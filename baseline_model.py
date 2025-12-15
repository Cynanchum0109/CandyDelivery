#!/usr/bin/env python3
"""
交互时间预测模型训练脚本
使用MLP模型，根据开始时间预测结束时间
"""

import re
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os


def parse_time_to_seconds(time_str):
    """将时间字符串（MM:SS）转换为秒数"""
    parts = time_str.strip().split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {time_str}")
    minutes = int(parts[0])
    seconds = int(parts[1])
    return minutes * 60 + seconds


def load_dataset(dataset_file='interaction_time_dataset.md'):
    """从markdown文件加载数据集"""
    start_times = []
    end_times = []
    durations = []
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过标题行，解析数据行
    for line in lines:
        line = line.strip()
        # 匹配表格行格式: | 00:13 | 02:20 | 127 |
        match = re.match(r'^\|\s*(\d{2}:\d{2})\s*\|\s*(\d{2}:\d{2})\s*\|\s*(\d+)\s*\|', line)
        if match:
            start_str = match.group(1)
            end_str = match.group(2)
            duration = int(match.group(3))
            
            start_seconds = parse_time_to_seconds(start_str)
            end_seconds = parse_time_to_seconds(end_str)
            
            start_times.append(start_seconds)
            end_times.append(end_seconds)
            durations.append(duration)
    
    return np.array(start_times), np.array(end_times), np.array(durations)


def create_features(start_times, durations_history=None):
    """
    创建特征向量
    输入：开始时间（秒）
    可选：历史交互时长（用于更复杂的特征）
    """
    features = []
    
    for i, start_time in enumerate(start_times):
        # 基础特征：开始时间
        feature = [start_time]
        
        # 如果提供了历史数据，添加统计特征
        if durations_history is not None and len(durations_history) > 0:
            # 使用最近N个交互的统计信息
            window_size = min(5, len(durations_history))
            recent_durations = durations_history[-window_size:]
            
            mean_duration = np.mean(recent_durations)
            std_duration = np.std(recent_durations) if len(recent_durations) > 1 else 0
            max_duration = np.max(recent_durations)
            min_duration = np.min(recent_durations)
            
            feature.extend([mean_duration, std_duration, max_duration, min_duration])
        else:
            # 如果没有历史数据，用0填充
            feature.extend([0, 0, 0, 0])
        
        features.append(feature)
    
    return np.array(features)


def train_model(X, y, use_cross_validation=True, cv_folds=5):
    """
    训练MLP模型
    
    Args:
        X: 特征矩阵 (开始时间 + 历史统计特征)
        y: 目标值 (结束时间，秒)
        use_cross_validation: 是否使用交叉验证
        cv_folds: 交叉验证折数
    """
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 创建MLP模型
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),  # 小网络，防止过拟合
        activation='relu',
        solver='adam',
        alpha=0.01,  # L2正则化
        learning_rate='adaptive',
        max_iter=2000,  # 增加迭代次数
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42,
        verbose=False,
        tol=1e-4  # 收敛容忍度
    )
    
    if use_cross_validation and len(X) >= cv_folds:
        # 使用K折交叉验证
        print(f"\n进行 {cv_folds} 折交叉验证...")
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # 交叉验证评估
        cv_scores = cross_val_score(mlp, X_scaled, y, cv=kfold, 
                                    scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"交叉验证 MAE: {cv_mae:.2f} ± {cv_std:.2f} 秒")
        
        # 在全部数据上训练最终模型
        print("\n在全部数据上训练最终模型...")
        mlp.fit(X_scaled, y)
    else:
        # 简单训练测试分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print(f"\n训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        # 训练
        mlp.fit(X_train, y_train)
        
        # 评估
        y_pred_train = mlp.predict(X_train)
        y_pred_test = mlp.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n训练集指标:")
        print(f"  MAE: {train_mae:.2f} 秒")
        print(f"  RMSE: {train_rmse:.2f} 秒")
        print(f"  R²: {train_r2:.4f}")
        print(f"\n测试集指标:")
        print(f"  MAE: {test_mae:.2f} 秒")
        print(f"  RMSE: {test_rmse:.2f} 秒")
        print(f"  R²: {test_r2:.4f}")
    
    return mlp, scaler


def save_model(model, scaler, model_file='interaction_time_model.pkl', scaler_file='interaction_time_scaler.pkl'):
    """保存模型和标准化器"""
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n模型已保存到: {model_file}")
    print(f"标准化器已保存到: {scaler_file}")


def predict_end_time(start_time_seconds, model, scaler, durations_history=None):
    """
    预测结束时间
    
    Args:
        start_time_seconds: 开始时间（秒）
        model: 训练好的MLP模型
        scaler: 标准化器
        durations_history: 历史交互时长列表（可选）
    
    Returns:
        预测的结束时间（秒）
    """
    # 创建特征
    if durations_history is not None:
        feature = create_features([start_time_seconds], durations_history)[0]
    else:
        feature = create_features([start_time_seconds])[0]
    
    # 标准化
    feature_scaled = scaler.transform([feature])
    
    # 预测
    end_time_pred = model.predict(feature_scaled)[0]
    
    return end_time_pred


def seconds_to_time_str(seconds):
    """将秒数转换为 MM:SS 格式"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def main():
    """主函数"""
    print("=" * 60)
    print("交互时间预测模型训练")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据集...")
    start_times, end_times, durations = load_dataset('interaction_time_dataset.md')
    print(f"   加载了 {len(start_times)} 条数据")
    print(f"   开始时间范围: {seconds_to_time_str(start_times.min())} - {seconds_to_time_str(start_times.max())}")
    print(f"   结束时间范围: {seconds_to_time_str(end_times.min())} - {seconds_to_time_str(end_times.max())}")
    print(f"   交互时长范围: {durations.min()} - {durations.max()} 秒")
    
    # 2. 创建特征（使用历史交互时长）
    print("\n2. 创建特征...")
    X = create_features(start_times, durations)
    y = end_times
    print(f"   特征维度: {X.shape}")
    print(f"   特征包括: 开始时间 + 历史平均时长 + 历史标准差 + 历史最大时长 + 历史最小时长")
    
    # 3. 训练模型
    print("\n3. 训练模型...")
    model, scaler = train_model(X, y, use_cross_validation=True, cv_folds=5)
    
    # 4. 保存模型
    print("\n4. 保存模型...")
    save_model(model, scaler)
    
    # 5. 测试预测
    print("\n5. 测试预测示例...")
    print("\n   预测示例:")
    for i in range(min(5, len(start_times))):
        start_time = start_times[i]
        actual_end = end_times[i]
        
        # 使用到当前为止的历史数据
        hist_durations = durations[:i].tolist() if i > 0 else []
        predicted_end = predict_end_time(start_time, model, scaler, hist_durations)
        
        error = abs(predicted_end - actual_end)
        print(f"   开始: {seconds_to_time_str(start_time)} | "
              f"实际结束: {seconds_to_time_str(actual_end)} | "
              f"预测结束: {seconds_to_time_str(predicted_end)} | "
              f"误差: {error:.1f}秒")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


def load_trained_model(model_file='interaction_time_model.pkl', scaler_file='interaction_time_scaler.pkl'):
    """加载已训练的模型和标准化器"""
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        raise FileNotFoundError(f"模型文件不存在。请先运行训练: python {__file__}")
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


def predict_from_start_time(start_time_str, durations_history=None):
    """
    便捷函数：根据开始时间字符串预测结束时间
    
    Args:
        start_time_str: 开始时间字符串，格式 "MM:SS" 或 "M:SS"
        durations_history: 历史交互时长列表（秒），可选
    
    Returns:
        预测的结束时间字符串 "MM:SS"
    
    Example:
        >>> predict_from_start_time("00:13", [122, 97, 101])
        '02:31'
    """
    try:
        model, scaler = load_trained_model()
        start_seconds = parse_time_to_seconds(start_time_str)
        end_seconds = predict_end_time(start_seconds, model, scaler, durations_history)
        return seconds_to_time_str(end_seconds)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return None


if __name__ == '__main__':
    main()

