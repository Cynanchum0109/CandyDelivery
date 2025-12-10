# feature_processor_live.py
import cv2
import time
import threading
import numpy as np
from collections import deque
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import joblib

FACE_SKIP_FRAMES = 5
HISTORY_LEN = 15

class FeatureProcessorLive:
    def __init__(self, cap, use_prediction=False):
        """
        Real-time YOLO + InsightFace feature extraction.
        
        Args:
            cap: shared VideoCapture
            use_prediction: If True, load MLP+scaler and compute prediction.
                            If False, only extract features and print them.
        """
        self.cap = cap
        self.running = False
        self.frame_count = 0

        # Store whether we want MLP prediction
        self.use_prediction = use_prediction

        # ---------------------------------------
        # LOAD YOLO, InsightFace
        # ---------------------------------------
        self.yolo = YOLO("yolov8n-pose.pt")
        self.face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))

        self.nose_hist = deque(maxlen=HISTORY_LEN)
        self.wrist_hist = deque(maxlen=HISTORY_LEN)

        # ---------------------------------------
        # OPTIONAL: LOAD DISENGAGEMENT MODEL
        # ---------------------------------------
        if self.use_prediction:
            print("Loading Scikit-Learn Brain...")
            try:
                self.model = joblib.load("robot_mlp_sklearn.pkl")
                self.scaler = joblib.load("robot_scaler.pkl")
                print("MLP model loaded.")
            except Exception as e:
                print(f"⚠ MLP model could not be loaded: {e}")
                print("Running WITHOUT prediction.")
                self.use_prediction = False
                self.model = None
                self.scaler = None
        else:
            print("Running with feature extraction ONLY (no prediction).")
            self.model = None
            self.scaler = None

        # Last prediction
        self.latest_prediction = None   # None when disabled
        self.latest_prob = None

        # Shared state
        self.state_lock = threading.Lock()
        self.latest_state = {
            "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
            "is_looking": 0,
            "smile": 0.0, "eyes": 0.0, "mouth": 0.0, "brow": 0.0,
            "face_vis_yolo": 0.0,
            "shake": 0.0,
            "nod": 0.0,
            "waving": 0,
            "shoulder_ratio": 0.0,
            "slouch_ratio": 0.0,
        }

    def update_body_hist(self, nose, lw, rw):
        self.nose_hist.append(nose)
        self.wrist_hist.append((lw, rw))

    def get_nod_shake(self):
        if len(self.nose_hist) < 5: 
            return 0.0, 0.0
        arr = np.array(self.nose_hist)
        return float(np.std(arr[:, 0])), float(np.std(arr[:, 1]))

    def get_action_units(self, landmarks):
        def dist(a, b):
            return np.linalg.norm(landmarks[a] - landmarks[b])
        face_w = dist(1, 17) or 1.0
        smile = dist(52, 61) / face_w
        eye = (dist(37, 41) + dist(89, 93)) / (2 * face_w)
        mouth = dist(63, 56) / face_w
        brow = dist(37, 24) / face_w
        return smile, eye, mouth, brow

    def process_frame(self, frame):
        """Process a single frame for features and optional prediction."""
        yolo_r = self.yolo(frame, verbose=False)[0]

        if yolo_r.keypoints is not None and len(yolo_r.keypoints.data) > 0:
            kpts = yolo_r.keypoints.data[0].cpu().numpy()

            # Basic body metrics
            l_sh, r_sh = kpts[5][:2], kpts[6][:2]
            l_hip, r_hip = kpts[11][:2], kpts[12][:2]
            nose = kpts[0][:2]
            torso_h = abs(l_hip[1] - l_sh[1])
            shoulder_w = np.linalg.norm(l_sh - r_sh)
            person_h = max((yolo_r.boxes.data[0][3] - yolo_r.boxes.data[0][1]), 1)

            slouch = torso_h / person_h
            shoulder_ratio = shoulder_w / person_h

            # Temporal
            body_center = (l_sh + r_sh) / 2
            self.update_body_hist(nose - body_center, kpts[9][:2], kpts[10][:2])
            nod, shake = self.get_nod_shake()

            # Face visibility
            face_vis = sum(1 for p in kpts[0:5] if p[2] > 0.5) / 5.0
        else:
            slouch = shoulder_ratio = nod = shake = face_vis = 0.0

        yaw = 0.0
        pitch = 0.0
        roll = 0.0
        smile = 0.0
        eyes = 0.0
        mouth = 0.0
        brow = 0.0
        is_looking = 0
        # Slow face loop
        smile = eyes = mouth = brow = 0.0
        yaw = pitch = roll = 0.0
        is_looking = 0

        if self.frame_count % FACE_SKIP_FRAMES == 0:
            faces = self.face_app.get(frame)
            if faces:
                f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                pose = f.pose
                pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])

                is_looking = 1 if abs(yaw) < 15 and abs(pitch) < 15 else 0

                if f.landmark_2d_106 is not None:
                    smile, eyes, mouth, brow = self.get_action_units(f.landmark_2d_106)

        # Save state
        with self.state_lock:
            self.latest_state.update({
                "yaw": yaw, "pitch": pitch, "roll": roll,
                "is_looking": is_looking,
                "smile": smile, "eyes": eyes, "mouth": mouth, "brow": brow,
                "face_vis_yolo": face_vis,
                "nod": nod, "shake": shake,
                "shoulder_ratio": shoulder_ratio,
                "slouch_ratio": slouch,
            })

        # Debug print (ALWAYS ON)
        print("FEATURES:", self.latest_state)

        # ---------------------------------------
        # OPTIONAL PREDICTION PIPELINE
        # ---------------------------------------
        if self.use_prediction:
            features_vector = np.array([
                slouch,
                shoulder_ratio,
                face_vis,
                is_looking,
                yaw, pitch, roll,
                smile, eyes, mouth, brow,
                nod, shake,
            ]).reshape(1, -1)

            scaled = self.scaler.transform(features_vector)
            pred_prob = self.model.predict_proba(scaled)[0][1]
            pred_label = int(pred_prob > 0.5)

            with self.state_lock:
                self.latest_prediction = pred_label
                self.latest_prob = float(pred_prob)
        else:
            # No prediction
            with self.state_lock:
                self.latest_prediction = None
                self.latest_prob = None

    def get_state(self):
        with self.state_lock:
            return dict(self.latest_state)

    def get_prediction(self):
        """If prediction disabled, return (None, None)."""
        with self.state_lock:
            return self.latest_prediction, self.latest_prob

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self.process_frame(frame)
            self.frame_count += 1

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1)