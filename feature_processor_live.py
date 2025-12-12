# feature_processor_live.py
import cv2
import time
import threading
import numpy as np
from collections import deque
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import joblib

# Constants
FACE_SKIP_FRAMES = 5
HISTORY_LEN = 15
# === MUST MATCH TRAINING CSV EXACTLY ===
FEATURE_COLUMNS = [
    "slouch_ratio",
    "shoulder_width_ratio",
    "engagement_zone",
    "dist_proxy",
    "face_vis_yolo",
    "face_vis_insight",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "is_looking_at_robot",
    "AU12_Smile",
    "AU45_EyeOpen",
    "AU25_MouthOpen",
    "AU01_BrowRaise",
    "nod_energy",
    "shake_energy",
    "is_waving",
]


class FeatureProcessorLive:
    """
    Real-time version of the offline FeatureProcessor used in video_processing.py.
    Extracts the same CSV features, but does not write CSV.
    Provides:
        - get_state(): full feature vector (dict)
        - get_prediction(): (label, prob) if model enabled
    """

    def __init__(self, cap, use_prediction=False):
        self.cap = cap
        self.running = False
        self.frame_count = 0

        # YOLO and Face models
        self.yolo = YOLO("yolov8n-pose.pt")
        self.face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))

        # Buffers for motion energy
        self.nose_hist = deque(maxlen=HISTORY_LEN)
        self.wrist_hist = deque(maxlen=HISTORY_LEN)

        # Prediction model (optional)
        self.use_prediction = use_prediction
        if use_prediction:
            try:
                self.model = joblib.load("robot_mlp_sklearn.pkl")
                self.scaler = joblib.load("robot_scaler.pkl")
                print("✓ Loaded disengagement model.")
            except Exception as e:
                print("⚠ Could not load model:", e)
                self.use_prediction = False
                self.model = None
                self.scaler = None
        else:
            self.model = None
            self.scaler = None

        # Shared state
        self.state_lock = threading.Lock()
        self.latest_state = self._empty_state()

        # Prediction outputs
        self.latest_prediction = None
        self.latest_prob = None

    # -----------------------------------------------------
    #   Full Feature State Structure
    # -----------------------------------------------------
    def _empty_state(self):
        return {
            "frame": 0,
            "timestamp": 0.0,

            # Body metrics
            "slouch_ratio": 0.0,
            "shoulder_width_ratio": 0.0,
            "engagement_zone": "Unknown",
            "dist_proxy": 0.0,

            # Visibility
            "face_vis_yolo": 0.0,
            "face_vis_insight": 0.0,

            # Head pose
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "head_roll": 0.0,
            "is_looking_at_robot": 0,

            # AU features
            "AU12_Smile": 0.0,
            "AU45_EyeOpen": 0.0,
            "AU25_MouthOpen": 0.0,
            "AU01_BrowRaise": 0.0,

            # Motion / Gesture
            "nod_energy": 0.0,
            "shake_energy": 0.0,
            "is_waving": 0,

            # Label (only used for prediction mode)
            "engage_label": None,
        }

    # -----------------------------------------------------
    #   Helpers
    # -----------------------------------------------------
    def _compute_au_106(self, landmarks):
        """Extract AU12, AU45, AU25, AU01 from InsightFace 106 landmarks."""
        def dist(i, j):
            return np.linalg.norm(landmarks[i] - landmarks[j])

        face_w = dist(1, 17)
        if face_w == 0: face_w = 1.0

        smile = dist(52, 61) / face_w
        eyes = (dist(37, 41) + dist(89, 93)) / (2 * face_w)
        mouth = dist(63, 56) / face_w
        brow = dist(37, 24) / face_w

        return smile, eyes, mouth, brow

    def _compute_temporal(self, nose, l_wrist, r_wrist):
        """Update for nod/shake and waving energy."""
        self.nose_hist.append(nose)
        self.wrist_hist.append((l_wrist, r_wrist))

        # nod / shake
        if len(self.nose_hist) >= 5:
            arr = np.array(self.nose_hist)
            nod = float(np.std(arr[:, 1]))   # vertical
            shake = float(np.std(arr[:, 0])) # horizontal
        else:
            nod = shake = 0.0

        # waving detection
        if len(self.wrist_hist) >= 5:
            hist = np.array(self.wrist_hist)
            l_wrist_var = np.std(hist[:, 0, 1])
            r_wrist_var = np.std(hist[:, 1, 1])
            is_waving = int(l_wrist_var > 15 or r_wrist_var > 15)
        else:
            is_waving = 0

        return nod, shake, is_waving

    # -----------------------------------------------------
    #   Real-Time Feature Processing
    # -----------------------------------------------------
    def process_frame(self, frame):

        # -------------------------
        # FAST BODY LOOP (every frame)
        # -------------------------
        yolo_r = self.yolo(frame, verbose=False)[0]

        slouch = shoulder_ratio = face_vis_yolo = 0.0
        zone = "Unknown"
        dist_proxy = 0.0
        nod = shake = is_waving = 0.0

        if yolo_r.keypoints is not None and len(yolo_r.keypoints.data) > 0:

            kpts = yolo_r.keypoints.data[0].cpu().numpy()
            box = yolo_r.boxes.data[0].cpu().numpy()

            # Keypoints
            l_sh, r_sh = kpts[5][:2], kpts[6][:2]
            l_hip, r_hip = kpts[11][:2], kpts[12][:2]
            nose = kpts[0][:2]
            l_wrist, r_wrist = kpts[9][:2], kpts[10][:2]

            # Person height
            person_h = max(box[3] - box[1], 1)

            # Slouch
            torso_h = abs(l_hip[1] - l_sh[1])
            slouch = torso_h / person_h

            # Shoulder width ratio
            shoulder_width = np.linalg.norm(l_sh - r_sh)
            shoulder_ratio = shoulder_width / person_h

            # Proxemics proxy
            dist_proxy = 1000 / (torso_h + 1)

            if dist_proxy < 3:
                zone = "Intimate"
            elif dist_proxy < 6:
                zone = "Social"
            else:
                zone = "Public"

            # YOLO face visibility
            face_vis_yolo = sum(1 for p in kpts[0:5] if p[2] > 0.5) / 5.0

            # Temporal energies
            body_center = (l_sh + r_sh) / 2
            rel_nose = nose - body_center
            nod, shake, is_waving = self._compute_temporal(rel_nose, l_wrist, r_wrist)

        # -------------------------
        # SLOW FACE LOOP (every N frames)
        # -------------------------
        yaw = pitch = roll = 0.0
        is_looking = 0
        smile = eyes = mouth = brow = 0.0
        face_vis_insight = 0.0

        if self.frame_count % FACE_SKIP_FRAMES == 0:

            faces = self.face_app.get(frame)

            if faces:
                f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                pose = f.pose
                pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])

                # Looking check
                is_looking = int(abs(yaw) < 15 and abs(pitch) < 15)

                # InsightFace visibility score
                if hasattr(f, "det_score"):
                    face_vis_insight = float(f.det_score)
                else:
                    face_vis_insight = 1.0

                # AU features
                if f.landmark_2d_106 is not None:
                    smile, eyes, mouth, brow = self._compute_au_106(f.landmark_2d_106)

        timestamp_ms = time.time() * 1000.0

        with self.state_lock:
            self.latest_state = {
                "frame": self.frame_count,
                "timestamp": timestamp_ms,

                "slouch_ratio": slouch,
                "shoulder_width_ratio": shoulder_ratio,
                "engagement_zone": zone,   # 保留 string，给人看
                "dist_proxy": dist_proxy,

                "face_vis_yolo": face_vis_yolo,
                "face_vis_insight": face_vis_insight,

                "head_yaw": yaw,
                "head_pitch": pitch,
                "head_roll": roll,
                "is_looking_at_robot": is_looking,

                "AU12_Smile": smile,
                "AU45_EyeOpen": eyes,
                "AU25_MouthOpen": mouth,
                "AU01_BrowRaise": brow,

                "nod_energy": nod,
                "shake_energy": shake,
                "is_waving": is_waving,

                "engage_label": None,
            }


        # --------------------------------
        # Optional: Prediction (SAFE)
        # --------------------------------
        if self.use_prediction and self.scaler and self.model:

            import pandas as pd
            import warnings

            # Encode zone EXACTLY like training
            ZONE_MAP = {
                "Intimate": 0,
                "Social": 1,
                "Public": 2,
                "Unknown": 3,
            }
            zone_code = ZONE_MAP.get(zone, 3)

            # Build feature dict (names MUST match CSV)
            feature_dict = {
                "slouch_ratio": slouch,
                "shoulder_width_ratio": shoulder_ratio,
                "engagement_zone": zone_code,
                "dist_proxy": dist_proxy,
                "face_vis_yolo": face_vis_yolo,
                "face_vis_insight": face_vis_insight,
                "head_yaw": yaw,
                "head_pitch": pitch,
                "head_roll": roll,
                "is_looking_at_robot": is_looking,
                "AU12_Smile": smile,
                "AU45_EyeOpen": eyes,
                "AU25_MouthOpen": mouth,
                "AU01_BrowRaise": brow,
                "nod_energy": nod,
                "shake_energy": shake,
                "is_waving": is_waving,
            }

            # DataFrame → column reorder (CRITICAL)
            X = pd.DataFrame([feature_dict])
            X = X[FEATURE_COLUMNS]

            # Scale + predict
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                X_scaled = self.scaler.transform(X)

            prob = float(self.model.predict_proba(X_scaled)[0][1])
            label = int(prob > 0.5)

            with self.state_lock:
                self.latest_prediction = label
                self.latest_prob = prob
                self.latest_state["engage_label"] = label

        
    # -----------------------------------------------------
    #   Public API
    # -----------------------------------------------------
    def get_state(self):
        with self.state_lock:
            return dict(self.latest_state)

    def get_prediction(self):
        with self.state_lock:
            return self.latest_prediction, self.latest_prob

    # -----------------------------------------------------
    #   Thread Loop
    # -----------------------------------------------------
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
