"""
Vision Module for Attendance Tracking and Engagement Estimation.
Powered by OpenCV YuNet DNN Face & Landmark Detector with fallback geometry processing.
Handles face tracking, sustained presence, head pose/attention, and movement dynamics.
"""
import os
import time
import cv2
import numpy as np


class VisionModule:
    """
    Analyzes visual streams for attendance validation and engagement indicators.
    """

    def __init__(self, sustained_presence_threshold: float = 3.0):
        self.sustained_presence_threshold = sustained_presence_threshold
        self.detector = None
        self.model_path = os.path.join(os.path.dirname(__file__), "yunet.onnx")

        if os.path.exists(self.model_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=self.model_path,
                    config="",
                    input_size=(320, 240),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=20,
                )
            except Exception as e:
                print(f"[VisionModule] YuNet init warning: {e}")

        # Entity tracking: participant_id -> {first_seen, last_seen, accumulated_presence}
        self.trackers = {}
        # Final attendance ledger: participant_id -> {status, verified_at, sustained_seconds}
        self.attendance_records = {}
        self.last_gray_frame = None
        self.inactivity_start_time = None

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single image frame (BGR format).
        Returns visual behavioral indicators and updated attendance records.
        """
        now = time.time()
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Motion Dynamics (Frame differencing)
        motion_score = 0.0
        if self.last_gray_frame is not None and self.last_gray_frame.shape == gray.shape:
            diff = cv2.absdiff(self.last_gray_frame, gray)
            motion_score = float(np.mean(diff) / 255.0)
            motion_score = min(motion_score * 4.0, 1.0)
        self.last_gray_frame = gray

        # 2. Inactivity tracking
        if motion_score < 0.05:
            if self.inactivity_start_time is None:
                self.inactivity_start_time = now
            inactivity_seconds = now - self.inactivity_start_time
        else:
            self.inactivity_start_time = None
            inactivity_seconds = 0.0

        # 3. Face Detection & Head Pose / Attention Analysis
        faces_detected = []
        attention_values = []
        detected_participants = []

        if self.detector is not None:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)
            if faces is not None:
                for face in faces:
                    # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
                    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    score = float(face[-1])
                    if score < 0.5:
                        continue

                    # Landmarks: Right Eye, Left Eye, Nose Tip
                    re_x, re_y = face[4], face[5]
                    le_x, le_y = face[6], face[7]
                    nose_x, nose_y = face[8], face[9]

                    # Gaze/Pose proxy: horizontal symmetry of nose between eyes
                    eye_mid_x = (re_x + le_x) / 2.0
                    eye_dist = max(1.0, abs(le_x - re_x))
                    nose_dev = abs(nose_x - eye_mid_x) / eye_dist

                    # If nose is centered between eyes, attention is high (frontal)
                    attention_score = max(0.2, min(1.0, 1.0 - (nose_dev * 1.5)))
                    attention_values.append(attention_score)
                    faces_detected.append((fx, fy, fw, fh))
        else:
            # Fallback color/contour detector if model unavailable
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 2000:
                    fx, fy, fw, fh = cv2.boundingRect(cnt)
                    faces_detected.append((fx, fy, fw, fh))
                    attention_values.append(0.75)

        # 4. Sustained Presence Tracking (Scenario 1)
        for (fx, fy, fw, fh) in faces_detected:
            slot_id = f"Student_{chr(65 + (fx // 120) % 26)}"
            detected_participants.append(slot_id)

            if slot_id not in self.trackers:
                self.trackers[slot_id] = {
                    "first_seen": now,
                    "last_seen": now,
                    "accumulated_presence": 0.0,
                }
            else:
                gap = now - self.trackers[slot_id]["last_seen"]
                if gap <= 2.5:
                    self.trackers[slot_id]["accumulated_presence"] += gap
                self.trackers[slot_id]["last_seen"] = now

            acc = self.trackers[slot_id]["accumulated_presence"]
            if acc >= self.sustained_presence_threshold:
                if slot_id not in self.attendance_records:
                    self.attendance_records[slot_id] = {
                        "participant": slot_id,
                        "status": "Verified Present",
                        "first_verified": time.strftime("%H:%M:%S", time.localtime(self.trackers[slot_id]["first_seen"])),
                        "sustained_seconds": round(acc, 1),
                    }
                else:
                    self.attendance_records[slot_id]["sustained_seconds"] = round(acc, 1)

        avg_attention = float(np.mean(attention_values)) if attention_values else 0.0

        return {
            "face_count": len(faces_detected),
            "attention_score": round(avg_attention, 2),
            "motion_score": round(motion_score, 2),
            "inactivity_duration_sec": round(inactivity_seconds, 1),
            "active_participants": detected_participants,
            "attendance": self.attendance_records,
        }
