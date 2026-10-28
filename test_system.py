"""
End-to-End System Test for Multimodal Attendance & Engagement Agent.
Verifies Vision Module, Audio Module, Fusion Engine, AI Advisor, and Flask API.
"""
import unittest
import numpy as np
import cv2

from vision.vision_module import VisionModule
from audio.audio_module import AudioModule
from analytics.fusion_engine import FusionEngine
from ai_agent.teaching_advisor import TeachingAdvisor
from app import app


class TestMultimodalAgent(unittest.TestCase):
    def setUp(self):
        self.vision = VisionModule(sustained_presence_threshold=1.0)
        self.audio = AudioModule()
        self.fusion = FusionEngine()
        self.advisor = TeachingAdvisor()
        self.client = app.test_client()

    def test_vision_module(self):
        # Create a synthetic test frame (640x480, 3 channels)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic circle/face approximation
        cv2.circle(dummy_frame, (320, 240), 60, (255, 255, 255), -1)

        result = self.vision.process_frame(dummy_frame)
        self.assertIn("face_count", result)
        self.assertIn("attention_score", result)
        self.assertIn("motion_score", result)
        self.assertIn("attendance", result)

    def test_audio_module(self):
        # Generate 1 second of synthetic 44.1kHz sine wave (simulating speech/voice)
        t = np.linspace(0, 1, 44100, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

        res_tone = self.audio.analyze_audio_chunk(tone)
        self.assertGreater(res_tone["rms_energy"], 0.0)
        self.assertTrue(res_tone["is_speech"])

        # Test silence
        silence = np.zeros(1000, dtype=np.float32)
        res_silence = self.audio.analyze_audio_chunk(silence)
        self.assertFalse(res_silence["is_speech"])

    def test_fusion_engine(self):
        vision_mock = {"attention_score": 0.8, "motion_score": 0.4, "inactivity_duration_sec": 0.0, "face_count": 5}
        audio_mock = {"interaction_score": 0.7, "silence_duration_sec": 0.0, "is_speech": True}

        fused = self.fusion.fuse_signals(vision_mock, audio_mock)
        self.assertIn("fused_engagement", fused)
        self.assertGreaterEqual(fused["fused_engagement"], 0.5)

        summary = self.fusion.get_session_summary()
        self.assertIn("avg_engagement", summary)
        self.assertIn("dynamics_label", summary)

    def test_ai_advisor(self):
        summary_mock = {
            "avg_engagement": 0.30,
            "attention_drops": 4,
            "interaction_peaks": 0,
            "dynamics_label": "Significant Attention Fatigue Detected",
            "active_duration_minutes": 25.0,
        }
        recs = self.advisor.generate_recommendations(summary_mock)
        self.assertIn("status_summary", recs)
        self.assertIn("suggestions", recs)
        self.assertGreaterEqual(len(recs["suggestions"]), 2)

    def test_flask_routes(self):
        # 1. Main Page
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)

        # 2. AI Advisor Endpoint
        res_ai = self.client.get("/api/ai_advisor")
        self.assertEqual(res_ai.status_code, 200)
        json_data = res_ai.get_json()
        self.assertEqual(json_data["status"], "success")

        # 3. CSV Export Endpoint
        res_csv = self.client.get("/api/export_attendance_csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn(b"Participant ID", res_csv.data)


if __name__ == "__main__":
    unittest.main()

