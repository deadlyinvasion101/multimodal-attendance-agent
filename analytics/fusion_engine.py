"""
Multimodal Signal Fusion and Classroom Dynamics Analytics Engine.
Fuses visual indicators with auditory signals into an Engagement Index,
detecting attention drops, interaction peaks, and session pacing issues.
"""
import time


class FusionEngine:
    def __init__(self, max_history: int = 150):
        self.max_history = max_history
        self.session_timeline = []
        self.session_start_time = time.time()
        self.total_attention_drops = 0
        self.total_interaction_peaks = 0

    def fuse_signals(self, vision_data: dict, audio_data: dict) -> dict:
        """
        Multimodal Fusion Formulation:
        Base = (0.50 * Attention) + (0.25 * Motion) + (0.25 * AudioInteraction)
        Adjusted for inactivity penalty if prolonged stillness occurs.
        """
        attention = vision_data.get("attention_score", 0.0)
        motion = vision_data.get("motion_score", 0.0)
        inactivity_sec = vision_data.get("inactivity_duration_sec", 0.0)

        audio_interaction = audio_data.get("interaction_score", 0.0)
        silence_sec = audio_data.get("silence_duration_sec", 0.0)

        # Baseline weighted combination
        base_engagement = (0.50 * attention) + (0.25 * motion) + (0.25 * audio_interaction)

        # Apply subtle inactivity penalty if both motion & audio are zero for > 10 seconds
        penalty = 0.0
        if inactivity_sec > 10.0 and silence_sec > 10.0:
            penalty = min(0.20, (inactivity_sec - 10.0) * 0.01)

        fused_score = max(0.0, min(1.0, round(base_engagement - penalty, 2)))

        # Track dynamics
        if fused_score < 0.35 and len(self.session_timeline) > 2:
            self.total_attention_drops += 1
        elif fused_score > 0.75:
            self.total_interaction_peaks += 1

        record = {
            "timestamp": time.strftime("%H:%M:%S"),
            "elapsed_seconds": int(time.time() - self.session_start_time),
            "fused_engagement": fused_score,
            "attention": attention,
            "motion": motion,
            "audio_interaction": audio_interaction,
            "face_count": vision_data.get("face_count", 0),
            "is_speech": audio_data.get("is_speech", False),
            "status": "Attentive" if fused_score >= 0.65 else ("Moderate" if fused_score >= 0.4 else "Disengaged"),
        }

        self.session_timeline.append(record)
        if len(self.session_timeline) > self.max_history:
            self.session_timeline.pop(0)

        return record

    def get_session_summary(self) -> dict:
        """
        Generates aggregated session dynamics report.
        """
        if not self.session_timeline:
            return {
                "avg_engagement": 0.0,
                "attention_drops": 0,
                "interaction_peaks": 0,
                "active_duration_minutes": 0.0,
                "session_state": "Waiting for stream",
                "timeline": [],
            }

        scores = [item["fused_engagement"] for item in self.session_timeline]
        avg_engagement = round(float(sum(scores) / len(scores)), 2)
        duration_minutes = round((time.time() - self.session_start_time) / 60.0, 1)

        # Classify overall trend
        if avg_engagement >= 0.70:
            dynamics = "High Interactive Engagement"
        elif avg_engagement >= 0.45:
            dynamics = "Steady Passive Listening"
        else:
            dynamics = "Significant Attention Fatigue Detected"

        return {
            "avg_engagement": avg_engagement,
            "attention_drops": self.total_attention_drops,
            "interaction_peaks": self.total_interaction_peaks,
            "active_duration_minutes": duration_minutes,
            "total_frames_analyzed": len(self.session_timeline),
            "dynamics_label": dynamics,
            "timeline": self.session_timeline,
        }

    def reset(self):
        self.session_timeline.clear()
        self.session_start_time = time.time()
        self.total_attention_drops = 0
        self.total_interaction_peaks = 0

