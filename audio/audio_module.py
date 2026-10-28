"""
Audio Signal Processing Module.
Extracts voice activity, energy levels, silence duration, and interaction metrics.
"""
import time
import numpy as np


class AudioModule:
    """
    Processes incoming PCM audio chunks to analyze auditory classroom engagement.
    """

    def __init__(self, silence_energy_threshold: float = 0.015):
        self.silence_energy_threshold = silence_energy_threshold
        self.last_speech_time = time.time()

    def analyze_audio_chunk(self, audio_data: np.ndarray) -> dict:
        """
        Calculates Root Mean Square (RMS) energy, Zero Crossing Rate (ZCR),
        and Voice Activity Detection (VAD) from float32 audio arrays (-1.0 to 1.0).
        """
        now = time.time()
        if audio_data is None or len(audio_data) == 0:
            silence_gap = now - self.last_speech_time
            return {
                "rms_energy": 0.0,
                "is_speech": False,
                "interaction_score": 0.05,
                "silence_duration_sec": round(silence_gap, 1),
            }

        # 1. Root Mean Square (RMS) Energy
        rms = float(np.sqrt(np.mean(audio_data**2)))

        # 2. Zero Crossing Rate (ZCR)
        zero_crossings = np.nonzero(np.diff(audio_data > 0))[0]
        zcr = float(len(zero_crossings) / float(len(audio_data)))

        # 3. Speech Classification Heuristic
        # Human conversational speech has characteristic energy and moderate ZCR
        is_speech = bool(rms > self.silence_energy_threshold and 0.005 < zcr < 0.50)

        if is_speech:
            self.last_speech_time = now
            silence_gap = 0.0
            # Higher energy indicates active question/response or classroom debate
            interaction_score = min(float(rms * 12.0), 1.0)
        else:
            silence_gap = now - self.last_speech_time
            interaction_score = 0.05

        return {
            "rms_energy": round(rms, 4),
            "zcr": round(zcr, 4),
            "is_speech": is_speech,
            "interaction_score": round(interaction_score, 2),
            "silence_duration_sec": round(silence_gap, 1),
        }
