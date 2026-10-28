"""
AI Teaching Advisor powered by Google Gemini 2.0 Flash.
Acts as an assistive pedagogical decision-support agent for instructors.
"""
import json
import os
from google import genai
from google.genai import types


class TeachingAdvisor:
    def __init__(self):
        # Supports GEMINI_API_KEY from environment or .env
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[TeachingAdvisor] Could not initialize Gemini client: {e}")

    def generate_recommendations(self, session_summary: dict) -> dict:
        """
        Synthesizes multimodal classroom telemetry into pedagogical recommendations.
        """
        avg_eng = session_summary.get("avg_engagement", 0.5)
        drops = session_summary.get("attention_drops", 0)
        peaks = session_summary.get("interaction_peaks", 0)
        dynamics = session_summary.get("dynamics_label", "Moderate Engagement")
        duration = session_summary.get("active_duration_minutes", 1.0)

        prompt = f"""
You are an expert AI Classroom Pedagogical Advisor assisting an educator in real-time.
Review the following multimodal session metrics:
- Overall Engagement Score: {avg_eng * 100:.1f}% ({dynamics})
- Session Elapsed Time: {duration} minutes
- Detected Attention Drops (Fatigue/Lapses): {drops}
- Detected Interaction Peaks (Debate/Participation): {peaks}

Generate an assistive diagnostic summary and 3 high-impact, actionable teaching suggestions:
1. Pacing & Tone adjustment (should the instructor slow down, pause, or speed up?)
2. Interactive micro-activity (e.g. 60-second pair-share, rapid quiz, conceptual check)
3. Cognitive re-focus tactic (visual cue, student prompt, real-world example)

Output strictly valid JSON matching this schema:
{{
    "status_summary": "1-2 sentence real-time classroom diagnostic.",
    "energy_level": "Low" | "Balanced" | "High",
    "suggestions": [
        {{"category": "Pacing", "action": "...", "priority": "High" | "Medium" | "Low"}},
        {{"category": "Interactivity", "action": "...", "priority": "High" | "Medium" | "Low"}},
        {{"category": "Focus & Engagement", "action": "...", "priority": "High" | "Medium" | "Low"}}
    ]
}}
"""

        # 1. Use Gemini 2.0 Flash if client available
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.3,
                    ),
                )
                if response.text:
                    return json.loads(response.text)
            except Exception as e:
                print(f"[TeachingAdvisor] Gemini API call error: {e}")

        # 2. Heuristic fallback (Ensures completely reliable offline operation)
        return self._generate_offline_heuristic(avg_eng, drops, peaks)

    def _generate_offline_heuristic(self, avg_eng: float, drops: int, peaks: int) -> dict:
        if avg_eng < 0.40 or drops > 3:
            return {
                "status_summary": f"Engagement is currently low ({avg_eng * 100:.0f}%) with {drops} attention lapses detected. The room needs a cognitive reset.",
                "energy_level": "Low",
                "suggestions": [
                    {
                        "category": "Pacing",
                        "action": "Slow down and pause for 45 seconds. Ask students to write down the single key takeaway so far.",
                        "priority": "High",
                    },
                    {
                        "category": "Interactivity",
                        "action": "Run a quick 'Think-Pair-Share' or ask a direct, open-ended question to re-energize the room.",
                        "priority": "High",
                    },
                    {
                        "category": "Focus & Engagement",
                        "action": "Switch slides or stand up / move closer to the center of the whiteboard to draw visual focus.",
                        "priority": "Medium",
                    },
                ],
            }
        elif avg_eng > 0.70 or peaks > 2:
            return {
                "status_summary": f"Excellent session momentum! Classroom engagement is high at {avg_eng * 100:.0f}%.",
                "energy_level": "High",
                "suggestions": [
                    {
                        "category": "Pacing",
                        "action": "Maintain this brisk tempo. Transition smoothly into the core problem or advanced example.",
                        "priority": "Medium",
                    },
                    {
                        "category": "Interactivity",
                        "action": "Capitalize on current energy: have a student explain their solution to the class.",
                        "priority": "High",
                    },
                    {
                        "category": "Focus & Engagement",
                        "action": "Reinforce correct understanding before moving to the next module.",
                        "priority": "Low",
                    },
                ],
            }
        else:
            return {
                "status_summary": f"Classroom engagement is steady at {avg_eng * 100:.0f}% with consistent focus.",
                "energy_level": "Balanced",
                "suggestions": [
                    {
                        "category": "Pacing",
                        "action": "Pacing is optimal. Check for non-verbal cues before proceeding to complex topics.",
                        "priority": "Low",
                    },
                    {
                        "category": "Interactivity",
                        "action": "Introduce a quick 2-minute hands-on check or quiz question.",
                        "priority": "Medium",
                    },
                    {
                        "category": "Focus & Engagement",
                        "action": "Acknowledge participant contributions to sustain positive classroom atmosphere.",
                        "priority": "Low",
                    },
                ],
            }

