"""
Flask Application & API Gateway for Multimodal Attendance & Engagement Agent.
Integrates Vision, Audio, Multimodal Fusion, and Gemini 2.0 Flash AI Advisor.
"""
import base64
import csv
import io
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, Response

from ai_agent.teaching_advisor import TeachingAdvisor
from analytics.fusion_engine import FusionEngine
from audio.audio_module import AudioModule
from vision.vision_module import VisionModule

load_dotenv()

app = Flask(__name__)

# Initialize Core Services
vision_engine = VisionModule(sustained_presence_threshold=3.0)
audio_engine = AudioModule()
fusion_engine = FusionEngine()
advisor = TeachingAdvisor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    """
    Receives real-time video frame and audio sample buffer from browser.
    """
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image", "")
    audio_data = payload.get("audio", [])

    # 1. Vision Processing
    vision_result = {
        "face_count": 0,
        "attention_score": 0.0,
        "motion_score": 0.0,
        "inactivity_duration_sec": 0.0,
        "active_participants": [],
        "attendance": vision_engine.attendance_records,
    }

    if image_b64:
        try:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            raw_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                vision_result = vision_engine.process_frame(frame)
        except Exception as err:
            app.logger.warning(f"Frame decoding error: {err}")

    # 2. Audio Processing
    if isinstance(audio_data, list) and len(audio_data) > 0:
        arr = np.array(audio_data, dtype=np.float32)
    else:
        arr = np.array([], dtype=np.float32)

    audio_result = audio_engine.analyze_audio_chunk(arr)

    # 3. Multimodal Signal Fusion
    fused_metrics = fusion_engine.fuse_signals(vision_result, audio_result)

    return jsonify(
        {
            "status": "success",
            "vision": vision_result,
            "audio": audio_result,
            "fused": fused_metrics,
            "attendance": vision_engine.attendance_records,
        }
    )


@app.route("/api/process_video_file", methods=["POST"])
def process_video_file():
    """
    Scenario 1: Support processing an uploaded recorded video file.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files["video"]
    temp_path = os.path.join(os.path.dirname(__file__), "temp_video.mp4")
    file.save(temp_path)

    cap = cv2.VideoCapture(temp_path)
    frame_idx = 0
    results = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx > 300:  # Sample up to ~300 frames
                break
            # Sample every 5th frame for efficiency
            if frame_idx % 5 == 0:
                v_res = vision_engine.process_frame(frame)
                a_res = {"rms_energy": 0.02, "is_speech": True, "interaction_score": 0.4, "silence_duration_sec": 0.0}
                f_res = fusion_engine.fuse_signals(v_res, a_res)
                results.append(f_res)
            frame_idx += 1
    finally:
        cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(
        {
            "status": "complete",
            "processed_frames": len(results),
            "attendance": vision_engine.attendance_records,
            "summary": fusion_engine.get_session_summary(),
        }
    )


@app.route("/api/ai_advisor", methods=["GET"])
def get_ai_advisor():
    """
    Scenario 3: AI-Driven Teaching Suggestions from Gemini.
    """
    summary = fusion_engine.get_session_summary()
    advisor_response = advisor.generate_recommendations(summary)
    return jsonify(
        {
            "status": "success",
            "classroom_summary": summary,
            "ai_advisor": advisor_response,
        }
    )


@app.route("/api/session_analytics", methods=["GET"])
def get_session_analytics():
    return jsonify(fusion_engine.get_session_summary())


@app.route("/api/export_attendance_csv", methods=["GET"])
def export_attendance_csv():
    """
    Exports sustained presence attendance logs as a downloadable CSV.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Participant ID", "Status", "First Verified Timestamp", "Sustained Seconds"])

    for pid, data in vision_engine.attendance_records.items():
        writer.writerow([pid, data["status"], data["first_verified"], data["sustained_seconds"]])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=attendance_report.csv"},
    )


@app.route("/api/reset_session", methods=["POST"])
def reset_session():
    vision_engine.trackers.clear()
    vision_engine.attendance_records.clear()
    fusion_engine.reset()
    return jsonify({"status": "session_reset_complete"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

