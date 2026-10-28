# Multimodal Attendance & Engagement Agent

An AI-powered, privacy-preserving classroom intelligence system integrating **Computer Vision**, **Audio Signal Processing**, and **Agent-Based Reasoning (Google Gemini 2.0 Flash)**.

---

## 🌟 Key Scenarios & Capabilities

### 1. Scenario 1: Automated Attendance Tracking
- Sustained presence detection eliminates false positives from temporary walk-bys.
- Automatically generates timestamped verification logs.
- Supports live webcam feeds or uploaded recorded video sessions.
- Downloadable attendance reports in CSV format.

### 2. Scenario 2: Engagement and Session Dynamics Analysis
- Tracks visual behavioral indicators: head pose, gaze orientation, movement dynamics, and inactivity periods.
- Analyzes acoustic energy via Voice Activity Detection (VAD) and silence intervals.
- Fuses multimodal signals into a unified **Engagement Index**.
- Flags attention drops, interaction peaks, and session pacing bottlenecks in real time.

### 3. Scenario 3: AI-Driven Teaching Suggestions
- Integrates Google Gemini 2.0 Flash as an assistive pedagogical coach.
- Provides context-aware recommendations for:
  - **Pacing**: Slowing down, speeding up, or inserting comprehension pauses.
  - **Interactivity**: Quick polls, think-pair-share, or conceptual checks.
  - **Dynamics**: Focus re-orientation and classroom atmosphere management.
- Runs with Gemini online or seamlessly with intelligent offline heuristics.

---

## 📁 Project Structure

```
multimodal-attendance-agent/
├── app.py                      # Flask API Gateway & Streaming Server
├── requirements.txt            # Python dependencies
├── test_system.py              # Automated test suite
├── vision/
│   ├── __init__.py
│   └── vision_module.py       # OpenCV face, gaze, and sustained presence tracker
├── audio/
│   ├── __init__.py
│   └── audio_module.py        # RMS audio energy, VAD, and silence analyzer
├── analytics/
│   ├── __init__.py
│   └── fusion_engine.py       # Multimodal signal fusion & session dynamics engine
├── ai_agent/
│   ├── __init__.py
│   └── teaching_advisor.py     # Gemini 2.0 Flash pedagogical advisor
├── templates/
│   └── index.html              # Modern Classroom Intelligence Dashboard
└── static/
    ├── css/style.css           # Styling
    └── js/dashboard.js         # WebRTC video capture & Chart.js live visualizer
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. (Optional) Set Google Gemini API Key
To enable live Gemini 2.0 Flash pedagogical suggestions:
```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
```
*(If no API key is provided, the agent automatically runs in offline heuristic mode).*

### 3. Run the Application
```powershell
python app.py
```

### 4. Open Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- Click **"Start Session"** to begin real-time attendance and engagement analysis.
- Click **"Consult Gemini Advisor"** for real-time pedagogical interventions.
- Click **"Export CSV"** to download the verified attendance ledger.

