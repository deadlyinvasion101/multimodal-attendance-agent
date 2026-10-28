/**
 * Frontend Controller for Multimodal Attendance & Engagement Agent.
 * Handles webcam stream, microphone audio capture, Chart.js dynamics, and AI advisor updates.
 */

let videoElement = document.getElementById("webcam");
let canvasElement = document.getElementById("snapshotCanvas");
let isStreaming = false;
let sessionInterval = null;
let audioContext = null;
let audioAnalyser = null;
let audioDataArray = null;

// Initialize Real-time Engagement Chart
const ctx = document.getElementById("timelineChart").getContext("2d");
const timelineChart = new Chart(ctx, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Fused Engagement",
        data: [],
        borderColor: "#38bdf8",
        backgroundColor: "rgba(56, 189, 248, 0.1)",
        tension: 0.35,
        fill: true,
      },
      {
        label: "Attention / Gaze",
        data: [],
        borderColor: "#22c55e",
        borderDash: [5, 5],
        tension: 0.2,
      },
      {
        label: "Audio Activity",
        data: [],
        borderColor: "#f59e0b",
        borderDash: [2, 4],
        tension: 0.2,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    scales: {
      y: {
        min: 0,
        max: 1,
        grid: { color: "rgba(255, 255, 255, 0.08)" },
        ticks: { color: "#94a3b8" },
      },
      x: {
        grid: { color: "rgba(255, 255, 255, 0.08)" },
        ticks: { color: "#94a3b8" },
      },
    },
    plugins: {
      legend: { labels: { color: "#e2e8f0" } },
    },
  },
});

// Start/Stop Live Classroom Session
document.getElementById("toggleSessionBtn").addEventListener("click", async () => {
  if (!isStreaming) {
    try {
      // 1. Request Webcam & Microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: true,
      });

      videoElement.srcObject = stream;
      isStreaming = true;

      // 2. Setup Web Audio API for real-time PCM audio capture
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      audioAnalyser = audioContext.createAnalyser();
      audioAnalyser.fftSize = 256;
      source.connect(audioAnalyser);
      audioDataArray = new Float32Array(audioAnalyser.fftSize);

      // UI Update
      document.getElementById("toggleSessionBtn").innerText = "⏹ End Session";
      document.getElementById("toggleSessionBtn").className = "btn btn-danger";
      document.getElementById("sessionBadge").innerText = "● LIVE SESSION";
      document.getElementById("sessionBadge").className = "badge bg-danger";

      // 3. Start periodic multimodal telemetry (every 1.5s)
      sessionInterval = setInterval(dispatchMultimodalTelemetry, 1500);
    } catch (err) {
      alert("Microphone/Camera permission required: " + err.message);
    }
  } else {
    // Stop session
    if (videoElement.srcObject) {
      videoElement.srcObject.getTracks().forEach((track) => track.stop());
    }
    if (audioContext) {
      audioContext.close();
    }
    clearInterval(sessionInterval);
    isStreaming = false;

    // UI Update
    document.getElementById("toggleSessionBtn").innerText = "▶ Start Session";
    document.getElementById("toggleSessionBtn").className = "btn btn-success";
    document.getElementById("sessionBadge").innerText = "IDLE";
    document.getElementById("sessionBadge").className = "badge bg-secondary";
  }
});

// Periodic Telemetry Dispatcher
async function dispatchMultimodalTelemetry() {
  if (!isStreaming) return;

  // Capture video frame snapshot
  canvasElement.width = videoElement.videoWidth || 320;
  canvasElement.height = videoElement.videoHeight || 240;
  const ctx2d = canvasElement.getContext("2d");
  ctx2d.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
  const imageBase64 = canvasElement.toDataURL("image/jpeg", 0.65);

  // Capture raw audio samples
  let audioSamples = [];
  if (audioAnalyser && audioDataArray) {
    audioAnalyser.getFloatTimeDomainData(audioDataArray);
    audioSamples = Array.from(audioDataArray).slice(0, 128);
  }

  try {
    const res = await fetch("/api/process_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: imageBase64,
        audio: audioSamples,
      }),
    });

    const data = await res.json();
    updateDashboardUI(data);
  } catch (e) {
    console.error("Telemetry error:", e);
  }
}

// Update Dashboard View with Backend Telemetry
function updateDashboardUI(data) {
  const fused = data.fused || {};
  const vision = data.vision || {};
  const audio = data.audio || {};
  const attendance = data.attendance || {};

  // Update KPI counters
  const engPercent = Math.round((fused.fused_engagement || 0) * 100);
  document.getElementById("kpiEngagement").innerText = `${engPercent}%`;
  document.getElementById("kpiStatus").innerText = fused.status || "Analyzing";
  document.getElementById("kpiFaces").innerText = vision.face_count || 0;
  document.getElementById("kpiAttention").innerText = `${Math.round((vision.attention_score || 0) * 100)}%`;
  document.getElementById("kpiAudio").innerText = `${Math.round((audio.interaction_score || 0) * 100)}%`;

  // Update Timeline Chart
  timelineChart.data.labels.push(fused.timestamp);
  timelineChart.data.datasets[0].data.push(fused.fused_engagement);
  timelineChart.data.datasets[1].data.push(vision.attention_score);
  timelineChart.data.datasets[2].data.push(audio.interaction_score);

  if (timelineChart.data.labels.length > 20) {
    timelineChart.data.labels.shift();
    timelineChart.data.datasets[0].data.shift();
    timelineChart.data.datasets[1].data.shift();
    timelineChart.data.datasets[2].data.shift();
  }
  timelineChart.update();

  // Update Automated Attendance Table (Scenario 1)
  const tbody = document.getElementById("attendanceTbody");
  const keys = Object.keys(attendance);
  if (keys.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" class="text-center text-muted">Tracking presence...</td></tr>`;
  } else {
    tbody.innerHTML = "";
    keys.forEach((pid) => {
      const rec = attendance[pid];
      tbody.innerHTML += `
        <tr>
          <td><strong>${rec.participant || pid}</strong></td>
          <td><span class="badge badge-verified">✓ ${rec.status}</span></td>
          <td>${rec.first_verified}</td>
          <td>${rec.sustained_seconds}s</td>
        </tr>
      `;
    });
  }
}

// Fetch Gemini AI Teaching Advisor Suggestions (Scenario 3)
document.getElementById("fetchAiAdvisorBtn").addEventListener("click", async () => {
  const btn = document.getElementById("fetchAiAdvisorBtn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Consulting Gemini...`;

  try {
    const res = await fetch("/api/ai_advisor");
    const data = await res.json();
    const advisor = data.ai_advisor || {};

    document.getElementById("advisorSummary").innerText = advisor.status_summary || "Session insights ready.";
    document.getElementById("energyLevelBadge").innerText = `Energy: ${advisor.energy_level || "Balanced"}`;

    const listContainer = document.getElementById("suggestionsContainer");
    listContainer.innerHTML = "";

    if (advisor.suggestions && advisor.suggestions.length > 0) {
      advisor.suggestions.forEach((item) => {
        const priorityClass =
          item.priority === "High"
            ? "priority-high"
            : item.priority === "Medium"
            ? "priority-medium"
            : "priority-low";

        listContainer.innerHTML += `
          <div class="suggestion-item ${priorityClass}">
            <div class="d-flex justify-content-between align-items-center mb-1">
              <strong class="text-info">${item.category}</strong>
              <span class="badge bg-dark">${item.priority} Priority</span>
            </div>
            <p class="mb-0 small text-light">${item.action}</p>
          </div>
        `;
      });
    }
  } catch (err) {
    alert("Error fetching AI recommendations: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `✨ Consult Gemini Advisor`;
  }
});

// Reset Session
document.getElementById("resetSessionBtn").addEventListener("click", async () => {
  if (confirm("Reset current session data and attendance records?")) {
    await fetch("/api/reset_session", { method: "POST" });
    timelineChart.data.labels = [];
    timelineChart.data.datasets.forEach((ds) => (ds.data = []));
    timelineChart.update();
    document.getElementById("attendanceTbody").innerHTML =
      `<tr><td colspan="4" class="text-center text-muted">Session reset.</td></tr>`;
    document.getElementById("kpiEngagement").innerText = "0%";
    document.getElementById("kpiFaces").innerText = "0";
    document.getElementById("kpiAttention").innerText = "0%";
    document.getElementById("kpiAudio").innerText = "0%";
  }
});

// Recorded Video File Upload (Scenario 1)
document.getElementById("videoFileInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("video", file);

  const uploadMsg = document.getElementById("uploadStatusMsg");
  uploadMsg.innerText = "Processing video through Vision & Engagement Engine...";

  try {
    const res = await fetch("/api/process_video_file", {
      method: "POST",
      body: formData,
    });
    const result = await res.json();
    uploadMsg.innerText = `Processed ${result.processed_frames} sampled frames. Attendance updated!`;

    // Refresh attendance table
    const tbody = document.getElementById("attendanceTbody");
    tbody.innerHTML = "";
    for (const [pid, rec] of Object.entries(result.attendance || {})) {
      tbody.innerHTML += `
        <tr>
          <td><strong>${rec.participant || pid}</strong></td>
          <td><span class="badge badge-verified">✓ ${rec.status}</span></td>
          <td>${rec.first_verified}</td>
          <td>${rec.sustained_seconds}s</td>
        </tr>
      `;
    }
  } catch (err) {
    uploadMsg.innerText = "Error processing video: " + err.message;
  }
});

