/**
 * OmniReID Live Dashboard Controller
 * =================================
 * Handles real-time polling, state-machine synchronization,
 * gallery card rendering, merge event logs, and camera controls.
 */

let isPolling = true;
let currentState = "IDLE";

document.addEventListener("DOMContentLoaded", () => {
  pollStatus();
  setInterval(pollStatus, 1500);
});

// -----------------------------------------------------------------------------
// Status & State Machine Synchronization
// -----------------------------------------------------------------------------
async function pollStatus() {
  if (!isPolling) return;
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();

    currentState = data.state;
    updateStateUI(data.state, data.session_id);
    updateCamerasHealth(data.cameras || {});
    if (data.gallery !== undefined) renderGallery(data.gallery);
    if (data.events !== undefined) renderEvents(data.events);
  } catch (err) {
    console.error("Status poll error:", err);
  }
}

function updateStateUI(state, sessionId) {
  const pill = document.getElementById("systemStatePill");
  const text = document.getElementById("systemStateText");
  const btnProbe = document.getElementById("btnProbe");
  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const sessionContainer = document.getElementById("sessionBadgeContainer");
  const sessionIdText = document.getElementById("sessionIdText");

  text.textContent = state;
  pill.className = "status-pill " + state.toLowerCase();

  if (sessionId) {
    sessionContainer.style.display = "flex";
    sessionIdText.textContent = sessionId;
  } else {
    sessionContainer.style.display = "none";
  }

  // Button State Logic and stream connection lifecycle based on state
  switch (state) {
    case "IDLE":
      btnProbe.disabled = false;
      btnStart.disabled = true;
      btnStop.disabled = true;
      // Disconnect stream sockets when idle to prevent browser socket exhaustion
      document.querySelectorAll('.video-feed').forEach(img => {
        const camId = img.dataset.camId;
        img.removeAttribute("src");
        const ph = document.getElementById(`placeholder_${camId}`);
        if (ph) ph.classList.remove("hidden");
      });
      break;
    case "PROBING":
      btnProbe.disabled = true;
      btnStart.disabled = true;
      btnStop.disabled = true;
      break;
    case "READY":
      btnProbe.disabled = false;
      btnStart.disabled = false;
      btnStop.disabled = true;
      break;
    case "RUNNING":
      btnProbe.disabled = true;
      btnStart.disabled = true;
      btnStop.disabled = false;
      // Connect live streams for enabled cameras when running
      document.querySelectorAll('.video-feed[data-enabled="true"]').forEach(img => {
        const camId = img.dataset.camId;
        if (!img.src || !img.src.includes(`/stream/${camId}`)) {
          img.src = `/stream/${camId}?t=${Date.now()}`;
          const ph = document.getElementById(`placeholder_${camId}`);
          if (ph) ph.classList.add("hidden");
        }
      });
      break;
    case "STOPPING":
      btnProbe.disabled = true;
      btnStart.disabled = true;
      btnStop.disabled = true;
      break;
  }
}

function updateCamerasHealth(cameras) {
  for (const [camId, health] of Object.entries(cameras)) {
    const fpsElem = document.getElementById(`fps_${camId}`);
    const statElem = document.getElementById(`stat_${camId}`);
    const resElem = document.getElementById(`res_${camId}`);

    if (fpsElem) fpsElem.textContent = `${health.fps.toFixed(1)} FPS`;
    if (resElem) resElem.textContent = health.resolution;
    if (statElem) {
      statElem.textContent = health.status;
      statElem.className = `status-indicator ${health.status.toLowerCase()}`;
    }
  }
}

// -----------------------------------------------------------------------------
// Discovered Persons Gallery
// -----------------------------------------------------------------------------
function renderGallery(list) {
  list = list || [];
  document.getElementById("personCount").textContent = list.length;
  const container = document.getElementById("galleryList");

  if (list.length === 0) {
    container.innerHTML = '<div class="empty-state">No targets discovered yet. Analyzing streams...</div>';
    return;
  }

  container.innerHTML = list.map(item => {
    const rgb = `rgb(${item.color_rgb[0]}, ${item.color_rgb[1]}, ${item.color_rgb[2]})`;
    const camBadges = item.active_cameras.map(c => `<span class="cam-chip">${c}</span>`).join(" ");

    return `
      <div class="person-card">
        <div class="person-color-chip" style="background: ${rgb}; box-shadow: 0 0 10px ${rgb}44;"></div>
        <div class="person-info">
          <div class="person-name">
            <span>${item.name}</span>
            <span style="font-size: 0.7rem; color: var(--text-muted);">${item.gid}</span>
          </div>
          <div class="person-meta">
            <span>Exemplars: ${item.exemplar_count}</span>
            <span>Matches: ${item.total_matches}</span>
          </div>
          <div class="person-cameras">
            ${camBadges || '<span style="font-size: 0.68rem; color: var(--text-muted);">Not currently visible</span>'}
          </div>
        </div>
      </div>
    `;
  }).join("");
}

// -----------------------------------------------------------------------------
// Live Activity & Merge Event Feed
// -----------------------------------------------------------------------------
function renderEvents(events) {
  events = events || [];
  const container = document.getElementById("eventFeed");

  if (events.length === 0) {
    container.innerHTML = '<div class="empty-state">Waiting for activity...</div>';
    return;
  }

  container.innerHTML = events.slice(0, 30).map(evt => {
    const typeClass = evt.type.toLowerCase();
    return `
      <div class="event-card ${typeClass}">
        <div class="event-header">
          <span class="event-type">${evt.type.replace("_", " ")}</span>
          <span class="event-time">${evt.timestamp.toFixed(1)}s</span>
        </div>
        <div class="event-msg">${evt.message}</div>
      </div>
    `;
  }).join("");
}

// -----------------------------------------------------------------------------
// User Action Handlers
// -----------------------------------------------------------------------------
async function probeCameras() {
  const btn = document.getElementById("btnProbe");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Probing Streams...';

  try {
    const res = await fetch("/api/probe", { method: "POST" });
    const data = await res.json();

    if (data.details) {
      for (const [camId, r] of Object.entries(data.details)) {
        const fpsElem = document.getElementById(`fps_${camId}`);
        const statElem = document.getElementById(`stat_${camId}`);
        const resElem = document.getElementById(`res_${camId}`);
        const img = document.getElementById(`stream_${camId}`);
        const ph = document.getElementById(`placeholder_${camId}`);

        if (fpsElem) fpsElem.textContent = `${r.fps.toFixed(1)} FPS`;
        if (resElem) resElem.textContent = r.resolution;
        if (statElem) {
          const isHealthy = r.ok && !r.standby;
          statElem.textContent = isHealthy ? "HEALTHY" : (r.standby ? "STANDBY" : "FAILED");
          statElem.className = `status-indicator ${isHealthy ? 'running' : (r.standby ? 'standby' : 'failed')}`;
        }
        // Load single instant snapshot to preview camera framing
        if (r.ok && !r.standby && img) {
          img.src = `/snapshot/${camId}?t=${Date.now()}`;
          if (ph) ph.classList.add("hidden");
        }
      }
    }

    alert(`Camera probe finished: ${data.passing_cameras}.\nClick Start Analysis to run every camera that passed.`);
  } catch (err) {
    alert("Error probing cameras: " + err);
  } finally {
    btn.innerHTML = '<span class="btn-icon">🔍</span> Test Cameras';
    pollStatus();
  }
}

async function startAnalysis() {
  const btn = document.getElementById("btnStart");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Starting...';

  try {
    const res = await fetch("/api/start", { method: "POST" });
    const data = await res.json();
    if (data.session_id) {
      console.log("Started session:", data.session_id);
    }
  } catch (err) {
    alert("Error starting analysis: " + err);
  } finally {
    btn.innerHTML = '<span class="btn-icon">▶</span> Start Analysis';
    pollStatus();
  }
}

async function stopAnalysis() {
  const btn = document.getElementById("btnStop");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Stopping...';

  try {
    await fetch("/api/stop", { method: "POST" });
  } catch (err) {
    alert("Error stopping analysis: " + err);
  } finally {
    btn.innerHTML = '<span class="btn-icon">⏹</span> Stop Analysis';
    pollStatus();
  }
}

// -----------------------------------------------------------------------------
// RTSP Modal Configuration
// -----------------------------------------------------------------------------
function openConfigModal() {
  document.getElementById("configModal").style.display = "flex";
}

function closeConfigModal() {
  document.getElementById("configModal").style.display = "none";
}

async function updateTripwire(val) {
  try {
    await fetch("/api/tripwire", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ y_pct: val }),
    });
    if (currentState !== "RUNNING") {
      const img = document.getElementById("stream_office_balcony");
      const ph = document.getElementById("placeholder_office_balcony");
      if (img) {
        img.src = `/snapshot/office_balcony?t=${Date.now()}`;
        if (ph) ph.classList.add("hidden");
      }
    }
  } catch (err) {
    console.error("Tripwire update failed:", err);
  }
}

async function saveCameraUrl(camId) {
  const input = document.getElementById(`cfg_input_${camId}`);
  const newUrl = input.value.trim();

  try {
    const res = await fetch("/api/camera/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam_id: camId, rtsp: newUrl }),
    });
    const data = await res.json();
    if (data.success) {
      alert(`Updated RTSP URL for ${camId}!`);
    } else {
      alert("Failed to update URL");
    }
  } catch (err) {
    alert("Error saving URL: " + err);
  }
}
