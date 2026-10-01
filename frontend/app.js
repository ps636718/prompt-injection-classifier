/**
 * Prompt Injection Classifier — Client-Side Application Logic
 * Communicates with FastAPI backend, manages cold-start retries, and updates UI.
 */

// ── Configuration ─────────────────────────────────────────────────────────────
// Default backend URL: pointing directly to the deployed Render backend
const DEFAULT_API_URL = "https://prompt-injection-classifier.onrender.com";

function getApiBaseUrl() {
  const saved = localStorage.getItem("PROMPT_CLASSIFIER_API_URL");
  if (saved && saved.trim()) {
    return saved.trim().replace(/\/+$/, "");
  }
  return DEFAULT_API_URL;
}

function setApiBaseUrl(url) {
  if (!url || !url.trim()) {
    localStorage.removeItem("PROMPT_CLASSIFIER_API_URL");
  } else {
    localStorage.setItem("PROMPT_CLASSIFIER_API_URL", url.trim().replace(/\/+$/, ""));
  }
  updateEndpointDisplay();
  checkBackendHealth();
}

// ── DOM Elements ──────────────────────────────────────────────────────────────
const promptInput = document.getElementById("prompt-input");
const checkButton = document.getElementById("check-button");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");
const charCountSpan = document.getElementById("char-count");
const wordCountSpan = document.getElementById("word-count");
const clearBtn = document.getElementById("clear-btn");
const sampleChips = document.getElementById("sample-chips");

// Status indicators
const statusPill = document.getElementById("backend-status-pill");
const statusDot = document.getElementById("status-indicator");
const statusText = document.getElementById("status-text");
const retryNotice = document.getElementById("retry-notice");
const retryNoticeText = document.getElementById("retry-notice-text");

// Results
const resultContainer = document.getElementById("result-container");
const resultCard = document.getElementById("result-card");
const verdictBadge = document.getElementById("verdict-badge");
const verdictIcon = document.getElementById("verdict-icon");
const verdictText = document.getElementById("verdict-text");
const confidenceValue = document.getElementById("confidence-value");
const confidenceFill = document.getElementById("confidence-fill");
const verdictSummary = document.getElementById("verdict-summary");
const latencyValue = document.getElementById("latency-value");
const riskRating = document.getElementById("risk-rating");

// Settings
const currentEndpointDisplay = document.getElementById("current-endpoint-display");
const toggleConfigBtn = document.getElementById("toggle-config-btn");
const configDrawer = document.getElementById("config-drawer");
const backendUrlInput = document.getElementById("backend-url-input");
const saveApiBtn = document.getElementById("save-api-btn");
const resetApiBtn = document.getElementById("reset-api-btn");

// ── State ─────────────────────────────────────────────────────────────────────
let isChecking = false;

// ── Text Counter & Input Management ───────────────────────────────────────────
function updateCounts() {
  const text = promptInput.value || "";
  const chars = text.length;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;

  charCountSpan.textContent = `${chars.toLocaleString()} character${chars === 1 ? "" : "s"}`;
  wordCountSpan.textContent = `${words.toLocaleString()} word${words === 1 ? "" : "s"}`;
}

promptInput.addEventListener("input", updateCounts);

clearBtn.addEventListener("click", () => {
  promptInput.value = "";
  updateCounts();
  promptInput.focus();
  resultContainer.classList.remove("active");
  hideRetryNotice();
});

// Keyboard shortcut: Ctrl+Enter / Cmd+Enter
promptInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    if (!isChecking) {
      handleCheckPrompt();
    }
  }
});

// Quick-sample prompt chips
sampleChips.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip-btn");
  if (!chip) return;
  const sample = chip.getAttribute("data-sample");
  if (sample) {
    promptInput.value = sample;
    updateCounts();
    promptInput.focus();
    handleCheckPrompt();
  }
});

// ── Backend Health Monitoring ─────────────────────────────────────────────────
async function checkBackendHealth() {
  const apiBase = getApiBaseUrl();
  updateEndpointDisplay();

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const res = await fetch(`${apiBase}/health`, { signal: controller.signal });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      if (data.model_loaded) {
        setOnlineStatus("Online & Ready", "online");
      } else {
        setOnlineStatus("Model Loading...", "waking");
      }
    } else {
      setOnlineStatus("Backend Unreachable", "error");
    }
  } catch (err) {
    setOnlineStatus("Offline / Sleeping", "waking");
  }
}

function setOnlineStatus(label, stateClass) {
  statusText.textContent = label;
  statusDot.className = `status-dot ${stateClass}`;
}

function showRetryNotice(message) {
  retryNotice.style.display = "flex";
  retryNoticeText.textContent = message;
}

function hideRetryNotice() {
  retryNotice.style.display = "none";
}

// ── Core Prediction Handler with Cold-Start Retries ───────────────────────────
async function handleCheckPrompt() {
  const rawText = promptInput.value ? promptInput.value.trim() : "";
  if (!rawText) {
    promptInput.focus();
    return;
  }

  isChecking = true;
  setButtonLoading(true);
  hideRetryNotice();

  const apiBase = getApiBaseUrl();
  const startTime = performance.now();
  const maxRetries = 6;
  const retryDelayMs = 3500;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      if (attempt > 1) {
        showRetryNotice(`Waking up server instance (Render cold start)... Retry attempt ${attempt} of ${maxRetries}`);
      }

      // Allow longer timeout for cold starts (up to 25s per request)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 25000);

      const response = await fetch(`${apiBase}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: rawText }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        const durationMs = Math.round(performance.now() - startTime);
        hideRetryNotice();
        renderResult(data, durationMs);
        setOnlineStatus("Online & Ready", "online");
        break;
      } else if (response.status === 503) {
        // Model initializing on backend
        showRetryNotice(`Backend is loading ML model into memory... Retrying in a moment (${attempt}/${maxRetries})`);
        await new Promise((r) => setTimeout(r, retryDelayMs));
      } else {
        throw new Error(`Server returned HTTP ${response.status}`);
      }
    } catch (err) {
      console.warn(`Predict attempt ${attempt} failed:`, err);
      if (attempt < maxRetries) {
        showRetryNotice(`Waking up backend server (free tier cold start can take 30-50s)... Retrying automatically (${attempt}/${maxRetries})`);
        await new Promise((r) => setTimeout(r, retryDelayMs));
      } else {
        // Reached end of retries
        showRetryNotice(`Could not reach the classifier at ${apiBase}. Please check your connection or backend URL in footer settings.`);
        setOnlineStatus("Unreachable", "error");
      }
    }
  }

  setButtonLoading(false);
  isChecking = false;
}

function setButtonLoading(loading) {
  if (loading) {
    checkButton.disabled = true;
    btnText.textContent = "Checking...";
    btnSpinner.style.display = "inline-block";
  } else {
    checkButton.disabled = false;
    btnText.textContent = "Check Prompt";
    btnSpinner.style.display = "none";
  }
}

// ── Render Prediction Result ──────────────────────────────────────────────────
function renderResult(data, durationMs) {
  const label = (data.label || "BENIGN").toUpperCase();
  const confidence = typeof data.confidence === "number" ? data.confidence : 1.0;
  const pct = Math.round(confidence * 1000) / 10; // e.g. 96.4%

  resultCard.className = `result-card ${label === "MALICIOUS" ? "verdict-malicious" : "verdict-benign"}`;

  // Verdict badge
  verdictBadge.className = `verdict-badge ${label === "MALICIOUS" ? "malicious" : "benign"}`;
  verdictText.textContent = label === "MALICIOUS" ? "MALICIOUS DETECTED" : "BENIGN PROMPT";
  verdictIcon.textContent = label === "MALICIOUS" ? "⚠️" : "🛡️";

  // Confidence & fill
  confidenceValue.textContent = `${pct}%`;
  confidenceFill.className = `confidence-fill ${label === "MALICIOUS" ? "malicious" : "benign"}`;
  // Trigger animation after DOM paint
  confidenceFill.style.width = "0%";
  setTimeout(() => {
    confidenceFill.style.width = `${pct}%`;
  }, 40);

  // Verdict Summary & Risk Rating
  if (label === "MALICIOUS") {
    verdictSummary.textContent =
      `Adversarial intent or injection pattern identified with ${pct}% certainty. This prompt should be blocked or sanitized before forwarding to any downstream LLM.`;
    riskRating.textContent = "HIGH RISK (BLOCK)";
    riskRating.style.color = "var(--color-negative)";
  } else {
    verdictSummary.textContent =
      `No adversarial jailbreak or prompt injection patterns detected (${pct}% confidence). The input appears safe for standard model generation.`;
    riskRating.textContent = "LOW RISK (ALLOW)";
    riskRating.style.color = "var(--color-positive)";
  }

  latencyValue.textContent = `${durationMs} ms`;
  resultContainer.classList.add("active");
  resultContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── API Configuration Drawer ──────────────────────────────────────────────────
function updateEndpointDisplay() {
  const current = getApiBaseUrl();
  currentEndpointDisplay.textContent = current;
  backendUrlInput.value = current;
}

toggleConfigBtn.addEventListener("click", () => {
  configDrawer.classList.toggle("open");
});

saveApiBtn.addEventListener("click", () => {
  const newUrl = backendUrlInput.value.trim();
  setApiBaseUrl(newUrl);
  configDrawer.classList.remove("open");
});

resetApiBtn.addEventListener("click", () => {
  localStorage.removeItem("PROMPT_CLASSIFIER_API_URL");
  updateEndpointDisplay();
  configDrawer.classList.remove("open");
  checkBackendHealth();
});

checkButton.addEventListener("click", handleCheckPrompt);

// ── Init on load ──────────────────────────────────────────────────────────────
updateCounts();
updateEndpointDisplay();
checkBackendHealth();
