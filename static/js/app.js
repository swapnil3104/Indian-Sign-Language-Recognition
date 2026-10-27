/**
 * Indian Sign Language (ISL) Recognition - Frontend Application
 * High-performance real-time hand tracking, gesture prediction, & translator
 */

// State Management
const state = {
  activeTab: 'studio',
  cameraMode: 'client', // 'client' (browser) or 'server' (opencv)
  isClientCamRunning: false,
  isServerCamRunning: false,
  stream: null,
  hands: null,
  camera: null,
  lastPredictionTime: 0,
  predictionInterval: 120, // ms between API requests (~8-10 fps inference)
  currentPrediction: null,
  currentConfidence: 0,
  stableSign: null,
  stableSignCount: 0,
  autoAddThreshold: 12, // stable frames before auto-adding letter
  composedText: '',
  audioEnabled: true,
  quiz: {
    active: false,
    targetSign: null,
    score: 0,
    streak: 0,
    timeLeft: 20,
    timerId: null,
    matchCounter: 0
  },
  classes: ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
};

// Audio Synthesizer (Web Audio API)
let audioCtx = null;
function getAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

function playBeep(frequency = 520, type = 'sine', duration = 0.12, volume = 0.08) {
  if (!state.audioEnabled) return;
  try {
    const ctx = getAudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(frequency, ctx.currentTime);
    gain.gain.setValueAtTime(volume, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) {
    console.warn("Audio play prevented:", e);
  }
}

function playTriumphChord() {
  if (!state.audioEnabled) return;
  [523.25, 659.25, 783.99, 1046.50].forEach((freq, idx) => {
    setTimeout(() => playBeep(freq, 'triangle', 0.35, 0.12), idx * 75);
  });
}

// DOM Elements
const elements = {
  navTabs: document.querySelectorAll('.nav-btn'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  clientVideo: document.getElementById('clientVideo'),
  clientCanvas: document.getElementById('clientCanvas'),
  serverStream: document.getElementById('serverStream'),
  videoPlaceholder: document.getElementById('videoPlaceholder'),
  startCamBtn: document.getElementById('startCamBtn'),
  stopCamBtn: document.getElementById('stopCamBtn'),
  switchCameraModeBtns: document.querySelectorAll('.mode-tab-btn'),
  detectedChar: document.getElementById('detectedChar'),
  confidenceBar: document.getElementById('confidenceBar'),
  confidenceVal: document.getElementById('confidenceVal'),
  candidatesList: document.getElementById('candidatesList'),
  composedDisplay: document.getElementById('composedDisplay'),
  btnSpace: document.getElementById('btnSpace'),
  btnBackspace: document.getElementById('btnBackspace'),
  btnClear: document.getElementById('btnClear'),
  btnSpeak: document.getElementById('btnSpeak'),
  btnCopy: document.getElementById('btnCopy'),
  btnSnapshot: document.getElementById('btnSnapshot'),
  audioToggleBtn: document.getElementById('audioToggleBtn'),
  fpsDisplay: document.getElementById('fpsDisplay'),
  latencyDisplay: document.getElementById('latencyDisplay'),
  // Quiz
  startQuizBtn: document.getElementById('startQuizBtn'),
  nextQuizBtn: document.getElementById('nextQuizBtn'),
  quizTargetChar: document.getElementById('quizTargetChar'),
  quizScore: document.getElementById('quizScore'),
  quizStreak: document.getElementById('quizStreak'),
  quizTime: document.getElementById('quizTime'),
  quizFeedback: document.getElementById('quizFeedback'),
  // Reference
  referenceGrid: document.getElementById('referenceGrid'),
  referenceSearch: document.getElementById('referenceSearch'),
  filterBtns: document.querySelectorAll('.filter-btn'),
  chartModal: document.getElementById('chartModal'),
  openChartBtn: document.getElementById('openChartBtn'),
  closeChartBtn: document.getElementById('closeChartBtn'),
  // Toast
  toast: document.getElementById('toast'),
  toastMessage: document.getElementById('toastMessage')
};

// Canvas 2D Context
const ctx = elements.clientCanvas.getContext('2d');

// -----------------------------------------------------------------------------
// Navigation Tabs
// -----------------------------------------------------------------------------
elements.navTabs.forEach(btn => {
  btn.addEventListener('click', () => {
    const targetTab = btn.getAttribute('data-tab');
    elements.navTabs.forEach(b => b.classList.remove('active'));
    elements.tabPanes.forEach(pane => pane.classList.remove('active'));

    btn.classList.add('active');
    const activePane = document.getElementById(targetTab);
    if (activePane) activePane.classList.add('active');
    state.activeTab = targetTab;
  });
});

// -----------------------------------------------------------------------------
// Toast Notification
// -----------------------------------------------------------------------------
function showToast(message) {
  elements.toastMessage.textContent = message;
  elements.toast.classList.add('show');
  setTimeout(() => {
    elements.toast.classList.remove('show');
  }, 2600);
}

// -----------------------------------------------------------------------------
// Audio Mute Toggle
// -----------------------------------------------------------------------------
elements.audioToggleBtn.addEventListener('click', () => {
  state.audioEnabled = !state.audioEnabled;
  elements.audioToggleBtn.innerHTML = state.audioEnabled
    ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>`
    : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line></svg>`;
  showToast(state.audioEnabled ? "Audio speech and effects ON" : "Audio MUTED");
});

// -----------------------------------------------------------------------------
// Camera Mode Switch (Browser vs Server OpenCV)
// -----------------------------------------------------------------------------
elements.switchCameraModeBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const mode = btn.getAttribute('data-mode');
    if (mode === state.cameraMode) return;

    elements.switchCameraModeBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    switchCameraMode(mode);
  });
});

function switchCameraMode(newMode) {
  // Stop current stream if running
  if (state.cameraMode === 'client' && state.isClientCamRunning) {
    stopClientCamera();
  } else if (state.cameraMode === 'server' && state.isServerCamRunning) {
    stopServerCamera();
  }

  state.cameraMode = newMode;
  if (newMode === 'client') {
    elements.clientVideo.style.display = 'block';
    elements.clientCanvas.style.display = 'block';
    elements.serverStream.style.display = 'none';
  } else {
    elements.clientVideo.style.display = 'none';
    elements.clientCanvas.style.display = 'none';
    elements.serverStream.style.display = 'block';
  }
  elements.videoPlaceholder.style.display = 'flex';
}

// -----------------------------------------------------------------------------
// Client-Side Browser Webcam & MediaPipe
// -----------------------------------------------------------------------------
async function initMediaPipeHands() {
  if (state.hands) return;

  state.hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  state.hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 0,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  state.hands.onResults(onHandResults);
}

async function startClientCamera() {
  try {
    await initMediaPipeHands();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      },
      audio: false
    });

    state.stream = stream;
    elements.clientVideo.srcObject = stream;
    await elements.clientVideo.play();

    // Match canvas dimensions to video
    elements.clientCanvas.width = elements.clientVideo.videoWidth || 640;
    elements.clientCanvas.height = elements.clientVideo.videoHeight || 480;

    state.camera = new Camera(elements.clientVideo, {
      onFrame: async () => {
        if (!state.isClientCamRunning) return;
        await state.hands.send({ image: elements.clientVideo });
      },
      width: 640,
      height: 480
    });
    await state.camera.start();

    state.isClientCamRunning = true;
    elements.videoPlaceholder.style.display = 'none';
    elements.startCamBtn.style.display = 'none';
    elements.stopCamBtn.style.display = 'inline-flex';
    showToast("Browser webcam active with MediaPipe tracking");
  } catch (err) {
    console.error("Camera access error:", err);
    showToast("Webcam error: " + (err.message || "Could not open camera"));
  }
}

function stopClientCamera() {
  if (state.camera) {
    state.camera.stop();
  }
  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
  }
  state.isClientCamRunning = false;
  ctx.clearRect(0, 0, elements.clientCanvas.width, elements.clientCanvas.height);
  elements.videoPlaceholder.style.display = 'flex';
  elements.startCamBtn.style.display = 'inline-flex';
  elements.stopCamBtn.style.display = 'none';
}

// -----------------------------------------------------------------------------
// Server OpenCV MJPEG Stream Handling
// -----------------------------------------------------------------------------
function startServerCamera() {
  fetch('/api/camera/start', { method: 'POST' })
    .then(r => r.json())
    .then(() => {
      elements.serverStream.src = '/video_feed?' + new Date().getTime();
      state.isServerCamRunning = true;
      elements.videoPlaceholder.style.display = 'none';
      elements.startCamBtn.style.display = 'none';
      elements.stopCamBtn.style.display = 'inline-flex';
      showToast("Server OpenCV stream active");
    })
    .catch(e => {
      showToast("Failed to start server camera stream");
    });
}

function stopServerCamera() {
  fetch('/api/camera/stop', { method: 'POST' })
    .finally(() => {
      elements.serverStream.src = '';
      state.isServerCamRunning = false;
      elements.videoPlaceholder.style.display = 'flex';
      elements.startCamBtn.style.display = 'inline-flex';
      elements.stopCamBtn.style.display = 'none';
    });
}

// Master Camera Buttons
elements.startCamBtn.addEventListener('click', () => {
  if (state.cameraMode === 'client') {
    startClientCamera();
  } else {
    startServerCamera();
  }
});

elements.stopCamBtn.addEventListener('click', () => {
  if (state.cameraMode === 'client') {
    stopClientCamera();
  } else {
    stopServerCamera();
  }
});

// -----------------------------------------------------------------------------
// Hand Landmark Drawing & Prediction Pipeline
// -----------------------------------------------------------------------------
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],         // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],         // Index
  [5, 9], [9, 10], [10, 11], [11, 12],    // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],  // Ring
  [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [0, 17]                                 // Palm base
];

let frameCount = 0;
let lastFpsCheck = performance.now();

function onHandResults(results) {
  // Update FPS counter
  frameCount++;
  const now = performance.now();
  if (now - lastFpsCheck >= 1000) {
    const fps = Math.round((frameCount * 1000) / (now - lastFpsCheck));
    elements.fpsDisplay.textContent = `${fps} FPS`;
    frameCount = 0;
    lastFpsCheck = now;
  }

  // Clear overlay canvas
  ctx.save();
  ctx.clearRect(0, 0, elements.clientCanvas.width, elements.clientCanvas.height);

  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    ctx.restore();
    return;
  }

  const w = elements.clientCanvas.width;
  const h = elements.clientCanvas.height;

  // Process first detected hand
  const landmarks = results.multiHandLandmarks[0];

  // 1. Draw clean hand skeleton lines
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = '#2563eb';
  ctx.shadowBlur = 0;

  HAND_CONNECTIONS.forEach(([i, j]) => {
    const p1 = landmarks[i];
    const p2 = landmarks[j];
    ctx.beginPath();
    ctx.moveTo(p1.x * w, p1.y * h);
    ctx.lineTo(p2.x * w, p2.y * h);
    ctx.stroke();
  });

  // 2. Draw crisp landmark joints
  landmarks.forEach((p, index) => {
    ctx.beginPath();
    const radius = (index === 4 || index === 8 || index === 12 || index === 16 || index === 20) ? 5 : 3.5;
    ctx.arc(p.x * w, p.y * h, radius, 0, 2 * Math.PI);
    ctx.fillStyle = (index === 0) ? '#10b981' : '#3b82f6';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
  });

  ctx.restore();

  // 3. Send landmarks to prediction API at throttled rate
  if (now - state.lastPredictionTime >= state.predictionInterval) {
    state.lastPredictionTime = now;
    predictLandmarks(landmarks);
  }
}

// -----------------------------------------------------------------------------
// Predict via Backend API
// -----------------------------------------------------------------------------
async function predictLandmarks(landmarks) {
  // Convert 21 normalized landmarks to [x, y] in pixel coordinates
  const w = elements.clientCanvas.width;
  const h = elements.clientCanvas.height;

  const rawPoints = landmarks.map(p => [
    Math.min(Math.floor(p.x * w), w - 1),
    Math.min(Math.floor(p.y * h), h - 1)
  ]);

  const startTime = performance.now();
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks: rawPoints })
    });
    const data = await res.json();
    const latency = Math.round(performance.now() - startTime);
    elements.latencyDisplay.textContent = `${latency} ms`;

    if (data.success) {
      handlePredictionResult(data);
    }
  } catch (err) {
    console.error("Prediction fetch failed:", err);
  }
}

function handlePredictionResult(data) {
  const label = data.label;
  const conf = data.confidence;

  state.currentPrediction = label;
  state.currentConfidence = conf;

  // Update HUD
  elements.detectedChar.textContent = label;
  elements.detectedChar.classList.add('bounce');
  setTimeout(() => elements.detectedChar.classList.remove('bounce'), 200);

  elements.confidenceBar.style.width = `${conf}%`;
  elements.confidenceVal.textContent = `${conf.toFixed(1)}%`;

  // Update Top 5 candidate bars
  if (data.top5) {
    elements.candidatesList.innerHTML = data.top5.map(c => `
      <div class="candidate-row">
        <span class="candidate-label">${c.label}</span>
        <div class="candidate-bar-bg">
          <div class="candidate-bar-fill" style="width: ${c.confidence}%"></div>
        </div>
        <span class="candidate-val" style="color: #94a3b8">${c.confidence}%</span>
      </div>
    `).join('');
  }

  // Quiz evaluation
  if (state.quiz.active && label === state.quiz.targetSign && conf >= 65) {
    state.quiz.matchCounter++;
    if (state.quiz.matchCounter >= 4) { // hold for ~400ms
      quizSignSuccess();
    }
  } else {
    state.quiz.matchCounter = 0;
  }

  // Auto-typing stability check
  if (conf >= 70) {
    if (label === state.stableSign) {
      state.stableSignCount++;
      if (state.stableSignCount === state.autoAddThreshold) {
        appendLetter(label);
        playBeep(740, 'sine', 0.1, 0.08);
      }
    } else {
      state.stableSign = label;
      state.stableSignCount = 1;
    }
  } else {
    state.stableSign = null;
    state.stableSignCount = 0;
  }
}

// -----------------------------------------------------------------------------
// Sentence Composer Actions
// -----------------------------------------------------------------------------
function appendLetter(char) {
  state.composedText += char;
  updateComposerDisplay();
}

function updateComposerDisplay() {
  elements.composedDisplay.innerHTML = state.composedText
    ? `${state.composedText}<span class="composer-cursor"></span>`
    : '';
}

elements.btnSpace.addEventListener('click', () => {
  state.composedText += ' ';
  updateComposerDisplay();
  playBeep(440, 'sine', 0.08);
});

elements.btnBackspace.addEventListener('click', () => {
  state.composedText = state.composedText.slice(0, -1);
  updateComposerDisplay();
  playBeep(330, 'sine', 0.08);
});

elements.btnClear.addEventListener('click', () => {
  state.composedText = '';
  updateComposerDisplay();
  playBeep(260, 'sine', 0.12);
  showToast("Sentence cleared");
});

elements.btnCopy.addEventListener('click', () => {
  if (!state.composedText.trim()) {
    showToast("Nothing to copy!");
    return;
  }
  navigator.clipboard.writeText(state.composedText)
    .then(() => showToast("Copied text to clipboard!"))
    .catch(() => showToast("Failed to copy"));
});

// Text-to-Speech (Web Speech API)
elements.btnSpeak.addEventListener('click', () => {
  const text = state.composedText.trim();
  if (!text) {
    showToast("Please compose some text first!");
    return;
  }
  if (!('speechSynthesis' in window)) {
    showToast("Speech synthesis not supported in this browser");
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.95;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
  showToast(`Speaking: "${text}"`);
});

// -----------------------------------------------------------------------------
// Practice & Quiz Arena
// -----------------------------------------------------------------------------
elements.startQuizBtn.addEventListener('click', () => {
  startQuizGame();
});

elements.nextQuizBtn.addEventListener('click', () => {
  pickNewQuizTarget();
});

function startQuizGame() {
  state.quiz.active = true;
  state.quiz.score = 0;
  state.quiz.streak = 0;
  elements.quizScore.textContent = '0';
  elements.quizStreak.textContent = '0';
  elements.startQuizBtn.style.display = 'none';
  elements.nextQuizBtn.style.display = 'inline-flex';

  pickNewQuizTarget();
  showToast("Quiz started! Show the target sign to the camera!");
}

function pickNewQuizTarget() {
  const randIdx = Math.floor(Math.random() * state.classes.length);
  state.quiz.targetSign = state.classes[randIdx];
  state.quiz.timeLeft = 20;
  state.quiz.matchCounter = 0;

  elements.quizTargetChar.textContent = state.quiz.targetSign;
  elements.quizFeedback.textContent = `Show sign for "${state.quiz.targetSign}"`;
  elements.quizFeedback.style.color = '#94a3b8';

  if (state.quiz.timerId) clearInterval(state.quiz.timerId);
  state.quiz.timerId = setInterval(() => {
    state.quiz.timeLeft--;
    elements.quizTime.textContent = `${state.quiz.timeLeft}s`;
    if (state.quiz.timeLeft <= 0) {
      clearInterval(state.quiz.timerId);
      quizTimeout();
    }
  }, 1000);
}

function quizSignSuccess() {
  clearInterval(state.quiz.timerId);
  state.quiz.score += 100 + (state.quiz.streak * 20);
  state.quiz.streak++;
  elements.quizScore.textContent = state.quiz.score;
  elements.quizStreak.textContent = state.quiz.streak;

  playTriumphChord();
  elements.quizFeedback.textContent = `🎯 Excellent! Sign "${state.quiz.targetSign}" recognized!`;
  elements.quizFeedback.style.color = '#10b981';

  showToast(`+100 Points! Streak: ${state.quiz.streak}`);

  setTimeout(() => {
    if (state.quiz.active) pickNewQuizTarget();
  }, 1400);
}

function quizTimeout() {
  state.quiz.streak = 0;
  elements.quizStreak.textContent = '0';
  playBeep(220, 'sawtooth', 0.25);
  elements.quizFeedback.textContent = `⏰ Time's up! Target was "${state.quiz.targetSign}"`;
  elements.quizFeedback.style.color = '#f43f5e';

  setTimeout(() => {
    if (state.quiz.active) pickNewQuizTarget();
  }, 1800);
}

// -----------------------------------------------------------------------------
// Reference Gallery & Search Filter
// -----------------------------------------------------------------------------
function renderReferenceCards(filter = 'all', searchQuery = '') {
  elements.referenceGrid.innerHTML = '';

  const vowels = ['A', 'E', 'I', 'O', 'U'];
  const numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9'];

  const filtered = state.classes.filter(char => {
    const isNum = numbers.includes(char);
    const isVowel = vowels.includes(char);

    let matchCategory = true;
    if (filter === 'numbers') matchCategory = isNum;
    else if (filter === 'letters') matchCategory = !isNum;
    else if (filter === 'vowels') matchCategory = isVowel;

    let matchSearch = true;
    if (searchQuery) {
      matchSearch = char.toLowerCase().includes(searchQuery.toLowerCase());
    }

    return matchCategory && matchSearch;
  });

  filtered.forEach(char => {
    const isNum = numbers.includes(char);
    const typeLabel = isNum ? 'Digit' : vowels.includes(char) ? 'Vowel' : 'Letter';

    const card = document.createElement('div');
    card.className = 'sign-card';
    card.innerHTML = `
      <div class="sign-card-char">${char}</div>
      <div class="sign-card-type">${typeLabel}</div>
    `;

    card.addEventListener('click', () => {
      playBeep(600, 'sine', 0.08);
      showToast(`Sign ${char} (${typeLabel}) selected`);
    });

    elements.referenceGrid.appendChild(card);
  });
}

elements.filterBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    elements.filterBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const filter = btn.getAttribute('data-filter');
    renderReferenceCards(filter, elements.referenceSearch.value.trim());
  });
});

elements.referenceSearch.addEventListener('input', (e) => {
  const activeBtn = document.querySelector('.filter-btn.active');
  const filter = activeBtn ? activeBtn.getAttribute('data-filter') : 'all';
  renderReferenceCards(filter, e.target.value.trim());
});

// Full Chart Modal
elements.openChartBtn.addEventListener('click', () => {
  elements.chartModal.classList.add('open');
});

elements.closeChartBtn.addEventListener('click', () => {
  elements.chartModal.classList.remove('open');
});

elements.chartModal.addEventListener('click', (e) => {
  if (e.target === elements.chartModal) {
    elements.chartModal.classList.remove('open');
  }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  renderReferenceCards();
  console.log("ISL Web Application initialized.");
});
