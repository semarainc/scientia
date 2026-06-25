const API_BASE = "http://127.0.0.1:8000";

// ---- Elements ----
const dropZone       = document.getElementById("dropZone");
const fileInput      = document.getElementById("fileInput");
const browseBtn      = document.getElementById("browseBtn");
const preview        = document.getElementById("preview");
const previewImg     = document.getElementById("previewImg");
const previewName    = document.getElementById("previewName");
const changeBtn      = document.getElementById("changeBtn");
const analyzeBtn     = document.getElementById("analyzeBtn");
const uploadSection  = document.getElementById("uploadSection");
const analysisLayout = document.getElementById("analysisLayout");
const resultCard     = document.getElementById("resultCard");
const loadingOverlay = document.getElementById("loadingOverlay");
const errorToast     = document.getElementById("errorToast");
const errorMsg       = document.getElementById("errorMsg");
const uploadHeading  = document.getElementById("uploadHeading");
const uploadDesc     = document.getElementById("uploadDesc");

const modeBadge      = document.getElementById("modeBadge");
const verdictIconWrap= document.getElementById("verdictIconWrap");
const verdictLabel   = document.getElementById("verdictLabel");
const verdictConf    = document.getElementById("verdictConf");
const barNormal      = document.getElementById("barNormal");
const barPneumonia   = document.getElementById("barPneumonia");
const pctNormal      = document.getElementById("pctNormal");
const pctPneumonia   = document.getElementById("pctPneumonia");
const resetBtn       = document.getElementById("resetBtn");

let selectedFile = null;

// ---- SVG icons ----
const ICON_NORMAL = `
  <svg viewBox="0 0 24 24" fill="none" stroke="#3fb950" stroke-width="2.5">
    <polyline points="20 6 9 17 4 12"/>
  </svg>`;

const ICON_PNEUMONIA = `
  <svg viewBox="0 0 24 24" fill="none" stroke="#f85149" stroke-width="2.5">
    <line x1="18" y1="6" x2="6" y2="18"/>
    <line x1="6" y1="6" x2="18" y2="18"/>
  </svg>`;

// ---- File selection ----
browseBtn.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("click", (e) => {
  if (e.target === browseBtn) return;
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// Drag & drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("File harus berupa gambar (JPEG atau PNG).");
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewName.textContent = file.name;
  dropZone.hidden = true;
  preview.hidden = false;
  analyzeBtn.disabled = false;
  analyzeBtn.hidden = false;
  resultCard.hidden = true;
  analysisLayout.classList.remove("has-result");
  uploadHeading.textContent = "Upload Citra X-Ray Dada";
  uploadDesc.textContent = "Format yang didukung: JPEG, PNG · Maks. 10 MB";
}

changeBtn.addEventListener("click", resetUpload);

function resetUpload() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  dropZone.hidden = false;
  preview.hidden = true;
  analyzeBtn.disabled = true;
  analyzeBtn.hidden = false;
  resultCard.hidden = true;
  analysisLayout.classList.remove("has-result");
  uploadHeading.textContent = "Upload Citra X-Ray Dada";
  uploadDesc.textContent = "Format yang didukung: JPEG, PNG · Maks. 10 MB";
}

// ---- Analyze ----
analyzeBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;

  showLoading(true);
  hideError();

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    console.log("[Scientia] Sending request to", `${API_BASE}/predict`);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    clearTimeout(timer);

    console.log("[Scientia] Response status:", res.status);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    console.log("[Scientia] Result:", data);
    showResult(data);
  } catch (e) {
    console.error("[Scientia] Error:", e);
    showError(`Gagal terhubung ke API: ${e.message}`);
  } finally {
    showLoading(false);
  }
}

// ---- Show result ----
function showResult(data) {
  const isPneumonia = data.label === "PNEUMONIA";

  // badge mode
  modeBadge.textContent = data.mode === "real" ? "Model Real" : "Mode Dummy";
  modeBadge.className = `badge badge--${data.mode === "real" ? "real" : "dummy"}`;

  // verdict
  verdictIconWrap.innerHTML = isPneumonia ? ICON_PNEUMONIA : ICON_NORMAL;
  verdictIconWrap.className = `verdict__icon-wrap verdict__icon-wrap--${isPneumonia ? "pneumonia" : "normal"}`;
  verdictLabel.textContent = data.label;
  verdictLabel.className   = `verdict__value verdict__value--${isPneumonia ? "pneumonia" : "normal"}`;
  verdictConf.textContent  = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  // prob bars
  const pN = Math.round(data.probabilities.NORMAL    * 100);
  const pP = Math.round(data.probabilities.PNEUMONIA * 100);

  // force reflow before animating
  barNormal.style.width    = "0%";
  barPneumonia.style.width = "0%";
  requestAnimationFrame(() => {
    barNormal.style.width    = `${pN}%`;
    barPneumonia.style.width = `${pP}%`;
  });

  pctNormal.textContent    = `${pN}%`;
  pctPneumonia.textContent = `${pP}%`;

  uploadSection.hidden = false;
  analyzeBtn.hidden = true;
  analysisLayout.classList.add("has-result");
  uploadHeading.textContent = "Citra yang Dianalisis";
  uploadDesc.textContent = "Gambar tetap ditampilkan sebagai konteks hasil model.";
  resultCard.hidden    = false;
}

// ---- Reset ----
resetBtn.addEventListener("click", () => {
  uploadSection.hidden = false;
  resetUpload();
});

// ---- Helpers ----
function showLoading(on) {
  loadingOverlay.hidden = !on;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorToast.hidden = false;
  clearTimeout(showError._timer);
  showError._timer = setTimeout(() => { errorToast.hidden = true; }, 5000);
}

function hideError() {
  errorToast.hidden = true;
}
