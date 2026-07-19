/**
 * MNIST live drawing demo — crop/scale adapted from Burn's mnist-inference-web example.
 * Pixel convention: black background = 0.0, white digit = 1.0 (matches MNIST training data).
 */

const PREDICT_INTERVAL_MS = 50;
const INK_THRESHOLD = 15;
const BRUSH_WIDTH = 22;
const PREVIEW_SIZE = 28;
const DIGIT_BOX_SIZE = 20;

const drawCanvas = document.getElementById("draw-canvas");
const previewCanvas = document.getElementById("preview-canvas");
const cropCanvas = document.getElementById("crop-canvas");
const chartCanvas = document.getElementById("chart-canvas");
const clearButton = document.getElementById("clear");
const predictionEl = document.getElementById("prediction");
const statusEl = document.getElementById("status");
const confusionSection = document.getElementById("confusion-section");
const confusionSummary = document.getElementById("confusion-summary");
const confusionTableContainer = document.getElementById("confusion-table-container");

const drawContext = drawCanvas.getContext("2d");
const previewContext = previewCanvas.getContext("2d", { willReadFrequently: true });
const cropContext = cropCanvas.getContext("2d");

let drawing = false;
let lastX = null;
let lastY = null;
let lastPredictAt = 0;
let pendingPredictTimer = null;
let requestSequence = 0;

const chart = createChart(chartCanvas);
resetChart();
loadConfusionMatrix();

drawContext.fillStyle = "#000000";
drawContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
drawContext.lineCap = "round";
drawContext.lineJoin = "round";
drawContext.strokeStyle = "#ffffff";
drawContext.lineWidth = BRUSH_WIDTH;

drawCanvas.addEventListener("pointerdown", (event) => {
  drawing = true;
  drawCanvas.setPointerCapture(event.pointerId);
  startStroke(event);
  schedulePredict(true);
});

drawCanvas.addEventListener("pointermove", (event) => {
  if (!drawing) {
    return;
  }
  extendStroke(event);
  schedulePredict(false);
});

drawCanvas.addEventListener("pointerup", () => {
  drawing = false;
  lastX = null;
  lastY = null;
  schedulePredict(true);
});

drawCanvas.addEventListener("pointercancel", () => {
  drawing = false;
  lastX = null;
  lastY = null;
});

clearButton.addEventListener("click", clearAll);

function canvasCoords(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (drawCanvas.width / rect.width),
    y: (event.clientY - rect.top) * (drawCanvas.height / rect.height),
  };
}

function startStroke(event) {
  const { x, y } = canvasCoords(event);
  lastX = x;
  lastY = y;

  drawContext.fillStyle = "#ffffff";
  drawContext.beginPath();
  drawContext.arc(x, y, BRUSH_WIDTH / 2, 0, Math.PI * 2);
  drawContext.fill();
}

function extendStroke(event) {
  const { x, y } = canvasCoords(event);

  drawContext.beginPath();
  drawContext.moveTo(lastX, lastY);
  drawContext.lineTo(x, y);
  drawContext.stroke();

  lastX = x;
  lastY = y;
}

function schedulePredict(immediate) {
  if (immediate) {
    clearTimeout(pendingPredictTimer);
    pendingPredictTimer = null;
    void runPredict();
    return;
  }

  const now = performance.now();
  const elapsed = now - lastPredictAt;
  const wait = PREDICT_INTERVAL_MS - elapsed;

  if (wait <= 0) {
    clearTimeout(pendingPredictTimer);
    pendingPredictTimer = null;
    void runPredict();
    return;
  }

  if (!pendingPredictTimer) {
    pendingPredictTimer = setTimeout(() => {
      pendingPredictTimer = null;
      void runPredict();
    }, wait);
  }
}

async function runPredict() {
  lastPredictAt = performance.now();
  const pixels = cropScaleToNormalizedArray(drawContext, cropContext, previewContext);
  if (!pixels) {
    resetChart();
    return;
  }

  const sequence = ++requestSequence;

  try {
    setStatus("");
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels: Array.from(pixels) }),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error ?? `predict failed (${response.status})`);
    }

    const { probabilities, prediction } = await response.json();
    if (sequence !== requestSequence) {
      return;
    }

    updateChart(probabilities, prediction);
  } catch (error) {
    if (sequence === requestSequence) {
      setStatus(error.message, true);
    }
  }
}

function configureSmoothScaling(ctx) {
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
}

function cropScaleToNormalizedArray(mainContext, cropCtx, previewCtx) {
  const bounds = findInkBounds(mainContext);
  if (!bounds) {
    return null;
  }

  const { x, y, width, height, centerOfMassX, centerOfMassY } = bounds;

  // MNIST convention: scale the ink's bounding box so its longer side is
  // DIGIT_BOX_SIZE px within the PREVIEW_SIZE frame (a fixed ~4px border,
  // not a fixed 5% margin), then center on the ink's center of mass rather
  // than the bounding box's geometric center.
  const scale = DIGIT_BOX_SIZE / Math.max(width, height);
  const scaledWidth = width * scale;
  const scaledHeight = height * scale;

  cropCtx.canvas.width = Math.max(1, Math.ceil(scaledWidth));
  cropCtx.canvas.height = Math.max(1, Math.ceil(scaledHeight));
  configureSmoothScaling(cropCtx);
  cropCtx.fillStyle = "#000000";
  cropCtx.fillRect(0, 0, cropCtx.canvas.width, cropCtx.canvas.height);
  cropCtx.drawImage(mainContext.canvas, x, y, width, height, 0, 0, scaledWidth, scaledHeight);

  const offsetX = PREVIEW_SIZE / 2 - (centerOfMassX - x) * scale;
  const offsetY = PREVIEW_SIZE / 2 - (centerOfMassY - y) * scale;

  previewCtx.save();
  previewCtx.setTransform(1, 0, 0, 1, 0, 0);
  configureSmoothScaling(previewCtx);
  previewCtx.fillStyle = "#000000";
  previewCtx.fillRect(0, 0, PREVIEW_SIZE, PREVIEW_SIZE);
  previewCtx.drawImage(cropCtx.canvas, offsetX, offsetY, scaledWidth, scaledHeight);
  previewCtx.restore();

  const imageData = previewCtx.getImageData(0, 0, PREVIEW_SIZE, PREVIEW_SIZE);
  return rgbaToNormalizedGrayscale(imageData.data);
}

function findInkBounds(ctx) {
  const { width, height, data } = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const xs = [];
  const ys = [];
  let massSum = 0;
  let massX = 0;
  let massY = 0;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const intensity = Math.max(r, g, b);
      if (intensity > INK_THRESHOLD) {
        xs.push(x);
        ys.push(y);
        massSum += intensity;
        massX += intensity * x;
        massY += intensity * y;
      }
    }
  }

  if (xs.length === 0) {
    return null;
  }

  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);

  const minX = xs[0];
  const maxX = xs[xs.length - 1];
  const minY = ys[0];
  const maxY = ys[ys.length - 1];

  return {
    x: minX,
    y: minY,
    width: maxX - minX + 1,
    height: maxY - minY + 1,
    centerOfMassX: massX / massSum,
    centerOfMassY: massY / massSum,
  };
}

function rgbaToNormalizedGrayscale(data) {
  const pixels = new Float64Array(data.length / 4);

  for (let index = 0; index < data.length; index += 4) {
    const r = data[index];
    const g = data[index + 1];
    const b = data[index + 2];
    const gray = (r + g + b) / (3 * 255);
    pixels[index / 4] = gray;
  }

  return pixels;
}

function createChart(canvas) {
  Chart.register(ChartDataLabels);

  return new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
      datasets: [
        {
          data: Array(10).fill(0),
          borderWidth: 0,
          backgroundColor: "#247abf",
        },
      ],
    },
    options: {
      responsive: false,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        datalabels: {
          color: "#ffffff",
          formatter: (value) => value.toFixed(2),
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
        },
      },
    },
  });
}

function updateChart(probabilities, prediction) {
  chart.data.datasets[0].data = probabilities;
  chart.data.datasets[0].backgroundColor = probabilities.map((_, index) =>
    index === prediction ? "#1b875b" : "#247abf",
  );
  chart.update("none");

  const confidence = (probabilities[prediction] * 100).toFixed(1);
  predictionEl.textContent = `Prediction: ${prediction} (${confidence}%)`;
}

function resetChart() {
  chart.data.datasets[0].data = Array(10).fill(0);
  chart.data.datasets[0].backgroundColor = "#247abf";
  chart.update("none");
  predictionEl.textContent = "";
}

function clearAll() {
  requestSequence += 1;
  clearTimeout(pendingPredictTimer);
  pendingPredictTimer = null;
  lastPredictAt = 0;
  drawing = false;
  lastX = null;
  lastY = null;
  drawContext.fillStyle = "#000000";
  drawContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  previewContext.fillStyle = "#000000";
  previewContext.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
  resetChart();
  setStatus("");
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

async function loadConfusionMatrix() {
  try {
    const response = await fetch("/confusion-matrix");
    if (!response.ok) {
      return;
    }

    const { matrix, accuracy, total } = await response.json();
    confusionSummary.textContent = `Overall accuracy: ${accuracy.toFixed(2)}% on ${total.toLocaleString()} test images. Rows = actual digit, columns = predicted digit.`;
    confusionTableContainer.replaceChildren(renderConfusionTable(matrix));
    confusionSection.hidden = false;
  } catch {
    // Test set unavailable — section stays hidden.
  }
}

function renderConfusionTable(matrix) {
  const table = document.createElement("table");
  table.className = "confusion-table";

  const maxCount = matrix.flat().reduce((max, value) => Math.max(max, value), 0);

  const header = document.createElement("tr");
  header.appendChild(document.createElement("th")).textContent = "Actual \\ Predicted";
  for (let digit = 0; digit < 10; digit += 1) {
    const cell = document.createElement("th");
    cell.textContent = String(digit);
    header.appendChild(cell);
  }
  table.appendChild(header);

  for (let actual = 0; actual < 10; actual += 1) {
    const row = document.createElement("tr");
    const label = document.createElement("td");
    label.className = "corner";
    label.textContent = String(actual);
    row.appendChild(label);

    for (let predicted = 0; predicted < 10; predicted += 1) {
      const count = matrix[actual][predicted];
      const cell = document.createElement("td");
      cell.className = "cell";
      cell.textContent = String(count);
      cell.style.backgroundColor = confusionCellColor(count, actual === predicted, maxCount);
      row.appendChild(cell);
    }

    table.appendChild(row);
  }

  return table;
}

function confusionCellColor(count, isCorrect, maxCount) {
  if (count === 0) {
    return "#ffffff";
  }

  const intensity = count / maxCount;
  if (isCorrect) {
    const green = Math.round(180 + intensity * 60);
    return `rgb(220, ${green}, 220)`;
  }

  const red = Math.round(220 + intensity * 35);
  return `rgb(${red}, 210, 210)`;
}
