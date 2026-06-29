/**
 * MNIST live drawing demo — crop/scale adapted from Burn's mnist-inference-web example.
 * Pixel convention: black background = 0.0, white digit = 1.0 (matches MNIST training data).
 */

const DEBOUNCE_MS = 75;
const INK_THRESHOLD = 15;
const BRUSH_WIDTH = 22;

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

const drawContext = drawCanvas.getContext("2d", { willReadFrequently: true });
const previewContext = previewCanvas.getContext("2d", { willReadFrequently: true });
const cropContext = cropCanvas.getContext("2d", { willReadFrequently: true });

let drawing = false;
let debounceTimer = null;
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
  plotPoint(event);
  schedulePredict();
});

drawCanvas.addEventListener("pointermove", (event) => {
  if (!drawing) {
    return;
  }
  plotPoint(event);
  schedulePredict();
});

drawCanvas.addEventListener("pointerup", () => {
  drawing = false;
  schedulePredict();
});

drawCanvas.addEventListener("pointercancel", () => {
  drawing = false;
});

clearButton.addEventListener("click", clearAll);

function plotPoint(event) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;

  drawContext.beginPath();
  drawContext.arc(x, y, BRUSH_WIDTH / 2, 0, Math.PI * 2);
  drawContext.fillStyle = "#ffffff";
  drawContext.fill();
}

function schedulePredict() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runPredict, DEBOUNCE_MS);
}

async function runPredict() {
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

  const { x, y, width, height } = bounds;

  const paddedSize = Math.ceil(Math.max(width, height) * 1.2);
  cropCtx.canvas.width = paddedSize;
  cropCtx.canvas.height = paddedSize;
  configureSmoothScaling(cropCtx);
  cropCtx.fillStyle = "#000000";
  cropCtx.fillRect(0, 0, paddedSize, paddedSize);

  const leftPadding = (paddedSize - width) / 2;
  const topPadding = (paddedSize - height) / 2;
  cropCtx.drawImage(
    mainContext.canvas,
    x,
    y,
    width,
    height,
    leftPadding,
    topPadding,
    width,
    height,
  );

  previewCtx.save();
  previewCtx.setTransform(1, 0, 0, 1, 0, 0);
  configureSmoothScaling(previewCtx);
  previewCtx.fillStyle = "#000000";
  previewCtx.fillRect(0, 0, 28, 28);
  previewCtx.drawImage(
    cropCtx.canvas,
    0,
    0,
    paddedSize,
    paddedSize,
    0,
    0,
    28,
    28,
  );
  previewCtx.restore();

  const imageData = previewCtx.getImageData(0, 0, 28, 28);
  return rgbaToNormalizedGrayscale(imageData.data);
}

function findInkBounds(ctx) {
  const { width, height, data } = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const xs = [];
  const ys = [];

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      if (Math.max(r, g, b) > INK_THRESHOLD) {
        xs.push(x);
        ys.push(y);
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
      animation: { duration: 150 },
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
  chart.update();

  const confidence = (probabilities[prediction] * 100).toFixed(1);
  predictionEl.textContent = `Prediction: ${prediction} (${confidence}%)`;
}

function resetChart() {
  chart.data.datasets[0].data = Array(10).fill(0);
  chart.data.datasets[0].backgroundColor = "#247abf";
  chart.update();
  predictionEl.textContent = "";
}

function clearAll() {
  requestSequence += 1;
  clearTimeout(debounceTimer);
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
