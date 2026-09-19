import { resizeCanvasToDisplaySize } from "/utils/canvas.js";
import { InputRanges } from "/utils/input.js";

const goldenRatio = (1 + Math.sqrt(5)) / 2;
const cAGoldenRatio = (2 / Math.PI) * Math.log(goldenRatio);

const container = document.getElementById("exponential-curve");
const inputRange = new InputRanges(container);
const cARange = inputRange.addRange({ label: "cA", shownLabel: "\\(c^A\\)", defaultValue: cAGoldenRatio, min: -1, max: 1 });

const canvas = document.createElement("canvas");
canvas.id = "canvas";
canvas.setAttribute("style", "width: 100%;");
container.appendChild(canvas);
resizeCanvasToDisplaySize(canvas);
const width = canvas.width;
const height = canvas.height;
const aspect = width / height;
const ctx = canvas.getContext("2d");

const black = `rgb(0 0 0)`;
const white = `rgb(255 255 255)`;
const brickRed = `rgb(${11 * 16 + 6}, ${3 * 16 + 2}, ${1 * 16 + 12})`;  // B6321C
const forestGreen = `rgb(${0 * 16 + 0}, ${9 * 11 + 2}, ${5 * 16 + 5})`; // 009B55
const royalBlue = `rgb(${0 * 16 + 0}, ${7 * 16 + 1}, ${11 * 16 + 12})`; // 0071BC
const axisColour = `rgb(0 0 0 / 0.1)`;

const lineWidth = 5;
const nSamples = 5000;
const tStart = -5 * Math.PI;
const tEnd = 5 * Math.PI;
const dt = (tEnd - tStart) / nSamples;

function project(x, y) {
  const unit = Math.min(width, height) / 4;
  return [width / 2 - unit * x, height / 2 + unit * y]
}

function plotSpiral() {
  const cA = inputRange.getValue("cA");
  const rSmall = 0.9;
  const rBig = 1.1;

  function spiralPoint(t, r) {
    return [r * Math.exp(t * cA) * Math.cos(t), r * Math.exp(t * cA) * Math.sin(t)]
  }
  ctx.fillStyle = white;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = royalBlue;
  ctx.fillStyle = royalBlue;
  ctx.beginPath();
  let [x, y] = project(...spiralPoint(tStart, rSmall));
  ctx.moveTo(x, y);
  for (let k = 1; k < nSamples; k++) {
    let t = tStart + k * dt;
    [x, y] = project(...spiralPoint(t, rSmall));
    ctx.lineTo(x, y);
  }
  [x, y] = project(...spiralPoint(tEnd, rBig));
  ctx.lineTo(x, y);
  for (let k = 1; k < nSamples; k++) {
    let t = tEnd - k * dt;
    [x, y] = project(...spiralPoint(t, rBig));
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
  ctx.fill();
}

plotSpiral();

cARange.addEventListener('input', plotSpiral);