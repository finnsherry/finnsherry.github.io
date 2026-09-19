import { resizeCanvasToDisplaySize } from "/utils/canvas.js";
import { InputRanges } from "/utils/input.js";

const container = document.getElementById("group");
const inputRange = new InputRanges(container);

const canvas = document.createElement("canvas");
canvas.id = "canvas-group";
canvas.setAttribute("style", "width: 100%;");
container.appendChild(canvas);
resizeCanvasToDisplaySize(canvas);
const width = canvas.width;
const height = canvas.height;
const minRadius = Math.min(width, height) / 2;
const aspect = width / height;
const ctx = canvas.getContext("2d");

const xRange = inputRange.addRange({ label: "x", shownLabel: "\\(x\\)", defaultValue: 0, min: -(width / (2 * minRadius)), max: (width / (2 * minRadius)) });
const yRange = inputRange.addRange({ label: "y", shownLabel: "\\(y\\)", defaultValue: 0, min: -(height / (2 * minRadius)), max: (height / (2 * minRadius)) });
const thetaRange = inputRange.addRange({ label: "theta", shownLabel: "\\(\\theta\\)", defaultValue: 0, min: -Math.PI, max: Math.PI });
const aRange = inputRange.addRange({ label: "a", shownLabel: "\\(a\\)", defaultValue: 1, min: 0.2, max: 5 });

const black = `rgb(0 0 0)`;
const white = `rgb(255 255 255)`;
const inputColours = {
  vertexColour: `rgb(${11 * 16 + 6}, ${3 * 16 + 2}, ${1 * 16 + 12}, 100%)`,
  lineColour: `rgb(${11 * 16 + 6}, ${3 * 16 + 2}, ${1 * 16 + 12}, 80%)`,
  fillColour: `rgb(${11 * 16 + 6}, ${3 * 16 + 2}, ${1 * 16 + 12}, 25%)`,
}
const transformedColours = {
  vertexColour: `rgb(${0 * 16 + 0}, ${7 * 16 + 1}, ${11 * 16 + 12}, 100%)`,
  lineColour: `rgb(${0 * 16 + 0}, ${7 * 16 + 1}, ${11 * 16 + 12}, 80%)`,
  fillColour: `rgb(${0 * 16 + 0}, ${7 * 16 + 1}, ${11 * 16 + 12}, 25%)`,
}

const vertexRadius = 10;
const lineWidth = 5;

let pointCount = 0;
let points = [];

function clearCanvas() {
  ctx.fillStyle = white;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function project([x, y]) {
  return [
    width / 2 + x * minRadius,
    height / 2 + y * minRadius
  ]
}

function drawPoint([x, y], colour = black) {
  ctx.beginPath();
  ctx.fillStyle = colour;
  ctx.arc(x, y, vertexRadius, 0, 2 * Math.PI, true);
  ctx.fill();
}

function drawTriangle(points, colours) {
  points.forEach(p => { drawPoint(project(p), colours.vertexColour) });
  ctx.fillStyle = colours.fillColour;
  ctx.strokeStyle = colours.lineColour;
  ctx.lineWidth = lineWidth;
  const projectedPoints = points.map(project);

  ctx.beginPath();
  ctx.moveTo(projectedPoints[0][0], projectedPoints[0][1]);
  ctx.lineTo(projectedPoints[1][0], projectedPoints[1][1]);
  ctx.lineTo(projectedPoints[2][0], projectedPoints[2][1]);
  ctx.closePath();
  ctx.stroke();
  ctx.fill();
}

function addPoint(e) {
  if (pointCount % 3 == 0) {
    clearCanvas();
    points = [];
  }
  const rect = canvas.getBoundingClientRect();
  const clickX = (2 * (e.clientX - rect.left) - width / 2) / minRadius;
  const clickY = (2 * (e.clientY - rect.top) - height / 2) / minRadius;
  drawPoint(project([clickX, clickY]), inputColours.vertexColour);
  points.push([clickX, clickY]);
  pointCount += 1;

  if (pointCount % 3 == 0) {
    drawScene()
  }
}

function transformPoint([xPoint, yPoint], [x, y, theta, a]) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    x + a * (xPoint * c - yPoint * s),
    y + a * (xPoint * s + yPoint * c),
  ]
}

function drawScene() {
  if (pointCount % 3 != 0) {
    return;
  }
  clearCanvas();
  drawTriangle(points, inputColours);

  const x = inputRange.getValue("x");
  const y = inputRange.getValue("y");
  const theta = inputRange.getValue("theta");
  const a = inputRange.getValue("a");
  const transformedPoints = points.map(p => transformPoint(p, [x, y, theta, a]));
  drawTriangle(transformedPoints, transformedColours);
}

clearCanvas();

canvas.addEventListener('pointerdown', addPoint);
xRange.addEventListener('input', drawScene);
yRange.addEventListener('input', drawScene);
thetaRange.addEventListener('input', drawScene);
aRange.addEventListener('input', drawScene);