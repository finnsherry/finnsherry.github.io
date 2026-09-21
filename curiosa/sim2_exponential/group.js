import { resizeCanvasToDisplaySize, canvasFromClient } from "/utils/canvas.js";
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

function worldFromCanvas([x, y]) {
  return [
    (x - width / 2) / minRadius,
    (y - height / 2) / minRadius,
  ]
}

function canvasFromWorld([x, y]) {
  return [
    x * minRadius + width / 2,
    y * minRadius + height / 2,
  ]
}

function drawPoint([x, y], colour = black) {
  ctx.beginPath();
  ctx.fillStyle = colour;
  ctx.arc(x, y, vertexRadius, 0, 2 * Math.PI, true);
  ctx.fill();
}

function drawTriangle(points, colours) {
  points.forEach(p => { drawPoint(canvasFromWorld(p), colours.vertexColour) });
  ctx.fillStyle = colours.fillColour;
  ctx.strokeStyle = colours.lineColour;
  ctx.lineWidth = lineWidth;
  const canvasPoints = points.map(canvasFromWorld);

  ctx.beginPath();
  ctx.moveTo(canvasPoints[0][0], canvasPoints[0][1]);
  ctx.lineTo(canvasPoints[1][0], canvasPoints[1][1]);
  ctx.lineTo(canvasPoints[2][0], canvasPoints[2][1]);
  ctx.closePath();
  ctx.stroke();
  ctx.fill();
}

function addPoint(e) {
  if (pointCount % 3 == 0) {
    clearCanvas();
    points = [];
  }
  const [ canvasX, canvasY ] = canvasFromClient(canvas, e.clientX, e.clientY);
  const [ x, y ] = worldFromCanvas([ canvasX, canvasY ]);
  drawPoint([canvasX, canvasY], inputColours.vertexColour);
  points.push([x, y]);
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