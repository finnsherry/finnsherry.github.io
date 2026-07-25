import { InputTable, InputButtons } from "/utils/input.js";
import { imageFileToArray } from "/utils/imageio.js";
import { TextureMaker, ComputePipelineMaker, setupWebGPU, renderImage, passMaker } from "/utils/webgpu.js";

const container = document.getElementsByClassName("content-container")[0];
const inputTable = new InputTable(container);
const parameters = [
  { label: "delta", shownLabel: "\\(\\delta\\)", defaultValue: 10 },
  { label: "threshold", shownLabel: "Threshold", defaultValue: 0.5 },
]

parameters.forEach(parameter => inputTable.addNumber(parameter));
const inputButtons = new InputButtons(container);
let binariseButton;
let dilateButton;
let connectButton;

const canvas = document.createElement("canvas");
canvas.id = "canvas";
canvas.setAttribute("style", "width: 100%;");
container.appendChild(canvas);

const { device: device, context: context, format: format } = await setupWebGPU(canvas);

const WORKGROUP = 8;
const texFormat = "r32float";

const computePipelineMaker = new ComputePipelineMaker(device, texFormat, WORKGROUP);

function roundUpToMultiple(number, multiplier) {
  return Math.ceil(number / multiplier) * multiplier;
}

const delta = inputTable.getNumber("delta");
const threshold = inputTable.getNumber("threshold");

const { array: intialImage, width: gridWidth, height: gridHeight } = await imageFileToArray("cross.png");
canvas.width = gridWidth;
canvas.height = gridHeight;
console.log(Math.max(...intialImage));

const workGroupGrid = [Math.ceil(gridWidth / WORKGROUP), Math.ceil(gridHeight / WORKGROUP)]

const textureMaker = new TextureMaker(device, texFormat, gridWidth, gridHeight);

let u0 = textureMaker.createStateTexture();
device.queue.writeTexture(
  { texture: u0 },
  intialImage,
  { bytesPerRow: gridWidth * 4 },
  [gridWidth, gridHeight],
);
let ubin = textureMaker.createStateTexture();
let udil = textureMaker.createStateTexture();

binariseButton = inputButtons.addButton({ label: "binarise", shownLabel: "Binarise" });
async function binarise() {
  console.log("binarise!");
  const binarisePipeline = computePipelineMaker.makeBinarisationPipeline(threshold);
  const binariseBind = computePipelineMaker.makeBinarisationBindGroup(binarisePipeline, u0, ubin);

  const overwritePipeline = computePipelineMaker.makeOverwritePipeline();
  const overwriteBind = computePipelineMaker.makeOverwriteBindGroup(overwritePipeline, ubin, u0);
  const encoder = device.createCommandEncoder();
  passMaker(encoder, binarisePipeline, binariseBind, workGroupGrid);
  passMaker(encoder, overwritePipeline, overwriteBind, workGroupGrid);
  device.queue.submit([encoder.finish()]);

  await renderImage(ubin, device, context, format);
  binarised = true;
  inputButtons.remove("binarise");
  dilateButton = inputButtons.addButton({ label: "dilate", shownLabel: "Dilate" });
  dilateButton.addEventListener("click", dilate);
}

async function dilate() {
  console.log("dilate!");
  const dtOpt = 1 / Math.sqrt(2);
  const n = Math.ceil(delta / dtOpt);
  const dt = delta / n;
  const dilatePipeline = computePipelineMaker.makeDilationPipeline(dt);
  let dilateBindA = computePipelineMaker.makeDilationBindGroup(dilatePipeline, u0, udil);
  let dilateBindB = computePipelineMaker.makeDilationBindGroup(dilatePipeline, udil, u0);

  const encoder = device.createCommandEncoder();
  for (let i = 0; i < n; i++) {
    passMaker(encoder, dilatePipeline, dilateBindA, workGroupGrid);
    [u0, udil] = [udil, u0];
    [dilateBindA, dilateBindB] = [dilateBindB, dilateBindA];
  }
  device.queue.submit([encoder.finish()]);

  await renderImage(udil, device, context, format);
  dilated = true;
  inputButtons.remove("dilate");
  connectButton = inputButtons.addButton({ label: "connect", shownLabel: "Connect" });
  connectButton.addEventListener("click", connect);
}

async function connect() {
  if (connected) {
    console.log("Already connected!");
    return;
  }
  console.log("connect!");
  const dtOpt = 1 / Math.sqrt(2);
  const n = Math.ceil(delta / dtOpt);
  const dt = delta / n;
  const binarisePipeline = computePipelineMaker.makeBinarisationPipeline(threshold);
  const binariseBind = device.createBindGroup({
    label: "binarise",
    layout: binarisePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: udil.createView() },
      { binding: 1, resource: u0.createView() },
    ]
  })
  const multiplyPipeline = computePipelineMaker.makeBinaryOperatorPipeline("*");
  const multiplyBind = computePipelineMaker.makeBinaryOperatorBindGroup(multiplyPipeline, u0, ubin, udil);

  const encoder = device.createCommandEncoder();
  passMaker(encoder, binarisePipeline, binariseBind, workGroupGrid);
  // passMaker(encoder, multiplyPipeline, multiplyBind, workGroupGrid);
  device.queue.submit([encoder.finish()]);

  await renderImage(u0, device, context, format);
  connected = true;
}

await renderImage(u0, device, context, format);

let binarised = false;
let dilated = false;
let connected = false;

binariseButton.addEventListener("click", binarise);
