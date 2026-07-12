import { getInputNumber, setInput } from "/utils/input.js";
import { imageFileToArray } from "/utils/imageio.js";
import { TextureMaker, ComputePipelineMaker, setupWebGPU, renderImage, passMaker } from "/utils/webgpu.js";

const { device: device, canvas: canvas, context: context, format: format } = await setupWebGPU();

const WORKGROUP = 8;
const texFormat = "r32float";

const computePipelineMaker = new ComputePipelineMaker(device, texFormat, WORKGROUP);

function roundUpToMultiple(number, multiplier) {
  return Math.ceil(number / multiplier) * multiplier;
}

setInput("delta", 10);
setInput("threshold", 0.5);
const delta = getInputNumber("delta", 10);
const threshold = getInputNumber("threshold", 0.5);

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

async function binarise() {
  console.log("binarise!");
  if (binarised) {
    console.log("Already binarised!");
    return;
  }
  const binarisePipeline = computePipelineMaker.makeBinarisationPipeline(threshold);
  const binariseBind = device.createBindGroup({
    label: "binarise",
    layout: binarisePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: u0.createView() },
      { binding: 1, resource: ubin.createView() },
    ]
  })
  const encoder = device.createCommandEncoder();
  passMaker(encoder, binarisePipeline, binariseBind, workGroupGrid);
  device.queue.submit([encoder.finish()]);

  await renderImage(ubin, device, context, format);
  binarised = true;
}

async function dilate() {
  console.log("dilate!");
  if (!binarised) {
    console.log("Can't dilate without first binarising!");
    return;
  }
  if (dilated) {
    console.log("Already dilated!");
    return;
  }
  const dtOpt = 1 / Math.sqrt(2);
  const n = Math.ceil(delta / dtOpt);
  const dt = delta / n;
  const dilatePipeline = computePipelineMaker.makeDilationPipeline(dt);
  let dilateBindA = device.createBindGroup({
    label: "binarise",
    layout: dilatePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: ubin.createView() },
      { binding: 1, resource: udil.createView() },
    ]
  })
  let dilateBindB = device.createBindGroup({
    label: "binarise",
    layout: dilatePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: udil.createView() },
      { binding: 1, resource: ubin.createView() },
    ]
  })

  const encoder = device.createCommandEncoder();
  for (let i = 0; i < n; i++) {
    passMaker(encoder, dilatePipeline, dilateBindA, workGroupGrid);
    [ubin, udil] = [udil, ubin];
    [dilateBindA, dilateBindB] = [dilateBindB, dilateBindA];
  }
  device.queue.submit([encoder.finish()]);

  await renderImage(ubin, device, context, format);
  dilated = true;
}

async function connect() {
  console.log("connect!");
  if (!dilated) {
    console.log("Can't connect without first dilating!");
    return;
  }
  if (connected) {
    console.log("Already connected!");
    return;
  }
  const dtOpt = 1 / Math.sqrt(2);
  const n = Math.ceil(delta / dtOpt);
  const dt = delta / n;
  const multiplicationPipeline = computePipelineMaker.makeBinaryOperatorPipeline("*");
  const multiplicationBind = device.createBindGroup({
    label: "binarise",
    layout: multiplicationPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: u0.createView() },
      { binding: 1, resource: udil.createView() },
      { binding: 2, resource: ubin.createView() },
    ]
  })

  const encoder = device.createCommandEncoder();
  passMaker(encoder, multiplicationPipeline, multiplicationBind, workGroupGrid);
  device.queue.submit([encoder.finish()]);

  await renderImage(ubin, device, context, format);
  connected = true;
}

await renderImage(u0, device, context, format);

let binarised = false;
let dilated = false;
let connected = false;

const binariseButton = document.getElementById("binarise");
binariseButton.addEventListener("click", binarise);

const dilateButton = document.getElementById("dilate");
dilateButton.addEventListener("click", dilate);

const connectButton = document.getElementById("connect");
connectButton.addEventListener("click", connect);
