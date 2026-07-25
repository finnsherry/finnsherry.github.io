import { InputTable, InputButtons } from "/utils/input.js";
import { imageFileToArray } from "/utils/imageio.js";
import { TextureMaker, setupWebGPU, ComputePipelineMaker, upwind_erosion } from "/utils/webgpu.js";

const container = document.getElementsByClassName("content-container")[0];
const inputTable = new InputTable(container);
const parameters = inputTable.addNumber({ label: "showEvery", shownLabel: "Show every #th frame", defaultValue: 1 });
const selector = {
  label: "colourScheme", shownLabel: "Colour scheme", options: [
    ["whiteBlack", "White-Black"],
    ["blueRed", "Blue-Red"],
    ["blueGreen", "Blue-Green"],
  ]
};
inputTable.addSelector(selector);
const inputButtons = new InputButtons(container);
const submit = inputButtons.addButton({ label: "submit", shownLabel: "Start" });

const canvas = document.createElement("canvas");
canvas.id = "canvas";
canvas.setAttribute("style", "width: 100%;");
container.appendChild(canvas);

const { device: device, context: context, format: format } = await setupWebGPU(canvas);

const WORKGROUP = 8;
const texFormat = "r32float";

const computePipelineMaker = new ComputePipelineMaker(device, texFormat, WORKGROUP);

function makeRenderPipeline(colourScheme) {
  let fragmentColouring = ``;
  switch (colourScheme) {
    case "blueRed":
      fragmentColouring = `
        let c = clamp(frac.r, 0., 1.);
        return vec4f(c * mazeMultiplier, 0.1 * mazeMultiplier, (1. - c) * mazeMultiplier, 1);
      `;
      break;
    case "blueGreen":
      fragmentColouring = `
        let c = clamp(0., frac.r, 1.);
        return vec4f(0.1 * mazeMultiplier, c * mazeMultiplier, (1. - c) * mazeMultiplier, 1);
      `;
      break;
    default:
      // White-Black.
      fragmentColouring = `
        let c = clamp(1 - frac.r, 0., 1.);
        return vec4f(c * mazeMultiplier, c * mazeMultiplier, c * mazeMultiplier, 1);
      `;
      break;
  }

  const renderWGSL = `
  struct Uniforms {
    grid: vec2f,
    maxValue: f32,
    dt: f32,
    origin: f32,
    mazeMax: f32,
  };

  @group(0) @binding(0) var tex : texture_2d<f32>;
  @group(0) @binding(1) var<uniform> uniforms: Uniforms;
  @group(0) @binding(2) var maze: texture_storage_2d<${texFormat}, read>;

  struct VSOut {
    @builtin(position) pos : vec4<f32>,
  };

  @vertex
  fn vs(@builtin(vertex_index) i : u32) -> VSOut {
    let pos = array<vec2<f32>, 6>(
      vec2(-1, -1), vec2(1, -1), vec2(-1, 1),
      vec2(-1, 1),  vec2(1, -1), vec2(1, 1)
    );
    var o : VSOut;
    o.pos = vec4(pos[i], 0, 1);
    return o;
  }

  @fragment
  fn fs(@builtin(position) p : vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(p.xy);
    let state = textureLoad(tex, coord, 0);
    let frac = state / uniforms.maxValue;
    let inMaze = textureLoad(maze, coord).r;
    let mazeMultiplier = inMaze + (1 - inMaze) * 0.1;
    ${fragmentColouring}
  }
`;

  return device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: "vs",
    },
    fragment: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: "fs",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });
}

function roundUpToMultiple(number, multiplier) {
  return Math.ceil(number / multiplier) * multiplier;
}

let rafId = null;
async function runSimulation() {
  // User parameters.
  const showEvery = inputTable.getNumber("showEvery", true);
  const colourScheme = inputTable.getSelector("colourScheme");
  console.log(colourScheme);


  // Load maze and make canvas.
  const { array: maze, width: width, height: height } = await imageFileToArray("maze.svg");
  function createStateTexture() {
    return device.createTexture({
      size: [width, height],
      format: texFormat,
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });
  }
  let texMaze = createStateTexture();
  device.queue.writeTexture(
    { texture: texMaze },
    maze,
    { bytesPerRow: width * 4 },
    [width, height]
  );
  canvas.width = width;
  canvas.height = height;

  // Define uniforms.
  const mazeMax = 2000;
  const maxValue =
    (Math.sqrt(height * height + width * width) * 2) / mazeMax;
  const dt = 1 / Math.sqrt(2);
  const origin = 0;

  const uniforms = new Float32Array([
    width,
    height,
    maxValue,
    dt,
    origin,
    mazeMax,
  ]);
  const uniformBuffer = device.createBuffer({
    label: "Grid uniforms",
    size: roundUpToMultiple(uniforms.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniforms);

  // Create textures for ping-pong.
  const init = new Float32Array(height * width);
  for (let i = 0; i < init.length; i += 1) {
    init[i] = i == origin ? 0 : maxValue;
  }
  let texA = createStateTexture();
  device.queue.writeTexture(
    { texture: texA },
    init,
    { bytesPerRow: width * 4 },
    [width, height]
  );
  let texB = createStateTexture();
  device.queue.writeTexture(
    { texture: texB },
    init,
    { bytesPerRow: width * 4 },
    [width, height]
  );

  // Compute pipeline and bind groups.
  const computePipeline = computePipelineMaker.makeEikonalPipeline();
  let computeBindA = computePipelineMaker.makeEikonalBindGroup(computePipeline, uniformBuffer, texA, texB, texMaze);
  let computeBindB = computePipelineMaker.makeEikonalBindGroup(computePipeline, uniformBuffer, texB, texA, texMaze);

  // Render pipeline and bind groups.
  const renderPipeline = makeRenderPipeline(colourScheme);

  function renderBind(tex, maze) {
    return device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: tex.createView() },
        { binding: 1, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: maze.createView() },
      ],
    });
  }
  let renderBindA = renderBind(texA, texMaze);
  let renderBindB = renderBind(texB, texMaze);

  let time = performance.now();
  function updateGrid() {
    let step = 0;
    const newtime = performance.now();
    const frametime = newtime - time;
    time = newtime;
    // console.log(frametime);

    const encoder = device.createCommandEncoder();

    for (let i = 0; i < showEvery; i++) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(computePipeline);
      pass.setBindGroup(0, computeBindA);
      pass.dispatchWorkgroups(
        Math.ceil(width / WORKGROUP),
        Math.ceil(height / WORKGROUP)
      );
      pass.end();
      [texA, texB] = [texB, texA];
      [computeBindA, computeBindB] = [computeBindB, computeBindA];
      step++;
    }
    console.log(step);

    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      pass.setPipeline(renderPipeline);
      pass.setBindGroup(0, renderBindB);
      pass.draw(6);
      pass.end();
      [renderBindA, renderBindB] = [renderBindB, renderBindA];
    }

    device.queue.submit([encoder.finish()]);
    rafId = requestAnimationFrame(updateGrid);
  }
  requestAnimationFrame(updateGrid);
}

async function startSimulation() {
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  await runSimulation();
}

await runSimulation();
submit.addEventListener("click", startSimulation);
