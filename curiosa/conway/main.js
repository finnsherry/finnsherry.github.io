import { InputTable, InputButtons } from "/utils/input.js";
import { TextureMaker, setupWebGPU, ComputePipelineMaker } from "/utils/webgpu.js";

const container = document.getElementsByClassName("content-container")[0];
const inputTable = new InputTable(container);
const parameters = [
  { label: "gridHeight", shownLabel: "Grid height", defaultValue: 128 },
  { label: "gridWidth", shownLabel: "Grid width", defaultValue: 128 },
  { label: "fraction", shownLabel: "Starting fraction alive", defaultValue: 0.4 },
  { label: "fps", shownLabel: "Frames per second", defaultValue: 10 },
]
parameters.forEach(parameter => inputTable.addNumber(parameter));
const inputButtons = new InputButtons(container);
const submit = inputButtons.addButton({ label: "submit", shownLabel: "Start" });

const canvas = document.createElement("canvas");
canvas.id = "canvas";
canvas.setAttribute("style", "width: 100%;");
container.appendChild(canvas);

const { device: device, context: context, format: format } = await setupWebGPU(canvas);

const WORKGROUP = 8;
const texFormat = "rgba8unorm";

const computePipelineMaker = new ComputePipelineMaker(device, texFormat, WORKGROUP);

const renderWGSL = `
  struct Uniforms {
    grid: vec2f,
  };

  @group(0) @binding(0) var tex : texture_2d<f32>;
  @group(0) @binding(1) var<uniform> uniforms: Uniforms;

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
    let alive = textureLoad(tex, coord, 0);
    let uv = p.xy / uniforms.grid;
    return (1 - alive) * vec4f(0, 0, 0.4, 1) + alive * vec4f(uv.x, 1 - uv.y, 1 - uv.x, 1);
  }
`;

const renderPipeline = device.createRenderPipeline({
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

function runSimulation() {
  const gridHeight = inputTable.getNumber("gridHeight", true);
  const gridWidth = inputTable.getNumber("gridWidth", true);
  const aliveFraction = inputTable.getNumber("fraction");
  const fps = inputTable.getNumber("fps", true)
  let updateInterval = 1000 / fps;

  const size = gridWidth * gridHeight;
  canvas.width = gridWidth;
  canvas.height = gridHeight;

  const uniformBuffer = device.createBuffer({
    label: "Grid uniforms",
    size: 8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Float32Array([gridWidth, gridHeight])
  );
  function renderBind(tex) {
    return device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: tex.createView() },
        { binding: 1, resource: { buffer: uniformBuffer } },
      ],
    });
  }

  let step = 0;

  const textureMaker = new TextureMaker(device, texFormat, gridWidth, gridHeight);

  let texA = textureMaker.createStateTexture();
  let texB = textureMaker.createStateTexture();

  const init = new Uint8Array(size * 4);
  for (let i = 0; i < init.length; i += 4) {
    const v = Math.random() > aliveFraction ? 255 : 0;
    init[i] = v;
    init[i + 1] = v;
    init[i + 2] = v;
    init[i + 3] = 255;
  }

  device.queue.writeTexture(
    { texture: texA },
    init,
    { bytesPerRow: gridWidth * 4 },
    [gridWidth, gridHeight]
  );

  const computePipeline = computePipelineMaker.makeConwayPipeline();
  let computeBindA = computePipelineMaker.makeConwayBindGroup(computePipeline, texA, texB);
  let computeBindB = computePipelineMaker.makeConwayBindGroup(computePipeline, texB, texA);

  let renderBindA = renderBind(texA);
  let renderBindB = renderBind(texB);

  let time = performance.now();
  function updateGrid() {
    const newtime = performance.now();
    const dt = newtime - time;
    time = newtime;
    // console.log(dt);
    const encoder = device.createCommandEncoder();

    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(computePipeline);
      pass.setBindGroup(0, computeBindA);
      pass.dispatchWorkgroups(
        Math.ceil(gridWidth / WORKGROUP),
        Math.ceil(gridHeight / WORKGROUP)
      );
      pass.end();
    }

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
    }

    step++;

    device.queue.submit([encoder.finish()]);
    [texA, texB] = [texB, texA];
    [computeBindA, computeBindB] = [computeBindB, computeBindA];
  }
  updateGrid();
  return setInterval(updateGrid, updateInterval);
}

function startSimulation() {
  if (previousSimulation) {
    clearInterval(previousSimulation);
  }
  previousSimulation = runSimulation();
}

let previousSimulation = runSimulation();
submit.addEventListener("click", startSimulation);
