import { getInputNumber, setInput } from "/utils/input.js";
import { svgFileToArray } from "/utils/imageio.js";
import { TextureMaker, setup, upwind_erosion } from "/utils/webgpu.js";

const { device: device, canvas: canvas, context: context, format: format } = await setup();

const WORKGROUP = 8;
const texFormat = "r32float";

function makeComputePipeline() {
  const computeWGSL = `
  struct Uniforms {
    grid: vec2f,
    maxValue: f32,
    dt: f32,
    origin: f32,
    mazeMax: f32,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var src : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(2) var dst : texture_storage_2d<${texFormat}, write>;
  @group(0) @binding(3) var maze: texture_storage_2d<${texFormat}, read>;

  ${upwind_erosion}

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(src);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let x = i32(id.x);
    let y = i32(id.y);

    let centre = textureLoad(src, vec2<i32>(x, y)).r;
    let xForward = textureLoad(src, vec2<i32>(clamp(x+1, 0, i32(dims.x) - 1), y)).r;
    let xBackward = textureLoad(src, vec2<i32>(clamp(x-1, 0, i32(dims.x) - 1), y)).r;
    let yForward = textureLoad(src, vec2<i32>(x, clamp(y+1, 0, i32(dims.y) - 1))).r;
    let yBackward = textureLoad(src, vec2<i32>(x, clamp(y-1, 0, i32(dims.y) - 1))).r;

    let dxForward = xForward - centre;
    let dxBackward = centre - xBackward;
    let dyForward = yForward - centre;
    let dyBackward = centre - yBackward;

    let dx = upwind_erosion(dxForward, dxBackward);
    let dy = upwind_erosion(dyForward, dyBackward);

    let cost = 1. / (1. + uniforms.mazeMax * textureLoad(maze, vec2<i32>(x, y)).r);

    let dWdt = cost - sqrt(dx * dx + dy * dy);
    var out = 0.;
    if abs(f32(f32(y) * uniforms.grid.x + f32(x)) - uniforms.origin) > 0.5 {
      out = centre + uniforms.dt * dWdt;
    }
    textureStore(
      dst,
      vec2<i32>(x, y),
      out * vec4f(1)
    );
  }
`;

  return device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: computeWGSL,
      }),
      entryPoint: "main",
    },
  });
}

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


setInput("gridwidth", 256);
setInput("showEvery", 1);

let rafId = null;
async function runSimulation() {
  // User parameters.
  const gridWidth = getInputNumber("gridwidth", 256, true);
  const showEvery = getInputNumber("showEvery", 1, true);
  let colourScheme = document.getElementById("colourScheme").value;
  console.log(colourScheme);

  // Load maze and make canvas.
  const { maze, height } = await svgFileToArray("maze.svg", gridWidth);
  const gridHeight = height;
  function createStateTexture() {
    return device.createTexture({
      size: [gridWidth, gridHeight],
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
    { bytesPerRow: gridWidth * 4 },
    [gridWidth, gridHeight]
  );
  canvas.width = gridWidth;
  canvas.height = gridHeight;

  // Define uniforms.
  const mazeMax = 2000;
  const maxValue =
    (Math.sqrt(gridHeight * gridHeight + gridWidth * gridWidth) * 2) / mazeMax;
  const dt = 1 / Math.sqrt(2);
  const origin = 0;

  const uniforms = new Float32Array([
    gridWidth,
    gridHeight,
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
  const init = new Float32Array(gridHeight * gridWidth);
  for (let i = 0; i < init.length; i += 1) {
    init[i] = i == origin ? 0 : maxValue;
  }
  let texA = createStateTexture();
  device.queue.writeTexture(
    { texture: texA },
    init,
    { bytesPerRow: gridWidth * 4 },
    [gridWidth, gridHeight]
  );
  let texB = createStateTexture();
  device.queue.writeTexture(
    { texture: texB },
    init,
    { bytesPerRow: gridWidth * 4 },
    [gridWidth, gridHeight]
  );

  // Compute pipeline and bind groups.
  const computePipeline = makeComputePipeline();

  function computeBind(src, dst, maze) {
    return device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: src.createView() },
        { binding: 2, resource: dst.createView() },
        { binding: 3, resource: maze.createView() },
      ],
    });
  }
  let computeBindA = computeBind(texA, texB, texMaze);
  let computeBindB = computeBind(texB, texA, texMaze);

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
        Math.ceil(gridWidth / WORKGROUP),
        Math.ceil(gridHeight / WORKGROUP)
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
const submit = document.getElementById("submit");
submit.addEventListener("click", startSimulation);
