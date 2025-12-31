const adapter = await navigator.gpu?.requestAdapter();
const device = await adapter?.requestDevice();
if (!device) throw new Error("WebGPU not supported");

const canvas = document.getElementById("canvas");

const context = canvas.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({
  device,
  format,
  alphaMode: "opaque",
});

const WORKGROUP = 8;

const texFormat = "r32float";

const computeWGSL = `
  @group(0) @binding(0) var src : texture_storage_2d<rgba8unorm, read>;
  @group(0) @binding(1) var dst : texture_storage_2d<rgba8unorm, write>;

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(src);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let x = i32(id.x);
    let y = i32(id.y);
    var n = 0;

    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) { continue; }
        let nx = (x + dx + i32(dims.x)) % i32(dims.x);
        let ny = (y + dy + i32(dims.y)) % i32(dims.y);
        if (textureLoad(src, vec2<i32>(nx, ny)).r > 0.5) {
          n++;
        }
      }
    }

    let alive = textureLoad(src, vec2<i32>(x, y)).r > 0.5;

    let outAlive =
      (alive && (n == 2 || n == 3)) ||
      (!alive && n == 3);

    textureStore(
      dst,
      vec2<i32>(x, y),
      select(
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
        vec4<f32>(1.0, 1.0, 1.0, 1.0),
        outAlive
      )
    );
  }
`;

const computePipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: device.createShaderModule({
      code: computeWGSL,
    }),
    entryPoint: "main",
  },
});

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

function computeBind(src, dst) {
  return device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: src.createView() },
      { binding: 1, resource: dst.createView() },
    ],
  });
}

function createLifeTexture(width, height) {
  return device.createTexture({
    size: [width, height],
    format: texFormat,
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST,
  });
}

function runSimulation() {
  let gridHeight = parseFloat(document.getElementById("gridheight").value);
  if (!gridHeight) {
    gridHeight = 128;
  }
  let gridWidth = parseFloat(document.getElementById("gridwidth").value);
  if (!gridWidth) {
    gridWidth = 128;
  }
  let aliveFraction = parseFloat(document.getElementById("fraction").value);
  if (!aliveFraction) {
    aliveFraction = 0.4;
  }
  let fps = parseInt(document.getElementById("fps").value);
  if (!fps) {
    fps = 60;
  }
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

  let texA = createLifeTexture(gridWidth, gridHeight);
  let texB = createLifeTexture(gridWidth, gridHeight);

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
      pass.setBindGroup(0, computeBind(texA, texB));
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
      pass.setBindGroup(0, renderBind(texB));
      pass.draw(6);
      pass.end();
    }

    step++;

    device.queue.submit([encoder.finish()]);
    [texA, texB] = [texB, texA];
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
const submit = document.getElementById("submit");
submit.addEventListener("click", startSimulation);
