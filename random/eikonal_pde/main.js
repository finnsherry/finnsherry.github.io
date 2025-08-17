const adapter = await navigator.gpu?.requestAdapter();
const device = await adapter?.requestDevice();
if (!device) throw new Error("WebGPU not supported");

const canvas = document.getElementById("canvas");

const context = canvas.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format });

const WORKGROUP_SIZE = 8;

async function svgFileToArray(filePath, width) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";

    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      const aspect = img.naturalHeight / img.naturalWidth;
      const height = Math.round(width * aspect);
      canvas.height = height;
      const ctx = canvas.getContext("2d");

      ctx.drawImage(img, 0, 0, width, height);
      const { data } = ctx.getImageData(0, 0, width, height);

      const maze = new Float32Array(width * height);
      for (let i = 0; i < width * height; i++) {
        if (data[i * 4 + 3] > 0) {
          maze[i] = 0;
        } else {
          maze[i] = 1;
        }
      }
      resolve({ maze, height });
    };

    img.onerror = reject;
    img.src = filePath;
  });
}

// svgFileToArray("maze.svg", 128).then((arr) => console.log(arr));

const shaderModule = device.createShaderModule({
  label: "Eikonal PDE shader",
  code: `
    struct Uniforms {
      grid: vec2f,
      maxValue: f32,
      dt: f32,
      origin: f32,
      mazeMax: f32,
    };
      
    struct VertexInput {
      @location(0) pos: vec2f,
      @builtin(instance_index) instance: u32,
    };

    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) state: f32,
      @location(1) inMaze: f32,
    };

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var<storage> cellStateIn: array<f32>;
    @group(0) @binding(2) var<storage, read_write> cellStateOut: array<f32>;
    @group(0) @binding(3) var<storage> maze: array<f32>;

    // Compute shaders
    fn cellIndex(cell: vec2u) -> u32 {
      let grid = uniforms.grid;
      let x = clamp(cell.x, 0, u32(grid.x - 1));
      let y = clamp(cell.y, 0, u32(grid.y - 1));
      return y * u32(grid.x) + x;
    }

    fn selectUpwind(forward: f32, backward: f32) -> f32 {
      return max(max(-forward, backward), 0.) * (2. * step(-forward, backward) - 1.);
    }

    @compute
    @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
    fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
      let centre = cellStateIn[cellIndex(cell.xy)];
      let xForward = cellStateIn[cellIndex(vec2u(cell.x+1, cell.y))];
      let xBackward = cellStateIn[cellIndex(vec2u(cell.x-1, cell.y))];
      let yForward = cellStateIn[cellIndex(vec2u(cell.x, cell.y+1))];
      let yBackward = cellStateIn[cellIndex(vec2u(cell.x, cell.y-1))];

      let dxForward = xForward - centre;
      let dxBackward = centre - xBackward;
      let dyForward = yForward - centre;
      let dyBackward = centre - yBackward;

      let dx = selectUpwind(dxForward, dxBackward);
      let dy = selectUpwind(dyForward, dyBackward);
      
      let i = cellIndex(cell.xy);
      let cost = 1. / (1. + uniforms.mazeMax * maze[i]);

      let dWdt = cost - sqrt(dx * dx + dy * dy);
      if abs(f32(i) - uniforms.origin) < 0.5 {
        cellStateOut[i] = 0.;
      } else {
        cellStateOut[i] = cellStateIn[i] + uniforms.dt * dWdt;
      }
    }

    // Vertex shaders
    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
      let i = f32(input.instance);
      let grid = uniforms.grid;
      let cell = vec2f(i % grid.x, floor(i / grid.x));
      var state = cellStateIn[input.instance];

      let gridPos = (input.pos + 1) / grid - 1 + cell / grid * 2;

      var output: VertexOutput;
      output.pos = vec4f(gridPos, 0, 1);
      output.state = state;
      output.inMaze = maze[input.instance];
      return output;
    }

    // Fragment shaders
    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
      let frac = input.state / uniforms.maxValue;
      let c = clamp(2. * frac, 0., 1.);
      let mazeMultiplier = input.inMaze + (1 - input.inMaze) * 0.1;
      return vec4f(c * mazeMultiplier, 0.1 * mazeMultiplier, (1. - c) * mazeMultiplier, 1);
    }
  `,
});

// Square in grid
const vertexSize = 1.0;
const vertices = new Float32Array([
  -1, -1, 1, -1, 1, 1,

  -1, -1, 1, 1, -1, 1,
]).map((x) => x * vertexSize);
const vertexBuffer = device.createBuffer({
  label: "Cell vertices",
  size: vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);
const vertexBufferLayout = {
  arrayStride: 8,
  attributes: [
    {
      format: "float32x2",
      offset: 0,
      shaderLocation: 0,
    },
  ],
};

const bindGroupLayout = device.createBindGroupLayout({
  label: "Cell bind group layout",
  entries: [
    {
      binding: 0,
      visibility:
        GPUShaderStage.VERTEX |
        GPUShaderStage.FRAGMENT |
        GPUShaderStage.COMPUTE,
      buffer: {},
    },
    {
      binding: 1,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      binding: 3,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    },
  ],
});

const pipelineLayout = device.createPipelineLayout({
  label: "Cell pipeline layout",
  bindGroupLayouts: [bindGroupLayout],
});

const cellPipeline = device.createRenderPipeline({
  label: "Cell pipeline",
  layout: pipelineLayout,
  vertex: {
    module: shaderModule,
    entryPoint: "vertexMain",
    buffers: [vertexBufferLayout],
  },
  fragment: {
    module: shaderModule,
    entryPoint: "fragmentMain",
    targets: [
      {
        format: format,
      },
    ],
  },
});

const simulationPipeline = device.createComputePipeline({
  label: "Simulation pipeline",
  layout: pipelineLayout,
  compute: {
    module: shaderModule,
    entryPoint: "computeMain",
  },
});

function roundUpToMultiple(number, multiplier) {
  const div = Math.floor(number / multiplier);
  const rem = number % multiplier;
  const roundUp = rem > 0 ? 1 : 0;
  return (div + roundUp) * multiplier;
}

function resizeCanvas(canvas, aspect) {
  const pixelRatio = window.devicePixelRatio;
  canvas.width = Math.ceil(canvas.clientWidth * pixelRatio);
  canvas.height = Math.ceil(canvas.width * aspect);
}

async function runSimulation() {
  isRunning = true;
  let gridWidth = parseFloat(document.getElementById("gridwidth").value);
  if (!gridWidth) {
    gridWidth = 256;
  }

  const { maze, height } = await svgFileToArray("maze.svg", gridWidth);
  const gridHeight = height;

  const mazeMax = 5000;
  const maxValue =
    (Math.sqrt(gridHeight * gridHeight + gridWidth * gridWidth) * 2) / mazeMax;
  const dt = 1 / Math.sqrt(2);
  const origin =
    Math.floor(gridHeight / 2) * gridWidth + Math.floor(gridWidth / 2);

  const aspect = gridHeight / gridWidth;
  resizeCanvas(canvas, aspect);

  let step = 0;

  const uniformArray = new Float32Array([
    gridWidth,
    gridHeight,
    maxValue,
    dt,
    origin,
    mazeMax,
  ]);
  const uniformBuffer = device.createBuffer({
    label: "Grid uniforms",
    size: roundUpToMultiple(uniformArray.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
  const workgroupHeight = Math.ceil(gridHeight / WORKGROUP_SIZE);
  const workgroupWidth = Math.ceil(gridWidth / WORKGROUP_SIZE);

  const cellStateArray = new Float32Array(gridHeight * gridWidth);
  for (let i = 0; i < cellStateArray.length; i += 1) {
    cellStateArray[i] = i == origin ? 0 : maxValue;
  }
  const cellStateStorage = [
    device.createBuffer({
      label: "Cell states A",
      size: cellStateArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
      label: "Cell states B",
      size: cellStateArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
  ];
  device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);
  device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray);

  const mazeStorage = device.createBuffer({
    label: "Maze",
    size: maze.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(mazeStorage, 0, maze);

  const bindGroups = [
    device.createBindGroup({
      label: "Cell renderer bind group A",
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer },
        },
        {
          binding: 1,
          resource: { buffer: cellStateStorage[0] },
        },
        {
          binding: 2,
          resource: { buffer: cellStateStorage[1] },
        },
        {
          binding: 3,
          resource: { buffer: mazeStorage },
        },
      ],
    }),
    device.createBindGroup({
      label: "Cell renderer bind group B",
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer },
        },
        {
          binding: 1,
          resource: { buffer: cellStateStorage[1] },
        },
        {
          binding: 2,
          resource: { buffer: cellStateStorage[0] },
        },
        {
          binding: 3,
          resource: { buffer: mazeStorage },
        },
      ],
    }),
  ];

  let time = performance.now();
  function updateGrid() {
    const newtime = performance.now();
    const frametime = newtime - time;
    time = newtime;
    const encoder = device.createCommandEncoder();

    step++;

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: [0.1, 0.1, 0.1, 0],
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(cellPipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setBindGroup(0, bindGroups[step % 2]);
    renderPass.draw(vertices.length / 2, gridWidth * gridHeight);
    renderPass.end();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);
    computePass.dispatchWorkgroups(workgroupWidth, workgroupHeight);
    computePass.end();
    device.queue.submit([encoder.finish()]);
    if (isRunning) {
      requestAnimationFrame(updateGrid);
    }
  }
  requestAnimationFrame(updateGrid);
}

async function startSimulation() {
  isRunning = false;
  await runSimulation();
}

let isRunning = false;
await runSimulation();
const submit = document.getElementById("submit");
submit.addEventListener("click", startSimulation);
