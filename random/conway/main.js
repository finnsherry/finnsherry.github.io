const adapter = await navigator.gpu?.requestAdapter();
const device = await adapter?.requestDevice();
if (!device) throw new Error("WebGPU not supported");

const canvas = document.getElementById("canvas");

const context = canvas.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format });

const WORKGROUP_SIZE = 8;

const cellShaderModule = device.createShaderModule({
  label: "Cell shader",
  code: `
    struct VertexInput {
      @location(0) pos: vec2f,
      @builtin(instance_index) instance: u32,
    };

    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) cell: vec2f,
    };

    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellState: array<u32>;

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
      let i = f32(input.instance);
      let cell = vec2f(i % grid.x, floor(i / grid.x));
      let state = f32(cellState[input.instance]);

      let gridPos = (input.pos * state + 1) / grid - 1 + cell / grid * 2;

      var output: VertexOutput;
      output.pos = vec4f(gridPos, 0, 1);
      output.cell = cell;
      return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let c = input.cell / grid;
      return vec4f(c, 1 - c.x, 1);
    }
  `,
});

const simulationShaderModule = device.createShaderModule({
  label: "Game of life simulation shader",
  code: `
    @group(0) @binding(0) var<uniform> grid: vec2f;

    @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
    @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

    fn cellIndex(cell: vec2u) -> u32 {
      return (cell.y % u32(grid.y)) * u32(grid.x) + cell.x % u32(grid.x);
    }

    fn cellActive(x: u32, y: u32) -> u32 {
      return cellStateIn[cellIndex(vec2(x, y))];
    }

    @compute
    @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
    fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
      let activeNeighbours = cellActive(cell.x+1, cell.y+1) +
                            cellActive(cell.x+1, cell.y) +
                            cellActive(cell.x+1, cell.y-1) +
                            cellActive(cell.x, cell.y-1) +
                            cellActive(cell.x-1, cell.y-1) +
                            cellActive(cell.x-1, cell.y) +
                            cellActive(cell.x-1, cell.y+1) +
                            cellActive(cell.x, cell.y+1);

      let i = cellIndex(cell.xy);
      
      switch activeNeighbours {
        case 2: {
          cellStateOut[i] = cellStateIn[i];
        }

        case 3: {
          cellStateOut[i] = 1;
        }

        default: {
          cellStateOut[i] = 0;
        }
      }
    }
  `,
});

// Square in grid
const vertexSize = 0.9;
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
    module: cellShaderModule,
    entryPoint: "vertexMain",
    buffers: [vertexBufferLayout],
  },
  fragment: {
    module: cellShaderModule,
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
    module: simulationShaderModule,
    entryPoint: "computeMain",
  },
});

function resizeCanvas(canvas, aspect) {
  const pixelRatio = window.devicePixelRatio;
  canvas.width = Math.ceil(canvas.clientWidth * pixelRatio);
  canvas.height = Math.ceil(canvas.width * aspect);
}

function runSimulation() {
  let gridHeight = parseFloat(document.getElementById("gridheight").value);
  if (!gridHeight) {
    gridHeight = 8;
  }
  let gridWidth = parseFloat(document.getElementById("gridwidth").value);
  if (!gridWidth) {
    gridWidth = 16;
  }
  let aliveFraction = parseFloat(document.getElementById("fraction").value);
  if (!aliveFraction) {
    aliveFraction = 0.4;
  }
  let updateInterval = parseInt(document.getElementById("frametime").value);
  if (!updateInterval) {
    updateInterval = 200;
  }

  const aspect = gridHeight / gridWidth;
  resizeCanvas(canvas, aspect);

  let step = 0;

  const uniformArray = new Float32Array([gridWidth, gridHeight]);
  const uniformBuffer = device.createBuffer({
    label: "Grid uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
  const workgroupHeight = Math.ceil(gridHeight / WORKGROUP_SIZE);
  const workgroupWidth = Math.ceil(gridWidth / WORKGROUP_SIZE);

  const cellStateArray = new Uint32Array(gridHeight * gridWidth);
  for (let i = 0; i < cellStateArray.length; i += 3) {
    cellStateArray[i] = Math.random() < aliveFraction ? 1 : 0;
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
      ],
    }),
  ];

  function updateGrid() {
    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);
    computePass.dispatchWorkgroups(workgroupWidth, workgroupHeight);
    computePass.end();

    step++;

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: [0, 0, 0.4, 0],
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(cellPipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setBindGroup(0, bindGroups[step % 2]);
    renderPass.draw(vertices.length / 2, gridWidth * gridHeight);
    renderPass.end();
    device.queue.submit([encoder.finish()]);
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
