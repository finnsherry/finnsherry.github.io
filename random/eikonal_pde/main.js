const adapter = await navigator.gpu?.requestAdapter();
const device = await adapter?.requestDevice();
if (!device) throw new Error("WebGPU not supported");

const canvas = document.getElementById("canvas");

const context = canvas.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format });

const WORKGROUP_SIZE = 8;

const UPDATE_INTERVAL = 200;
let step = 0;

const GRID_SIZE = 16;
const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
const uniformBuffer = device.createBuffer({
  label: "Grid uniforms",
  size: uniformArray.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
const WORKGROUP_COUNT = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);

const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
for (let i = 0; i < cellStateArray.length; i += 3) {
  cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
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

function updateGrid() {
  const encoder = device.createCommandEncoder();

  const computePass = encoder.beginComputePass();
  computePass.setPipeline(simulationPipeline);
  computePass.setBindGroup(0, bindGroups[step % 2]);
  computePass.dispatchWorkgroups(WORKGROUP_COUNT, WORKGROUP_COUNT);
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
  renderPass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);
  renderPass.end();
  device.queue.submit([encoder.finish()]);
}
setInterval(updateGrid, UPDATE_INTERVAL);

// resizeCanvas(canvas);
// const context = canvas.getContext("webgpu");
// const format = navigator.gpu.getPreferredCanvasFormat();
// context.configure({ device, format });

// const NUM_BALLS = 100;
// const BUFFER_SIZE = NUM_BALLS * 6;
// let inputBalls = new Float32Array(new ArrayBuffer(BUFFER_SIZE));
// for (let i = 0; i < BUFFER_SIZE; i += 6) {
//   inputBalls[i + 0] = randomBetween(2, 10);
//   inputBalls[i + 1] = 0;
//   inputBalls[i + 2] = randomBetween(0, context.canvas.width);
//   inputBalls[i + 3] = randomBetween(0, context.canvas.height);
//   inputBalls[i + 4] = randomBetween(-100, 100);
//   inputBalls[i + 5] = randomBetween(-100, 100);
// }

// async function initWebGPU() {
//   const module = device.createShaderModule({
//     code: `
//             struct Ball {
//                 radius: f32,
//                 position: vec2<f32>,
//                 velocity: vec2<f32>,
//             }

//             @group(0) @binding(0)
//             var<storage, read_write> input: array<Ball>;

//             @group(0) @binding(1)
//             var<storage, read_write> output: array<Ball>;

//             const TIME_STEP: f32 = 0.016;

//             @compute @workgroup_size(64)
//             fn main(
//                 @builtin(global_invocation_id) global_id: vec3<u32>,
//             ) {
//                 let num_balls = arrayLength(&output);
//                 if(global_id.x >= num_balls) {
//                     return;
//                 }
//                 output[global_id.x].position = output[global_id.x].position + output[global_id.x].velocity * TIME_STEP;
//             }
//         `,
//   });

//   const bindGroupLayout = device.createBindGroupLayout({
//     entries: [
//       {
//         binding: 0,
//         visibility: GPUShaderStage.COMPUTE,
//         buffer: { type: "read-only-storage" },
//       },
//       {
//         binding: 1,
//         visibility: GPUShaderStage.COMPUTE,
//         buffer: { type: "storage" },
//       },
//     ],
//   });

//   const pipeline = device.createComputePipeline({
//     layout: device.createPipelineLayout({
//       bindGroupLayouts: [bindGroupLayout],
//     }),
//     compute: {
//       module,
//       entryPoint: "main",
//     },
//   });

//   const input = device.createBuffer({
//     size: BUFFER_SIZE,
//     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//   });

//   const output = device.createBuffer({
//     size: BUFFER_SIZE,
//     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
//   });

//   const stagingBuffer = device.createBuffer({
//     size: BUFFER_SIZE,
//     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
//   });

//   const bindGroup = device.createBindGroup({
//     layout: bindGroupLayout,
//     entries: [
//       {
//         binding: 0,
//         resource: { buffer: input },
//       },
//       {
//         binding: 1,
//         resource: { buffer: output },
//       },
//     ],
//   });

//   device.queue.writeBuffer(input, 0, inputBalls);

//   const commandEncoder = device.createCommandEncoder();
//   const passEncoder = commandEncoder.beginComputePass();
//   passEncoder.setPipeline(pipeline);
//   passEncoder.setBindGroup(0, bindGroup);
//   passEncoder.dispatchWorkgroups(Math.ceil(BUFFER_SIZE / 64));
//   passEncoder.end();

//   commandEncoder.copyBufferToBuffer(output, 0, stagingBuffer, 0, BUFFER_SIZE);

//   const commands = commandEncoder.finish();
//   device.queue.submit([commands]);

//   await stagingBuffer.mapAsync(GPUMapMode.READ, 0, BUFFER_SIZE);

//   const copyArrayBuffer = stagingBuffer.getMappedRange(0, BUFFER_SIZE);
//   const data = copyArrayBuffer.slice();
//   stagingBuffer.unmap();
//   console.log(new Float32Array(data));
// }

// function randomBetween(lower, upper) {
//   return Math.floor(lower + Math.random() * (upper - lower + 1));
// }

// function resizeCanvas(canvas) {
//   const pixelRatio = window.devicePixelRatio;
//   canvas.width = Math.ceil(canvas.clientWidth * pixelRatio);
//   canvas.height = Math.ceil(canvas.clientHeight * pixelRatio);
// }

// async function main() {
//   await initWebGPU();

//   console.log();
// }

// main();
