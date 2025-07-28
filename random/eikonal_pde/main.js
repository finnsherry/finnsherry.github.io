const canvas = document.getElementById('canvas');
const fileInput = document.getElementById('image');

const alpha = 0.1; // thermal diffusivity
const dt = 0.1;    // time step
const workgroupSize = 8;

let device, context;
let GRID_WIDTH = 0, GRID_HEIGHT = 0;
let stateTexA, stateTexB;
let computePipeline, renderPipeline;
let computeBindA, renderBind;
let paramBuffer;
let mousePos = [0, 0];
let isDrawing = false;

async function initWebGPU() {
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter();
    device = await adapter.requestDevice();
    context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ code: computeShader() }),
            entryPoint: 'main'
        }
    });

    renderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: device.createShaderModule({ code: vertexShader() }),
            entryPoint: 'main'
        },
        fragment: {
            module: device.createShaderModule({ code: fragmentShader() }),
            entryPoint: 'main',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list' }
    });

    paramBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
}

function computeShader() {
    return `
struct Params {
    mouse : vec2<f32>,
    brushRadius : f32,
    isDrawing : u32,
};

@group(0) @binding(0) var srcTex : texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var dstTex : texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params : Params;

const alpha : f32 = ${alpha};
const dt : f32 = ${dt};
const dx : f32 = 1.0;

@compute @workgroup_size(${workgroupSize}, ${workgroupSize})
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x >= ${GRID_WIDTH}u || y >= ${GRID_HEIGHT}u) { return; }

    let center = textureLoad(srcTex, vec2<i32>(i32(x), i32(y)), 0).r;
    let left = textureLoad(srcTex, vec2<i32>(i32(x)-1, i32(y)), 0).r;
    let right = textureLoad(srcTex, vec2<i32>(i32(x)+1, i32(y)), 0).r;
    let up = textureLoad(srcTex, vec2<i32>(i32(x), i32(y)-1), 0).r;
    let down = textureLoad(srcTex, vec2<i32>(i32(x), i32(y)+1), 0).r;

    let lap = (left + right + up + down - 4.0 * center) / (dx*dx);
    var newVal = center + dt * alpha * lap;

    // Apply mouse interaction
    if (params.isDrawing == 1u) {
        let fx = f32(x);
        let fy = f32(y);
        let dist = distance(vec2<f32>(fx, fy), params.mouse);
        if (dist < params.brushRadius) {
            newVal = 1.0; // inject heat
        }
    }

    textureStore(dstTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(newVal, 0.0, 0.0, 1.0));
}`;
}

function vertexShader() {
    return `
@vertex
fn main(@builtin(vertex_index) idx : u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );
    return vec4<f32>(pos[idx], 0.0, 1.0);
}`;
}

function fragmentShader() {
    return `
@group(0) @binding(0) var img : texture_2d<f32>;

@fragment
fn main(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(${GRID_WIDTH}.0, ${GRID_HEIGHT}.0);
    let uv = pos.xy / dims;
    let value = textureLoad(img, vec2<i32>(i32(pos.x), i32(pos.y)), 0).r;
    return vec4<f32>(value, value, value, 1.0);
}`;
}

async function loadImageData(source) {
    return new Promise((resolve) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = source instanceof File ? URL.createObjectURL(source) : source;
        img.onload = () => {
            const width = img.width;
            const height = img.height;
            const offCanvas = document.createElement("canvas");
            offCanvas.width = width;
            offCanvas.height = height;
            const ctx = offCanvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            const imgData = ctx.getImageData(0, 0, width, height).data;

            const grayscale = new Float32Array(width * height * 4);
            for (let i = 0; i < width * height; i++) {
                const r = imgData[i * 4 + 0];
                const g = imgData[i * 4 + 1];
                const b = imgData[i * 4 + 2];
                const gray = (r + g + b) / (3 * 255);
                grayscale[i * 4 + 0] = gray;
                grayscale[i * 4 + 1] = 0;
                grayscale[i * 4 + 2] = 0;
                grayscale[i * 4 + 3] = 1;
            }
            resolve({ grayscale, width, height });
        };
    });
}

async function initSimulation(source) {
    const { grayscale, width, height } = await loadImageData(source);
    GRID_WIDTH = width;
    GRID_HEIGHT = height;

    canvas.width = GRID_WIDTH;
    canvas.height = GRID_HEIGHT;

    stateTexA = device.createTexture({
        size: [GRID_WIDTH, GRID_HEIGHT],
        format: "rgba32float",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });

    stateTexB = device.createTexture({
        size: [GRID_WIDTH, GRID_HEIGHT],
        format: "rgba32float",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });

    device.queue.writeTexture(
        { texture: stateTexA },
        grayscale,
        { bytesPerRow: GRID_WIDTH * 16 },
        [GRID_WIDTH, GRID_HEIGHT]
    );

    computeBindA = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: stateTexA.createView() },
            { binding: 1, resource: stateTexB.createView() },
            { binding: 2, resource: { buffer: paramBuffer } }
        ]
    });

    renderBind = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: stateTexA.createView() }
        ]
    });
}

function updateParams() {
    const data = new Float32Array([
        mousePos[0], mousePos[1], 10.0, // brush radius
        isDrawing ? 1 : 0
    ]);
    device.queue.writeBuffer(paramBuffer, 0, data.buffer);
}

function frame() {
    updateParams();

    const commandEncoder = device.createCommandEncoder();

    // Compute pass
    const passCompute = commandEncoder.beginComputePass();
    passCompute.setPipeline(computePipeline);
    passCompute.setBindGroup(0, computeBindA);
    passCompute.dispatchWorkgroups(Math.ceil(GRID_WIDTH / workgroupSize), Math.ceil(GRID_HEIGHT / workgroupSize));
    passCompute.end();

    // Render pass
    const textureView = context.getCurrentTexture().createView();
    const passRender = commandEncoder.beginRenderPass({
        colorAttachments: [{ view: textureView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }]
    });
    passRender.setPipeline(renderPipeline);
    passRender.setBindGroup(0, renderBind);
    passRender.draw(6);
    passRender.end();

    device.queue.submit([commandEncoder.finish()]);

    [stateTexA, stateTexB] = [stateTexB, stateTexA];
    computeBindA = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: stateTexA.createView() },
            { binding: 1, resource: stateTexB.createView() },
            { binding: 2, resource: { buffer: paramBuffer } }
        ]
    });
    renderBind = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: stateTexA.createView() }
        ]
    });

    requestAnimationFrame(frame);
}

// Mouse interaction
canvas.addEventListener('mousedown', () => { isDrawing = true; });
canvas.addEventListener('mouseup', () => { isDrawing = false; });
canvas.addEventListener('mouseleave', () => { isDrawing = false; });
canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * GRID_WIDTH;
    const y = (e.clientY - rect.top) / rect.height * GRID_HEIGHT;
    mousePos = [x, y];
});

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        await initSimulation(file);
    }
});

(async () => {
    await initWebGPU();
    await initSimulation('riemannian_ball.png'); // fallback image
    requestAnimationFrame(frame);
})();
