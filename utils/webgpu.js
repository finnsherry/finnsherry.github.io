export class TextureMaker {
    constructor(device, texFormat, gridWidth, gridHeight) {
        this.device = device;
        this.texFormat = texFormat;
        this.gridWidth = gridWidth;
        this.gridHeight = gridHeight;
    }

    createStateTexture() {
        return this.device.createTexture({
            size: [this.gridWidth, this.gridHeight],
            format: this.texFormat,
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST,
        });
    }
}

export class ComputePipelineMaker {
    constructor(device, texFormat, texFormatVec3, workgroupSize) {
        this.device = device;
        this.texFormat = texFormat;
        this.texFormatVec3 = texFormatVec3;
        this.workgroupSize = workgroupSize;
    }

    makeConvolutionPipelines(vec3) {
        let format
        let swizzle
        let out
        if (vec3) {
            format = this.texFormatVec3;
            swizzle = "rgb";
            out = `
                textureStore(
                    u_reg,
                    id.xy,
                    vec4f(out, 1)
                );
            `;
        } else {
            format = this.texFormat;
            swizzle = "r";
            out = `
                textureStore(
                    u_reg,
                    id.xy,
                    out * vec4f(1)
                );
            `;
        }

        const bodyWGSL = `
            let dims = textureDimensions(u);
            if (id.x >= dims.x || id.y >= dims.y) {
              return;
            }

            let dimx = i32(dims.x);
            let dimy = i32(dims.y);
            let x = i32(id.x);
            let y = i32(id.y);
            let r = i32(arrayLength(&k)) - 1;

            var out = textureLoad(u, vec2<i32>(x, y)).${swizzle} * k[0];
        `

        const computeWGSL = `

            @group(0) @binding(0) var<storage, read> k : array<f32>;
            @group(0) @binding(1) var u : texture_storage_2d<${format}, read>;
            @group(0) @binding(2) var u_reg : texture_storage_2d<${format}, write>;

            ${sanitise_index}

            @compute @workgroup_size(${this.workgroupSize}, ${this.workgroupSize})
            fn convolve_horizontal(@builtin(global_invocation_id) id : vec3<u32>) {
                ${bodyWGSL}

                for (var i = 1; i <= r; i++) {
                let xl = sanitise_index(x - i , dimx);
                let xr = sanitise_index(x + i , dimx);
                out += (textureLoad(u, vec2<i32>(xl, y)).${swizzle} + textureLoad(u, vec2<i32>(xr, y)).${swizzle})* k[i];
                }

                ${out}
            }

            @compute @workgroup_size(${this.workgroupSize}, ${this.workgroupSize})
            fn convolve_vertical(@builtin(global_invocation_id) id : vec3<u32>) {
                ${bodyWGSL}

                for (var i = 1; i <= r; i++) {
                let yl = sanitise_index(y - i , dimy);
                let yr = sanitise_index(y + i , dimy);
                out += (textureLoad(u, vec2<i32>(x, yl)).${swizzle} + textureLoad(u, vec2<i32>(x, yr)).${swizzle})* k[i];
                }

                ${out}
            }
            `;

        return [
            this.device.createComputePipeline({
                label: "horizontal convolution",
                layout: "auto",
                compute: {
                    module: this.device.createShaderModule({
                        code: computeWGSL,
                    }),
                    entryPoint: "convolve_horizontal",
                },
            }),
            this.device.createComputePipeline({
                label: "vertical convolution",
                layout: "auto",
                compute: {
                    module: this.device.createShaderModule({
                        code: computeWGSL,
                    }),
                    entryPoint: "convolve_vertical",
                },
            }),
        ];
    }

    makeBinarisationPipeline(threshold) {
        const computeWGSL = `
            @group(0) @binding(0) var u : texture_storage_2d<${this.texFormat}, read>;
            @group(0) @binding(1) var u_bin : texture_storage_2d<${this.texFormat}, write>;

            @compute @workgroup_size(${this.workgroupSize}, ${this.workgroupSize})
            fn binarise(@builtin(global_invocation_id) id : vec3<u32>) {
                let dims = textureDimensions(u);
                if (id.x >= dims.x || id.y >= dims.y) {
                return;
                }
                
                let u_cur = textureLoad(u, id.xy).r;
                var u_out = 0.;
                if (u_cur > ${threshold}) {
                    u_out = 255.;
                }

                textureStore(
                    u_bin,
                    id.xy,
                    u_out * vec4f(1)
                );
            }
        `;

        return this.device.createComputePipeline({
            label: "binarisation",
            layout: "auto",
            compute: {
                module: this.device.createShaderModule({
                    code: computeWGSL,
                }),
                entryPoint: "binarise",
            },
        });
    }
}

export const sanitise_index = `
    fn sanitise_index(i : i32, n : i32) -> i32 {
        let m = abs(i);
        return select(m, 2 * n - m - 2, m >= n);
    }
`

export const upwind_dilation = `
  fn upwind_dilation(df : f32, db : f32) -> f32 {
    return max(max(df, -db), 0.) * select(1., -1., df <= -db);
  }
`

export const upwind_erosion = `
  fn upwind_erosion(df : f32, db : f32) -> f32 {
    return max(max(-df, db), 0.) * select(1., -1., -df >= db);
  }
`

export async function renderImage(tex, device, context, format) {
    const renderWGSL = `
        @group(0) @binding(0) var tex : texture_2d<f32>;

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
            let c = textureLoad(tex, coord, 0).r / 255.;
            return vec4f(c, c, c, 1);
        }
    `;

    const renderPipeline = device.createRenderPipeline({
        label: "render",
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

    const renderBind = device.createBindGroup({
        label: "render",
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: tex.createView() }],
    });

    const encoder = device.createCommandEncoder();

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
    pass.setBindGroup(0, renderBind);
    pass.draw(6);
    pass.end();

    device.queue.submit([encoder.finish()]);
}

export async function setup() {
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

    return { device: device, canvas: canvas, context: context, format: format };
}

export function passMaker(encoder, pipeline, bindGroup, workGroupGrid) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...workGroupGrid);
    pass.end();
}