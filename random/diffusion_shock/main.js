const adapter = await navigator.gpu?.requestAdapter();
const requiredLimits = {};
requiredLimits.maxStorageTexturesPerShaderStage = 8;
const device = await adapter?.requestDevice({ requiredLimits });
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

async function pngFileToArray(filePath) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";

    img.onload = () => {
      const width = img.naturalWidth;
      const height = img.naturalHeight;
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");

      ctx.drawImage(img, 0, 0, width, height);
      const { data } = ctx.getImageData(0, 0, width, height);

      const array = new Float32Array(width * height);
      for (let i = 0; i < width * height; i++) {
        const R = data[i * 4];
        const G = data[i * 4 + 1];
        const B = data[i * 4 + 2];
        array[i] = 0.2126 * R + 0.7152 * G + 0.0722 * B;
      }
      resolve({ array, width, height });
    };

    img.onerror = reject;
    img.src = filePath;
  });
}
const texFormat = "r32float";

function makeConvolutionPipelines() {
  const computeWGSL = `

  @group(0) @binding(0) var<storage, read> k : array<f32>;
  @group(0) @binding(1) var u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(2) var u_reg : texture_storage_2d<${texFormat}, write>;

  fn sanitise_index(i : i32, n : i32) -> i32 {
    let m = abs(i);
    return select(m, 2 * n - m - 2, m >= n);
  }

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn convolve_horizontal(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(u);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let dimx = i32(dims.x);
    let x = i32(id.x);
    let y = i32(id.y);
    let r = i32(arrayLength(&k)) - 1;

    var out = textureLoad(u, vec2<i32>(x, y)).r * k[0];
    for (var i = 1; i <= r; i++) {
      let xl = sanitise_index(x - i , dimx);
      let xr = sanitise_index(x + i , dimx);
      out += (textureLoad(u, vec2<i32>(xl, y)).r + textureLoad(u, vec2<i32>(xr, y)).r)* k[i];
    }

    textureStore(
      u_reg,
      id.xy,
      out * vec4f(1)
    );
  }

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn convolve_vertical(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(u);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let dimy = i32(dims.y);
    let x = i32(id.x);
    let y = i32(id.y);
    let r = i32(arrayLength(&k)) - 1;

    var out = textureLoad(u, vec2<i32>(x, y)).r * k[0];
    for (var i = 1; i <= r; i++) {
      let yl = sanitise_index(y - i , dimy);
      let yr = sanitise_index(y + i , dimy);
      out += (textureLoad(u, vec2<i32>(x, yl)).r + textureLoad(u, vec2<i32>(x, yr)).r)* k[i];
    }

    textureStore(
      u_reg,
      id.xy,
      out * vec4f(1)
    );
  }
`;

  return [
    device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: computeWGSL,
        }),
        entryPoint: "convolve_horizontal",
      },
    }),
    device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: computeWGSL,
        }),
        entryPoint: "convolve_vertical",
      },
    }),
  ];
}

function makeSobelPipeline() {
  const computeWGSL = `

  @group(0) @binding(0) var u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(1) var d_dx : texture_storage_2d<${texFormat}, write>;
  @group(0) @binding(2) var d_dy : texture_storage_2d<${texFormat}, write>;

  fn sanitise_index(i : i32, n : i32) -> i32 {
    let m = abs(i);
    return select(m, 2 * n - m - 2, m >= n);
  }

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn sobel_gradient(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(u);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let dimx = i32(dims.x);
    let dimy = i32(dims.y);
    let x = i32(id.x);
    let y = i32(id.y);

    let I_dxf = sanitise_index(x + 1, dimx);
    let I_dxb = sanitise_index(x - 1, dimx);
    let I_dyf = sanitise_index(y + 1, dimy);
    let I_dyb = sanitise_index(y - 1, dimy);

    let u_dxf = textureLoad(u, vec2<i32>(I_dxf, y)).r;
    let u_dxb = textureLoad(u, vec2<i32>(I_dxb, y)).r;
    let u_dyf = textureLoad(u, vec2<i32>(x, I_dyf)).r;
    let u_dyb = textureLoad(u, vec2<i32>(x, I_dyb)).r;
    let u_dpf = textureLoad(u, vec2<i32>(I_dxf, I_dyf)).r;
    let u_dpb = textureLoad(u, vec2<i32>(I_dxb, I_dyb)).r;
    let u_dmf = textureLoad(u, vec2<i32>(I_dxf, I_dyb)).r;
    let u_dmb = textureLoad(u, vec2<i32>(I_dxb, I_dyf)).r;

    textureStore(
      d_dx,
      id.xy,
      (-1 * u_dmb - 2 * u_dxb - 1 * u_dpb + 1 * u_dmf + 2 * u_dxf + 1 * u_dpf) / 8 * vec4f(1)
    );
    
    textureStore(
      d_dy,
      id.xy,
      (-1 * u_dmf - 2 * u_dyb - 1 * u_dpb + 1 * u_dmb + 2 * u_dyf + 1 * u_dpf) / 8 * vec4f(1)
    );
  }
`;

  return device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: computeWGSL,
      }),
      entryPoint: "sobel_gradient",
    },
  });
}

function makeDSSwitchPipeline() {
  const computeWGSL = `
  struct Uniforms {
    delta: f32,
    dt: f32,
    lambda: f32,
    epsilon: f32,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var d_dx : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(2) var d_dy : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(3) var switchDS : texture_storage_2d<${texFormat}, write>;

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(switchDS);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let dx = textureLoad(d_dx, id.xy).r;
    let dy = textureLoad(d_dy, id.xy).r;

    textureStore(
      switchDS,
      id.xy,
      1 / sqrt(1 + (dx*dx + dy*dy) / (uniforms.lambda * uniforms.lambda)) * vec4f(1)
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

function makeStructureTensorPipeline() {
  const computeWGSL = `

  @group(0) @binding(0) var d_dx : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(1) var d_dy : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(2) var J_11 : texture_storage_2d<${texFormat}, write>;
  @group(0) @binding(3) var J_12 : texture_storage_2d<${texFormat}, write>;
  @group(0) @binding(4) var J_22 : texture_storage_2d<${texFormat}, write>;

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(d_dx);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let dimx = i32(dims.x);
    let dimy = i32(dims.y);
    let x = i32(id.x);
    let y = i32(id.y);

    let dx = textureLoad(d_dx, vec2<i32>(x, y)).r;
    let dy = textureLoad(d_dy, vec2<i32>(x, y)).r;

    textureStore(
      J_11,
      id.xy,
      (dx * dx) * vec4f(1)
    );
    
    textureStore(
      J_12,
      id.xy,
      (dx * dy) * vec4f(1)
    );
    
    textureStore(
      J_22,
      id.xy,
      (dy * dy) * vec4f(1)
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

function makeSecondOrderPipeline() {
  const computeWGSL = `

  @group(0) @binding(0) var u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(1) var d_dxx : texture_storage_2d<${texFormat}, write>;
  @group(0) @binding(2) var d_dxy : texture_storage_2d<${texFormat}, write>;
  @group(0) @binding(3) var d_dyy : texture_storage_2d<${texFormat}, write>;

  fn sanitise_index(i : i32, n : i32) -> i32 {
    let m = abs(i);
    return select(m, 2 * n - m - 2, m >= n);
  }

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn central_derivatives_second_order(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(u);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let dimx = i32(dims.x);
    let dimy = i32(dims.y);
    let x = i32(id.x);
    let y = i32(id.y);

    let I_dxf = sanitise_index(x + 1, dimx);
    let I_dxb = sanitise_index(x - 1, dimx);
    let I_dyf = sanitise_index(y + 1, dimy);
    let I_dyb = sanitise_index(y - 1, dimy);

    let u_c = textureLoad(u, vec2<i32>(x, y)).r;
    let u_dxf = textureLoad(u, vec2<i32>(I_dxf, y)).r;
    let u_dxb = textureLoad(u, vec2<i32>(I_dxb, y)).r;
    let u_dyf = textureLoad(u, vec2<i32>(x, I_dyf)).r;
    let u_dyb = textureLoad(u, vec2<i32>(x, I_dyb)).r;
    let u_dpf = textureLoad(u, vec2<i32>(I_dxf, I_dyf)).r;
    let u_dpb = textureLoad(u, vec2<i32>(I_dxb, I_dyb)).r;
    let u_dmf = textureLoad(u, vec2<i32>(I_dxf, I_dyb)).r;
    let u_dmb = textureLoad(u, vec2<i32>(I_dxb, I_dyf)).r;

    textureStore(
      d_dxx,
      id.xy,
      (u_dxf - 2 * u_c + u_dxb) * vec4f(1)
    );
    
    textureStore(
      d_dxy,
      id.xy,
      (u_dpf - u_dmf - u_dmb + u_dpb) * vec4f(1)
    );
    
    textureStore(
      d_dyy,
      id.xy,
      (u_dyf - 2 * u_c + u_dyb) * vec4f(1)
    );
  }
`;

  return device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: computeWGSL,
      }),
      entryPoint: "central_derivatives_second_order",
    },
  });
}

function makeMorphologicalSwitchPipeline() {
  const computeWGSL = `
  struct Uniforms {
    delta: f32,
    dt: f32,
    lambda: f32,
    epsilon: f32,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var d_dxx : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(2) var d_dxy : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(3) var d_dyy : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(4) var J_rho_11 : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(5) var J_rho_12 : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(6) var J_rho_22 : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(7) var switchMorph : texture_storage_2d<${texFormat}, write>;

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(switchMorph);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let dxx = textureLoad(d_dxx, id.xy).r;
    let dxy = textureLoad(d_dxy, id.xy).r;
    let dyy = textureLoad(d_dyy, id.xy).r;
    let A11 = textureLoad(J_rho_11, id.xy).r;
    let A12 = textureLoad(J_rho_12, id.xy).r;
    let A22 = textureLoad(J_rho_22, id.xy).r;

    let v1 = -(-A11 + A22 - sqrt((A11 - A22)*(A11 - A22) + 4 * A12 * A12));
    let norm = sqrt(v1 * v1 + 4 * A12 * A12) + 0.000001;
    let c = v1 / norm;
    let s = 2 * A12 / norm;
    let d_dww = c * c * dxx + 2 * c * s * dxy + s * s * dyy;

    textureStore(
      switchMorph,
      id.xy,
      select(
        2 * atan2(d_dww, uniforms.epsilon) / 3.1416,
        sign(d_dww),
        uniforms.epsilon == 0
      ) * vec4f(1)
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

function makeLaplacianPipeline() {
  const computeWGSL = `
  struct Uniforms {
    delta: f32,
    dt: f32,
    lambda: f32,
    epsilon: f32,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(2) var laplacian_u : texture_storage_2d<${texFormat}, write>;

  fn sanitise_index(i : i32, n : i32) -> i32 {
    let m = abs(i);
    return select(m, 2 * n - m - 2, m >= n);
  }

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn laplacian(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(u);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let dimx = i32(dims.x);
    let dimy = i32(dims.y);
    let x = i32(id.x);
    let y = i32(id.y);

    let I_dxf = sanitise_index(x + 1, dimx);
    let I_dxb = sanitise_index(x - 1, dimx);
    let I_dyf = sanitise_index(y + 1, dimy);
    let I_dyb = sanitise_index(y - 1, dimy);

    let u_c = textureLoad(u, vec2<i32>(x, y)).r;
    let u_dxf = textureLoad(u, vec2<i32>(I_dxf, y)).r;
    let u_dxb = textureLoad(u, vec2<i32>(I_dxb, y)).r;
    let u_dyf = textureLoad(u, vec2<i32>(x, I_dyf)).r;
    let u_dyb = textureLoad(u, vec2<i32>(x, I_dyb)).r;
    let u_dpf = textureLoad(u, vec2<i32>(I_dxf, I_dyf)).r;
    let u_dpb = textureLoad(u, vec2<i32>(I_dxb, I_dyb)).r;
    let u_dmf = textureLoad(u, vec2<i32>(I_dxf, I_dyb)).r;
    let u_dmb = textureLoad(u, vec2<i32>(I_dxb, I_dyf)).r;

    textureStore(
      laplacian_u,
      id.xy,
      (
        (1 - uniforms.delta) * (u_dxf + u_dxb + u_dyf + u_dyb - 4 * u_c) +
        (uniforms.delta / 2) * (u_dpf + u_dpb + u_dmf + u_dmb - 4 * u_c)
      ) * vec4f(1)
    );
  }
`;

  return device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: computeWGSL,
      }),
      entryPoint: "laplacian",
    },
  });
}

function makeMorphologicalPipeline() {
  const computeWGSL = `
  struct Uniforms {
    delta: f32,
    dt: f32,
    lambda: f32,
    epsilon: f32,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(2) var dilation_u : texture_storage_2d<${texFormat}, write>;
  @group(0) @binding(3) var erosion_u : texture_storage_2d<${texFormat}, write>;

  fn sanitise_index(i : i32, n : i32) -> i32 {
    let m = abs(i);
    return select(m, 2 * n - m - 2, m >= n);
  }

  fn upwind_dilation(df : f32, db : f32) -> f32 {
    return max(max(df, -db), 0.) * select(1., -1., df <= -db);
  }

  fn upwind_erosion(df : f32, db : f32) -> f32 {
    return max(max(-df, db), 0.) * select(1., -1., -df >= db);
  }

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(u);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let dimx = i32(dims.x);
    let dimy = i32(dims.y);
    let x = i32(id.x);
    let y = i32(id.y);

    let I_dxf = sanitise_index(x + 1, dimx);
    let I_dxb = sanitise_index(x - 1, dimx);
    let I_dyf = sanitise_index(y + 1, dimy);
    let I_dyb = sanitise_index(y - 1, dimy);

    let u_c = textureLoad(u, vec2<i32>(x, y)).r;
    let u_dxf = textureLoad(u, vec2<i32>(I_dxf, y)).r - u_c;
    let u_dxb = u_c - textureLoad(u, vec2<i32>(I_dxb, y)).r;
    let u_dyf = textureLoad(u, vec2<i32>(x, I_dyf)).r - u_c;
    let u_dyb = u_c - textureLoad(u, vec2<i32>(x, I_dyb)).r;
    let u_dpf = textureLoad(u, vec2<i32>(I_dxf, I_dyf)).r - u_c;
    let u_dpb = u_c - textureLoad(u, vec2<i32>(I_dxb, I_dyb)).r;
    let u_dmf = textureLoad(u, vec2<i32>(I_dxf, I_dyb)).r - u_c;
    let u_dmb = u_c - textureLoad(u, vec2<i32>(I_dxb, I_dyf)).r;

    let u_dx_dil = upwind_dilation(u_dxf, u_dxb);
    let u_dy_dil = upwind_dilation(u_dyf, u_dyb);
    let u_dp_dil = upwind_dilation(u_dpf, u_dpb);
    let u_dm_dil = upwind_dilation(u_dmf, u_dmb);

    let u_dx_ero = upwind_erosion(u_dxf, u_dxb);
    let u_dy_ero = upwind_erosion(u_dyf, u_dyb);
    let u_dp_ero = upwind_erosion(u_dpf, u_dpb);
    let u_dm_ero = upwind_erosion(u_dmf, u_dmb);

    textureStore(
      dilation_u,
      id.xy,
      (
        (1 - uniforms.delta) * sqrt(u_dx_dil * u_dx_dil + u_dy_dil * u_dy_dil) +
        (uniforms.delta / sqrt(2)) * sqrt(u_dp_dil * u_dp_dil + u_dm_dil * u_dm_dil)
      ) * vec4f(1)
    );

    textureStore(
      erosion_u,
      id.xy,
      -(
        (1 - uniforms.delta) * sqrt(u_dx_ero * u_dx_ero + u_dy_ero * u_dy_ero) +
        (uniforms.delta / sqrt(2)) * sqrt(u_dp_ero * u_dp_ero + u_dm_ero * u_dm_ero)
      ) * vec4f(1)
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

function makeStepPipeline() {
  const computeWGSL = `
  struct Uniforms {
    delta: f32,
    dt: f32,
    lambda: f32,
    epsilon: f32,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var mask : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(2) var switch_DS : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(3) var switch_morph : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(4) var laplacian_u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(5) var dilation_u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(6) var erosion_u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(7) var u : texture_storage_2d<${texFormat}, read_write>;

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(u);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let dimx = i32(dims.x);
    let dimy = i32(dims.y);
    let x = i32(id.x);
    let y = i32(id.y);

    let u_cur = textureLoad(u, vec2<i32>(x, y)).r;
    let mask_cur = textureLoad(mask, vec2<i32>(x, y)).r;
    let switch_DS_cur = textureLoad(switch_DS, vec2<i32>(x, y)).r;
    let switch_morph_cur = textureLoad(switch_morph, vec2<i32>(x, y)).r;
    let laplacian_u_cur = textureLoad(laplacian_u, vec2<i32>(x, y)).r;
    let dilation_u_cur = textureLoad(dilation_u, vec2<i32>(x, y)).r;
    let erosion_u_cur = textureLoad(erosion_u, vec2<i32>(x, y)).r;

    textureStore(
      u,
      id.xy,
      (
        u_cur + uniforms.dt * (mask_cur / 255) * (
          laplacian_u_cur * switch_DS_cur +
          (1 - switch_DS_cur) * (
            select(dilation_u_cur, erosion_u_cur, switch_morph_cur > 0) * abs(switch_morph_cur)
          )
        )
      ) * vec4f(1)
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

function makeRenderPipeline() {
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
    let state = textureLoad(tex, coord, 0).r;
    return (state / 255.) * vec4f(1);
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

function createStateTexture(width, height) {
  return device.createTexture({
    size: [width, height],
    format: texFormat,
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST,
  });
}

function roundUpToMultiple(number, multiplier) {
  return Math.ceil(number / multiplier) * multiplier;
}

function createGaussianKernel(sigma, radiusMultiplier) {
  const r = Math.ceil(sigma * radiusMultiplier + 0.5);
  let k = new Float32Array(r + 1);
  let sum = 0;
  for (let i = 0; i < r + 1; i++) {
    const s = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i] = s;
    sum += s;
  }
  for (let i = 0; i < r + 1; i++) {
    k[i] /= sum;
  }
  const kBuffer = device.createBuffer({
    label: "Gaussian kernel",
    size: k.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(kBuffer, 0, k);
  return kBuffer;
}

let rafId = null;
async function runInpainting() {
  // User parameters.
  let showEvery = parseFloat(document.getElementById("showEvery").value);
  if (!showEvery) {
    showEvery = 1;
  }
  let lambda = parseFloat(document.getElementById("lambda").value);
  if (!lambda) {
    lambda = 2;
  }
  let nu = parseFloat(document.getElementById("nu").value);
  if (!nu) {
    nu = 2;
  }
  let sigma = parseFloat(document.getElementById("sigma").value);
  if (!sigma) {
    sigma = 2;
  }
  let rho = parseFloat(document.getElementById("rho").value);
  if (!rho) {
    rho = 5;
  }

  // Load data and make canvas.
  const { array: u0 } = await pngFileToArray("cross.png");
  const {
    array: mask,
    width: gridWidth,
    height: gridHeight,
  } = await pngFileToArray("mask.png");

  let texMask = createStateTexture(gridWidth, gridHeight);
  device.queue.writeTexture(
    { texture: texMask },
    mask,
    { bytesPerRow: gridWidth * 4 },
    [gridWidth, gridHeight]
  );
  canvas.width = gridWidth;
  canvas.height = gridHeight;

  // Define uniforms.
  const delta = Math.sqrt(2) - 1;
  const dt_diffusion = 1 / (4 - 2 * delta);
  const dt_shock = 1 / (Math.sqrt(2) * (1 - delta) + delta);
  const dt = Math.min(dt_diffusion, dt_shock);
  const epsilon = 0.15 * lambda;
  const radiusMultiplier = 5;

  const k_nu = createGaussianKernel(nu, radiusMultiplier);
  const k_sigma = createGaussianKernel(sigma, radiusMultiplier);
  const k_rho = createGaussianKernel(rho, radiusMultiplier);

  const uniforms = new Float32Array([delta, dt, lambda, epsilon]);
  const uniformBuffer = device.createBuffer({
    label: "Grid uniforms",
    size: roundUpToMultiple(uniforms.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniforms);

  // (Intermediate) textures.
  let u = createStateTexture(gridWidth, gridHeight);
  device.queue.writeTexture(
    { texture: u },
    u0,
    { bytesPerRow: gridWidth * 4 },
    [gridWidth, gridHeight]
  );
  const laplacian_u = createStateTexture(gridWidth, gridHeight);
  const dilation_u = createStateTexture(gridWidth, gridHeight);
  const erosion_u = createStateTexture(gridWidth, gridHeight);
  const convolutionStorage = createStateTexture(gridWidth, gridHeight);
  const d_dx_DS = createStateTexture(gridWidth, gridHeight);
  const d_dy_DS = createStateTexture(gridWidth, gridHeight);
  const switch_DS = createStateTexture(gridWidth, gridHeight);
  const u_sigma = createStateTexture(gridWidth, gridHeight);
  const d_dx_morph = createStateTexture(gridWidth, gridHeight);
  const d_dy_morph = createStateTexture(gridWidth, gridHeight);
  const J_11 = createStateTexture(gridWidth, gridHeight);
  const J_12 = createStateTexture(gridWidth, gridHeight);
  const J_22 = createStateTexture(gridWidth, gridHeight);
  const J_rho_11 = createStateTexture(gridWidth, gridHeight);
  const J_rho_12 = createStateTexture(gridWidth, gridHeight);
  const J_rho_22 = createStateTexture(gridWidth, gridHeight);
  const d_dxx = createStateTexture(gridWidth, gridHeight);
  const d_dxy = createStateTexture(gridWidth, gridHeight);
  const d_dyy = createStateTexture(gridWidth, gridHeight);
  const switch_morph = createStateTexture(gridWidth, gridHeight);

  // Compute pipeline and bind groups.

  const [horizontalConvolutionPipeline, verticalConvolutionPipeline] =
    makeConvolutionPipelines();
  function horizontalConvolutionBind(k, src, dst) {
    return device.createBindGroup({
      layout: horizontalConvolutionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: k } },
        { binding: 1, resource: src.createView() },
        { binding: 2, resource: dst.createView() },
      ],
    });
  }
  function verticalConvolutionBind(k, src, dst) {
    return device.createBindGroup({
      layout: verticalConvolutionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: k } },
        { binding: 1, resource: src.createView() },
        { binding: 2, resource: dst.createView() },
      ],
    });
  }

  const sobelPipeline = makeSobelPipeline();
  function sobelBind(src, d_dx, d_dy) {
    return device.createBindGroup({
      layout: sobelPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: src.createView() },
        { binding: 1, resource: d_dx.createView() },
        { binding: 2, resource: d_dy.createView() },
      ],
    });
  }

  const DSSwitchPipeline = makeDSSwitchPipeline();
  function DSSwitchBind(d_dx, d_dy, switchDS) {
    return device.createBindGroup({
      layout: DSSwitchPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: d_dx.createView() },
        { binding: 2, resource: d_dy.createView() },
        { binding: 3, resource: switchDS.createView() },
      ],
    });
  }

  const structureTensorPipeline = makeStructureTensorPipeline();
  function structureTensorBind(d_dx, d_dy, J_11, J_12, J_22) {
    return device.createBindGroup({
      layout: structureTensorPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: d_dx.createView() },
        { binding: 1, resource: d_dy.createView() },
        { binding: 2, resource: J_11.createView() },
        { binding: 3, resource: J_12.createView() },
        { binding: 4, resource: J_22.createView() },
      ],
    });
  }

  const secondOrderPipeline = makeSecondOrderPipeline();
  function secondOrderBind(src, d_dxx, d_dxy, d_dyy) {
    return device.createBindGroup({
      layout: secondOrderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: src.createView() },
        { binding: 1, resource: d_dxx.createView() },
        { binding: 2, resource: d_dxy.createView() },
        { binding: 3, resource: d_dyy.createView() },
      ],
    });
  }

  const morphologicalSwitchPipeline = makeMorphologicalSwitchPipeline();
  function morphologicalSwitchBind(
    d_dxx,
    d_dxy,
    d_dyy,
    J_rho_11,
    J_rho_12,
    J_rho_22,
    switchMorph
  ) {
    return device.createBindGroup({
      layout: morphologicalSwitchPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: d_dxx.createView() },
        { binding: 2, resource: d_dxy.createView() },
        { binding: 3, resource: d_dyy.createView() },
        { binding: 4, resource: J_rho_11.createView() },
        { binding: 5, resource: J_rho_12.createView() },
        { binding: 6, resource: J_rho_22.createView() },
        { binding: 7, resource: switchMorph.createView() },
      ],
    });
  }

  const laplacianPipeline = makeLaplacianPipeline();
  function laplacianBind(src, laplacian_u) {
    return device.createBindGroup({
      layout: laplacianPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: src.createView() },
        { binding: 2, resource: laplacian_u.createView() },
      ],
    });
  }

  const morphologicalPipeline = makeMorphologicalPipeline();
  function morphologicalBind(src, dilation_u, erosion_u) {
    return device.createBindGroup({
      layout: morphologicalPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: src.createView() },
        { binding: 2, resource: dilation_u.createView() },
        { binding: 3, resource: erosion_u.createView() },
      ],
    });
  }

  const stepPipeline = makeStepPipeline();
  function stepBind(
    mask,
    switchDS,
    switchMorph,
    laplacian_u,
    dilation_u,
    erosion_u,
    u
  ) {
    return device.createBindGroup({
      layout: stepPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: mask.createView() },
        { binding: 2, resource: switchDS.createView() },
        { binding: 3, resource: switchMorph.createView() },
        { binding: 4, resource: laplacian_u.createView() },
        { binding: 5, resource: dilation_u.createView() },
        { binding: 6, resource: erosion_u.createView() },
        { binding: 7, resource: u.createView() },
      ],
    });
  }

  // Render pipeline and bind groups.
  const renderPipeline = makeRenderPipeline();

  function renderBind(tex) {
    return device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: tex.createView() }],
    });
  }

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
      // DS switch
      {
        pass.setPipeline(horizontalConvolutionPipeline);
        pass.setBindGroup(
          0,
          horizontalConvolutionBind(k_nu, u, convolutionStorage)
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(verticalConvolutionPipeline);
        pass.setBindGroup(
          0,
          verticalConvolutionBind(k_nu, convolutionStorage, switch_DS)
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(sobelPipeline);
        pass.setBindGroup(0, sobelBind(switch_DS, d_dx_DS, d_dy_DS));
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(DSSwitchPipeline);
        pass.setBindGroup(0, DSSwitchBind(d_dx_DS, d_dy_DS, switch_DS));
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );
      }

      // Morphological switch
      {
        pass.setPipeline(horizontalConvolutionPipeline);
        pass.setBindGroup(
          0,
          horizontalConvolutionBind(k_sigma, u, convolutionStorage)
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(verticalConvolutionPipeline);
        pass.setBindGroup(
          0,
          verticalConvolutionBind(k_sigma, convolutionStorage, u_sigma)
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(sobelPipeline);
        pass.setBindGroup(0, sobelBind(u_sigma, d_dx_morph, d_dy_morph));
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(structureTensorPipeline);
        pass.setBindGroup(
          0,
          structureTensorBind(d_dx_morph, d_dy_morph, J_11, J_12, J_22)
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(horizontalConvolutionPipeline);
        pass.setBindGroup(
          0,
          horizontalConvolutionBind(k_rho, J_11, convolutionStorage)
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(verticalConvolutionPipeline);
        pass.setBindGroup(
          0,
          verticalConvolutionBind(k_rho, convolutionStorage, J_rho_11)
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(horizontalConvolutionPipeline);
        pass.setBindGroup(
          0,
          horizontalConvolutionBind(k_rho, J_12, convolutionStorage)
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(verticalConvolutionPipeline);
        pass.setBindGroup(
          0,
          verticalConvolutionBind(k_rho, convolutionStorage, J_rho_12)
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(horizontalConvolutionPipeline);
        pass.setBindGroup(
          0,
          horizontalConvolutionBind(k_rho, J_22, convolutionStorage)
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(verticalConvolutionPipeline);
        pass.setBindGroup(
          0,
          verticalConvolutionBind(k_rho, convolutionStorage, J_rho_22)
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(secondOrderPipeline);
        pass.setBindGroup(0, secondOrderBind(u_sigma, d_dxx, d_dxy, d_dyy));
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(morphologicalSwitchPipeline);
        pass.setBindGroup(
          0,
          morphologicalSwitchBind(
            d_dxx,
            d_dxy,
            d_dyy,
            J_rho_11,
            J_rho_12,
            J_rho_22,
            switch_morph
          )
        );
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );
      }

      // Derivatives
      {
        pass.setPipeline(laplacianPipeline);
        pass.setBindGroup(0, laplacianBind(u, laplacian_u));
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );

        pass.setPipeline(morphologicalPipeline);
        pass.setBindGroup(0, morphologicalBind(u, dilation_u, erosion_u));
        pass.dispatchWorkgroups(
          Math.ceil(gridWidth / WORKGROUP),
          Math.ceil(gridHeight / WORKGROUP)
        );
      }

      // Step
      pass.setPipeline(stepPipeline);
      pass.setBindGroup(
        0,
        stepBind(
          texMask,
          switch_DS,
          switch_morph,
          laplacian_u,
          dilation_u,
          erosion_u,
          u
        )
      );
      pass.dispatchWorkgroups(
        Math.ceil(gridWidth / WORKGROUP),
        Math.ceil(gridHeight / WORKGROUP)
      );

      pass.end();

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
      pass.setBindGroup(0, renderBind(u));
      pass.draw(6);
      pass.end();
    }

    device.queue.submit([encoder.finish()]);
    rafId = requestAnimationFrame(updateGrid);
  }
  requestAnimationFrame(updateGrid);
}

async function startInpainting() {
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  await runInpainting();
}

await runInpainting();
const submit = document.getElementById("submit");
submit.addEventListener("click", startInpainting);
