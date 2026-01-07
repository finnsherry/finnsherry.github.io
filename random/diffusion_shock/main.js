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
const texVec2Format = "rg32float";
const texVec3Format = "rgba32float";

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
      label: "horizontal convolution",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: computeWGSL,
        }),
        entryPoint: "convolve_horizontal",
      },
    }),
    device.createComputePipeline({
      label: "vertical convolution",
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

function makeConvolutionVec3Pipelines() {
  const computeWGSL = `

  @group(0) @binding(0) var<storage, read> k : array<f32>;
  @group(0) @binding(1) var u : texture_storage_2d<${texVec3Format}, read>;
  @group(0) @binding(2) var u_reg : texture_storage_2d<${texVec3Format}, write>;

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

    var out = textureLoad(u, vec2<i32>(x, y)).rgb * k[0];
    for (var i = 1; i <= r; i++) {
      let xl = sanitise_index(x - i , dimx);
      let xr = sanitise_index(x + i , dimx);
      out += (textureLoad(u, vec2<i32>(xl, y)).rgb + textureLoad(u, vec2<i32>(xr, y)).rgb) * k[i];
    }

    textureStore(
      u_reg,
      id.xy,
      vec4f(out, 1)
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

    var out = textureLoad(u, vec2<i32>(x, y)).rgb * k[0];
    for (var i = 1; i <= r; i++) {
      let yl = sanitise_index(y - i , dimy);
      let yr = sanitise_index(y + i , dimy);
      out += (textureLoad(u, vec2<i32>(x, yl)).rgb + textureLoad(u, vec2<i32>(x, yr)).rgb)* k[i];
    }

    textureStore(
      u_reg,
      id.xy,
      vec4f(out, 1)
    );
  }
`;

  return [
    device.createComputePipeline({
      label: "horizontal 3d convolution",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: computeWGSL,
        }),
        entryPoint: "convolve_horizontal",
      },
    }),
    device.createComputePipeline({
      label: "vertical 3d convolution",
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

function makeDSSwitchPipeline() {
  const computeWGSL = `
  struct Uniforms {
    delta: f32,
    dt: f32,
    lambda: f32,
    epsilon: f32,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(2) var switchDS : texture_storage_2d<${texFormat}, write>;\

  fn sanitise_index(i : i32, n : i32) -> i32 {
    let m = abs(i);
    return select(m, 2 * n - m - 2, m >= n);
  }

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(switchDS);
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

    let dx = (-1 * u_dmb - 2 * u_dxb - 1 * u_dpb + 1 * u_dmf + 2 * u_dxf + 1 * u_dpf) / 8;
    let dy = (-1 * u_dmf - 2 * u_dyb - 1 * u_dpb + 1 * u_dmb + 2 * u_dyf + 1 * u_dpf) / 8;

    textureStore(
      switchDS,
      id.xy,
      1 / sqrt(1 + (dx*dx + dy*dy) / (uniforms.lambda * uniforms.lambda)) * vec4f(1)
    );
  }
`;

  return device.createComputePipeline({
    label: "DS switch",
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

  @group(0) @binding(0) var u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(1) var J : texture_storage_2d<${texVec3Format}, write>;

  fn sanitise_index(i : i32, n : i32) -> i32 {
    let m = abs(i);
    return select(m, 2 * n - m - 2, m >= n);
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

    let u_dxf = textureLoad(u, vec2<i32>(I_dxf, y)).r;
    let u_dxb = textureLoad(u, vec2<i32>(I_dxb, y)).r;
    let u_dyf = textureLoad(u, vec2<i32>(x, I_dyf)).r;
    let u_dyb = textureLoad(u, vec2<i32>(x, I_dyb)).r;
    let u_dpf = textureLoad(u, vec2<i32>(I_dxf, I_dyf)).r;
    let u_dpb = textureLoad(u, vec2<i32>(I_dxb, I_dyb)).r;
    let u_dmf = textureLoad(u, vec2<i32>(I_dxf, I_dyb)).r;
    let u_dmb = textureLoad(u, vec2<i32>(I_dxb, I_dyf)).r;

    let dx = (-1 * u_dmb - 2 * u_dxb - 1 * u_dpb + 1 * u_dmf + 2 * u_dxf + 1 * u_dpf) / 8;
    let dy = (-1 * u_dmf - 2 * u_dyb - 1 * u_dpb + 1 * u_dmb + 2 * u_dyf + 1 * u_dpf) / 8;

    textureStore(
      J,
      id.xy,
      vec4f(
        dx * dx,
        dx * dy,
        dy * dy,
        1
      )
    );
  }
`;

  return device.createComputePipeline({
    label: "structure tensor",
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
  @group(0) @binding(1) var d_dd : texture_storage_2d<${texVec3Format}, write>;

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
      d_dd,
      id.xy,
      vec4f(
        u_dxf - 2 * u_c + u_dxb,
        u_dpf - u_dmf - u_dmb + u_dpb,
        u_dyf - 2 * u_c + u_dyb,
        1
      )
    );
  }
`;

  return device.createComputePipeline({
    label: "second order",
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
  @group(0) @binding(1) var d_dd : texture_storage_2d<${texVec3Format}, read>;
  @group(0) @binding(2) var J_rho : texture_storage_2d<${texVec3Format}, read>;
  @group(0) @binding(3) var switchMorph : texture_storage_2d<${texFormat}, write>;

  @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
  fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let dims = textureDimensions(switchMorph);
    if (id.x >= dims.x || id.y >= dims.y) {
      return;
    }

    let dd = textureLoad(d_dd, id.xy).rgb;
    let dxx = dd.r;
    let dxy = dd.g;
    let dyy = dd.b;
    let A = textureLoad(J_rho, id.xy).rgb;
    let A11 = A.r;
    let A12 = A.g;
    let A22 = A.b;

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
    label: "morphological switch",
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: computeWGSL,
      }),
      entryPoint: "main",
    },
  });
}

// function makeLaplacianPipeline() {
//   const computeWGSL = `
//   struct Uniforms {
//     delta: f32,
//     dt: f32,
//     lambda: f32,
//     epsilon: f32,
//   };

//   @group(0) @binding(0) var<uniform> uniforms: Uniforms;
//   @group(0) @binding(1) var u : texture_storage_2d<${texFormat}, read>;
//   @group(0) @binding(2) var laplacian_u : texture_storage_2d<${texFormat}, write>;

//   fn sanitise_index(i : i32, n : i32) -> i32 {
//     let m = abs(i);
//     return select(m, 2 * n - m - 2, m >= n);
//   }

//   @compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
//   fn laplacian(@builtin(global_invocation_id) id : vec3<u32>) {
//     let dims = textureDimensions(u);
//     if (id.x >= dims.x || id.y >= dims.y) {
//       return;
//     }

//     let dimx = i32(dims.x);
//     let dimy = i32(dims.y);
//     let x = i32(id.x);
//     let y = i32(id.y);

//     let I_dxf = sanitise_index(x + 1, dimx);
//     let I_dxb = sanitise_index(x - 1, dimx);
//     let I_dyf = sanitise_index(y + 1, dimy);
//     let I_dyb = sanitise_index(y - 1, dimy);

//     let u_c = textureLoad(u, vec2<i32>(x, y)).r;
//     let u_dxf = textureLoad(u, vec2<i32>(I_dxf, y)).r;
//     let u_dxb = textureLoad(u, vec2<i32>(I_dxb, y)).r;
//     let u_dyf = textureLoad(u, vec2<i32>(x, I_dyf)).r;
//     let u_dyb = textureLoad(u, vec2<i32>(x, I_dyb)).r;
//     let u_dpf = textureLoad(u, vec2<i32>(I_dxf, I_dyf)).r;
//     let u_dpb = textureLoad(u, vec2<i32>(I_dxb, I_dyb)).r;
//     let u_dmf = textureLoad(u, vec2<i32>(I_dxf, I_dyb)).r;
//     let u_dmb = textureLoad(u, vec2<i32>(I_dxb, I_dyf)).r;

//     textureStore(
//       laplacian_u,
//       id.xy,
//       (
//         (1 - uniforms.delta) * (u_dxf + u_dxb + u_dyf + u_dyb - 4 * u_c) +
//         (uniforms.delta / 2) * (u_dpf + u_dpb + u_dmf + u_dmb - 4 * u_c)
//       ) * vec4f(1)
//     );
//   }
// `;

//   return device.createComputePipeline({
//     layout: "auto",
//     compute: {
//       module: device.createShaderModule({
//         code: computeWGSL,
//       }),
//       entryPoint: "laplacian",
//     },
//   });
// }

function makeDiffusionPipeline() {
  const computeWGSL = `
  struct Uniforms {
    delta: f32,
    dt: f32,
    lambda: f32,
    epsilon: f32,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(2) var switch_DS : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(3) var diffusion : texture_storage_2d<${texFormat}, write>;

  fn sanitise_index(i : i32, n : i32) -> i32 {
    let m = abs(i);
    return select(m, 2 * n - m - 2, m >= n);
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
    let u_dxf = textureLoad(u, vec2<i32>(I_dxf, y)).r;
    let u_dxb = textureLoad(u, vec2<i32>(I_dxb, y)).r;
    let u_dyf = textureLoad(u, vec2<i32>(x, I_dyf)).r;
    let u_dyb = textureLoad(u, vec2<i32>(x, I_dyb)).r;
    let u_dpf = textureLoad(u, vec2<i32>(I_dxf, I_dyf)).r;
    let u_dpb = textureLoad(u, vec2<i32>(I_dxb, I_dyb)).r;
    let u_dmf = textureLoad(u, vec2<i32>(I_dxf, I_dyb)).r;
    let u_dmb = textureLoad(u, vec2<i32>(I_dxb, I_dyf)).r;

    let switch_DS_cur = textureLoad(switch_DS, vec2<i32>(x, y)).r;

    textureStore(
      diffusion,
      id.xy,
      switch_DS_cur * (
        (1 - uniforms.delta) * (u_dxf + u_dxb + u_dyf + u_dyb - 4 * u_c) +
        (uniforms.delta / 2) * (u_dpf + u_dpb + u_dmf + u_dmb - 4 * u_c)
      ) * vec4f(1)
    );
  }
`;

  return device.createComputePipeline({
    label: "diffusion",
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: computeWGSL,
      }),
      entryPoint: "main",
    },
  });
}

function makeShockPipeline() {
  const computeWGSL = `
  struct Uniforms {
    delta: f32,
    dt: f32,
    lambda: f32,
    epsilon: f32,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var u : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(2) var switch_DS : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(3) var switch_morph : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(4) var shock : texture_storage_2d<${texFormat}, write>;

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

    let dilation_u = (
        (1 - uniforms.delta) * sqrt(u_dx_dil * u_dx_dil + u_dy_dil * u_dy_dil) +
        (uniforms.delta / sqrt(2)) * sqrt(u_dp_dil * u_dp_dil + u_dm_dil * u_dm_dil)
    );
    let erosion_u = -(
        (1 - uniforms.delta) * sqrt(u_dx_ero * u_dx_ero + u_dy_ero * u_dy_ero) +
        (uniforms.delta / sqrt(2)) * sqrt(u_dp_ero * u_dp_ero + u_dm_ero * u_dm_ero)
    );

    let switch_DS_cur = textureLoad(switch_DS, vec2<i32>(x, y)).r;
    let switch_morph_cur = textureLoad(switch_morph, vec2<i32>(x, y)).r;

    textureStore(
      shock,
      id.xy,
      (1 - switch_DS_cur) * (
        select(dilation_u, erosion_u, switch_morph_cur > 0) * abs(switch_morph_cur)
      ) * vec4f(1)
    );
  }
`;

  return device.createComputePipeline({
    label: "shock",
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
  @group(0) @binding(2) var diffusion : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(3) var shock : texture_storage_2d<${texFormat}, read>;
  @group(0) @binding(4) var u : texture_storage_2d<${texFormat}, read_write>;

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
    let diffusion_cur = textureLoad(diffusion, vec2<i32>(x, y)).r;
    let shock_cur = textureLoad(shock, vec2<i32>(x, y)).r;

    textureStore(
      u,
      id.xy,
      u_cur + uniforms.dt * (mask_cur / 255) * (diffusion_cur + shock_cur) * vec4f(1)
    );
  }
`;

  return device.createComputePipeline({
    label: "step",
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
    let c = textureLoad(tex, coord, 0).r / 255.;
    return vec4f(c, c, c, 1);
  }
`;

  return device.createRenderPipeline({
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
  function createStateTextureVec3() {
    return device.createTexture({
      size: [gridWidth, gridHeight],
      format: texVec3Format,
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });
  }
  const u = createStateTexture();
  device.queue.writeTexture(
    { texture: u },
    u0,
    { bytesPerRow: gridWidth * 4 },
    [gridWidth, gridHeight]
  );
  const texMask = createStateTexture();
  device.queue.writeTexture(
    { texture: texMask },
    mask,
    { bytesPerRow: gridWidth * 4 },
    [gridWidth, gridHeight]
  );
  // const laplacian_u = createStateTexture();
  const diffusion = createStateTexture();
  const shock = createStateTexture();
  const convolutionStorage = createStateTexture();
  const u_nu = createStateTexture();
  const switch_DS = createStateTexture();
  const u_sigma = createStateTexture();
  const J = createStateTextureVec3();
  const J_rho = createStateTextureVec3();
  const convolutionStorageVec3 = createStateTextureVec3();
  const d_dd = createStateTextureVec3();
  const switch_morph = createStateTexture();

  // Compute pipeline and bind groups.
  function passMaker(pass, pipeline, bindGroup) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(gridWidth / WORKGROUP),
      Math.ceil(gridHeight / WORKGROUP)
    );
  }
  const [horizontalConvolutionPipeline, verticalConvolutionPipeline] =
    makeConvolutionPipelines();
  function horizontalConvolutionBind(k, src, dst) {
    return device.createBindGroup({
      label: "horizontal convolution",
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
      label: "vertical convolution",
      layout: verticalConvolutionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: k } },
        { binding: 1, resource: src.createView() },
        { binding: 2, resource: dst.createView() },
      ],
    });
  }

  const horizontalConvolutionNuBind = horizontalConvolutionBind(
    k_nu,
    u,
    convolutionStorage
  );
  const verticalConvolutionNuBind = verticalConvolutionBind(
    k_nu,
    convolutionStorage,
    u_nu
  );
  function convolutionNuPass(pass) {
    passMaker(pass, horizontalConvolutionPipeline, horizontalConvolutionNuBind);
    passMaker(pass, verticalConvolutionPipeline, verticalConvolutionNuBind);
  }

  const horizontalConvolutionSigmaBind = horizontalConvolutionBind(
    k_sigma,
    u,
    convolutionStorage
  );
  const verticalConvolutionSigmaBind = verticalConvolutionBind(
    k_sigma,
    convolutionStorage,
    u_sigma
  );
  function convolutionSigmaPass(pass) {
    passMaker(
      pass,
      horizontalConvolutionPipeline,
      horizontalConvolutionSigmaBind
    );
    passMaker(pass, verticalConvolutionPipeline, verticalConvolutionSigmaBind);
  }

  const [horizontalConvolutionVec3Pipeline, verticalConvolutionVec3Pipeline] =
    makeConvolutionVec3Pipelines();
  function horizontalConvolutionVec3Bind(k, src, dst) {
    return device.createBindGroup({
      label: "horizontal 3d convolution",
      layout: horizontalConvolutionVec3Pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: k } },
        { binding: 1, resource: src.createView() },
        { binding: 2, resource: dst.createView() },
      ],
    });
  }
  function verticalConvolutionVec3Bind(k, src, dst) {
    return device.createBindGroup({
      label: "vertical 3d convolution",
      layout: verticalConvolutionVec3Pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: k } },
        { binding: 1, resource: src.createView() },
        { binding: 2, resource: dst.createView() },
      ],
    });
  }
  const horizontalConvolutionJBind = horizontalConvolutionVec3Bind(
    k_rho,
    J,
    convolutionStorageVec3
  );
  const verticalConvolutionJBind = verticalConvolutionVec3Bind(
    k_rho,
    convolutionStorageVec3,
    J_rho
  );
  function regulariseStructureTensorPass(pass) {
    passMaker(
      pass,
      horizontalConvolutionVec3Pipeline,
      horizontalConvolutionJBind
    );
    passMaker(pass, verticalConvolutionVec3Pipeline, verticalConvolutionJBind);
  }

  const DSSwitchPipeline = makeDSSwitchPipeline();
  const DSSwitchBind = device.createBindGroup({
    label: "DS switch",
    layout: DSSwitchPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: u_nu.createView() },
      { binding: 2, resource: switch_DS.createView() },
    ],
  });
  function DSSwitchPass(pass) {
    passMaker(pass, DSSwitchPipeline, DSSwitchBind);
  }

  const structureTensorPipeline = makeStructureTensorPipeline();
  const structureTensorBind = device.createBindGroup({
    label: "structure tensor",
    layout: structureTensorPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: u_sigma.createView() },
      { binding: 1, resource: J.createView() },
    ],
  });
  function structureTensorPass(pass) {
    passMaker(pass, structureTensorPipeline, structureTensorBind);
  }

  const secondOrderPipeline = makeSecondOrderPipeline();
  const secondOrderBind = device.createBindGroup({
    label: "second order",
    layout: secondOrderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: u_sigma.createView() },
      { binding: 1, resource: d_dd.createView() },
    ],
  });
  function secondOrderPass(pass) {
    passMaker(pass, secondOrderPipeline, secondOrderBind);
  }

  const morphologicalSwitchPipeline = makeMorphologicalSwitchPipeline();
  const morphologicalSwitchBind = device.createBindGroup({
    label: "morphological switch",
    layout: morphologicalSwitchPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: d_dd.createView() },
      { binding: 2, resource: J_rho.createView() },
      { binding: 3, resource: switch_morph.createView() },
    ],
  });
  function morphologicalSwitchPass(pass) {
    passMaker(pass, morphologicalSwitchPipeline, morphologicalSwitchBind);
  }

  // const laplacianPipeline = makeLaplacianPipeline();
  // const laplacianBind = device.createBindGroup({
  //   layout: laplacianPipeline.getBindGroupLayout(0),
  //   entries: [
  //     { binding: 0, resource: { buffer: uniformBuffer } },
  //     { binding: 1, resource: u.createView() },
  //     { binding: 2, resource: laplacian_u.createView() },
  //   ],
  // });
  // function laplacianPass(pass) {
  //   passMaker(pass, laplacianPipeline, laplacianBind);
  // }

  const diffusionPipeline = makeDiffusionPipeline();
  const diffusionBind = device.createBindGroup({
    label: "diffusion",
    layout: diffusionPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: u.createView() },
      { binding: 2, resource: switch_DS.createView() },
      { binding: 3, resource: diffusion.createView() },
    ],
  });
  function diffusionPass(pass) {
    passMaker(pass, diffusionPipeline, diffusionBind);
  }

  const shockPipeline = makeShockPipeline();
  const shockBind = device.createBindGroup({
    label: "shock",
    layout: shockPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: u.createView() },
      { binding: 2, resource: switch_DS.createView() },
      { binding: 3, resource: switch_morph.createView() },
      { binding: 4, resource: shock.createView() },
    ],
  });
  function shockPass(pass) {
    passMaker(pass, shockPipeline, shockBind);
  }

  const stepPipeline = makeStepPipeline();
  const stepBind = device.createBindGroup({
    label: "step",
    layout: stepPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: texMask.createView() },
      { binding: 2, resource: diffusion.createView() },
      { binding: 3, resource: shock.createView() },
      { binding: 4, resource: u.createView() },
    ],
  });
  function stepPass(pass) {
    passMaker(pass, stepPipeline, stepBind);
  }

  // Render pipeline and bind groups.
  const renderPipeline = makeRenderPipeline();
  const renderBind = device.createBindGroup({
    label: "render",
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: u.createView() }],
  });

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
      convolutionNuPass(pass);
      DSSwitchPass(pass);

      // Morphological switch
      convolutionSigmaPass(pass);
      structureTensorPass(pass);
      regulariseStructureTensorPass(pass);
      secondOrderPass(pass);
      morphologicalSwitchPass(pass);

      // Derivatives
      diffusionPass(pass);
      shockPass(pass);

      // Step
      stepPass(pass);
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
      pass.setBindGroup(0, renderBind);
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
