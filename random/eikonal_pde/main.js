function createVertexShader(gl, vertexstr) {
    let vertexshader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexshader, vertexstr);
    gl.compileShader(vertexshader);

    if (!gl.getShaderParameter(vertexshader, gl.COMPILE_STATUS))
        throw gl.getShaderInfoLog(vertexshader);

    return vertexshader;
}

function createFragmentShader(gl, fragmentstr) {
    let fragmentshader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentshader, fragmentstr);
    gl.compileShader(fragmentshader);

    if (!gl.getShaderParameter(fragmentshader, gl.COMPILE_STATUS))
        throw gl.getShaderInfoLog(fragmentshader);

    return fragmentshader;
}

function bytes_in_type(gl, type) {
    switch (type) {
        case gl.FLOAT:
            4
            break;

        case gl.INT:
            4
            break;

        default:
            throw `CAN NOT DETERMINE BYTES IN TYPE ${type}`
            break;
    }

}

function getGLTypesDictionary(gl) {
    let dict = {
        'gl.FLOAT': gl.FLOAT,
        'gl.FLOAT_VEC2': gl.FLOAT_VEC2,
        'gl.FLOAT_VEC3': gl.FLOAT_VEC3,
        'gl.FLOAT_VEC4': gl.FLOAT_VEC4,
        'gl.INT': gl.INT,
        'gl.INT_VEC2': gl.INT_VEC2,
        'gl.INT_VEC3': gl.INT_VEC3,
        'gl.INT_VEC4': gl.INT_VEC4,
        'gl.BOOL': gl.BOOL,
        'gl.BOOL_VEC2': gl.BOOL_VEC2,
        'gl.BOOL_VEC3': gl.BOOL_VEC3,
        'gl.BOOL_VEC4': gl.BOOL_VEC4,
        'gl.FLOAT_MAT2': gl.FLOAT_MAT2,
        'gl.FLOAT_MAT3': gl.FLOAT_MAT3,
        'gl.FLOAT_MAT4': gl.FLOAT_MAT4,
        'gl.SAMPLER_2D': gl.SAMPLER_2D,
        'gl.SAMPLER_CUBE': gl.SAMPLER_CUBE,
        'gl.UNSIGNED_INT': gl.UNSIGNED_INT,
        'gl.UNSIGNED_INT_VEC2': gl.UNSIGNED_INT_VEC2,
        'gl.UNSIGNED_INT_VEC3': gl.UNSIGNED_INT_VEC3,
        'gl.UNSIGNED_INT_VEC4': gl.UNSIGNED_INT_VEC4,
        'gl.FLOAT_MAT2x3': gl.FLOAT_MAT2x3,
        'gl.FLOAT_MAT2x4': gl.FLOAT_MAT2x4,
        'gl.FLOAT_MAT3x2': gl.FLOAT_MAT3x2,
        'gl.FLOAT_MAT3x4': gl.FLOAT_MAT3x4,
        'gl.FLOAT_MAT4x2': gl.FLOAT_MAT4x2,
        'gl.FLOAT_MAT4x3': gl.FLOAT_MAT4x3,
        'gl.SAMPLER_3D': gl.SAMPLER_3D,
        'gl.SAMPLER_2D_SHADOW': gl.SAMPLER_2D_SHADOW,
        'gl.SAMPLER_2D_ARRAY': gl.SAMPLER_2D_ARRAY,
        'gl.SAMPLER_2D_ARRAY_SHADOW': gl.SAMPLER_2D_ARRAY_SHADOW,
        'gl.SAMPLER_CUBE_SHADOW': gl.SAMPLER_CUBE_SHADOW,
        'gl.INT_SAMPLER_2D': gl.INT_SAMPLER_2D,
        'gl.INT_SAMPLER_3D': gl.INT_SAMPLER_3D,
        'gl.INT_SAMPLER_CUBE': gl.INT_SAMPLER_CUBE,
        'gl.INT_SAMPLER_2D_ARRAY': gl.INT_SAMPLER_2D_ARRAY,
        'gl.UNSIGNED_INT_SAMPLER_2D': gl.UNSIGNED_INT_SAMPLER_2D,
        'gl.UNSIGNED_INT_SAMPLER_3D': gl.UNSIGNED_INT_SAMPLER_3D,
        'gl.UNSIGNED_INT_SAMPLER_CUBE': gl.UNSIGNED_INT_SAMPLER_CUBE,
        'gl.UNSIGNED_INT_SAMPLER_2D_ARRAY': gl.UNSIGNED_INT_SAMPLER_2D_ARRAY,
    }
    for (let key in dict) {
        dict[dict[key]] = key
    }
    return dict;
}

function createProgram(gl, vertexShader, fragmentShader) {
    let typeDictionary = getGLTypesDictionary(gl);

    let P = gl.createProgram();

    //attach the shaders to our program
    gl.attachShader(P, vertexShader);
    gl.attachShader(P, fragmentShader);

    //link the program
    gl.linkProgram(P);
    if (!gl.getProgramParameter(P, gl.LINK_STATUS))
        throw gl.getProgramInfoLog(P);

    //find uniforms of program
    P.uniforms = {}
    let numUniforms = gl.getProgramParameter(P, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < numUniforms; ++i) {
        let uniform = gl.getActiveUniform(P, i);
        uniform.type_name = typeDictionary[uniform.type];
        uniform.location = gl.getUniformLocation(P, uniform.name);

        if (uniform.location == -1) {
            console.log(`could not find the location of the uniform ${uniform.name}`);
        }

        P.uniforms[uniform.name] = uniform;
    }

    P.attributes = {}
    let numAttributes = gl.getProgramParameter(P, gl.ACTIVE_ATTRIBUTES)
    for (let i = 0; i < numAttributes; ++i) {
        let attribute = gl.getActiveAttrib(P, i);
        attribute.type_name = typeDictionary[attribute.type]
        attribute.location = gl.getAttribLocation(P, attribute.name);

        if (attribute.location == -1) {
            console.log(`could not find the location of the attribute ${attribute.name}`);
        }

        P.attributes[attribute.name] = attribute;
    }

    console.log("created a program: ", P)

    return P
}

function createProgramFromSource(gl, vertexSource, fragmentSource) {
    let vertexShader = createVertexShader(gl, vertexSource);
    let fragmentShader = createFragmentShader(gl, fragmentSource);
    return createProgram(gl, vertexShader, fragmentShader);
}

function setUniform(gl, program, name, value) {
    //console.log("setting "+name+" to "+value);

    gl.useProgram(program);
    if (!program.uniforms[name]) {
        //console.log(name+" is no uniform of program and will be disregarded");
        return;
    }
    let type = program.uniforms[name].type;
    let location = program.uniforms[name].location;

    switch (type) {
        case gl.FLOAT:
            gl.uniform1f(location, value);
            break;

        case gl.FLOAT_VEC2:
            gl.uniform2fv(location, value);
            break;

        case gl.FLOAT_VEC3:
            gl.uniform3fv(location, value);
            break;

        case gl.FLOAT_VEC4:
            gl.uniform4fv(location, value);
            break;

        case gl.INT:
            gl.uniform1i(location, value);
            break;

        case gl.INT_VEC2:
            gl.uniform2iv(location, value);
            break;

        case gl.INT_VEC3:
            gl.uniform3iv(location, value);
            break;

        case gl.INT_VEC4:
            gl.uniform4iv(location, value);
            break;

        case gl.BOOL:
            gl.uniform1i(location, value);
            break;

        case gl.FLOAT_MAT2:
            gl.uniformMatrix2fv(location, true, value);
            break;

        case gl.FLOAT_MAT3:
            gl.uniformMatrix3fv(location, true, value);
            break;

        case gl.FLOAT_MAT4:
            gl.uniformMatrix4fv(location, true, value);
            break;

        case gl.SAMPLER_2D:
            gl.uniform1i(location, value);
            break;

        default:
            console.log("Type " + type + " is not supported")
            break;
    }
}

function setUniforms(gl, program, data) {
    for (let uniformName in data) {
        setUniform(gl, program, uniformName, data[uniformName])
    }
}

function checkFrameBufferStatus(gl) {
    switch (gl.checkFramebufferStatus(gl.FRAMEBUFFER)) {
        case gl.FRAMEBUFFER_COMPLETE:
            console.log("FRAMEBUFFER_COMPLETE")
            break;

        case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            throw "FRAMEBUFFER_INCOMPLETE_ATTACHMENT"
            break;

        case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            throw "FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT"
            break;

        case gl.FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            throw "FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER"
            break;

        case gl.FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            throw "FRAMEBUFFER_INCOMPLETE_READ_BUFFER"
            break;

        case gl.FRAMEBUFFER_UNSUPPORTED:
            throw "FRAMEBUFFER_UNSUPPORTED"
            break;

        case gl.FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            throw "FRAMEBUFFER_INCOMPLETE_MULTISAMPLE"
            break;

        case gl.FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            throw "FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS"
            break;

        default:
            throw "FRAMEBUFFER UKNOWN STATUS"
            break;
    }
}


//
let dt_sim = 1.0;
let dx = 1.0;

//pointers
let pointersPrev = new Map();
let pointers = new Map();

//textures
let vptextures = [];
let mtextures = [];

//thingies
let view = 0;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const maze = new Image();
maze.src = "eikonal_pde/maze.svg";
maze.onload = function () {
    const mazeData = ctx.drawImage(maze, 0, 0, canvas.clientWidth, canvas.clientHeight);
    // const data = mazeData.data
}

maze.onerror = function () {
    console.log("Failed to load maze")
}

// const gl = canvas.getContext("webgl2");
// if (!gl)
//     throw "Could not get webgl2 context";

// gl.preserveDrawingBuffer = true;

// if (!gl.getExtension('EXT_color_buffer_float'))
//     throw "Extension EXT_color_buffer_float not supported";

// if (!gl.getExtension('OES_texture_float_linear'))
//     throw "Extension OES_texture_float_linear not supported";

// const framebuffer = gl.createFramebuffer();

// const ms_label = document.getElementById("ms_label");
// const resolution_label = document.getElementById("resolution_label");
// const resolution_x_input = document.getElementById("resolutionX");
// const resolution_y_input = document.getElementById("resolutionY");
// const iterations_input = document.getElementById("iterations");
// const save_button = document.getElementById("save");
// const fullscreen_button = document.getElementById("fullscreen");
// const reset_button = document.getElementById("reset");

// window.addEventListener("error", function (e) {
//     console.log(e);
//     alert('Error message: ' + e.message + '\nLine Number: ' + e.lineno + '\nColumn Number: ' + e.colno + "\nStack:\n" + e.error.stack);
//     return true;
// })

// canvas.addEventListener("pointerdown", function (e) {
//     console.log("pointer down", e);
//     canvas.setPointerCapture(e.pointerId);
//     pointers.set(e.pointerId, e);
//     pointersPrev.set(e.pointerId, e);
// });

// canvas.addEventListener("pointermove", function (e) {
//     // console.log("pointer move", e);
//     pointers.set(e.pointerId, e);
// });

// canvas.addEventListener("pointerup", function (e) {
//     console.log("pointer up", e);
//     pointers.delete(e.pointerId);
// });

// canvas.addEventListener("pointercancel", function (e) {
//     console.log("pointer cancel", e);
//     pointers.delete(e.pointerId);
// });

// save_button.addEventListener("click", function (e) {
//     drawToCanvas()
//     captureCanvas(canvas, "image.png");
// })

// reset_button.addEventListener("click", createTextures)

// fullscreen_button.addEventListener("click", function (e) {
//     document.getElementById('canvas container').requestFullscreen();
// });


// function worldFromClient(clientX, clientY) {
//     const [canvasX, canvasY] = canvasFromClient(canvas, clientX, clientY);
//     return [canvasX, canvas.height - canvasY];
// }


// const quadBuffer = gl.createBuffer();
// gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
// gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

// const sharedShader = `#version 300 es

// #ifdef GL_FRAGMENT_PRECISION_HIGH
// precision highp float;
// #else
// precision mediump float;
// #endif

// uniform sampler2D vp;
// uniform sampler2D m;
// uniform float dx;
// uniform float dt;
// uniform float iTime;
// uniform vec2  iResolution;
// uniform bool  interacting;
// uniform vec2  pointer0_prev;
// uniform vec2  pointer0;
// `

// const fragmentShaders = {
//     post: `

// uniform int iView;

// out vec4 fragColor;

// void main(){
//     vec4 vps = texelFetch(vp, ivec2(gl_FragCoord.xy), 0);
//     vec4 ms = texelFetch(m, ivec2(gl_FragCoord.xy), 0);  
    
//     if(iView==0){
//         fragColor = vec4(ms.x, ms.y, ms.z, ms.x + ms.y + ms.z);
//     }else{
//         fragColor = vec4(abs(vps.xy),abs(vps.z)*10.0,1);
//     }
// }
//     `,

//     advectdye: `

// out vec4 fragColor;

// void main()
// {
//     vec4 o = texture(vp, (gl_FragCoord.xy + vec2( 0, 0))/iResolution.xy);
  
//     fragColor = texture(m, (gl_FragCoord.xy-o.xy*dt)/iResolution);
// }
//     `,

//     advectfluid: `

// out vec4 fragColor;

// void main()
// {
//     vec4 o = texture(vp, (gl_FragCoord.xy + vec2( 0, 0))/iResolution.xy);
  
//     fragColor = texture(vp, (gl_FragCoord.xy-o.xy*dt)/iResolution);
// }
//     `,

//     boundaryfluid: `

// out vec4 fragColor;

// void main()
// {
//     vec4 o = texture(vp, (gl_FragCoord.xy + vec2( 0, 0))/iResolution.xy);
//     vec4 n = texture(vp, (gl_FragCoord.xy + vec2( 0, 1))/iResolution.xy);
//     vec4 e = texture(vp, (gl_FragCoord.xy + vec2( 1, 0))/iResolution.xy);
//     vec4 s = texture(vp, (gl_FragCoord.xy + vec2( 0,-1))/iResolution.xy);
//     vec4 w = texture(vp, (gl_FragCoord.xy + vec2(-1, 0))/iResolution.xy);
    
//     fragColor = o;
    
//     if(gl_FragCoord.x <= 2.5f)
//     {
//         fragColor.xy = -e.xy;
//         fragColor.z = e.z;
//     }
    
//     if(gl_FragCoord.y <= 2.5f)
//     {
//         fragColor.xy = -n.xy;
//         fragColor.z = n.z;
//     }
    
//     if(gl_FragCoord.x >= iResolution.x-2.5f)
//     {
//         fragColor.xy = -w.xy;
//         fragColor.z = w.z;
//     }
    
//     if(gl_FragCoord.y >= iResolution.y-2.5f)
//     {
//         fragColor.xy = -s.xy;
//         fragColor.z = s.z;
//     }
// }
//     `,

//     interactivityfluid: `

// out vec4 fragColor;

// void main()
// {
//     vec4 o = texture(vp, (gl_FragCoord.xy + vec2( 0, 0))/iResolution.xy);

//     const float interaction_falloff = 0.05;
//     const float interaction_strength = 0.1;

//     if (interacting) {
//         vec2 m0 = pointer0_prev;
//         vec2 m1 = pointer0;
//         vec2 m = m1 - m0;

//         float t = clamp(dot(gl_FragCoord.xy - m0, m) / (dot(m, m) + 1e-7), 0.0, 1.0);
//         vec2 mt = m0 + m * t;

//         vec2 diff = gl_FragCoord.xy - mt;
//         float dist = interaction_falloff * length(diff);
//         float strength = interaction_strength * exp( - dist * dist);

//         o.xy += strength * (m / dt - o.xy) * dt;
//     }
    
//     fragColor = o;
// }
//     `,

//     pressurepoisson: `

// out vec4 fragColor;

// void main()
// {
//     vec4 o = texture(vp, (gl_FragCoord.xy + vec2( 0, 0))/iResolution.xy);
//     vec4 n = texture(vp, (gl_FragCoord.xy + vec2( 0, 1))/iResolution.xy);
//     vec4 e = texture(vp, (gl_FragCoord.xy + vec2( 1, 0))/iResolution.xy);
//     vec4 s = texture(vp, (gl_FragCoord.xy + vec2( 0,-1))/iResolution.xy);
//     vec4 w = texture(vp, (gl_FragCoord.xy + vec2(-1, 0))/iResolution.xy);
  
//     // float divergence of the velocity
//     float div = (e.x - w.x + n.y - s.y) / (2.0f * dx * dx);
    
//     // one jacobi iteration
//     float a = 1.0f / ( dx * dx);
//     float p = 1.0f / ( -4.0f * a ) * ( div - a * (n.z + e.z + s.z + w.z));

//     fragColor = vec4(o.xy, p, o.w);
// }
//     `,

//     project: `

// out vec4 fragColor;

// void main()
// {
//     vec4 o = texture(vp, (gl_FragCoord.xy + vec2( 0, 0))/iResolution.xy);
//     vec4 n = texture(vp, (gl_FragCoord.xy + vec2( 0, 1))/iResolution.xy);
//     vec4 e = texture(vp, (gl_FragCoord.xy + vec2( 1, 0))/iResolution.xy);
//     vec4 s = texture(vp, (gl_FragCoord.xy + vec2( 0,-1))/iResolution.xy);
//     vec4 w = texture(vp, (gl_FragCoord.xy + vec2(-1, 0))/iResolution.xy);
    
//     // gradient of the pressure
//     vec2 grad = vec2( e.z - w.z, n.z - s.z ) / (2.0f * dx * dx);

//     // project
//     fragColor = vec4(o.xy - grad, o.zw);
// }
//     `,

//     interactivitydye: `

// vec3 palette(float t){
//     return 0.5 + 0.5*cos(t+vec3(0,.33,.66)*6.2830);
// }

// out vec4 fragColor;

// void main()
// {
//     vec4 o = texture(m, (gl_FragCoord.xy + vec2( 0, 0))/iResolution.xy);
    
//     if(interacting)
//     {
//         vec2 m0 = pointer0_prev;
//         vec2 m1 = pointer0;
//         vec2 m = m1 - m0;

//         float t = clamp(dot(gl_FragCoord.xy - m0, m) / (dot(m, m) + 1e-7), 0.0, 1.0);
//         vec2 mt = m0 + m * t;

//         float d = distance(gl_FragCoord.xy, mt);
//         o.xyz += palette(iTime*0.001) * exp(-d*0.2);
//     }
    
//     o = min(o, vec4(.8));
    
//     fragColor = o;
// }
//     `,
// }
// const vertexShaders = {
//     "standard": `#version 300 es

// in vec4 a_position;

// void main() {
//     gl_Position = a_position;
// }
//     `,
// }

// const programs = {};

// for (let frag in fragmentShaders) {
//     console.log(`creating program ${frag}`)
//     programs[frag] = createProgramFromSource(gl, vertexShaders["standard"], sharedShader + fragmentShaders[frag]);
// }

// createTextures()

// function createTexture(width, height) {
//     let texture = gl.createTexture();
//     gl.bindTexture(gl.TEXTURE_2D, texture);
//     {
//         gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
//         gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
//         gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
//     }
//     gl.bindTexture(gl.TEXTURE_2D, null);
//     return texture;
// }

// function createTextures() {
//     for (let i = 0; i < 2; i++) {
//         vptextures[i] = createTexture(canvas.width, canvas.height);
//     }
//     for (let i = 0; i < 2; i++) {
//         mtextures[i] = createTexture(canvas.width, canvas.height);
//     }
// }

// function setAllUniforms() {
//     for (const programName in programs) {
//         let program = programs[programName];

//         setUniforms(gl, program, {
//             iResolution: [canvas.width, canvas.height],
//             iTime: time,
//             dx: dx,
//             dt: dt_sim,
//             vp: 0, //VP corresponds to TEXTURE0
//             m: 1,  //M corresponds to TEXTURE1
//             iView: view,
//             interacting: false,
//         })


//         if (pointers.size == 1) {
//             const pointerId = Array.from(pointers.keys())[0];

//             const pointerPrev = pointersPrev.get(pointerId);
//             if (pointerPrev) {
//                 setUniforms(gl, program, {
//                     pointer0_prev: worldFromClient(pointerPrev.clientX, pointerPrev.clientY)
//                 })
//             }

//             const pointer = pointers.get(pointerId);
//             if (pointer) {
//                 setUniforms(gl, program, {
//                     interacting: pointer.buttons & 1,
//                     pointer0: worldFromClient(pointer.clientX, pointer.clientY)
//                 })
//             }

//         }

//     }
// }

// function bind() {
//     //bind vptextures[1] to TEXTURE0
//     gl.activeTexture(gl.TEXTURE0);
//     gl.bindTexture(gl.TEXTURE_2D, vptextures[1]);

//     //bind mtextures[1] to TEXTURE1
//     gl.activeTexture(gl.TEXTURE1);
//     gl.bindTexture(gl.TEXTURE_2D, mtextures[1]);
// }

// function triangle() {
//     gl.enableVertexAttribArray(0);
//     gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
//     gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
// }

// function advectdye() {
//     bind();
//     gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, mtextures[0], 0);
//     gl.useProgram(programs["advectdye"]);
//     triangle();
//     mtextures.reverse();
// }

// function advectfluid() {
//     bind();
//     gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, vptextures[0], 0);
//     gl.useProgram(programs["advectfluid"]);
//     triangle();
//     vptextures.reverse();
// }

// function pressurePoisson() {
//     bind();
//     gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, vptextures[0], 0);
//     gl.useProgram(programs["pressurepoisson"]);
//     triangle()
//     vptextures.reverse();
// }

// function project() {
//     bind();
//     gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, vptextures[0], 0);
//     gl.useProgram(programs["project"]);
//     triangle()
//     vptextures.reverse();
// }

// function boundaryfluid() {
//     bind();
//     gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, vptextures[0], 0);
//     gl.useProgram(programs["boundaryfluid"]);
//     triangle();
//     vptextures.reverse();
// }

// function interactivitydye() {
//     bind();
//     gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, mtextures[0], 0);
//     gl.useProgram(programs["interactivitydye"]);
//     triangle();
//     mtextures.reverse();
// }

// function interactivityfluid() {
//     bind();
//     gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, vptextures[0], 0);
//     gl.useProgram(programs["interactivityfluid"]);
//     triangle();
//     vptextures.reverse();
// }

// function simulate() {
//     gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

//     gl.viewport(0, 0, canvas.width, canvas.height);

//     setAllUniforms()

//     interactivitydye();
//     advectdye();

//     interactivityfluid();
//     advectfluid();

//     for (let i = 0; i < parseInt(iterations_input.value); i++) {
//         pressurePoisson();
//         boundaryfluid();
//     }

//     project();

//     //increment frame counter
//     frames++;
// }

// function drawToCanvas() {
//     gl.viewport(0, 0, canvas.width, canvas.height);

//     //use the canvas
//     gl.bindFramebuffer(gl.FRAMEBUFFER, null);

//     bind();

//     //use post program
//     gl.useProgram(programs["post"]);

//     //draw triangles
//     triangle();
// }

// let time = performance.now();
// function loop() {
//     const newtime = performance.now();
//     const dt = newtime - time;
//     time = newtime

//     let resx = resolution_x_input.value ? resolution_x_input.value : 1;
//     let resy = resolution_y_input.value ? resolution_y_input.value : 1;

//     if (canvas.width != resx || canvas.height != resy) {
//         canvas.width = resx;
//         canvas.height = resy;
//         createTextures();
//     }


//     simulate();
//     drawToCanvas();

//     //update dom
//     ms_label.innerHTML = dt.toFixed(0) + "ms";
//     resolution_label.innerHTML = "Resolution : " + canvas.width + "x" + canvas.height

//     pointersPrev = new Map(pointers);

//     requestAnimationFrame(loop);
// }

// requestAnimationFrame(loop);