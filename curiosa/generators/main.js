import { NDArray, vec2, vec3, vec4, mat3, mat4 } from "/utils/linalg.js";
import { resizeCanvasToDisplaySize } from "/utils/canvas.js";
import { Camera } from "/utils/camera.js";
import { InputState, getInputNumber, setInput } from "/utils/input.js";

const canvas = document.getElementById("canvas");
resizeCanvasToDisplaySize(canvas);
const width = canvas.width;
const height = canvas.height;
const length = Math.min(width, height) / 2;
const aspect = width / height;
const ctx = canvas.getContext("2d");

const black = `rgb(0 0 0)`;
const white = `rgb(255 255 255)`;
const brickRed = `rgb(${11 * 16 + 6}, ${3 * 16 + 2}, ${1 * 16 + 12})`;  // B6321C
const forestGreen = `rgb(${0 * 16 + 0}, ${9 * 11 + 2}, ${5 * 16 + 5})`; // 009B55
const royalBlue = `rgb(${0 * 16 + 0}, ${7 * 16 + 1}, ${11 * 16 + 12})`; // 0071BC
const axisColour = `rgb(0 0 0 / 0.1)`;

const origin = vec3(0, 0, 0);
const nx = vec3(1, 0, 0);
const ny = vec3(0, 1, 0);
const nz = vec3(0, 0, 1);
const epsilon = 0.00001;

const inputState = new InputState();
inputState.attach(canvas);

const camera = new Camera(vec3(0, 3, 3), origin);
camera.perspective = true;
camera.zoom = 1 / 5;

function project(point, perspective) {
  const projected = NDArray.matmul(perspective, vec4(point.x, point.y, point.z, 2));
  return vec2(
    width / 2 + length * projected.x / projected.w,
    height - (height / 2 + length * projected.y / projected.w),
  );
}

function drawPoint(point, perspective, colour = black, radius = 0.05) {
  const projected = project(point, perspective);
  ctx.beginPath();
  ctx.fillStyle = colour;
  ctx.arc(projected.x, projected.y, radius * length, 0, 2 * Math.PI, true);
  ctx.fill();
}

function drawArrow(start, end, perspective, colour = black, headLength = 0.7, width = 0.01) {
  const projectedStart = project(start, perspective);
  const projectedEnd = project(end, perspective);
  const difference = NDArray.sub(projectedEnd, projectedStart);
  const startOfHead = NDArray.add(projectedStart, NDArray.mul(difference, headLength));
  const orthogonal = NDArray.normalize(vec2(-difference.y, difference.x));
  const rightPoint = NDArray.add(startOfHead, NDArray.mul(orthogonal, width * length * 1.5));
  const leftPoint = NDArray.add(startOfHead, NDArray.mul(orthogonal, -width * length * 1.5));

  ctx.lineWidth = length * width;
  ctx.strokeStyle = colour;
  ctx.fillStyle = colour;
  ctx.beginPath();
  ctx.moveTo(projectedStart.x, projectedStart.y);
  ctx.lineTo(startOfHead.x, startOfHead.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(leftPoint.x, leftPoint.y);
  ctx.lineTo(rightPoint.x, rightPoint.y);
  ctx.lineTo(projectedEnd.x, projectedEnd.y);
  ctx.closePath();
  ctx.fill();
}

class Generator {
  constructor(velocity, rotationGenerator) {
    this.v = velocity;
    this.omega = rotationGenerator;
  }

  get mat() {
    let matrix = NDArray.fromMat3(this.omega);
    matrix.data[4 * 0 + 3] = this.v.x;
    matrix.data[4 * 1 + 3] = this.v.y;
    matrix.data[4 * 2 + 3] = this.v.z;
    matrix.data[4 * 3 + 3] = 0;
    return matrix;
  }

  //   exp() {
  //     // Rodrigues
  //     const omega_vec = vec3(-this.omega.data[5], this.omega.data[2], -this.omega.data[1]);
  //     const theta = NDArray.norm(omega_vec);
  //     if (theta < epsilon) {
  //       const R = NDArray.eye(3)
  //     } else {
  //       const omega_s = NDArray.mul(this.omega, 1 / theta);
  //       const cos = Math.cos(theta);
  //       const sin = Math.sin(theta);
  //       const R = NDArray.add(
  //         NDArray.eye(3),
  //         NDArray.add(
  //           NDArray.mul(
  //             omega_s,
  //             sin,
  //           ),
  //           NDArray.mul(
  //             NDArray.matmul(omega_s, omega_s),
  //             (1 - cos),
  //           ),
  //         ),
  //       );
  //     }


  //     return 
  //   }
}

class PositionOrientation {
  constructor(position, orientation) {
    this.x = position;
    this.n = orientation;
  }

  get mat() {
    let matrix = new NDArray(
      [
        this.x.x, this.n.x,
        this.x.y, this.n.y,
        this.x.z, this.n.z,
        1, 0,
      ],
      [4, 2],
    );
    return matrix;
  }

  static fromMat(matrix) {
    const x = vec3(matrix.data[0], matrix.data[2], matrix.data[4]);
    const n = vec3(matrix.data[1], matrix.data[3], matrix.data[5]);
    return new PositionOrientation(x, n);
  }

  draw(perspective, colour = black, headLength = 0.7, width = 0.01) {
    drawArrow(this.x, NDArray.add(this.x, this.n), perspective, colour, headLength, width);
  }

  static getGenerator(p1, p2, generator) {
    if (!(generator === "mav" || generator === "pure_rotation" || typeof (generator) === "number")) {
      throw new Error("Generator.getGenerator: generator must be either 'mav', 'pure_rotation', or a number.");
    }

    const x1 = p1.x;
    const n1 = p1.n;
    const x2 = p2.x;
    const n2 = p2.n;

    const x_m = NDArray.mul(NDArray.add(x1, x2), 0.5);
    const x_diff = NDArray.sub(x2, x1);
    const cross_n = NDArray.cross(n1, n2);
    const sinTheta = NDArray.norm(cross_n);
    const cosTheta = NDArray.dot(n1, n2);
    const theta_mav = Math.atan2(sinTheta, cosTheta);

    if (theta_mav < epsilon) {
      console.log("Small angle so only translating.");
      return new Generator(x_diff, NDArray.zeros([3, 3]));
    }

    const k0 = NDArray.mul(cross_n, 1 / sinTheta);
    const x_perp_mav = NDArray.mul(k0, NDArray.dot(k0, x_diff));
    const x_par_mav = NDArray.sub(x_diff, x_perp_mav);

    const centre_mav = NDArray.add(
      x_m,
      NDArray.mul(
        NDArray.cross(k0, x_par_mav),
        (0.5 / Math.tan(theta_mav / 2.0))
      )
    );
    const v_mav = x_perp_mav;
    const omega_vec_mav = NDArray.mul(k0, theta_mav);
    const omega_mav = mat3([
      0, -omega_vec_mav.z, omega_vec_mav.y,
      omega_vec_mav.z, 0, -omega_vec_mav.x,
      -omega_vec_mav.y, omega_vec_mav.x, 0,
    ]);

    if (NDArray.norm(x_diff) < epsilon) {
      return new Generator(
        NDArray.sub(v_mav, NDArray.cross(omega_vec_mav, centre_mav)),
        omega_mav
      );
    }

    let k
    switch (typeof generator) {
      case "string":
        switch (generator) {
          case "mav":
            return new Generator(
              NDArray.sub(v_mav, NDArray.cross(omega_vec_mav, centre_mav)),
              omega_mav
            );

          case "pure_rotation":
            k = NDArray.cross(x_diff, NDArray.sub(n2, n1));
            break;

        }
        break;

      case "number":
        const phi = generator;
        const n1plusn2 = NDArray.add(n1, n2);
        const khalfpi = NDArray.mul(n1plusn2, 1 / NDArray.norm(n1plusn2))
        k = NDArray.add(
          NDArray.mul(k0, Math.cos(phi)),
          NDArray.mul(khalfpi, Math.sin(phi)),
        );
        break;
    }
    k = NDArray.normalize(k);

    const n1p = NDArray.sub(n1, NDArray.mul(k, NDArray.dot(k, n1)));
    const n2p = NDArray.sub(n2, NDArray.mul(k, NDArray.dot(k, n2)));
    const theta = Math.atan2(
      NDArray.dot(k, NDArray.cross(n1p, n2p)),
      NDArray.dot(n1p, n2p),
    );

    const x_perp = NDArray.mul(k, NDArray.dot(k, x_diff));
    const x_par = NDArray.sub(x_diff, x_perp);

    const centre = NDArray.add(
      x_m, NDArray.mul(
        NDArray.cross(k, x_par),
        (0.5 / Math.tan(theta / 2.0))
      )
    );
    const v = x_perp;

    const omega_vec = NDArray.mul(k, theta);
    const omega = mat3([
      0, -omega_vec.z, omega_vec.y,
      omega_vec.z, 0, -omega_vec.x,
      -omega_vec.y, omega_vec.x, 0,
    ]);

    return new Generator(
      NDArray.sub(v, NDArray.cross(omega_vec, centre)),
      omega
    );
  }
}
setInput("x1x", -1);
setInput("x1y", 0);
setInput("x1z", 0);

setInput("n1x", 0);
setInput("n1y", 1);
setInput("n1z", 0);

setInput("x2x", 1);
setInput("x2y", 0);
setInput("x2z", 0);

setInput("n2x", 0);
setInput("n2y", 0);
setInput("n2z", 1);

setInput("phi", 0.5);

let rafId = null;
function runSimulation() {
  const x1x = getInputNumber("x1x", -1);
  const x1y = getInputNumber("x1y", 0);
  const x1z = getInputNumber("x1z", 0);

  const n1x = getInputNumber("n1x", 0);
  const n1y = getInputNumber("n1y", 1);
  const n1z = getInputNumber("n1z", 0);

  const x2x = getInputNumber("x2x", 1);
  const x2y = getInputNumber("x2y", 0);
  const x2z = getInputNumber("x2z", 0);

  const n2x = getInputNumber("n2x", 0);
  const n2y = getInputNumber("n2y", 0);
  const n2z = getInputNumber("n2z", 1);

  const phioverpi = getInputNumber("phi", 0.5);
  const phi = phioverpi * Math.PI;

  const p1 = new PositionOrientation(vec3(x1x, x1y, x1z), NDArray.normalize(vec3(n1x, n1y, n1z)));
  const p2 = new PositionOrientation(vec3(x2x, x2y, x2z), NDArray.normalize(vec3(n2x, n2y, n2z)));

  function render() {
    ctx.fillStyle = white;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const perspective = NDArray.matmul(
      camera.clipFromCam(1, -1, 1),
      camera.camFromWorld()
    );

    drawArrow(origin, NDArray.mul(nx, 0.5), perspective, axisColour);
    drawArrow(origin, NDArray.mul(ny, 0.5), perspective, axisColour);
    drawArrow(origin, NDArray.mul(nz, 0.5), perspective, axisColour);

    let generator = PositionOrientation.getGenerator(p1, p2, "mav");
    for (let i = 1; i < 10; i++) {
      const t = i / 10;
      const inbetween = NDArray.expm(NDArray.mul(generator.mat, t), 4);
      const p = PositionOrientation.fromMat(NDArray.matmul(inbetween, p1.mat));
      p.draw(perspective, brickRed, 0.85);
    }
    generator = PositionOrientation.getGenerator(p1, p2, "pure_rotation");
    for (let i = 1; i < 10; i++) {
      const t = i / 10;
      const inbetween = NDArray.expm(NDArray.mul(generator.mat, t), 4);
      const p = PositionOrientation.fromMat(NDArray.matmul(inbetween, p1.mat));
      p.draw(perspective, forestGreen, 0.85);
    }
    generator = PositionOrientation.getGenerator(p1, p2, phi);
    for (let i = 1; i < 10; i++) {
      const t = i / 10;
      const inbetween = NDArray.expm(NDArray.mul(generator.mat, t), 4);
      const p = PositionOrientation.fromMat(NDArray.matmul(inbetween, p1.mat));
      p.draw(perspective, royalBlue, 0.85);
    }

    p1.draw(perspective, black, 0.85);
    p2.draw(perspective, black, 0.85);
  }

  render();
  let time = performance.now();
  function loop() {
    const newtime = performance.now();
    const dt = newtime - time;
    time = newtime;
    const changed = camera.update(inputState, dt * 0.001);
    if (changed) {
      render();
    }
    inputState.flush();
    rafId = requestAnimationFrame(loop);
  }

  loop();
}

runSimulation();

function startSimulation() {
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  runSimulation();
}
submit.addEventListener("click", startSimulation);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    startSimulation();
  }
});

function startRandomSimulation() {
  setInput("x1x", (2. * (Math.random() - 0.5)).toFixed(2));
  setInput("x1y", (2. * (Math.random() - 0.5).toFixed(2)));
  setInput("x1z", (2. * (Math.random() - 0.5).toFixed(2)));

  setInput("n1x", (2. * (Math.random() - 0.5).toFixed(2)));
  setInput("n1y", (2. * (Math.random() - 0.5).toFixed(2)));
  setInput("n1z", (2. * (Math.random() - 0.5).toFixed(2)));

  setInput("x2x", (2. * (Math.random() - 0.5).toFixed(2)));
  setInput("x2y", (2. * (Math.random() - 0.5).toFixed(2)));
  setInput("x2z", (2. * (Math.random() - 0.5).toFixed(2)));

  setInput("n2x", (2. * (Math.random() - 0.5).toFixed(2)));
  setInput("n2y", (2. * (Math.random() - 0.5).toFixed(2)));
  setInput("n2z", (2. * (Math.random() - 0.5).toFixed(2)));

  setInput("phi", (2. * (Math.random() - 0.5).toFixed(2)));

  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  runSimulation();
}
randomise.addEventListener("click", startRandomSimulation);