// eikonal_pde.js

import { arrayToImage } from "./imshow.js";

const N = 101

const xs = Array.from({length: N}, (x, i) => i);
const ys = Array.from({length: N}, (x, i) => i);

const cost = Array.from({ length: N }, () => new Array(N));
for (const [i, c] of cost.entries()) {
    for (const [j, d] of c.entries()) {
        let loc = Math.sqrt(
                Math.pow(2 * (xs[i] - (N-1.)/2.) / N, 2.) + 
                Math.pow(2 * (ys[j] - (N-1.)/2.) / N, 2.)
        )
        cost[i][j] = 1. / (1. + loc)
    }
}
console.log(cost)

const source = [50, 50]

function eikonalStep(W, cost, epsilon, dW_dt) {
    for (const [i, a] of dW_dt.entries()) {
        for (const [j, b] of a.entries()) {
            let dW_dx_forward  = W[i+2][j+1] - W[i+1][j+1]
            let dW_dx_backward = W[i+1][j+1] - W[i][j+1]
            let dW_dy_forward  = W[i+1][j+2] - W[i+1][j+1]
            let dW_dy_backward = W[i+1][j+1] - W[i+1][j]

            let dW_dx = Math.max(-dW_dx_forward, dW_dx_backward, 0)
            let dW_dy = Math.max(-dW_dy_forward, dW_dy_backward, 0)
            
            dW_dt[i][j] = (
                1. - Math.sqrt(
                    Math.pow(dW_dx, 2) + Math.pow(dW_dy, 2)
                ) / cost[i][j]
            ) * epsilon
        }
    }
    for (const [i, a] of dW_dt.entries()) {
        for (const [j, b] of a.entries()) {
            W[i+1][j+1] += dW_dt[i][j]
        }
    }
}

function eikonalSolver(cost, source, nMax) {
    // Heuristic, so that W does not become negative.
    const epsilon = 1. / Math.sqrt(2.) * Math.min(...cost.flat())

    const shiftedSource = [source[0]+1, source[1]+1]
    const W = Array.from({ length: N+2 }, () => new Array(N+2).fill(10.));
    const dW_dt = Array.from({ length: N }, () => new Array(N).fill(10.));
    W[shiftedSource[0]][shiftedSource[1]] = 0.

    for (let i = 0; i < nMax; i++) {
        eikonalStep(W, cost, epsilon, dW_dt)
        // Reapply boundary condition.
        W[shiftedSource[0]][shiftedSource[1]] = 0.
    }

    return W.slice(1, -1).map(a => a.slice(1, -1))
}

let W = eikonalSolver(cost, source, 1000)
console.log(W)

const canvasCost = document.createElement("canvas");
const contextCost = canvasCost.getContext("2d");
const imageCost = arrayToImage(contextCost, cost);
imageCost.classList.add("scaled-image");
const containerCost = document.getElementById("distancemap")
containerCost.appendChild(imageCost);

const canvasW = document.createElement("canvas");
const contextW = canvasW.getContext("2d");
const imageW = arrayToImage(contextW, W);
imageW.classList.add("scaled-image");
const containerW = document.getElementById("distancemap")
containerW.appendChild(imageW);