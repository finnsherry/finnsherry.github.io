import { NDArray, vec3, mat3, mat4, rotateTogether } from "/utils/linalg.js";

export class Camera {
    
    constructor(position, target = vec3(0, 0, 0)) {
        this.position = position;
        this.target = target;
        this.setFrame();

        this.zoom = 1.0;
        this.near = 0.01;
        this.far = 1000;

        this.perspective = true;
        
        this.rollSpeed = 0.5; // radians per second
        this.lookSensitivity = 0.005; // radians per pixel
        this.wheelSensitivity = 0.5;

        this.looked = false;

        this.controlsExplanation = [
            "Look: Left mouse button / one-finger drag.",
            "Speed: Mouse wheel.",
            "Zoom: PageUp / PageDown / Home.",
            "Perspective/orthographic: Backquote (`).",
        ];
    }

    getInfo() {
        return [
            `Position: (${this.position.x.toPrecision(2)}, ${this.position.y.toPrecision(2)}, ${this.position.z.toPrecision(2)})`,
            `Zoom: ${this.zoom.toPrecision(3)}`,
            `Projection: ${this.perspective ? "perspective" : "orthographic"}`,
        ]
    }

    setFrame() {
        this.forward = NDArray.normalize(NDArray.sub(this.target, this.position));
        const right = NDArray.cross(this.forward, vec3(0, 1, 0));
        if (NDArray.norm(right) > 0.001) {
            this.right = NDArray.normalize(right);
        } else {
            this.right = NDArray.normalize(NDArray.cross(this.forward, vec3(1, 0, 0)));
        }
        this.upward = NDArray.cross(this.right, this.forward);
    }

    turnRight(a) {
        const cos = Math.cos(a);
        const sin = Math.sin(a);
        this.position = NDArray.add(
            this.target,
            NDArray.matmul(
                mat3([
                    cos, 0, -sin,
                    0,   1, 0,
                    sin, 0, cos,
                ]),
                NDArray.sub(this.target, this.position)
            )
        );
        this.setFrame();
        this.changed = true;
    }

    tiltUp(a) {
        const cos = Math.cos(a);
        const sin = Math.sin(a);
        this.position = NDArray.add(
            this.target,
            NDArray.matmul(
                mat3([
                    1, 0, 0,
                    0, cos, -sin,
                    0, sin, cos,
                ]),
                NDArray.sub(this.target, this.position)
            )
        );
        this.setFrame();
        this.changed = true;
    }

    camFromWorld() {
        const p = this.position;
        const f = this.forward;
        let r = this.right;
        const u = this.upward;
        const worldFromCam = mat4([
            r.x, u.x, f.x,  p.x,
            r.y, u.y, f.y,  p.y,
            r.z, u.z, f.z,  p.z,
            0,  0,   0,  1,
        ]);
        const camFromWorld = NDArray.inv4(worldFromCam);
        return camFromWorld;
    }

    clipFromCam(aspect, ndcZMin = 0, ndcZMax = 1) {
        if(this.perspective)
            return NDArray.perspective(this.zoom, aspect, this.near, this.far, ndcZMin, ndcZMax);
        else
            return NDArray.orthographic(this.zoom, aspect, this.near, this.far, ndcZMin, ndcZMax);
    }

    update(inputState, dt) // dt in seconds
    {
        // shorthand because we will be using this a lot in this function
        const S = inputState;

        const zoomStep = Math.exp(2.0 * dt);
        if(S.keyCodeDown["PageUp"])   {this.zoom /= zoomStep; this.changed = true;}
        if(S.keyCodeDown["PageDown"]) {this.zoom *= zoomStep; this.changed = true;}
        if(S.keyCodeDown["Home"])     {this.zoom = 1.0; this.changed = true;}
        
        if(S.keyCodeWentDown.has("Backquote")) {this.perspective = !this.perspective; this.changed = true;}

        if (S.pointers.size === 1) {
            const id = Array.from(S.pointers.keys())[0];
            const pointer = S.pointers.get(id);

            const leftClickDragging = pointer.pointerType == "mouse" && pointer.buttons & 1
            const singleTouchDragging = pointer.pointerType == "touch"
            
            if(leftClickDragging || singleTouchDragging) {
                const dx = (S.pointersMovementX.get(id) || 0) * this.zoom * this.lookSensitivity;
                const dy = (S.pointersMovementY.get(id) || 0) * this.zoom * this.lookSensitivity;
                this.turnRight(dx);
                this.tiltUp(-dy);
                this.changed = true;
            }
        }

        const copy = this.changed;

        this.changed = false;
        
        return copy;
    }

}

// 

export class InputState {
    constructor() {
        this.pointers = new Map();
        this.pointersMovementX = new Map();
        this.pointersMovementY = new Map();
        this.toBeFlushed = new Set();

        this.pointerWentDown = new Set();
        this.pointerWentUp = new Set();

        this.wheelTicks = { x: 0, y: 0 };

        this.keyCodeDown = {};
        this.keyCodeWentDown = new Set();
        this.keyCodeWentUp = new Set();
    }

    // Attach event listeners to an element to track input state
    // this will prevent default behavior for most events, so it should be used on a dedicated input element (e.g. a canvas)
    attach(element) {
        element.tabIndex = 0; // Make the element focusable

        element.style.touchAction = "none"; // disable default touch actions


        element.addEventListener("dragstart", e => {
            e.preventDefault();
        });

        element.addEventListener("contextmenu", e => {
            e.preventDefault()
        });

        element.addEventListener("pointerdown", e => {
            // console.log("pointerdown", e);
            element.setPointerCapture(e.pointerId);
            this.pointers.set(e.pointerId, e);
            this.pointersMovementX.set(e.pointerId, 0);
            this.pointersMovementY.set(e.pointerId, 0);
            this.pointerWentDown.add(e.pointerId);
        });

        element.addEventListener("pointermove", e => {
            // console.log("pointermove", e);
            // Update the pointer event with the latest position, but accumulate properties such as movementX/Y
            // unfortunately, e.movementX/Y can not be set (read-only)
            // also, we can not rely on the browser to provide movementX/Y, because it can be inconsistent?
            // so we will calculate movementX/Y ourselves by comparing the current event with the previous event for the same pointerId

            const prev = this.pointers.get(e.pointerId);

            const movementX = (prev ? e.clientX - prev.clientX : 0) + this.pointersMovementX.get(e.pointerId) || 0;
            const movementY = (prev ? e.clientY - prev.clientY : 0) + this.pointersMovementY.get(e.pointerId) || 0;

            this.pointers.set(e.pointerId, e);
            this.pointersMovementX.set(e.pointerId, movementX );
            this.pointersMovementY.set(e.pointerId, movementY );
        });

        element.addEventListener("pointerup", e => {
            // console.log("pointerup", e);
            this.pointerWentUp.add(e.pointerId);
            this.toBeFlushed.add(e.pointerId);
        });

        element.addEventListener("pointercancel", e => {
            // console.log("pointercancel", e);
            this.toBeFlushed.add(e.pointerId);
        });

        element.addEventListener("wheel", e => {
            e.preventDefault();
            this.wheelTicks.x += Math.sign(e.deltaX);
            this.wheelTicks.y += Math.sign(e.deltaY);
        });

        element.addEventListener("keydown", e => {
            // console.log("keydown", e);
            e.preventDefault();
            this.keyCodeDown[e.code] = true;
            this.keyCodeWentDown.add(e.code);
        });

        element.addEventListener("keyup", e => {
            // console.log("keyup", e);
            this.keyCodeDown[e.code] = false;
            this.keyCodeWentUp.add(e.code);
        });
    }

    flush() {
        // clear accumulated movementX/Y for all pointers
        this.pointersMovementX.forEach((_, key) => this.pointersMovementX.set(key, 0));
        this.pointersMovementY.forEach((_, key) => this.pointersMovementY.set(key, 0));
        // clear wheel ticks
        this.wheelTicks = { x: 0, y: 0 };
        // clear keyCodeWentDown and Up
        this.keyCodeWentDown.clear();
        this.keyCodeWentUp.clear();
        // clear pointerWentDown and Up
        this.pointerWentDown.clear();
        this.pointerWentUp.clear();
        // remove pointers that went up or were canceled
        this.toBeFlushed.forEach(pointerId => {
            this.pointers.delete(pointerId);
            this.pointersMovementX.delete(pointerId);
            this.pointersMovementY.delete(pointerId);
        });
        this.toBeFlushed.clear();
    }
}
