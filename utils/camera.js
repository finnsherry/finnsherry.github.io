import { NDArray, vec3, mat3, mat4, rotateTogether } from "/utils/linalg.js";

export class Camera {
    // Adapted from https://bellaard.com/js/3dcamera.js
    // This camera only rotates around a fixed target.
    constructor(position, target = vec3(0, 0, 0)) {
        this.position = position;
        this.target = target;
        this.setFrame();

        this.zoom = 1.0;
        this.near = 0.01;
        this.far = 1000;

        this.perspective = true;
        
        this.rollSpeed = 0.5; // radians per second
        this.lookSensitivity = 0.03; // radians per pixel
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

        this.lookSensitivity *= Math.pow(2, - S.wheelTicks.y * this.wheelSensitivity);

        const copy = this.changed;

        this.changed = false;
        
        return copy;
    }

}