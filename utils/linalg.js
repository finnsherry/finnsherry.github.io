export class NDArray {
    constructor(data, shape) {
        this.data = data instanceof Float32Array ? data : new Float32Array(data);
        this.shape = shape.slice();
        this._validate();
    }

    _validate() {
        const size = this.shape.reduce((a, b) => a * b, 1);
        if (size !== this.data.length) throw new Error("NDArray: shape does not match data length");
    }

    _sameShape(other) {
        return this.shape.length === other.shape.length && this.shape.every((v, i) => v === other.shape[i]);
    }

    _assertSameShape(other, op) {
        if (!this._sameShape(other)) throw new Error("NDArray." + op + ": shape mismatch");
    }

    _assertVector(op) {
        if (this.shape.length !== 1) throw new Error("NDArray." + op + ": expected vector shape [n]");
    }

    _assertMatrix(op) {
        if (this.shape.length !== 2) throw new Error("NDArray." + op + ": expected matrix shape [r,c]");
    }

    clone() { return new NDArray(this.data.slice(), this.shape); }

    // Convenience accessors for vectors.
    get x() { return this.data[0]; }
    set x(v) { this.data[0] = v; }
    get y() { return this.data[1]; }
    set y(v) { this.data[1] = v; }
    get z() { return this.data[2]; }
    set z(v) { this.data[2] = v; }
    get w() { return this.data[3]; }
    set w(v) { this.data[3] = v; }

    static array(data, shape) { return new NDArray(data, shape); }
    static zeros(shape) { return new NDArray(new Float32Array(shape.reduce((a, b) => a * b, 1)), shape); }
    static ones(shape) { return new NDArray(new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(1), shape); }
    static eye(n) {
        const d = new Float32Array(n * n);
        for (let i = 0; i < n; i++) d[i * n + i] = 1;
        return new NDArray(d, [n, n]);
    }

    static add(a, b) {
        if (!a._sameShape(b)) throw new Error("NDArray.add: shape mismatch");
        return new NDArray(a.data.map((e, i) => e + b.data[i]), a.shape);
    }

    static sub(a, b) {
        if (!a._sameShape(b)) throw new Error("NDArray.sub: shape mismatch");
        return new NDArray(a.data.map((e, i) => e - b.data[i]), a.shape);
    }

    static mul(a, s) { return new NDArray(a.data.map(e => e * s), a.shape); }
    static div(a, s) { return new NDArray(a.data.map(e => e / s), a.shape); }

    static dot(a, b) {
        a._assertVector("dot");
        b._assertVector("dot");
        if (!a._sameShape(b)) throw new Error("NDArray.dot: shape mismatch");
        let out = 0;
        for (let i = 0; i < a.data.length; i++) out += a.data[i] * b.data[i];
        return out;
    }

    static norm(v) {
        v._assertVector("norm");
        return Math.sqrt(NDArray.dot(v, v));
    }

    static normalize(v) {
        v._assertVector("normalize");
        return NDArray.div(v, NDArray.norm(v));
    }

    static dist(a, b) {
        a._assertVector("dist");
        b._assertVector("dist");
        return NDArray.norm(NDArray.sub(a, b));
    }

    static cross(a, b) {
        a._assertVector("cross");
        b._assertVector("cross");
        if (a.data.length !== 3 || b.data.length !== 3) throw new Error("NDArray.cross: expected shape [3]");
        return new NDArray([
            a.data[1] * b.data[2] - a.data[2] * b.data[1],
            a.data[2] * b.data[0] - a.data[0] * b.data[2],
            a.data[0] * b.data[1] - a.data[1] * b.data[0],
        ], [3]);
    }

    static transpose(A) {
        A._assertMatrix("transpose");
        const r = A.shape[0], c = A.shape[1];
        const out = new Float32Array(r * c);
        for (let i = 0; i < r; i++)
            for (let j = 0; j < c; j++)
                out[j * r + i] = A.data[i * c + j];
        return new NDArray(out, [c, r]);
    }

    static trace(A) {
        A._assertMatrix("trace");
        const r = A.shape[0], c = A.shape[1];
        if (r !== c) throw new Error("NDArray.trace: expected square matrix");
        let s = 0;
        for (let i = 0; i < r; i++) s += A.data[i * c + i];
        return s;
    }

    static outer(u, v) {
        u._assertVector("outer");
        v._assertVector("outer");
        const n = u.data.length, m = v.data.length;
        const out = new Float32Array(n * m);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < m; j++)
                out[i * m + j] = u.data[i] * v.data[j];
        return new NDArray(out, [n, m]);
    }

    static matmul(A, B) {
        A._assertMatrix("matmul");

        if (B.shape.length === 1) {
            const r = A.shape[0], k = A.shape[1];
            if (B.shape[0] !== k) throw new Error("NDArray.matmul: shape mismatch for mat-vec");
            const out = new Float32Array(r);
            for (let i = 0; i < r; i++) {
                let s = 0;
                for (let j = 0; j < k; j++) s += A.data[i * k + j] * B.data[j];
                out[i] = s;
            }
            return new NDArray(out, [r]);
        }

        B._assertMatrix("matmul");
        const r = A.shape[0], k = A.shape[1], c = B.shape[1];
        if (B.shape[0] !== k) throw new Error("NDArray.matmul: shape mismatch for mat-mat");

        const out = new Float32Array(r * c);
        for (let i = 0; i < r; i++) {
            for (let j = 0; j < c; j++) {
                let s = 0;
                for (let t = 0; t < k; t++) s += A.data[i * k + t] * B.data[t * c + j];
                out[i * c + j] = s;
            }
        }
        return new NDArray(out, [r, c]);
    }

    static expm(A, n) {
        A._assertMatrix("expm");
        const r = A.shape[0], c = A.shape[1];
        if (r !== n || c !== n) throw new Error("NDArray.expm: expected shape [n,n]");
        if (!(n == 3 || n == 4)) throw new Error("NDArray.expm: only works for n = 3, 4");

        let norm = 0; 
        for (let i = 0; i < n; i++) {
            let s = 0;
            for (let j = 0; j < n; j++) {
                s += Math.abs(A.data[n * i + j]);
            }
            norm = Math.max(norm, s);
        }

        let s = 0;
        if (norm > 1) {
            s = Math.max(0, Math.ceil(Math.log2(norm)));
        }

        const As = NDArray.mul(A, 1 / Math.pow(2, s));

        const I = NDArray.eye(n);
        const A2 = NDArray.matmul(As, As);
        const A3 = NDArray.matmul(A2, As);
        const A4 = NDArray.matmul(A2, A2);
        const A5 = NDArray.matmul(A3, A2);
        // const A6 = NDArray.matmul(A3, A3);

        const N = NDArray.add(
            NDArray.add(I, NDArray.mul(As, 1/2)),
            NDArray.add(
                NDArray.mul(A2, 1/9),
                NDArray.add(
                    NDArray.mul(A3, 1/72),
                    NDArray.add(
                        NDArray.mul(A4, 1/1008),
                        NDArray.mul(A5, 1/30240)
                    )
                ),
            ),
        );
        const D = NDArray.add(
            NDArray.add(I, NDArray.mul(As, -1/2)),
            NDArray.add(
                NDArray.mul(A2, 1/9),
                NDArray.add(
                    NDArray.mul(A3, -1/72),
                    NDArray.add(
                        NDArray.mul(A4, 1/1008),
                        NDArray.mul(A5, -1/30240)
                    )
                ),
            ),
        );

        let invD
        if (n == 4) {
            invD = NDArray.inv4(D);
        } else {
            invD = NDArray.inv3(D);
        }

        let R = NDArray.matmul(N, invD);

        for (let i = 0; i < s; i++) {
            R = NDArray.matmul(R, R);
        }

        return R
    }

    static det3(A) {
        if (A.shape.length !== 2 || A.shape[0] !== 3 || A.shape[1] !== 3) throw new Error("NDArray.det3: expected shape [3,3]");
        const m = A.data;
        return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    }

    static adj3(A) {
        if (A.shape.length !== 2 || A.shape[0] !== 3 || A.shape[1] !== 3) throw new Error("NDArray.adj3: expected shape [3,3]");
        const m = A.data;
        return new NDArray([
             m[4] * m[8] - m[7] * m[5], -m[1] * m[8] + m[7] * m[2],  m[1] * m[5] - m[4] * m[2],
            -m[3] * m[8] + m[6] * m[5],  m[0] * m[8] - m[6] * m[2], -m[0] * m[5] + m[3] * m[2],
             m[3] * m[7] - m[6] * m[4], -m[0] * m[7] + m[6] * m[1],  m[0] * m[4] - m[3] * m[1],
        ], [3, 3]);
    }

    static inv3(A) { return NDArray.mul(NDArray.adj3(A), 1 / NDArray.det3(A)); }

    static adj4(A) {
        if (A.shape.length !== 2 || A.shape[0] !== 4 || A.shape[1] !== 4) throw new Error("NDArray.adj4: expected shape [4,4]");

        const m = A.data;
        const m00 = m[0],  m01 = m[1],  m02 = m[2],  m03 = m[3];
        const m10 = m[4],  m11 = m[5],  m12 = m[6],  m13 = m[7];
        const m20 = m[8],  m21 = m[9],  m22 = m[10], m23 = m[11];
        const m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];

        const s0 = m00 * m11 - m10 * m01;
        const s1 = m00 * m12 - m10 * m02;
        const s2 = m00 * m13 - m10 * m03;
        const s3 = m01 * m12 - m11 * m02;
        const s4 = m01 * m13 - m11 * m03;
        const s5 = m02 * m13 - m12 * m03;

        const c5 = m22 * m33 - m32 * m23;
        const c4 = m21 * m33 - m31 * m23;
        const c3 = m21 * m32 - m31 * m22;
        const c2 = m20 * m33 - m30 * m23;
        const c1 = m20 * m32 - m30 * m22;
        const c0 = m20 * m31 - m30 * m21;

        return new NDArray([
            ( m11 * c5 - m12 * c4 + m13 * c3),
            (-m01 * c5 + m02 * c4 - m03 * c3),
            ( m31 * s5 - m32 * s4 + m33 * s3),
            (-m21 * s5 + m22 * s4 - m23 * s3),

            (-m10 * c5 + m12 * c2 - m13 * c1),
            ( m00 * c5 - m02 * c2 + m03 * c1),
            (-m30 * s5 + m32 * s2 - m33 * s1),
            ( m20 * s5 - m22 * s2 + m23 * s1),

            ( m10 * c4 - m11 * c2 + m13 * c0),
            (-m00 * c4 + m01 * c2 - m03 * c0),
            ( m30 * s4 - m31 * s2 + m33 * s0),
            (-m20 * s4 + m21 * s2 - m23 * s0),

            (-m10 * c3 + m11 * c1 - m12 * c0),
            ( m00 * c3 - m01 * c1 + m02 * c0),
            (-m30 * s3 + m31 * s1 - m32 * s0),
            ( m20 * s3 - m21 * s1 + m22 * s0),
        ], [4, 4]);
    }

    static inv4(A) {
        if (A.shape.length !== 2 || A.shape[0] !== 4 || A.shape[1] !== 4) throw new Error("NDArray.inv4: expected shape [4,4]");

        const adj = NDArray.adj4(A);
        const m = A.data;
        const a = adj.data;
        const det = m[0] * a[0] + m[1] * a[4] + m[2] * a[8] + m[3] * a[12];

        return NDArray.mul(adj, 1 / det);
    }

    static rotAxis(axis, angle) {
        if (axis.shape.length !== 1 || axis.shape[0] !== 3) throw new Error("NDArray.rotAxis: expected axis shape [3]");
        const c = Math.cos(angle), s = Math.sin(angle);
        return NDArray.eye(3)
            .mul(c)
            .add(NDArray.mul(NDArray.skew(axis), s))
            .add(NDArray.mul(NDArray.outer(axis, axis), 1 - c));
    }

    static skew(v) {
        if (v.shape.length !== 1 || v.shape[0] !== 3) throw new Error("NDArray.skew: expected shape [3]");
        const x = v.data[0], y = v.data[1], z = v.data[2];
        return new NDArray([0, -z, y, z, 0, -x, -y, x, 0], [3, 3]);
    }

    static rotAxes(a) {
        const cx = Math.cos(a[0]), sx = Math.sin(a[0]);
        const cy = Math.cos(a[1]), sy = Math.sin(a[1]);
        const cz = Math.cos(a[2]), sz = Math.sin(a[2]);

        const Rx = new NDArray([1, 0, 0, 0, cx, sx, 0, -sx, cx], [3, 3]);
        const Ry = new NDArray([cy, 0, -sy, 0, 1, 0, sy, 0, cy], [3, 3]);
        const Rz = new NDArray([cz, sz, 0, -sz, cz, 0, 0, 0, 1], [3, 3]);
        return NDArray.matmul(Rx, NDArray.matmul(Ry, Rz));
    }

    static translate(x, y, z) {
        return new NDArray([
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1,
        ], [4, 4]);
    }

    static scale(sx, sy, sz) {
        return new NDArray([
            sx, 0, 0, 0,
            0, sy, 0, 0,
            0, 0, sz, 0,
            0, 0, 0, 1,
        ], [4, 4]);
    }

    static fromMat3(M) {
        if (M.shape.length !== 2 || M.shape[0] !== 3 || M.shape[1] !== 3) throw new Error("NDArray.fromMat3: expected shape [3,3]");
        const m = M.data;
        return new NDArray([
            m[0], m[1], m[2], 0,
            m[3], m[4], m[5], 0,
            m[6], m[7], m[8], 0,
            0, 0, 0, 1,
        ], [4, 4]);
    }

    // perspective projection matrix.
    // puts z=near to z=ndc_z_min and z=far to z=ndc_z_max after the perspective divide.
    static perspective(zoom, aspect, near, far, ndc_z_min, ndc_z_max) {
        const f = 1.0 / zoom;
        const A = (ndc_z_max * far - ndc_z_min * near) / (far - near);
        const B = near * (ndc_z_min - A);
        
        return new NDArray([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, A, B,
            0, 0, 1, 0,
        ], [4, 4]);
    }

    // orthographic projection matrix.
    // puts z=near to z=ndc_z_min and z=far to z=ndc_z_max after the perspective divide.
    static orthographic(zoom, aspect, near, far, ndc_z_min, ndc_z_max) {
        const f = 1.0 / zoom;
        const A = (ndc_z_max - ndc_z_min) / (far - near);
        const B = ndc_z_min - near * A;
        
        return new NDArray([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, A, B,
            0, 0, 0, 1,
        ], [4, 4]);
    }

    add(b) { this._assertSameShape(b, "add"); for (let i = 0; i < this.data.length; i++) this.data[i] += b.data[i]; return this; }
    sub(b) { this._assertSameShape(b, "sub"); for (let i = 0; i < this.data.length; i++) this.data[i] -= b.data[i]; return this; }
    mul(s) { for (let i = 0; i < this.data.length; i++) this.data[i] *= s; return this; }
    div(s) { for (let i = 0; i < this.data.length; i++) this.data[i] /= s; return this; }

    matmul(B) {
        const out = NDArray.matmul(this, B);
        this.data = out.data;
        this.shape = out.shape;
        return this;
    }

    normalize() {
        this._assertVector("normalize");
        const n = NDArray.norm(this);
        if (n === 0) return this;
        return this.div(n);
    }
}


export const vec2 = (x, y) => new NDArray([x, y], [2]);
export const vec3 = (x, y, z) => new NDArray([x, y, z], [3]);
export const vec4 = (x, y, z, w) => new NDArray([x, y, z, w], [4]);
export const mat3 = (values) => new NDArray(values, [3, 3]);
export const mat4 = (values) => new NDArray(values, [4, 4]);

export function rotateTogether(u, v, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    const nu = NDArray.add(NDArray.mul(u, c), NDArray.mul(v, s));
    const nv = NDArray.add(NDArray.mul(u, -s), NDArray.mul(v, c));
    u.data.set(nu.data);
    v.data.set(nv.data);
}
