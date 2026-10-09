// Parametric, animated heart valves.
//
// Atrioventricular valves (mitral, tricuspid)
//   • Saddle-shaped, D-shaped (mitral) / oval (tricuspid) annulus.
//   • Each leaflet is a sheet hinged on the annulus. Closed: it runs to a shared
//     coaptation line (a posteriorly displaced "smile" for the mitral, a Y for the
//     tricuspid), billows slightly toward the atrium, and the remaining length forms the
//     coaptation zone hanging into the ventricle. Open: leaflets hang into the ventricle
//     as a funnel. Free edges are scalloped (P1–P3 on the posterior mitral leaflet).
// Semilunar valves (aortic, pulmonary)
//   • Three cusps hinged on a crown-shaped line (nadir → commissures). Closed: free edges
//     meet in a Y at the centre and the cusp bellies sag toward the ventricle. Open: free
//     edges move toward the sinus wall.
//
// update(opening, time) rewrites vertex positions every frame; geometry topology is fixed.

import * as THREE from 'three';
import { add, sub, mul, norm, lerp, comb, frame, dot, clamp } from '../core/vec.js';
import { ensureHeartAttributes } from '../core/geometry.js';

const ease = (x) => x * x * (3 - 2 * x);

export class LeafletValve {
  /**
   * cfg: { type:'av'|'semilunar', center, axis, ref, rx, ry, saddle, flatAnterior,
   *        leaflets:[{name,a0,a1,len(s)}], coapt:[cu,cv], depth, hNadir, hComm }
   */
  constructor(cfg, nS = 26, nT = 14) {
    this.cfg = cfg;
    const f = frame(cfg.axis, cfg.ref);
    if (cfg.anteriorDir && dot(f.v, cfg.anteriorDir) < 0) {
      f.u = mul(f.u, -1);
      f.v = mul(f.v, -1);
    }
    this.f = f;
    this.nS = nS;
    this.nT = nT;
    this.opening = 0;
    this.time = 0;
  }

  radial(phi) {
    const { u, v } = this.f;
    return norm(comb([u, Math.cos(phi)], [v, Math.sin(phi)]));
  }

  annulus(phi) {
    const { center, rx, ry, saddle = 0, flatAnterior = 1 } = this.cfg;
    const { u, v, w } = this.f;
    const s = Math.sin(phi);
    const ryy = s > 0 ? ry * flatAnterior : ry;
    return comb([center, 1], [u, rx * Math.cos(phi)], [v, ryy * s], [w, saddle * Math.cos(2 * phi)]);
  }

  /** Point on leaflet `li` at across-parameter s∈[0,1], hinge→free-edge t∈[0,1]. */
  point(li, s, t, o = this.opening) {
    return this.cfg.type === 'av' ? this.avPoint(li, s, t, o) : this.slPoint(li, s, t, o);
  }

  avPoint(li, s, t, o) {
    const { center, depth = 0.75, coapt = [0, 0] } = this.cfg;
    const { u, v, w } = this.f;
    const L = this.cfg.leaflets[li];
    const phi = L.a0 + s * (L.a1 - L.a0);
    const A = this.annulus(phi);
    const len = L.len(s);
    // coaptation "V": commissure → coaptation centre → commissure
    const K = (a) => comb([lerp(this.annulus(a), center, 0.16), 1], [w, depth * 0.55]);
    const Cc = comb([center, 1], [u, coapt[0]], [v, coapt[1]], [w, depth]);
    const Q = s < 0.5 ? lerp(K(L.a0), Cc, s * 2) : lerp(Cc, K(L.a1), (s - 0.5) * 2);
    const AQ = sub(Q, A);
    const dAQ = Math.hypot(...AQ);
    const t1 = clamp(dAQ / len, 0.2, 1);
    let Pc;
    const taper = Math.sin(Math.PI * s) ** 0.5;
    if (t <= t1) {
      const tt = t / t1;
      Pc = comb([A, 1], [AQ, tt], [w, -Math.sin(Math.PI * tt) * 0.2 * taper]);
    } else {
      const back = mul(norm(AQ), -0.025);
      Pc = comb([Q, 1], [w, (t - t1) * len], [back, 1]);
    }
    // open: funnel into the ventricle, slightly flared toward the walls
    const rd = this.radial(phi);
    const flutter = Math.sin(this.time * 0.021 + s * 6 + li) * 0.035 * t * o;
    const Po = comb([A, 1], [w, t * len * 0.96], [rd, t * len * (0.1 + flutter)]);
    return lerp(Pc, Po, ease(o));
  }

  slPoint(li, s, t, o) {
    const { center, rx: R, hNadir = -0.2, hComm = 1.3 } = this.cfg;
    const { w } = this.f;
    const L = this.cfg.leaflets[li];
    const sg = s * 2 - 1;
    const mid = (L.a0 + L.a1) / 2, half = (L.a1 - L.a0) / 2;
    const phi = mid + sg * half;
    const H = comb([center, 1], [this.radial(phi), R], [w, hNadir + (hComm - hNadir) * sg * sg]);
    const K0 = comb([center, 1], [this.radial(L.a0), R * 0.98], [w, hComm]);
    const K1 = comb([center, 1], [this.radial(L.a1), R * 0.98], [w, hComm]);
    const Cc = comb([center, 1], [w, hComm * 0.9]);
    const Fc = sg < 0 ? lerp(K0, Cc, sg + 1) : lerp(Cc, K1, sg);
    const flutter = Math.sin(this.time * 0.025 + li * 2.1) * 0.05 * o;
    const Fo = comb([center, 1], [this.radial(phi), R * (0.84 + flutter)], [w, hComm * 1.03]);
    const oo = ease(o);
    const F = lerp(Fc, Fo, oo);
    const bell = Math.sin(Math.PI * t) * (1 - sg * sg);
    return comb([lerp(H, F, t), 1], [w, -R * 0.42 * bell * (1 - oo)], [this.radial(phi), R * 0.07 * bell * oo]);
  }

  /** Build a THREE.BufferGeometry with all leaflets (+ fibrous annulus for AV valves). */
  buildGeometry() {
    const { nS, nT } = this;
    const nL = this.cfg.leaflets.length;
    const perLeaf = (nS + 1) * (nT + 1);
    const idx = [];
    for (let l = 0; l < nL; l++)
      for (let i = 0; i < nS; i++)
        for (let j = 0; j < nT; j++) {
          const a = l * perLeaf + i * (nT + 1) + j;
          const b = a + nT + 1;
          idx.push(a, b, a + 1, b, b + 1, a + 1);
        }
    this.leafVerts = nL * perLeaf;
    let ringPos = [], ringNrm = [];
    if (this.cfg.type === 'av') {
      // fibrous annulus: a thin torus following the saddle annulus (static)
      const nA = 96, nR = 8, r = 0.09;
      const base = this.leafVerts;
      for (let i = 0; i <= nA; i++) {
        const phi = (i / nA) * Math.PI * 2;
        const c = this.annulus(phi);
        const tng = norm(sub(this.annulus(phi + 0.01), c));
        const rd = norm(sub(c, this.cfg.center));
        const bn = norm(sub(rd, mul(tng, dot(rd, tng))));
        const b2 = [tng[1] * bn[2] - tng[2] * bn[1], tng[2] * bn[0] - tng[0] * bn[2], tng[0] * bn[1] - tng[1] * bn[0]];
        for (let j = 0; j < nR; j++) {
          const a = (j / nR) * Math.PI * 2;
          const d = comb([bn, Math.cos(a)], [b2, Math.sin(a)]);
          ringPos.push(...add(c, mul(d, r)));
          ringNrm.push(...d);
        }
      }
      for (let i = 0; i < nA; i++)
        for (let j = 0; j < nR; j++) {
          const a = base + i * nR + j, b = base + i * nR + ((j + 1) % nR);
          idx.push(a, a + nR, b, b, a + nR, b + nR);
        }
    }
    const total = this.leafVerts + ringPos.length / 3;
    const pos = new Float32Array(total * 3);
    const nrm = new Float32Array(total * 3);
    pos.set(ringPos, this.leafVerts * 3);
    nrm.set(ringNrm, this.leafVerts * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
    g.setIndex(idx);
    ensureHeartAttributes(g);
    this.geometry = g;
    this.leafIndexCount = nL * nS * nT * 6;
    this.update(0, 0, true);
    return g;
  }

  update(opening, time = 0, force = false) {
    if (!force && Math.abs(opening - this.opening) < 1e-4 && Math.abs(time - this.time) < 1 && this._built) return;
    this.opening = opening;
    this.time = time;
    const { nS, nT } = this;
    const pos = this.geometry.attributes.position.array;
    let k = 0;
    for (let l = 0; l < this.cfg.leaflets.length; l++)
      for (let i = 0; i <= nS; i++)
        for (let j = 0; j <= nT; j++) {
          const p = this.point(l, i / nS, j / nT, opening);
          pos[k++] = p[0]; pos[k++] = p[1]; pos[k++] = p[2];
        }
    this.computeLeafNormals();
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.normal.needsUpdate = true;
    this.geometry.computeBoundingSphere();
    this._built = true;
  }

  computeLeafNormals() {
    const pos = this.geometry.attributes.position.array;
    const nrm = this.geometry.attributes.normal.array;
    const idx = this.geometry.index.array;
    nrm.fill(0, 0, this.leafVerts * 3);
    for (let t = 0; t < this.leafIndexCount; t += 3) {
      const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
      const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
      const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      for (const q of [a, b, c]) { nrm[q] += nx; nrm[q + 1] += ny; nrm[q + 2] += nz; }
    }
    for (let q = 0; q < this.leafVerts * 3; q += 3) {
      const l = Math.hypot(nrm[q], nrm[q + 1], nrm[q + 2]) || 1;
      nrm[q] /= l; nrm[q + 1] /= l; nrm[q + 2] /= l;
    }
  }
}

const D = Math.PI / 180;

/** Create the four valves from landmarks. */
export function createValves(lm, fields, tricuspidLayout) {
  const m = lm.mitral;
  const mitral = new LeafletValve({
    type: 'av', center: m.center, axis: m.axis, ref: m.ref, anteriorDir: m.anteriorDir,
    rx: m.rx, ry: m.ry, saddle: 0.22, flatAnterior: 0.82, depth: 0.8, coapt: [0, -0.38 * m.ry],
    leaflets: [
      { name: 'anterior', a0: 0, a1: Math.PI, len: (s) => 2.3 * (0.38 + 0.62 * Math.sin(Math.PI * s) ** 0.7) },
      { name: 'posterior', a0: Math.PI, a1: 2 * Math.PI,
        len: (s) => 1.45 * (0.42 + 0.58 * Math.sin(Math.PI * s) ** 0.55) * (0.8 + 0.2 * Math.abs(Math.sin(3 * Math.PI * s))) },
    ],
  });
  const t = lm.tricuspid;
  const tricuspid = new LeafletValve({
    type: 'av', center: t.center, axis: t.axis, ref: t.ref, rx: t.rx, ry: t.ry, saddle: 0.12, depth: 0.8, coapt: [0, 0],
    leaflets: tricuspidLayout.leaflets.map((L) => ({
      ...L, len: (s) => L.len * (0.4 + 0.6 * Math.sin(Math.PI * s) ** 0.6) * (0.88 + 0.12 * Math.abs(Math.sin(2 * Math.PI * s))),
    })),
  });
  const ao = lm.aortic;
  const aortic = new LeafletValve({
    type: 'semilunar', center: ao.center, axis: ao.axis, ref: [0, 0, 1], rx: ao.radius, ry: ao.radius,
    hNadir: -0.25, hComm: 1.25,
    leaflets: [
      { name: 'right coronary', a0: -60 * D, a1: 60 * D },
      { name: 'left coronary', a0: 60 * D, a1: 180 * D },
      { name: 'non-coronary', a0: 180 * D, a1: 300 * D },
    ],
  });
  const pv = lm.pulmonary;
  const pulmonary = new LeafletValve({
    type: 'semilunar', center: pv.center, axis: pv.axis, ref: [0, 0, 1], rx: pv.radius, ry: pv.radius,
    hNadir: -0.2, hComm: 1.3,
    leaflets: [
      { name: 'anterior', a0: -60 * D, a1: 60 * D },
      { name: 'left', a0: 60 * D, a1: 180 * D },
      { name: 'right', a0: 180 * D, a1: 300 * D },
    ],
  });
  return { mitral, tricuspid, aortic, pulmonary };
}
