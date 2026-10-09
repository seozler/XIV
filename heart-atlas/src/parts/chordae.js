// Chordae tendineae: branching fibrous cords strung between papillary-muscle heads and
// leaflet attachment points (primary → free edge, secondary → rough zone).
//
// Endpoints are re-evaluated every frame from the live leaflet geometry, so the cords
// follow valve motion. Explode offsets of the valve and papillary-muscle parts are
// baked into the vertex positions and also written to `aShift`, which the deformation
// shader subtracts before (and re-adds after) applying the cardiac field — chordae stay
// attached to both ends, stretching like strings when the parts are pulled apart.

import * as THREE from 'three';
import { add, sub, mul, norm, lerp, cross } from '../core/vec.js';
import { ensureHeartAttributes } from '../core/geometry.js';

const RAD = 5;

export class Chordae {
  /**
   * @param valve LeafletValve
   * @param pms   [{ phi, heads:[pt] }]
   */
  constructor(valve, pms, { primaries = 6, secondaries = 3, radius = 0.032 } = {}) {
    this.valve = valve;
    this.pms = pms;
    this.radius = radius;
    // attachment points
    const att = [];
    valve.cfg.leaflets.forEach((L, li) => {
      for (let i = 0; i < primaries; i++) att.push({ li, s: (i + 0.5) / primaries, t: 1, primary: true });
      for (let i = 0; i < secondaries; i++) att.push({ li, s: (i + 0.5) / secondaries, t: 0.68, primary: false });
    });
    // assign each attachment to the papillary muscle nearest in angle
    const angDist = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
    for (const a of att) {
      const L = valve.cfg.leaflets[a.li];
      a.phi = L.a0 + a.s * (L.a1 - L.a0);
      let best = 0, bd = Infinity;
      pms.forEach((pm, i) => { const d = angDist(a.phi, pm.phi); if (d < bd) { bd = d; best = i; } });
      a.pm = best;
    }
    // group neighbours into trunks (2–3 branches per trunk)
    this.trunks = [];
    pms.forEach((pm, pi) => {
      const mine = att.filter((a) => a.pm === pi).sort((a, b) => a.li - b.li || a.phi - b.phi);
      let h = 0;
      for (let i = 0; i < mine.length; ) {
        const n = mine.length - i === 4 ? 2 : Math.min(3, mine.length - i);
        this.trunks.push({ pm: pi, head: h++ % pm.heads.length, atts: mine.slice(i, i + n) });
        i += n;
      }
    });
    this.segCount = this.trunks.reduce((s, t) => s + 1 + t.atts.length, 0);
  }

  buildGeometry() {
    const nv = this.segCount * RAD * 2;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nv * 3), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(nv * 3), 3));
    g.setAttribute('aShift', new THREE.BufferAttribute(new Float32Array(nv * 3), 3));
    const idx = [];
    for (let s = 0; s < this.segCount; s++) {
      const o = s * RAD * 2;
      for (let j = 0; j < RAD; j++) {
        const j2 = (j + 1) % RAD;
        idx.push(o + j, o + j2, o + RAD + j, o + j2, o + RAD + j2, o + RAD + j);
      }
    }
    g.setIndex(idx);
    ensureHeartAttributes(g);
    this.geometry = g;
    this.update([0, 0, 0], [0, 0, 0]);
    return g;
  }

  /** valveOffset / pmOffset: current explode offsets of the valve and papillary parts. */
  update(valveOffset, pmOffset) {
    const pos = this.geometry.attributes.position.array;
    const nrm = this.geometry.attributes.normal.array;
    const shf = this.geometry.attributes.aShift.array;
    let seg = 0;
    const write = (a, b, ra, rb, sa, sb) => {
      const ax = norm(sub(b, a));
      const ref = Math.abs(ax[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      const u = norm(cross(ax, ref));
      const v = cross(ax, u);
      const o = seg * RAD * 2 * 3;
      for (let e = 0; e < 2; e++) {
        const c = e ? b : a, r = e ? rb : ra, sh = e ? sb : sa;
        for (let j = 0; j < RAD; j++) {
          const ang = (j / RAD) * Math.PI * 2;
          const d = add(mul(u, Math.cos(ang)), mul(v, Math.sin(ang)));
          const k = o + (e * RAD + j) * 3;
          pos[k] = c[0] + d[0] * r; pos[k + 1] = c[1] + d[1] * r; pos[k + 2] = c[2] + d[2] * r;
          nrm[k] = d[0]; nrm[k + 1] = d[1]; nrm[k + 2] = d[2];
          shf[k] = sh[0]; shf[k + 1] = sh[1]; shf[k + 2] = sh[2];
        }
      }
      seg++;
    };
    for (const tr of this.trunks) {
      const head = add(this.pms[tr.pm].heads[tr.head], pmOffset);
      const ends = tr.atts.map((a) => add(this.valve.point(a.li, a.s, a.t), valveOffset));
      const c = ends.reduce((s, p) => add(s, mul(p, 1 / ends.length)), [0, 0, 0]);
      const split = lerp(head, c, 0.62);
      const splitShift = lerp(pmOffset, valveOffset, 0.62);
      const r = this.radius * (tr.atts[0].primary ? 1 : 1.25);
      write(head, split, r * 1.5, r * 1.25, pmOffset, splitShift);
      for (const e of ends) write(split, e, r, r * 0.8, splitShift, valveOffset);
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.normal.needsUpdate = true;
    this.geometry.attributes.aShift.needsUpdate = true;
    this.geometry.computeBoundingSphere();
  }
}
