// Blood-flow particles along the two circuits through the heart:
//   systemic veins → RA → tricuspid → RV → infundibulum → pulmonary valve → PA branches
//   pulmonary veins → LA → mitral → LV → LVOT → aortic valve → aorta / arch branches
// Each path is split into flow-gated segments whose speed follows the live
// haemodynamics: venous return, AV-valve inflow (E and A waves), ventricular ejection.
// Particles therefore pool in the atria during systole, rush into the ventricles in
// early diastole and are expelled through the outflow tracts in systole.

import * as THREE from 'three';
import { add, mul, comb, lerp, rng } from '../core/vec.js';

const COLORS = { venous: new THREE.Color('#3f6dff'), arterial: new THREE.Color('#ff3b30') };

export class BloodFlow {
  constructor(lm, { count = 1800 } = {}) {
    this.lm = lm;
    this.count = count;
    this.paths = this.definePaths(lm);
    const R = rng(11);
    this.R = R;
    this.p = [];
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const size = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const path = this.pickPath();
      const q = { path, seg: Math.floor(R() * path.segs.length), s: R(), j: [R() - 0.5, R() - 0.5, R() - 0.5], v: 0.75 + R() * 0.5 };
      this.p.push(q);
      const c = COLORS[path.blood];
      col.set([c.r, c.g, c.b], i * 3);
      size[i] = 0.16 + R() * 0.1;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('size', new THREE.BufferAttribute(size, 1));
    const m = new THREE.ShaderMaterial({
      uniforms: { uScale: { value: 600 }, uOpacity: { value: 1 } },
      vertexShader: /* glsl */ `
        attribute float size; attribute vec3 color; varying vec3 vColor; uniform float uScale;
        void main() {
          vColor = color;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * uScale / -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        varying vec3 vColor; uniform float uOpacity;
        void main() {
          vec2 c = gl_PointCoord - 0.5; float d = dot(c, c);
          if (d > 0.25) discard;
          float shade = 0.75 + 0.5 * (0.25 - d) * 4.0 - c.y * 0.4;
          gl_FragColor = vec4(vColor * shade, uOpacity);
          #include <colorspace_fragment>
        }`,
      transparent: true,
      depthWrite: false,
    });
    this.object = new THREE.Points(g, m);
    this.object.frustumCulled = false;
    this.object.renderOrder = 5;
    this.update({ ejection: 0, filling: 0 }, 0);
  }

  definePaths(lm) {
    const { lv, mitral, tricuspid, aortic, pulmonary, RA, LA, RV } = lm;
    const curve = (pts) => new THREE.CatmullRomCurve3(pts.map((p) => new THREE.Vector3(...p)), false, 'centripetal');
    const seg = (rate, pts, radius) => ({ rate, curve: curve(pts), radius, len: 0 });
    const tvOut = add(tricuspid.center, mul(tricuspid.axis, 1.8));
    const rvApex = lv(5.0, 2.2, -1.6);
    const right = (inflow, branch) => ({
      blood: 'venous', weight: 1,
      segs: [
        seg('venous', inflow, 0.55),
        seg('avIn', [inflow[inflow.length - 1], lerp(RA.center, tricuspid.center, 0.55), tricuspid.center], 0.9),
        seg('fill', [tricuspid.center, tvOut, rvApex], 0.9),
        seg('eject', [rvApex, comb([RV.center, 1], [lm.e1, 0.6]), RV.outflowStart, add(pulmonary.center, mul(pulmonary.axis, -0.6)), pulmonary.center], 0.7),
        seg('out', [pulmonary.center, ...lm.pulmonaryTrunk.path.slice(1), ...branch.path.slice(1)], 0.6),
      ],
    });
    const lvApex = lv(6.1, 0, 0);
    const lvot = [lvApex, lv(4.0, 0.6, -0.35), lv(1.4, 0.8, -0.45), add(aortic.center, mul(aortic.axis, -0.5)), aortic.center];
    const left = (pv, outPath) => ({
      blood: 'arterial', weight: 1,
      segs: [
        seg('venous', [...pv.path, LA.center], 0.4),
        seg('avIn', [LA.center, lerp(LA.center, mitral.center, 0.6), mitral.center], 0.9),
        seg('fill', [mitral.center, add(mitral.center, mul(mitral.axis, 2.6)), lvApex], 1.0),
        seg('eject', lvot, 0.75),
        seg('out', [aortic.center, ...outPath], 0.7),
      ],
    });
    const svcIn = [...lm.svc.path, add(RA.center, [0, 0.4, 0])];
    const ivcIn = [...lm.ivc.path, add(RA.center, [0, -0.5, 0])];
    const aoPath = lm.aorta.path.slice(1);
    const branchPath = (b) => [...lm.aorta.path.slice(1, 6), b.from, ...b.path];
    const paths = [
      right(svcIn, lm.pulmonaryTrunk.right), right(svcIn, lm.pulmonaryTrunk.left),
      right(ivcIn, lm.pulmonaryTrunk.right), right(ivcIn, lm.pulmonaryTrunk.left),
      ...lm.pulmonaryVeins.map((v) => left(v, aoPath)),
      ...lm.pulmonaryVeins.map((v, i) => left(v, branchPath(lm.aorta.branches[i % 3]))),
    ];
    for (const p of paths) for (const s of p.segs) s.len = s.curve.getLength();
    return paths;
  }

  pickPath(blood = null) {
    const list = blood ? this.paths.filter((p) => p.blood === blood) : this.paths;
    return list[Math.floor(this.R() * list.length)];
  }

  rates(st) {
    const ej = Math.min(1.6, (st.ejection || 0) / 0.42);
    const fi = Math.min(1.6, (st.filling || 0) / 0.75);
    return {
      venous: 0.3 + 0.5 * Math.min(1, ej) + 0.35 * fi,
      avIn: 0.05 + fi * 1.1,
      fill: 0.02 + fi * 1.0,
      eject: 0.02 + ej * 1.25,
      out: 0.25 + ej * 1.1,
    };
  }

  /** dtSim in ms of simulated time. */
  update(st, dtSim) {
    const rates = this.rates(st);
    const pos = this.object.geometry.attributes.position.array;
    const v3 = new THREE.Vector3();
    for (let i = 0; i < this.count; i++) {
      const q = this.p[i];
      let seg = q.path.segs[q.seg];
      q.s += (rates[seg.rate] * 0.03 * q.v * dtSim) / seg.len;
      while (q.s >= 1) {
        q.s -= 1;
        q.seg++;
        if (q.seg >= q.path.segs.length) {
          const np = this.pickPath(q.path.blood);
          q.path = np;
          q.seg = 0;
          q.s = this.R() * 0.2;
        }
        seg = q.path.segs[q.seg];
      }
      seg.curve.getPointAt(Math.min(q.s, 1), v3);
      const r = seg.radius;
      pos[i * 3] = v3.x + q.j[0] * r;
      pos[i * 3 + 1] = v3.y + q.j[1] * r;
      pos[i * 3 + 2] = v3.z + q.j[2] * r;
    }
    this.object.geometry.attributes.position.needsUpdate = true;
  }
}
