// HeartAnatomy — procedural, animated, anatomically structured three.js heart.
//
//   const heart = new HeartAnatomy({ quality: 'high' });
//   await heart.build((p, label) => …);       // generates all geometry (no assets)
//   scene.add(heart.root);
//   renderer.localClippingEnabled = true;      // for section views
//   // per frame:
//   heart.update(deltaMs);
//
// Parts are independent THREE.Groups (heart.parts.get(id).group) authored in a shared
// body frame, so any subset assembles perfectly; explode/isolate/visibility/section/
// palette/opacity are per part. One CardiacCycle drives every part through a shared
// GPU deformation field, valve kinematics, chordae and blood-flow particles.

import * as THREE from 'three';
import { createLandmarks } from './anatomy/landmarks.js';
import { createFields } from './anatomy/fields.js';
import { PARTS, PART_MAP, SYSTEMS } from './anatomy/catalog.js';
import { createHeartUniforms, createHeartMaterial } from './core/heartMaterial.js';
import { definePapillaryMuscles, makeAttributeFns, buildChambers } from './parts/chambers.js';
import { createValves } from './parts/valves.js';
import { Chordae } from './parts/chordae.js';
import { buildVessels } from './parts/vessels.js';
import { routeCoronaries, buildCoronaryTubes } from './parts/coronaries.js';
import { buildConduction } from './parts/conduction.js';
import { buildPericardium } from './parts/pericardium.js';
import { CardiacCycle } from './sim/CardiacCycle.js';
import { BloodFlow } from './sim/BloodFlow.js';

const QUALITY = { low: 0.17, medium: 0.125, high: 0.095, ultra: 0.075 };

// Explode directions (body frame) × distance (cm). Chordae have none: they re-string
// themselves between the (moving) valves and papillary muscles.
const EXPLODE = {
  leftVentricle: [[1, -0.55, 0.15], 4.2], septum: [[0.05, -1, 0.35], 3.6], rightVentricle: [[-0.55, -0.15, 1], 4.6],
  leftAtrium: [[0.45, 0.55, -1], 4.6], rightAtrium: [[-1, 0.3, -0.15], 4.8],
  mitralValve: [[0.25, 0.75, -0.25], 1.6], tricuspidValve: [[-0.7, 0.35, 0.45], 2.0],
  aorticValve: [[-0.25, 1, 0.3], 2.2], pulmonaryValve: [[0.2, 1, 0.65], 3.0],
  lvPapillary: [[0.6, -0.75, 0], 2.4], rvPapillary: [[-0.45, -0.35, 0.85], 2.8], moderatorBand: [[-0.3, -0.5, 0.85], 3.0],
  aorta: [[-0.3, 1, 0.05], 4.4], pulmonaryTrunk: [[0.55, 1, 0.75], 4.2], superiorVenaCava: [[-1, 1, -0.1], 4.6],
  inferiorVenaCava: [[-0.5, -1, -0.45], 4.4], pulmonaryVeins: [[0.25, 0.4, -1], 6.6], ligamentumArteriosum: [[0.45, 1, -0.4], 5.6],
  saNode: [[-1, 0.3, -0.15], 4.8], internodalTracts: [[-0.6, 0.35, -0.3], 2.4], pericardium: [[0, 0.1, 1], 11],
};
const SYSTEM_EXPLODE = { coronary: 6.6, veins: 7.2 };

export class HeartAnatomy {
  constructor({ quality = 'high', palette = 'atlas', parts = null, heartRate = 72, rhythm = 'nsr' } = {}) {
    this.quality = quality;
    this.h = typeof quality === 'number' ? quality : QUALITY[quality] ?? QUALITY.high;
    this.palette = palette;
    this.only = parts ? new Set(parts) : null;
    this.root = new THREE.Group();
    this.root.name = 'HeartAnatomy';
    this.parts = new Map();
    this.lm = createLandmarks();
    this.uniforms = createHeartUniforms(this.lm);
    this.cycle = new CardiacCycle({ heartRate, rhythm });
    this.clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    this.section = { mode: 'none', offset: 0 };
    this.explode = 0;
    this.explodeTarget = 0;
    this.wallOpacity = 1;
    this.selected = null;
    this.hovered = null;
    this.isolated = null;
    this.flowVisible = false;
    this.systemVisible = Object.fromEntries(SYSTEMS.map((s) => [s.id, s.id !== 'pericardium']));
    this.partVisible = {};
    this.time = 0;
    this.assembly = null;
  }

  want(id) {
    return !this.only || this.only.has(id);
  }

  /** Generate all geometry. onProgress(fraction, label). */
  async build(onProgress = () => {}) {
    const lm = this.lm;
    const steps = 16;
    let k = 0;
    const tick = async (label) => {
      onProgress(Math.min(1, k++ / steps), label);
      await new Promise((r) => setTimeout(r, 0));
    };
    await tick('Anatomik alanlar');
    const fields = (this.fields = createFields(lm));
    await tick('Koroner oluklar');
    const cor = (this.coronaries = routeCoronaries(lm, fields));
    const pm = (this.pm = definePapillaryMuscles(lm, fields));
    const attrs = makeAttributeFns(lm, fields, cor.groove);

    const chambers = buildChambers(lm, fields, pm, attrs, this.h);
    await tick('Odacıklar');
    for (const id in chambers) if (this.want(id)) this.addPart(id, [chambers[id].build()]);

    await tick('Kapakçıklar');
    this.valves = createValves(lm, fields, pm.tricuspid);
    const valveIds = { mitral: 'mitralValve', tricuspid: 'tricuspidValve', aortic: 'aorticValve', pulmonary: 'pulmonaryValve' };
    for (const key in valveIds) {
      const g = this.valves[key].buildGeometry();
      if (this.want(valveIds[key])) this.addPart(valveIds[key], [g], { dynamic: true });
    }
    await tick('Korda tendinealar');
    this.chordae = {
      left: new Chordae(this.valves.mitral, pm.lv, { primaries: 7, secondaries: 3 }),
      right: new Chordae(this.valves.tricuspid, pm.rv, { primaries: 5, secondaries: 2 }),
    };
    if (this.want('chordaeLeft')) this.addPart('chordaeLeft', [this.chordae.left.buildGeometry()], { shift: true, dynamic: true });
    if (this.want('chordaeRight')) this.addPart('chordaeRight', [this.chordae.right.buildGeometry()], { shift: true, dynamic: true });

    await tick('Büyük damarlar');
    const vessels = buildVessels(lm, fields);
    for (const id of ['aorta', 'pulmonaryTrunk', 'superiorVenaCava', 'inferiorVenaCava', 'pulmonaryVeins', 'ligamentumArteriosum'])
      if (this.want(id)) this.addPart(id, [vessels[id].build()]);

    await tick('Koroner damarlar');
    const tubes = buildCoronaryTubes(cor.routes);
    for (const id in tubes) if (this.want(id)) this.addPart(id, [tubes[id].build()]);

    await tick('İleti sistemi');
    const cond = buildConduction(lm, fields, pm, this.valves);
    this.conductionPoints = cond.points;
    for (const id in cond.builders) if (this.want(id)) this.addPart(id, [cond.builders[id].build()]);

    await tick('Perikard');
    if (this.want('pericardium')) this.addPart('pericardium', [buildPericardium(lm, fields, Math.max(0.14, this.h * 1.5)).build()]);

    await tick('Kan akışı');
    this.flow = new BloodFlow(lm);
    this.flow.object.visible = this.flowVisible;
    this.root.add(this.flow.object);

    this.computeExplodeVectors();
    this.applyVisibility();
    this.setPalette(this.palette);
    onProgress(1, 'Hazır');
    return this;
  }

  addPart(id, geometries, { shift = false, dynamic = false } = {}) {
    const meta = PART_MAP.get(id);
    if (!meta) throw new Error('Unknown part ' + id);
    const group = new THREE.Group();
    group.name = id;
    const materials = [];
    const meshes = geometries.map((g) => {
      const transparent = !!meta.transparent;
      const mat = createHeartMaterial(this.uniforms, {
        color: meta.colors[this.palette],
        roughness: meta.system === 'valves' || meta.system === 'subvalvular' ? 0.62 : meta.system === 'vessels' ? 0.42 : 0.5,
        clearcoat: meta.system === 'chambers' || meta.system === 'coronary' || meta.system === 'veins' ? 0.45 : 0.25,
        glow: !!meta.glow,
        tissue: meta.system === 'chambers' ? 0.38 : meta.system === 'valves' ? 0.18 : 0.2,
        opacity: transparent ? 0.22 : 1,
        side: transparent ? THREE.FrontSide : THREE.DoubleSide,
        shift,
      });
      if (meta.glow) mat.emissive = new THREE.Color(meta.colors.atlas).multiplyScalar(0.12);
      materials.push(mat);
      const mesh = new THREE.Mesh(g, mat);
      mesh.name = id;
      mesh.userData.partId = id;
      // the GPU field moves vertices — keep bounds generous instead of frustum culling
      mesh.frustumCulled = false;
      if (transparent) mesh.renderOrder = 10;
      group.add(mesh);
      return mesh;
    });
    this.root.add(group);
    const box = new THREE.Box3();
    for (const m of meshes) { m.geometry.computeBoundingBox(); box.union(m.geometry.boundingBox); }
    const part = {
      id, meta, group, meshes, materials, dynamic,
      center: box.getCenter(new THREE.Vector3()), box,
      explodeDir: new THREE.Vector3(), explodeDist: 0, local: 0,
      pickId: this.parts.size + 1,
    };
    for (const mat of materials) mat.userData.local.uPickColor.value.setRGB((part.pickId & 255) / 255, ((part.pickId >> 8) & 255) / 255, 0.5);
    this.parts.set(id, part);
    return part;
  }

  computeExplodeVectors() {
    const c = new THREE.Vector3(...this.lm.heartCenter);
    for (const p of this.parts.values()) {
      const e = EXPLODE[p.id];
      if (p.id === 'chordaeLeft' || p.id === 'chordaeRight') continue;
      if (e) { p.explodeDir.set(...e[0]).normalize(); p.explodeDist = e[1]; }
      else if (SYSTEM_EXPLODE[p.meta.system]) {
        p.explodeDir.copy(p.center).sub(c).normalize();
        p.explodeDist = SYSTEM_EXPLODE[p.meta.system];
      }
    }
  }

  // ── Public controls ──────────────────────────────────────────────────────
  setExplode(v, animate = true) {
    this.explodeTarget = THREE.MathUtils.clamp(v, 0, 1);
    if (!animate) this.explode = this.explodeTarget;
    this.assembly = null;
  }

  /** Cinematic assembly: parts fly in system by system. */
  playAssembly(duration = 5200) {
    const order = ['pericardium', 'conduction', 'subvalvular', 'valves', 'chambers', 'vessels', 'coronary', 'veins'];
    const list = [...this.parts.values()].sort((a, b) => order.indexOf(a.meta.system) - order.indexOf(b.meta.system));
    const n = list.length;
    list.forEach((p, i) => { p.assemblyDelay = (i / n) * duration * 0.62; });
    this.assembly = { t: 0, duration, per: duration * 0.38 };
    this.explode = 1;
    this.explodeTarget = 0;
  }

  setSystemVisible(sys, on) { this.systemVisible[sys] = on; this.applyVisibility(); }
  setPartVisible(id, on) { this.partVisible[id] = on; this.applyVisibility(); }
  isolate(id) { this.isolated = id; this.applyVisibility(); }
  setFlowVisible(on) { this.flowVisible = on; if (this.flow) this.flow.object.visible = on; }

  isPartVisible(id) {
    const p = this.parts.get(id);
    if (!p) return false;
    if (this.isolated) return id === this.isolated;
    if (this.partVisible[id] === false) return false;
    return this.systemVisible[p.meta.system] !== false;
  }

  applyVisibility() {
    for (const p of this.parts.values()) p.group.visible = this.isPartVisible(p.id);
  }

  select(id) {
    this.selected = id;
  }
  hover(id) {
    this.hovered = id;
  }

  setPalette(name) {
    this.palette = name;
    for (const p of this.parts.values())
      for (const m of p.materials) {
        m.color.set(p.meta.colors[name]);
        m.userData.local.uCutColor.value.set(p.meta.colors[name]).multiplyScalar(name === 'real' ? 0.78 : 0.7);
      }
    this.uniforms.uFat.value = name === 'real' ? 1 : 0;
  }

  /** Opacity of the myocardial walls and great vessels (ghost view to see valves, conduction and flow). */
  setWallOpacity(o) {
    this.wallOpacity = o;
    for (const p of this.parts.values()) {
      const sys = p.meta.system;
      if (sys !== 'chambers' && sys !== 'vessels') continue;
      // great vessels fade a little less than the walls so the outflow tracts stay readable
      const op = sys === 'vessels' ? Math.min(1, o * 1.6) : o;
      for (const m of p.materials) {
        const t = op < 0.999;
        if (m.transparent !== t) m.needsUpdate = true;
        m.transparent = t;
        m.opacity = op;
        m.depthWrite = !t;
        m.side = t ? THREE.FrontSide : THREE.DoubleSide;
      }
      for (const mesh of p.meshes) mesh.renderOrder = op < 0.999 ? 8 : 0;
    }
  }

  setElectricView(on) { this.uniforms.uElectric.value = on ? 1 : 0; }
  setDeformation(on) { this.uniforms.uDeform.value = on ? 1 : 0; }

  /** Section views: none | fourChamber | longAxis | shortAxis | coronal | sagittal. offset in cm. */
  setSection(mode, offset = 0) {
    const was = this.section.mode;
    this.section = { mode, offset };
    const lm = this.lm;
    const V = (a) => new THREE.Vector3(...a);
    const A = V(lm.APEX), M = V(lm.mitral.center), T = V(lm.tricuspid.center), Ao = V(lm.aortic.center);
    const n = new THREE.Vector3();
    let p0 = A.clone();
    if (mode === 'fourChamber') { n.crossVectors(T.clone().sub(A), M.clone().sub(A)).normalize(); if (n.z > 0) n.negate(); }
    else if (mode === 'longAxis') { n.crossVectors(Ao.clone().sub(A), M.clone().sub(A)).normalize(); if (n.x < 0) n.negate(); }
    else if (mode === 'shortAxis') { n.set(...lm.n); p0 = V(lm.B).addScaledVector(n, lm.L * 0.45); }
    else if (mode === 'coronal') { n.set(0, 0, -1); p0 = V(lm.heartCenter); }
    else if (mode === 'sagittal') { n.set(1, 0, 0); p0 = V(lm.heartCenter); }
    if (mode !== 'none') this.clipPlane.setFromNormalAndCoplanarPoint(n, p0.addScaledVector(n, -offset));
    if ((mode === 'none') !== (was === 'none') || was === undefined) {
      for (const p of this.parts.values())
        for (const m of p.materials) { m.clippingPlanes = mode === 'none' ? [] : [this.clipPlane]; m.needsUpdate = true; }
    }
    // the pericardium is meaningless in a section — hide it there
    return this.clipPlane;
  }

  /** Suggested camera direction for a section (unit vector from target to camera). */
  sectionViewDir() {
    if (this.section.mode === 'none') return null;
    return this.clipPlane.normal.clone().negate();
  }

  /** World-space bounds of the visible parts (for camera framing). */
  visibleBounds(ids = null) {
    const box = new THREE.Box3();
    for (const p of this.parts.values()) {
      if (ids ? !ids.includes(p.id) : !p.group.visible) continue;
      const b = p.box.clone().translate(p.group.position);
      box.union(b);
    }
    return box.applyMatrix4(this.root.matrixWorld);
  }

  // ── Picking (GPU id pass) ────────────────────────────────────────────────
  pick(renderer, scene, camera, x, y) {
    const rt = (this._pickRT ||= new THREE.WebGLRenderTarget(1, 1));
    const size = renderer.getSize(new THREE.Vector2());
    const flowVis = this.flow?.object.visible;
    if (this.flow) this.flow.object.visible = false;
    const hidden = [];
    scene.traverse((o) => { if (o.visible && (o.isLight === undefined) && o.isMesh && !o.userData.partId) { hidden.push(o); o.visible = false; } });
    this.uniforms.uPickMode.value = 1;
    const bg = scene.background;
    scene.background = null;
    const cc = renderer.getClearColor(new THREE.Color()), ca = renderer.getClearAlpha();
    renderer.setClearColor(0x000000, 0);
    camera.setViewOffset(size.x, size.y, Math.floor(x), Math.floor(y), 1, 1);
    renderer.setRenderTarget(rt);
    renderer.clear();
    renderer.render(scene, camera);
    const px = new Uint8Array(4);
    renderer.readRenderTargetPixels(rt, 0, 0, 1, 1, px);
    renderer.setRenderTarget(null);
    camera.clearViewOffset();
    renderer.setClearColor(cc, ca);
    scene.background = bg;
    this.uniforms.uPickMode.value = 0;
    for (const o of hidden) o.visible = true;
    if (this.flow) this.flow.object.visible = flowVis;
    if (px[3] === 0) return null;
    const id = px[0] + (px[1] << 8);
    for (const p of this.parts.values()) if (p.pickId === id) return p.id;
    return null;
  }

  // ── Frame update ────────────────────────────────────────────────────────
  update(dtMs) {
    this.time += dtMs;
    const prevT = this.cycle.t;
    const st = this.cycle.update(dtMs);
    const dtSim = this.cycle.t - prevT;
    const u = this.uniforms;
    u.uVc.value = st.vContraction;
    u.uAc.value = st.aContraction;
    u.uQuiverV.value = st.quiverV;
    u.uQuiverA.value = st.quiverA;
    u.uTime.value = this.cycle.t;
    u.uAoP.value = st.aoPulse;
    u.uPaP.value = st.paPulse;
    u.uVenP.value = st.venPulse;
    u.uSinceP.value = st.sinceP;
    u.uSinceQRS.value = st.sinceQRS;
    u.uPR.value = st.pr;
    u.uAF.value = st.rhythm === 'afib' ? 1 : 0;
    u.uVF.value = st.rhythm === 'vf' ? 1 : 0;

    // explode (eased) and assembly choreography
    if (this.assembly) {
      const a = this.assembly;
      a.t += dtMs;
      for (const p of this.parts.values()) {
        const x = THREE.MathUtils.clamp((a.t - p.assemblyDelay) / a.per, 0, 1);
        p.local = 1 - (1 - (1 - x) ** 3);
      }
      if (a.t > a.duration + 100) { this.assembly = null; this.explode = 0; }
    } else {
      this.explode += (this.explodeTarget - this.explode) * Math.min(1, dtMs / 140);
      for (const p of this.parts.values()) p.local = this.explode;
    }
    for (const p of this.parts.values()) p.group.position.copy(p.explodeDir).multiplyScalar(p.explodeDist * p.local);

    // valves + chordae
    if (this.valves) {
      const vt = this.cycle.t;
      this.valves.mitral.update(st.mitral, vt);
      this.valves.tricuspid.update(st.tricuspid, vt);
      this.valves.aortic.update(st.aortic, vt);
      this.valves.pulmonary.update(st.pulmonary, vt);
      const off = (id) => (this.parts.get(id)?.group.position.toArray() ?? [0, 0, 0]);
      if (this.parts.has('chordaeLeft')) this.chordae.left.update(off('mitralValve'), off('lvPapillary'));
      if (this.parts.has('chordaeRight')) this.chordae.right.update(off('tricuspidValve'), off('rvPapillary'));
    }

    // blood flow (hidden while exploded)
    if (this.flow) {
      this.flow.object.visible = this.flowVisible && this.explode < 0.03 && !this.assembly;
      if (this.flow.object.visible) this.flow.update(st, dtSim);
    }

    // highlight
    const pulse = 0.5 + 0.5 * Math.sin(this.time * 0.006);
    for (const p of this.parts.values()) {
      const v = p.id === this.selected ? 0.16 + 0.12 * pulse : p.id === this.hovered ? 0.1 : 0;
      for (const m of p.materials) m.userData.local.uHighlight.value = v;
    }
    return st;
  }

  dispose() {
    for (const p of this.parts.values()) {
      for (const m of p.meshes) m.geometry.dispose();
      for (const m of p.materials) m.dispose();
    }
    this.flow?.object.geometry.dispose();
    this.flow?.object.material.dispose();
    this._pickRT?.dispose();
  }
}

export { PARTS, SYSTEMS };
