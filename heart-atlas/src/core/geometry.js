// Geometry assembly helpers: a vertex/attribute accumulator shared by every part,
// a variable-radius tube (vessels, conduction fibres, chordae) with true wall
// thickness and parallel-transport frames, and conversion to THREE.BufferGeometry.
//
// Every heart geometry carries the same attribute layout so one deformation shader
// can animate all of them coherently:
//   aVessel      vec4  centre-line point (xyz) + radial compliance (w)
//   aVesselKind  float 0 systemic artery · 1 pulmonary artery · 2 vein
//   aFat         float epicardial fat weight (realistic palette)
//   aAct         float electrical activation time (ms; ≥1000 → QRS-relative; <0 none)

import * as THREE from 'three';
import { meshSDF } from './surfaceNets.js';

export class GeoBuilder {
  constructor() {
    this.pos = [];
    this.nrm = [];
    this.vessel = [];
    this.kind = [];
    this.fat = [];
    this.act = [];
    this.idx = [];
  }
  get count() {
    return this.pos.length / 3;
  }
  vertex(p, n, { vessel = null, kind = 0, fat = 0, act = -1 } = {}) {
    this.pos.push(p[0], p[1], p[2]);
    this.nrm.push(n[0], n[1], n[2]);
    if (vessel) this.vessel.push(vessel[0], vessel[1], vessel[2], vessel[3]);
    else this.vessel.push(0, 0, 0, 0);
    this.kind.push(kind);
    this.fat.push(fat);
    this.act.push(act);
    return this.count - 1;
  }
  tri(a, b, c) {
    this.idx.push(a, b, c);
  }
  /** Append a raw mesh {positions, normals, indices}. */
  appendMesh(m, attrs = {}) {
    const base = this.count;
    const P = m.positions, N = m.normals;
    for (let i = 0; i < P.length; i += 3) {
      const p = [P[i], P[i + 1], P[i + 2]];
      this.vertex(p, [N[i], N[i + 1], N[i + 2]], {
        act: attrs.act ? attrs.act(p) : -1,
        fat: attrs.fat ? attrs.fat(p) : 0,
      });
    }
    for (let i = 0; i < m.indices.length; i++) this.idx.push(m.indices[i] + base);
  }
  build() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nrm, 3));
    g.setAttribute('aVessel', new THREE.Float32BufferAttribute(this.vessel, 4));
    g.setAttribute('aVesselKind', new THREE.Float32BufferAttribute(this.kind, 1));
    g.setAttribute('aFat', new THREE.Float32BufferAttribute(this.fat, 1));
    g.setAttribute('aAct', new THREE.Float32BufferAttribute(this.act, 1));
    g.setIndex(this.count > 65535 ? new THREE.Uint32BufferAttribute(this.idx, 1) : new THREE.Uint16BufferAttribute(this.idx, 1));
    g.computeBoundingSphere();
    g.computeBoundingBox();
    return g;
  }
}

/** Mesh an SDF over bounds into a GeoBuilder. */
export function sdfToBuilder(sdf, bounds, h, attrs = {}, builder = new GeoBuilder()) {
  const m = meshSDF(sdf, bounds, h);
  builder.appendMesh(m, attrs);
  return builder;
}

/** Make sure a THREE geometry has the shared heart attribute layout. */
export function ensureHeartAttributes(g) {
  const n = g.attributes.position.count;
  if (!g.attributes.aVessel) g.setAttribute('aVessel', new THREE.Float32BufferAttribute(new Float32Array(n * 4), 4));
  if (!g.attributes.aVesselKind) g.setAttribute('aVesselKind', new THREE.Float32BufferAttribute(new Float32Array(n), 1));
  if (!g.attributes.aFat) g.setAttribute('aFat', new THREE.Float32BufferAttribute(new Float32Array(n), 1));
  if (!g.attributes.aAct) g.setAttribute('aAct', new THREE.Float32BufferAttribute(new Float32Array(n).fill(-1), 1));
  return g;
}

export function curveFrom(points, tension = 0.5) {
  return new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(p[0], p[1], p[2])), false, 'catmullrom', tension);
}

/**
 * Parallel-transport frames along a curve; `initialNormal` fixes the angular origin
 * (needed for the aortic sinuses so they line up with the valve cusps).
 */
export function transportFrames(curve, segments, initialNormal = null) {
  const T = [], N = [], B = [], P = [];
  for (let i = 0; i <= segments; i++) {
    const u = i / segments;
    P.push(curve.getPointAt(u));
    T.push(curve.getTangentAt(u).normalize());
  }
  let n0;
  if (initialNormal) n0 = new THREE.Vector3(...initialNormal);
  else {
    const t = T[0];
    n0 = Math.abs(t.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  }
  n0.sub(T[0].clone().multiplyScalar(n0.dot(T[0]))).normalize();
  N.push(n0);
  B.push(new THREE.Vector3().crossVectors(T[0], n0));
  for (let i = 1; i <= segments; i++) {
    const axis = new THREE.Vector3().crossVectors(T[i - 1], T[i]);
    const n = N[i - 1].clone();
    if (axis.length() > 1e-6) {
      axis.normalize();
      const ang = Math.acos(THREE.MathUtils.clamp(T[i - 1].dot(T[i]), -1, 1));
      n.applyAxisAngle(axis, ang);
    }
    n.sub(T[i].clone().multiplyScalar(n.dot(T[i]))).normalize();
    N.push(n);
    B.push(new THREE.Vector3().crossVectors(T[i], n));
  }
  return { T, N, B, P };
}

/**
 * Variable-radius tube.
 * opts.radius(t, angle) → outer radius; opts.wall → wall thickness (0 = single surface)
 * opts.capStart/capEnd → close ends with a dome (for blind vessel ends / fibres)
 * opts.kind / compliance → pulse distension; opts.act(t) → activation time.
 */
export function tube(builder, points, opts = {}) {
  const {
    radius = () => 0.5,
    wall = 0,
    segments = 64,
    radial = 20,
    capStart = false,
    capEnd = false,
    kind = 0,
    compliance = 0,
    act = null,
    fat = 0,
    tension = 0.5,
    initialNormal = null,
  } = opts;
  const curve = typeof points.getPointAt === "function" ? points : curveFrom(points, tension);
  const { T, N, B, P } = transportFrames(curve, segments, initialNormal);
  const rings = [];
  const inner = [];
  const vtx = (p, n, i) => builder.vertex(p, n, {
    vessel: compliance ? [P[i].x, P[i].y, P[i].z, compliance] : null,
    kind,
    fat,
    act: act ? act(i / segments) : -1,
  });
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const ring = [], ringIn = [];
    for (let j = 0; j < radial; j++) {
      const a = (j / radial) * Math.PI * 2;
      const ca = Math.cos(a), sa = Math.sin(a);
      const dir = new THREE.Vector3().addScaledVector(N[i], ca).addScaledVector(B[i], sa);
      const r = radius(t, a);
      const p = P[i].clone().addScaledVector(dir, r);
      ring.push(vtx([p.x, p.y, p.z], [dir.x, dir.y, dir.z], i));
      if (wall > 0) {
        const q = P[i].clone().addScaledVector(dir, Math.max(r - wall, r * 0.35));
        ringIn.push(vtx([q.x, q.y, q.z], [-dir.x, -dir.y, -dir.z], i));
      }
    }
    rings.push(ring);
    inner.push(ringIn);
  }
  for (let i = 0; i < segments; i++)
    for (let j = 0; j < radial; j++) {
      const j2 = (j + 1) % radial;
      const a = rings[i][j], b = rings[i][j2], c = rings[i + 1][j], d = rings[i + 1][j2];
      builder.tri(a, b, c);
      builder.tri(b, d, c);
      if (wall > 0) {
        const a2 = inner[i][j], b2 = inner[i][j2], c2 = inner[i + 1][j], d2 = inner[i + 1][j2];
        builder.tri(a2, c2, b2);
        builder.tri(b2, c2, d2);
      }
    }
  // Ends: rims for walled vessels (cut vessel look), domes for capped fibres.
  const end = (i, sign, cap) => {
    const t = T[i].clone().multiplyScalar(sign);
    if (wall > 0 && !cap) {
      const ro = [], ri = [];
      for (let j = 0; j < radial; j++) {
        const po = builder.pos.slice(rings[i][j] * 3, rings[i][j] * 3 + 3);
        const pi = builder.pos.slice(inner[i][j] * 3, inner[i][j] * 3 + 3);
        ro.push(vtx(po, [t.x, t.y, t.z], i));
        ri.push(vtx(pi, [t.x, t.y, t.z], i));
      }
      for (let j = 0; j < radial; j++) {
        const j2 = (j + 1) % radial;
        if (sign > 0) { builder.tri(ro[j], ro[j2], ri[j]); builder.tri(ro[j2], ri[j2], ri[j]); }
        else { builder.tri(ro[j], ri[j], ro[j2]); builder.tri(ro[j2], ri[j], ri[j2]); }
      }
    } else if (cap) {
      const steps = 4;
      let prev = rings[i];
      for (let s = 1; s <= steps; s++) {
        const phi = (s / steps) * Math.PI * 0.5;
        const ring = [];
        for (let j = 0; j < radial; j++) {
          const a = (j / radial) * Math.PI * 2;
          const r = radius(i / segments, a);
          const dir = new THREE.Vector3().addScaledVector(N[i], Math.cos(a)).addScaledVector(B[i], Math.sin(a));
          const nn = dir.clone().multiplyScalar(Math.cos(phi)).addScaledVector(t, Math.sin(phi)).normalize();
          const p = P[i].clone().addScaledVector(dir, r * Math.cos(phi)).addScaledVector(t, r * Math.sin(phi));
          ring.push(vtx([p.x, p.y, p.z], [nn.x, nn.y, nn.z], i));
        }
        for (let j = 0; j < radial; j++) {
          const j2 = (j + 1) % radial;
          if (sign > 0) { builder.tri(prev[j], prev[j2], ring[j]); builder.tri(prev[j2], ring[j2], ring[j]); }
          else { builder.tri(prev[j], ring[j], prev[j2]); builder.tri(prev[j2], ring[j], ring[j2]); }
        }
        prev = ring;
      }
    }
  };
  end(0, -1, capStart);
  end(segments, 1, capEnd);
  return { curve, frames: { T, N, B, P } };
}

/** Sample a curve into arrays of points (plain arrays). */
export function sampleCurve(points, n, tension = 0.5) {
  const c = typeof points.getPointAt === "function" ? points : curveFrom(points, tension);
  const out = [];
  for (let i = 0; i <= n; i++) {
    const p = c.getPointAt(i / n);
    out.push([p.x, p.y, p.z]);
  }
  return out;
}
