// Myocardial parts: LV free wall, interventricular septum, RV, LA, RA,
// papillary muscles and the moderator band. Meshed from the SDFs in anatomy/fields.js.
//
// Per-vertex attributes:
//   aAct — electrical activation time (atria: ms after the P wave from the SA node;
//          ventricles: 1500 + ms after QRS onset, earliest at mid-septal/apical
//          endocardium and latest at the posterobasal epicardium — the normal sequence).
//   aFat — epicardial fat weight along the AV and interventricular grooves.

import { add, sub, mul, norm, dot, dist, clamp, frame, comb } from '../core/vec.js';
import { roundCone, smoothUnion, smax } from '../core/sdf.js';
import { sdfToBuilder } from '../core/geometry.js';
import { boxPoints } from '../anatomy/fields.js';

/** March from `p` along `dir` until `f` becomes positive (leaves a cavity). */
export function marchOut(f, p, dir, step = 0.04, max = 8) {
  for (let t = 0; t < max; t += step) {
    const q = add(p, mul(dir, t));
    if (f(q[0], q[1], q[2]) > 0) return q;
  }
  return add(p, mul(dir, max));
}

/** Papillary muscle definitions (rest coordinates) — shared with chordae and conduction. */
export function definePapillaryMuscles(lm, fields) {
  const mf = fields.mitralFrame;
  const m = lm.mitral;
  const lvPM = [
    { id: 'anterolateral', phi: 0.15 },
    { id: 'posteromedial', phi: Math.PI + 0.25 },
  ].map(({ id, phi }) => {
    const radial = norm(comb([mf.u, Math.cos(phi)], [mf.v, Math.sin(phi) * 0.8]));
    const tip = comb([m.center, 1], [mf.w, 2.45], [radial, m.rx * 0.62]);
    const dir = norm(comb([radial, 1], [mf.w, 1.1]));
    const wallPt = marchOut(fields.lvCavity, tip, dir);
    const base = add(wallPt, mul(dir, 0.35));
    return { id, phi, tip, base, heads: heads(tip, base, 3, id.length) };
  });
  const tf = fields.tricuspidFrame;
  const t = lm.tricuspid;
  const rvDefs = valveLeafletsTricuspid(lm, fields);
  // Papillary muscles sit beneath the commissures.
  const rvPM = [
    { id: 'anterior', phi: rvDefs.commAP, depth: 2.7, r: 0.6, size: 1 },
    { id: 'posterior', phi: rvDefs.commPS, depth: 2.4, r: 0.55, size: 0.8 },
    { id: 'septal', phi: rvDefs.commAS, depth: 1.5, r: 0.45, size: 0.55 },
  ].map(({ id, phi, depth, r, size }) => {
    const radial = norm(comb([tf.u, Math.cos(phi)], [tf.v, Math.sin(phi)]));
    const tip = comb([t.center, 1], [tf.w, depth], [radial, t.rx * r]);
    const dir = norm(comb([radial, 1], [tf.w, 0.6]));
    const wallPt = marchOut(fields.rvCavity, tip, dir);
    const base = add(wallPt, mul(dir, 0.25));
    return { id, phi, tip, base, size, heads: heads(tip, base, id === 'septal' ? 2 : 3, id.length) };
  });
  return { lv: lvPM, rv: rvPM, tricuspid: rvDefs };
}

function heads(tip, base, n, seed) {
  const ax = norm(sub(tip, base));
  const f = frame(ax, [0.3, 0.9, 0.2]);
  const out = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2 + seed;
    out.push(comb([tip, 1], [ax, 0.18], [f.u, Math.cos(a) * 0.32], [f.v, Math.sin(a) * 0.32]));
  }
  return out;
}

/** Tricuspid leaflet layout (angles in the tricuspid frame), decided from the septum direction. */
export function valveLeafletsTricuspid(lm, fields) {
  const tf = fields.tricuspidFrame;
  const toSeptum = norm(sub(lm.lv(2.0, 0, 0), lm.tricuspid.center));
  const phiS = Math.atan2(dot(toSeptum, tf.v), dot(toSeptum, tf.u));
  const D = Math.PI / 180;
  // anterior centred on u (by construction of tricuspid.ref)
  if (phiS > 0) {
    return {
      leaflets: [
        { name: 'anterior', a0: -70 * D, a1: 70 * D, len: 2.0 },
        { name: 'septal', a0: 70 * D, a1: 172 * D, len: 1.55 },
        { name: 'posterior', a0: 172 * D, a1: 290 * D, len: 1.65 },
      ],
      commAS: 70 * D, commPS: 172 * D, commAP: 290 * D,
    };
  }
  return {
    leaflets: [
      { name: 'anterior', a0: -70 * D, a1: 70 * D, len: 2.0 },
      { name: 'posterior', a0: 70 * D, a1: 188 * D, len: 1.65 },
      { name: 'septal', a0: 188 * D, a1: 290 * D, len: 1.55 },
    ],
    commAP: 70 * D, commPS: 188 * D, commAS: 290 * D,
  };
}

// ── activation & fat attribute functions ──────────────────────────────────
export function makeAttributeFns(lm, fields, grooveSamples) {
  const SA = lm.conduction.saNode;
  const bach = add(lm.LA.center, [0.4, 0.9, 1.0]); // Bachmann's bundle insertion (anterior LA)
  const atrialAct = (p) => Math.min(dist(p, SA) / 0.085, 22 + dist(p, bach) / 0.09, 110);
  const ventAct = (wallFn, cavityFn, septal, rv) => (p) => {
    const s = fields.sOf(p[0], p[1], p[2]);
    const sn = clamp(s / lm.L, 0, 1);
    const depth = clamp(cavityFn(p[0], p[1], p[2]) / (rv ? 0.4 : 0.95), 0, 1);
    let t = 8 + 46 * (1 - sn) ** 1.15 + 16 * depth;
    if (septal) t -= 6;
    if (rv) t += 4;
    // posterobasal LV is the last region to depolarise
    const back = clamp(-dot(sub(p, lm.B), lm.e1) / 3, 0, 1);
    t += 8 * back * (1 - sn);
    return 1500 + t;
  };
  // Spatial hash of groove samples for fat.
  const cell = 1.2;
  const grid = new Map();
  const key = (x, y, z) => `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
  for (const s of grooveSamples) {
    const k = key(s.p[0], s.p[1], s.p[2]);
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(s);
  }
  const fat = (p) => {
    if (fields.epicardium(p[0], p[1], p[2]) < -0.12) return 0; // only the epicardial surface
    let best = 0;
    const cx = Math.floor(p[0] / cell), cy = Math.floor(p[1] / cell), cz = Math.floor(p[2] / cell);
    for (let i = -1; i <= 1; i++)
      for (let j = -1; j <= 1; j++)
        for (let k = -1; k <= 1; k++) {
          const list = grid.get(`${cx + i},${cy + j},${cz + k}`);
          if (!list) continue;
          for (const s of list) {
            const d = dist(p, s.p);
            const w = clamp(1 - (d - 0.2) / (s.r * 0.95), 0, 1) ** 1.3 * s.w;
            if (w > best) best = w;
          }
        }
    return best;
  };
  return { atrialAct, ventAct, fat };
}

/** Build all myocardial part geometries. Returns { id: GeoBuilder }. */
export function buildChambers(lm, fields, pm, attrs, h, onStep = () => {}) {
  const out = {};
  const { lvWall, septumMask, rvWall, laWall, raWall, lvCavity, rvCavity } = fields;
  const lvFree = (x, y, z) => Math.max(lvWall(x, y, z), -septumMask(x, y, z));
  const sept = (x, y, z) => Math.max(lvWall(x, y, z), septumMask(x, y, z));
  const vLV = attrs.ventAct(lvWall, lvCavity, false, false);
  const vS = attrs.ventAct(lvWall, lvCavity, true, false);
  const vRV = attrs.ventAct(rvWall, rvCavity, false, true);

  onStep('Sol ventrikül');
  out.leftVentricle = sdfToBuilder(lvFree, fields.bounds.LV, h, { act: vLV, fat: attrs.fat });
  onStep('İnterventriküler septum');
  out.septum = sdfToBuilder(sept, fields.bounds.LV, h, { act: vS, fat: attrs.fat });
  onStep('Sağ ventrikül');
  out.rightVentricle = sdfToBuilder(rvWall, fields.bounds.RV, h, { act: vRV, fat: attrs.fat });
  onStep('Sol atriyum');
  out.leftAtrium = sdfToBuilder(laWall, fields.bounds.LA, h * 0.8, { act: attrs.atrialAct, fat: attrs.fat });
  onStep('Sağ atriyum');
  out.rightAtrium = sdfToBuilder(raWall, fields.bounds.RA, h * 0.8, { act: attrs.atrialAct, fat: attrs.fat });

  // Papillary muscles (multi-headed tapered cones, trimmed to the ventricular wall).
  onStep('Papiller kaslar');
  const pmField = (list, outer, scale = 1) => {
    const prims = [];
    for (const m of list) {
      const s = (m.size || 1) * scale;
      prims.push(roundCone(m.base, m.tip, 0.62 * s, 0.4 * s));
      for (const hd of m.heads) prims.push(roundCone(m.tip, hd, 0.3 * s, 0.17 * s));
    }
    const u = smoothUnion(prims, 0.22);
    return (x, y, z) => smax(u(x, y, z), outer(x, y, z) + 0.1, 0.1);
  };
  const lvPts = pm.lv.flatMap((m) => [m.base, m.tip, ...m.heads]);
  const rvPts = pm.rv.flatMap((m) => [m.base, m.tip, ...m.heads]);
  out.lvPapillary = sdfToBuilder(pmField(pm.lv, fields.lvOuter), boxPoints(lvPts, 1.0), h * 0.7, { act: vLV });
  out.rvPapillary = sdfToBuilder(pmField(pm.rv, fields.rvOuter), boxPoints(rvPts, 0.9), h * 0.7, { act: vRV });

  // Moderator band: septum (RV face) → base of the anterior papillary muscle.
  const antPM = pm.rv.find((m) => m.id === 'anterior');
  const septPt = moderatorSeptalPoint(lm, fields);
  const band = roundCone(septPt, add(antPM.base, mul(norm(sub(antPM.tip, antPM.base)), 0.4)), 0.3, 0.34);
  out.moderatorBand = sdfToBuilder(band, boxPoints([septPt, antPM.base], 0.8), h * 0.7, { act: vRV });
  pm.moderator = { septal: septPt, end: antPM.base };
  return out;
}

export function moderatorSeptalPoint(lm, fields) {
  // A point on the RV face of the septum, about 45% of the way from base to apex.
  const inner = lm.lv(3.4, 0.9, -1.1);
  const toRV = norm(sub(lm.RV.center, inner));
  return marchOut(fields.lvOuter, inner, toRV);
}
