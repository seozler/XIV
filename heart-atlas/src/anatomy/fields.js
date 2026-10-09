// Signed-distance definitions of the four chambers and their shared interfaces.
//
// Design rules that keep the parts separable yet perfectly joinable:
//   • The interventricular septum belongs to the LV wall; the RV wraps around the LV
//     outer surface (crescentic RV), so the RV cavity is bounded directly by the septum.
//   • The interatrial septum belongs to the LA wall; the RA wraps around it and carries
//     the fossa ovalis (a dent pressed into the LA's right face).
//   • Atria are cut at their atrioventricular valve planes; ventricles have a fibrous
//     roof with the valve orifices carved out — every great vessel and valve opens
//     into a real lumen.

import { add, sub, mul, norm, comb, cross, frame } from '../core/vec.js';
import {
  ellipsoid, roundCone, tubeChain, ellipticCylinder, halfSpace,
  smoothUnion, intersect, smax, ridged, noise3, projectToSurface,
} from '../core/sdf.js';

// ── bounding box helpers ──
export const box = (min, max) => ({ min: [...min], max: [...max] });
export function boxUnion(...b) {
  return {
    min: [0, 1, 2].map((i) => Math.min(...b.map((x) => x.min[i]))),
    max: [0, 1, 2].map((i) => Math.max(...b.map((x) => x.max[i]))),
  };
}
export const boxExpand = (b, m) => ({ min: b.min.map((v) => v - m), max: b.max.map((v) => v + m) });
export function boxEllipsoid(c, basis, r) {
  const h = [0, 1, 2].map((i) => Math.sqrt(basis.reduce((s, ax, j) => s + (r[j] * ax[i]) ** 2, 0)));
  return { min: c.map((v, i) => v - h[i]), max: c.map((v, i) => v + h[i]) };
}
export function boxPoints(pts, r) {
  return {
    min: [0, 1, 2].map((i) => Math.min(...pts.map((p) => p[i])) - r),
    max: [0, 1, 2].map((i) => Math.max(...pts.map((p) => p[i])) + r),
  };
}

export function createFields(lm) {
  const { B, n, e1, eL, lv, LV, RV, RA, LA, aortic, mitral, tricuspid, pulmonary } = lm;
  const lvBasis = [n, e1, eL];
  const sOf = (x, y, z) => (x - B[0]) * n[0] + (y - B[1]) * n[1] + (z - B[2]) * n[2];

  // ── Aortic root envelope (used to indent the atria: "torus aorticus") ──
  const aoRoot = roundCone(add(aortic.center, mul(aortic.axis, -0.4)), add(aortic.center, mul(aortic.axis, 3.2)), 1.62, 1.55);
  const paRoot = roundCone(add(pulmonary.center, mul(pulmonary.axis, -0.3)), add(pulmonary.center, mul(pulmonary.axis, 2.5)), 1.5, 1.5);

  // ── Left ventricle ──
  const lvEll = ellipsoid(LV.outer.center, lvBasis, LV.outer.radii);
  const lvRoof = halfSpace(lv(LV.outer.roof, 0, 0), mul(n, -1));
  const lvotOuter = roundCone(lv(1.0, 0.85, -0.6), add(aortic.center, mul(aortic.axis, 0.15)), 1.6, 1.48);
  const lvOuter = smoothUnion([intersect(lvEll, lvRoof, 0.7), lvotOuter], 0.6);

  const cav = ellipsoid(LV.cavity.center, lvBasis, LV.cavity.radii);
  const cavRoof = halfSpace(lv(LV.cavity.roof, 0, 0), mul(n, -1));
  const lvotCav = roundCone(lv(1.3, 0.75, -0.45), add(aortic.center, mul(aortic.axis, 0.4)), 1.12, 1.05);
  const lvCavSmooth = smoothUnion([intersect(cav, cavRoof, 0.45), lvotCav], 0.45);
  // Trabeculae carneae: fine ridges toward the apex (LV is finely trabeculated).
  const lvCavity = (x, y, z) => {
    const d = lvCavSmooth(x, y, z);
    if (d < -0.6 || d > 0.6) return d;
    const s = sOf(x, y, z);
    const w = Math.min(1, Math.max(0, (s - 2.2) / 3.5));
    if (w <= 0) return d;
    return d + w * 0.17 * ridged(x * 1.15 + 3.1, y * 1.15, z * 1.15 - 7.3);
  };

  const mf = frame(mitral.axis, mitral.ref);
  const mitralHole = ellipticCylinder(add(mitral.center, mul(mitral.axis, -0.2)), mitral.axis, mf.u, mitral.rx - 0.08, mitral.ry - 0.08, -2.0, 0.9);
  const aorticHole = roundCone(add(aortic.center, mul(aortic.axis, -0.5)), add(aortic.center, mul(aortic.axis, 3.0)), aortic.radius - 0.08, aortic.radius - 0.08);

  const lvWallRaw = (x, y, z) => Math.max(lvOuter(x, y, z), -lvCavity(x, y, z));
  const lvWall = (x, y, z) => Math.max(lvWallRaw(x, y, z), -mitralHole(x, y, z), -aorticHole(x, y, z));

  // ── Right ventricle (crescent wrapped around the LV) ──
  const rvEll = ellipsoid(RV.center, lvBasis, RV.radii);
  const rvRoof = halfSpace(lv(RV.roof, 0, 0), mul(n, -1));
  const rvBody = intersect(rvEll, rvRoof, 0.6);
  const rvOutflow = roundCone(RV.outflowStart, add(pulmonary.center, mul(pulmonary.axis, 0.1)), 1.8, 1.45);
  const tvInflow = roundCone(add(tricuspid.center, mul(tricuspid.axis, -0.15)), add(tricuspid.center, mul(tricuspid.axis, 1.6)), 2.05, 2.15);
  const rvRaw = smoothUnion([rvBody, rvOutflow, tvInflow], 0.9);
  // The RV overlaps the outermost ~1 mm of the LV so the interventricular grooves close cleanly.
  const rvOuter = (x, y, z) => smax(smax(rvRaw(x, y, z), -lvOuter(x, y, z) - 0.12, 0.22), -aoRoot(x, y, z), 0.3);
  // Coarse trabeculation of the RV (plus the septomarginal trabecula, built separately).
  const rvCavity = (x, y, z) => {
    let d = rvRaw(x, y, z) + RV.wall;
    if (d > -0.7 && d < 0.7) {
      const s = sOf(x, y, z);
      const w = Math.min(1, Math.max(0, (s - 1.0) / 3.0));
      // The infundibulum (outflow) is smooth-walled: fade trabeculae near the pulmonary valve.
      const dp = Math.hypot(x - pulmonary.center[0], y - pulmonary.center[1], z - pulmonary.center[2]);
      const wo = Math.min(1, Math.max(0, (dp - 2.2) / 1.6));
      if (w * wo > 0) d += w * wo * 0.26 * ridged(x * 0.8 - 2.0, y * 0.8 + 5.0, z * 0.8);
    }
    return Math.max(d, -lvOuter(x, y, z), -aoRoot(x, y, z) - RV.wall);
  };
  const tf = frame(tricuspid.axis, tricuspid.ref);
  const tvHole = ellipticCylinder(add(tricuspid.center, mul(tricuspid.axis, -0.2)), tricuspid.axis, tf.u, tricuspid.rx - 0.08, tricuspid.ry - 0.08, -2.0, 1.0);
  const pvHole = roundCone(add(pulmonary.center, mul(pulmonary.axis, -0.45)), add(pulmonary.center, mul(pulmonary.axis, 3.0)), pulmonary.radius - 0.08, pulmonary.radius - 0.08);
  const rvWall = (x, y, z) => Math.max(rvOuter(x, y, z), -rvCavity(x, y, z), -tvHole(x, y, z), -pvHole(x, y, z));

  // Septum mask: the part of the LV wall covered by the right ventricle.
  const septumMask = (x, y, z) => rvRaw(x, y, z) + 0.12;

  // ── Left atrium ──
  const laEll = ellipsoid(LA.center, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], LA.radii);
  // Left auricle: long, narrow, hooked and crenated — curls forward around the pulmonary trunk.
  const laAuricleBase = tubeChain([LA.auricleBase, LA.auricleMid, LA.auricleTip], [1.15, 0.9, 0.5], 0.5);
  const laAuricle = (x, y, z) => {
    const d = laAuricleBase(x, y, z);
    if (d > 0.5) return d;
    return d + 0.07 * Math.sin(x * 6.5 + noise3(x, y, z) * 2.0) * Math.sin(y * 5.0);
  };
  const laVestibule = roundCone(LA.center, add(mitral.center, mul(mitral.axis, -0.15)), 1.85, 1.75);
  const pvStubs = lm.pulmonaryVeins.map((v) => roundCone(LA.center, v.path[1], 1.0, v.r + 0.16));
  const laRaw0 = smoothUnion([laEll, laAuricle, laVestibule, ...pvStubs], 0.7);
  // Atria are trimmed at their AV valve plane — only locally (within the vestibule),
  // so the auricles that reach beyond the valve plane are kept.
  // Returns a field that is positive (= removed by smax) only beyond the plane AND near c.
  const localCut = (plane, c, r) => (x, y, z) => -Math.max(-plane(x, y, z), Math.hypot(x - c[0], y - c[1], z - c[2]) - r);
  const laCut = localCut(halfSpace(add(mitral.center, mul(mitral.axis, 0.05)), mitral.axis), mitral.center, 2.6);
  const laRaw = (x, y, z) => smax(smax(laRaw0(x, y, z), laCut(x, y, z), 0.25), -aoRoot(x, y, z), 0.35);

  // Fossa ovalis: shallow oval dent on the septal (right) face of the LA.
  const septCenter = projectToSurface(laRaw, comb([LA.center, 0.45], [RA.center, 0.55]));
  const septNormal = norm(sub(RA.center, LA.center));
  const fossa = ellipsoid(add(septCenter, mul(septNormal, 0.12)), [septNormal, norm(cross(septNormal, [0, 0, 1])), norm(cross(septNormal, cross(septNormal, [0, 0, 1])))], [0.26, 0.85, 0.7]);
  const laOuter = (x, y, z) => smax(laRaw(x, y, z), -fossa(x, y, z), 0.15);

  const pectinate = (x, y, z, axisDir, freq) => {
    const t = x * axisDir[0] + y * axisDir[1] + z * axisDir[2];
    const wob = noise3(x * 0.9, y * 0.9, z * 0.9) * 1.6;
    const s = Math.abs(Math.sin(t * freq + wob));
    return s ** 10;
  };
  const laCavity = (x, y, z) => {
    let d = laRaw0(x, y, z) + LA.wall;
    // Pectinate muscles are confined to the left auricle (smooth venous LA body).
    const da = laAuricle(x, y, z);
    if (da < 0.2 && d > -0.6) d += 0.17 * pectinate(x, y, z, norm(sub(LA.auricleTip, LA.auricleBase)), 5.2) * Math.min(1, -da / 0.4 + 0.5);
    return smax(smax(d, laCut(x, y, z) - 0.02, 0.2), -aoRoot(x, y, z) - LA.wall, 0.35);
  };
  const pvHoles = lm.pulmonaryVeins.map((v) => roundCone(LA.center, v.path[0], v.r - 0.12, v.r - 0.12));
  const mitralHoleA = ellipticCylinder(mitral.center, mitral.axis, mf.u, mitral.rx - 0.05, mitral.ry - 0.05, -0.8, 1.0);
  const laWall = (x, y, z) => {
    let d = Math.max(laOuter(x, y, z), -laCavity(x, y, z), -mitralHoleA(x, y, z));
    for (const h of pvHoles) d = Math.max(d, -h(x, y, z));
    return d;
  };

  // ── Right atrium ──
  const raEll = ellipsoid(RA.center, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], RA.radii);
  // Right auricle: broad, flattened triangular flap lying on the aortic root.
  const raAxA = norm(sub(RA.auricleTip, RA.auricleBase));
  const raAxC = norm(cross(raAxA, [0, 1, 0.35]));
  const raAxB = cross(raAxC, raAxA);
  const raAuricle = smoothUnion([
    tubeChain([RA.auricleBase, RA.auricleMid], [1.35, 0.95], 0.5),
    ellipsoid(comb([RA.auricleBase, 0.35], [RA.auricleTip, 0.65], [raAxB, 0.15]), [raAxA, raAxB, raAxC], [1.55, 1.05, 0.5]),
    roundCone(RA.auricleMid, RA.auricleTip, 0.62, 0.32),
  ], 0.55);
  const raVestibule = roundCone(RA.center, add(tricuspid.center, mul(tricuspid.axis, -0.15)), 1.75, 1.95);
  const svcEnd = lm.svc.path[lm.svc.path.length - 1];
  const ivcEnd = lm.ivc.path[lm.ivc.path.length - 1];
  const svcStub = roundCone(RA.center, add(svcEnd, [0, 1.1, 0]), 1.35, lm.svc.r + 0.16);
  const ivcStub = roundCone(RA.center, add(ivcEnd, [0.12, -0.9, -0.3]), 1.45, lm.ivc.r + 0.16);
  const raRaw0 = smoothUnion([raEll, raAuricle, raVestibule, svcStub, ivcStub], 0.75);
  const raCut = localCut(halfSpace(add(tricuspid.center, mul(tricuspid.axis, 0.05)), tricuspid.axis), tricuspid.center, 2.7);
  const raRaw = (x, y, z) => smax(smax(smax(raRaw0(x, y, z), raCut(x, y, z), 0.25), -aoRoot(x, y, z) - 0.15, 0.3), -laRaw(x, y, z) + 0.02, 0.15);
  const raOuter = raRaw;
  const crista = norm(sub(svcEnd, ivcEnd));
  const raCavity = (x, y, z) => {
    let d = raRaw0(x, y, z) + RA.wall;
    if (d > -0.6 && d < 0.6) {
      // Pectinate muscles run from the crista terminalis across the anterolateral wall
      // and into the auricle; the posterior (sinus venarum) wall is smooth.
      const lateral = (x - RA.center[0]) * -0.6 + (z - RA.center[2]) * 0.8;
      const w = Math.min(1, Math.max(0, (lateral + 0.4) / 1.2));
      if (w > 0) d += w * 0.12 * pectinate(x, y, z, crista, 4.6);
    }
    d = smax(d, raCut(x, y, z) - 0.02, 0.2);
    d = smax(d, -aoRoot(x, y, z) - 0.15 - RA.wall, 0.3);
    return Math.max(d, -laOuter(x, y, z));
  };
  const svcHole = roundCone(add(svcEnd, [0, -0.8, 0]), lm.svc.path[1], lm.svc.r - 0.14, lm.svc.r - 0.14);
  const ivcHole = roundCone(add(ivcEnd, [0.0, 0.8, 0.3]), lm.ivc.path[1], lm.ivc.r - 0.14, lm.ivc.r - 0.14);
  const tvHoleA = ellipticCylinder(tricuspid.center, tricuspid.axis, tf.u, tricuspid.rx - 0.05, tricuspid.ry - 0.05, -0.8, 1.0);
  const extraRaHoles = []; // coronary sinus ostium is appended by the coronary builder
  const raWall = (x, y, z) => {
    let d = Math.max(raOuter(x, y, z), -raCavity(x, y, z), -svcHole(x, y, z), -ivcHole(x, y, z), -tvHoleA(x, y, z));
    for (const h of extraRaHoles) d = Math.max(d, -h(x, y, z));
    return d;
  };

  // ── Epicardial envelope (for coronary routing and the pericardium) ──
  const epicardium = (x, y, z) => Math.min(lvOuter(x, y, z), rvOuter(x, y, z), laOuter(x, y, z), raOuter(x, y, z));

  // ── Bounds ──
  const bLV = boxExpand(boxEllipsoid(B, lvBasis, LV.outer.radii), 0.3);
  bLV.max = bLV.max.map((v, i) => Math.max(v, aortic.center[i] + 1.6));
  bLV.min = bLV.min.map((v, i) => Math.min(v, aortic.center[i] - 1.6));
  const bRV = boxExpand(boxUnion(boxEllipsoid(RV.center, lvBasis, RV.radii), boxPoints([RV.outflowStart, pulmonary.center, tricuspid.center], 2.2)), 0.3);
  const bLA = boxExpand(boxUnion(boxEllipsoid(LA.center, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], LA.radii), boxPoints([LA.auricleTip, LA.auricleMid, mitral.center, ...lm.pulmonaryVeins.map((v) => v.path[1])], 1.3)), 0.3);
  const bRA = boxExpand(boxUnion(boxEllipsoid(RA.center, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], RA.radii), boxPoints([RA.auricleTip, tricuspid.center, add(svcEnd, [0, 1.4, 0]), add(ivcEnd, [0, -1.2, 0])], 1.6)), 0.3);

  return {
    sOf, lvOuter, lvCavity, lvWall, septumMask, rvRaw, rvOuter, rvCavity, rvWall,
    laOuter, laCavity, laWall, raOuter, raCavity, raWall, extraRaHoles,
    epicardium, aoRoot, paRoot, fossaCenter: septCenter, septNormal,
    bounds: { LV: bLV, RV: bRV, LA: bLA, RA: bRA },
    mitralFrame: mf, tricuspidFrame: tf,
  };
}
