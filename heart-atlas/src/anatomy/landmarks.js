// Anatomical reference frame and landmarks.
//
// Body coordinates, centimetres:  +X = patient's LEFT,  +Y = SUPERIOR,  +Z = ANTERIOR.
// (A camera on +Z looking at the origin gives the classic anterior atlas view, with the
// patient's left on the viewer's right.)
//
// Everything is derived from a handful of measured adult dimensions:
//   • LV long axis (base → epicardial apex) ≈ 8 cm, directed anteriorly, inferiorly
//     and to the left (apex in the 5th left intercostal space).
//   • LV end-diastolic internal diameter ≈ 4.6 cm, free-wall thickness ≈ 0.9–1.0 cm.
//   • RV free wall ≈ 0.4 cm; atrial walls ≈ 0.2–0.3 cm.
//   • Annuli: mitral ≈ 3.1 × 2.4 cm (D-shaped, saddle), tricuspid ≈ 3.4 cm,
//     aortic ≈ 2.3 cm, pulmonary ≈ 2.4 cm.
//   • The pulmonary valve lies anterior, superior and to the left of the aortic valve;
//     the aortic valve is wedged between the mitral and tricuspid valves
//     (aorto-mitral fibrous continuity).
//   • The septal tricuspid leaflet inserts ~0.8–1 cm more apically than the mitral.

import { add, sub, mul, norm, cross, dot, comb, frame, len } from '../core/vec.js';

export function createLandmarks() {
  const B = [0.6, 0.8, -0.6]; // centre of the LV base (fibrous skeleton level)
  const APEX = [4.2, -5.6, 2.6]; // epicardial apex
  const n = norm(sub(APEX, B)); // LV long axis, base → apex
  const L = len(sub(APEX, B));
  const e1 = norm(sub([0, 0, 1], mul(n, n[2]))); // "anterior" within the base plane
  const eL = norm(cross(e1, n)); // "left" within the base plane
  const lv = (s, a, b) => comb([B, 1], [n, s], [e1, a], [eL, b]);

  // ── Valves ────────────────────────────────────────────────────────────
  const aortic = {
    center: lv(-1.25, 1.4, -1.0),
    axis: norm(comb([n, -1], [e1, 0.6], [[-1, 0, 0], 0.12])), // up, right, slightly anterior
    radius: 1.15,
  };
  const mitral = {
    center: lv(-1.05, -0.75, 0.55),
    rx: 1.55, // commissural (anterolateral ↔ posteromedial)
    ry: 1.2, // antero-posterior
  };
  mitral.axis = norm(add(mul(n, 0.75), mul(norm(sub(APEX, mitral.center)), 0.25)));
  // In-plane reference: from posteromedial to anterolateral commissure.
  mitral.ref = norm(comb([eL, 0.8], [e1, 0.35], [n, 0.0]));
  // "anterior" of the mitral frame points to the aortic valve (anterior leaflet side).
  mitral.anteriorDir = norm(sub(aortic.center, mitral.center));

  const tricuspid = {
    center: lv(0.15, 0.6, -3.25),
    rx: 1.75,
    ry: 1.45,
  };
  tricuspid.axis = norm(comb([n, 0.62], [eL, 0.66], [e1, 0.25]));
  tricuspid.ref = norm(comb([n, 0.2], [e1, 1], [eL, -0.3])); // toward the anterior leaflet

  const pulmonary = {
    center: add(aortic.center, [1.35, 1.25, 1.55]),
    axis: norm([0.32, 0.78, -0.54]), // up, back and to the left
    radius: 1.2,
  };

  // ── Chamber parameters ───────────────────────────────────────────────
  const LV = {
    outer: { center: B, radii: [L, 3.2, 3.4], roof: -1.0 },
    cavity: { center: lv(0.05, 0, 0), radii: [7.15, 2.25, 2.4], roof: -0.42 },
    wall: 0.95,
  };
  const RV = {
    center: lv(2.35, 1.75, -1.35),
    radii: [4.65, 2.9, 3.55],
    roof: -0.55,
    wall: 0.38,
    outflowStart: lv(1.4, 2.7, -0.7),
  };
  const RA = {
    center: [-3.45, 1.0, -1.4],
    radii: [2.3, 2.65, 2.2],
    wall: 0.27,
    auricleBase: [-3.05, 2.2, 0.15],
    auricleMid: [-2.85, 2.85, 1.7],
    auricleTip: [-1.6, 2.7, 2.6],
  };
  const LA = {
    center: [0.45, 2.3, -3.2],
    radii: [2.75, 1.8, 1.7],
    wall: 0.27,
    auricleBase: [2.2, 2.45, -1.95],
    auricleMid: [3.35, 2.7, -0.75],
    auricleTip: [2.7, 2.95, 0.55],
  };

  // ── Great vessels (centre-lines) ─────────────────────────────────────
  const ao = aortic.center, aa = aortic.axis;
  const aorta = {
    path: [
      add(ao, mul(aa, -0.25)),
      add(ao, mul(aa, 1.3)),
      add(ao, mul(aa, 2.6)), // sinotubular junction
      [-2.3, 5.3, 0.55],
      [-1.95, 6.9, -0.2],
      [-0.9, 8.0, -1.35],
      [0.45, 8.35, -2.75],
      [1.55, 7.75, -4.55],
      [1.95, 6.2, -6.0],
      [1.95, 3.6, -6.55],
      [1.8, 0.0, -6.65],
      [1.6, -4.5, -6.4],
    ],
    radius: [1.18, 1.38, 1.42, 1.48, 1.46, 1.38, 1.3, 1.22, 1.15, 1.1, 1.06, 1.02],
    branches: [
      { id: 'brachiocephalic', from: [-1.15, 7.95, -1.15], path: [[-1.35, 8.9, -1.05], [-2.05, 10.4, -0.8], [-2.75, 11.6, -0.55]], r: [0.62, 0.6, 0.56] },
      { id: 'leftCarotid', from: [-0.05, 8.3, -2.05], path: [[0.0, 9.3, -1.95], [0.15, 10.8, -1.75], [0.3, 12.0, -1.6]], r: [0.42, 0.4, 0.38] },
      { id: 'leftSubclavian', from: [0.95, 8.15, -3.45], path: [[1.25, 9.1, -3.5], [2.15, 10.3, -3.35], [3.3, 10.9, -3.1]], r: [0.5, 0.48, 0.45] },
    ],
  };
  const pc = pulmonary.center, pa = pulmonary.axis;
  const pulmonaryTrunk = {
    path: [
      add(pc, mul(pa, -0.25)),
      add(pc, mul(pa, 1.4)),
      [1.35, 4.95, 0.35],
      [1.3, 5.75, -0.85],
      [1.05, 6.0, -1.75],
    ],
    radius: [1.22, 1.38, 1.4, 1.36, 1.28],
    right: { path: [[1.05, 6.0, -1.75], [-0.4, 5.95, -2.35], [-2.3, 5.75, -2.55], [-4.6, 5.35, -2.35]], r: [1.05, 1.0, 0.95, 0.9] },
    left: { path: [[1.05, 6.0, -1.75], [2.35, 6.15, -2.55], [3.6, 5.95, -3.25], [4.9, 5.5, -3.55]], r: [1.02, 0.98, 0.92, 0.86] },
  };
  const svc = { path: [[-3.15, 11.0, -1.35], [-3.2, 8.5, -1.4], [-3.3, 5.6, -1.5], [-3.35, 3.0, -1.55]], r: 1.0 };
  const ivc = { path: [[-2.15, -6.4, -3.0], [-2.35, -4.4, -2.75], [-2.65, -2.4, -2.25], [-2.95, -0.75, -1.85]], r: 1.15 };
  const pulmonaryVeins = [
    { id: 'rspv', path: [[-4.6, 3.55, -4.8], [-3.0, 3.05, -4.55], [-1.6, 2.75, -4.05], [-0.9, 2.6, -3.5]], r: 0.68 },
    { id: 'ripv', path: [[-4.5, 0.75, -5.0], [-2.95, 1.05, -4.7], [-1.5, 1.45, -4.15], [-0.85, 1.75, -3.6]], r: 0.64 },
    { id: 'lspv', path: [[5.2, 3.55, -4.6], [3.95, 3.05, -4.4], [2.75, 2.75, -4.0], [1.9, 2.6, -3.5]], r: 0.66 },
    { id: 'lipv', path: [[5.1, 0.75, -4.85], [3.9, 1.1, -4.55], [2.7, 1.5, -4.1], [1.85, 1.75, -3.6]], r: 0.62 },
  ];

  // ── Conduction landmarks ──────────────────────────────────────────────
  const conduction = {
    saNode: [-3.05, 3.45, -0.35], // SVC–RA junction, crista terminalis (sulcus terminalis)
    avNode: null, // derived in parts/conduction.js from the tricuspid annulus (triangle of Koch)
  };

  return {
    B, APEX, n, L, e1, eL, lv,
    aortic, mitral, tricuspid, pulmonary,
    LV, RV, RA, LA,
    aorta, pulmonaryTrunk, svc, ivc, pulmonaryVeins,
    conduction,
    /** Approximate centre of mass of the cardiac silhouette (for explode/camera). */
    heartCenter: lv(2.6, -0.2, -0.9),
    valveFrame(v) {
      return frame(v.axis, v.ref || [0, 1, 0]);
    },
    dot,
  };
}
