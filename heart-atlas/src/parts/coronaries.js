// Coronary arteries and cardiac veins, routed in the real anatomical grooves.
//
// The grooves are *found* from the chamber fields rather than hand-placed:
//   • Interventricular grooves = the edges of the RV crescent on the LV surface
//     (anterior edge → LAD/great cardiac vein, inferior edge → PDA/middle cardiac vein).
//   • Coronary (atrioventricular) sulcus = a ring around the base, sampled on the
//     epicardial envelope (RCA + small cardiac vein on the right, LCx + great cardiac
//     vein / coronary sinus on the left and posteriorly).
// Control points are then projected onto the epicardium and lifted by the vessel radius,
// so every vessel lies on the surface and follows the heart as it beats.
// Right-dominant circulation (~70–80% of people): the PDA arises from the RCA.

import { add, sub, mul, norm, dot, lerp, comb, frame } from '../core/vec.js';
import { projectToSurface, surfaceNormal, roundCone } from '../core/sdf.js';
import { GeoBuilder, tube, curveFrom } from '../core/geometry.js';
import { marchOut } from './chambers.js';

export function computeGrooves(lm, fields) {
  const { e1, eL, n } = lm;
  const ant = [], post = [];
  for (let s = 0.35; s <= lm.L - 0.9; s += 0.3) {
    const c = lm.lv(s, 0, 0);
    const smp = [];
    const N = 144;
    for (let k = 0; k < N; k++) {
      const th = (k / N) * Math.PI * 2;
      const dir = norm(comb([e1, Math.cos(th)], [eL, Math.sin(th)]));
      const p = marchOut(fields.lvOuter, c, dir, 0.03, 7);
      const q = add(p, mul(dir, 0.3));
      smp.push({ th, p, cov: fields.rvOuter(q[0], q[1], q[2]) < 0 });
    }
    const tr = [];
    for (let k = 0; k < N; k++) if (smp[k].cov !== smp[(k + 1) % N].cov) tr.push(lerp(smp[k].p, smp[(k + 1) % N].p, 0.5));
    if (tr.length < 2) break;
    tr.sort((a, b) => dot(sub(b, c), e1) - dot(sub(a, c), e1));
    ant.push({ s, p: tr[0] });
    post.push({ s, p: tr[tr.length - 1] });
  }
  // AV ring
  const cav = add(lerp(lm.mitral.center, lm.tricuspid.center, 0.5), mul(n, 0.15));
  const ring = (th, ds = 0) => {
    const c = add(cav, mul(n, ds));
    const dir = norm(comb([e1, Math.cos(th)], [eL, Math.sin(th)]));
    return marchOut(fields.epicardium, c, dir, 0.03, 9);
  };
  const angleOf = (p) => Math.atan2(dot(sub(p, cav), eL), dot(sub(p, cav), e1));
  const crux = angleOf(post[0].p);
  return { ant, post, ring, angleOf, crux, cav };
}

/** Project a control polyline onto a surface field, lift it by `lift`, and smooth. */
export function surfaceRoute(field, ctrl, { n = 60, lift = 0.15, smooth = 6 } = {}) {
  const c = curveFrom(ctrl, 0.5);
  let pts = [];
  for (let i = 0; i <= n; i++) {
    const p = c.getPointAt(i / n);
    pts.push([p.x, p.y, p.z]);
  }
  const place = (p) => {
    const q = projectToSurface(field, p, 10);
    return add(q, mul(surfaceNormal(field, q), lift));
  };
  pts = pts.map(place);
  for (let it = 0; it < smooth; it++) {
    for (let k = 0; k < 2; k++) pts = pts.map((p, i) => (i === 0 || i === pts.length - 1 ? p : lerp(p, lerp(pts[i - 1], pts[i + 1], 0.5), 0.6)));
    pts = pts.map((p, i) => (i === 0 || i === pts.length - 1 ? p : place(p)));
  }
  return pts;
}

const taper = (r0, r1) => (t) => r0 + (r1 - r0) * t;

export function routeCoronaries(lm, fields) {
  const G = computeGrooves(lm, fields);
  const epi = fields.epicardium;
  const ao = lm.aortic;
  const vf = frame(ao.axis, [0, 0, 1]);
  const radial = (deg) => norm(comb([vf.u, Math.cos((deg * Math.PI) / 180)], [vf.v, Math.sin((deg * Math.PI) / 180)]));
  // ostia sit in the upper third of the right and left sinuses of Valsalva
  const ostR = comb([ao.center, 1], [ao.axis, 1.05], [radial(0), 1.5]);
  const ostL = comb([ao.center, 1], [ao.axis, 1.05], [radial(120), 1.5]);
  const R = {}; // routes: id → { pts, r(t) }
  const art = (id, pts, r0, r1, part) => (R[id] = { pts, r: taper(r0, r1), part, kind: 'artery' });
  const vein = (id, pts, r0, r1, part) => (R[id] = { pts, r: taper(r0, r1), part, kind: 'vein' });
  const arc = (a0, a1, ds = 0, step = 0.18) => {
    const out = [];
    const nS = Math.max(2, Math.ceil(Math.abs(a1 - a0) / step));
    for (let i = 0; i <= nS; i++) out.push(G.ring(a0 + ((a1 - a0) * i) / nS, ds));
    return out;
  };
  const lift = (r) => r * 0.7;
  // angle around the LV long axis (diagonals / marginals are laid out in this frame)
  const angLV = (p) => Math.atan2(dot(sub(p, lm.B), lm.eL), dot(sub(p, lm.B), lm.e1));
  const onLV = (s, a, field = fields.lvOuter) => marchOut(field, lm.lv(s, 0, 0), norm(comb([lm.e1, Math.cos(a)], [lm.eL, Math.sin(a)])), 0.03, 8);

  // ── RCA: right sinus → right AV groove → acute margin → crux ──
  const cruxA = G.crux;
  const rcaCtrl = [ostR, comb([ostR, 1], [radial(0), 0.6], [[-1, 0, 0], 0.35]), ...arc(-0.42, cruxA, -0.05)];
  const rca = surfaceRoute(epi, rcaCtrl, { n: 110, lift: lift(0.18) });
  rca[0] = ostR;
  art('rca', rca, 0.2, 0.13, 'rca');
  // PDA: crux → down the posterior interventricular groove
  const pdaCtrl = [rca[rca.length - 1], ...G.post.filter((g) => g.s < lm.L * 0.78).map((g) => g.p)];
  art('pda', surfaceRoute(epi, pdaCtrl, { n: 60, lift: lift(0.11) }), 0.12, 0.05, 'pda');
  // Posterolateral branch beyond the crux (onto the inferior LV)
  {
    const a = cruxA - 0.45;
    const pts = [rca[rca.length - 1], G.ring(cruxA - 0.25, 0.25), G.ring(a, 0.9), lm.lv(2.8, Math.cos(a) * 3.5, Math.sin(a) * 3.5)];
    art('rcaPL', surfaceRoute(epi, pts, { n: 30, lift: lift(0.09) }), 0.1, 0.05, 'rca');
  }
  // Acute marginal: from the RCA at the acute margin down the RV free wall
  {
    const a = -1.3;
    const start = rca.reduce((b, p) => (Math.abs(G.angleOf(p) - a) < Math.abs(G.angleOf(b) - a) ? p : b), rca[10]);
    const dir = norm(comb([lm.e1, Math.cos(a)], [lm.eL, Math.sin(a)]));
    const pts = [start, ...[1.4, 2.6, 3.8, 5.0].map((s) => marchOut(epi, lm.lv(s, 0, 0), norm(add(dir, mul(lm.e1, 0.25))), 0.03, 8))];
    art('acuteMarginal', surfaceRoute(epi, pts, { n: 40, lift: lift(0.1) }), 0.11, 0.05, 'rca');
  }
  // Conus branch: first RCA branch, over the infundibulum
  {
    const p0 = rca[4];
    const tgt = comb([lm.pulmonary.center, 1], [lm.pulmonary.axis, -0.9], [lm.e1, 0.9], [[1, 0, 0], 0.3]);
    art('conus', surfaceRoute(epi, [p0, lerp(p0, tgt, 0.5), tgt], { n: 20, lift: lift(0.07) }), 0.08, 0.04, 'rca');
  }
  // SA nodal artery: proximal RCA → up behind the right auricle → SVC–RA junction
  {
    const p0 = rca[7];
    const pts = [p0, [-2.15, 3.3, 0.35], [-2.65, 3.75, -0.45], lm.conduction.saNode];
    art('saNodal', surfaceRoute(epi, pts, { n: 26, lift: lift(0.06) }), 0.07, 0.045, 'rca');
  }

  // ── Left main → LAD + LCx ──
  const lmEnd0 = G.ring(0.95, -0.05);
  const bif = projectToSurface(epi, lerp(G.ant[0].p, lmEnd0, 0.5));
  const lmca = surfaceRoute(epi, [ostL, comb([ostL, 1], [radial(120), 0.5], [lm.eL, 0.45]), bif], { n: 24, lift: lift(0.24), smooth: 1 });
  lmca[0] = ostL;
  art('lmca', lmca, 0.25, 0.23, 'leftMainCoronary');
  const B0 = lmca[lmca.length - 1];
  // LAD: anterior interventricular groove, wrapping the apex onto the inferior surface
  const apexWrap = lm.lv(lm.L - 0.05, -0.9, -0.6);
  const lad = surfaceRoute(epi, [B0, ...G.ant.map((g) => g.p), apexWrap], { n: 120, lift: lift(0.19) });
  art('lad', lad, 0.21, 0.06, 'lad');
  // Diagonals across the anterolateral LV wall
  for (const [k, sFrac, spread] of [[1, 0.2, 0.62], [2, 0.42, 0.55], [3, 0.6, 0.45]]) {
    const i0 = Math.floor(lad.length * sFrac);
    const p0 = lad[i0];
    const th = angLV(p0);
    const s0 = fields.sOf(...p0);
    const pts = [p0];
    for (const [ds, dth] of [[0.9, 0.22], [2.0, 0.42], [3.1, 0.58]]) pts.push(onLV(Math.min(s0 + ds, lm.L - 0.9), th + dth * spread * 1.6));
    art('diag' + k, surfaceRoute(epi, pts, { n: 30, lift: lift(0.1) }), k === 1 ? 0.13 : 0.1, 0.045, 'lad');
  }
  // LCx: left AV groove toward the posterolateral wall
  const lcx = surfaceRoute(epi, [B0, ...arc(1.05, 2.55, -0.05)], { n: 70, lift: lift(0.17) });
  art('lcx', lcx, 0.18, 0.09, 'circumflex');
  // Obtuse marginals down the lateral LV wall
  const thLAD = angLV(G.ant[Math.floor(G.ant.length / 2)].p);
  for (const [k, da, len] of [[1, 0.95, 5.4], [2, 1.5, 4.8]]) {
    const a = thLAD + da;
    const p0 = lcx.reduce((b, p) => (Math.abs(angLV(p) - a) < Math.abs(angLV(b) - a) ? p : b), lcx[5]);
    const pts = [p0, ...[1.3, 2.6, len].map((s, i) => onLV(s, a + 0.08 * (i + 1)))];
    art('om' + k, surfaceRoute(epi, pts, { n: 34, lift: lift(0.11) }), 0.12, 0.05, 'circumflex');
  }

  // ── Veins ──
  // Coronary sinus ostium: RA floor between the IVC orifice and the tricuspid valve.
  const ivcEnd = lm.ivc.path[lm.ivc.path.length - 1];
  const csOst = comb([ivcEnd, 0.42], [lm.tricuspid.center, 0.58], [[0, 0, -1], 0.35], [[0, 1, 0], -0.15]);
  const csStart = G.ring(2.45, -0.4);
  const cruxV = G.ring(cruxA + Math.PI * 2 - 0.05, -0.35);
  const cs = surfaceRoute(epi, [csStart, ...arc(2.65, cruxA + Math.PI * 2 - 0.12, -0.42), cruxV], { n: 60, lift: lift(0.38) });
  cs.push(lerp(cs[cs.length - 1], csOst, 0.55), csOst);
  // end the sinus exactly at the RA endocardium (its ostium opens into the RA)
  const enter = cs.findIndex((p, i) => i > cs.length - 6 && fields.raCavity(p[0], p[1], p[2]) < -0.05);
  if (enter > 0) cs.length = enter + 1;
  vein('cs', cs, 0.36, 0.45, 'coronarySinus');
  fields.extraRaHoles.push(roundCone(lerp(cs[cs.length - 3], csOst, 0.2), add(csOst, mul(norm(sub(csOst, cs[cs.length - 3])), 0.6)), 0.3, 0.3));
  // Great cardiac vein: apex → anterior IV groove (beside the LAD) → left AV groove → CS
  const gcvAnt = G.ant.filter((g) => g.s < lm.L * 0.82).reverse().map((g) => {
    const th = G.angleOf(g.p) + 0.1;
    return marchOut(epi, lm.lv(g.s, 0, 0), norm(comb([lm.e1, Math.cos(th)], [lm.eL, Math.sin(th)])), 0.03, 7);
  });
  const gcv = surfaceRoute(epi, [...gcvAnt, G.ring(0.85, -0.35), ...arc(1.15, 2.45, -0.4)], { n: 110, lift: lift(0.18) });
  vein('gcv', gcv, 0.11, 0.3, 'greatCardiacVein');
  // Middle cardiac vein beside the PDA
  const mcvPts = G.post.filter((g) => g.s < lm.L * 0.75).reverse().map((g) => {
    const th = G.angleOf(g.p) - 0.12;
    return marchOut(epi, lm.lv(g.s, 0, 0), norm(comb([lm.e1, Math.cos(th)], [lm.eL, Math.sin(th)])), 0.03, 7);
  });
  vein('mcv', surfaceRoute(epi, [...mcvPts, cs[cs.length - 4]], { n: 60, lift: lift(0.16) }), 0.09, 0.22, 'middleCardiacVein');
  // Small cardiac vein: right AV groove → CS terminal part
  vein('scv', surfaceRoute(epi, [...arc(-1.25, cruxA - 0.08, -0.3), cs[cs.length - 4]], { n: 40, lift: lift(0.1) }), 0.07, 0.13, 'smallCardiacVein');
  // Posterior vein of the LV
  {
    const a = 2.85;
    const dir = norm(comb([lm.e1, Math.cos(a)], [lm.eL, Math.sin(a)]));
    const pts = [4.8, 3.2, 1.6].map((s) => marchOut(epi, lm.lv(s, 0, 0), dir, 0.03, 8));
    vein('plv', surfaceRoute(epi, [...pts, cs[Math.floor(cs.length * 0.25)]], { n: 36, lift: lift(0.12) }), 0.08, 0.16, 'posteriorVeinLV');
  }
  // Oblique vein of the LA (Marshall)
  vein('marshall', surfaceRoute(epi, [[2.3, 3.6, -4.0], [2.1, 2.5, -3.9], cs[2]], { n: 20, lift: lift(0.05) }), 0.045, 0.06, 'posteriorVeinLV');
  // Anterior cardiac veins: RV anterior wall → across the right AV groove → RA
  for (const [k, a] of [[1, -0.35], [2, -0.7], [3, -1.0]]) {
    const dir = norm(comb([lm.e1, Math.cos(a)], [lm.eL, Math.sin(a)]));
    const pts = [marchOut(epi, lm.lv(1.9, 0, 0), dir, 0.03, 8), marchOut(epi, lm.lv(0.9, 0, 0), dir, 0.03, 8), G.ring(a - 0.05, -0.15), G.ring(a - 0.1, -0.9)];
    vein('acv' + k, surfaceRoute(epi, pts, { n: 26, lift: lift(0.07) }), 0.05, 0.08, 'anteriorCardiacVeins');
  }

  // Groove samples for epicardial fat
  const groove = [];
  const addG = (pts, r, w) => pts.forEach((p, i) => i % 2 === 0 && groove.push({ p, r, w }));
  addG(rca, 1.1, 1); addG(lcx, 1.0, 0.95); addG(lad.slice(0, Math.floor(lad.length * 0.8)), 0.85, 0.9);
  addG(R.pda.pts, 0.8, 0.85); addG(cs, 1.0, 1); addG(gcv, 0.8, 0.85); addG(R.acuteMarginal.pts, 0.8, 0.7);
  addG(lmca, 0.9, 1); addG(R.om1.pts.slice(0, 15), 0.6, 0.5);
  return { routes: R, grooves: G, groove, ostia: { right: ostR, left: ostL } };
}

/** Build tube geometry per catalog part from the routes. */
export function buildCoronaryTubes(routes) {
  const byPart = {};
  for (const id in routes) {
    const rt = routes[id];
    const b = (byPart[rt.part] ||= new GeoBuilder());
    tube(b, rt.pts, {
      radius: (t) => rt.r(t),
      segments: Math.max(16, Math.min(160, rt.pts.length * 2)),
      radial: rt.r(0) > 0.2 ? 16 : 10,
      capStart: true, capEnd: true,
      kind: rt.kind === 'vein' ? 2 : 0,
    });
  }
  return byPart;
}
