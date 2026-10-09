// Cardiac conduction system with per-vertex activation times (aAct encoding):
//   0…999       ms after P-wave onset (SA node, internodal tracts, Bachmann's bundle)
//   1400…1900   1500 + ms relative to QRS onset (His −30…−12, bundles −12…0, Purkinje 0…25)
//   2000…3000   AV node: fraction through the node; the shader maps it onto the current
//               PR interval, so the AV delay stretches in 1st-degree AV block.

import { add, sub, mul, norm, lerp, comb, dist, dot, rng, cross } from '../core/vec.js';
import { projectToSurface, surfaceNormal } from '../core/sdf.js';
import { GeoBuilder, tube, curveFrom } from '../core/geometry.js';
import { marchOut } from './chambers.js';
import { surfaceRoute } from './coronaries.js';

/** Route on the *inside* of a cavity wall (endocardium), offset into the lumen. */
function endoRoute(cavity, ctrl, { n = 40, inset = 0.06, smooth = 2 } = {}) {
  // cavity field is negative inside the lumen; its zero set is the endocardium.
  return surfaceRoute(cavity, ctrl, { n, lift: -inset, smooth });
}

export function buildConduction(lm, fields, pm, valves) {
  const out = {};
  const svcEnd = lm.svc.path[lm.svc.path.length - 1];
  const ivcEnd = lm.ivc.path[lm.ivc.path.length - 1];
  const crista = norm(sub(svcEnd, ivcEnd));
  const SA = projectToSurface(fields.raOuter, lm.conduction.saNode);

  // ── SA node: subepicardial spindle along the crista terminalis ──
  {
    const b = new GeoBuilder();
    const pts = surfaceRoute(fields.raOuter, [add(SA, mul(crista, 0.55)), SA, add(SA, mul(crista, -0.9))], { n: 16, lift: -0.04, smooth: 1 });
    tube(b, pts, { radius: (t) => 0.08 + 0.17 * Math.sin(Math.PI * Math.min(1, t * 1.25)) ** 0.7, segments: 24, radial: 14, capStart: true, capEnd: true, act: (t) => Math.abs(t - 0.35) * 8 });
    out.saNode = b;
  }

  // ── AV node: triangle of Koch, above the septal tricuspid annulus near the CS ostium ──
  const tri = valves.tricuspid;
  const septal = tri.cfg.leaflets.findIndex((l) => l.name === 'septal');
  const Ls = tri.cfg.leaflets[septal];
  const sepAnn = tri.annulus(Ls.a0 + (Ls.a1 - Ls.a0) * 0.35);
  const avGuess = comb([sepAnn, 1], [tri.f.w, -0.55], [norm(sub(lm.RA.center, sepAnn)), 0.3]);
  const AV = projectToSurface(fields.raCavity, avGuess);
  const AVn = surfaceNormal(fields.raCavity, AV);
  const AVc = add(AV, mul(AVn, 0.12)); // embedded just under the endocardium
  {
    const b = new GeoBuilder();
    const ax = norm(sub(lm.aortic.center, AVc));
    tube(b, [add(AVc, mul(ax, -0.35)), AVc, add(AVc, mul(ax, 0.35))], { radius: (t) => 0.12 + 0.12 * Math.sin(Math.PI * t), segments: 14, radial: 12, capStart: true, capEnd: true, act: (t) => 2000 + t * 1000 });
    out.avNode = b;
  }

  // ── Internodal pathways + Bachmann's bundle ──
  {
    const b = new GeoBuilder();
    const raEndo = (ctrl, n) => endoRoute(fields.raCavity, ctrl, { n, inset: 0.05 });
    const antTop = [-1.95, 2.95, -0.75];
    const ant = raEndo([SA, [-2.45, 3.35, -0.3], antTop, lerp(antTop, AV, 0.5), AV], 40);
    const mid = raEndo([SA, [-2.9, 2.6, -2.1], add(fields.fossaCenter, [0, 0.9, 0]), add(fields.fossaCenter, [0.1, -0.8, 0.2]), AV], 40);
    const post = raEndo([SA, comb([SA, 1], [crista, -1.6], [[-1, 0, 0], 0.3]), comb([ivcEnd, 1], [[0, 1, 0], 0.9], [[1, 0, 0], 0.4]), AV], 40);
    const actFor = (pts) => { const L = pts.reduce((s, p, i) => (i ? s + dist(p, pts[i - 1]) : 0), 0); return (t) => 4 + (t * L) / 0.12; };
    for (const pts of [ant, mid, post]) tube(b, pts, { radius: () => 0.055, segments: 50, radial: 8, capStart: true, capEnd: true, act: actFor(pts) });
    // Bachmann's bundle: subepicardial, across the anterior interatrial groove to the LA
    const bach = surfaceRoute(fields.epicardium, [antTop, [-0.6, 3.6, -1.45], [0.9, 3.75, -2.0], [2.15, 3.15, -2.15]], { n: 34, lift: 0.02 });
    tube(b, bach, { radius: () => 0.07, segments: 40, radial: 8, capStart: true, capEnd: true, act: (t) => 10 + t * 40 });
    out.internodalTracts = b;
  }

  // ── His bundle: through the central fibrous body to the crest of the muscular septum ──
  const sepDir = norm(sub(lm.RV.center, lm.B).map((v, i) => v - lm.n[i] * dot(sub(lm.RV.center, lm.B), lm.n)));
  const crestS = 0.15;
  const pLV = marchOut(fields.lvCavity, lm.lv(crestS, 0.4, 0), sepDir); // LV septal endocardium
  const pRV = marchOut(fields.lvOuter, pLV, sepDir); // RV face of the septum
  const crest = lerp(pLV, pRV, 0.5);
  {
    const b = new GeoBuilder();
    tube(b, [AVc, lerp(AVc, crest, 0.45), crest], { radius: () => 0.1, segments: 20, radial: 10, capStart: true, capEnd: true, act: (t) => 1500 - 30 + 18 * t });
    out.hisBundle = b;
  }

  // ── Left bundle branch: fans over the LV septal endocardium ──
  const lvEndo = (ctrl, n) => endoRoute(fields.lvCavity, ctrl, { n, inset: 0.07 });
  const alPM = pm.lv.find((m) => m.id === 'anterolateral');
  const pmPM = pm.lv.find((m) => m.id === 'posteromedial');
  const sepAt = (s, side) => marchOut(fields.lvCavity, lm.lv(s, 0, 0), norm(comb([sepDir, 1], [lm.e1, side])), 0.03, 4);
  const lbbStart = add(pLV, mul(sepDir, -0.05));
  const lbb = {
    anterior: lvEndo([lbbStart, sepAt(1.2, 0.8), sepAt(2.6, 1.3), lerp(alPM.base, alPM.tip, 0.2)], 46),
    posterior: lvEndo([lbbStart, sepAt(1.2, -0.7), sepAt(2.4, -1.2), lerp(pmPM.base, pmPM.tip, 0.2)], 46),
    septal: lvEndo([lbbStart, sepAt(1.6, 0.05), sepAt(3.6, 0.1), sepAt(5.0, 0.0)], 40),
  };
  {
    const b = new GeoBuilder();
    tube(b, [crest, lbbStart, lbb.anterior[2]], { radius: () => 0.09, segments: 10, radial: 8, capStart: true, act: (t) => 1500 - 12 + 4 * t });
    for (const k in lbb) tube(b, lbb[k], { radius: (t) => 0.075 - 0.03 * t, segments: 40, radial: 8, capStart: true, capEnd: true, act: (t) => 1500 - 8 + 10 * t });
    out.leftBundle = b;
  }

  // ── Right bundle branch: RV septal surface → moderator band → anterior papillary muscle ──
  const rbbPts = [crest, ...[0.9, 1.9, 2.9].map((s) => {
    const q = marchOut(fields.lvOuter, lm.lv(s, 0, 0), norm(comb([sepDir, 1], [lm.e1, 0.35])), 0.03, 5);
    return add(q, mul(surfaceNormal(fields.lvOuter, q), 0.06));
  }), pm.moderator.septal];
  const rbb = curveFrom(rbbPts);
  const rbbAll = [...Array(31)].map((_, i) => { const p = rbb.getPointAt(i / 30); return [p.x, p.y, p.z]; });
  const band = [...Array(11)].map((_, i) => lerp(pm.moderator.septal, pm.moderator.end, i / 10));
  {
    const b = new GeoBuilder();
    tube(b, [...rbbAll, ...band.slice(1)], { radius: () => 0.065, segments: 50, radial: 8, capStart: true, capEnd: true, act: (t) => 1500 - 12 + 16 * t });
    out.rightBundle = b;
  }

  // ── Purkinje network: branching random walks on the endocardium ──
  {
    const b = new GeoBuilder();
    const R = rng(7);
    const grow = (cavity, seed, startAct, depth, dirHint) => {
      const pts = [seed];
      let p = seed;
      let dir = dirHint;
      const steps = 5 + Math.floor(R() * 4);
      for (let i = 0; i < steps; i++) {
        const jitter = [R() - 0.5, R() - 0.5, R() - 0.5];
        dir = norm(comb([dir, 1], [jitter, 0.9], [lm.n, 0.25]));
        const q = projectToSurface(cavity, add(p, mul(dir, 0.55)), 8);
        const nn = surfaceNormal(cavity, q);
        p = add(q, mul(nn, -0.05));
        pts.push(p);
      }
      if (pts.length > 2) tube(b, pts, { radius: () => 0.03, segments: pts.length * 3, radial: 5, capEnd: true, act: (t) => startAct + t * 12 });
      if (depth > 0) for (let k = 0; k < 2; k++) {
        const j = 2 + Math.floor(R() * (pts.length - 2));
        const d2 = norm(cross(lm.n, sub(pts[Math.min(j + 1, pts.length - 1)], pts[j - 1])).map((v, i) => v * (k ? 1 : -1) + lm.n[i] * 0.5));
        grow(cavity, pts[j], startAct + (j / pts.length) * 12, depth - 1, d2);
      }
    };
    const seedsLV = [...lbb.anterior.slice(-12), ...lbb.posterior.slice(-12), ...lbb.septal.slice(-14)].filter((_, i) => i % 5 === 0);
    for (const s of seedsLV) grow(fields.lvCavity, s, 1500 + 2, 1, norm(add(lm.n, [R() - 0.5, R() - 0.5, R() - 0.5])));
    const seedsRV = [...band.slice(-3), ...rbbAll.slice(-10).filter((_, i) => i % 4 === 0)];
    for (const s of seedsRV) grow(fields.rvCavity, s, 1500 + 4, 1, norm(add(lm.n, [R() - 0.5, R() - 0.5, R() - 0.5])));
    out.purkinje = b;
  }

  return { builders: out, points: { SA, AV: AVc, crest } };
}
