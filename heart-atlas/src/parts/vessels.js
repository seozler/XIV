// Great vessels as thick-walled tubes with cut (open) distal ends.
// • Aortic root: three sinuses of Valsalva bulging behind the cusps, angularly aligned
//   with the valve frame (parallel-transport frames seeded with the valve's u-axis),
//   narrowing at the sinotubular junction.
// • Venous tubes are trimmed exactly where they meet the receiving chamber's cavity,
//   so their lumens open into the atria through the carved orifices.
// • Every vertex stores its centre-line point + compliance for pulse distension.

import { add, sub, mul, dist, frame } from '../core/vec.js';
import { GeoBuilder, tube, curveFrom } from '../core/geometry.js';

/** Radius interpolation over control points by approximate arc length. */
function radiusFn(points, radii) {
  const acc = [0];
  for (let i = 1; i < points.length; i++) acc.push(acc[i - 1] + dist(points[i], points[i - 1]));
  const T = acc.map((a) => a / acc[acc.length - 1]);
  return (t) => {
    for (let i = 1; i < T.length; i++)
      if (t <= T[i]) {
        const f = (t - T[i - 1]) / (T[i] - T[i - 1] || 1);
        return radii[i - 1] + (radii[i] - radii[i - 1]) * f;
      }
    return radii[radii.length - 1];
  };
}

/** Cut a vessel path where it first enters a cavity (path ordered from outside → inside). */
export function trimToCavity(points, cavity, extra = 0.08) {
  const c = curveFrom(points);
  const N = 240;
  const out = [];
  for (let i = 0; i <= N; i++) {
    const p = c.getPointAt(i / N);
    const q = [p.x, p.y, p.z];
    out.push(q);
    if (cavity(q[0], q[1], q[2]) < -extra) break;
  }
  // re-sample to a modest number of control points
  const step = Math.max(1, Math.floor(out.length / 10));
  const pts = out.filter((_, i) => i % step === 0);
  if (pts[pts.length - 1] !== out[out.length - 1]) pts.push(out[out.length - 1]);
  return pts;
}

export function buildVessels(lm, fields) {
  const out = {};
  const ao = lm.aortic;

  // ── Aorta ──
  {
    const b = new GeoBuilder();
    const path = lm.aorta.path;
    const rf = radiusFn(path, lm.aorta.radius);
    const vf = frame(ao.axis, [0, 0, 1]); // same frame as the aortic valve → cusp angles match
    const curve = curveFrom(path);
    const len = curve.getLength();
    const sinusEnd = 2.7 / len;
    const cusps = [0, 120, 240].map((d) => (d * Math.PI) / 180);
    const sinus = (t, a) => {
      if (t > sinusEnd) return 0;
      const x = t / sinusEnd;
      const bump = Math.sin(Math.PI * Math.min(1, x * 1.15)) ** 0.8;
      let lobe = 0;
      for (const c of cusps) lobe = Math.max(lobe, Math.cos(1.5 * Math.atan2(Math.sin(a - c), Math.cos(a - c))));
      return 0.24 * bump * Math.max(0, lobe);
    };
    out.aortaInfo = { curve, frame: vf, sinusEnd, len };
    tube(b, curve, {
      radius: (t, a) => rf(t) * (1 + sinus(t, a)),
      wall: 0.2, segments: 220, radial: 40, kind: 0, compliance: 0.07, initialNormal: vf.u,
    });
    for (const br of lm.aorta.branches) {
      const pts = [br.from, ...br.path];
      const r = radiusFn(pts, [br.r[0] * 1.15, ...br.r]);
      tube(b, pts, { radius: (t) => r(t), wall: 0.11, segments: 40, radial: 20, kind: 0, compliance: 0.05 });
    }
    out.aorta = b;
  }

  // ── Pulmonary trunk + arteries ──
  {
    const b = new GeoBuilder();
    const pt = lm.pulmonaryTrunk;
    const rf = radiusFn(pt.path, pt.radius);
    const vf = frame(lm.pulmonary.axis, [0, 0, 1]);
    const curve = curveFrom(pt.path);
    const len = curve.getLength();
    const se = 2.2 / len;
    const cusps = [0, 120, 240].map((d) => (d * Math.PI) / 180);
    tube(b, curve, {
      radius: (t, a) => {
        let s = 0;
        if (t < se) {
          let lobe = 0;
          for (const c of cusps) lobe = Math.max(lobe, Math.cos(1.5 * Math.atan2(Math.sin(a - c), Math.cos(a - c))));
          s = 0.14 * Math.sin(Math.PI * Math.min(1, (t / se) * 1.1)) * Math.max(0, lobe);
        }
        return rf(t) * (1 + s);
      },
      wall: 0.15, segments: 120, radial: 36, kind: 1, compliance: 0.09, initialNormal: vf.u,
    });
    for (const side of [pt.right, pt.left]) {
      const r = radiusFn(side.path, side.r);
      tube(b, side.path, { radius: (t) => r(t), wall: 0.13, segments: 70, radial: 28, kind: 1, compliance: 0.09 });
    }
    out.pulmonaryTrunk = b;
  }

  // ── Venae cavae (trimmed into the RA) ──
  for (const [id, v] of [['superiorVenaCava', lm.svc], ['inferiorVenaCava', lm.ivc]]) {
    const b = new GeoBuilder();
    const pts = trimToCavity(v.path, fields.raCavity);
    tube(b, pts, { radius: () => v.r, wall: 0.14, segments: 60, radial: 30, kind: 2, compliance: 0.05 });
    out[id] = b;
  }

  // ── Pulmonary veins (trimmed into the LA) ──
  {
    const b = new GeoBuilder();
    for (const v of lm.pulmonaryVeins) {
      const pts = trimToCavity(v.path, fields.laCavity);
      tube(b, pts, { radius: (t) => v.r * (1 + 0.12 * t), wall: 0.12, segments: 40, radial: 24, kind: 2, compliance: 0.05 });
    }
    out.pulmonaryVeins = b;
  }

  // ── Ligamentum arteriosum: LPA origin → inferior surface of the arch ──
  {
    const b = new GeoBuilder();
    const from = add(lm.pulmonaryTrunk.left.path[1], [-0.35, 0.55, 0.15]);
    const to = [1.05, 7.15, -3.45];
    tube(b, [from, add(from, mul(sub(to, from), 0.5)), to], { radius: () => 0.17, segments: 16, radial: 12, capStart: true, capEnd: true });
    out.ligamentumArteriosum = b;
  }
  return out;
}
