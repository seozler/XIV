// Pericardial sac: a smooth envelope around the heart and the intrapericardial great
// vessels (ascending aorta up to the arch, pulmonary trunk up to its bifurcation, the
// terminal SVC/IVC and pulmonary veins), opened along its superior reflection line so
// the vessels emerge — rendered translucent.

import { tubeChain, smoothUnion, halfSpace, smax } from '../core/sdf.js';
import { meshSDF } from '../core/surfaceNets.js';
import { GeoBuilder } from '../core/geometry.js';

export function buildPericardium(lm, fields, h) {
  const ao = tubeChain(lm.aorta.path.slice(0, 5), lm.aorta.radius.slice(0, 5), 0.3);
  const pa = tubeChain(lm.pulmonaryTrunk.path, lm.pulmonaryTrunk.radius, 0.3);
  const svc = tubeChain(lm.svc.path.slice(1), [lm.svc.r, lm.svc.r, lm.svc.r], 0.3);
  const ivc = tubeChain(lm.ivc.path.slice(1), [lm.ivc.r, lm.ivc.r, lm.ivc.r], 0.3);
  const pvs = lm.pulmonaryVeins.map((v) => tubeChain(v.path.slice(1), [v.r, v.r, v.r], 0.2));
  const all = smoothUnion([fields.epicardium, ao, pa, svc, ivc, ...pvs], 1.4);
  const top = 6.35;
  const cut = halfSpace([0, top, 0], [0, 1, 0]);
  const bottom = halfSpace([0, -5.2, 0], [0, -1, 0]);
  const sac = (x, y, z) => smax(smax(all(x, y, z) - 0.55, cut(x, y, z), 0.3), bottom(x, y, z), 0.4);
  const m = meshSDF(sac, { min: [-7.5, -7.5, -8.5], max: [7.5, 7.0, 6.5] }, h);
  // Open the sac: drop the faces lying on the superior and inferior cut planes.
  const P = m.positions, I = m.indices;
  const keep = [];
  for (let t = 0; t < I.length; t += 3) {
    const ys = [I[t], I[t + 1], I[t + 2]].map((i) => P[i * 3 + 1]);
    if (ys.every((y) => y > top - 0.25) || ys.every((y) => y < -5.2 + 0.25)) continue;
    keep.push(I[t], I[t + 1], I[t + 2]);
  }
  const b = new GeoBuilder();
  b.appendMesh({ positions: P, normals: m.normals, indices: keep });
  return b;
}
