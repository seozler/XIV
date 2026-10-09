// Narrow-band Surface Nets mesher for signed distance fields.
//
// 1. A coarse grid (every `C` cells) is sampled first.
// 2. Fine samples are evaluated exactly only inside coarse cells that can contain
//    the surface (|d| below the coarse cell diagonal); elsewhere the coarse field is
//    trilinearly interpolated, which is sign-safe for Lipschitz fields.
// 3. One vertex per sign-changing cell, placed at the mean of edge crossings and then
//    Newton-projected onto the true surface; normals come from the field gradient.

const C = 4;

export function meshSDF(sdf, bounds, h, { project = true, maxMove = 0.8 } = {}) {
  const [x0, y0, z0] = bounds.min;
  const [x1, y1, z1] = bounds.max;
  const nx = Math.ceil((x1 - x0) / h) + 1;
  const ny = Math.ceil((y1 - y0) / h) + 1;
  const nz = Math.ceil((z1 - z0) / h) + 1;
  const sxy = nx * ny;
  const vals = new Float32Array(nx * ny * nz);

  // ── coarse pass ──
  const cnx = Math.ceil((nx - 1) / C) + 1, cny = Math.ceil((ny - 1) / C) + 1, cnz = Math.ceil((nz - 1) / C) + 1;
  const cvals = new Float32Array(cnx * cny * cnz);
  const ch = h * C;
  for (let k = 0; k < cnz; k++)
    for (let j = 0; j < cny; j++)
      for (let i = 0; i < cnx; i++) cvals[i + cnx * (j + cny * k)] = sdf(x0 + i * ch, y0 + j * ch, z0 + k * ch);

  const band = ch * 1.75 * 1.6;
  const done = new Uint8Array(nx * ny * nz);
  for (let ck = 0; ck < cnz - 1; ck++)
    for (let cj = 0; cj < cny - 1; cj++)
      for (let ci = 0; ci < cnx - 1; ci++) {
        const c000 = cvals[ci + cnx * (cj + cny * ck)];
        const c100 = cvals[ci + 1 + cnx * (cj + cny * ck)];
        const c010 = cvals[ci + cnx * (cj + 1 + cny * ck)];
        const c110 = cvals[ci + 1 + cnx * (cj + 1 + cny * ck)];
        const c001 = cvals[ci + cnx * (cj + cny * (ck + 1))];
        const c101 = cvals[ci + 1 + cnx * (cj + cny * (ck + 1))];
        const c011 = cvals[ci + cnx * (cj + 1 + cny * (ck + 1))];
        const c111 = cvals[ci + 1 + cnx * (cj + 1 + cny * (ck + 1))];
        const mn = Math.min(Math.abs(c000), Math.abs(c100), Math.abs(c010), Math.abs(c110), Math.abs(c001), Math.abs(c101), Math.abs(c011), Math.abs(c111));
        const exact = mn < band;
        const iEnd = Math.min(ci * C + C, nx - 1), jEnd = Math.min(cj * C + C, ny - 1), kEnd = Math.min(ck * C + C, nz - 1);
        for (let k = ck * C; k <= kEnd; k++)
          for (let j = cj * C; j <= jEnd; j++)
            for (let i = ci * C; i <= iEnd; i++) {
              const idx = i + nx * j + sxy * k;
              if (done[idx] === 2) continue;
              if (exact) {
                vals[idx] = sdf(x0 + i * h, y0 + j * h, z0 + k * h);
                done[idx] = 2;
              } else if (!done[idx]) {
                const u = (i - ci * C) / C, v = (j - cj * C) / C, w = (k - ck * C) / C;
                const a = c000 + (c100 - c000) * u, b = c010 + (c110 - c010) * u;
                const c = c001 + (c101 - c001) * u, d = c011 + (c111 - c011) * u;
                const e = a + (b - a) * v, f = c + (d - c) * v;
                vals[idx] = e + (f - e) * w;
                done[idx] = 1;
              }
            }
      }

  // ── vertices ──
  const cellIndex = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const cxy = (nx - 1) * (ny - 1);
  const pos = [];
  const corner = new Float32Array(8);
  const EDGES = [
    [0, 1], [2, 3], [4, 5], [6, 7],
    [0, 2], [1, 3], [4, 6], [5, 7],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const OFF = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
  ];
  for (let k = 0; k < nz - 1; k++)
    for (let j = 0; j < ny - 1; j++)
      for (let i = 0; i < nx - 1; i++) {
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const o = OFF[c];
          const v = vals[i + o[0] + nx * (j + o[1]) + sxy * (k + o[2])];
          corner[c] = v;
          if (v < 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) continue;
        let sx = 0, sy = 0, sz = 0, n = 0;
        for (const [a, b] of EDGES) {
          const va = corner[a], vb = corner[b];
          if ((va < 0) === (vb < 0)) continue;
          const t = va / (va - vb);
          const oa = OFF[a], ob = OFF[b];
          sx += oa[0] + (ob[0] - oa[0]) * t;
          sy += oa[1] + (ob[1] - oa[1]) * t;
          sz += oa[2] + (ob[2] - oa[2]) * t;
          n++;
        }
        cellIndex[i + (nx - 1) * j + cxy * k] = pos.length / 3;
        pos.push(x0 + (i + sx / n) * h, y0 + (j + sy / n) * h, z0 + (k + sz / n) * h);
      }

  // ── faces: one quad per sign-changing grid edge ──
  const idx = [];
  const cell = (i, j, k) => cellIndex[i + (nx - 1) * j + cxy * k];
  for (let k = 1; k < nz - 1; k++)
    for (let j = 1; j < ny - 1; j++)
      for (let i = 1; i < nx - 1; i++) {
        const v0 = vals[i + nx * j + sxy * k];
        const in0 = v0 < 0;
        // x-edge
        if (i < nx - 1) {
          const v1 = vals[i + 1 + nx * j + sxy * k];
          if (in0 !== v1 < 0) quad(cell(i, j - 1, k - 1), cell(i, j, k - 1), cell(i, j, k), cell(i, j - 1, k), in0);
        }
        if (j < ny - 1) {
          const v1 = vals[i + nx * (j + 1) + sxy * k];
          if (in0 !== v1 < 0) quad(cell(i - 1, j, k - 1), cell(i - 1, j, k), cell(i, j, k), cell(i, j, k - 1), in0);
        }
        if (k < nz - 1) {
          const v1 = vals[i + nx * j + sxy * (k + 1)];
          if (in0 !== v1 < 0) quad(cell(i - 1, j - 1, k), cell(i, j - 1, k), cell(i, j, k), cell(i - 1, j, k), in0);
        }
      }
  function quad(a, b, c, d, flip) {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) idx.push(a, b, c, a, c, d);
    else idx.push(a, c, b, a, d, c);
  }

  // ── projection + normals ──
  const P = new Float32Array(pos);
  const N = new Float32Array(P.length);
  const e = h * 0.25;
  const lim = h * maxMove;
  for (let v = 0; v < P.length; v += 3) {
    let x = P[v], y = P[v + 1], z = P[v + 2];
    const ox = x, oy = y, oz = z;
    let gx = 0, gy = 0, gz = 0;
    const iters = project ? 2 : 0;
    for (let it = 0; it <= iters; it++) {
      const d = sdf(x, y, z);
      gx = sdf(x + e, y, z) - sdf(x - e, y, z);
      gy = sdf(x, y + e, z) - sdf(x, y - e, z);
      gz = sdf(x, y, z + e) - sdf(x, y, z - e);
      if (it === iters) break;
      const g2 = (gx * gx + gy * gy + gz * gz) / (4 * e * e) || 1;
      const s = d / g2 / (2 * e);
      x -= gx * s;
      y -= gy * s;
      z -= gz * s;
      const mx = x - ox, my = y - oy, mz = z - oz;
      const m = Math.sqrt(mx * mx + my * my + mz * mz);
      if (m > lim) {
        x = ox + (mx / m) * lim;
        y = oy + (my / m) * lim;
        z = oz + (mz / m) * lim;
      }
    }
    P[v] = x; P[v + 1] = y; P[v + 2] = z;
    const gl = Math.sqrt(gx * gx + gy * gy + gz * gz) || 1;
    N[v] = gx / gl; N[v + 1] = gy / gl; N[v + 2] = gz / gl;
  }

  // Orientation: make every triangle's winding agree with the field gradient (outward).
  const I = new Uint32Array(idx);
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    const s = cx * (N[a] + N[b] + N[c]) + cy * (N[a + 1] + N[b + 1] + N[c + 1]) + cz * (N[a + 2] + N[b + 2] + N[c + 2]);
    if (s < 0) { const tmp = I[t + 1]; I[t + 1] = I[t + 2]; I[t + 2] = tmp; }
  }

  return { positions: P, normals: N, indices: I };
}
