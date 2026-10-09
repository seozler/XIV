// Signed distance field primitives and operators. Every SDF is a plain function
// (x, y, z) => distance (negative inside). Chambers, papillary muscles and the
// pericardium are modelled as fields and meshed with surface nets, which gives
// organic blends (smooth unions), real wall thickness and carved orifices.

import { norm, sub, dot, cross } from './vec.js';

export function sphere(c, r) {
  const [cx, cy, cz] = c;
  return (x, y, z) => {
    const dx = x - cx, dy = y - cy, dz = z - cz;
    return Math.sqrt(dx * dx + dy * dy + dz * dz) - r;
  };
}

/**
 * Oriented ellipsoid (Quilez' bound-corrected approximation).
 * basis = [a, b, c] orthonormal axes, r = radii along them.
 */
export function ellipsoid(c, basis, r) {
  const [cx, cy, cz] = c;
  const [a, b, d] = basis;
  const [r0, r1, r2] = r;
  const i0 = 1 / r0, i1 = 1 / r1, i2 = 1 / r2;
  const j0 = i0 * i0, j1 = i1 * i1, j2 = i2 * i2;
  const rmin = Math.min(r0, r1, r2);
  return (x, y, z) => {
    const dx = x - cx, dy = y - cy, dz = z - cz;
    const p0 = dx * a[0] + dy * a[1] + dz * a[2];
    const p1 = dx * b[0] + dy * b[1] + dz * b[2];
    const p2 = dx * d[0] + dy * d[1] + dz * d[2];
    const k0 = Math.sqrt(p0 * p0 * j0 + p1 * p1 * j1 + p2 * p2 * j2);
    if (k0 < 1e-9) return -rmin;
    const k1 = Math.sqrt(p0 * p0 * j0 * j0 + p1 * p1 * j1 * j1 + p2 * p2 * j2 * j2);
    return (k0 * (k0 - 1)) / k1;
  };
}

/** Tapered capsule between a (radius r1) and b (radius r2). */
export function roundCone(a, b, r1, r2) {
  const ba = sub(b, a);
  const l2 = dot(ba, ba);
  const rr = r1 - r2;
  const a2 = l2 - rr * rr;
  const il2 = 1 / l2;
  const [ax, ay, az] = a;
  const [bx, by, bz] = ba;
  return (x, y, z) => {
    const pax = x - ax, pay = y - ay, paz = z - az;
    const yv = pax * bx + pay * by + paz * bz;
    const zv = yv - l2;
    const qx = pax * l2 - bx * yv, qy = pay * l2 - by * yv, qz = paz * l2 - bz * yv;
    const x2 = qx * qx + qy * qy + qz * qz;
    const y2 = yv * yv * l2;
    const z2 = zv * zv * l2;
    const k = Math.sign(rr) * rr * rr * x2;
    if (Math.sign(zv) * a2 * z2 > k) return Math.sqrt(x2 + z2) * il2 - r2;
    if (Math.sign(yv) * a2 * y2 < k) return Math.sqrt(x2 + y2) * il2 - r1;
    return (Math.sqrt(x2 * a2 * il2) + yv * rr) * il2 - r1;
  };
}

/** Chain of tapered capsules through points with per-point radii, smoothly joined. */
export function tubeChain(points, radii, k = 0.2) {
  const parts = [];
  for (let i = 0; i < points.length - 1; i++) parts.push(roundCone(points[i], points[i + 1], radii[i], radii[i + 1]));
  return smoothUnion(parts, k);
}

/** Elliptic cylinder used to carve orifices: centre c, axis w, in-plane u, half-heights h0..h1 along w. */
export function ellipticCylinder(c, w, u, rx, ry, h0, h1) {
  w = norm(w);
  u = norm(sub(u, w.map((x) => x * dot(u, w))));
  const v = cross(w, u);
  const [cx, cy, cz] = c;
  const rm = Math.min(rx, ry);
  return (x, y, z) => {
    const dx = x - cx, dy = y - cy, dz = z - cz;
    const pw = dx * w[0] + dy * w[1] + dz * w[2];
    const pu = dx * u[0] + dy * u[1] + dz * u[2];
    const pv = dx * v[0] + dy * v[1] + dz * v[2];
    const e = (Math.sqrt((pu / rx) ** 2 + (pv / ry) ** 2) - 1) * rm;
    const h = Math.max(h0 - pw, pw - h1);
    return Math.max(e, h);
  };
}

/** Half-space: negative on the side opposite to the normal (i.e. dot(p - p0, n) < 0). */
export function halfSpace(p0, n) {
  n = norm(n);
  const d0 = dot(p0, n);
  return (x, y, z) => x * n[0] + y * n[1] + z * n[2] - d0;
}

// ── Operators ────────────────────────────────────────────────────────────

export const smin = (a, b, k) => {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
};
export const smax = (a, b, k) => -smin(-a, -b, k);

export function union(list) {
  return (x, y, z) => {
    let d = Infinity;
    for (let i = 0; i < list.length; i++) {
      const v = list[i](x, y, z);
      if (v < d) d = v;
    }
    return d;
  };
}

export function smoothUnion(list, k) {
  if (list.length === 1) return list[0];
  return (x, y, z) => {
    let d = list[0](x, y, z);
    for (let i = 1; i < list.length; i++) d = smin(d, list[i](x, y, z), k);
    return d;
  };
}

/** a minus b (smooth when k > 0). */
export function subtract(a, b, k = 0) {
  return k > 0 ? (x, y, z) => smax(a(x, y, z), -b(x, y, z), k) : (x, y, z) => Math.max(a(x, y, z), -b(x, y, z));
}

export function intersect(a, b, k = 0) {
  return k > 0 ? (x, y, z) => smax(a(x, y, z), b(x, y, z), k) : (x, y, z) => Math.max(a(x, y, z), b(x, y, z));
}

/** Material between `outer` surface and `cavity` (both SDFs): the wall of a hollow organ. */
export function wall(outer, cavity) {
  return (x, y, z) => Math.max(outer(x, y, z), -cavity(x, y, z));
}

export function offset(f, d) {
  return (x, y, z) => f(x, y, z) - d;
}

/** Numerical gradient (forward differences) — returns [gx, gy, gz, d]. */
export function gradient(f, x, y, z, e = 1e-3) {
  const d = f(x, y, z);
  return [(f(x + e, y, z) - d) / e, (f(x, y + e, z) - d) / e, (f(x, y, z + e) - d) / e, d];
}

/** Pull a point onto the zero set of f (Newton steps along the gradient). */
export function projectToSurface(f, p, iterations = 12) {
  let [x, y, z] = p;
  for (let i = 0; i < iterations; i++) {
    const [gx, gy, gz, d] = gradient(f, x, y, z);
    const g2 = gx * gx + gy * gy + gz * gz || 1;
    x -= (d * gx) / g2;
    y -= (d * gy) / g2;
    z -= (d * gz) / g2;
    if (Math.abs(d) < 1e-4) break;
  }
  return [x, y, z];
}

export function surfaceNormal(f, p) {
  const [gx, gy, gz] = gradient(f, p[0], p[1], p[2]);
  return norm([gx, gy, gz]);
}

// ── Noise (for trabeculae carneae, pectinate muscles, tissue texture) ────

const PERM = new Uint8Array(512);
{
  let s = 1337;
  const p = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) {
    s = (s * 16807) % 2147483647;
    const j = s % (i + 1);
    [p[i], p[j]] = [p[j], p[i]];
  }
  for (let i = 0; i < 512; i++) PERM[i] = p[i & 255];
}
const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
function grad3(h, x, y, z) {
  const u = h < 8 ? x : y;
  const v = h < 4 ? y : h === 12 || h === 14 ? x : z;
  return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
}
/** Classic Perlin noise in [-1, 1]. */
export function noise3(x, y, z) {
  const X = Math.floor(x) & 255, Y = Math.floor(y) & 255, Z = Math.floor(z) & 255;
  x -= Math.floor(x);
  y -= Math.floor(y);
  z -= Math.floor(z);
  const u = fade(x), v = fade(y), w = fade(z);
  const A = PERM[X] + Y, AA = PERM[A] + Z, AB = PERM[A + 1] + Z;
  const B = PERM[X + 1] + Y, BA = PERM[B] + Z, BB = PERM[B + 1] + Z;
  const l = (a, b, t) => a + t * (b - a);
  return l(
    l(l(grad3(PERM[AA] & 15, x, y, z), grad3(PERM[BA] & 15, x - 1, y, z), u),
      l(grad3(PERM[AB] & 15, x, y - 1, z), grad3(PERM[BB] & 15, x - 1, y - 1, z), u), v),
    l(l(grad3(PERM[AA + 1] & 15, x, y, z - 1), grad3(PERM[BA + 1] & 15, x - 1, y, z - 1), u),
      l(grad3(PERM[AB + 1] & 15, x, y - 1, z - 1), grad3(PERM[BB + 1] & 15, x - 1, y - 1, z - 1), u), v),
    w,
  );
}

/** Ridged multi-octave noise in [0, 1] — sharp crests read as muscular trabeculae. */
export function ridged(x, y, z, octaves = 2) {
  let a = 0, f = 1, amp = 1, norm_ = 0;
  for (let i = 0; i < octaves; i++) {
    a += (1 - Math.abs(noise3(x * f, y * f, z * f))) ** 2 * amp;
    norm_ += amp;
    f *= 2.1;
    amp *= 0.5;
  }
  return a / norm_;
}
