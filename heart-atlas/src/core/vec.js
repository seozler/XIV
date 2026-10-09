// Minimal 3-vector helpers on plain arrays. Geometry generation runs millions of
// evaluations, so these stay allocation-light and framework-free.

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const len = (a) => Math.sqrt(dot(a, a));
export const norm = (a) => {
  const l = len(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
export const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
export const dist = (a, b) => len(sub(a, b));

/** Sum of scaled vectors: comb([v1, s1], [v2, s2], ...) */
export function comb(...pairs) {
  const r = [0, 0, 0];
  for (const [v, s] of pairs) {
    r[0] += v[0] * s;
    r[1] += v[1] * s;
    r[2] += v[2] * s;
  }
  return r;
}

/** Orthonormal frame whose first axis is `w`; `ref` picks the in-plane orientation. */
export function frame(w, ref) {
  w = norm(w);
  let u = sub(ref, mul(w, dot(ref, w)));
  if (len(u) < 1e-6) u = Math.abs(w[1]) < 0.9 ? cross(w, [0, 1, 0]) : cross(w, [1, 0, 0]);
  u = norm(u);
  const v = cross(w, u);
  return { w, u, v };
}

/** Rodrigues rotation of v about unit axis k by angle a. */
export function rotate(v, k, a) {
  const c = Math.cos(a), s = Math.sin(a);
  const kv = cross(k, v), kd = dot(k, v) * (1 - c);
  return [v[0] * c + kv[0] * s + k[0] * kd, v[1] * c + kv[1] * s + k[1] * kd, v[2] * c + kv[2] * s + k[2] * kd];
}

export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
export const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

/** Deterministic PRNG (mulberry32) so every build produces identical anatomy. */
export function rng(seed = 1) {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
