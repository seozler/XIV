// Heart tissue material: MeshPhysicalMaterial + a shared cardiac deformation field.
//
// All parts are authored in rest (end-diastolic) body coordinates. A single vertex-shader
// field driven by CardiacCycle uniforms moves every vertex, so separately built parts
// (walls, valves, chordae, coronaries, conduction fibres) stay perfectly registered while
// the heart beats — and still move as one when they are exploded apart (explode offsets
// are object transforms applied after the deformation).
//
// Field components (all physiologically motivated):
//   1. Arterial/venous distension around each vessel centre-line (pressure pulse).
//   2. Longitudinal shortening: the AV plane descends toward a nearly fixed apex
//      (MAPSE/TAPSE); atria and great-vessel roots are dragged with it.
//   3. Radial wall thickening with myocardial incompressibility: r'² = r² − c, so the
//      endocardium moves inward far more than the epicardium and the wall thickens.
//   4. LV torsion (wringing): base and apex rotate in opposite directions.
//   5. Atrial contraction toward the atrial centres; fibrillatory "quiver" in AF / VF.
//   6. Electrical activation glow from per-vertex activation times.

import * as THREE from 'three';

export function createHeartUniforms(lm) {
  const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
  return {
    uB: { value: v3(lm.B) },
    uN: { value: v3(lm.n) },
    uL: { value: lm.L },
    uRAc: { value: v3(lm.RA.center) },
    uLAc: { value: v3(lm.LA.center) },
    uVc: { value: 0 }, // ventricular contraction 0 (end-diastole) … 1 (end-systole)
    uAc: { value: 0 }, // atrial contraction 0 … 1
    uLong: { value: 1.25 }, // cm of AV-plane excursion at full contraction
    uRad: { value: 2.9 }, // cm² of radial "area" removed at full contraction
    uTwist: { value: 0.24 }, // rad of torsion between base and apex
    uQuiverV: { value: 0 },
    uQuiverA: { value: 0 },
    uTime: { value: 0 },
    uAoP: { value: 0 },
    uPaP: { value: 0 },
    uVenP: { value: 0 },
    uDeform: { value: 1 },
    uSinceP: { value: 1e4 },
    uSinceQRS: { value: 1e4 },
    uPR: { value: 160 },
    uAF: { value: 0 },
    uVF: { value: 0 },
    uElectric: { value: 0 },
    uFat: { value: 0 },
    uFatColor: { value: new THREE.Color('#e7c46d') },
    uGlowColor: { value: new THREE.Color('#ffd84a') },
    uPickMode: { value: 0 },
  };
}

const VERT_PARS = /* glsl */ `
uniform vec3 uB; uniform vec3 uN; uniform float uL;
uniform vec3 uRAc; uniform vec3 uLAc;
uniform float uVc; uniform float uAc; uniform float uLong; uniform float uRad; uniform float uTwist;
uniform float uQuiverV; uniform float uQuiverA; uniform float uTime;
uniform float uAoP; uniform float uPaP; uniform float uVenP; uniform float uDeform;
attribute vec4 aVessel; attribute float aVesselKind; attribute float aFat; attribute float aAct;
#ifdef HD_SHIFT
attribute vec3 aShift;
#endif
varying vec3 vRest; varying float vFat; varying float vAct;

vec3 hdRot(vec3 v, vec3 k, float a) {
  float c = cos(a), s = sin(a);
  return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
}

vec3 hdDeform(vec3 p) {
  vec3 q = p;
  if (aVessel.w > 0.0) {
    float pulse = aVesselKind < 0.5 ? uAoP : (aVesselKind < 1.5 ? uPaP : uVenP);
    q = aVessel.xyz + (q - aVessel.xyz) * (1.0 + aVessel.w * pulse);
  }
  vec3 d = q - uB;
  float s = dot(d, uN);
  vec3 rad = d - uN * s;
  float r = length(rad);
  float sn = clamp(s / uL, 0.0, 1.0);
  float wV = smoothstep(-1.4, 0.2, s);
  float longW = s > 0.0 ? (1.0 - sn) : exp(s / 2.2);
  float dz = uVc * uLong * longW;
  float cc = uVc * uRad * (1.0 - 0.6 * sn * sn) * wV;
  float rn = sqrt(max(r * r - cc, r * r * 0.12));
  float tw = uVc * uTwist * wV * (sn - 0.3);
  vec3 radN = r > 1e-4 ? rad / r : vec3(0.0);
  radN = hdRot(radN, uN, tw);
  q = uB + uN * (s + dz) + radN * rn;
  float wA = 1.0 - smoothstep(-0.6, 0.6, s);
  vec3 dl = p - uLAc; float gl = exp(-dot(dl, dl) / 9.0) * wA;
  vec3 dr = p - uRAc; float gr = exp(-dot(dr, dr) / 9.0) * wA;
  q -= dl * (uAc * 0.13 * gl) + dr * (uAc * 0.13 * gr);
  if (uQuiverV + uQuiverA > 0.0) {
    vec3 nz = vec3(sin(p.y * 2.3 + uTime * 0.047 + p.x), sin(p.z * 2.1 - uTime * 0.053 + p.y * 1.3), sin(p.x * 1.9 + uTime * 0.061 - p.z));
    q += nz * (uQuiverV * wV * 0.07 + uQuiverA * wA * 0.06);
  }
  return mix(p, q, uDeform);
}
`;

const VERT_NORMAL = /* glsl */ `
vec3 hdP0 = position;
#ifdef HD_SHIFT
hdP0 -= aShift;
#endif
vec3 hdT1 = normalize(abs(normal.y) < 0.95 ? cross(normal, vec3(0.0, 1.0, 0.0)) : cross(normal, vec3(1.0, 0.0, 0.0)));
vec3 hdT2 = cross(normal, hdT1);
vec3 hdD0 = hdDeform(hdP0);
vec3 hdD1 = hdDeform(hdP0 + hdT1 * 0.03);
vec3 hdD2 = hdDeform(hdP0 + hdT2 * 0.03);
vec3 hdNrm = cross(hdD1 - hdD0, hdD2 - hdD0);
vec3 objectNormal = length(hdNrm) > 1e-8 ? normalize(hdNrm) : normal;
#ifdef USE_TANGENT
vec3 objectTangent = vec3(tangent.xyz);
#endif
vRest = hdP0; vFat = aFat; vAct = aAct;
`;

const VERT_BEGIN = /* glsl */ `
vec3 transformed = hdD0;
#ifdef HD_SHIFT
transformed += aShift;
#endif
`;

const FRAG_PARS = /* glsl */ `
uniform float uFat; uniform vec3 uFatColor; uniform float uTissue; uniform vec3 uCutColor;
uniform float uElectric; uniform float uSinceP; uniform float uSinceQRS; uniform float uPR;
uniform float uAF; uniform float uVF; uniform float uTime; uniform float uGlow; uniform vec3 uGlowColor;
uniform float uHighlight; uniform vec3 uHighlightColor;
uniform float uPickMode; uniform vec3 uPickColor;
varying vec3 vRest; varying float vFat; varying float vAct;

float hdHash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float hdNoise(vec3 x) {
  vec3 i = floor(x); vec3 f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hdHash(i), hdHash(i + vec3(1,0,0)), f.x), mix(hdHash(i + vec3(0,1,0)), hdHash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(hdHash(i + vec3(0,0,1)), hdHash(i + vec3(1,0,1)), f.x), mix(hdHash(i + vec3(0,1,1)), hdHash(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float hdActivation(float a) {
  if (a < -0.5) return 0.0;
  bool atrial = a < 1000.0;
  bool nodal = a >= 2000.0;
  if (uAF > 0.5 && (atrial || nodal)) {
    float n = hdNoise(vRest * 0.75 + vec3(uTime * 0.013, uTime * 0.009, -uTime * 0.007));
    return smoothstep(0.6, 0.8, n) * 0.9;
  }
  if (uVF > 0.5 && !atrial && !nodal) {
    float n = hdNoise(vRest * 0.6 + vec3(-uTime * 0.008, uTime * 0.011, uTime * 0.012));
    return smoothstep(0.58, 0.78, n);
  }
  float dt;
  if (atrial) dt = uSinceP - a;
  else if (!nodal) dt = uSinceQRS - (a - 1500.0);
  else dt = uSinceP - (40.0 + (a - 2000.0) / 1000.0 * max(uPR - 80.0, 20.0));
  if (dt < 0.0) return 0.0;
  return exp(-dt / 20.0) + 0.12 * (1.0 - smoothstep(120.0, 300.0, dt));
}
`;

const FRAG_COLOR = /* glsl */ `
float hdN1 = hdNoise(vRest * 2.2);
float hdN2 = hdNoise(vRest * 7.0 + 3.1);
diffuseColor.rgb *= 1.0 + ((hdN1 - 0.5) * 0.6 + (hdN2 - 0.5) * 0.35) * uTissue;
// lobulated fat with an irregular edge
float hdFat = smoothstep(0.32, 0.62, vFat + (hdN1 - 0.5) * 0.45 + (hdN2 - 0.5) * 0.2) * uFat;
diffuseColor.rgb = mix(diffuseColor.rgb, uFatColor * (0.86 + 0.28 * hdN2), hdFat);
if (!gl_FrontFacing) diffuseColor.rgb = uCutColor * (0.82 + 0.36 * hdN2);
`;

const FRAG_EMISSIVE = /* glsl */ `
float hdA = hdActivation(vAct);
totalEmissiveRadiance += uGlowColor * hdA * mix(uElectric * 0.85, 1.7, uGlow);
totalEmissiveRadiance += uHighlightColor * uHighlight;
`;

// GPU picking: an ID pass renders each part's id colour through the same deformation.
const FRAG_PICK = /* glsl */ `
if (uPickMode > 0.5) {
  if (opacity < 0.5) discard;
  gl_FragColor = vec4(uPickColor, 1.0);
}
`;

/**
 * @param shared  uniforms from createHeartUniforms (shared by reference → one update animates all)
 * @param opts    { color, roughness, clearcoat, sheen, opacity, side, glow, tissue, cutColor, shift, clippingPlanes }
 */
export function createHeartMaterial(shared, opts = {}) {
  const {
    color = '#b0413e', roughness = 0.5, metalness = 0, clearcoat = 0.35, clearcoatRoughness = 0.35,
    sheen = 0.25, sheenColor = '#ffb3a8', opacity = 1, side = THREE.DoubleSide, glow = false,
    tissue = 0.35, cutColor = null, shift = false, clippingPlanes = null,
  } = opts;
  const m = new THREE.MeshPhysicalMaterial({
    color, roughness, metalness, clearcoat, clearcoatRoughness, sheen, sheenColor, side,
    sheenRoughness: 0.6, transparent: opacity < 1, opacity, depthWrite: opacity >= 1,
  });
  if (clippingPlanes) m.clippingPlanes = clippingPlanes;
  if (shift) m.defines = { HD_SHIFT: '' };
  const local = {
    uTissue: { value: tissue },
    uCutColor: { value: new THREE.Color(cutColor || new THREE.Color(color).multiplyScalar(0.72).getStyle()) },
    uGlow: { value: glow ? 1 : 0 },
    uHighlight: { value: 0 },
    uHighlightColor: { value: new THREE.Color('#3aa7ff') },
    uPickColor: { value: new THREE.Color(0, 0, 0) },
  };
  m.userData.local = local;
  m.onBeforeCompile = (sh) => {
    for (const k in shared) sh.uniforms[k] = shared[k];
    for (const k in local) sh.uniforms[k] = local[k];
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\n' + VERT_PARS)
      .replace('#include <beginnormal_vertex>', VERT_NORMAL)
      .replace('#include <begin_vertex>', VERT_BEGIN);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + FRAG_PARS)
      .replace('#include <color_fragment>', '#include <color_fragment>\n' + FRAG_COLOR)
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n' + FRAG_EMISSIVE)
      .replace('#include <dithering_fragment>', '#include <dithering_fragment>\n' + FRAG_PICK);
  };
  m.customProgramCacheKey = () => 'heart-v1' + (shift ? '-s' : '');
  return m;
}
