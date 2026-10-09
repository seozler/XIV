// Cardiac cycle engine: electrical events, mechanics, haemodynamics and ECG.
//
// Electrical layer — two clocks (atrial P events, ventricular QRS events) so that
// dissociated rhythms work:
//   nsr   sinus rhythm with respiratory sinus arrhythmia, PR adapts to rate
//   afib  no P waves, fibrillatory baseline, irregularly irregular RR, no atrial kick
//   avb1  first-degree AV block (PR ≈ 300 ms)
//   avb3  complete heart block: sinus P waves + independent wide-QRS escape ≈ 38/min
//   vf    ventricular fibrillation: chaotic ECG, quivering ventricles, no output
//
// Mechanical layer (per beat, timed from QRS onset — Wiggers diagram):
//   MVC = QRS + 45 ms → isovolumetric contraction (≈50 ms)
//   AVO → ejection, LVET = 413 − 1.7·HR ms (Weissler), ejection profile 12x(1−x)²
//   AVC → isovolumetric relaxation (≈70 ms) → MVO → rapid filling (E) → diastasis →
//   atrial systole (A, ≈20 % of filling) 70–170 ms after the P wave.
// Haemodynamics: two-element Windkessel aorta (R·C ≈ 1.6 s, baroreflex-scaled R) fed by the ejection flow;
// LV pressure from the EDPVR in diastole, rising to aortic pressure in IVC, tracking
// aortic pressure in ejection and decaying exponentially in IVR.

const g = (x, s) => Math.exp(-(x * x) / (2 * s * s));
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

export class CardiacCycle {
  constructor({ heartRate = 72, rhythm = 'nsr', speed = 1, seed = 3 } = {}) {
    this.heartRate = heartRate;
    this.rhythm = rhythm;
    this.speed = speed;
    this.paused = false;
    this._seed = seed;
    this.listeners = {};
    this.reset();
  }

  on(evt, fn) { (this.listeners[evt] ||= []).push(fn); return () => this.off(evt, fn); }
  off(evt, fn) { this.listeners[evt] = (this.listeners[evt] || []).filter((f) => f !== fn); }
  emit(evt, data) { for (const f of this.listeners[evt] || []) f(data); }

  rand() {
    let t = (this._seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  reset() {
    this.t = 0;
    this.P = [];
    this.Q = [];
    this.wideQRS = new Set();
    this.nextPAt = 0;
    this.nextQAt = 160;
    this.V = 118; // LV volume (mL)
    this.Pao = 85; // aortic pressure (mmHg)
    this.Rs = 1250; // systemic resistance (mmHg·ms/mL), baroreflex-adjusted
    this.mvOpen = true; this.mvT = -300; this.mvAtClose = 0.6;
    this.avOpen = false; this.avT = -1000;
    this.Vavo = 118;
    this.lastPhase = '';
    this.history = [];
    this.state = {};
    this._schedule();
    this._compute(0);
  }

  setHeartRate(hr) { this.heartRate = clamp(hr, 30, 200); }
  setRhythm(r) {
    if (r === this.rhythm) return;
    this.rhythm = r;
    // drop future events and reschedule from now with the new rhythm
    this.P = this.P.filter((p) => p <= this.t);
    this.Q = this.Q.filter((q) => q <= this.t);
    this.nextPAt = this.t + 60;
    this.nextQAt = this.t + (r === 'avb3' ? 900 : 220);
    this._schedule();
  }

  get rr() { return 60000 / this.heartRate; }
  get pr() { return this.rhythm === 'avb1' ? 300 : clamp(175 - (this.heartRate - 60) * 0.4, 120, 200); }

  _schedule() {
    const horizon = this.t + 4000;
    const r = this.rhythm;
    if (r === 'nsr' || r === 'avb1' || r === 'vf' || r === 'avb3') {
      const atrialRR = r === 'avb3' ? 60000 / Math.max(this.heartRate, 60) : this.rr;
      while (this.nextPAt < horizon) {
        const p = this.nextPAt;
        this.P.push(p);
        if (r === 'nsr' || r === 'avb1') this.Q.push(p + this.pr);
        const rsa = r === 'nsr' ? 1 + 0.035 * Math.sin((p / 4200) * Math.PI * 2) : 1;
        this.nextPAt = p + atrialRR * rsa;
      }
    }
    if (r === 'afib') {
      while (this.nextQAt < horizon) {
        this.Q.push(this.nextQAt);
        this.nextQAt += Math.max(330, this.rr * (0.55 + this.rand() * 0.9));
      }
    }
    if (r === 'avb3') {
      while (this.nextQAt < horizon) {
        this.Q.push(this.nextQAt);
        this.wideQRS.add(this.nextQAt);
        this.nextQAt += 60000 / 38;
      }
    }
    this.P.sort((a, b) => a - b);
    this.Q.sort((a, b) => a - b);
    const keep = this.t - 8000;
    this.P = this.P.filter((p) => p > keep);
    this.Q = this.Q.filter((q) => q > keep);
  }

  _last(list, t) { let v = -Infinity; for (const x of list) if (x <= t) v = x; else break; return v; }
  _next(list, t) { for (const x of list) if (x > t) return x; return Infinity; }

  /** Beat timing for the QRS at time q. */
  beatTiming(q) {
    const nq = this._next(this.Q, q);
    const rr = Number.isFinite(nq) ? nq - q : this.rr;
    const hr = 60000 / rr;
    const MVC = 45;
    const IVCT = 48 * (0.8 + 0.2 * Math.min(1.4, rr / 833));
    const AVO = MVC + IVCT;
    const LVET = clamp(413 - 1.7 * hr, 150, 360) * (this.wideQRS.has(q) ? 1.08 : 1);
    const AVC = AVO + LVET;
    const MVO = AVC + 70 * (0.8 + 0.2 * Math.min(1.4, rr / 833));
    const QT = clamp(400 * Math.sqrt(rr / 1000), 260, 500) + (this.wideQRS.has(q) ? 60 : 0);
    return { rr, hr, MVC, AVO, AVC, MVO, LVET, IVCT, QT };
  }

  update(dtRealMs) {
    if (this.paused) dtRealMs = 0;
    let dt = Math.min(dtRealMs, 100) * this.speed;
    // sub-step integration for stable haemodynamics
    while (dt > 0) {
      const h = Math.min(dt, 2);
      this._step(h);
      dt -= h;
    }
    this._compute();
    return this.state;
  }

  _atrial(t) {
    if (this.rhythm === 'afib') return 0;
    const lp = this._last(this.P, t);
    const ta = t - lp;
    if (ta < 70 || ta > 170) return 0;
    return Math.sin(Math.PI * ((ta - 70) / 100)) ** 1.5;
  }

  _step(h) {
    this.t += h;
    const t = this.t;
    if (this.nextPAt < t + 3000 || this.nextQAt < t + 3000) this._schedule();
    const vf = this.rhythm === 'vf';
    const lq = this._last(this.Q, t);
    const tq = t - lq;
    const B = this.beatTiming(lq);
    const aC = this._atrial(t);

    if (vf) {
      if (this.mvOpen) { this.mvOpen = false; this.mvT = t; this.mvAtClose = this.mvOpenness ?? 0.4; }
      if (this.avOpen) { this.avOpen = false; this.avT = t; }
      this.V += (138 - this.V) * h * 0.0004;
      this.Pao += (-this.Pao / 1430) * h / 1.3 * 1.0 + (18 - this.Pao) * h * 0.0006;
      this.flow = 0; this.fill = 0;
      return;
    }
    // valve events
    if (this.mvOpen && tq >= B.MVC && tq < B.MVO) {
      this.mvOpen = false; this.mvT = t; this.mvAtClose = this.mvOpenness ?? 0.5; this.emit('S1', { t });
    } else if (!this.mvOpen && (tq >= B.MVO || (tq < B.MVC && t - this.mvT > B.MVO))) {
      this.mvOpen = true; this.mvT = t;
    }
    if (!this.avOpen && tq >= B.AVO && tq < B.AVC) { this.avOpen = true; this.avT = t; this.Vavo = this.V; }
    else if (this.avOpen && (tq >= B.AVC || tq < B.AVO)) { this.avOpen = false; this.avT = t; this.emit('S2', { t }); }

    // volumes
    let Q = 0, fill = 0;
    if (this.avOpen) {
      const x = clamp((tq - B.AVO) / B.LVET, 0, 1);
      const SV = Math.max(8, this.Vavo - 48) * 0.96;
      Q = (SV * 12 * x * (1 - x) ** 2) / B.LVET; // mL/ms
      this.V -= Q * h;
    } else if (this.mvOpen) {
      fill = (108 - this.V) / 65 + 0.37 * aC;
      fill = Math.max(0, fill);
      this.V += fill * h;
    }
    // Baroreflex: resistance adapts to rate so mean pressure stays near-normal.
    const Rt = clamp(1250 * Math.pow(B.hr / 72, -0.85), 700, 2600);
    this.Rs += (Rt - this.Rs) * Math.min(1, h * 0.0008);
    // two-element Windkessel (C = 1.3 mL/mmHg)
    this.Pao += ((Q - this.Pao / this.Rs) / 1.3) * h;
    this.flow = Q;
    this.fill = fill;
  }

  _compute() {
    const t = this.t;
    const r = this.rhythm;
    const vf = r === 'vf';
    const lq = this._last(this.Q, t), nq = this._next(this.Q, t);
    const lp = this._last(this.P, t), np = this._next(this.P, t);
    const tq = t - lq;
    const B = this.beatTiming(lq);
    const aC = this._atrial(t);

    // valve openings
    let mo;
    if (vf) mo = Math.max(0, this.mvAtClose * (1 - (t - this.mvT) / 40)) + 0.06 * (1 + Math.sin(t * 0.05));
    else if (this.mvOpen) {
      const d = t - this.mvT;
      mo = sstep(0, 45, d) * (1 - 0.42 * sstep(110, 280, d)) + (r === 'afib' ? 0 : 0.45 * aC);
      mo = clamp(mo, 0, 1);
    } else mo = Math.max(0, this.mvAtClose * (1 - (t - this.mvT) / 28));
    this.mvOpenness = mo;
    let ao;
    if (this.avOpen) {
      const x = clamp((tq - B.AVO) / B.LVET, 0, 1);
      ao = sstep(0, 0.12, x) * (1 - 0.22 * x);
    } else ao = Math.max(0, 0.78 * (1 - (t - this.avT) / 28));
    if (vf) ao = 0;

    // ventricular contraction state from volume (shape follows volume)
    const vc = vf ? 0.06 : clamp((120 - this.V) / 70, -0.15, 1.15) + (tq > B.MVC && tq < B.AVO ? 0.05 : 0);

    // pressures
    const EDP = 3 + 0.0009 * (this.V - 45) ** 2;
    let Plv;
    if (vf) Plv = 10 + 3 * Math.sin(t * 0.03);
    else if (this.avOpen) Plv = this.Pao + this.flow * 9;
    else if (tq >= B.MVC && tq < B.AVO) Plv = EDP + (this.Pao + 1 - EDP) * ((tq - B.MVC) / B.IVCT) ** 1.7;
    else if (tq >= B.AVC && tq < B.MVO) Plv = Math.max(EDP, (this.PavcCache ?? 100) * Math.exp(-(tq - B.AVC) / 30));
    else Plv = EDP + (r === 'afib' ? 0 : 3.5 * aC);
    if (this.avOpen) this.PavcCache = Plv;
    let Pla;
    const vWave = this.mvOpen ? 0 : clamp((tq - B.MVC) / Math.max(1, B.MVO - B.MVC), 0, 1);
    if (vf) Pla = 9 + 4 * aC;
    else if (this.mvOpen) Pla = Plv + 1.2 + 5 * Math.exp(-(t - this.mvT) / 60) + 2.5 * aC;
    else Pla = 4.5 + 7.5 * vWave + 1.6 * g(tq - B.MVC - 22, 12) + 11 * aC;
    const Ppa = 9 + Math.max(0, (this.Pao - 72)) * 0.36;

    // phase
    let phase;
    if (vf) phase = 'fibrillation';
    else if (tq >= B.MVC && tq < B.AVO) phase = 'isovolumetricContraction';
    else if (tq >= B.AVO && tq < B.AVO + B.LVET * 0.4) phase = 'rapidEjection';
    else if (tq >= B.AVO && tq < B.AVC) phase = 'reducedEjection';
    else if (tq >= B.AVC && tq < B.MVO) phase = 'isovolumetricRelaxation';
    else if (aC > 0.05) phase = 'atrialSystole';
    else if (this.mvOpen && t - this.mvT < 160) phase = 'rapidFilling';
    else phase = 'diastasis';
    if (phase !== this.lastPhase) { this.emit('phase', { phase, t }); this.lastPhase = phase; }

    // electrical clocks for the activation shader (upcoming events count as negative times)
    const sinceQRS = nq - t < 60 ? t - nq : t - lq;
    const sinceP = np - t < 60 ? t - np : t - lp;
    const ecg = this.ecgAt(t);

    this.state = {
      t, phase, rhythm: r, heartRate: vf ? 0 : this._displayRate(t),
      sinceP: r === 'afib' ? 1e4 : sinceP, sinceQRS: vf ? 1e4 : sinceQRS, pr: this.pr,
      mitral: mo, tricuspid: clamp(mo * 1.02 + 0.02 * Math.sin(t * 0.01) * mo, 0, 1),
      aortic: ao, pulmonary: vf ? 0 : clamp(this.avOpen ? ao : Math.max(0, 0.78 * (1 - (t - this.avT - 14) / 28)), 0, 1),
      vContraction: vc, aContraction: r === 'afib' ? 0 : aC,
      quiverV: vf ? 1 : 0, quiverA: r === 'afib' ? 1 : 0,
      lvVolume: this.V, lvPressure: Plv, aoPressure: this.Pao, laPressure: Pla, paPressure: Ppa,
      aoPulse: clamp((this.Pao - 72) / 50, 0, 1.4), paPulse: clamp((Ppa - 9) / 14, 0, 1.4),
      venPulse: 0.5 * aC + 0.25 * vWave,
      ejection: this.flow ?? 0, filling: this.fill ?? 0,
      ecg, timing: B, sinceQ: tq,
    };
    this.history.push([t, ecg, Plv, this.Pao, Pla, this.V]);
    const cut = t - 6000;
    while (this.history.length && this.history[0][0] < cut) this.history.shift();
    return this.state;
  }

  /** Ventricular rate averaged over the last few RR intervals (what a monitor shows). */
  _displayRate(t) {
    const past = this.Q.filter((q) => q <= t).slice(-5);
    if (past.length < 3) return Math.round(60000 / this.beatTiming(this._last(this.Q, t)).rr);
    return Math.round((60000 * (past.length - 1)) / (past[past.length - 1] - past[0]));
  }

  /** Synthetic ECG (lead II, mV) at time tau, from the scheduled beats. */
  ecgAt(tau) {
    const r = this.rhythm;
    let v = 0.015 * Math.sin(tau * 0.0016);
    if (r === 'vf') {
      return v + 0.55 * Math.sin((2 * Math.PI * tau * 4.7) / 1000 + 1.6 * Math.sin(tau * 0.0021)) * (0.55 + 0.45 * Math.sin(tau * 0.0011))
        + 0.18 * Math.sin((2 * Math.PI * tau * 7.3) / 1000);
    }
    if (r === 'afib') v += 0.045 * Math.sin((2 * Math.PI * tau * 6.3) / 1000 + 0.8 * Math.sin(tau * 0.0013)) + 0.028 * Math.sin((2 * Math.PI * tau * 8.9) / 1000 + 1.3);
    for (const p of this.P) {
      if (p > tau + 10) break;
      if (tau - p > 300) continue;
      v += 0.16 * g(tau - (p + 48), 19);
    }
    for (const q of this.Q) {
      if (q > tau + 10) break;
      if (tau - q > 700) continue;
      const B = this.beatTiming(q);
      const x = tau - q;
      if (this.wideQRS.has(q)) {
        v += -0.1 * g(x - 18, 14) + 0.9 * g(x - 60, 20) - 0.5 * g(x - 108, 22);
        v += -0.32 * g(x - (B.QT - 90), 58);
      } else {
        v += -0.12 * g(x - 16, 7) + 1.15 * g(x - 38, 8.5) - 0.3 * g(x - 56, 8);
        const tc = B.QT - 88;
        v += 0.3 * g(x - tc, x < tc ? 50 : 36);
      }
    }
    return v;
  }
}
