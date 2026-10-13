"""Новый бит для ремикса.

1. (необязательно) Demucs делит трек на дорожки: вокал, бас, барабаны, остальное.
2. Анализ: сетка долей (librosa beat tracking), сильная доля такта, басовая нота каждого такта.
3. Генератор паттернов в стиле жанра + аранжировка (вступление, дроп, сбивки, вариации).
4. Синтез ударных, 808 и саба по нотам трека, сведение с «качанием» под бочку.
"""

import random

import numpy as np
from scipy.signal import butter, sosfilt

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

BEAT_STYLES = {
    "lofi":  {"name": "Lo-Fi boom bap",             "half": True,  "swing": .16, "side": .25, "filter": .35, "lofi": True},
    "phonk": {"name": "Drift Phonk: ковбелл и 808", "half": True,  "swing": 0.0, "side": .45, "filter": .45},
    "trap":  {"name": "Trap: 808 и роллы хэтов",    "half": True,  "swing": 0.0, "side": .35, "filter": .30},
    "house": {"name": "House 4×4",                  "half": False, "swing": .05, "side": .60, "filter": .25},
    "synth": {"name": "Synthwave",                  "half": False, "swing": 0.0, "side": .30, "filter": .25},
    "dance": {"name": "Sped Up / Dance",            "half": False, "swing": 0.0, "side": .50, "filter": .30},
}


def cutoff_hz(amount):
    """0 → фильтр выключен, 1 → срез около 400 Гц."""
    return 20000 * 2 ** (-amount * 5.6)


def _sos(btype, freq, sr, order=2):
    nyq = sr * 0.45
    if isinstance(freq, (list, tuple)):
        freq = [max(20.0, min(f, nyq)) for f in freq]
    else:
        freq = max(20.0, min(freq, nyq))
    return butter(order, freq, btype=btype, fs=sr, output="sos")


def _lowpass(x, sr, f):
    return sosfilt(_sos("low", f, sr), x, axis=-1)


def _highpass(x, sr, f):
    return sosfilt(_sos("high", f, sr), x, axis=-1)


def _hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


# =====================================================================
# DEMUCS (разделение на дорожки)
# =====================================================================


def demucs_available():
    try:
        import demucs.apply  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def load_demucs_model(name="htdemucs"):
    from demucs.pretrained import get_model
    model = get_model(name)
    model.eval()
    return model


def separate(y, sr, model):
    """Возвращает dict дорожек {drums, bass, other, vocals}, каждая формы (каналы, сэмплы)."""
    import librosa
    import torch
    from demucs.apply import apply_model

    msr = model.samplerate
    x = y if y.shape[0] == 2 else np.repeat(y[:1], 2, axis=0)
    if sr != msr:
        x = librosa.resample(x, orig_sr=sr, target_sr=msr, axis=-1)
    wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        out = apply_model(model, ((wav - mean) / std)[None], device=device,
                          split=True, overlap=0.25, progress=False)[0]
    out = (out * std + mean).cpu().numpy()

    stems, n = {}, y.shape[1]
    for name, s in zip(model.sources, out):
        if sr != msr:
            s = librosa.resample(s, orig_sr=msr, target_sr=sr, axis=-1)
        s = s[:, :n]
        if s.shape[1] < n:
            s = np.pad(s, ((0, 0), (0, n - s.shape[1])))
        if y.shape[0] == 1:
            s = s.mean(0, keepdims=True)
        stems[name] = s.astype(np.float32)
    return stems


# =====================================================================
# АНАЛИЗ ТРЕКА
# =====================================================================


def analyze(y, sr, drums=None, bass=None):
    """Темп, доли, сильная доля и басовые ноты по тактам."""
    import librosa

    hop = 512
    mono = y.mean(0)
    rhythm = drums.mean(0) if drums is not None else mono
    dur = y.shape[1] / sr

    env = librosa.onset.onset_strength(y=rhythm, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop,
                                           start_bpm=110, units="time")
    tempo = float(np.atleast_1d(tempo)[0]) or 90.0
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 8:
        beats = np.arange(0.0, dur, 60.0 / tempo)

    # Сильная доля — та из четырёх, где больше всего атак в басу (бочка)
    low_env = librosa.onset.onset_strength(y=_lowpass(rhythm, sr, 150), sr=sr, hop_length=hop)
    frames = np.clip(librosa.time_to_frames(beats, sr=sr, hop_length=hop), 0, len(low_env) - 1)
    phase = int(np.argmax([low_env[frames[b::4]].sum() for b in range(4)]))

    # Басовые ноты: CQT от C1 до B3, сворачиваем в 12 нот и суммируем по тактам
    bsrc = bass.mean(0) if bass is not None else _lowpass(mono, sr, 300)
    sr_b = 11025
    bsrc = librosa.resample(bsrc, orig_sr=sr, target_sr=sr_b)
    C = np.abs(librosa.cqt(bsrc, sr=sr_b, hop_length=512, fmin=librosa.note_to_hz("C1"),
                           n_bins=36, bins_per_octave=12))
    chroma = C.reshape(3, 12, -1).sum(0)
    ctimes = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr_b, hop_length=512)

    bar_starts = beats[phase::4]
    edges = np.concatenate([[-np.inf], bar_starts, [np.inf]])
    seg_idx = np.searchsorted(edges, ctimes, side="right") - 1
    nseg = len(bar_starts) + 1
    seg = np.zeros((nseg, 12))
    np.add.at(seg, np.clip(seg_idx, 0, nseg - 1), chroma.T)
    strength, roots = seg.sum(1), seg.argmax(1)
    strong = strength >= 0.2 * (np.median(strength) or 0)
    votes = np.zeros(12)
    np.add.at(votes, roots[strong], strength[strong])
    key = int(votes.argmax())
    for i in range(nseg):
        if not strong[i]:
            roots[i] = roots[i - 1] if i else key

    return {
        "tempo": 60.0 / float(np.median(np.diff(beats))),
        "beats": beats, "phase": phase, "bar_edges": edges,
        "roots": roots.astype(int), "key": key,
    }


def _root_at(an, t):
    i = int(np.searchsorted(an["bar_edges"], t, side="right") - 1)
    return int(an["roots"][min(max(i, 0), len(an["roots"]) - 1)])


def _beat_grid(beats, phase, duration, scale):
    """Сетка долей, продлённая до начала и после конца трека. Возвращает (сетка, индекс сильной доли, bpm)."""
    b = np.asarray(beats, dtype=float)
    if scale == 2:
        b = np.sort(np.concatenate([b, (b[:-1] + b[1:]) / 2]))
        phase *= 2
    elif scale == 0.5:
        b = b[phase % 2::2]
        phase //= 2
    p = float(np.median(np.diff(b)))
    pre = b[0] - p * np.arange(1, int(b[0] / p) + 9)
    post = b[-1] + p * np.arange(1, int(max(0.0, duration - b[-1]) / p) + 9)
    grid = np.concatenate([pre[::-1], b, post])
    return grid, len(pre) + phase, 60.0 / p


def _time_at(grid, pos):
    i = int(np.floor(pos))
    if i < 0:
        return grid[0] + pos * (grid[1] - grid[0])
    if i >= len(grid) - 1:
        return grid[-1] + (pos - (len(grid) - 1)) * (grid[-1] - grid[-2])
    return grid[i] + (pos - i) * (grid[i + 1] - grid[i])


# =====================================================================
# ГЕНЕРАТОР ПАТТЕРНОВ  (такт = 16 шагов; n — смещение ноты от баса трека)
# =====================================================================


def _ev(s, i, v, n=0, length=2):
    return {"s": s, "i": i, "v": v, "n": n, "len": length}


def _gen_lofi(r, d):
    e, kicks = [], [0, r.choice([7, 10, 10, 11])]
    if r.random() < .4 + d * .5:
        kicks.append(r.choice([3, 6, 13]))
    for s in kicks:
        e += [_ev(s, "kick", .85 if s else 1), _ev(s, "sub", .55, 0, 3)]
    e += [_ev(4, "snare", .9), _ev(12, "snare", .9)]
    if r.random() < d:
        e.append(_ev(r.choice([7, 9, 15]), "snare", .28))
    e += [_ev(s, "hat", .32 if s % 4 else .5) for s in range(0, 16, 2) if r.random() > .1 * d]
    if r.random() < .6:
        e.append(_ev(r.choice([6, 14]), "ohat", .25))
    if r.random() < d:
        e.append(_ev(r.choice([3, 11]), "shaker", .3))
    return e


def _gen_phonk(r, d):
    e, kicks = [], [0, r.choice([3, 6]), 10]
    if r.random() < d:
        kicks.append(r.choice([11, 13, 14]))
    notes = [0, r.choice([0, -2]), r.choice([0, 3, -5]), r.choice([5, 7])]
    for k, s in enumerate(kicks):
        e += [_ev(s, "kick", 1), _ev(s, "808", .9, notes[k % 4], 4)]
    e.append(_ev(8, "clap", 1))
    if r.random() < d * .7:
        e.append(_ev(r.choice([14, 15]), "clap", .4))
    e += [_ev(s, "hat", .38 if s % 4 else .55) for s in range(0, 16, 2)]
    scale = [0, 3, 5, 7, 10, 12]
    for s in [0, 2, 3, 5, 6, 8, 10, 11, 13, 14]:
        if r.random() < .5 + .35 * d:
            e.append(_ev(s, "cowbell", .5 + .3 * r.random(), r.choice(scale)))
    return e


def _gen_trap(r, d):
    e, kicks = [], [0, r.choice([7, 10, 11])]
    if r.random() < .5 + d * .4:
        kicks.append(r.choice([3, 13, 14]))
    notes = [0, r.choice([0, 7, -5]), r.choice([0, 3])]
    for k, s in enumerate(kicks):
        e += [_ev(s, "kick", 1), _ev(s, "808", .95, notes[k % 3], 6)]
    e.append(_ev(8, "clap", 1))
    for s in range(0, 16, 2):
        if r.random() < .3 * d + .08:
            div = r.choice([3, 4, 6])
            e += [_ev(s + 2 * j / div, "hat", .3 + .3 * j / div) for j in range(div)]
        else:
            e += [_ev(s, "hat", .5), _ev(s + 1, "hat", .3)]
    if r.random() < .6:
        e.append(_ev(14, "ohat", .35))
    return e


def _gen_house(r, d):
    e = [_ev(s, "kick", 1) for s in (0, 4, 8, 12)] + [_ev(s, "clap", .85) for s in (4, 12)]
    for s in (2, 6, 10, 14):
        e += [_ev(s, "ohat", .4), _ev(s, "sub", .5, 7 if r.random() < .25 * d else 0, 1.5)]
    e += [_ev(s, "hat", .22) for s in range(1, 16, 2) if r.random() < .5 + .4 * d]
    e += [_ev(s, "shaker", .25) for s in range(16) if r.random() < .25 * d]
    return e


def _gen_synth(r, d):
    e = [_ev(0, "kick", 1), _ev(8, "kick", 1), _ev(4, "gsnare", .95), _ev(12, "gsnare", .95)]
    if r.random() < .4 + .5 * d:
        e.append(_ev(r.choice([6, 10, 11]), "kick", .8))
    e += [_ev(s, "sub", .4, 0, 6) for s in (0, 8)]
    e += [_ev(s, "hat", .3 if s % 4 else .45) for s in range(0, 16, 2)]
    if r.random() < d:
        e += [_ev(s, "hat", .2) for s in range(1, 16, 4)]
    return e


def _gen_dance(r, d):
    e = [_ev(s, "kick", 1) for s in (0, 4, 8, 12)]
    for s in (4, 12):
        e += [_ev(s, "snare", .7), _ev(s, "clap", .6)]
    for s in (2, 6, 10, 14):
        e += [_ev(s, "ohat", .45), _ev(s, "sub", .5, 0, 1.5)]
    e += [_ev(s, "hat", .3 if s % 2 else .2) for s in range(16) if s % 2 or r.random() < .3 * d]
    return e


GENERATORS = {"lofi": _gen_lofi, "phonk": _gen_phonk, "trap": _gen_trap,
              "house": _gen_house, "synth": _gen_synth, "dance": _gen_dance}


def _mutate(r, d, base, style):
    e = [dict(ev) for ev in base]
    hats = [ev for ev in e if ev["i"] == "hat"]
    if hats and r.random() < .6:
        e.remove(r.choice(hats))
    if r.random() < .4 + .5 * d:
        s = r.choice([5, 9, 13, 14, 15])
        if not any(ev["i"] == "kick" and ev["s"] == s for ev in e):
            e.append(_ev(s, "kick", .7))
            bass = next((ev for ev in e if ev["i"] in ("808", "sub")), None)
            if bass:
                e.append({**bass, "s": s, "n": r.choice([0, 7, -2])})
    if style == "phonk":
        for ev in e:
            if ev["i"] == "cowbell" and r.random() < .3:
                ev["n"] = r.choice([0, 3, 7, 10])
    return e


def _fill(style, base):
    e = [ev for ev in base if ev["s"] < 12 or ev["i"] in ("hat", "ohat", "shaker", "cowbell")]
    if style in ("synth", "lofi"):
        e += [_ev(s, "tom", .6 + .1 * j, 7 - 3 * j) for j, s in enumerate((12, 13, 14, 15))]
    else:
        step = .5 if style in ("trap", "phonk", "dance") else 1
        inst = "clap" if style == "house" else "snare"
        e += [_ev(s, inst, .35 + .55 * (s - 12) / 4) for s in np.arange(12, 16, step)]
    if not any(ev["i"] == "kick" and ev["s"] == 12 for ev in e):
        e.append(_ev(12, "kick", .9))
    return e


PATTERN_ROWS = [("Бочка", ("kick",)), ("Снейр", ("snare", "clap", "gsnare", "tom")),
                ("Хэты", ("hat", "ohat", "shaker")), ("Бас/перк", ("808", "sub", "cowbell"))]


def pattern_text(pattern):
    lines = []
    for name, insts in PATTERN_ROWS:
        cells = ["·"] * 16
        for ev in pattern:
            if ev["i"] in insts:
                cells[int(ev["s"])] = "■"
        groups = " ".join("".join(cells[i:i + 4]) for i in range(0, 16, 4))
        lines.append(f"{name:<9} {groups}")
    return "\n".join(lines)


# =====================================================================
# СИНТЕЗ УДАРНЫХ
# =====================================================================


class _Kit:
    def __init__(self, sr, seed):
        self.sr = sr
        self.noise = np.random.default_rng(seed).standard_normal(int(sr * 2)).astype(np.float32)
        self.cache = {}

    def _t(self, dur):
        return np.arange(int(self.sr * dur)) / self.sr

    @staticmethod
    def _dec(t, dec):
        return np.exp(-t * 7.6 / dec)   # до −66 дБ за dec секунд

    def _noise(self, kind, f, dec, width=1.5):
        n = int(self.sr * (dec + .02))
        sos = _sos("high", f, self.sr) if kind == "high" else _sos("band", [f / width, f * width], self.sr)
        x = sosfilt(sos, self.noise[:n])
        x /= np.max(np.abs(x)) + 1e-9
        return x * self._dec(self._t(dec + .02)[:n], dec)

    def _tone(self, wave, f0, f1, tau, dec, dur):
        t = self._t(dur)
        f = f1 + (f0 - f1) * np.exp(-t / tau)
        ph = np.cumsum(f) / self.sr
        x = np.sin(2 * np.pi * ph) if wave == "sine" else 2 * np.abs(2 * (ph % 1) - 1) - 1
        return x * self._dec(t, dec)

    @staticmethod
    def _mix(*parts):
        out = np.zeros(max(len(p) for _, p in parts))
        for offset, p in parts:
            out[offset:offset + len(p)] += p[:len(out) - offset]
        return out

    def get(self, name, n=0):
        key = (name, n)
        if key in self.cache:
            return self.cache[key]
        sr = self.sr
        if name == "kick":
            x = self._mix((0, self._tone("sine", 160, 46, .035, .42, .45) * 1.1),
                          (0, self._noise("high", 3500, .012) * .25))
        elif name == "snare":
            x = self._mix((0, self._noise("band", 1900, .17) * .8), (0, self._tone("tri", 210, 170, .02, .09, .12) * .45))
        elif name == "gsnare":
            x = self._mix((0, self._noise("band", 1500, .34) * .9), (0, self._tone("tri", 190, 150, .02, .12, .15) * .5))
        elif name == "clap":
            parts = [(int(o * sr), self._noise("band", 1250, .025, 1.2) * .7) for o in (0, .011, .023)]
            x = self._mix(*parts, (int(.03 * sr), self._noise("band", 1250, .16, 1.3) * .55))
        elif name == "hat":
            x = self._noise("high", 8000, .045) * .55
        elif name == "ohat":
            x = self._noise("high", 7500, .28) * .45
        elif name == "shaker":
            x = self._noise("band", 6500, .06, 1.3) * .5
        elif name == "crash":
            x = self._noise("high", 5000, 1.6) * .35
        elif name == "tom":
            f = _hz(43 + n)
            x = self._tone("sine", f * 1.6, f, .05, .3, .32) * .8
        elif name == "cowbell":
            f, t = _hz(n), self._t(.25)
            sq = (np.sign(np.sin(2 * np.pi * f * t)) + np.sign(np.sin(2 * np.pi * f * 1.48 * t))) * .5
            sq = sosfilt(_sos("band", [f * 1.3 / 1.5, f * 1.3 * 1.5], sr), sq)
            sq /= np.max(np.abs(sq)) + 1e-9
            x = np.tanh(sq * self._dec(t, .22) * 2.5) / np.tanh(2.5) * .35
        else:
            raise ValueError(name)
        x = x.astype(np.float32)
        self.cache[key] = x
        return x

    def bass(self, midi, prev_midi, dur, hard, glide, peak):
        t = self._t(dur)
        if len(t) == 0:
            return np.zeros(0, np.float32)
        f = _hz(midi)
        if glide and prev_midi is not None:
            ft = f + (_hz(prev_midi) - f) * np.exp(-t / .025)
        else:
            ft = f + (f * (2.0 if hard else 1.2) - f) * np.exp(-t / .012)
        att, body = .006, max(.8 * dur, .01)
        env = np.where(t < att, t / att, np.exp(np.log(.35) * (t - att) / max(body - att, 1e-3)))
        tail = t > body
        env[tail] = .35 * (1 - (t[tail] - body) / max(dur - body, 1e-3))
        x = np.sin(2 * np.pi * np.cumsum(ft) / self.sr) * env * peak
        if hard:
            x = np.tanh(x * 2.5) / np.tanh(2.5) * .9
        else:
            x *= .6
        return x.astype(np.float32)


def _place(bus, x, t, gain, sr, pan=0.0):
    i = int(round(t * sr))
    n = bus.shape[1]
    if i >= n or len(x) == 0:
        return
    if i < 0:
        x, i = x[-i:], 0
    m = min(len(x), n - i)
    if m <= 0:
        return
    bus[0, i:i + m] += x[:m] * gain * (1 - pan)
    bus[1, i:i + m] += x[:m] * gain * (1 + pan)


# =====================================================================
# СВЕДЕНИЕ
# =====================================================================


def render_beat_mix(y, sr, an, style, *, bed=None, vol=.9, filter_amt=.35, side=.3,
                    density=.5, new_bass=True, seed=1, scale=1):
    """Возвращает (новый микс той же длины, что y; инфо для интерфейса).

    bed — дорожки после Demucs: {"main": вокал + остальное, "bass": бас}. Без них старый бит
    приглушается фильтром.
    """
    st_ = BEAT_STYLES[style]
    r = random.Random(seed)
    n, dur = y.shape[1], y.shape[1] / sr
    kit = _Kit(sr, seed)

    grid, down_idx, bpm = _beat_grid(an["beats"], an["phase"], dur, scale)
    mult = 2 if st_["half"] and bpm >= 110 else (.5 if not st_["half"] and bpm < 95 else 1)
    bpb = int(4 * mult)                      # долей в такте паттерна: 2, 4 или 8
    start = down_idx % bpb

    A = GENERATORS[style](r, density)
    B = _mutate(r, density, A, style)
    FA, FB = _fill(style, A), _fill(style, B)

    drums = np.zeros((2, n), np.float32)
    bass_bus = np.zeros((2, n), np.float32)
    kicks, bass_events = [], []
    bar_len = _time_at(grid, start + bpb) - _time_at(grid, start)
    intro = 1 if bar_len > 3 else 2

    k = 0
    while True:
        b0 = start + k * bpb
        t0 = _time_at(grid, b0)
        if t0 >= dur:
            break
        use_b = (k // 4) % 2 == 1
        evs = (FB if use_b else FA) if k % 4 == 3 else (B if use_b else A)
        if k < intro:
            evs = [ev for ev in evs if ev["i"] in ("hat", "ohat", "shaker", "cowbell")]
        if t0 >= 0 and (k == intro or (k > intro and (k - intro) % 8 == 0)):
            _place(drums, kit.get("crash"), t0, 1.0, sr)
        for ev in evs:
            pos = b0 + ev["s"] / 16 * bpb
            if float(ev["s"]).is_integer() and int(ev["s"]) % 2 == 1:
                pos += st_["swing"] * bpb / 16
            t = _time_at(grid, pos)
            if t < 0 or t >= dur - .05:
                continue
            root = _root_at(an, t)
            name = ev["i"]
            if name in ("808", "sub"):
                if new_bass:
                    bass_events.append((t, ev, root))
                continue
            if name == "kick":
                kicks.append(t)
            if name == "cowbell":
                x = kit.get("cowbell", 72 + root + ev["n"])
            elif name == "tom":
                x = kit.get("tom", ev["n"])
            else:
                x = kit.get(name)
            pan = (r.random() - .5) * .3 if name in ("hat", "shaker") else 0.0
            _place(drums, x, t, ev["v"], sr, pan)
        k += 1

    # Бас одноголосный: каждая нота длится до следующей, 808 иногда скользит между нотами
    bass_events.sort(key=lambda e: e[0])
    prev = None
    step_sec = bar_len / 16
    for j, (t, ev, root) in enumerate(bass_events):
        nxt = bass_events[j + 1][0] if j + 1 < len(bass_events) else dur
        if nxt - t < .02:
            continue
        d = max(.06, min(ev["len"] * step_sec, nxt - t - .01))
        midi = 24 + (root + ev["n"]) % 12
        if midi < 31:
            midi += 12
        hard = ev["i"] == "808"
        glide = hard and prev is not None and prev != midi and r.random() < .5
        _place(bass_bus, kit.bass(midi, prev, d, hard, glide, ev["v"]), t, 1.0, sr)
        prev = midi

    if st_.get("lofi"):
        drums = _lowpass(drums, sr, 6500).astype(np.float32)

    # Подложка: дорожки без барабанов или исходный трек под фильтром
    if bed is not None:
        music = bed["main"] + (0 if new_bass else bed["bass"])
    else:
        music = y.copy()
        if filter_amt > 0:
            music = _lowpass(music, sr, cutoff_hz(filter_amt))
        if new_bass:
            music = _highpass(music, sr, 110)
    if music.shape[0] == 1:
        music = np.repeat(music, 2, axis=0)
    music = music * (.5 / (np.max(np.abs(music)) + 1e-9))

    # Качание: подложка приседает под каждую бочку
    duck = np.ones(n, np.float32)
    depth = side * .7
    if depth > 0:
        kicks.sort()
        for j, t in enumerate(kicks):
            nxt = kicks[j + 1] if j + 1 < len(kicks) else t + 1
            rel = min(60 / bpm * .75, nxt - t - .01)
            if rel < .03:
                continue
            i0, a, rl = int(t * sr), int(.01 * sr), int(rel * sr)
            curve = np.concatenate([np.linspace(1, 1 - depth, a, endpoint=False),
                                    np.linspace(1 - depth, 1, max(rl - a, 1))])
            m = min(len(curve), n - i0)
            if m > 0:
                duck[i0:i0 + m] = np.minimum(duck[i0:i0 + m], curve[:m])

    mix = music * duck + (drums + bass_bus) * vol
    peak = np.max(np.abs(mix))
    if peak > .95:
        mix *= .95 / peak
    if y.shape[0] == 1:
        mix = mix.mean(0, keepdims=True)

    info = {"style": st_["name"], "bpm": bpm, "mult": mult, "key": NOTE_NAMES[an["key"]],
            "pattern": pattern_text(A), "separated": bed is not None}
    return mix.astype(np.float32), info
