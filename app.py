"""AI YouTube Music & Remix Studio.

Запуск:  streamlit run app.py
"""

import hashlib
import hmac
import io
import os
import random
import re
import secrets
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import shutil
import subprocess

from scipy.signal import butter, lfilter, oaconvolve, resample_poly, sosfilt


# =====================================================================
# НАСТРОЙКИ
# =====================================================================

# На своём сервере база лежит в постоянной папке NBS_DATA_DIR (её не стирают перезапуски и обновления)
DATA_DIR = Path(os.environ.get("NBS_DATA_DIR") or Path(__file__).resolve().parent)
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:  # нет прав на папку — работаем из папки приложения
    DATA_DIR = Path(__file__).resolve().parent
DB_PATH = DATA_DIR / "users.db"
PBKDF2_ITERATIONS = 200_000
MAX_TRACK_MINUTES = 15
COVER_SIZE = (1280, 720)
COVER_MAX_BYTES = 2 * 1024 * 1024  # лимит YouTube на превью — 2 МБ

# Жанры: подпись -> параметры (короткое имя, пресет по умолчанию, фото, цвет акцента)
GENRES = {
    "LO-FI / CHILL MUSIC (Мягкий расслабленный вайб)": {
        "short": "Lo-Fi", "preset": "Lo-Fi Vinyl", "beat": "lofi",
        "photos": [1060, 1068, 1074, 1081], "accent": (255, 196, 140),
    },
    "SLOWED & REVERB (Глубокий атмосферный замедленный вайб)": {
        "short": "Slowed + Reverb", "preset": "Slowed & Reverb", "beat": "trap",
        "photos": [1047, 1058, 1084, 1039], "accent": (170, 190, 255),
    },
    "PHONK / DRIFT PHONK (Глубокий бас, темный кач)": {
        "short": "Phonk", "preset": "Phonk Drift", "beat": "phonk",
        "photos": [1078, 1062, 1048, 111], "accent": (255, 70, 90),
    },
    "NIGHTCORE / FAST VIBE (Энергичный высокий темп)": {
        "short": "Sped Up", "preset": "Sped Up / Nightcore", "beat": "dance",
        "photos": [1015, 1016, 1042], "accent": (120, 240, 255),
    },
    "DEEP HOUSE / CLUB (Клубный ритм)": {
        "short": "Deep House", "preset": "Deep House", "beat": "house",
        "photos": [1060, 1068, 1074, 1081], "accent": (90, 255, 200),
    },
    "SYNTHWAVE / RETRO 80s (Космический ретро-вайб)": {
        "short": "Synthwave", "preset": "Synthwave", "beat": "synth",
        "photos": [1015, 1047, 1084], "accent": (255, 110, 220),
    },
}

# Пресеты ремикса. link=True — «эффект пластинки»: тон меняется вместе со скоростью.
PRESETS = {
    "Slowed & Reverb": {"label": "Slowed + Reverb", "link": True, "speed": 0.85,
                        "tempo": 1.0, "pitch": 0.0, "reverb": 0.55, "vinyl": 0.2, "bass": 0.1},
    "Sped Up / Nightcore": {"label": "Sped Up", "link": True, "speed": 1.20,
                            "tempo": 1.0, "pitch": 0.0, "reverb": 0.1, "vinyl": 0.0, "bass": 0.0},
    "Phonk Drift": {"label": "Phonk Remix", "link": False, "speed": 1.0,
                    "tempo": 1.05, "pitch": -1.0, "reverb": 0.2, "vinyl": 0.0, "bass": 0.6},
    "Lo-Fi Vinyl": {"label": "Lo-Fi Remix", "link": True, "speed": 0.93,
                    "tempo": 1.0, "pitch": 0.0, "reverb": 0.25, "vinyl": 0.6, "bass": 0.2},
    "Deep House": {"label": "Deep House Remix", "link": False, "speed": 1.0,
                   "tempo": 1.02, "pitch": 0.0, "reverb": 0.15, "vinyl": 0.0, "bass": 0.4},
    "Synthwave": {"label": "Synthwave Remix", "link": False, "speed": 1.0,
                  "tempo": 0.97, "pitch": 0.0, "reverb": 0.45, "vinyl": 0.1, "bass": 0.2},
    "Свои настройки": {"label": "Remix", "link": False, "speed": 1.0,
                       "tempo": 1.0, "pitch": 0.0, "reverb": 0.0, "vinyl": 0.0, "bass": 0.0},
}

# =====================================================================
# ПОЛЬЗОВАТЕЛИ (SQLite + PBKDF2 с солью)
# =====================================================================


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with closing(get_conn()) as conn, conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                   username  TEXT PRIMARY KEY,
                   password  TEXT NOT NULL,
                   is_active INTEGER DEFAULT 1
               )"""
        )
        cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        if "ai_credits" not in cols:      # обновление старой базы без потери пользователей
            conn.execute("ALTER TABLE users ADD COLUMN ai_credits INTEGER DEFAULT 0")
        if "created_at" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
        conn.execute("""CREATE TABLE IF NOT EXISTS credit_log (
                            id       INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT, delta INTEGER, reason TEXT,
                            ts       TEXT DEFAULT (datetime('now'))
                        )""")


def get_setting(name, default=""):
    """Настройка из переменных окружения (Render, свой сервер) или из Secrets (Streamlit Cloud)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def is_admin(username):
    admins = {u.strip().lower() for u in get_setting("ADMIN_USERS").split(",") if u.strip()}
    return username.strip().lower() in admins


def get_user(username):
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT username, is_active, ai_credits, created_at FROM users WHERE username = ?",
                           (username,)).fetchone()
    return None if row is None else dict(zip(("username", "is_active", "ai_credits", "created_at"), row))


def list_users():
    with closing(get_conn()) as conn:
        rows = conn.execute("SELECT username, is_active, ai_credits, created_at FROM users "
                            "ORDER BY created_at DESC, username").fetchall()
    return [dict(zip(("username", "is_active", "ai_credits", "created_at"), r)) for r in rows]


def add_credits(username, delta, reason=""):
    with closing(get_conn()) as conn, conn:
        conn.execute("UPDATE users SET ai_credits = MAX(0, COALESCE(ai_credits, 0) + ?) WHERE username = ?",
                     (int(delta), username))
        conn.execute("INSERT INTO credit_log (username, delta, reason) VALUES (?, ?, ?)",
                     (username, int(delta), reason))


def spend_credit(username, reason):
    """Атомарно списывает 1 генерацию. False — если баланс пуст."""
    with closing(get_conn()) as conn, conn:
        cur = conn.execute("UPDATE users SET ai_credits = ai_credits - 1 WHERE username = ? AND ai_credits > 0",
                           (username,))
        if cur.rowcount:
            conn.execute("INSERT INTO credit_log (username, delta, reason) VALUES (?, -1, ?)", (username, reason))
        return cur.rowcount == 1


def set_active(username, active):
    with closing(get_conn()) as conn, conn:
        conn.execute("UPDATE users SET is_active = ? WHERE username = ?", (1 if active else 0, username))


def set_password(username, password):
    with closing(get_conn()) as conn, conn:
        conn.execute("UPDATE users SET password = ? WHERE username = ?", (hash_password(password), username))


def credit_history(username, limit=20):
    with closing(get_conn()) as conn:
        return conn.execute("SELECT ts, delta, reason FROM credit_log WHERE username = ? ORDER BY id DESC LIMIT ?",
                            (username, limit)).fetchall()


def hash_password(password, salt=None, iterations=PBKDF2_ITERATIONS):
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def check_password(password, stored):
    """Возвращает (совпал ли пароль, нужно ли обновить старый хэш)."""
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, iterations, salt_hex, hash_hex = stored.split("$")
            dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations))
        except ValueError:
            return False, False
        return hmac.compare_digest(dk.hex(), hash_hex), False
    # Старый формат (голый SHA-256 из прошлой версии) — проверяем и переводим на новый
    legacy = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(legacy, stored), True


def register_user(username, password):
    """Возвращает текст ошибки или None при успехе."""
    username = username.strip()
    if len(username) < 3:
        return "Логин должен быть не короче 3 символов."
    if len(password) < 6:
        return "Пароль должен быть не короче 6 символов."
    try:
        with closing(get_conn()) as conn, conn:
            conn.execute("INSERT INTO users (username, password, created_at) VALUES (?, ?, datetime('now'))",
                         (username, hash_password(password)))
    except sqlite3.IntegrityError:
        return "Такой логин уже занят."
    return None


def verify_user(username, password):
    username = username.strip()
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT password, is_active FROM users WHERE username = ?",
                           (username,)).fetchone()
        if not row:
            return False
        ok, needs_upgrade = check_password(password, row[0])
        if not ok or not row[1]:
            return False
        if needs_upgrade:
            with conn:
                conn.execute("UPDATE users SET password = ? WHERE username = ?",
                             (hash_password(password), username))
    return True


# =====================================================================
# ОБРАБОТКА АУДИО  (массивы формы (каналы, сэмплы), стерео сохраняется)
# =====================================================================


def _filter(y, sr, cutoff, btype, order=2):
    cutoff = min(max(cutoff, 20.0), sr * 0.45)
    sos = butter(order, cutoff, btype=btype, fs=sr, output="sos")
    return sosfilt(sos, y, axis=-1)


def lowpass(y, sr, cutoff, order=2):
    return _filter(y, sr, cutoff, "low", order)


def highpass(y, sr, cutoff, order=2):
    return _filter(y, sr, cutoff, "high", order)


def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    return y.astype(np.float32), sr


def change_speed(y, sr, factor):
    """Эффект пластинки: скорость и тон меняются вместе (классический slowed / sped up)."""
    if abs(factor - 1.0) < 1e-3:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=int(round(sr / factor)), axis=-1)


def rubberband_available():
    return shutil.which("rubberband") is not None


def _rubberband(y, sr, pitch, tempo):
    """Студийная смена темпа/тона через Rubber Band (движок R3): чище, без «фазового» призвука."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        sf.write(src, y.T, sr, subtype="FLOAT")
        base = ["rubberband", "-q", "-T", f"{tempo:.6f}", "-p", f"{pitch:.4f}"]
        for engine in (["--fine"], []):            # R3, если версия умеет; иначе стандартный движок
            res = subprocess.run(base + engine + [src, dst], capture_output=True, timeout=900)
            if res.returncode == 0 and os.path.exists(dst):
                out, _ = sf.read(dst, dtype="float32", always_2d=True)
                return out.T
        raise RuntimeError(res.stderr.decode(errors="ignore")[:300])


def shift_pitch_and_tempo(y, sr, pitch, tempo):
    if not pitch and abs(tempo - 1.0) <= 1e-3:
        return y
    if rubberband_available():
        try:
            return _rubberband(y, sr, pitch, tempo)
        except Exception:
            pass                                    # запасной путь — librosa
    if pitch:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
    if abs(tempo - 1.0) > 1e-3:
        y = librosa.effects.time_stretch(y, rate=tempo)
    return y


def bass_boost(y, sr, amount):
    if amount <= 0:
        return y
    return y + lowpass(y, sr, 110) * amount * 1.5


def _impulse_response(sr, size, rng):
    decay = 0.6 + 2.4 * size                      # время затухания до -60 дБ, сек
    n = int(sr * decay)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.91 * t / decay)
    ir = lowpass(ir, sr, 7000 - 3000 * size)      # хвост «темнее» в большой комнате
    ir = np.concatenate([np.zeros(int(sr * 0.015)), ir])  # pre-delay 15 мс
    return ir / np.sqrt(np.sum(ir ** 2))


def apply_reverb(y, sr, amount, seed=7):
    """Свёрточный ревербератор с синтетической импульсной характеристикой."""
    if amount <= 0:
        return y
    rng = np.random.default_rng(seed)
    # Для каждого канала — своя ИХ, это даёт стереоширину
    wet = np.stack([oaconvolve(ch, _impulse_response(sr, amount, rng)) for ch in y])
    dry = np.pad(y, ((0, 0), (0, wet.shape[1] - y.shape[1])))
    mix = 0.15 + 0.35 * amount
    return (1 - mix) * dry + mix * wet


def apply_vinyl(y, sr, amount, seed=11):
    """Тёплое звучание пластинки: срез верхов, мягкая сатурация, шипение и щелчки."""
    if amount <= 0:
        return y
    rng = np.random.default_rng(seed)
    n = y.shape[1]
    y = lowpass(y, sr, 16000 - 6000 * amount)
    drive = 1 + amount
    y = np.tanh(y * drive) / np.tanh(drive)

    hiss = highpass(lowpass(rng.standard_normal(n), sr, 5000), sr, 300)
    hiss *= 0.004 * amount / (np.std(hiss) + 1e-9)

    clicks = np.zeros(n)
    count = int(n / sr * (3 + 10 * amount))
    idx = rng.integers(0, n, count)
    clicks[idx] = rng.choice([-1.0, 1.0], count) * rng.uniform(0.02, 0.12, count) * amount
    clicks = highpass(clicks, sr, 1500)
    return y + hiss + clicks


def normalize(y, peak_db=-1.0):
    y = y - np.mean(y, axis=-1, keepdims=True)   # убираем постоянную составляющую
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 10 ** (peak_db / 20)
    return y


def apply_fades(y, sr, fade_in=0.02, fade_out=0.05):
    n_in, n_out = int(sr * fade_in), int(sr * fade_out)
    if y.shape[1] > n_in + n_out:
        y[:, :n_in] *= np.linspace(0, 1, n_in)
        y[:, -n_out:] *= np.linspace(1, 0, n_out)
    return y


# =====================================================================
# ЭФФЕКТЫ (массивы (2, сэмплы); bpm нужен эффектам «в темп»)
# =====================================================================

FX_DEFAULTS = dict(
    eq_low=0.0, eq_mid=0.0, eq_high=0.0, hp=20, lp=20000,
    saturation=0.0, distortion=0.0, bitcrush=0.0, radio=0.0,
    chorus=0.0, flanger=0.0, phaser=0.0, autofilter=0.0, tremolo=0.0,
    eight_d=0.0, autopan=0.0, width=1.0, echo=0.0, echo_beats=0.75, echo_fb=0.4, room=0.0,
    gate=0.0, comp=0.0, tapestop=0.0,
)
FX_PRESETS = {
    "Без эффектов": {},
    "🎧 8D Audio": {"eight_d": .9, "room": .35, "width": 1.2},
    "🌙 Мечтательный": {"chorus": .45, "echo": .35, "room": .4, "eq_high": 2.0, "width": 1.5},
    "🌊 Под водой": {"lp": 700, "chorus": .5, "room": .5, "eq_low": 4.0},
    "📼 Кассета": {"saturation": .4, "chorus": .2, "lp": 9000, "eq_high": -3.0},
    "📻 Радио 90-х": {"radio": .8, "saturation": .3, "room": .15},
    "📞 Телефон": {"radio": 1.0, "bitcrush": .3},
    "🖤 Тёмный фонк": {"distortion": .25, "eq_low": 5.0, "comp": .5, "lp": 12000},
    "🔊 Клубный памп": {"comp": .6, "autofilter": .4, "eq_low": 3.0, "width": 1.3},
    "⚡ Транс-гейт": {"gate": .8, "echo": .3, "echo_beats": .5},
    "🕹️ Ретро 8-бит": {"bitcrush": .8, "gate": .3},
    "🌀 Психоделика": {"phaser": .7, "flanger": .4, "autopan": .5, "echo": .3},
    "⏹️ Тейп-стоп в конце": {"tapestop": .6},
}
FX_NAMES = {
    "eq_low": "низы", "eq_mid": "середина", "eq_high": "верха", "hp": "срез низов", "lp": "срез верхов",
    "saturation": "ламповая теплота", "distortion": "дисторшн", "bitcrush": "биткраш", "radio": "радио",
    "chorus": "хорус", "flanger": "фленжер", "phaser": "фэйзер", "autofilter": "свип-фильтр",
    "tremolo": "тремоло", "eight_d": "8D", "autopan": "автопанорама", "width": "ширина стерео",
    "echo": "эхо", "room": "зал", "gate": "транс-гейт", "comp": "компрессор", "tapestop": "тейп-стоп",
}
SYNCED_FX = ("echo", "gate", "tremolo", "autofilter", "autopan")


def active_fx(fx):
    out = []
    for k, v in fx.items():
        d = FX_DEFAULTS.get(k)
        if k in FX_NAMES and d is not None and abs(float(v) - float(d)) > 1e-6:
            out.append(FX_NAMES[k])
    return out


def _rbj(kind, f0, sr, gain_db=0.0, q=0.707):
    """Биквад-фильтры по RBJ Audio EQ Cookbook, в формате sos."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * min(max(f0, 20.0), sr * .45) / sr
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / (2 * q)
    if kind in ("lowshelf", "highshelf"):
        sa = 2 * np.sqrt(A) * alpha
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
            a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
            a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    else:  # резонансный лоупасс
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return np.concatenate([b, a])[None, :]


def _one_pole(x, sr, seconds):
    a = np.exp(-1 / (sr * seconds))
    return lfilter([1 - a], [1, -a], x, axis=-1)


def _match_level(new, ref):
    return new * (np.max(np.abs(ref)) / (np.max(np.abs(new)) + 1e-9))


def fx_eq(y, sr, low, mid, high):
    if low:
        y = sosfilt(_rbj("lowshelf", 120, sr, low), y, axis=-1)
    if mid:
        y = sosfilt(_rbj("peak", 1000, sr, mid, q=.8), y, axis=-1)
    if high:
        y = sosfilt(_rbj("highshelf", 6000, sr, high), y, axis=-1)
    return y


def fx_compressor(y, sr, amount):
    thr_db, ratio = -10 - 20 * amount, 2 + 6 * amount
    env = _one_pole(np.max(np.abs(y), axis=0), sr, .08)
    over = np.maximum(0, 20 * np.log10(env + 1e-9) - thr_db)
    gain = 10 ** (-over * (1 - 1 / ratio) / 20)
    makeup = 10 ** (-thr_db * (1 - 1 / ratio) * .4 / 20)
    return y * gain * makeup


def fx_saturation(y, amount):
    d = 1 + 4 * amount
    return np.tanh(y * d) / np.tanh(d)


def fx_distortion(y, sr, amount):
    dist = lowpass(np.clip(y * (1 + 20 * amount), -1, 1), sr, 6000)
    m = min(1.0, amount * 1.2)
    return (1 - m) * y + m * _match_level(dist, y)


def fx_bitcrush(y, amount):
    q = 2 ** (15 - 11 * amount)
    y = np.round(y * q) / q
    hold = 1 + int(amount * 15)
    if hold > 1:
        y = np.repeat(y[:, ::hold], hold, axis=1)[:, :y.shape[1]]
    return y


def fx_radio(y, sr, amount):
    band = sosfilt(butter(2, [350, 3200], btype="band", fs=sr, output="sos"), y, axis=-1)
    band = np.tanh(band * 3) / np.tanh(3)
    return y * (1 - amount) + _match_level(band, y) * amount


def fx_mod_delay(y, sr, base_ms, depth_ms, rate, mix):
    """Модулированная задержка: основа хоруса и фленжера."""
    n = y.shape[1]
    idx = np.arange(n, dtype=np.float64)
    out = np.empty_like(y)
    for c in range(y.shape[0]):
        d = (base_ms + depth_ms * np.sin(2 * np.pi * rate * idx / sr + c * np.pi / 2)) / 1000 * sr
        out[c] = np.interp(idx - d, idx, y[c], left=0.0)
    return y + out * mix


def fx_phaser(y, sr, amount):
    n, B = y.shape[1], 1024
    out = np.empty_like(y)
    zi = np.zeros((4, y.shape[0], 1))
    for s0 in range(0, n, B):
        f = 300 * 6 ** (.5 + .5 * np.sin(2 * np.pi * .35 * s0 / sr))
        t = np.tan(np.pi * f / sr)
        c = (t - 1) / (t + 1)
        x = y[:, s0:s0 + B]
        for k in range(4):
            x, zi[k] = lfilter([c, 1.0], [1.0, c], x, axis=-1, zi=zi[k])
        out[:, s0:s0 + B] = x
    return y * (1 - amount * .5) + out * amount * .5


def fx_autofilter(y, sr, amount, bpm):
    n, B = y.shape[1], 512
    out = np.empty_like(y)
    zi = np.zeros((1, y.shape[0], 2))
    rate = bpm / 60 / 8                            # один проход фильтра за 2 такта
    for s0 in range(0, n, B):
        f = 300 * 30 ** (.5 + .5 * np.sin(2 * np.pi * rate * s0 / sr - np.pi / 2))
        out[:, s0:s0 + B], zi = sosfilt(_rbj("low", f, sr, q=2.5), y[:, s0:s0 + B], axis=-1, zi=zi)
    return y * (1 - amount) + out * amount


def fx_tremolo(y, sr, amount, bpm):
    t = np.arange(y.shape[1]) / sr
    return y * (1 - amount * .5 * (1 + np.sin(2 * np.pi * bpm / 60 * 2 * t)))


def fx_gate(y, sr, amount, bpm):
    pattern = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], float)
    step = 60 / bpm / 4
    env = pattern[(np.arange(y.shape[1]) / sr / step).astype(int) % 16]
    env = _one_pole(env, sr, .004)
    return y * (1 - amount + amount * env)


def fx_eight_d(y, sr, amount):
    """8D: звук плавно облетает голову; «сзади» он темнее и тише."""
    t = np.arange(y.shape[1]) / sr
    th = 2 * np.pi * .12 * t
    mid = y.mean(0)
    behind = (1 - np.cos(th)) / 2
    src = mid * (1 - .6 * behind) + lowpass(mid, sr, 2500) * .6 * behind
    pan = (np.sin(th) + 1) * np.pi / 4
    rot = np.stack([src * np.cos(pan), src * np.sin(pan)]) * np.sqrt(2)
    return y * (1 - amount) + rot * amount


def fx_autopan(y, sr, amount, bpm):
    g = amount * np.sin(2 * np.pi * bpm / 60 / 4 * np.arange(y.shape[1]) / sr)
    return np.stack([y[0] * (1 - np.maximum(g, 0)), y[1] * (1 + np.minimum(g, 0))])


def fx_width(y, w):
    mid, side = y.mean(0), (y[0] - y[1]) / 2 * w
    return np.stack([mid + side, mid - side])


def fx_echo(y, sr, amount, beats, fb, bpm):
    n, delay = y.shape[1], int(beats * 60 / bpm * sr)
    src = lowpass(y.mean(0), sr, 6000)
    wet, g = np.zeros_like(y), 1.0
    for k in range(1, 7):
        sh = delay * k
        if delay <= 0 or sh >= n:
            break
        g *= fb
        wet[k % 2, sh:] += src[:n - sh] * g
    return y + wet * amount * 1.5


def fx_tape_stop(y, sr, amount):
    T = .5 + 2 * amount
    m, n = int(T * sr), y.shape[1]
    if m >= n:
        return y
    tt = np.arange(m) / sr
    src_idx = (n - m) + (tt - tt ** 2 / (2 * T)) * sr  # плёнка замедляется до полной остановки
    out = y.copy()
    base = np.arange(n, dtype=np.float64)
    for c in range(y.shape[0]):
        out[c, n - m:] = np.interp(src_idx, base, y[c])
    return out


def apply_effects(y, sr, fx, bpm):
    f = {**FX_DEFAULTS, **(fx or {})}
    if f["hp"] > 25:
        y = highpass(y, sr, f["hp"], order=4)
    if f["lp"] < 19000:
        y = lowpass(y, sr, f["lp"], order=4)
    y = fx_eq(y, sr, f["eq_low"], f["eq_mid"], f["eq_high"])
    if f["saturation"] > 0:
        y = fx_saturation(y, f["saturation"])
    if f["distortion"] > 0:
        y = fx_distortion(y, sr, f["distortion"])
    if f["radio"] > 0:
        y = fx_radio(y, sr, f["radio"])
    if f["bitcrush"] > 0:
        y = fx_bitcrush(y, f["bitcrush"])
    if f["chorus"] > 0:
        y = fx_mod_delay(y, sr, 18, 6, .9, .7 * f["chorus"])
    if f["flanger"] > 0:
        y = fx_mod_delay(y, sr, 2.5, 2, .25, .8 * f["flanger"])
    if f["phaser"] > 0:
        y = fx_phaser(y, sr, f["phaser"])
    if f["autofilter"] > 0:
        y = fx_autofilter(y, sr, f["autofilter"], bpm)
    if f["tremolo"] > 0:
        y = fx_tremolo(y, sr, f["tremolo"], bpm)
    if f["gate"] > 0:
        y = fx_gate(y, sr, f["gate"], bpm)
    if f["comp"] > 0:
        y = fx_compressor(y, sr, f["comp"])
    return y


def apply_space(y, sr, fx, bpm):
    f = {**FX_DEFAULTS, **(fx or {})}
    if f["eight_d"] > 0:
        y = fx_eight_d(y, sr, f["eight_d"])
    if f["autopan"] > 0:
        y = fx_autopan(y, sr, f["autopan"], bpm)
    if abs(f["width"] - 1) > 1e-3:
        y = fx_width(y, f["width"])
    if f["echo"] > 0:
        y = fx_echo(y, sr, f["echo"], f["echo_beats"], f["echo_fb"], bpm)
    if f["room"] > 0:
        y = apply_reverb(y, sr, f["room"], seed=23)
    return y


# =====================================================================
# МАСТЕРИНГ: громкость по стандарту LUFS (ITU-R BS.1770), true-peak лимитер, моно-низ, склейка
# =====================================================================

MASTER_TARGETS = {
    "youtube": {"name": "YouTube / Spotify / Apple Music (−14 LUFS)", "lufs": -14.0, "ceiling": -1.0},
    "social": {"name": "TikTok / Reels / Shorts, громче (−11 LUFS)", "lufs": -11.0, "ceiling": -1.0},
    "loud": {"name": "Максимально громко, клубный (−9 LUFS)", "lufs": -9.0, "ceiling": -0.8},
    "soft": {"name": "Мягко, больше динамики (−16 LUFS)", "lufs": -16.0, "ceiling": -1.5},
}
MASTER_DEFAULTS = dict(on=True, target="youtube", mono_bass=True, glue=0.3, air=0.0, warmth=0.0)


def _k_weight(y, sr):
    """K-фильтр BS.1770: полка +4 дБ на верхах и срез ниже ~38 Гц."""
    y = sosfilt(_rbj("highshelf", 1500.0, sr, 4.0, q=0.7071), y, axis=-1)
    w = 2 * np.pi * 38.0 / sr
    cw, alpha = np.cos(w), np.sin(w) / (2 * 0.5)
    b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]) / (1 + alpha)
    a = np.array([1, -2 * cw / (1 + alpha), (1 - alpha) / (1 + alpha)])
    return lfilter(b, a, y, axis=-1)


def integrated_lufs(y, sr):
    """Интегральная громкость в LUFS с абсолютным (−70) и относительным (−10 LU) гейтами."""
    if y.ndim == 1:
        y = y[None, :]
    k = _k_weight(y.astype(np.float64), sr)
    block, step = int(.4 * sr), int(.1 * sr)
    if k.shape[1] < block:
        return -70.0
    csum = np.concatenate([np.zeros((k.shape[0], 1)), np.cumsum(k ** 2, axis=1)], axis=1)
    starts = np.arange(0, k.shape[1] - block + 1, step)
    ms = ((csum[:, starts + block] - csum[:, starts]) / block).sum(0)
    lk = -0.691 + 10 * np.log10(ms + 1e-12)
    gated = ms[lk > -70]
    if not len(gated):
        return -70.0
    rel = -0.691 + 10 * np.log10(gated.mean()) - 10
    gated = ms[(lk > -70) & (lk > rel)]
    return float(-0.691 + 10 * np.log10(gated.mean() + 1e-12)) if len(gated) else -70.0


def true_peak_db(y):
    """Истинный пик (с учётом межвыборочных пиков) через 4-кратную передискретизацию."""
    up = resample_poly(y, 4, 1, axis=-1)
    return float(20 * np.log10(np.max(np.abs(up)) + 1e-12))


def limiter(y, sr, ceiling_db=-1.0, release_ms=80.0, block=64):
    """Лимитер с упреждением: заранее плавно убирает громкость перед пиком — без щелчков и клиппинга."""
    ceiling = 10 ** (ceiling_db / 20)
    n = y.shape[1]
    nb = -(-n // block)
    pad = nb * block - n
    # пики по каждому блоку на 4× передискретизированном сигнале (ловит межвыборочные пики)
    up = np.abs(resample_poly(y, 4, 1, axis=-1)).max(0)
    up = np.pad(up, (0, max(0, nb * block * 4 - len(up))))[:nb * block * 4]
    peaks = up.reshape(nb, block * 4).max(1)
    req = np.minimum(1.0, ceiling / (peaks + 1e-12))
    # упреждение: блок «видит» по два соседних блока с каждой стороны
    padded = np.pad(req, 2, constant_values=1.0)
    g = np.min(np.stack([padded[i:i + nb] for i in range(5)]), axis=0)
    # плавное восстановление (release) в децибелах
    g_db = 20 * np.log10(g)
    step = 20 * np.log10(2) / max(1.0, release_ms / 1000 * sr / block)   # +6 дБ за release_ms
    out = np.empty(nb)
    cur = 0.0
    for i in range(nb):
        cur = min(g_db[i], cur + step)
        out[i] = cur
    centers = np.arange(nb) * block + block / 2
    gain = 10 ** (np.interp(np.arange(n + pad), centers, out) / 20)[:n]
    return y * gain


def glue_compressor(y, sr, amount):
    """Мягкая шинная компрессия 2:1 — собирает бит, бас и вокал в один плотный звук."""
    if amount <= 0:
        return y
    rms = np.sqrt(_one_pole(np.mean(y ** 2, axis=0), sr, .03) + 1e-12)
    lvl = 20 * np.log10(rms)
    loud = lvl[lvl > np.percentile(lvl, 30)]
    thr = (np.percentile(loud, 85) if len(loud) else lvl.max()) - 2 - 6 * amount
    ratio = 1.5 + 2.0 * amount
    over = np.maximum(0, lvl - thr)
    gr = _one_pole(over * (1 - 1 / ratio), sr, .12)   # плавная огибающая: без «насоса»
    return y * 10 ** (-gr / 20)


def mono_bass(y, sr, freq=120.0):
    """Бас ниже 120 Гц — в центр: плотнее и громче на телефонах и в клубе."""
    mid, side = (y[0] + y[1]) / 2, (y[0] - y[1]) / 2
    side = highpass(side, sr, freq, order=4)
    return np.stack([mid + side, mid - side])


def master_audio(y, sr, opts=None):
    """Мастеринг-цепочка. Возвращает (сигнал, отчёт о громкости)."""
    o = {**MASTER_DEFAULTS, **(opts or {})}
    tgt = MASTER_TARGETS[o["target"]]
    report = {"lufs_in": integrated_lufs(y, sr), "tp_in": true_peak_db(y)}
    y = highpass(y, sr, 22, order=2)                         # убираем инфразвук и постоянную составляющую
    if y.shape[0] == 2 and o["mono_bass"]:
        y = mono_bass(y, sr)
    if o["warmth"] > 0:                                      # тёплая середина-низ
        y = sosfilt(_rbj("peak", 220, sr, 2.5 * o["warmth"], q=.7), y, axis=-1)
    if o["air"] > 0:                                         # «воздух» на верхах
        y = sosfilt(_rbj("highshelf", 11000, sr, 3.5 * o["air"]), y, axis=-1)
    y = glue_compressor(y, sr, o["glue"])
    for _ in range(4):                                       # громкость → лимитер → проверка
        cur = integrated_lufs(y, sr)
        if cur <= -69:
            break
        diff = tgt["lufs"] - cur
        if abs(diff) < .15:
            break
        y = limiter(y * 10 ** (diff / 20), sr, tgt["ceiling"])
    tp = true_peak_db(y)
    if tp > tgt["ceiling"]:                                  # страховка от остаточных пиков
        y = y * 10 ** ((tgt["ceiling"] - tp - .05) / 20)
    report.update(lufs_out=integrated_lufs(y, sr), tp_out=true_peak_db(y), target=tgt["name"])
    return y, report


def process_audio(y, sr, s, fx=None, bpm=120.0, master=None, fades=(0.02, 0.05)):
    """Скорость/тон → эффекты → пространство → винил → мастеринг. Возвращает (сигнал, отчёт)."""
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    if s["link"]:
        y = change_speed(y, sr, s["speed"])
    else:
        y = shift_pitch_and_tempo(y, sr, s["pitch"], s["tempo"])
    y = normalize(y, -3.0)
    y = apply_effects(y, sr, fx, bpm)
    y = bass_boost(y, sr, s["bass"])
    if fx and fx.get("tapestop", 0) > 0:
        y = fx_tape_stop(y, sr, fx["tapestop"])   # до зала: хвост ревербера звучит после остановки
    y = apply_space(y, sr, fx, bpm)
    y = apply_reverb(y, sr, s["reverb"])
    y = normalize(y, -3.0)
    y = apply_vinyl(y, sr, s["vinyl"])
    m = {**MASTER_DEFAULTS, **(master or {})}
    report = None
    if m["on"]:
        y, report = master_audio(y, sr, m)
    else:
        y = normalize(y, -1.0)
    y = apply_fades(y, sr, *fades)
    return y.astype(np.float32), report


@st.cache_resource(show_spinner=False)
def get_demucs_model():
    """Модель Demucs грузится один раз на весь сервер (при первом запуске скачивается ~80 МБ)."""
    return load_demucs_model()


def _tpdf_dither(y, bits):
    """Треугольный дизеринг перед округлением до 16 бит — убирает «цифровой» призвук на тихих местах."""
    rng = np.random.default_rng(0)
    lsb = 1.0 / (2 ** (bits - 1))
    return y + (rng.random(y.shape) - rng.random(y.shape)) * lsb


def _ffmpeg_mp3(y, sr):
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.mp3")
        sf.write(src, y.T, sr, subtype="FLOAT")
        ar = ["-ar", "48000"] if sr > 48000 else []
        res = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, *ar,
                              "-c:a", "libmp3lame", "-b:a", "320k", dst], capture_output=True, timeout=600)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.decode(errors="ignore")[:300])
        with open(dst, "rb") as fh:
            return fh.read()


def encode_audio(y, sr, fmt, bits=16):
    """WAV 16/24 бит или MP3 320 кбит/с."""
    y = np.clip(y, -1.0, 1.0)
    buf = io.BytesIO()
    if fmt == "WAV":
        if bits == 24:
            sf.write(buf, y.T, sr, format="WAV", subtype="PCM_24")
        else:
            sf.write(buf, np.clip(_tpdf_dither(y, 16), -1, 1).T, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    if shutil.which("ffmpeg"):
        try:
            return _ffmpeg_mp3(y, sr)
        except Exception:
            pass
    try:   # максимальное качество MP3 в libsndfile (soundfile ≥ 0.13)
        sf.write(buf, y.T, sr, format="MP3", compression_level=0.0, bitrate_mode="CONSTANT")
    except TypeError:
        sf.write(buf, y.T, sr, format="MP3")
    return buf.getvalue()


def make_video(cover_jpg, wav_bytes, accent, size=(1920, 1080)):
    """MP4 для YouTube: обложка + анимированная волна звука в цвет жанра. Нужен ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("на сервере нет ffmpeg")
    w, h = size
    color = "0x%02X%02X%02X" % tuple(accent)
    with tempfile.TemporaryDirectory() as tmp:
        img, aud, out = (os.path.join(tmp, x) for x in ("cover.jpg", "audio.wav", "video.mp4"))
        with open(img, "wb") as fh:
            fh.write(cover_jpg)
        with open(aud, "wb") as fh:
            fh.write(wav_bytes)
        wave_h = h // 6
        graph = (f"[0:v]scale={w}:{h}:flags=lanczos,format=rgba[bg];"
                 f"[1:a]aformat=channel_layouts=mono,showwaves=s={w}x{wave_h}:mode=cline:rate=25:"
                 f"colors={color}@0.92:scale=sqrt:draw=full,format=rgba[wv];"
                 f"[bg][wv]overlay=0:H-h-{h // 18}:format=auto,format=yuv420p[v]")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "25", "-i", img, "-i", aud,
               "-filter_complex", graph, "-map", "[v]", "-map", "1:a",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-r", "25",
               "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", out]
        res = subprocess.run(cmd, capture_output=True, timeout=1800)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.decode(errors="ignore")[-400:])
        with open(out, "rb") as fh:
            return fh.read()


# =====================================================================
# НОВЫЙ БИТ: Demucs (необязательно), анализ трека, генератор паттернов, синтез, сведение
# =====================================================================

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


def analyze(y, sr, drums=None, bass=None, harm=None):
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
    else:
        # Точная подгонка: каждую долю подтягиваем к ближайшей реальной атаке (до 35 мс),
        # смещения сглаживаем, чтобы сетка не дёргалась
        peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5, post_avg=5,
                                       delta=float(env.std()) * .2, wait=3)
        ptimes = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
        if len(ptimes) > 1:
            j = np.clip(np.searchsorted(ptimes, beats), 1, len(ptimes) - 1)
            near = np.where(np.abs(ptimes[j - 1] - beats) < np.abs(ptimes[j] - beats), ptimes[j - 1], ptimes[j])
            off = np.where(np.abs(near - beats) < .035, near - beats, 0.0)
            beats = np.maximum.accumulate(beats + np.convolve(off, np.ones(8) / 8, mode="same"))

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

    # Аккорды: мажор или минор над басовой нотой каждого такта (по гармонии трека)
    hsrc = harm.mean(0) if harm is not None else mono
    hc = librosa.feature.chroma_cqt(y=librosa.resample(hsrc, orig_sr=sr, target_sr=22050), sr=22050, hop_length=1024)
    ht = librosa.frames_to_time(np.arange(hc.shape[1]), sr=22050, hop_length=1024)
    hseg = np.zeros((nseg, 12))
    np.add.at(hseg, np.clip(np.searchsorted(edges, ht, side="right") - 1, 0, nseg - 1), hc.T)
    minor = np.array([hseg[i, (roots[i] + 3) % 12] > hseg[i, (roots[i] + 4) % 12] for i in range(nseg)])

    return {
        "tempo": 60.0 / float(np.median(np.diff(beats))),
        "beats": beats, "phase": phase, "bar_edges": edges,
        "roots": roots.astype(int), "key": key, "minor": minor,
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


# ---------- Музыкальные слои: аккорды, клавиши, эхо, переходы ----------

LAYER_STYLES = {
    #        громкость пэда/клавиш, инструмент клавиш, яркость пэда, септаккорды
    "lofi":  {"pad": .25, "keys": .55, "keys_type": "rhodes", "bright": .2, "seventh": True, "echo": .25},
    "phonk": {"pad": .35, "keys": .30, "keys_type": "bell",   "bright": .1, "seventh": False, "echo": .35},
    "trap":  {"pad": .30, "keys": .45, "keys_type": "bell",   "bright": .3, "seventh": False, "echo": .35},
    "house": {"pad": .30, "keys": .50, "keys_type": "organ",  "bright": .4, "seventh": True, "echo": .20},
    "synth": {"pad": .45, "keys": .45, "keys_type": "arp",    "bright": .6, "seventh": True, "echo": .30},
    "dance": {"pad": .35, "keys": .50, "keys_type": "pluck",  "bright": .5, "seventh": False, "echo": .30},
}
# режим, шаги (для аккордов — пары «шаг, длина»), длина ноты в шагах
KEY_PATTERNS = {
    "rhodes": ("chord", [(0, 6), (7, 5)], None),
    "organ":  ("chord", [(2, 1.5), (6, 1.5), (10, 1.5), (14, 1.5)], None),
    "bell":   ("arp", [0, 3, 6, 10, 12], 3),
    "arp":    ("arp", list(range(16)), 1),
    "pluck":  ("arp", [0, 2, 3, 6, 8, 10, 11, 14], 1.5),
}


def _chord_at(an, t):
    i = int(np.clip(np.searchsorted(an["bar_edges"], t, side="right") - 1, 0, len(an["roots"]) - 1))
    minor = an.get("minor")
    return int(an["roots"][i]), bool(minor[i]) if minor is not None else False


def chord_name(root, minor):
    return NOTE_NAMES[root % 12] + ("m" if minor else "")


def _chord_tones(root, minor, seventh):
    tones = [0, 3 if minor else 4, 7] + ([10 if minor else 11] if seventh else [])
    return [root + x for x in tones]


def _saw(f, t, phase0=0.0):
    return 2 * ((f * t + phase0) % 1) - 1


def _synth_pad(midis, dur, sr, bright, rng):
    """Широкий пэд: по три расстроенных пилы на ноту, разные в левом и правом канале."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros((2, n))
    for m in midis:
        f = _hz(m)
        for ch, cents in ((0, (-9, 0, 9)), (1, (-7, 2, 11))):
            for c in cents:
                out[ch] += _saw(f * 2 ** (c / 1200), t, rng.random())
    out = sosfilt(_sos("low", 700 + 2500 * bright, sr), out, axis=-1)
    env = np.ones(n)
    na, nr = int(min(.5, dur * .3) * sr), int(min(.6, dur * .4) * sr)
    if na:
        env[:na] = np.linspace(0, 1, na)
    if nr:
        env[-nr:] *= np.linspace(1, 0, nr)
    return (out * env / (3 * len(midis))).astype(np.float32)


def _note(kind, midi, dur, sr):
    n = int(max(dur, .05) * sr)
    t = np.arange(n) / sr
    f = _hz(midi)
    w = 2 * np.pi * f * t
    if kind == "rhodes":     # электропиано: мягкая FM-атака и тремоло
        x = np.sin(w + 1.2 * np.exp(-t / .25) * np.sin(w)) + .25 * np.sin(2 * w) * np.exp(-t / .4)
        x *= np.exp(-t / 1.4) * (1 + .15 * np.sin(2 * np.pi * 4.5 * t))
    elif kind == "bell":     # колокольчик (FM)
        x = np.sin(w + 2.5 * np.exp(-t / .5) * np.sin(3.5 * w)) * np.exp(-t / .9)
    elif kind == "organ":    # хаус-орган, короткий стаб
        x = (np.sin(w) + .5 * np.sin(2 * w) + .3 * np.sin(3 * w) + .2 * np.sin(4 * w)) / 2 * np.exp(-t / .35)
    elif kind == "pluck":
        x = sosfilt(_sos("low", min(f * 6, 8000), sr), _saw(f, t)) * np.exp(-t / .22) + .3 * np.sin(w) * np.exp(-t / .3)
    else:                    # арпеджио-синт
        x = sosfilt(_sos("low", min(f * 8, 9000), sr), .6 * np.sign(np.sin(w)) + .4 * _saw(f * 1.005, t, .3)) * np.exp(-t / .16)
    nr = int(min(.03, dur / 4) * sr)
    if nr > 0:
        x[-nr:] *= np.linspace(1, 0, nr)
    return x.astype(np.float32)


def _riser(dur, sr, rng):
    """Шумовой райзер: полоса фильтра поднимается от 400 Гц до 8 кГц, громкость растёт."""
    n = int(dur * sr)
    if n < 100:
        return np.zeros(0, np.float32)
    x, out, chunks = rng.standard_normal(n), np.zeros(n), 24
    for c in range(chunks):
        a, b = c * n // chunks, (c + 1) * n // chunks
        f = 400 * 20 ** (c / (chunks - 1))
        out[a:b] = sosfilt(_sos("band", [f / 1.6, f * 1.6], sr), x[a:b])
    out /= np.max(np.abs(out)) + 1e-9
    return (out * (np.arange(n) / n) ** 2).astype(np.float32)


def _impact(sr):
    t = np.arange(int(1.4 * sr)) / sr
    f = 32 + 30 * np.exp(-t / .08)
    return (np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / .45)).astype(np.float32)


def _ping_pong_echo(bus, sr, bpm, amount):
    """Эхо в темп трека (пунктирная восьмая), по очереди слева и справа."""
    if amount <= 0:
        return bus
    n, delay = bus.shape[1], int(.75 * 60 / bpm * sr)
    src, wet = bus.mean(0), np.zeros_like(bus)
    src = _lowpass(src, sr, 5000)
    for tap in range(1, 5):
        sh = delay * tap
        if sh >= n:
            break
        wet[tap % 2, sh:] += src[:n - sh] * (.3 + .3 * amount) ** tap * 1.6
    return bus + wet * amount


def render_beat_mix(y, sr, an, style, *, bed=None, vol=.9, filter_amt=.35, side=.3,
                    density=.5, new_bass=True, seed=1, scale=1, layers=None):
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

    lay = layers or {}
    fx_on = lay.get("on") and lay.get("fx")
    section = 8 if bar_len >= 3 else 16          # новая часть примерно каждые 40 секунд
    is_drop = lambda kk: kk == intro or (kk > intro and (kk - intro) % section == 0)  # noqa: E731
    bars, drops = [], []

    k = 0
    while True:
        b0 = start + k * bpb
        t0 = _time_at(grid, b0)
        if t0 >= dur:
            break
        bars.append((k, b0))
        use_b = (k // 4) % 2 == 1
        evs = (FB if use_b else FA) if k % 4 == 3 else (B if use_b else A)
        if k < intro:
            evs = [ev for ev in evs if ev["i"] in ("hat", "ohat", "shaker", "cowbell")]
        elif fx_on and is_drop(k + 1):
            # брейк: вторая половина такта перед дропом без бочки и баса — дроп бьёт сильнее
            evs = [ev for ev in evs if ev["s"] < 8 or ev["i"] in ("hat", "snare", "clap", "tom")]
        if t0 >= 0 and is_drop(k):
            drops.append(t0)
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

    # Музыкальные слои по аккордам трека
    layer_bus = np.zeros((2, n), np.float32)
    fx_bus = np.zeros((2, n), np.float32)
    nrng = np.random.default_rng(seed)
    if lay.get("on"):
        cfg = LAYER_STYLES[style]
        # Пэд: аккорд на каждый такт песни, вступает с первого дропа
        if lay.get("pad", 0) > 0:
            pad_bus = np.zeros((2, n), np.float32)
            starts = [s for s in an["bar_edges"][1:-1] if s < dur]
            if not starts or starts[0] > 0:
                starts = [0.0] + starts
            pad_from = drops[0] if drops else 0.0
            for i, bs in enumerate(starts):
                be = starts[i + 1] if i + 1 < len(starts) else dur
                if be <= pad_from:
                    continue
                s0 = max(bs, pad_from)
                root, minor = _chord_at(an, (bs + be) / 2)
                base = 48 + root + (12 if 48 + root < 52 else 0)
                midis = [base - 12] + _chord_tones(base, minor, cfg["seventh"])
                x = _synth_pad(midis, be - s0 + .35, sr, cfg["bright"], nrng)
                i0 = int(s0 * sr)
                m = min(x.shape[1], n - i0)
                if m > 0:
                    pad_bus[:, i0:i0 + m] += x[:, :m]
            layer_bus += pad_bus * lay["pad"]
        # Клавиши / арпеджио: в такт паттерна, с первого такта
        if lay.get("keys", 0) > 0:
            keys_bus = np.zeros((2, n), np.float32)
            kind = cfg["keys_type"]
            mode, steps, note_len = KEY_PATTERNS[kind]
            if style == "phonk":
                steps = [0, 6, 12] + ([3, 9] if density > .6 else [])
            arp_i = 0
            for kk, b0 in bars:
                step_sec = (_time_at(grid, b0 + bpb) - _time_at(grid, b0)) / 16
                for item in steps:
                    s, length = item if mode == "chord" else (item, note_len)
                    t = _time_at(grid, b0 + s / 16 * bpb)
                    if t < 0 or t >= dur - .05:
                        continue
                    root, minor = _chord_at(an, t)
                    tail = .5 if kind in ("rhodes", "bell") else .05
                    if mode == "chord":
                        base = 48 + root + (12 if 48 + root < 52 else 0)
                        for v, midi in enumerate(_chord_tones(base, minor, cfg["seventh"])):
                            _place(keys_bus, _note(kind, midi, length * step_sec + tail, sr), t + .012 * v, .22, sr,
                                   (v - 1.5) * .15)
                    else:
                        tones = _chord_tones(root, minor, cfg["seventh"])
                        ladder = tones + [x + 12 for x in tones]
                        if kk % 2:
                            ladder = ladder[::-1]           # каждый второй такт арпеджио идёт вниз
                        midi = 72 + ladder[arp_i % len(ladder)] - (12 if 72 + root > 79 else 0)
                        arp_i += 1
                        _place(keys_bus, _note(kind, midi, length * step_sec + tail, sr), t, .35, sr,
                               .25 if arp_i % 2 else -.25)
            keys_bus = _ping_pong_echo(keys_bus, sr, bpm, lay.get("echo", 0))
            layer_bus += keys_bus * lay["keys"]
        # Общий зал для слоёв
        if np.any(layer_bus):
            layer_bus = apply_reverb(layer_bus, sr, .4)[:, :n].astype(np.float32)
        # Переходы: райзер и реверс-тарелка перед дропом, низкий удар на дропе
        if lay.get("fx"):
            crash_rev = kit.get("crash")[::-1]
            for d_t in drops:
                rl = min(4.0, bar_len)
                if d_t - rl > 0:
                    _place(fx_bus, _riser(rl, sr, nrng), d_t - rl, .22, sr)
                if d_t - len(crash_rev) / sr > 0:
                    _place(fx_bus, crash_rev, d_t - len(crash_rev) / sr, .5, sr)
                _place(fx_bus, _impact(sr), d_t, .55, sr)

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

    if lay.get("on") and lay.get("polish"):
        # шире стерео для слоёв
        mid, sid = layer_bus.mean(0), (layer_bus[0] - layer_bus[1]) / 2 * 1.5
        layer_bus = np.stack([mid + sid, mid - sid])
    soft_duck = 1 - (1 - duck) * .7               # слои качаются мягче подложки
    mix = music * duck + (drums + bass_bus) * vol + layer_bus * soft_duck + fx_bus
    peak = np.max(np.abs(mix)) + 1e-9
    if lay.get("on") and lay.get("polish"):
        # склейка: мягкая сатурация собирает бит, бас и музыку в один звук
        mix = np.tanh(mix / peak * 1.3) / np.tanh(1.3) * .95
    elif peak > .95:
        mix *= .95 / peak
    if y.shape[0] == 1:
        mix = mix.mean(0, keepdims=True)

    names = []
    for i in range(1, len(an["roots"])):
        name = chord_name(*_chord_at(an, (an["bar_edges"][i] + min(an["bar_edges"][i + 1], dur)) / 2))
        if not names or names[-1] != name:
            names.append(name)
        if len(names) == 8:
            break
    info = {"style": st_["name"], "bpm": bpm, "mult": mult, "key": NOTE_NAMES[an["key"]],
            "pattern": pattern_text(A), "separated": bed is not None,
            "chords": " → ".join(names), "drops": len(drops)}
    return mix.astype(np.float32), info


# =====================================================================
# ОБЛОЖКА 1280x720
# =====================================================================

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


@st.cache_resource
def _font_path():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def load_font(size):
    path = _font_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def fit_text(draw, text, max_width, max_lines, start_size, min_size):
    """Подбирает размер шрифта, чтобы текст влез в max_lines строк."""
    for size in range(start_size, min_size - 1, -4):
        font = load_font(size)
        lines = wrap_text(draw, text, font, max_width)
        if len(lines) <= max_lines and all(draw.textlength(l, font=font) <= max_width for l in lines):
            return font, size, lines
    return font, size, lines[:max_lines]


def fetch_background(photo_ids):
    url = f"https://picsum.photos/id/{random.choice(photo_ids)}/1280/720"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except (requests.RequestException, OSError):
        return None


def gradient_background(accent):
    w, h = COVER_SIZE
    top = np.array([18, 18, 32], dtype=float)
    bottom = np.array(accent, dtype=float) * 0.35
    t = np.linspace(0, 1, h)[:, None, None]
    arr = (top * (1 - t) + bottom * t).repeat(w, axis=1)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def make_cover(title, artist, genre_text, accent, background):
    w, h = COVER_SIZE
    bg = ImageOps.fit(background.convert("RGB"), COVER_SIZE, Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(4)).convert("RGBA")

    # Затемнение слева направо: текст читается, фото остаётся видно
    alpha = np.tile(np.linspace(225, 110, w).astype(np.uint8), (h, 1))
    shade = Image.new("RGBA", COVER_SIZE, (0, 0, 0, 0))
    shade.putalpha(Image.fromarray(alpha, "L"))
    cover = Image.alpha_composite(bg, shade)
    draw = ImageDraw.Draw(cover)

    x, max_w = 90, w - 180
    title_font, title_size, title_lines = fit_text(draw, title.upper(), max_w, 2, 104, 48)
    artist_font, artist_size = load_font(44), 44
    badge_font, badge_size = load_font(30), 30

    badge_h = badge_size + 24
    title_line_h = int(title_size * 1.12)
    block_h = badge_h + 30 + title_line_h * len(title_lines) + 24 + artist_size
    y = (h - block_h) // 2

    # Плашка жанра
    badge_text = genre_text.upper()
    badge_w = int(draw.textlength(badge_text, font=badge_font)) + 40
    draw.rounded_rectangle([x, y, x + badge_w, y + badge_h], radius=badge_h // 2, fill=accent + (255,))
    draw.text((x + 20, y + badge_h // 2), badge_text, font=badge_font, fill=(15, 15, 20), anchor="lm")
    y += badge_h + 30

    # Название
    for line in title_lines:
        draw.text((x, y), line, font=title_font, fill=(255, 255, 255),
                  stroke_width=max(2, title_size // 30), stroke_fill=(0, 0, 0))
        y += title_line_h
    y += 24

    # Артист с цветной чертой
    if artist.strip():
        draw.rectangle([x, y + 6, x + 8, y + artist_size], fill=accent)
        draw.text((x + 24, y), artist.strip(), font=artist_font, fill=accent,
                  stroke_width=2, stroke_fill=(0, 0, 0))

    rgb = cover.convert("RGB")
    for quality in (92, 85, 75, 65):
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=quality, optimize=True)
        if buf.tell() <= COVER_MAX_BYTES:
            break
    return buf.getvalue()


# =====================================================================
# YOUTUBE SEO
# =====================================================================


def hashtag(text):
    words = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    return "#" + "".join(w[:1].upper() + w[1:] for w in words) if words else ""


def make_seo(title, artist, genre_short, preset_label):
    title, artist = title.strip(), artist.strip()
    # «Song (Remix)» -> «Song», чтобы не было «(Remix) (Lo-Fi Remix)»
    title = re.sub(r"\s*[\(\[][^\)\]]*remix[^\)\]]*[\)\]]\s*$", "", title, flags=re.I) or title
    by = f"{artist} - " if artist else ""
    titles = [
        f"{by}{title} ({preset_label})",
        f"{title} [{preset_label}] | {genre_short} Vibes",
        f"{title} — {genre_short} Remix (HQ Audio)",
    ]
    titles = [t[:100] for t in titles]  # лимит YouTube — 100 символов

    tags_list = list(dict.fromkeys(filter(None, [
        hashtag(title), hashtag(artist), hashtag(genre_short), hashtag(preset_label),
        "#Remix", "#Music", "#Vibes",
    ])))

    description = f"""🎵 {by}{title}
🎧 Version: {preset_label} · {genre_short}

Immerse yourself in the atmosphere. Enjoy the vibe, drop a like and subscribe for more releases!

📌 Follow & Support:
• Telegram: your link here
• Spotify / Apple Music: your link here

{' '.join(tags_list)}
"""

    keywords = [title, artist, f"{title} {preset_label.lower()}", f"{genre_short} remix",
                preset_label.lower(), f"{genre_short} music", "remix"]
    tags, total, seen = [], 0, set()
    for k in filter(None, (k.strip() for k in keywords)):
        if k.lower() in seen:
            continue
        seen.add(k.lower())
        if total + len(k) + 2 > 500:  # лимит поля тегов — 500 символов
            break
        tags.append(k)
        total += len(k) + 2
    return titles, description, ", ".join(tags)


def slugify(text):
    return re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_") or "track"


# =====================================================================
# ИНТЕРФЕЙС
# =====================================================================


# =====================================================================
# ДИЗАЙН: NBS SOFT MUSIC
# =====================================================================

NBS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700&family=Manrope:wght@400;500;600;700&display=swap');
:root {
  --nbs-bg: #0d0b14; --nbs-panel: rgba(23, 18, 34, .78); --nbs-line: #2b2440;
  --nbs-fg: #ece7f5; --nbs-muted: #a39bb8; --nbs-lilac: #b8a4ff; --nbs-peach: #ffb59c;
}
.stApp {
  background:
    radial-gradient(1100px 560px at 90% -12%, rgba(184,164,255,.16), transparent 60%),
    radial-gradient(900px 520px at -15% 105%, rgba(255,181,156,.10), transparent 60%),
    var(--nbs-bg);
}
.stApp, .stApp p, .stApp label, .stApp li { font-family: 'Manrope', system-ui, sans-serif; }
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Unbounded', 'Manrope', sans-serif !important; letter-spacing: -.01em; }
.stApp h3 { font-size: 1.02rem !important; font-weight: 600 !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #140f1f, #0d0b14); border-right: 1px solid var(--nbs-line); }
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--nbs-panel); border: 1px solid var(--nbs-line) !important; border-radius: 18px !important;
  backdrop-filter: blur(6px);
}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 12px; font-weight: 600; transition: transform .12s ease, filter .12s ease;
}
.stButton > button[kind="primary"], .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
  background: linear-gradient(100deg, var(--nbs-lilac), var(--nbs-peach)); color: #1a1426; border: 0; font-weight: 700;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover { filter: brightness(1.07); transform: translateY(-1px); }
[data-testid="stFileUploaderDropzone"] { background: rgba(184,164,255,.05); border: 1px dashed #4a3f6b; border-radius: 14px; }
[data-testid="stMetric"] { background: rgba(255,255,255,.03); border: 1px solid var(--nbs-line); border-radius: 14px; padding: 10px 14px; }
[data-testid="stExpander"] details { border-radius: 14px; border-color: var(--nbs-line); }
.stTabs [data-baseweb="tab-list"] { gap: 2px; flex-wrap: wrap; }
.stApp code { color: var(--nbs-peach); }

.nbs-hero { position: relative; padding: 30px 28px 24px; margin: 4px 0 18px; border-radius: 22px;
  border: 1px solid var(--nbs-line); overflow: hidden;
  background: linear-gradient(135deg, rgba(184,164,255,.14), rgba(255,181,156,.07) 55%, rgba(13,11,20,0)); }
.nbs-logo { font-family: 'Unbounded', sans-serif; font-weight: 700; font-size: clamp(34px, 7vw, 54px);
  line-height: .95; letter-spacing: -.02em; color: var(--nbs-fg); }
.nbs-logo span { display: block; margin-top: 10px; font-weight: 500; font-size: .36em; letter-spacing: .38em;
  background: linear-gradient(90deg, var(--nbs-lilac), var(--nbs-peach)); -webkit-background-clip: text;
  background-clip: text; color: transparent; }
.nbs-hero p { color: var(--nbs-muted); margin: 16px 0 0; max-width: 46ch; font-size: 15px; line-height: 1.55; }
.nbs-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 16px; }
.nbs-tags b { font-weight: 600; font-size: 12px; padding: 4px 11px; border-radius: 999px;
  border: 1px solid var(--nbs-line); color: var(--nbs-muted); background: rgba(255,255,255,.02); }
.nbs-eq { position: absolute; right: 28px; top: 30px; display: flex; gap: 5px; align-items: flex-end; height: 62px; }
.nbs-eq i { display: block; width: 6px; border-radius: 3px; height: 30%;
  background: linear-gradient(180deg, var(--nbs-peach), var(--nbs-lilac)); animation: nbs-eq 1.25s ease-in-out infinite; }
.nbs-eq i:nth-child(2) { animation-delay: -.2s } .nbs-eq i:nth-child(3) { animation-delay: -.55s }
.nbs-eq i:nth-child(4) { animation-delay: -.9s } .nbs-eq i:nth-child(5) { animation-delay: -.35s }
.nbs-eq i:nth-child(6) { animation-delay: -.7s } .nbs-eq i:nth-child(7) { animation-delay: -1.05s }
@keyframes nbs-eq { 0%, 100% { height: 18%; } 50% { height: 100%; } }
@media (prefers-reduced-motion: reduce) { .nbs-eq i { animation: none; height: 60%; } }
@media (max-width: 560px) { .nbs-eq { display: none; } .nbs-hero { padding: 22px 18px 18px; } }
.nbs-side-logo { font-family: 'Unbounded', sans-serif; font-weight: 700; font-size: 24px; line-height: 1; color: var(--nbs-fg); }
.nbs-side-logo span { display: block; font-size: 10px; letter-spacing: .38em; margin-top: 6px; font-weight: 500; color: var(--nbs-lilac); }
.nbs-step { font-family: 'Unbounded', sans-serif; font-size: 12px; letter-spacing: .14em; color: var(--nbs-lilac); text-transform: uppercase; }
</style>
"""


def inject_design():
    st.markdown(NBS_CSS, unsafe_allow_html=True)


def render_hero(text="Ремиксы с новым битом, 20+ эффектов, студийный мастеринг, музыка с нуля на нейросети, готовое видео, обложка и SEO для YouTube."):
    st.markdown(f"""
<div class="nbs-hero">
  <div class="nbs-eq"><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
  <div class="nbs-logo">NBS<span>SOFT MUSIC</span></div>
  <p>{text}</p>
  <div class="nbs-tags"><b>Ремиксы</b><b>Новый бит</b><b>8D и 20+ эффектов</b><b>Мастеринг −14 LUFS</b><b>Видео для YouTube</b><b>ИИ-генерация</b></div>
</div>""", unsafe_allow_html=True)


def step_title(num, title):
    st.markdown(f"<div class='nbs-step'>Шаг {num}</div>", unsafe_allow_html=True)
    st.subheader(title)


def render_auth():
    render_hero("Войдите или зарегистрируйтесь, чтобы открыть студию.")
    tab_login, tab_reg = st.tabs(["🔑 Вход", "📝 Регистрация"])

    with tab_login:
        with st.form("login_form"):
            user = st.text_input("Логин")
            password = st.text_input("Пароль", type="password")
            if st.form_submit_button("Войти", type="primary"):
                if verify_user(user, password):
                    st.session_state.update(logged_in=True, username=user.strip())
                    st.rerun()
                else:
                    st.error("Неверный логин или пароль.")

    with tab_reg:
        with st.form("register_form"):
            user = st.text_input("Придумайте логин")
            password = st.text_input("Придумайте пароль (от 6 символов)", type="password")
            password2 = st.text_input("Повторите пароль", type="password")
            if st.form_submit_button("Зарегистрироваться"):
                if password != password2:
                    st.error("Пароли не совпадают.")
                elif (error := register_user(user, password)):
                    st.error(error)
                else:
                    st.success("Готово! Теперь войдите во вкладке «Вход».")


# =====================================================================
# ПРОФИ ИИ-РЕМИКС: Stability AI · Stable Audio (audio-to-audio)
# =====================================================================

STABILITY_URL = "https://api.stability.ai/v2beta/audio/stable-audio-2/audio-to-audio"
AI_MAX_SECONDS = 190
AI_STYLE_PROMPTS = {
    "lofi": "lo-fi hip hop remix, dusty boom bap drums with swing, warm Rhodes electric piano chords, "
            "vinyl texture, soft deep sub bass",
    "phonk": "drift phonk remix, aggressive distorted 808 bass, Memphis cowbell melody, hard trap drums, "
             "dark atmosphere",
    "trap": "melodic trap remix, booming 808 bass, crisp hi-hat rolls, punchy claps, airy bell melody, "
            "atmospheric pads",
    "house": "deep house remix, four on the floor kick, groovy offbeat bassline, organ chord stabs, "
             "shimmering hi-hats, warm pads",
    "synth": "synthwave remix, gated reverb snare, retro analog arpeggios, lush 80s pads, driving bassline",
    "dance": "energetic sped up dance remix, punchy four on the floor drums, bright plucks, uplifting supersaw chords",
}


def get_stability_key():
    """Ключ из Secrets (Streamlit Cloud → Settings → Secrets: STABILITY_API_KEY = "...") или из поля ввода."""
    return get_setting("STABILITY_API_KEY", "")


def ai_context(username):
    """Чей ключ используется и как платится генерация: own — свой ключ, admin — безлимит, credits — баланс."""
    own = st.session_state.get("own_key", "").strip()
    if own:
        return own, "own"
    server_key = get_stability_key()
    if not server_key:
        return "", "none"
    return server_key, "admin" if is_admin(username) else "credits"


def ai_block_reason(api_key, mode, username):
    if not api_key:
        return "Нейросеть не подключена. Добавьте свой ключ Stability AI в боковой панели слева."
    if mode == "credits" and (get_user(username) or {}).get("ai_credits", 0) <= 0:
        return "На балансе 0 ИИ-генераций. " + get_setting("BUY_INFO", "Чтобы пополнить, напишите администратору.")
    return None


def charge_ai(username, mode, reason):
    return True if mode in ("own", "admin") else spend_credit(username, reason)


def refund_ai(username, mode, reason):
    if mode == "credits":
        add_credits(username, 1, "возврат: " + reason)


def build_ai_prompt(style, extra="", bpm=None, key_name=None):
    parts = [f"Professional {AI_STYLE_PROMPTS[style]}"]
    if bpm:
        parts.append(f"{bpm:.0f} BPM")
    if key_name:
        parts.append(f"key of {key_name}")
    parts.append("studio quality mix and master, high fidelity, wide stereo")
    if extra.strip():
        parts.append(extra.strip())
    return ", ".join(parts)


def stability_audio_to_audio(audio_bytes, filename, mime, prompt, strength, duration, api_key, model, seed):
    data = {"prompt": prompt, "strength": f"{strength:.2f}", "duration": str(int(duration)),
            "seed": str(int(seed)), "output_format": "mp3"}
    if model:
        data["model"] = model
    resp = requests.post(STABILITY_URL,
                         headers={"authorization": f"Bearer {api_key}", "accept": "audio/*"},
                         files={"audio": (filename, audio_bytes, mime)}, data=data, timeout=600)
    if resp.status_code == 200:
        return resp.content
    _raise_stability_error(resp)


# --- Генерация трека с нуля (text-to-audio): своя музыка без загрузки чужого файла ---

STABILITY_T2A_URL = "https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio"
GEN_MOODS = {
    "Спокойный, мечтательный": "calm, dreamy, soft, emotional",
    "Тёмный, ночной": "dark, nocturnal, moody, mysterious",
    "Энергичный": "energetic, uplifting, driving",
    "Грустный, ностальгия": "sad, melancholic, nostalgic",
    "Агрессивный": "aggressive, hard-hitting, intense",
    "Романтичный": "romantic, warm, intimate",
}
GEN_BPM = {"lofi": 80, "phonk": 130, "trap": 140, "house": 122, "synth": 105, "dance": 128}


def build_gen_prompt(style, mood, bpm, extra=""):
    core = AI_STYLE_PROMPTS[style].replace(" remix", "")
    parts = [f"Original instrumental track, {core}", mood, f"{bpm} BPM",
             "professional studio production, polished mix and master, high fidelity, wide stereo"]
    if extra.strip():
        parts.append(extra.strip())
    return ", ".join(parts)


def stability_text_to_audio(prompt, duration, api_key, model, seed):
    data = {"prompt": prompt, "duration": str(int(duration)), "seed": str(int(seed)), "output_format": "mp3"}
    if model:
        data["model"] = model
    resp = requests.post(STABILITY_T2A_URL,
                         headers={"authorization": f"Bearer {api_key}", "accept": "audio/*"},
                         files={"none": ""}, data=data, timeout=600)
    if resp.status_code == 200:
        return resp.content
    _raise_stability_error(resp)


def _raise_stability_error(resp):
    try:
        body = resp.json()
        msg = "; ".join(body.get("errors") or []) or body.get("message") or body.get("name") or resp.text
    except ValueError:
        msg = resp.text[:300]
    hints = {401: "проверьте API-ключ", 402: "на счёте Stability закончились кредиты",
             403: "запрос отклонён модерацией Stability (например, загружен защищённый трек)",
             413: "файл слишком большой", 429: "слишком много запросов, подождите минуту"}
    hint = hints.get(resp.status_code, "")
    raise RuntimeError(f"Stability AI ответил {resp.status_code}: {msg}" + (f" — {hint}" if hint else ""))


def render_admin(me):
    st.subheader("🛠 Админ-панель")
    users = list_users()
    c1, c2, c3 = st.columns(3)
    c1.metric("Пользователей", len(users))
    c2.metric("Активных", sum(1 for u in users if u["is_active"]))
    c3.metric("Генераций на балансах", sum(int(u["ai_credits"] or 0) for u in users))
    st.dataframe([{"Логин": u["username"], "Статус": "активен" if u["is_active"] else "заблокирован",
                   "ИИ-генераций": int(u["ai_credits"] or 0), "Регистрация": u["created_at"] or "—"} for u in users],
                 hide_index=True, width="stretch")
    if not users:
        st.info("Пока никто не зарегистрировался.")
        return
    sel = st.selectbox("Пользователь", [u["username"] for u in users], key="adm_user")
    u = get_user(sel)
    st.markdown(f"**{sel}** · баланс: **{int(u['ai_credits'] or 0)}** · "
                f"{'активен' if u['is_active'] else 'заблокирован'}")
    with st.form("adm_credits", clear_on_submit=True):
        c1, c2 = st.columns([1, 2])
        amount = c1.number_input("Начислить генераций (минус — списать)", -1000, 1000, 10, 1)
        note = c2.text_input("Комментарий", placeholder="Kaspi 2 000 ₸, 23.10")
        if st.form_submit_button("💳 Изменить баланс", type="primary"):
            add_credits(sel, int(amount), f"админ {me}: {note}".strip())
            st.success(f"Баланс {sel} изменён на {int(amount):+d}.")
            st.rerun()
    c1, c2 = st.columns(2)
    if sel.lower() != me.lower():
        if u["is_active"]:
            if c1.button("🚫 Заблокировать", key="adm_block"):
                set_active(sel, False)
                st.rerun()
        elif c1.button("✅ Разблокировать", key="adm_unblock"):
            set_active(sel, True)
            st.rerun()
    with c2.popover("🔑 Новый пароль"):
        new_pw = st.text_input("Новый пароль (от 6 символов)", type="password", key="adm_pw")
        if st.button("Сохранить пароль", key="adm_pw_btn"):
            if len(new_pw) < 6:
                st.error("Пароль короче 6 символов.")
            else:
                set_password(sel, new_pw)
                st.success("Пароль изменён. Передайте его пользователю.")
    hist = credit_history(sel)
    if hist:
        st.markdown("**История баланса**")
        st.dataframe([{"Время (UTC)": t, "Изменение": f"{d:+d}", "Причина": r} for t, d, r in hist],
                     hide_index=True, width="stretch")


def render_results(res):
    st.success("🎉 Готово! Трек, обложка и SEO созданы.")
    st.subheader("📥 Результаты")
    st.image(res["cover"], caption="Обложка для YouTube (1280×720)")
    st.markdown("**🎧 Сравните: до и после**")
    c1, c2 = st.columns(2)
    if res.get("orig"):
        c1.caption("Оригинал")
        c1.audio(res["orig"])
    c2.caption("Результат")
    c2.audio(res["wav"], format="audio/wav")
    rep = res.get("master")
    if rep:
        c1, c2, c3 = st.columns(3)
        c1.metric("Громкость", f"{rep['lufs_out']:.1f} LUFS", f"{rep['lufs_out'] - rep['lufs_in']:+.1f} дБ")
        c2.metric("Пик (true peak)", f"{rep['tp_out']:.1f} dBTP")
        c3.metric("Длительность", f"{int(res['duration'] // 60)}:{int(res['duration'] % 60):02d}")
        st.caption(f"Мастеринг: {rep['target']}. Пик ниже потолка — на YouTube и в плеерах не будет искажений.")
    if res.get("video"):
        st.markdown("**🎬 Видео для YouTube**")
        st.video(res["video"])
    elif res.get("video_error"):
        st.warning(f"Видео не получилось: {res['video_error']}")
    if res.get("fx"):
        st.markdown("**🎚️ Эффекты:** " + ", ".join(res["fx"]))

    beat = res.get("beat")
    if beat:
        timing = {2: " · half-time", 0.5: " · double-time"}.get(beat["mult"], "")
        st.markdown(f"**🥁 Новый бит:** {beat['style']}{timing}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Темп ремикса", f"{beat['bpm_out']:.1f} BPM",
                  None if abs(beat["bpm_out"] - beat["bpm"]) < .05 else f"{beat['bpm_out'] - beat['bpm']:+.1f}")
        c2.metric("Бас трека", beat["key"])
        c3.metric("Старые барабаны", "вырезаны" if beat["separated"] else "приглушены")
        if beat.get("chords"):
            st.markdown(f"**🎹 Аккорды трека:** {beat['chords']}")
        if beat.get("drops"):
            st.caption(f"Дропов в аранжировке: {beat['drops']}")
        st.code(beat["pattern"], language="text")

    if res.get("ai_mp3"):
        st.markdown("**🤖 Профи ИИ-версия (Stable Audio)**")
        st.audio(res["ai_mp3"], format="audio/mpeg")
        st.download_button("🤖 Скачать ИИ-ремикс (MP3)", res["ai_mp3"], f"{res['slug']}_ai.mp3",
                           "audio/mpeg", key="dl_ai")
        with st.expander("Промпт, отправленный ИИ"):
            st.code(res.get("ai_prompt", ""), language="text")
    elif res.get("ai_error"):
        st.warning(f"ИИ-версия не получилась. {res['ai_error']}")

    st.markdown("**📥 Скачать**")
    items = [("🖼️ Обложка JPG", res["cover"], f"{res['slug']}_cover.jpg", "image/jpeg"),
             (f"🎧 WAV {res.get('bits', 16)} бит", res["wav"], f"{res['slug']}.wav", "audio/wav")]
    if res.get("mp3"):
        items.append(("🎵 MP3 320 кбит/с", res["mp3"], f"{res['slug']}.mp3", "audio/mpeg"))
    if res.get("video"):
        items.append(("🎬 Видео MP4", res["video"], f"{res['slug']}.mp4", "video/mp4"))
    cols = st.columns(len(items))
    for i, (label, data, name, mime) in enumerate(items):
        cols[i].download_button(label, data, name, mime, key=f"dl_{i}", width="stretch")

    st.markdown("---")
    st.subheader("🚀 YouTube SEO (English)")
    st.markdown("**💡 Варианты названия:**")
    for t in res["titles"]:
        st.code(t, language="text")
    st.markdown("**📝 Описание:**")
    st.code(res["description"], language="text")
    st.markdown("**🏷️ Теги (через запятую):**")
    st.code(res["tags"], language="text")


def main():
    st.set_page_config(page_title="NBS SOFT MUSIC", page_icon="🎧", layout="centered")
    inject_design()
    init_db()
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("username", "")

    if not st.session_state["logged_in"]:
        render_auth()
        st.stop()

    session_keys = ("logged_in", "username", "result", "file_key", "beat_cache", "gen_track", "admin_mode")
    username = st.session_state["username"]
    user = get_user(username)
    if not user or not user["is_active"]:
        for key in session_keys:
            st.session_state.pop(key, None)
        st.error("Аккаунт заблокирован или удалён. Обратитесь к администратору.")
        st.stop()
    admin = is_admin(username)

    # --- Боковая панель: аккаунт, баланс ИИ, админка ---
    with st.sidebar:
        st.markdown("<div class='nbs-side-logo'>NBS<span>SOFT MUSIC</span></div>", unsafe_allow_html=True)
        st.write("")
        st.write(f"👤 **{username}**" + (" · администратор" if admin else ""))
        st.markdown("**🤖 ИИ-генерации**")
        if admin:
            st.caption("Безлимит (администратор)")
        else:
            st.metric("Баланс", int(user["ai_credits"] or 0))
            st.caption(get_setting("BUY_INFO", "Чтобы пополнить, напишите администратору."))
        with st.expander("Свой ключ Stability AI"):
            st.text_input("API-ключ", type="password", key="own_key",
                          help="platform.stability.ai → API Keys")
            st.caption("Со своим ключом генерации не списываются с баланса: платит ваш счёт Stability.")
        if admin:
            st.toggle("🛠 Админ-панель", key="admin_mode")
        if st.button("🚪 Выйти"):
            for key in session_keys:
                st.session_state.pop(key, None)
            st.rerun()

    api_key, ai_mode = ai_context(username)
    if admin and st.session_state.get("admin_mode"):
        render_admin(username)
        return

    render_hero()

    # --- Шаг 1. Музыка ---
    with st.container(border=True):
        step_title(1, "🎵 Музыка")
        mode = st.radio("Источник", ["upload", "generate"], horizontal=True, label_visibility="collapsed",
                        format_func={"upload": "🎵 Загрузить свой трек",
                                     "generate": "✨ Создать трек с нуля (ИИ)"}.get)
        source_bytes, source_name, source_bpm = None, None, None
        if mode == "upload":
            uploaded = st.file_uploader("Загрузите трек (.mp3 или .wav)", type=["mp3", "wav"])
            if uploaded is not None:
                source_bytes, source_name = uploaded.getvalue(), uploaded.name
            else:
                st.caption("Загрузите свой трек или трек, на который у вас есть права.")
        else:
            gen_styles = {k: BEAT_STYLES[k]["name"] for k in AI_STYLE_PROMPTS}
            c1, c2 = st.columns(2)
            g_style = c1.selectbox("Стиль", list(gen_styles), format_func=gen_styles.get, key="gen_style")
            g_mood = c2.selectbox("Настроение", list(GEN_MOODS), key="gen_mood")
            c3, c4 = st.columns(2)
            g_bpm = c3.slider("Темп, BPM", 60, 180, GEN_BPM[g_style], key=f"gen_bpm_{g_style}")
            g_dur = c4.slider("Длительность, сек", 30, AI_MAX_SECONDS, 120, 10, key="gen_dur")
            g_extra = st.text_input("Пожелания к звучанию (по-английски, необязательно)",
                                    placeholder="soft female humming, rain sounds, acoustic guitar", key="gen_extra")
            g_model = st.selectbox("Модель", ["stable-audio-2.5", "stable-audio-2"], key="gen_model")
            g_prompt = build_gen_prompt(g_style, GEN_MOODS[g_mood], g_bpm, g_extra)
            st.caption(f"Промпт для нейросети: {g_prompt}")
            cost_note = {"credits": "списывается 1 генерация с баланса", "admin": "безлимит",
                         "own": "оплата с вашего ключа Stability", "none": "нейросеть не подключена"}[ai_mode]
            if st.button(f"✨ Сгенерировать трек · {cost_note}", type="primary", key="gen_btn"):
                reason = ai_block_reason(api_key, ai_mode, username)
                if reason:
                    st.error(reason)
                elif not charge_ai(username, ai_mode, "трек с нуля"):
                    st.error(ai_block_reason(api_key, ai_mode, username) or "Не удалось списать генерацию.")
                else:
                    generated = False
                    with st.spinner("Нейросеть сочиняет трек, обычно это до минуты…"):
                        try:
                            seed = random.randint(1, 4_000_000)
                            audio = stability_text_to_audio(g_prompt, g_dur, api_key, g_model, seed)
                            st.session_state["gen_track"] = {"bytes": audio, "name": f"nbs_{g_style}_{seed}.mp3",
                                                             "prompt": g_prompt, "bpm": g_bpm}
                            st.session_state.pop("result", None)
                            generated = True
                        except Exception as err:
                            refund_ai(username, ai_mode, "трек с нуля")
                            st.error(f"{err}. Генерация возвращена на баланс." if ai_mode == "credits" else str(err))
                    if generated:
                        st.rerun()          # обновить баланс в боковой панели
            gen = st.session_state.get("gen_track")
            if gen:
                source_bytes, source_name, source_bpm = gen["bytes"], gen["name"], gen["bpm"]
                st.success("Трек готов. Это ваша оригинальная музыка: дальше её можно обработать как обычно.")
                st.download_button("⬇️ Скачать сгенерированный трек (MP3)", gen["bytes"], gen["name"],
                                   "audio/mpeg", key="dl_gen")
            else:
                st.caption("Выберите стиль и настроение и нажмите «Сгенерировать».")
        if source_bytes is not None:
            st.audio(source_bytes)

    if source_bytes is None:
        return

    file_key = f"{source_name}-{len(source_bytes)}-{hashlib.md5(source_bytes[:1_000_000]).hexdigest()[:10]}"
    if st.session_state.get("file_key") != file_key:
        st.session_state["file_key"] = file_key
        st.session_state.pop("result", None)

    # --- Шаг 2. Релиз и обложка ---
    with st.container(border=True):
        step_title(2, "🎨 Релиз и обложка")
        genre_label = st.selectbox("Ниша / стиль канала", list(GENRES))
        genre = GENRES[genre_label]
        c1, c2 = st.columns(2)
        song_title = c1.text_input("Название трека", "I WELCOME OCTOBER (Remix)")
        artist_name = c2.text_input("Артист", "Smagulov")
        genre_text = st.text_input("Текст жанра на обложке", genre["short"], key=f"gtext_{genre_label}")
        bg_file = st.file_uploader("Свой фон для обложки (необязательно)", type=["jpg", "jpeg", "png", "webp"])

    # --- Шаг 3. Скорость и звучание ---
    with st.container(border=True):
        step_title(3, "🎛️ Скорость и звучание")
        preset_names = list(PRESETS)
        preset_name = st.selectbox("Пресет", preset_names, index=preset_names.index(genre["preset"]),
                                   key=f"preset_{genre_label}")
        p = PRESETS[preset_name]
        k = f"{genre_label}_{preset_name}"  # ключи сбрасывают ползунки при смене пресета
        with st.expander("Тонкая настройка", expanded=preset_name == "Свои настройки"):
            link = st.checkbox("Эффект пластинки: тон меняется вместе со скоростью",
                               value=p["link"], key=f"link_{k}")
            if link:
                speed = st.slider("Скорость", 0.70, 1.40, p["speed"], 0.01, key=f"speed_{k}")
                tempo, pitch = 1.0, 0.0
            else:
                speed = 1.0
                tempo = st.slider("Темп (без изменения тона)", 0.75, 1.30, p["tempo"], 0.01, key=f"tempo_{k}")
                pitch = st.slider("Тональность, полутоны", -6.0, 6.0, p["pitch"], 0.5, key=f"pitch_{k}")
            reverb = st.slider("Реверберация", 0.0, 1.0, p["reverb"], 0.05, key=f"reverb_{k}")
            bass = st.slider("Усиление баса", 0.0, 1.0, p["bass"], 0.05, key=f"bass_{k}")
            vinyl = st.slider("Винил (тепло, шипение, щелчки)", 0.0, 1.0, p["vinyl"], 0.05, key=f"vinyl_{k}")
    settings = dict(link=link, speed=speed, tempo=tempo, pitch=pitch, reverb=reverb, bass=bass, vinyl=vinyl)

    # --- Шаг 4. Новый бит и слои ---
    with st.container(border=True):
        step_title(4, "🥁 Новый бит и музыкальные слои")
        beat_on = st.checkbox("Сделать новый бит", value=True,
                              help="Определяет темп, сильные доли и басовые ноты трека и пишет под них свой ритм.")
        style_labels = {"auto": f"Как у жанра — {BEAT_STYLES[genre['beat']]['name']}"}
        style_labels.update({key: s["name"] for key, s in BEAT_STYLES.items()})
        style_choice = st.selectbox("Стиль бита", list(style_labels), format_func=style_labels.get,
                                    key=f"beatstyle_{genre_label}", disabled=not beat_on)
        style_key = genre["beat"] if style_choice == "auto" else style_choice
        bst = BEAT_STYLES[style_key]
        kb = f"{genre_label}_{style_key}"

        has_demucs = demucs_available()
        use_sep = st.checkbox("Вырезать старые барабаны нейросетью (Demucs)", value=has_demucs,
                              disabled=not (beat_on and has_demucs),
                              help="Делит трек на вокал, бас, барабаны и остальное; старые барабаны убираются полностью.")
        if not has_demucs:
            st.caption("Demucs не установлен: старый бит будет приглушён фильтром.")

        st.session_state.setdefault("beat_seed", 1)
        with st.expander("Настройки бита"):
            beat_vol = st.slider("Громкость нового бита", 0.0, 1.5, 0.9, 0.05, key=f"bvol_{kb}")
            beat_filter = st.slider("Приглушить старый бит (фильтр)", 0.0, 1.0, bst["filter"], 0.05,
                                    key=f"bfilter_{kb}", disabled=use_sep,
                                    help="Работает, когда Demucs выключен. Чем больше, тем ниже срез частот.")
            beat_side = st.slider("Качание под бочку", 0.0, 1.0, bst["side"], 0.05, key=f"bside_{kb}")
            beat_density = st.slider("Сложность и сбивки", 0.0, 1.0, 0.5, 0.05, key=f"bdens_{kb}")
            beat_bass = st.checkbox("Новый бас (808 / саб) по нотам трека", value=True, key=f"bbass_{kb}")
            tempo_choice = st.radio("Темп бита", ["Как найден", "½× (в 2 раза медленнее)", "2× (в 2 раза быстрее)"],
                                    horizontal=True, help="Если бит спешит или тянется, переключите здесь.")
            c1, c2 = st.columns([2, 1])
            c1.number_input("Номер вариации бита", 1, 99999, key="beat_seed",
                            help="Тот же номер даёт тот же бит, другой — новый ритм.")
            c2.button("🎲 Другая", on_click=lambda: st.session_state.update(beat_seed=random.randint(1, 99999)))
        beat_scale = {"Как найден": 1, "½× (в 2 раза медленнее)": 0.5}.get(tempo_choice, 2)

        lst = LAYER_STYLES[style_key]
        keys_names = {"rhodes": "электропиано", "bell": "колокольчики", "organ": "хаус-орган",
                      "arp": "арпеджио-синт", "pluck": "плак-синт"}
        layers_on = st.checkbox("🎹 Добавить музыкальные слои по аккордам трека", value=True, disabled=not beat_on,
                                help="Аккорды и мелодия строятся по гармонии трека, поэтому звучат в его тональности.")
        with st.expander("Слои, переходы и сведение"):
            layer_pad = st.slider("Аккорды (пэд)", 0.0, 1.0, lst["pad"], 0.05, key=f"lpad_{kb}")
            layer_keys = st.slider(f"Клавиши: {keys_names[lst['keys_type']]}", 0.0, 1.0, lst["keys"], 0.05,
                                   key=f"lkeys_{kb}")
            layer_echo = st.slider("Эхо клавиш в темп", 0.0, 1.0, lst["echo"], 0.05, key=f"lecho_{kb}")
            layer_fx = st.checkbox("Переходы: брейк, райзер, реверс-тарелка и удар на дропе", value=True, key=f"lfx_{kb}")
            layer_polish = st.checkbox("Красивое сведение: склейка и широкое стерео", value=True, key=f"lpol_{kb}")
        layers = dict(on=beat_on and layers_on, pad=layer_pad, keys=layer_keys, echo=layer_echo,
                      fx=layer_fx, polish=layer_polish)

    # --- Шаг 5. Эффекты ---
    with st.container(border=True):
        step_title(5, "🎚️ Эффекты")
        fx_preset = st.selectbox("Набор эффектов", list(FX_PRESETS), key="fx_preset",
                                 help="Готовый набор выставляет ползунки ниже; дальше их можно менять.")
        base = {**FX_DEFAULTS, **FX_PRESETS[fx_preset]}
        kf = f"fx_{fx_preset}_"
        fx = {}
        t1, t2, t3, t4, t5 = st.tabs(["Тембр", "Окраска", "Движение", "Пространство", "Ритм и финал"])
        with t1:
            c1, c2, c3 = st.columns(3)
            fx["eq_low"] = c1.slider("Низы, дБ", -12.0, 12.0, float(base["eq_low"]), .5, key=kf + "eq_low")
            fx["eq_mid"] = c2.slider("Середина, дБ", -12.0, 12.0, float(base["eq_mid"]), .5, key=kf + "eq_mid")
            fx["eq_high"] = c3.slider("Верха, дБ", -12.0, 12.0, float(base["eq_high"]), .5, key=kf + "eq_high")
            fx["hp"] = st.slider("Срез низов, Гц (20 — выключен)", 20, 1000, int(base["hp"]), 10, key=kf + "hp")
            fx["lp"] = st.slider("Срез верхов, Гц (20000 — выключен)", 500, 20000, int(base["lp"]), 100, key=kf + "lp",
                                 help="Низкие значения дают эффект «из-за стены» или «под водой».")
        with t2:
            fx["saturation"] = st.slider("Ламповая теплота (сатурация)", 0.0, 1.0, float(base["saturation"]), .05, key=kf + "sat")
            fx["distortion"] = st.slider("Дисторшн", 0.0, 1.0, float(base["distortion"]), .05, key=kf + "dist")
            fx["bitcrush"] = st.slider("Биткраш (звук ретро-приставки)", 0.0, 1.0, float(base["bitcrush"]), .05, key=kf + "bit")
            fx["radio"] = st.slider("Старое радио / телефон", 0.0, 1.0, float(base["radio"]), .05, key=kf + "radio")
        with t3:
            fx["chorus"] = st.slider("Хорус (объём, «плывущий» звук)", 0.0, 1.0, float(base["chorus"]), .05, key=kf + "chorus")
            fx["flanger"] = st.slider("Фленжер (реактивный «вжух»)", 0.0, 1.0, float(base["flanger"]), .05, key=kf + "flanger")
            fx["phaser"] = st.slider("Фэйзер (вращение)", 0.0, 1.0, float(base["phaser"]), .05, key=kf + "phaser")
            fx["autofilter"] = st.slider("Свип-фильтр в темп", 0.0, 1.0, float(base["autofilter"]), .05, key=kf + "af")
            fx["tremolo"] = st.slider("Тремоло (пульсация громкости)", 0.0, 1.0, float(base["tremolo"]), .05, key=kf + "trem")
        with t4:
            fx["eight_d"] = st.slider("8D Audio (звук вокруг головы, слушать в наушниках)", 0.0, 1.0,
                                      float(base["eight_d"]), .05, key=kf + "8d")
            fx["autopan"] = st.slider("Автопанорама (лево–право в темп)", 0.0, 1.0, float(base["autopan"]), .05, key=kf + "pan")
            fx["width"] = st.slider("Ширина стерео (1 — как есть, 0 — моно)", 0.0, 2.0, float(base["width"]), .05, key=kf + "width")
            fx["room"] = st.slider("Зал (дополнительная реверберация)", 0.0, 1.0, float(base["room"]), .05, key=kf + "room")
            fx["echo"] = st.slider("Эхо в темп", 0.0, 1.0, float(base["echo"]), .05, key=kf + "echo")
            c1, c2 = st.columns(2)
            echo_opts = {0.25: "1/16", 0.5: "1/8", 0.75: "пунктирная 1/8", 1.0: "1/4"}
            fx["echo_beats"] = c1.selectbox("Шаг эха", list(echo_opts), format_func=echo_opts.get,
                                            index=list(echo_opts).index(base["echo_beats"]), key=kf + "eb")
            fx["echo_fb"] = c2.slider("Повторы эха", 0.1, 0.8, float(base["echo_fb"]), .05, key=kf + "efb")
        with t5:
            fx["gate"] = st.slider("Транс-гейт (рубленый ритм 1/16)", 0.0, 1.0, float(base["gate"]), .05, key=kf + "gate")
            fx["comp"] = st.slider("Компрессор (плотнее и громче)", 0.0, 1.0, float(base["comp"]), .05, key=kf + "comp")
            fx["tapestop"] = st.slider("Тейп-стоп в конце (плёнка останавливается)", 0.0, 1.0,
                                       float(base["tapestop"]), .05, key=kf + "tape")
        chosen = active_fx(fx)
        st.caption("Включено: " + (", ".join(chosen) if chosen else "ничего"))

    # --- Шаг 6. Профи ИИ-ремикс ---
    with st.container(border=True):
        step_title(6, "🤖 Профи ИИ-ремикс")
        ai_on = st.checkbox("Перегенерировать нейросетью Stable Audio", value=False,
                            help="Нейросеть берёт трек как основу и заново исполняет его в выбранном стиле: "
                                 "живее и «дороже» по звуку. 20 кредитов Stability (≈ $0.20) за генерацию.")
        ai_source, ai_strength, ai_extra, ai_model = "remix", 0.75, "", "stable-audio-2.5"
        if ai_on:
            reason = ai_block_reason(api_key, ai_mode, username)
            if reason:
                st.warning(reason)
            elif ai_mode == "credits":
                st.caption("За ИИ-ремикс с баланса списывается 1 генерация (если нейросеть вернёт ошибку — вернётся).")
            ai_source = st.radio("Что отправить нейросети", ["remix", "original"], horizontal=True,
                                 format_func={"remix": "Готовый трек со всеми эффектами (рекомендуется)",
                                              "original": "Исходный трек"}.get)
            ai_strength = st.slider("Насколько сильно переделать", 0.5, 0.95, 0.75, 0.05,
                                    help="0.6–0.7 — близко к оригиналу; 0.8–0.9 — смелее, больше нового.")
            ai_extra = st.text_input("Пожелания к звучанию (по-английски, необязательно)",
                                     placeholder="female vocal chops, rainy night mood, more reverb")
            ai_model = st.selectbox("Модель", ["stable-audio-2.5", "stable-audio-2"], key="ai_model")
            st.caption(f"Нейросеть обрабатывает до {AI_MAX_SECONDS // 60} мин {AI_MAX_SECONDS % 60} с трека. "
                       "Загружайте только свои треки или треки с лицензией: Stability проверяет загрузки.")

    # --- Шаг 7. Мастеринг и экспорт ---
    has_ffmpeg = shutil.which("ffmpeg") is not None
    with st.container(border=True):
        step_title(7, "💎 Мастеринг и экспорт")
        master_on = st.checkbox("Профессиональный мастеринг", value=True,
                                help="Ровная громкость по стандарту площадки, лимитер без искажений, плотный бас.")
        tnames = {key: t["name"] for key, t in MASTER_TARGETS.items()}
        m_target = st.selectbox("Громкость под площадку", list(tnames), format_func=tnames.get,
                                disabled=not master_on, key="m_target")
        with st.expander("Тонкая настройка мастеринга"):
            m_glue = st.slider("Склейка (шинный компрессор)", 0.0, 1.0, 0.3, 0.05, key="m_glue",
                               help="Собирает все звуки в один плотный микс. Больше — плотнее, меньше — живее.")
            m_mono = st.checkbox("Плотный бас: всё ниже 120 Гц в центр", value=True, key="m_mono")
            m_warm = st.slider("Теплота (середина-низ)", 0.0, 1.0, 0.0, 0.05, key="m_warm")
            m_air = st.slider("Воздух (блеск верхов)", 0.0, 1.0, 0.0, 0.05, key="m_air")
        master_opts = dict(on=master_on, target=m_target, glue=m_glue, mono_bass=m_mono, warmth=m_warm, air=m_air)

        st.markdown("**✂️ Обрезка и плавность**")
        c1, c2 = st.columns(2)
        trim_start = c1.number_input("Начать с, сек", 0.0, 900.0, 0.0, 0.5, key="trim_start")
        trim_end = c2.number_input("Закончить на, сек (0 — до конца)", 0.0, 900.0, 0.0, 0.5, key="trim_end")
        c1, c2 = st.columns(2)
        fade_in = c1.slider("Плавное начало, сек", 0.0, 10.0, 0.0, 0.5, key="fade_in")
        fade_out = c2.slider("Плавный конец, сек", 0.0, 15.0, 3.0, 0.5, key="fade_out")

        st.markdown("**📦 Форматы**")
        wav24 = st.checkbox("WAV 24 бит (студийный; иначе 16 бит с дизерингом)", value=False, key="wav24")
        make_mp4 = st.checkbox("🎬 Видео для YouTube: MP4 1080p, обложка + живая волна звука",
                               value=has_ffmpeg, disabled=not has_ffmpeg, key="make_mp4")
        if not has_ffmpeg:
            st.caption("Видео собирается на сервере с ffmpeg (Render, свой сервер).")

    if st.button("🚀 Создать трек, обложку и SEO", type="primary", width="stretch"):
        if trim_end and trim_end <= trim_start:
            st.warning("Конец обрезки должен быть позже начала.")
            st.stop()
        if not song_title.strip():
            st.warning("Введите название трека.")
            st.stop()

        suffix = Path(source_name).suffix.lower() or ".mp3"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(source_bytes)
            tmp.close()

            with st.status("Обработка…", expanded=True) as status:
                st.write("🔄 Загружаем аудио…")
                if librosa.get_duration(path=tmp.name) > MAX_TRACK_MINUTES * 60:
                    status.update(label="Трек слишком длинный", state="error")
                    st.error(f"Трек длиннее {MAX_TRACK_MINUTES} минут.")
                    st.stop()
                y, sr = load_audio(tmp.name)
                i0 = int(trim_start * sr)
                i1 = int(trim_end * sr) if trim_end else y.shape[1]
                if i0 >= y.shape[1] - sr:
                    status.update(label="Неверная обрезка", state="error")
                    st.error("Начало обрезки дальше конца трека.")
                    st.stop()
                y = y[:, i0:i1]
                y_original = y

                beat_info = None
                if beat_on:
                    # Анализ и разделение на дорожки дорогие — кэшируем на текущий файл
                    cache = st.session_state.get("beat_cache")
                    cache_key = (file_key, use_sep, trim_start, trim_end)
                    if not cache or cache["key"] != cache_key:
                        stems = None
                        if use_sep:
                            st.write("🧠 Делим трек на дорожки (Demucs). На обычном процессоре это 1–3 минуты…")
                            try:
                                stems = separate(y, sr, get_demucs_model())
                            except Exception as err:
                                st.warning(f"Demucs не сработал ({err}). Приглушаю старый бит фильтром.")
                        st.write("📈 Ищем темп, сильные доли, басовые ноты и аккорды…")
                        analysis = analyze(y, sr, drums=stems["drums"] if stems else None,
                                           bass=stems["bass"] if stems else None,
                                           harm=stems["other"] if stems else None)
                        bed = {"main": stems["vocals"] + stems["other"], "bass": stems["bass"]} if stems else None
                        cache = {"key": cache_key, "analysis": analysis, "bed": bed}
                        st.session_state["beat_cache"] = cache
                    st.write(f"🥁 Пишем и сводим бит: {bst['name']}…")
                    y, beat_info = render_beat_mix(
                        y, sr, cache["analysis"], style_key, bed=cache["bed"], vol=beat_vol,
                        filter_amt=beat_filter, side=beat_side, density=beat_density,
                        new_bass=beat_bass, seed=int(st.session_state["beat_seed"]), scale=beat_scale,
                        layers=layers)
                    beat_info["bpm_out"] = beat_info["bpm"] * (speed if link else tempo)

                # Темп для эффектов «в темп» (эхо, гейт, тремоло, свип, автопанорама)
                fx_bpm = beat_info["bpm_out"] if beat_info else None
                if fx_bpm is None and any(fx.get(name, 0) > 0 for name in SYNCED_FX):
                    base_bpm = source_bpm or float(np.atleast_1d(librosa.beat.beat_track(y=y.mean(0), sr=sr)[0])[0])
                    fx_bpm = (base_bpm or 120.0) * (speed if link else tempo)

                st.write("🎚️ Эффекты и мастеринг…" if master_on else "🎚️ Применяем эффекты…")
                y, master_report = process_audio(y, sr, settings, fx, fx_bpm or 120.0, master=master_opts,
                                                 fades=(max(fade_in, 0.02), max(fade_out, 0.05)))
                bits = 24 if wav24 else 16
                wav = encode_audio(y, sr, "WAV", bits=bits)
                try:
                    mp3 = encode_audio(y, sr, "MP3")
                except Exception:  # нет ни ffmpeg, ни MP3 в libsndfile — отдаём только WAV
                    mp3 = None

                ai_mp3, ai_error, ai_prompt = None, None, ""
                if ai_on:
                    block = ai_block_reason(api_key, ai_mode, username)
                    if block:
                        ai_error = block
                    elif not charge_ai(username, ai_mode, "профи ИИ-ремикс"):
                        ai_error = ai_block_reason(api_key, ai_mode, username) or "Не удалось списать генерацию."
                    else:
                        st.write("🤖 Нейросеть Stable Audio перегенерирует трек (обычно до минуты)…")
                        src = y if ai_source == "remix" else y_original
                        src = src[:, :int(AI_MAX_SECONDS * sr)]
                        try:
                            payload, fname, mime = encode_audio(src, sr, "MP3"), "track.mp3", "audio/mpeg"
                        except Exception:
                            payload, fname, mime = encode_audio(src, sr, "WAV"), "track.wav", "audio/wav"
                        key_name = None
                        if beat_info and st.session_state.get("beat_cache"):
                            an = st.session_state["beat_cache"]["analysis"]
                            same = an["roots"] == an["key"]
                            minor_arr = an.get("minor")
                            is_minor = bool(minor_arr[same].mean() > .5) if minor_arr is not None and same.any() else False
                            key_name = NOTE_NAMES[an["key"]] + (" minor" if is_minor else " major")
                        bpm_out = fx_bpm if ai_source == "remix" else (beat_info["bpm"] if beat_info else source_bpm)
                        ai_prompt = build_ai_prompt(style_key, ai_extra, bpm_out, key_name)
                        try:
                            ai_mp3 = stability_audio_to_audio(
                                payload, fname, mime, ai_prompt, ai_strength,
                                min(AI_MAX_SECONDS, src.shape[1] / sr), api_key, ai_model,
                                int(st.session_state.get("beat_seed", 1)))
                        except Exception as err:
                            refund_ai(username, ai_mode, "профи ИИ-ремикс")
                            ai_error = str(err) + (". Генерация возвращена на баланс." if ai_mode == "credits" else "")

                st.write("🎨 Рисуем обложку…")
                background = None
                if bg_file is not None:
                    try:
                        background = Image.open(bg_file).convert("RGB")
                    except OSError:
                        st.warning("Не удалось открыть ваш фон — использую стандартный.")
                background = background or fetch_background(genre["photos"]) or gradient_background(genre["accent"])
                cover = make_cover(song_title, artist_name, genre_text, genre["accent"], background)

                video, video_error = None, None
                if make_mp4 and has_ffmpeg:
                    st.write("🎬 Собираем видео для YouTube (для 3-минутного трека — около 2–5 минут)…")
                    try:
                        video = make_video(cover, encode_audio(y, sr, "WAV", bits=16), genre["accent"])
                    except Exception as err:
                        video_error = str(err)

                st.write("🚀 Готовим SEO…")
                titles, description, tags = make_seo(song_title, artist_name, genre["short"], p["label"])
                status.update(label="Готово", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Ошибка обработки: {e}")
            st.stop()
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass

        # Сохраняем в сессии, чтобы результаты не пропадали после нажатия «Скачать»
        st.session_state["result"] = dict(
            wav=wav, mp3=mp3, cover=cover, titles=titles, description=description,
            tags=tags, slug=slugify(f"{artist_name}_{song_title}"), beat=beat_info,
            ai_mp3=ai_mp3, ai_error=ai_error, ai_prompt=ai_prompt, fx=active_fx(fx),
            orig=source_bytes, master=master_report, bits=bits, duration=y.shape[1] / sr,
            video=video, video_error=video_error,
        )

    if "result" in st.session_state:
        with st.container(border=True):
            render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
