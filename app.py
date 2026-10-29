"""AI YouTube Music & Remix Studio.

Запуск:  streamlit run app.py
"""

import functools
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
    "DISCO / ДИСКОТЕКА (Танцпол, 120 BPM)": {
        "short": "Disco", "preset": "Disco", "beat": "disco",
        "photos": [1015, 1060, 1068], "accent": (255, 205, 80),
    },
    "POP REMIX (Яркий радио-звук)": {
        "short": "Pop Remix", "preset": "Bright Pop", "beat": "dance",
        "photos": [1016, 1042, 1015], "accent": (255, 150, 90),
    },
}

# Пресеты ремикса. link=True — «эффект пластинки»: тон меняется вместе со скоростью.
PRESETS = {
    "Disco": {"label": "Disco Remix", "link": False, "speed": 1.0,
              "tempo": 1.0, "pitch": 0.0, "reverb": 0.12, "vinyl": 0.0, "bass": 0.1},
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
    "Bright Pop": {"label": "Pop Remix", "link": False, "speed": 1.0,
                   "tempo": 1.04, "pitch": 2.0, "reverb": 0.15, "vinyl": 0.0, "bass": 0.25},
    "Deep / Dark": {"label": "Deep Dark Remix", "link": False, "speed": 1.0,
                    "tempo": 0.94, "pitch": -3.0, "reverb": 0.35, "vinyl": 0.0, "bass": 0.5},
    "Свои настройки": {"label": "Remix", "link": False, "speed": 1.0,
                       "tempo": 1.0, "pitch": 0.0, "reverb": 0.0, "vinyl": 0.0, "bass": 0.0},
}

# =====================================================================
# ПОЛЬЗОВАТЕЛИ (SQLite + PBKDF2 с солью)
# =====================================================================


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)   # фоновые задачи и интерфейс пишут одновременно
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


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
        conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
                            id TEXT PRIMARY KEY, username TEXT, kind TEXT, title TEXT,
                            status TEXT, progress REAL DEFAULT 0, message TEXT,
                            created_at TEXT DEFAULT (datetime('now')), finished_at TEXT, result TEXT
                        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS presets (
                            username TEXT, name TEXT, data TEXT, created_at TEXT DEFAULT (datetime('now')),
                            PRIMARY KEY (username, name)
                        )""")
        conn.execute("CREATE TABLE IF NOT EXISTS yt_cache (q TEXT PRIMARY KEY, ts REAL, data TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token TEXT PRIMARY KEY, username TEXT, expires REAL)")
    with closing(get_conn()) as conn:
        try:
            conn.execute("PRAGMA journal_mode=WAL")      # чтение не блокирует запись
        except sqlite3.OperationalError:
            pass


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


LOGIN_DAYS = 30


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()    # в базе только хэш — утечка базы не даёт войти


def create_login_token(username):
    """«Запомнить меня»: после перезагрузки страницы или обрыва связи вход восстанавливается сам."""
    import time as _t
    token = secrets.token_urlsafe(24)
    with closing(get_conn()) as conn, conn:
        conn.execute("DELETE FROM auth_tokens WHERE expires < ?", (_t.time(),))
        conn.execute("INSERT INTO auth_tokens (token, username, expires) VALUES (?, ?, ?)",
                     (_token_hash(token), username, _t.time() + LOGIN_DAYS * 86400))
    return token


def user_from_token(token):
    import time as _t
    if not token or len(token) > 100:
        return None
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT t.username FROM auth_tokens t JOIN users u ON u.username = t.username "
                           "WHERE t.token = ? AND t.expires > ? AND u.is_active = 1",
                           (_token_hash(token), _t.time())).fetchone()
    return row[0] if row else None


def delete_login_token(token):
    if token:
        with closing(get_conn()) as conn, conn:
            conn.execute("DELETE FROM auth_tokens WHERE token = ?", (_token_hash(token),))


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
    sos = butter(order, cutoff, btype=btype, fs=sr, output="sos").astype(np.float32)
    return sosfilt(sos, np.asarray(y, dtype=np.float32), axis=-1)   # float32 — вдвое меньше памяти


def _sosf(sos, y):
    return sosfilt(np.asarray(sos, np.float32), np.asarray(y, dtype=np.float32), axis=-1)


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
    from fractions import Fraction
    fr = Fraction(1 / factor).limit_denominator(400)      # точность скорости лучше 0,1%
    return resample_poly(np.asarray(y, np.float32), fr.numerator, fr.denominator, axis=-1).astype(np.float32)


def rubberband_available():
    return shutil.which("rubberband") is not None


def _rubberband(y, sr, pitch, tempo, formant=False):
    """Студийная смена темпа/тона через Rubber Band (движок R3): чище, без «фазового» призвука.
    formant=True — тембр голоса сохраняется (нет «бурундука» и «великана»)."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        sf.write(src, y.T, sr, subtype="FLOAT")
        base = ["rubberband", "-q", "-T", f"{tempo:.6f}", "-p", f"{pitch:.4f}"] + (["-F"] if formant and pitch else [])
        for engine in (["--fine"], []):            # R3, если версия умеет; иначе стандартный движок
            res = subprocess.run(base + engine + [src, dst], capture_output=True, timeout=900)
            if res.returncode == 0 and os.path.exists(dst):
                out, _ = sf.read(dst, dtype="float32", always_2d=True)
                return out.T
        raise RuntimeError(res.stderr.decode(errors="ignore")[:300])


def pv_stretch(y, rate, n_fft=4096, hop=1024):
    """Смена темпа без изменения тона: фазовый вокодер с фиксацией фаз (identity phase locking).
    Фазы считаются по «середине» стерео и одинаково применяются к обоим каналам — стерео не «плывёт».
    rate > 1 — быстрее, < 1 — медленнее."""
    y = np.asarray(y, np.float32)
    if y.ndim == 1:
        y = y[None, :]
    if abs(rate - 1) < 1e-3:
        return y
    win = np.hanning(n_fft).astype(np.float32)
    pad = np.pad(y, ((0, 0), (n_fft, n_fft)))
    frames = 1 + (pad.shape[1] - n_fft) // hop
    X = np.stack([np.fft.rfft(np.lib.stride_tricks.sliding_window_view(pad[c], n_fft)[::hop][:frames] * win, axis=1)
                  for c in range(y.shape[0])]).astype(np.complex64)
    M = X.mean(0)
    mag_c, ph_c = np.abs(X), np.angle(X)
    ph_m, mag_m = np.angle(M), np.abs(M)
    bins = X.shape[2]
    omega = (2 * np.pi * hop * np.arange(bins) / n_fft).astype(np.float32)
    steps = np.arange(0, frames - 1, rate)
    out_len = len(steps) * hop + n_fft
    out = np.zeros((y.shape[0], out_len), np.float32)
    norm = np.zeros(out_len, np.float32)
    acc = ph_m[0].copy()
    idx = np.arange(bins)
    for i, t in enumerate(steps):
        k = int(t)
        a = np.float32(t - k)
        m = (1 - a) * mag_m[k] + a * mag_m[k + 1]
        # пики спектра и «зоны влияния» каждого пика (граница — середина между пиками)
        pk = np.where((m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]))[0] + 1
        if len(pk) == 0:
            pk = np.array([int(np.argmax(m))])
        bounds = (pk[:-1] + pk[1:]) // 2
        owner = pk[np.searchsorted(bounds, idx, side="right")]
        out_ph = acc[owner] + (ph_m[k] - ph_m[k][owner])
        rot = out_ph - ph_m[k]
        for c in range(y.shape[0]):
            mc = (1 - a) * mag_c[c, k] + a * mag_c[c, k + 1]
            frame = np.fft.irfft(mc * np.exp(1j * (ph_c[c, k] + rot)), n_fft).astype(np.float32) * win
            out[c, i * hop:i * hop + n_fft] += frame
        norm[i * hop:i * hop + n_fft] += win ** 2
        dphi = ph_m[k + 1] - ph_m[k] - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        acc = out_ph + omega + dphi
    out /= np.maximum(norm, 1e-3)
    target = int(round(y.shape[1] / rate))
    return out[:, n_fft:n_fft + target]


def stretch_tempo(y, sr, rate):
    """Темп ×rate без изменения тона: Rubber Band (студийный движок), если есть, иначе фазовый вокодер."""
    if abs(rate - 1) < 1e-3:
        return np.asarray(y, np.float32)
    if rubberband_available():
        try:
            return _rubberband(np.asarray(y, np.float32), sr, 0.0, rate).astype(np.float32)
        except Exception:
            pass
    return pv_stretch(y, rate)


HZ432_SEMITONES = 12 * np.log2(432 / 440)        # −0,318 полутона: строй A = 432 Гц


def _envelope(logmag, lifter, iters=12):
    """Спектральная огибающая (форманты) кадра — «истинная огибающая»: кепстральное сглаживание,
    поднятое до вершин гармоник (обычное сглаживание проваливается между гармониками и зависит от ноты)."""
    a = logmag.copy()
    for _ in range(iters):
        c = np.fft.irfft(a, axis=-1)
        c[..., lifter:-lifter] = 0
        v = np.fft.rfft(c, axis=-1).real[..., :logmag.shape[-1]]
        a = np.maximum(logmag, v)
    return v


LIFTER_MS = 2.2


def formant_fix(y_out, y_ref, sr, rate=1.0, n_fft=2048, hop=512, max_db=15.0, batch=256):
    """Возвращает сигналу после смены тона тембр оригинала: огибающая спектра (форманты голоса)
    берётся из y_ref, тонкая структура (ноты) — из y_out. rate — во сколько раз y_out быстрее y_ref.
    Считается пачками кадров — память не растёт с длиной трека."""
    y_out, y_ref = np.asarray(y_out, np.float32), np.asarray(y_ref, np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    lifter = max(8, int(sr * LIFTER_MS / 1000))  # 2,2 мс: F1 и F2 голоса различимы, гармоники — нет
    lim = max_db / 20 * np.log(10)
    low_bins = int(80 / sr * n_fft)
    half = n_fft // 2
    ref_pad = np.pad(y_ref.mean(0), (half, half + n_fft))
    n_frames = y_out.shape[1] // hop + 1
    out = np.zeros_like(y_out)
    for c in range(y_out.shape[0]):
        pad = np.pad(y_out[c], (half, half + n_fft))
        acc = np.zeros(len(pad), np.float32)
        norm = np.zeros(len(pad), np.float32)
        for b0 in range(0, n_frames, batch):
            ii = np.arange(b0, min(n_frames, b0 + batch))
            X = np.fft.rfft(np.stack([pad[i * hop:i * hop + n_fft] for i in ii]) * win, axis=1)
            jj = np.clip(np.round(ii * rate).astype(int), 0, (len(ref_pad) - n_fft) // hop)
            R = np.fft.rfft(np.stack([ref_pad[j * hop:j * hop + n_fft] for j in jj]) * win, axis=1)
            env_r = _envelope(np.log(np.abs(R) + 1e-6), lifter)
            env_o = _envelope(np.log(np.abs(X) + 1e-6), lifter)
            g = np.exp(np.clip(env_r - env_o, -lim, lim))
            g[:, :low_bins] = 1.0                 # бас и бочку не трогаем
            fr = np.fft.irfft(X * g, n_fft, axis=1).astype(np.float32) * win
            for k, i in enumerate(ii):
                acc[i * hop:i * hop + n_fft] += fr[k]
                norm[i * hop:i * hop + n_fft] += win ** 2
        acc /= np.maximum(norm, 1e-3)
        out[c] = acc[half:half + y_out.shape[1]]
    return out


def shift_pitch_and_tempo(y, sr, pitch, tempo, formant=False):
    """Тон (полутоны) и темп (×) независимо. formant=True — голос остаётся живым при сильном сдвиге."""
    if not pitch and abs(tempo - 1.0) <= 1e-3:
        return y
    if rubberband_available():
        try:
            return _rubberband(y, sr, pitch, tempo, formant)
        except Exception:
            pass                                    # запасной путь — свой фазовый вокодер
    y = np.asarray(y, np.float32)
    ref = y
    if pitch:
        q = 2 ** (pitch / 12)
        y = change_speed(pv_stretch(y, tempo / q), sr, q)   # растянуть, затем «пластинкой» поднять тон
    elif abs(tempo - 1.0) > 1e-3:
        y = pv_stretch(y, tempo)
    if formant and abs(pitch) >= .5:
        y = formant_fix(y, ref, sr, rate=tempo)
    return y


def vinyl_speed(y, sr, factor, formant=False):
    """Эффект пластинки (тон и скорость вместе). formant=True — тот же сдвиг, но тембр голоса живой."""
    if abs(factor - 1.0) < 1e-3:
        return y
    if not formant or abs(12 * np.log2(factor)) < .5:
        return change_speed(y, sr, factor)
    if rubberband_available():
        try:
            return _rubberband(np.asarray(y, np.float32), sr, 12 * np.log2(factor), factor, True)
        except Exception:
            pass
    return formant_fix(change_speed(y, sr, factor), y, sr, rate=factor)


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
        y = _sosf(_rbj("lowshelf", 120, sr, low), y)
    if mid:
        y = _sosf(_rbj("peak", 1000, sr, mid, q=.8), y)
    if high:
        y = _sosf(_rbj("highshelf", 6000, sr, high), y)
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
# ПРОВЕРКА ТРЕКА: клиппинг, громкость, баланс частот, стерео, тишина
# =====================================================================


def diagnose(y, sr):
    """Возвращает (показатели, список замечаний). Замечание: (уровень, заголовок, совет)."""
    if y.ndim == 1:
        y = y[None, :]
    mono = y.mean(0)
    loudest = np.abs(y).max(0)
    dur = y.shape[1] / sr
    lufs = integrated_lufs(y, sr)
    tp = true_peak_db(y)
    clip = np.abs(y) >= .999
    runs = int(np.sum(clip[:, 2:] & clip[:, 1:-1] & clip[:, :-2]))       # 3+ сэмпла подряд у потолка
    clip_ratio = runs / y.size
    dc = float(np.abs(y.mean(1)).max())
    # баланс частот по средней мощности спектра
    seg = y[:, : int(min(dur, 120) * sr)]
    if seg.shape[1] > 1024:                      # мощность по каналам (не страдает от противофазы)
        f = (np.abs(np.fft.rfft(seg * np.hanning(seg.shape[1]), axis=1)) ** 2).sum(0)
        freqs = np.fft.rfftfreq(seg.shape[1], 1 / sr)
    else:
        f, freqs = np.ones(10), np.arange(10)
    bands = {"sub": (20, 60), "low": (60, 250), "mid": (250, 2000), "pres": (2000, 6000), "air": (6000, 16000)}
    power = {k: float(f[(freqs >= a) & (freqs < b)].sum()) + 1e-12 for k, (a, b) in bands.items()}
    total = sum(power.values())
    share = {k: v / total for k, v in power.items()}
    # баланс относительно середины, дБ (для розового шума: низ −1,8, присутствие −2,7, воздух −3,3)
    bass_db = 10 * np.log10((power["sub"] + power["low"]) / power["mid"])
    pres_db = 10 * np.log10(power["pres"] / power["mid"])
    air_db = 10 * np.log10(power["air"] / power["mid"])
    corr, width = 1.0, 0.0
    if y.shape[0] == 2:
        l, r = y[0] - y[0].mean(), y[1] - y[1].mean()
        corr = float((l * r).sum() / (np.sqrt((l * l).sum() * (r * r).sum()) + 1e-12))
        width = float(np.sqrt(np.mean(((l - r) / 2) ** 2)) / (np.sqrt(np.mean(((l + r) / 2) ** 2)) + 1e-12))
    loud = np.where(loudest > 10 ** (-50 / 20))[0]
    lead = loud[0] / sr if len(loud) else dur
    tail = (len(mono) - loud[-1]) / sr if len(loud) else dur
    plr = tp - lufs

    m = dict(duration=dur, lufs=lufs, true_peak=tp, clip_ratio=clip_ratio, dc=dc, share=share,
             bass_db=bass_db, presence_db=pres_db, air_db=air_db,
             correlation=corr, width=width, lead_silence=lead, tail_silence=tail, plr=plr, sr=sr)
    issues = []
    if clip_ratio > 1e-4 or tp > 0.0:
        issues.append(("bad", "Перегруз и клиппинг",
                       f"Пик {tp:+.1f} dBTP, срезанные вершины в {clip_ratio * 100:.2f}% звука — слышны хрипы. "
                       "Включите мастеринг (лимитер уберёт новые перегрузы), уменьшите «Усиление баса» и дисторшн."))
    if lufs < -24:
        issues.append(("warn", "Трек очень тихий", f"{lufs:.1f} LUFS. Мастеринг поднимет до нужной громкости."))
    elif lufs > -8:
        issues.append(("warn", "Трек пережат по громкости",
                       f"{lufs:.1f} LUFS — YouTube всё равно сделает его тише. Выберите мастеринг −14 LUFS."))
    if plr < 6 and lufs > -14:
        issues.append(("warn", "Мало динамики", f"Разница пика и громкости {plr:.1f} дБ — звук «плоский». "
                       "Склейку в мастеринге поставьте 0–0,2."))
    if bass_db > 14:
        issues.append(("warn", "Много баса", f"Низ громче середины на {bass_db:.0f} дБ — бас «гудит» и маскирует мелодию. "
                       "Уменьшите «Усиление баса», «Низы» −2…−4 дБ, включите «Плотный бас»."))
    elif bass_db < -2:
        issues.append(("warn", "Мало баса", "Низ слабый. Добавьте «Усиление баса» или новый бас 808."))
    if air_db < -28:
        issues.append(("info", "Глухой звук", "Почти нет верхов. В мастеринге добавьте «Воздух», «Верха» +2 дБ."))
    if pres_db > 2:
        issues.append(("warn", "Резкие верхние середины", "Звук может «резать» уши. «Середина» −2 дБ или эффект «Теплота»."))
    if corr < 0:
        issues.append(("bad", "Проблема с фазой", f"Корреляция каналов {corr:+.2f}: на телефоне в моно звук пропадёт. "
                       "Поставьте «Ширина стерео» 0,8 и включите «Плотный бас»."))
    elif y.shape[0] == 2 and width < .03:
        issues.append(("info", "Почти моно", "Звук узкий. Попробуйте «Ширина стерео» 1,3, хорус или 8D."))
    if dc > .01:
        issues.append(("info", "Смещение нуля (DC)", "Мастеринг уберёт его автоматически."))
    if lead > 2:
        issues.append(("info", "Тишина в начале", f"{lead:.1f} с тишины — обрежьте в «Обрезке» (начать с {lead - .3:.1f} с)."))
    if tail > 3:
        issues.append(("info", "Тишина в конце", f"{tail:.1f} с тишины в конце — обрежьте."))
    if sr < 44100:
        issues.append(("warn", "Низкое качество файла", f"Частота {sr} Гц. Лучше загрузить WAV или MP3 44,1 кГц."))
    if not issues:
        issues.append(("ok", "Трек в хорошем состоянии", "Явных проблем нет — можно сразу делать ремикс."))
    return m, issues


# =====================================================================
# ЗВУКИ АТМОСФЕРЫ (синтез, бесшовные петли)
# =====================================================================

AMBIENCES = {
    "rain": "🌧️ Дождь",
    "storm": "⛈️ Гроза",
    "fire": "🔥 Костёр",
    "sea": "🌊 Море",
    "wind": "🌬️ Ветер",
    "drive": "🚗 Ночная поездка (шум дороги)",
}


def _noise(n, rng, ch=2):
    return rng.standard_normal((ch, n)).astype(np.float32)


def _pinkish(x, sr):
    """Мягкий «розовый» шум: белый через пару фильтров."""
    return lowpass(x, sr, 6000) * .6 + lowpass(x, sr, 1200) * .8


def _events(n, sr, rate, rng, length_s, shape):
    """Случайные короткие события (капли, щелчки) с частотой rate в секунду."""
    out = np.zeros((2, n), np.float32)
    count = rng.poisson(rate * n / sr)
    L = max(8, int(length_s * sr))
    for _ in range(count):
        i = int(rng.integers(0, max(1, n - L)))
        ev = shape(L, rng) * rng.uniform(.3, 1.0)
        pan = rng.uniform(-.8, .8)
        out[0, i:i + L] += ev * (1 - pan) / 2
        out[1, i:i + L] += ev * (1 + pan) / 2
    return out


def _synth_ambience(kind, n, sr, rng):
    t = np.arange(n) / sr
    if kind in ("rain", "storm"):
        bed = _pinkish(highpass(_noise(n, rng), sr, 250), sr)
        bed *= 1 + .15 * np.sin(2 * np.pi * .05 * t + rng.uniform(0, 6))
        drops = _events(n, sr, 60 if kind == "rain" else 110, rng, .015,
                        lambda L, r: (highpass(r.standard_normal(L), sr, 2500) * np.exp(-np.arange(L) / (L / 5))))
        rumble = lowpass(_noise(n, rng), sr, 300) * .6
        out = bed + drops * 1.5 + rumble
        if kind == "storm":
            out *= 1.3
            k = 0
            for start in np.sort(rng.uniform(3, n / sr - 8, max(2, int(n / sr / 30)))):
                L = int(rng.uniform(4, 7) * sr)
                i = int(start * sr)
                m = min(L, n - i)
                tt = np.arange(m) / sr
                env = (1 - np.exp(-tt / rng.uniform(.05, .4))) * np.exp(-tt / rng.uniform(1.2, 2.5))
                boom = lowpass(rng.standard_normal(m), sr, 140) * 6 + lowpass(rng.standard_normal(m), sr, 900) * env * 2
                pan = rng.uniform(-.5, .5)
                out[0, i:i + m] += boom * env * (1 - pan)
                out[1, i:i + m] += boom * env * (1 + pan)
                k += 1
        return out
    if kind == "fire":
        roar = lowpass(_noise(n, rng), sr, 220) * (1 + .4 * np.sin(2 * np.pi * .3 * t))
        hiss = highpass(lowpass(_noise(n, rng), sr, 6000), sr, 2000) * .15
        crackle = _events(n, sr, 18, rng, .006,
                          lambda L, r: r.standard_normal(L) * np.exp(-np.arange(L) / (L / 4)) * 2)
        pops = _events(n, sr, 2, rng, .03, lambda L, r: (lowpass(r.standard_normal(L), sr, 3000)
                                                         * np.exp(-np.arange(L) / (L / 6)) * 6))
        return roar + hiss + crackle + pops
    if kind == "sea":
        base = lowpass(_noise(n, rng), sr, 900)
        per = rng.uniform(6, 10)
        phase = 2 * np.pi * t / per                      # волны с периодом 6–10 с
        wave = (.25 + .75 * (.5 + .5 * np.sin(phase)) ** 3)
        foam = highpass(_noise(n, rng), sr, 1500) * .4 * wave ** 2
        return base * wave + foam
    if kind == "wind":
        x = _noise(n, rng)
        gust = lowpass(rng.standard_normal(n), sr, .3)
        gust = .5 + .5 * gust / (np.abs(gust).max() + 1e-9)
        out = np.zeros_like(x)
        B = int(.05 * sr)
        zi = None
        for a in range(0, n, B):                         # полоса фильтра «гуляет» вместе с порывами
            fc = 250 + 650 * float(gust[a])
            sos = butter(2, [fc / 1.5, fc * 1.5], btype="band", fs=sr, output="sos")
            if zi is None:
                zi = np.zeros((sos.shape[0], 2, 2))
            out[:, a:a + B], zi = sosfilt(sos, x[:, a:a + B], axis=-1, zi=zi)
        return out * (.3 + gust)
    if kind == "drive":
        f0 = 38 + 2 * np.sin(2 * np.pi * .07 * t)
        ph = 2 * np.pi * np.cumsum(f0) / sr
        engine = (np.sin(ph) + .5 * np.sin(2 * ph) + .25 * np.sin(3 * ph)) * .3
        road = lowpass(_noise(n, rng), sr, 350) * .9
        tire = highpass(lowpass(_noise(n, rng), sr, 3000), sr, 900) * .12
        return road + tire + engine
    raise ValueError(kind)


@functools.lru_cache(maxsize=3)
def ambience_loop(kind, sr, seconds=90, seed=5):
    """Бесшовная стерео-петля атмосферы, нормированная до −20 LUFS."""
    rng = np.random.default_rng(seed)
    n, xf = int(seconds * sr), int(3 * sr)
    x = highpass(_synth_ambience(kind, n + xf, sr, rng), sr, 30).astype(np.float32)
    a = np.sqrt(np.linspace(0, 1, xf, dtype=np.float32))
    x[:, :xf] = x[:, :xf] * a + x[:, n:n + xf] * a[::-1]
    x = x[:, :n]
    return (x * 10 ** ((-20 - integrated_lufs(x, sr)) / 20)).astype(np.float32)


def custom_ambience(path, sr):
    """Свой файл атмосферы (кафе, город…): приводим к частоте и −20 LUFS."""
    x, _ = librosa.load(path, sr=sr, mono=False)
    if x.ndim == 1:
        x = np.stack([x, x])
    x = x[:2].astype(np.float32)
    return x * 10 ** ((-20 - integrated_lufs(x, sr)) / 20)


def mix_ambience(y, sr, loop, level_db, offset=0):
    """Добавляет атмосферу (петлю) к сигналу. level_db — громкость относительно трека."""
    n = y.shape[1]
    idx = (np.arange(n) + offset) % loop.shape[1]
    track_lufs = integrated_lufs(y, sr)
    gain = 10 ** ((track_lufs + level_db - (-20)) / 20) if track_lufs > -69 else 10 ** (level_db / 20)
    return y + loop[:, idx] * gain


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
    """Интегральная громкость в LUFS (BS.1770) с гейтами −70 LUFS и −10 LU. Считается кусками по 10 с."""
    if y.ndim == 1:
        y = y[None, :]
    sub = int(.1 * sr)                                   # подблоки 100 мс → блоки 400 мс с шагом 100 мс
    shelf = _rbj("highshelf", 1500.0, sr, 4.0, q=0.7071)
    w = 2 * np.pi * 38.0 / sr
    cw, alpha = np.cos(w), np.sin(w) / (2 * 0.5)
    hb = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]) / (1 + alpha)
    ha = np.array([1, -2 * cw / (1 + alpha), (1 - alpha) / (1 + alpha)])
    ms_sub = []
    seg = sub * 100
    zi_s = np.zeros((shelf.shape[0], y.shape[0], 2))
    zi_h = np.zeros((y.shape[0], 2))
    rest = np.zeros((y.shape[0], 0))
    for a0 in range(0, y.shape[1], seg):
        x = y[:, a0:a0 + seg].astype(np.float64)
        x, zi_s = sosfilt(shelf, x, axis=-1, zi=zi_s)
        x, zi_h = lfilter(hb, ha, x, axis=-1, zi=zi_h)
        x = np.concatenate([rest, x], axis=1)
        k = x.shape[1] // sub
        if k:
            ms_sub.append((x[:, :k * sub] ** 2).reshape(x.shape[0], k, sub).mean(2).sum(0))
        rest = x[:, k * sub:]
    if not ms_sub:
        return -70.0
    ms_sub = np.concatenate(ms_sub)
    if len(ms_sub) < 4:
        return -70.0
    ms = (ms_sub[:-3] + ms_sub[1:-2] + ms_sub[2:-1] + ms_sub[3:]) / 4
    lk = -0.691 + 10 * np.log10(ms + 1e-12)
    gated = ms[lk > -70]
    if not len(gated):
        return -70.0
    rel = -0.691 + 10 * np.log10(gated.mean()) - 10
    gated = ms[(lk > -70) & (lk > rel)]
    return float(-0.691 + 10 * np.log10(gated.mean() + 1e-12)) if len(gated) else -70.0


def _block_peaks(y, block, seg_blocks=4096, pad=64):
    """Истинные пики (4× передискретизация) по блокам из block сэмплов. Кусками — экономно по памяти."""
    n = y.shape[1]
    nb = -(-n // block)
    out = np.zeros(nb, np.float32)
    seg = block * seg_blocks
    for b0 in range(0, nb, seg_blocks):
        a, e = b0 * block, min(n, (b0 + seg_blocks) * block)
        lo, hi = max(0, a - pad), min(n, e + pad)
        peak = None
        for c in range(y.shape[0]):
            up = np.abs(resample_poly(y[c, lo:hi].astype(np.float32), 4, 1))
            up = up[(a - lo) * 4:(a - lo) * 4 + (e - a) * 4]
            peak = up if peak is None else np.maximum(peak, up)
        cnt = -(-(e - a) // block)
        peak = np.pad(peak, (0, cnt * block * 4 - len(peak)))
        out[b0:b0 + cnt] = peak.reshape(cnt, block * 4).max(1)
    return out


def true_peak_db(y):
    """Истинный пик (с учётом межвыборочных пиков) через 4-кратную передискретизацию."""
    if y.ndim == 1:
        y = y[None, :]
    return float(20 * np.log10(float(_block_peaks(y, 4096).max()) + 1e-12))


def limiter(y, sr, ceiling_db=-1.0, release_ms=80.0, block=64):
    """Лимитер с упреждением: заранее плавно убирает громкость перед пиком — без щелчков и клиппинга."""
    ceiling = 10 ** (ceiling_db / 20)
    n = y.shape[1]
    peaks = _block_peaks(y, block)
    nb = len(peaks)
    req = np.minimum(1.0, ceiling / (peaks.astype(np.float64) + 1e-12))
    padded = np.pad(req, 2, constant_values=1.0)          # упреждение: по два блока с каждой стороны
    g = np.min(np.stack([padded[i:i + nb] for i in range(5)]), axis=0)
    g_db = 20 * np.log10(g)
    step = 20 * np.log10(2) / max(1.0, release_ms / 1000 * sr / block)   # +6 дБ за release_ms
    out = np.empty(nb)
    cur = 0.0
    for i in range(nb):
        cur = min(g_db[i], cur + step)
        out[i] = cur
    centers = np.arange(nb) * block + block / 2
    gain = (10 ** (np.interp(np.arange(n), centers, out) / 20)).astype(np.float32)
    return y * gain


def glue_compressor(y, sr, amount, block=64):
    """Мягкая шинная компрессия 2:1 — собирает бит, бас и вокал в один плотный звук.
    Огибающая считается по блокам 64 сэмпла — мало памяти даже на длинных треках."""
    if amount <= 0:
        return y
    n = y.shape[1]
    nb = n // block
    if nb < 16:
        return y
    blk = (y[:, :nb * block].astype(np.float32) ** 2).reshape(y.shape[0], nb, block).mean(2).mean(0)
    bsr = sr / block
    env = _one_pole(blk, bsr, .03)
    lvl = 10 * np.log10(env + 1e-12)
    loud = lvl[lvl > np.percentile(lvl, 30)]
    thr = (np.percentile(loud, 85) if len(loud) else lvl.max()) - 2 - 6 * amount
    ratio = 1.5 + 2.0 * amount
    gr = _one_pole(np.maximum(0, lvl - thr) * (1 - 1 / ratio), bsr, .12)   # плавная огибающая: без «насоса»
    centers = np.arange(nb) * block + block / 2
    gain = (10 ** (-np.interp(np.arange(n), centers, gr) / 20)).astype(np.float32)
    return y * gain


def mono_bass(y, sr, freq=120.0):
    """Бас ниже 120 Гц — в центр: плотнее и громче на телефонах и в клубе."""
    side = highpass((y[0] - y[1]) * .5, sr, freq, order=4)
    mid = (y[0] + y[1]) * .5
    out = np.empty_like(y, dtype=np.float32)
    np.add(mid, side, out=out[0])
    np.subtract(mid, side, out=out[1])
    return out


def master_audio(y, sr, opts=None):
    """Мастеринг-цепочка. Возвращает (сигнал, отчёт о громкости)."""
    o = {**MASTER_DEFAULTS, **(opts or {})}
    tgt = MASTER_TARGETS.get(o["target"], MASTER_TARGETS["youtube"])
    if o.get("lufs") is not None:                         # громкость «как у образца»
        tgt = {"name": f"Как у образца ({o['lufs']:.0f} LUFS)", "lufs": float(o["lufs"]), "ceiling": -1.0}
    report = {"lufs_in": integrated_lufs(y, sr), "tp_in": true_peak_db(y)}
    y = highpass(np.asarray(y, np.float32), sr, 22, order=2)                         # убираем инфразвук и постоянную составляющую
    if y.shape[0] == 2 and o["mono_bass"]:
        y = mono_bass(y, sr)
    if o["warmth"] > 0:                                      # тёплая середина-низ
        y = _sosf(_rbj("peak", 220, sr, 2.5 * o["warmth"], q=.7), y)
    if o["air"] > 0:                                         # «воздух» на верхах
        y = _sosf(_rbj("highshelf", 11000, sr, 3.5 * o["air"]), y)
    y = glue_compressor(y, sr, o["glue"])
    for _ in range(4):                                       # громкость → лимитер → проверка
        cur = integrated_lufs(y, sr)
        if cur <= -69:
            break
        diff = tgt["lufs"] - cur
        if abs(diff) < .15:
            break
        y = limiter(y * np.float32(10 ** (diff / 20)), sr, tgt["ceiling"])
    tp = true_peak_db(y)
    if tp > tgt["ceiling"]:                                  # страховка от остаточных пиков
        y = y * 10 ** ((tgt["ceiling"] - tp - .05) / 20)
    report.update(lufs_out=integrated_lufs(y, sr), tp_out=true_peak_db(y), target=tgt["name"])
    return y, report


def process_audio(y, sr, s, fx=None, bpm=120.0, master=None, fades=(0.02, 0.05), ambience=None, match=None):
    """Скорость/тон → эффекты → пространство → винил → мастеринг. Возвращает (сигнал, отчёт)."""
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    fm = s.get("formant", True)
    if s["link"]:
        y = vinyl_speed(y, sr, s["speed"], fm)
        if s.get("hz432"):
            y = shift_pitch_and_tempo(y, sr, HZ432_SEMITONES, 1.0)
    else:
        y = shift_pitch_and_tempo(y, sr, s["pitch"] + (HZ432_SEMITONES if s.get("hz432") else 0), s["tempo"], fm)
    y = normalize(y, -3.0)
    y = apply_effects(y, sr, fx, bpm)
    y = bass_boost(y, sr, s["bass"])
    if fx and fx.get("tapestop", 0) > 0:
        y = fx_tape_stop(y, sr, fx["tapestop"])   # до зала: хвост ревербера звучит после остановки
    y = apply_space(y, sr, fx, bpm)
    y = apply_reverb(y, sr, s["reverb"])
    y = normalize(y, -3.0)
    y = apply_vinyl(y, sr, s["vinyl"])
    if ambience and ambience.get("loop") is not None:
        y = mix_ambience(y, sr, ambience["loop"], ambience.get("level_db", -14.0))
    if match and match.get("curve"):                       # тембр и ширина «как у образца»
        y, _ = match_eq(y, sr, match["curve"], amount=match.get("amount", .8))
        if match.get("width") and y.shape[0] == 2:
            w0 = stereo_width(y)
            if w0 > 1e-3:
                y = fx_width(y, float(np.clip(match["width"] / w0, .7, 1.6)))
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


# =====================================================================
# ДЛИННЫЕ ВЕРСИИ (1–3 часа) И МИКСЫ: потоковая запись на диск, плавные стыки
# =====================================================================


def trim_silence(y, sr, thresh_db=-50.0, keep=.15):
    loud = np.where(np.abs(y).max(0) > 10 ** (thresh_db / 20))[0]
    if not len(loud):
        return y
    a = max(0, loud[0] - int(keep * sr))
    b = min(y.shape[1], loud[-1] + int(keep * sr))
    return y[:, a:b]


class CrossfadeWriter:
    """Пишет WAV кусками: плавные стыки между фрагментами, атмосфера поверх, лимитер, финальное затухание.
    В памяти держит только текущий фрагмент — подходит для роликов на несколько часов."""

    def __init__(self, path, sr, *, fade_out=8.0, ambience=None, amb_gain=0.0, ceiling_db=-1.0):
        self.f = sf.SoundFile(path, "w", sr, 2, "PCM_16")
        self.sr, self.pos, self.tail = sr, 0, None
        self.fade_out = int(fade_out * sr)
        self.amb, self.amb_gain, self.ceiling = ambience, amb_gain, ceiling_db

    def _emit(self, x):
        if x.shape[1] == 0:
            return
        if self.amb is not None:
            idx = (np.arange(x.shape[1]) + self.pos) % self.amb.shape[1]
            x = x + self.amb[:, idx] * self.amb_gain
            x = limiter(x, self.sr, self.ceiling)
        x = np.clip(_tpdf_dither(x, 16), -1, 1)
        self.f.write(x.T.astype(np.float32))
        self.pos += x.shape[1]

    def append(self, y, xf_s):
        """Добавляет фрагмент; возвращает время (с), с которого он звучит в полную силу (для таймкодов)."""
        y = y.astype(np.float32)
        hold = max(int(xf_s * self.sr), self.fade_out, 1)
        if self.tail is None:
            start = 0.0
            body = y
        else:
            k = min(int(xf_s * self.sr), self.tail.shape[1], y.shape[1] // 2)
            t = np.linspace(0, np.pi / 2, k, dtype=np.float32)
            region = self.tail[:, self.tail.shape[1] - k:] * np.cos(t) + y[:, :k] * np.sin(t)
            self._emit(self.tail[:, :self.tail.shape[1] - k])
            start = (self.pos + k / 2) / self.sr
            body = np.concatenate([region, y[:, k:]], axis=1)
        cut = max(0, body.shape[1] - hold)
        self._emit(body[:, :cut])
        self.tail = body[:, cut:]
        return start

    @property
    def duration(self):
        return (self.pos + (self.tail.shape[1] if self.tail is not None else 0)) / self.sr

    def close(self):
        if self.tail is not None:
            t = self.tail.copy()
            k = min(self.fade_out, t.shape[1])
            if k:
                t[:, -k:] *= np.cos(np.linspace(0, np.pi / 2, k, dtype=np.float32)) ** 2
            self._emit(t)
        self.f.close()
        return self.pos / self.sr


def build_long_version(y, sr, out_wav, minutes, *, crossfade=6.0, ambience=None, amb_level_db=-14.0,
                       progress=None):
    """Зацикливает готовый трек на minutes минут с плавными стыками (и непрерывной атмосферой)."""
    y = trim_silence(y, sr)
    lufs = integrated_lufs(y, sr)
    amb_gain = 10 ** ((lufs + amb_level_db + 20) / 20) if ambience is not None else 0.0
    w = CrossfadeWriter(out_wav, sr, fade_out=10.0, ambience=ambience, amb_gain=amb_gain)
    target = minutes * 60
    reps = 0
    while w.duration < target:
        left = target - w.duration
        piece = y if left > y.shape[1] / sr else y[:, :int((left + crossfade) * sr)]
        w.append(piece, crossfade)
        reps += 1
        if progress:
            progress(min(1.0, w.duration / target))
    return w.close(), reps


def fmt_ts(t):
    t = int(round(t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def build_mix(tracks, sr, out_wav, *, crossfade=5.0, process=None, progress=None):
    """Микс из нескольких треков: каждый обрабатывается и выводится на одну громкость,
    стыки плавные. tracks — список (загрузчик, название). Возвращает (длительность, таймкоды)."""
    w = CrossfadeWriter(out_wav, sr, fade_out=6.0)
    stamps = []
    for i, (loader, title) in enumerate(tracks):
        y = loader()
        if y.shape[0] == 1:
            y = np.repeat(y, 2, axis=0)
        if process:
            y = process(y)
        y = trim_silence(y, sr)
        start = w.append(y, crossfade if i else 0)
        stamps.append((0.0 if i == 0 else start, title))
        del y
        if progress:
            progress((i + 1) / len(tracks))
    return w.close(), stamps


def timestamps_text(stamps):
    return "\n".join(f"{fmt_ts(t)} {title}" for t, title in stamps)


# =====================================================================
# НОВЫЙ БИТ: Demucs (необязательно), анализ трека, генератор паттернов, синтез, сведение
# =====================================================================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

BEAT_STYLES = {
    "template": {"name": "По образцу", "half": False, "swing": 0.0, "side": .35, "filter": .20},
    "disco": {"name": "Disco / Дискотека", "half": False, "swing": 0.0, "side": .40, "filter": .20,
              "retempo": (110, 132, 120)},
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


def _to_sr(x, sr, target):
    """Передискретизация моно-сигнала (точно, полифазным фильтром)."""
    if sr == target:
        return x.astype(np.float32)
    from math import gcd
    g = gcd(int(sr), int(target))
    return resample_poly(x, int(target) // g, int(sr) // g).astype(np.float32)


def _frames_mag(x, n_fft, hop, chunk=4096):
    """Модуль спектра по кадрам (кадры × частоты), считается кусками — экономно по памяти."""
    win = np.hanning(n_fft).astype(np.float32)
    pad = np.concatenate([np.zeros(n_fft // 2, np.float32), x.astype(np.float32), np.zeros(n_fft, np.float32)])
    n_frames = 1 + (len(pad) - n_fft) // hop
    out = np.empty((n_frames, n_fft // 2 + 1), np.float32)
    view = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop]
    for a in range(0, n_frames, chunk):
        out[a:a + chunk] = np.abs(np.fft.rfft(view[a:a + chunk] * win, axis=1))
    return out


def _flux(mag, lo_bin=0, hi_bin=None):
    """Спектральный поток: насколько резко растёт энергия (атаки ударных)."""
    m = np.log1p(100 * mag[:, lo_bin:hi_bin])
    d = np.diff(m, axis=0, prepend=m[:1])
    return np.maximum(d, 0).sum(1)


def _detrend(env, w=16):
    cs = np.concatenate([[0.0], np.cumsum(env)])
    i = np.arange(len(env))
    a, b = np.maximum(0, i - w), np.minimum(len(env), i + w + 1)
    return np.maximum(0, env - (cs[b] - cs[a]) / (b - a))


def _pulse_salience(env, lag, fps, win_s=20.0):
    """Насколько сильно атаки ложатся на сетку с шагом lag (максимум по фазе, среднее по окнам)."""
    n, win = len(env), int(win_s * fps)
    scores = []
    for start in range(0, max(1, n - win // 2), max(1, win // 2)):
        seg = env[start:start + win]
        if len(seg) < lag * 4:
            continue
        k = np.arange(int((len(seg) - 1) // lag))
        best = 0.0
        for ph in range(int(np.ceil(lag))):
            idx = np.round(ph + k * lag).astype(int)
            idx = idx[idx < len(seg)]
            best = max(best, float(seg[idx].mean()))
        scores.append(best / (seg.mean() + 1e-9))
    return float(np.mean(scores)) if scores else 0.0


def _estimate_period(env, fps, bpm_min=55, bpm_max=200, prior_bpm=105):
    """Период доли в кадрах. Кандидаты — пики автокорреляции и их кратные (×2, ×½, ×1,5, ×⅔);
    побеждает тот, на чью сетку лучше ложатся бочка и малый барабан, с мягким предпочтением 70–160 BPM."""
    e = env - env.mean()
    n = len(e)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(e, nfft)
    acf = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    acf /= acf[0] + 1e-9
    lo, hi = int(fps * 60 / bpm_max), min(int(fps * 60 / bpm_min), n // 4)
    peaks = [l for l in range(lo + 1, hi) if acf[l] >= acf[l - 1] and acf[l] >= acf[l + 1] and acf[l] > 0]
    peaks = sorted(peaks, key=lambda l: -acf[l])[:6] or [int(fps * 60 / prior_bpm)]
    cands = set()
    for l in peaks:
        for m in (1, 2, .5, 1.5, 2 / 3):
            c = l * m
            if lo <= c <= hi:
                cands.add(round(c, 1))

    def refine(l):                                           # уточнение лага до долей кадра
        best_l, best_s = l, -1.0
        for d in np.linspace(-1.0, 1.0, 9):
            s_ = _pulse_salience(env, l + d, fps)
            if s_ > best_s:
                best_l, best_s = l + d, s_
        return best_l, best_s

    scored = []
    for c in cands:
        sal = _pulse_salience(env, c, fps)
        bpm = 60 * fps / c
        prior = np.exp(-.5 * (np.log2(bpm / prior_bpm) / 1.0) ** 2)
        scored.append((sal * (.6 + .4 * prior), c))
    _, best = max(scored)
    # Если более быстрая связанная сетка (×2 или ×1,5) почти так же хорошо ложится на удары — берём её:
    # медленная сетка «перепрыгивает» через доли, а для нового бита нужна настоящая доля
    best_sal = _pulse_salience(env, best, fps)
    changed = True
    while changed:
        changed = False
        # ×2: быстрая сетка верна, если её «чётные» и «нечётные» позиции почти одинаково сильные
        c = best / 2
        if c >= lo and 60 * fps / c <= 185 and _alternation(env, c) >= .55:
            best, best_sal, changed = c, _pulse_salience(env, c, fps), True
            continue
        # ×1,5: медленная сетка в 2/3 настоящей доли
        c = best / 1.5
        if c >= lo and 60 * fps / c <= 185:
            sal = _pulse_salience(env, c, fps)
            if sal >= .85 * best_sal:
                best, best_sal, changed = c, sal, True
    return refine(best)[0]


def _alternation(env, lag):
    """Отношение слабых позиций сетки к сильным (1 — одинаково, 0 — через одну пусто)."""
    k = np.arange(int((len(env) - 1) // lag))
    best_ph, best_m = 0, -1.0
    for ph in range(int(np.ceil(lag))):
        idx = np.round(ph + k * lag).astype(int)
        m = float(env[idx[idx < len(env)]].mean())
        if m > best_m:
            best_ph, best_m = ph, m
    idx = np.round(best_ph + k * lag).astype(int)
    idx = idx[idx < len(env)]
    even, odd = float(env[idx[0::2]].mean()), float(env[idx[1::2]].mean())
    return min(even, odd) / (max(even, odd) + 1e-9)


def _dp_beats(env, period, tightness=120.0):
    """Динамическое программирование (Эллис): доли на сильных атаках при почти ровном шаге.
    Терпит живое «плавание» темпа, не теряет сетку на паузах."""
    n = len(env)
    e = env / (env.std() + 1e-9)
    score = e.copy()
    back = -np.ones(n, dtype=np.int64)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    offs = np.arange(lo, hi + 1)
    pen = -tightness * np.log(offs / period) ** 2
    for t in range(lo, n):
        cand = t - offs
        ok = cand >= 0
        if not ok.any():
            continue
        vals = score[cand[ok]] + pen[ok]
        j = int(np.argmax(vals))
        if vals[j] > 0:
            score[t] = e[t] + vals[j]
            back[t] = cand[ok][j]
    tail = slice(max(0, n - int(period * 1.5)), n)
    t = int(np.arange(n)[tail][np.argmax(score[tail])])
    beats = []
    while t >= 0:
        beats.append(t)
        t = int(back[t])
    return np.array(beats[::-1], dtype=float)


def _refine_onsets(beats_t, x, sr):
    """Подтягивает каждую долю к началу реальной атаки (точность ~1–2 мс).
    Короткое окно 11,6 мс с шагом 1 мс, без частот ниже 170 Гц (бас не мешает)."""
    asr, n_fft, hop = 22050, 256, 22
    r = _to_sr(x, sr, asr)
    w = np.hanning(n_fft).astype(np.float32)
    shifts, conf = np.zeros(len(beats_t)), np.zeros(len(beats_t), bool)
    for i, bt in enumerate(beats_t):
        a = int((bt - .02) * asr) - n_fft
        b = int((bt + .045) * asr) + n_fft
        if a < 0 or b > len(r):
            continue
        fr = np.lib.stride_tricks.sliding_window_view(r[a:b], n_fft)[::hop] * w
        m = np.log1p(100 * np.abs(np.fft.rfft(fr, axis=1))[:, 2:])
        fl = np.maximum(np.diff(m, axis=0), 0).sum(1)
        k = int(np.argmax(fl))
        if fl[k] > 3 * (np.median(fl) + 1e-9):
            onset = (a + (k + 1) * hop + n_fft) / asr - .002   # правый край окна − 2 мс = начало атаки
            shifts[i], conf[i] = onset - bt, True
    if not conf.any():
        return beats_t + .0157                                 # калиброванная задержка грубой сетки
    out = shifts.copy()
    for i in range(len(shifts)):                                # выбросы и пропуски — по соседям
        nb = shifts[max(0, i - 3):i + 4][conf[max(0, i - 3):i + 4]]
        med = float(np.median(nb)) if len(nb) else float(np.median(shifts[conf]))
        if not conf[i] or abs(shifts[i] - med) > .012:
            out[i] = med
    return np.maximum.accumulate(beats_t + out)


def analyze(y, sr, drums=None, bass=None, harm=None, bpm_hint=None):
    """Темп, доли (с точностью до миллисекунд), сильная доля, басовые ноты и аккорды по тактам."""
    mono = y.mean(0) if y.ndim == 2 else y
    rhythm = drums.mean(0) if drums is not None else mono
    dur = len(mono) / sr

    # 1. Огибающие атак на 22 кГц, шаг 5 мс
    asr = 22050
    r = _to_sr(rhythm, sr, asr)
    hop, n_fft = 110, 1024
    fps = asr / hop
    mag = _frames_mag(r, n_fft, hop)
    bin_hz = asr / n_fft
    env = _detrend(_flux(mag))
    low = _detrend(_flux(mag, 1, int(150 / bin_hz) + 1))                  # бочка
    mid = _detrend(_flux(mag, int(1200 / bin_hz), int(5000 / bin_hz)))   # малый барабан / хлопок
    del mag

    # 2. Темп и доли
    drum_env = low / (low.std() + 1e-9) + mid / (mid.std() + 1e-9)
    tempo_env = drum_env if drum_env.max() > 0 else env
    if bpm_hint:                                   # темп задан вручную
        period = fps * 60 / float(bpm_hint)
    else:
        period = _estimate_period(tempo_env, fps)
    confidence = _pulse_salience(tempo_env, period, fps)
    frames = _dp_beats(env, period)
    if len(frames) < 8:
        frames = np.arange(0, len(env), period)
    beats = frames / fps
    # 3. Точная подгонка к атакам по полному сигналу
    beats = _refine_onsets(beats, rhythm, sr)

    # 4. Сильная доля: бочка на 1 и 3, малый барабан на 2 и 4
    fi = np.clip(np.round(beats * fps).astype(int), 0, len(env) - 1)
    def at(arr, k):
        return float(arr[fi[k::4]].mean()) if len(fi[k::4]) else 0.0
    lw = np.array([at(low, k) for k in range(4)])
    md = np.array([at(mid, k) for k in range(4)])
    lw /= lw.max() + 1e-9
    md /= md.max() + 1e-9
    drum_score = np.array([lw[p] + .5 * lw[(p + 2) % 4] + .6 * md[(p + 1) % 4] + .6 * md[(p + 3) % 4]
                           for p in range(4)])
    # аккорды меняются в начале такта: «новизна» гармонии на каждой доле
    csr0 = 11025
    hs = _to_sr(harm.mean(0) if harm is not None else mono, sr, csr0)
    hm = _frames_mag(hs, 4096, 512)
    hf = np.arange(hm.shape[1]) * csr0 / 4096
    sel = np.where((hf >= 60) & (hf <= 2000))[0]
    pcs = np.round(69 + 12 * np.log2(hf[sel] / 440)).astype(int) % 12
    chf = np.stack([hm[:, sel[pcs == c]].sum(1) for c in range(12)], 1)
    del hm
    bf = np.clip(np.round(beats * csr0 / 512).astype(int), 0, len(chf) - 1)
    beat_ch = np.stack([chf[bf[i]:bf[i + 1]].mean(0) if bf[i + 1] > bf[i] else chf[bf[i]]
                        for i in range(len(bf) - 1)] + [chf[bf[-1]]])
    beat_ch /= np.linalg.norm(beat_ch, axis=1, keepdims=True) + 1e-9
    nov = np.concatenate([[0.0], 1 - (beat_ch[1:] * beat_ch[:-1]).sum(1)])
    hn = np.array([float(nov[k::4].mean()) if len(nov[k::4]) else 0.0 for k in range(4)])
    hn /= hn.max() + 1e-9
    phase = int(np.argmax(drum_score + 1.2 * hn))

    # 5. Басовые ноты по тактам (38–280 Гц) и аккорды (мажор/минор, 110–2000 Гц)
    bar_starts = beats[phase::4]
    edges = np.concatenate([[-np.inf], bar_starts, [np.inf]])
    nseg = len(bar_starts) + 1
    csr = 11025
    bsrc = _to_sr(bass.mean(0) if bass is not None else lowpass(mono, sr, 300), sr, csr)
    hsrc = _to_sr(harm.mean(0) if harm is not None else mono, sr, csr)

    def chroma_per_bar(sig, n_fft, f_lo, f_hi, weight_low=False):
        mg = _frames_mag(sig, n_fft, n_fft // 2)
        freqs = np.arange(mg.shape[1]) * csr / n_fft
        sel = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
        pcs = (np.round(69 + 12 * np.log2(freqs[sel] / 440)).astype(int) % 12)
        w = np.where(freqs[sel] < 130, 1.5, 1.0) if weight_low else np.ones(len(sel))
        ch = np.zeros((mg.shape[0], 12), np.float32)
        for pc in range(12):
            m = pcs == pc
            if m.any():
                ch[:, pc] = (mg[:, sel[m]] * w[m]).sum(1)
        times = (np.arange(mg.shape[0]) * (n_fft // 2)) / csr
        seg = np.zeros((nseg, 12))
        np.add.at(seg, np.clip(np.searchsorted(edges, times, side="right") - 1, 0, nseg - 1), ch)
        return seg

    seg = chroma_per_bar(bsrc, 8192, 38, 280, weight_low=True)
    strength, roots = seg.sum(1), seg.argmax(1)
    strong = strength >= 0.2 * (np.median(strength) or 0)
    votes = np.zeros(12)
    np.add.at(votes, roots[strong], strength[strong])
    key = int(votes.argmax())
    for i in range(nseg):
        if not strong[i]:
            roots[i] = roots[i - 1] if i else key
    hseg = chroma_per_bar(hsrc, 4096, 110, 2000)
    minor = np.array([hseg[i, (roots[i] + 3) % 12] > hseg[i, (roots[i] + 4) % 12] for i in range(nseg)])
    # Привязка к тональности: аккорды только из лада песни — новые бас и струнные не сфальшивят
    tonic = roots == key
    key_minor = bool(minor[tonic].mean() >= .5) if tonic.any() else bool(minor.mean() >= .5)
    if key_minor:   # натуральный минор; V может быть и мажорной (гармонический минор)
        quality = {0: True, 2: True, 3: False, 5: True, 7: None, 8: False, 10: False}
    else:
        quality = {0: False, 2: True, 4: True, 5: False, 7: False, 9: True, 11: True}
    for i in range(nseg):
        deg = (roots[i] - key) % 12
        if deg not in quality:
            roots[i] = roots[i - 1] if i else key
            deg = (roots[i] - key) % 12
        q = quality.get(deg)
        if q is not None:
            minor[i] = q

    return {
        "tempo": 60.0 / float(np.median(np.diff(beats))) if len(beats) > 1 else 60 * fps / period,
        "beats": beats, "phase": phase, "bar_edges": edges,
        "roots": roots.astype(int), "key": key, "minor": minor, "duration": dur, "key_minor": key_minor,
        "confidence": confidence, "sure": confidence >= 2.2,
    }


def remove_drums_hpss(y, sr, strength=1.0, progress=None):
    """Убирает старые барабаны без нейросети (HPSS): тональные звуки тянутся во времени,
    удары — по частотам. Оставляет мелодию, аккорды и почти весь вокал. Работает кусками по ~20 с."""
    from scipy.ndimage import median_filter
    n_fft, hop = 4096, 1024
    win = np.hanning(n_fft).astype(np.float32)
    chunk, ov = int(20 * sr), int(2 * sr)
    n = y.shape[1]
    out = np.zeros_like(y, dtype=np.float32)
    margin = 1.0 + 1.5 * strength
    starts = list(range(0, n, chunk))
    for ci, a in enumerate(starts):
        lo, hi = max(0, a - ov), min(n, a + chunk + ov)
        for c in range(y.shape[0]):
            seg = y[c, lo:hi].astype(np.float32)
            pad = np.concatenate([np.zeros(n_fft, np.float32), seg, np.zeros(n_fft, np.float32)])
            fr = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop] * win
            X = np.fft.rfft(fr, axis=1)
            mag = np.abs(X)
            H = median_filter(mag, size=(17, 1), mode="nearest")        # протяжённые во времени — тон
            P = median_filter(mag, size=(1, 17), mode="nearest")        # широкие по частоте — удары
            mask = H ** 2 / (H ** 2 + (margin * P) ** 2 + 1e-12)
            frames = np.fft.irfft(X * mask, n_fft, axis=1).astype(np.float32) * win
            rec = np.zeros(len(pad), np.float32)
            norm = np.zeros(len(pad), np.float32)
            for i in range(frames.shape[0]):
                rec[i * hop:i * hop + n_fft] += frames[i]
                norm[i * hop:i * hop + n_fft] += win ** 2
            rec = (rec / np.maximum(norm, 1e-3))[n_fft:n_fft + len(seg)]
            # склейка кусков плавными перекрытиями
            w = np.ones(len(seg), np.float32)
            if lo > 0:
                w[:2 * ov] = np.linspace(0, 1, 2 * ov)
            if hi < n:
                w[-2 * ov:] = np.minimum(w[-2 * ov:], np.linspace(1, 0, 2 * ov))
            out[c, lo:hi] += rec * w
        if progress:
            progress((ci + 1) / len(starts))
    return out


def retempo_plan(bpm, style, target=None, cfg=None):
    """Как привести трек к темпу стиля: (множитель сетки m, коэффициент скорости r, итоговый темп).
    m ∈ {1, 2, ½} выбирается ближе к цели, r — плавная подгонка без изменения тона (0,8…1,35)."""
    cfg = cfg or BEAT_STYLES.get(style, {}).get("retempo")
    if not cfg or not bpm:
        return 1, 1.0, bpm
    lo, hi, ideal = cfg
    target = float(min(max(target or ideal, lo), hi))
    m = min((1, 2, .5), key=lambda k: abs(np.log(target / (bpm * k))))
    r = float(np.clip(target / (bpm * m), .8, 1.35))
    return m, r, bpm * m * r


def scale_analysis(an, rate):
    """Анализ трека после ускорения в rate раз: все времена делятся на rate."""
    out = dict(an)
    out["beats"] = np.asarray(an["beats"]) / rate
    out["bar_edges"] = np.asarray(an["bar_edges"]) / rate
    out["tempo"] = an["tempo"] * rate
    if "duration" in an:
        out["duration"] = an["duration"] / rate
    return out


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


def _gen_disco(r, d):
    """Диско: бочка на каждую долю, открытый хэт между долями, хлопок на 2 и 4, октавный бас восьмыми."""
    e = [_ev(s, "kick", 1) for s in (0, 4, 8, 12)]
    e += [_ev(s, "clap", .8) for s in (4, 12)] + [_ev(s, "snare", .4) for s in (4, 12)]
    e += [_ev(s, "ohat", .55) for s in (2, 6, 10, 14)]
    e += [_ev(s, "hat", .2 if s % 2 == 0 else .14) for s in range(16) if s % 4 != 2 and r.random() < .55 + .4 * d]
    for i, s_ in enumerate(range(0, 16, 2)):
        n = 0 if (i == 0 or r.random() > .2 * d) else r.choice([7, 5])
        e.append({**_ev(s_, "dbass", .95 if i % 2 == 0 else .75, n, 1.7), "oct": i % 2})
    if r.random() < .5 * d:
        e.append(_ev(15, "shaker", .3))
    return e


GENERATORS = {"disco": _gen_disco, "lofi": _gen_lofi, "phonk": _gen_phonk, "trap": _gen_trap,
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

    def disco_bass(self, midi, dur, peak):
        """Упругий диско-бас: синус + яркая пила с быстро закрывающимся фильтром."""
        t = self._t(max(dur, .05))
        f = _hz(midi)
        saw = 2 * ((f * t) % 1) - 1
        bright = sosfilt(_sos("low", min(f * 10, 2500), self.sr), saw)
        x = .75 * np.sin(2 * np.pi * f * t) + .5 * bright * np.exp(-t / .07)
        env = np.minimum(1, t / .004) * np.exp(-t / max(dur * .8, .05))
        k = min(len(t), int(.015 * self.sr))
        if k:
            env[-k:] *= np.linspace(1, 0, k)
        return (np.tanh(x * env * 1.4) * peak * .75).astype(np.float32)

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
# РЕМИКС ПО ОБРАЗЦУ: изучаем готовый ремикс и повторяем его рецепт на любой песне
# =====================================================================

THIRD_OCT = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
             2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]


def third_octave_curve(y, sr, max_s=120):
    """Средний спектр по третьоктавным полосам, дБ относительно среднего (форма звучания, без громкости)."""
    x = (y.mean(0) if y.ndim == 2 else y)[: int(max_s * sr)].astype(np.float32)
    n_fft, hop = 8192, 4096
    if len(x) < n_fft * 2:
        return [0.0] * len(THIRD_OCT)
    mag = _frames_mag(x, n_fft, hop)
    p = (mag.astype(np.float64) ** 2).mean(0)
    f = np.arange(len(p)) * sr / n_fft
    out = []
    for fc in THIRD_OCT:
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        sel = (f >= lo) & (f < hi)
        out.append(10 * np.log10(p[sel].mean() + 1e-12) if sel.any() else -120.0)
    out = np.array(out)
    return list(out - out[(np.array(THIRD_OCT) >= 200) & (np.array(THIRD_OCT) <= 4000)].mean())


def match_eq(y, sr, ref_curve, amount=.8, max_db=8.0):
    """Подгоняет тембр под образец: линейно-фазовый эквалайзер по третьоктавным полосам (±max_db)."""
    from scipy.signal import firwin2
    cur = np.array(third_octave_curve(y, sr))
    ref = np.array(ref_curve)
    diff = np.clip((ref - cur) * amount, -max_db, max_db)
    diff = np.convolve(np.pad(diff, 1, mode="edge"), [.25, .5, .25], mode="valid")   # сглаживание
    freqs = np.concatenate([[0], THIRD_OCT, [sr / 2]]) / (sr / 2)
    gains = 10 ** (np.concatenate([[diff[0]], diff, [diff[-1]]]) / 20)
    taps = firwin2(4097, np.clip(freqs, 0, 1), gains)
    d = len(taps) // 2
    out = np.stack([oaconvolve(ch, taps)[d:d + y.shape[1]] for ch in y]).astype(np.float32)
    return out, list(diff)


def stereo_width(y):
    if y.shape[0] < 2:
        return 0.0
    mid, side = (y[0] + y[1]) / 2, (y[0] - y[1]) / 2
    return float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-12))


def groove_profiles(y, sr, an):
    """Рисунок бита по 16 шагам такта в четырёх полосах: бочка, бас, хлопок/снейр, хэты."""
    asr, hop = 22050, 55
    r = _to_sr(y.mean(0), sr, asr)
    fps = asr / hop
    mag = _frames_mag(r, 1024, hop)
    bh = asr / 1024
    bands = {"kick": (int(30 / bh), int(100 / bh)), "bass": (int(100 / bh), int(250 / bh)),
             "snare": (int(1500 / bh), int(5000 / bh)), "hat": (int(8000 / bh), None)}
    beats = an["beats"]
    bars = beats[an["phase"]::4]
    out, offs = {}, []
    for k, (a, b) in bands.items():
        e = _detrend(_flux(mag, a, b), w=24)
        prof = np.zeros(16)
        for i in range(len(bars) - 1):
            for s_ in range(16):
                f = int((bars[i] + (bars[i + 1] - bars[i]) * s_ / 16) * fps)
                if f + 3 < len(e):
                    prof[s_] += e[max(0, f - 3):f + 4].max()
        out[k] = list(np.round(prof / (prof.max() + 1e-9), 3))
        if k == "hat":
            for i in range(len(bars) - 1):
                st_ = (bars[i + 1] - bars[i]) / 16
                for s_ in range(1, 16, 2):
                    f, w = int((bars[i] + st_ * s_) * fps), max(1, int(st_ * .5 * fps))
                    if w < f < len(e) - w:
                        offs.append((np.argmax(e[f - w:f + w]) - w) / fps / st_)
    out["swing"] = float(np.clip(np.median(offs), -.2, .3)) if offs else 0.0
    return out


def extract_style(y, sr, name="Мой образец"):
    """Изучает готовый ремикс: темп, рисунок бита, тембр, ширину стерео и громкость."""
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    an = analyze(y, sr)
    tpl = {"name": name, "tempo": float(an["tempo"]), "curve": third_octave_curve(y, sr),
           "lufs": float(integrated_lufs(y, sr)), "width": stereo_width(y)}
    tpl.update(groove_profiles(y, sr, an))
    return tpl


def _gen_template(r, d, tpl):
    """Бит по рисунку образца: удары там, где они есть в образце, с той же силой."""
    e = []
    kick, bass, snare, hat = (np.array(tpl[k]) for k in ("kick", "bass", "snare", "hat"))
    for s_ in range(16):
        if kick[s_] >= .6:
            e.append(_ev(s_, "kick", min(1.0, .45 + .6 * kick[s_])))
        elif kick[s_] >= .45 and r.random() < d:
            e.append(_ev(s_, "kick", .5))
        if snare[s_] >= .85 and s_ != 0:
            e += [_ev(s_, "clap", min(1.0, snare[s_])), _ev(s_, "snare", .45 * snare[s_])]
        elif .6 <= snare[s_] < .85 and s_ % 2 == 0 and r.random() < .5 + .4 * d:
            e.append(_ev(s_, "snare", .3))
        if hat[s_] >= .6:
            e.append(_ev(s_, "hat", .18 + .4 * hat[s_]))
        elif hat[s_] >= .4 and r.random() < d:
            e.append(_ev(s_, "hat", .14))
    steps = [s_ for s_ in range(16) if bass[s_] >= .8] or [s_ for s_ in range(16) if kick[s_] >= .6] or [0, 8]
    high_bass = bass.mean() > kick.mean() * 1.15            # в образце бас звучит выше (100–250 Гц) — слышно на телефоне
    for j, s_ in enumerate(steps):
        nxt = steps[j + 1] if j + 1 < len(steps) else 16
        if high_bass:
            e.append({**_ev(s_, "dbass", .9, 0, max(1.0, nxt - s_ - .2)), "oct": 1})
        else:
            e.append(_ev(s_, "sub", .85, 0, max(1.0, nxt - s_ - .2)))
    return e


def find_chorus(y, sr, an):
    """Начало припева: самый повторяющийся и громкий фрагмент из 4 тактов в первых ¾ песни."""
    edges = np.asarray(an["bar_edges"][1:-1])
    if len(edges) < 16:
        return 0.0
    csr = 11025
    xs = _to_sr(y.mean(0), sr, csr)
    mg = _frames_mag(xs, 4096, 1024)
    fr = np.arange(mg.shape[1]) * csr / 4096
    sel = np.where((fr >= 100) & (fr <= 2500))[0]
    pcs = np.round(69 + 12 * np.log2(fr[sel] / 440)).astype(int) % 12
    ch = np.stack([mg[:, sel[pcs == k]].sum(1) for k in range(12)], 1)
    en = mg.sum(1)
    t = np.arange(len(ch)) * 1024 / csr
    bar_ch, bar_en = [], []
    for i in range(len(edges) - 1):
        m = (t >= edges[i]) & (t < edges[i + 1])
        bar_ch.append(ch[m].mean(0) if m.any() else np.zeros(12))
        bar_en.append(en[m].mean() if m.any() else 0)
    bar_ch = np.array(bar_ch)
    bar_ch /= np.linalg.norm(bar_ch, axis=1, keepdims=True) + 1e-9
    bar_en = np.array(bar_en) / (max(bar_en) + 1e-9)
    nb = len(bar_ch) - 3
    win = np.stack([bar_ch[i:i + 4].ravel() for i in range(nb)])
    win /= np.linalg.norm(win, axis=1, keepdims=True) + 1e-9
    S = win @ win.T
    best, best_i = -1, 0
    for i in range(nb):
        if edges[i] < 8 or i > nb * .75:
            continue
        others = np.concatenate([S[i, :max(0, i - 4)], S[i, i + 5:]])
        rep = np.sort(others)[-3:].mean() if len(others) >= 3 else 0
        score = rep * (.5 + .5 * bar_en[i:i + 4].mean())
        if score > best:
            best, best_i = score, i
    return float(edges[best_i])


def save_style(username, tpl):
    with closing(get_conn()) as conn, conn:
        conn.execute("CREATE TABLE IF NOT EXISTS styles (username TEXT, name TEXT, data TEXT, "
                     "created_at TEXT DEFAULT (datetime('now')), PRIMARY KEY (username, name))")
        conn.execute("INSERT OR REPLACE INTO styles (username, name, data) VALUES (?, ?, ?)",
                     (username, tpl["name"][:60], __import__("json").dumps(tpl, ensure_ascii=False)))


def list_styles(username):
    with closing(get_conn()) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS styles (username TEXT, name TEXT, data TEXT, "
                     "created_at TEXT DEFAULT (datetime('now')), PRIMARY KEY (username, name))")
        rows = conn.execute("SELECT name, data FROM styles WHERE username = ? ORDER BY created_at DESC",
                            (username,)).fetchall()
    return {n: __import__("json").loads(d) for n, d in rows}


def delete_style(username, name):
    with closing(get_conn()) as conn, conn:
        conn.execute("DELETE FROM styles WHERE username = ? AND name = ?", (username, name))


# =====================================================================
# СВЕДЕНИЕ
# =====================================================================


# ---------- Музыкальные слои: аккорды, клавиши, эхо, переходы ----------

LAYER_STYLES = {
    "template": {"pad": .22, "keys": .35, "keys_type": "pluck", "bright": .5, "seventh": False, "echo": .25},
    "disco": {"pad": .20, "keys": .55, "keys_type": "disco", "bright": .6, "seventh": True, "echo": .12},
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
    "disco":  ("disco", [], None),
    "piano":  ("chord", [(0, 7), (8, 7)], None),
    "strings_long": ("chord", [(0, 16)], None),
}
# Настроение аккордов поверх песни: меняет инструмент и громкости слоёв
CHORD_MOODS = {
    "auto":    None,
    "warm":    {"name": "Тёплое пианино", "keys_type": "piano", "seventh": True, "bright": .2,
                "pad": .15, "keys": .6, "echo": .12},
    "strings": {"name": "Струнные", "keys_type": "strings_long", "seventh": False, "bright": .3,
                "pad": .0, "keys": .5, "echo": .0},
    "dream":   {"name": "Мечтательный пэд", "keys_type": "bell", "seventh": True, "bright": .55,
                "pad": .45, "keys": .28, "echo": .45},
    "club":    {"name": "Клубное", "keys_type": "pluck", "seventh": False, "bright": .6,
                "pad": .3, "keys": .5, "echo": .3},
}
DISCO_STRINGS = [(0, 3.0), (6, 1.2), (10, 2.5)]          # синкопированные стабы струнных
DISCO_GUITAR = [1, 3, 5, 7, 9, 11, 13, 15]               # «чикен-скретч» на слабых шестнадцатых


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
    elif kind == "strings":  # струнные: три расстроенные пилы, мягкая атака
        x = sum(_saw(f * 2 ** (c / 1200), t, ph) for c, ph in ((-8, 0), (0, .33), (9, .66))) / 3
        x = sosfilt(_sos("low", 4500, sr), x) * np.minimum(1, t / .02) * np.exp(-t / .5)
    elif kind == "gtr":      # глушёная фанк-гитара (Карплус–Стронг)
        L = max(2, int(sr / f))
        exc = np.random.default_rng(int(midi * 7)).uniform(-1, 1, min(n, L))
        x = lfilter([1.0], np.r_[1.0, np.zeros(L - 1), -.48, -.48], np.pad(exc, (0, n - len(exc))))
        x = sosfilt(_sos("band", [400, 4000], sr), x) * np.exp(-t / .045)
    elif kind == "piano":    # фортепиано: обертоны с лёгкой негармоничностью, верхние гаснут быстрее
        x = np.zeros(n)
        for k in range(1, 9):
            fk = f * k * np.sqrt(1 + .0004 * k * k)
            if fk > sr * .45:
                break
            x += np.sin(2 * np.pi * fk * t) * np.exp(-t * (.5 + .45 * k) * (f / 260) ** .3) / k ** 1.25
        x *= np.minimum(1, t / .004)
        x += np.random.default_rng(int(midi)).standard_normal(n) * np.exp(-t / .006) * .05   # удар молоточка
    elif kind == "strings_long":   # струнная группа: медленная атака, держит весь аккорд
        x = sum(_saw(f * 2 ** (c / 1200), t, ph) for c, ph in ((-10, 0), (-3, .2), (4, .5), (11, .8))) / 4
        x = sosfilt(_sos("low", 3200, sr), x) * np.minimum(1, t / .35) * (1 + .04 * np.sin(2 * np.pi * 5 * t))
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


BEAT_LEVEL_DB = {"disco": 1.0, "lofi": -2.0, "phonk": 1.0, "trap": 0.5, "house": 0.0, "synth": -1.0, "dance": 0.0}


def render_beat_mix(y, sr, an, style, *, bed=None, vol=.9, filter_amt=.35, side=.3,
                    density=.5, new_bass=True, seed=1, scale=1, layers=None, hpss_bed=None, template=None):
    """Возвращает (новый микс той же длины, что y; инфо для интерфейса).

    bed — дорожки после Demucs: {"main": вокал + остальное, "bass": бас}. Без них старый бит
    приглушается фильтром.
    """
    st_ = BEAT_STYLES[style]
    if style == "template" and template:
        st_ = dict(st_, name=f"По образцу: {template.get('name', '')}", swing=float(template.get("swing", 0)))
    r = random.Random(seed)
    n, dur = y.shape[1], y.shape[1] / sr
    kit = _Kit(sr, seed)

    grid, down_idx, bpm = _beat_grid(an["beats"], an["phase"], dur, scale)
    mult = 2 if st_["half"] and bpm >= 110 else (.5 if not st_["half"] and bpm < 75 else 1)
    bpb = int(4 * mult)                      # долей в такте паттерна: 2, 4 или 8
    start = down_idx % bpb

    A = _gen_template(r, density, template) if style == "template" and template else GENERATORS[style](r, density)
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
            if name in ("808", "sub", "dbass"):
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
        midi += 12 * ev.get("oct", 0)
        if ev["i"] == "dbass":
            _place(bass_bus, kit.disco_bass(midi, d, ev["v"]), t, 1.0, sr)
            prev = midi
            continue
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
        cfg = {**LAYER_STYLES[style], **(CHORD_MOODS.get(lay.get("mood") or "auto") or {})}
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
            if kind == "disco":                                # струнные и фанк-гитара по аккордам трека
                for kk, b0 in bars:
                    step_sec = (_time_at(grid, b0 + bpb) - _time_at(grid, b0)) / 16
                    for s_, ln in DISCO_STRINGS:
                        t = _time_at(grid, b0 + s_ / 16 * bpb)
                        if 0 <= t < dur - .05 and kk >= 1:
                            root, minor = _chord_at(an, t)
                            for v, midi in enumerate(_chord_tones(60 + root - (12 if root > 7 else 0), minor, True)):
                                _place(keys_bus, _note("strings", midi, ln * step_sec + .25, sr), t, .13, sr, (v - 1.5) * .25)
                    for j, s_ in enumerate(DISCO_GUITAR):
                        if r.random() > .7 + .3 * density:
                            continue
                        t = _time_at(grid, b0 + s_ / 16 * bpb)
                        if 0 <= t < dur - .05:
                            root, minor = _chord_at(an, t)
                            for midi in _chord_tones(64 + (root + 8) % 12 - 8, minor, False):
                                _place(keys_bus, _note("gtr", midi, .12, sr), t, .1, sr, .45 if j % 2 else -.45)
                steps = []
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
                    tail = .5 if kind in ("rhodes", "bell", "piano") else .05
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

    # Подложка: дорожки без барабанов (Demucs или HPSS) или исходный трек под фильтром
    if bed is not None:
        music = bed["main"] + (0 if new_bass else bed["bass"])
    elif hpss_bed is not None:
        music = hpss_bed.copy()
        if filter_amt > 0:
            music = _lowpass(music, sr, cutoff_hz(filter_amt * .5))
        if new_bass:
            music = _highpass(music, sr, 110)
    else:
        music = y.copy()
        if filter_amt > 0:
            music = _lowpass(music, sr, cutoff_hz(filter_amt))
        if new_bass:
            music = _highpass(music, sr, 110)
    if music.shape[0] == 1:
        music = np.repeat(music, 2, axis=0)
    # Баланс по громкости (LUFS), а не по пикам: трек и бит звучат на правильных уровнях
    music = music * 10 ** ((-20.0 - integrated_lufs(music, sr)) / 20)

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
    beat_bus = drums + bass_bus
    if hpss_bed is not None or bed is not None:   # в треке есть голос: освобождаем ему 2–3 кГц
        carve = _rbj("peak", 2600, sr, -3.0, q=.9)
        beat_bus = _sosf(carve, beat_bus)
        layer_bus = _sosf(carve, layer_bus)
    beat_lufs = integrated_lufs(beat_bus, sr)
    if beat_lufs > -69:                           # громкость бита относительно трека
        target = -20.0 + BEAT_LEVEL_DB.get(style, 0.0) + 20 * np.log10(max(vol, 1e-3) / .9)
        beat_bus = beat_bus * 10 ** ((target - beat_lufs) / 20)
    mix = music * duck + beat_bus + layer_bus * soft_duck + fx_bus
    peak = np.max(np.abs(mix)) + 1e-9
    if lay.get("on") and lay.get("polish"):
        # склейка: мягкая сатурация собирает бит, бас и музыку в один звук
        mix = np.tanh(mix / peak * 1.15) / np.tanh(1.15) * .95
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
            "pattern": pattern_text(A), "separated": bed is not None or hpss_bed is not None,
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
# ВИДЕО: анимированный фон, волна, текст песни, водяной знак
# =====================================================================

VIDEO_W, VIDEO_H, VIDEO_FPS = 1920, 1080, 25
LYRICS_FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts"]


def _run_ffmpeg(cmd, cwd=None, timeout=7200, stdin=None):
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout, input=stdin)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode(errors="ignore")[-500:])


def render_watermark(text="", logo_bytes=None, opacity=.8):
    """PNG 1920×1080 с ником или логотипом канала в правом нижнем углу (прозрачный фон)."""
    canvas = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
    if logo_bytes:
        logo = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")
        h = 110
        logo = logo.resize((max(1, int(logo.width * h / logo.height)), h), Image.LANCZOS)
        a = logo.getchannel("A").point(lambda v: int(v * opacity))
        logo.putalpha(a)
        canvas.alpha_composite(logo, (VIDEO_W - logo.width - 48, VIDEO_H - logo.height - 40))
    elif text.strip():
        d = ImageDraw.Draw(canvas)
        font = load_font(40)
        tw = d.textlength(text.strip(), font=font)
        x, y = VIDEO_W - tw - 52, VIDEO_H - 92
        d.text((x + 2, y + 2), text.strip(), font=font, fill=(0, 0, 0, int(160 * opacity)))
        d.text((x, y), text.strip(), font=font, fill=(255, 255, 255, int(235 * opacity)))
    return canvas


def make_background_loop(cover_jpg, out_path, animated=True, watermark=None, seconds=12, crf=20):
    """Зацикленный фон-клип: плавный «наезд» камеры, мерцание света и пыль. Бесшовный стык."""
    base = ImageOps.fit(Image.open(io.BytesIO(cover_jpg)).convert("RGB"), (VIDEO_W, VIDEO_H), Image.LANCZOS)
    wm = None
    if watermark is not None:
        wa = np.asarray(watermark, np.float32) / 255.0
        wm = (wa[..., :3], wa[..., 3:4])
    frames = int(seconds * VIDEO_FPS) if animated else VIDEO_FPS * 2
    rng = np.random.default_rng(11)
    # пылинки: дрейф вверх ровно на высоту кадра за цикл + покачивание → идеальная петля
    P = 110
    px, py = rng.uniform(0, VIDEO_W, P), rng.uniform(0, VIDEO_H, P)
    wob_a, wob_k, wob_ph = rng.uniform(6, 22, P), rng.integers(1, 3, P), rng.uniform(0, 2 * np.pi, P)
    bright, size = rng.uniform(.25, .9, P), rng.integers(1, 4, P)
    blink_k, blink_ph = rng.integers(1, 4, P), rng.uniform(0, 2 * np.pi, P)
    yy, xx = np.mgrid[0:VIDEO_H, 0:VIDEO_W]
    vign = (1 - .35 * (((xx - VIDEO_W / 2) / (VIDEO_W / 2)) ** 2 + ((yy - VIDEO_H / 2) / (VIDEO_H / 2)) ** 2) / 2)
    vign = np.clip(vign, .6, 1)[..., None].astype(np.float32)
    del yy, xx
    big = base.resize((int(VIDEO_W * 1.1), int(VIDEO_H * 1.1)), Image.LANCZOS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(VIDEO_FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-g", str(VIDEO_FPS * 2), out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in range(frames):
            ph = 2 * np.pi * f / frames
            if animated:
                z = 1.0 + .035 * (1 - np.cos(ph)) / 2                       # 1.0 → 1.035 → 1.0
                cw, ch = int(big.width / z / 1.02), int(big.height / z / 1.02)
                cx = (big.width - cw) / 2 + 14 * np.sin(ph)
                cy = (big.height - ch) / 2 + 9 * np.cos(ph)
                img = big.crop((int(cx), int(cy), int(cx) + cw, int(cy) + ch)).resize((VIDEO_W, VIDEO_H), Image.BILINEAR)
                arr = np.asarray(img, np.float32) / 255.0
                arr *= 1 + .03 * np.sin(3 * ph) + .015 * np.sin(7 * ph + 1)   # мерцание света
                arr *= vign
                x = (px + wob_a * np.sin(wob_k * ph + wob_ph)) % VIDEO_W
                y = (py - VIDEO_H * f / frames) % VIDEO_H
                b = bright * (.6 + .4 * np.sin(blink_k * ph + blink_ph))
                for i in range(P):
                    s_ = int(size[i])
                    x0, y0 = int(x[i]), int(y[i])
                    arr[max(0, y0 - s_):y0 + s_, max(0, x0 - s_):x0 + s_] += b[i] * .55
            else:
                arr = np.asarray(base, np.float32) / 255.0
            if wm is not None:
                arr = arr * (1 - wm[1]) + wm[0] * wm[1]
            proc.stdin.write((np.clip(arr, 0, 1) * 255).astype(np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError(err.decode(errors="ignore")[-400:])
    finally:
        if proc.poll() is None:
            proc.kill()
    return out_path


_LRC = re.compile(r"^\s*\[(\d{1,2}):(\d{2}(?:[.,]\d{1,3})?)\]\s*(.*)$")


def parse_lyrics(text, duration):
    """Строки текста с таймингом. Формат LRC «[01:23.45] строка» — точно; без меток — равномерно."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    timed = []
    for l in lines:
        m = _LRC.match(l)
        if m:
            timed.append((int(m.group(1)) * 60 + float(m.group(2).replace(",", ".")), m.group(3).strip()))
    out = []
    if timed and len(timed) >= len(lines) * .6:
        timed = sorted(t for t in timed if t[1])
        for i, (t0, line) in enumerate(timed):
            t1 = timed[i + 1][0] if i + 1 < len(timed) else min(duration, t0 + 6)
            out.append((t0, min(t1, t0 + 8), line))
        return out
    plain = [l for l in lines if not _LRC.match(l)]
    if not plain:
        return []
    start, end = duration * .06, duration * .94
    step = (end - start) / len(plain)
    return [(start + i * step, start + (i + 1) * step - .15, line) for i, line in enumerate(plain)]


def lyrics_to_ass(items, accent):
    def ts(t):
        t = max(0.0, t)
        return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"
    r, g, b = accent
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Lyric,DejaVu Sans,64,&H00FFFFFF,&H00{b:02X}{g:02X}{r:02X},&H64000000,&H96000000,"
            "-1,0,0,0,100,100,0,0,1,4,2,2,120,120,250,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    ev = []
    for t0, t1, line in items:
        safe = line.replace("{", "(").replace("}", ")").replace("\\", "/")
        ev.append(f"Dialogue: 0,{ts(t0)},{ts(t1)},Lyric,,0,0,0,,{{\\fad(250,250)}}{safe}")
    return head + "\n".join(ev) + "\n"


def make_video(cover_jpg, audio_path, out_path, accent, *, duration, animated=True, waveform=True,
               lyrics=None, watermark=None, workdir=None):
    """MP4 1080p для YouTube. Для длинных роликов (> 20 мин) — быстрый путь без перекодирования фона."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("на сервере нет ffmpeg")
    work = os.path.abspath(workdir or tempfile.mkdtemp())
    os.makedirs(work, exist_ok=True)
    audio_path, out_path = os.path.abspath(audio_path), os.path.abspath(out_path)
    bg = os.path.join(work, "bg_loop.mp4")
    long_mode = duration > 20 * 60
    if long_mode or animated:
        make_background_loop(cover_jpg, bg, animated=animated, watermark=watermark if long_mode else None,
                             crf=27 if long_mode else 20)
    if long_mode:
        # фон-петля копируется без перекодирования — час видео собирается за минуты
        _run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", bg, "-i", audio_path,
                     "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                     "-t", f"{duration:.2f}", "-movflags", "+faststart", out_path])
        return out_path
    color = "0x%02X%02X%02X" % tuple(accent)
    if animated:
        inputs = ["-stream_loop", "-1", "-i", bg]
    else:
        cov = os.path.join(work, "cover.jpg")
        with open(cov, "wb") as fh:
            fh.write(cover_jpg)
        inputs = ["-loop", "1", "-framerate", str(VIDEO_FPS), "-i", cov]
    inputs += ["-i", audio_path]
    chain = [f"[0:v]scale={VIDEO_W}:{VIDEO_H}:flags=lanczos,format=rgba[v0]"]
    last = "v0"
    if waveform:
        wave_h = VIDEO_H // 7
        chain.append(f"[1:a]aformat=channel_layouts=mono,showwaves=s={VIDEO_W}x{wave_h}:mode=cline:"
                     f"rate={VIDEO_FPS}:colors={color}@0.9:scale=sqrt:draw=full,format=rgba[wv]")
        chain.append(f"[{last}][wv]overlay=0:H-h-{VIDEO_H // 20}:format=auto[v1]")
        last = "v1"
    if watermark is not None:
        wpath = os.path.join(work, "wm.png")
        watermark.save(wpath)
        inputs += ["-loop", "1", "-i", wpath]
        chain.append(f"[{last}][2:v]overlay=0:0:format=auto[v2]")
        last = "v2"
    if lyrics:
        with open(os.path.join(work, "lyrics.ass"), "w", encoding="utf-8") as fh:
            fh.write(lyrics_to_ass(lyrics, accent))
        fonts = next((d for d in LYRICS_FONT_DIRS if os.path.isdir(d)), "")
        chain.append(f"[{last}]ass=lyrics.ass" + (f":fontsdir={fonts}" if fonts else "") + "[v3]")
        last = "v3"
    chain.append(f"[{last}]format=yuv420p[vout]")
    _run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(chain),
                 "-map", "[vout]", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-r", str(VIDEO_FPS), "-c:a", "aac", "-b:a", "320k", "-t", f"{duration:.2f}",
                 "-movflags", "+faststart", os.path.abspath(out_path)], cwd=work)
    return out_path


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
# SEO ДЛЯ ДЛИННЫХ ВЕРСИЙ И МИКСОВ
# =====================================================================


def _limit_tags(keywords):
    tags, total, seen = [], 0, set()
    for k in filter(None, (k.strip() for k in keywords)):
        if k.lower() in seen:
            continue
        seen.add(k.lower())
        if total + len(k) + 2 > 500:
            break
        tags.append(k)
        total += len(k) + 2
    return ", ".join(tags)


def make_seo_long(title, artist, genre_short, label, minutes, ambience_name=""):
    title = re.sub(r"\s*[\(\[][^\)\]]*remix[^\)\]]*[\)\]]\s*$", "", title.strip(), flags=re.I) or title.strip()
    dur = f"{minutes // 60} Hour" + ("s" if minutes >= 120 else "") if minutes % 60 == 0 else f"{minutes} Minutes"
    amb = re.sub(r"[^\w\s]", "", ambience_name).strip()
    amb_en = {"Дождь": "Rain", "Гроза": "Thunderstorm", "Костёр": "Fireplace", "Море": "Ocean Waves",
              "Ветер": "Wind", "Ночная поездка шум дороги": "Night Drive"}.get(amb, "")
    extra = f" + {amb_en} Sounds" if amb_en else ""
    by = f"{artist.strip()} - " if artist.strip() else ""
    titles = [t[:100] for t in (f"{by}{title} ({label}) {dur}{extra}",
                                f"{title} {label} | {dur} Loop{extra}",
                                f"{dur} of {title} — {genre_short} for Sleep, Study & Relax")]
    desc = (f"🎵 {by}{title} — {label}, {dur.lower()} version{extra.lower()}.\n\n"
            "Perfect for studying, sleeping, working, driving or just relaxing. Seamless loop, no ads in the middle.\n\n"
            "📌 Subscribe for more long versions!\n\n"
            f"{hashtag(title)} {hashtag(genre_short)} #{dur.replace(' ', '')} #Relax #Study #Sleep")
    tags = _limit_tags([f"{title} {dur.lower()}", f"{title} {label.lower()}", f"{genre_short} {dur.lower()}",
                        f"{label.lower()} {dur.lower()}", f"{genre_short} for sleep", f"{genre_short} for study",
                        f"{amb_en.lower()} sounds" if amb_en else "", title, artist])
    return titles, desc, tags


def make_seo_mix(mix_title, genre_short, label, minutes, stamps):
    year = __import__("datetime").date.today().year
    dur = f"{max(1, round(minutes / 60))} Hour" if minutes >= 50 else f"{int(minutes)} Min"
    style = genre_short if label.lower() in genre_short.lower() else f"{genre_short} {label}"
    titles = [t[:100] for t in (f"{mix_title} | {genre_short} Mix {year} ({dur})",
                                f"Best {style} Mix {year} — {dur}",
                                f"{mix_title} 🎧 {style} Playlist")]
    desc = (f"🎧 {mix_title} — {genre_short} {label} mix, {dur.lower()}.\n\n"
            f"⏱ Tracklist:\n{timestamps_text(stamps)}\n\n"
            "📌 Like, subscribe and turn on notifications for new mixes!\n\n"
            f"{hashtag(genre_short)}Mix {hashtag(label)} #Playlist #Mix{year} #Music")
    tags = _limit_tags([f"{genre_short} mix", f"{genre_short} mix {year}", f"{label.lower()} mix",
                        f"{genre_short} playlist", f"{genre_short} {dur.lower()}", mix_title]
                       + [t for _, t in stamps[:8]])
    return titles, desc, tags


# =====================================================================
# ФОНОВЫЕ ЗАДАЧИ И ЛИЧНЫЙ КАБИНЕТ
# =====================================================================

JOBS_DIR = DATA_DIR / "jobs"
CACHE_DIR = DATA_DIR / "cache"
JOB_KINDS = {"studio": "🎛 Трек", "long": "♾ Длинная версия", "mix": "🎚 Микс"}
JOB_STATUS = {"queued": "⏳ в очереди", "running": "⚙️ в работе", "done": "✅ готово", "error": "❌ ошибка"}


class JobCtx:
    def __init__(self, job_id, username):
        self.id, self.username = job_id, username
        self.dir = JOBS_DIR / job_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._last = 0.0

    def path(self, name):
        return str(self.dir / name)

    def progress(self, p, msg=None, force=False):
        now = __import__("time").time()
        if not force and now - self._last < 1.0:          # не чаще раза в секунду
            return
        self._last = now
        with closing(get_conn()) as conn, conn:
            if msg is None:
                conn.execute("UPDATE jobs SET progress = ? WHERE id = ?", (float(p), self.id))
            else:
                conn.execute("UPDATE jobs SET progress = ?, message = ? WHERE id = ?", (float(p), msg, self.id))

    def stage(self, a, b, msg):
        """Прогресс для отрезка работы [a, b]: возвращает функцию f(доля внутри этапа)."""
        self.progress(a, msg, force=True)
        return lambda frac: self.progress(a + (b - a) * min(1.0, max(0.0, frac)))


def _run_job(job_id, username, fn, params):
    import traceback
    import gc
    ctx = JobCtx(job_id, username)
    with closing(get_conn()) as conn, conn:
        conn.execute("UPDATE jobs SET status = 'running', message = 'Начинаем…' WHERE id = ?", (job_id,))
    try:
        result = fn(ctx, params) or {}
        with closing(get_conn()) as conn, conn:
            conn.execute("UPDATE jobs SET status = 'done', progress = 1, message = 'Готово', "
                         "finished_at = datetime('now'), result = ? WHERE id = ?",
                         (__import__("json").dumps(result, ensure_ascii=False), job_id))
    except Exception as err:
        with open(ctx.path("error.log"), "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        with closing(get_conn()) as conn, conn:
            conn.execute("UPDATE jobs SET status = 'error', message = ?, finished_at = datetime('now') WHERE id = ?",
                         (f"{type(err).__name__}: {err}"[:600], job_id))
    finally:
        gc.collect()


def cleanup_jobs(keep_days=None):
    """Удаляет старые результаты (по умолчанию старше 7 дней) и кэш анализа старше 2 дней."""
    keep_days = keep_days or float(get_setting("JOB_KEEP_DAYS", "7"))
    import time as _t
    with closing(get_conn()) as conn, conn:
        old = [r[0] for r in conn.execute(
            "SELECT id FROM jobs WHERE status IN ('done','error') AND created_at < datetime('now', ?)",
            (f"-{keep_days} days",))]
        for jid in old:
            shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)
            conn.execute("DELETE FROM jobs WHERE id = ?", (jid,))
    if CACHE_DIR.exists():
        for f in CACHE_DIR.iterdir():
            if _t.time() - f.stat().st_mtime > 2 * 86400:
                f.unlink(missing_ok=True)


@st.cache_resource(show_spinner=False)
def job_executor():
    """Один исполнитель на весь сервер. Задачи, оборванные перезапуском, помечаются ошибкой."""
    from concurrent.futures import ThreadPoolExecutor
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    with closing(get_conn()) as conn, conn:
        conn.execute("UPDATE jobs SET status = 'error', message = 'Сервер перезапускался во время обработки — "
                     "запустите задачу заново' WHERE status IN ('queued', 'running')")
    cleanup_jobs()
    return ThreadPoolExecutor(max_workers=max(1, int(get_setting("JOB_WORKERS", "1"))))


def submit_job(username, kind, title, fn, params, prepare=None):
    """Создаёт задачу. prepare(ctx) сохраняет входные файлы в папку задачи до постановки в очередь."""
    active = [j for j in list_jobs(username) if j["status"] in ("queued", "running")]
    if len(active) >= int(get_setting("MAX_ACTIVE_JOBS", "2")) and not is_admin(username):
        raise RuntimeError("У вас уже идут 2 задачи. Дождитесь их завершения в «Моём кабинете».")
    job_id = secrets.token_hex(6)
    ctx = JobCtx(job_id, username)
    if prepare:
        prepare(ctx)
    with closing(get_conn()) as conn, conn:
        conn.execute("INSERT INTO jobs (id, username, kind, title, status, message) VALUES (?, ?, ?, ?, 'queued', ?)",
                     (job_id, username, kind, title[:120], "В очереди"))
    job_executor().submit(_run_job, job_id, username, fn, params)
    return job_id


def list_jobs(username, limit=30):
    with closing(get_conn()) as conn:
        rows = conn.execute("SELECT id, kind, title, status, progress, message, created_at, finished_at, result "
                            "FROM jobs WHERE username = ? ORDER BY created_at DESC LIMIT ?",
                            (username, limit)).fetchall()
    keys = ("id", "kind", "title", "status", "progress", "message", "created_at", "finished_at", "result")
    out = [dict(zip(keys, r)) for r in rows]
    for j in out:
        j["result"] = __import__("json").loads(j["result"]) if j["result"] else {}
    return out


def delete_job(username, job_id):
    with closing(get_conn()) as conn, conn:
        conn.execute("DELETE FROM jobs WHERE id = ? AND username = ? AND status IN ('done','error')", (job_id, username))
    shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)


def _file_entry(ctx, name, label, kind, mime):
    p = Path(ctx.path(name))
    return {"name": name, "label": label, "kind": kind, "mime": mime, "size": p.stat().st_size} if p.exists() else None


def _write_audio_files(ctx, y, sr, bits, base="track", mp3_kbps=320):
    files = []
    with open(ctx.path(f"{base}.wav"), "wb") as fh:
        fh.write(encode_audio(y, sr, "WAV", bits=bits))
    files.append(_file_entry(ctx, f"{base}.wav", f"🎧 WAV {bits} бит", "audio", "audio/wav"))
    try:
        with open(ctx.path(f"{base}.mp3"), "wb") as fh:
            fh.write(encode_audio(y, sr, "MP3"))
        files.append(_file_entry(ctx, f"{base}.mp3", f"🎵 MP3 {mp3_kbps} кбит/с", "audio", "audio/mpeg"))
    except Exception:
        pass
    return [f for f in files if f]


def _wav_file_to_mp3(src, dst, kbps=192):
    _run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:a", "libmp3lame", "-b:a", f"{kbps}k", dst])


def _cover_for(ctx, p):
    genre = GENRES[p["genre_label"]]
    background = None
    if p.get("bg_path"):
        try:
            background = Image.open(p["bg_path"]).convert("RGB")
        except OSError:
            background = None
    background = background or fetch_background(genre["photos"]) or gradient_background(genre["accent"])
    cover = make_cover(p["title"], p["artist"], p["genre_text"], genre["accent"], background)
    with open(ctx.path("cover.jpg"), "wb") as fh:
        fh.write(cover)
    return cover, genre


def _video_for(ctx, p, cover, genre, audio_path, duration, lyrics=None, stage=None):
    v = p.get("video") or {}
    if not v.get("on") or not shutil.which("ffmpeg"):
        return None
    wm = None
    if v.get("logo_path") or v.get("wm_text"):
        logo = open(v["logo_path"], "rb").read() if v.get("logo_path") else None
        wm = render_watermark(v.get("wm_text", ""), logo)
    if stage:
        stage(0.0)
    make_video(cover, audio_path, ctx.path("video.mp4"), genre["accent"], duration=duration,
               animated=v.get("animated", True), waveform=v.get("waveform", True),
               lyrics=lyrics, watermark=wm, workdir=ctx.path("video_tmp"))
    shutil.rmtree(ctx.path("video_tmp"), ignore_errors=True)
    return _file_entry(ctx, "video.mp4", "🎬 Видео MP4 1080p", "video", "video/mp4")


def _cached(key, fn):
    """Кэш тяжёлых расчётов (анализ, HPSS) на диске — повторные запуски того же трека быстрее."""
    import pickle
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest() + ".pkl")
    if path.exists():
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            path.unlink(missing_ok=True)
    val = fn()
    try:
        with open(path, "wb") as fh:
            pickle.dump(val, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass
    return val


def _ambience_from(p, sr):
    a = p.get("ambience") or {}
    if not a.get("kind"):
        return None, ""
    if a["kind"] == "custom":
        return custom_ambience(a["path"], sr), "Своя атмосфера"
    return ambience_loop(a["kind"], sr), AMBIENCES.get(a["kind"], "")


def run_studio_job(ctx, p):
    """Полный цикл студии в фоне: анализ → бит → эффекты → мастеринг → ИИ → обложка → видео → SEO."""
    ctx.progress(0.0, "Загружаем трек…", force=True)
    y, sr = load_audio(p["src_path"])
    i0 = int(p["trim_start"] * sr)
    i1 = int(p["trim_end"] * sr) if p["trim_end"] else y.shape[1]
    y = y[:, i0:i1]
    if y.shape[1] < sr:
        raise RuntimeError("После обрезки трек короче 1 секунды — проверьте «Начать с» и «Закончить на».")
    file_key = f"{p['src_hash']}|{p['trim_start']}|{p['trim_end']}"
    y, y_original, analysis, beat_info, fx_bpm, master_report, amb_name = studio_audio(ctx, y, sr, p, file_key)
    ctx.progress(.62, "Сохраняем WAV и MP3…", force=True)
    files = _write_audio_files(ctx, y, sr, p["bits"])
    duration = y.shape[1] / sr

    ai_note = None
    if p.get("ai"):
        ai = p["ai"]
        ctx.progress(.66, "Нейросеть Stable Audio перегенерирует трек…", force=True)
        if not charge_ai(ctx.username, ai["mode"], "профи ИИ-ремикс"):
            ai_note = "На балансе нет ИИ-генераций — ИИ-версия не создана."
        else:
            src = (y if ai["source"] == "remix" else y_original)[:, :int(AI_MAX_SECONDS * sr)]
            try:
                payload, fname, mime = encode_audio(src, sr, "MP3"), "track.mp3", "audio/mpeg"
            except Exception:
                payload, fname, mime = encode_audio(src, sr, "WAV"), "track.wav", "audio/wav"
            key_name = None
            if analysis is not None:
                same = analysis["roots"] == analysis["key"]
                mn = analysis.get("minor")
                is_minor = bool(mn[same].mean() > .5) if mn is not None and same.any() else False
                key_name = NOTE_NAMES[analysis["key"]] + (" minor" if is_minor else " major")
            prompt = build_ai_prompt(p["style_key"], ai["extra"], fx_bpm, key_name)
            try:
                data = stability_audio_to_audio(payload, fname, mime, prompt, ai["strength"],
                                                min(AI_MAX_SECONDS, src.shape[1] / sr), ai["key"], ai["model"],
                                                int(p["beat_seed"]))
                with open(ctx.path("ai.mp3"), "wb") as fh:
                    fh.write(data)
                files.append(_file_entry(ctx, "ai.mp3", "🤖 ИИ-версия MP3", "audio", "audio/mpeg"))
            except Exception as err:
                refund_ai(ctx.username, ai["mode"], "профи ИИ-ремикс")
                ai_note = f"ИИ-версия не получилась: {err}" + (" Генерация возвращена." if ai["mode"] == "credits" else "")

    ctx.progress(.72, "Рисуем обложку…", force=True)
    cover, genre = _cover_for(ctx, p)
    files.insert(0, _file_entry(ctx, "cover.jpg", "🖼️ Обложка JPG", "image", "image/jpeg"))
    lyrics = parse_lyrics(p["video"].get("lyrics", ""), duration) if p.get("video", {}).get("lyrics") else None
    vstage = ctx.stage(.75, .97, "Собираем видео для YouTube (для 3-минутного трека — около 2–5 минут)…")
    try:
        vf = _video_for(ctx, p, cover, genre, ctx.path("track.wav"), duration, lyrics=lyrics, stage=vstage)
        if vf:
            files.append(vf)
    except Exception as err:
        ai_note = (ai_note + " " if ai_note else "") + f"Видео не получилось: {err}"
    titles, description, tags = make_seo(p["title"], p["artist"], genre["short"], p["label"])
    return {"files": [f for f in files if f], "titles": titles, "description": description, "tags": tags,
            "master": master_report, "beat": beat_info, "fx": active_fx(p["fx"]), "duration": duration,
            "note": ai_note, "ambience": amb_name}


RU_NOTES = ["до", "до-диез", "ре", "ре-диез", "ми", "фа", "фа-диез", "соль", "соль-диез", "ля", "ля-диез", "си"]
KEY_SHIFT_TIPS = {-3: "тёмно и глубоко", -2: "глубже и грустнее, как в slowed", -1: "мягче и теплее",
                  1: "ярче и энергичнее", 2: "ещё ярче, для sped up и поп", 3: "очень ярко, nightcore"}


def key_label(root, minor, long=False):
    short = NOTE_NAMES[int(root) % 12] + ("m" if minor else "")
    return f"{short} ({RU_NOTES[int(root) % 12]} {'минор' if minor else 'мажор'})" if long else short


def song_key_info(y, sr):
    """Тональность и темп для подсказок (первых полутора минут хватает, памяти нужно мало)."""
    an = analyze(y[:, :int(90 * sr)], sr)
    return {"root": int(an["key"]), "minor": bool(an.get("key_minor", False)), "bpm": float(an["tempo"])}


def total_shift(s):
    """Итоговый сдвиг тона в полутонах при текущих настройках."""
    base = 12 * np.log2(s["speed"]) if s["link"] else s["pitch"]
    return base + (HZ432_SEMITONES if s.get("hz432") else 0.0)


TONE_MOODS = {
    "bright": "☀️ Ярко",
    "sad": "🌧 Грустно",
    "dance": "💃 Танцевально",
    "calm": "🌙 Спокойно",
}


def beautiful_tone(mood, minor, bpm):
    """«Красивая тональность»: сдвиг тона, темп и зал под настроение с учётом лада и темпа песни."""
    def tempo_to(target, lo, hi):
        if not bpm:
            return 1.0
        cand = min((bpm, bpm * 2, bpm / 2), key=lambda b: abs(np.log(target / b)))
        return float(np.clip(round(target / cand, 2), lo, hi))
    if mood == "bright":   # мажор и так светлый — хватает +1; минору нужно +2
        return {"pitch": 2.0 if minor else 1.0, "tempo": tempo_to(bpm * 1.05 if bpm else 120, 1.0, 1.08), "reverb": .12}
    if mood == "sad":
        return {"pitch": -2.0 if minor else -3.0, "tempo": tempo_to(bpm * .9 if bpm else 80, .85, .95), "reverb": .45}
    if mood == "dance":
        return {"pitch": 1.0, "tempo": tempo_to(124, .9, 1.15), "reverb": .15}
    return {"pitch": -1.0, "tempo": tempo_to(85, .85, 1.0), "reverb": .35}


def _genre_by_short(short):
    return next(g for g, v in GENRES.items() if v["short"] == short)


RECIPES = {
    "⚡ Sped Up": dict(genre="Sped Up", preset="Sped Up / Nightcore", link=True, speed=1.2, reverb=.1, beat=False, air=.3,
                      tip="+3 полутона, ×1,2, без нового бита"),
    "🌙 Slowed + Reverb": dict(genre="Slowed + Reverb", preset="Slowed & Reverb", link=True, speed=.85, reverb=.55,
                               vinyl=.15, beat=False, warm=.2, tip="−3 полутона, ×0,85, большой зал"),
    "🎌 Nightcore": dict(genre="Sped Up", preset="Sped Up / Nightcore", link=True, speed=1.28, reverb=.12, beat=True,
                        style="dance", chords="club", air=.4, hi=2.0, tip="×1,28, танцевальный бит и плак-синт"),
    "🖤 Deep / Dark": dict(genre="Slowed + Reverb", preset="Deep / Dark", link=False, tempo=.94, pitch=-3.0,
                          reverb=.35, bass=.5, beat=True, style="trap", chords="dream", warm=.3, tip="−3 полутона, тёмный трэп"),
    "✨ Bright Pop": dict(genre="Pop Remix", preset="Bright Pop", link=False, tempo=1.04, pitch=2.0, reverb=.15,
                         bass=.25, beat=True, style="dance", chords="club", air=.7, hi=3.5, tip="+2 полутона, яркий поп-бит"),
    "☕ Lo-Fi": dict(genre="Lo-Fi", preset="Lo-Fi Vinyl", link=True, speed=.93, reverb=.25, vinyl=.35, bass=.35,
                    beat=True, style="lofi", chords="warm", warm=.4, tip="винил, тёплое пианино, мягкий бит"),
    "🪩 Disco": dict(genre="Disco", preset="Disco", link=False, beat=True, style="disco", retempo=True, air=.45, hi=2.5,
                    tip="темп танцпола, струнные и фанк-гитара"),
}


def recipe_values(name):
    """Значения виджетов для рецепта (ключи как у pv/wk)."""
    r = RECIPES[name]
    gl = _genre_by_short(r["genre"])
    base = PRESETS[r["preset"]]
    k = f"{gl}_{r['preset']}"
    vals = {"s_genre": gl, f"preset_{gl}": r["preset"], "beat_on": r["beat"], "formant": True, "hz432": False,
            "m_air": r.get("air", 0.0), "m_warm": r.get("warm", 0.0), "fx_preset": "Без эффектов",
            "fx_Без эффектов_eq_high": r.get("hi", 0.0)}
    for f in ("link", "speed", "tempo", "pitch", "reverb", "bass", "vinyl"):
        vals[f"{f}_{k}"] = r.get(f, base[f])
    if r["beat"]:
        style = r["style"]
        vals[f"beatstyle_{gl}"] = "auto" if style == GENRES[gl]["beat"] else style
        vals["layers_on"] = True
        vals["chord_mood"] = r.get("chords", "auto")
        if r.get("retempo"):
            vals[f"retempo_{style}"] = True
    return vals


def _loudest_window(y, sr, seconds):
    """Начало самого яркого (громкого) куска длиной seconds — обычно это припев."""
    m = np.abs(y).mean(0)
    sec = m[:len(m) // sr * sr].reshape(-1, sr).mean(1) if len(m) >= sr else m[:1]
    w = int(seconds)
    if len(sec) <= w:
        return 0
    run = np.convolve(sec, np.ones(w), "valid")
    lo = int(len(run) * .15)                                   # не берём самое начало (интро)
    return int(lo + np.argmax(run[lo:])) if len(run) > lo + 1 else int(np.argmax(run))


def studio_preview(source_bytes, source_name, p, seconds=20, lead=8):
    """Быстрая прослушка: тот же звук, что в полной обработке, но на самом ярком куске песни.
    Возвращает (было MP3, стало MP3, начало куска в секундах). «Было» выровнено по громкости со «стало»."""
    with tempfile.NamedTemporaryFile(suffix=Path(source_name).suffix or ".mp3") as tf:
        tf.write(source_bytes)
        tf.flush()
        y, sr = load_audio(tf.name)
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    i0 = int(p["trim_start"] * sr)
    i1 = int(p["trim_end"] * sr) if p["trim_end"] else y.shape[1]
    y = y[:, i0:i1]
    win = seconds + lead
    start = _loudest_window(y, sr, win)
    exc = y[:, start * sr:(start + win) * sr]
    del y
    q = {**p, "fade_in": .02, "fade_out": .3, "ai": None,
         "tpl_opts": {**(p.get("tpl_opts") or {}), "chorus": False}}
    if (q.get("ambience") or {}).get("kind") == "custom":
        q["ambience"] = {"kind": None, "level_db": -18}
    key = f"{p['src_hash']}|pv|{p['trim_start']}|{p['trim_end']}|{start}"
    out, _, _, _, _, _, _ = studio_audio(_NullCtx(), exc, sr, q, key)
    n = int(seconds * sr)
    after = out[:, -n:] if out.shape[1] > n else out
    ratio = exc.shape[1] / max(1, out.shape[1])                  # во сколько раз изменилась длина
    nb = min(exc.shape[1], int(after.shape[1] * ratio))
    before = exc[:, -nb:]
    gain = 10 ** ((integrated_lufs(after, sr) - integrated_lufs(before, sr)) / 20)
    before = np.clip(before * gain, -.98, .98)
    fade = np.minimum(1, np.arange(after.shape[1]) / (.03 * sr))
    after = after * fade * fade[::-1]
    fb = np.minimum(1, np.arange(before.shape[1]) / (.03 * sr))
    before = before * fb * fb[::-1]
    return encode_audio(before, sr, "MP3"), encode_audio(after, sr, "MP3"), i0 / sr + start + lead


class _NullCtx:
    """Заглушка прогресса для быстрой прослушки (без фоновой задачи)."""
    def progress(self, *a, **k):
        pass

    def stage(self, *a, **k):
        return lambda *x, **y: None


def studio_audio(ctx, y, sr, p, file_key):
    """Звуковая часть студии: анализ → новый бит и слои → тон и темп → эффекты → мастеринг.
    Общая для полной обработки и для прослушки 20 секунд — звучит одинаково."""
    y_original = y
    beat_info, analysis = None, None
    p = {**p, "settings": dict(p["settings"])}

    if p["beat_on"]:
        st_a = ctx.stage(.05, .12, "Ищем темп, доли, басовые ноты и аккорды…")
        analysis = _cached(f"an|{file_key}|{p.get('bpm_hint') or 0}",
                           lambda: analyze(y, sr, bpm_hint=p.get("bpm_hint") or None))
        st_a(1)
        tpl = p.get("template") if p["style_key"] == "template" else None
        topt = p.get("tpl_opts") or {}
        if tpl and topt.get("tempo", True):                  # темп задаёт образец — скорость пресета не добавляем
            p["settings"] = {**p["settings"], "speed": 1.0, "tempo": 1.0}
        if tpl and topt.get("chorus"):                       # как у диджеев: ремикс начинается с припева
            ctx.progress(.1, "Ищем припев…", force=True)
            start = find_chorus(y, sr, analysis)
            if start > 1.0 and y.shape[1] / sr - start > 45:
                y = y[:, int((start - .05) * sr):]
                y_original = y
                file_key += f"|ch{start:.2f}"
                analysis = _cached(f"an|{file_key}", lambda: analyze(y, sr))
        hpss_bed = None
        if p["drum_removal"] == "hpss":
            st_h = ctx.stage(.12, .35, "Убираем старые барабаны (без нейросети)…")
            hpss_bed = _cached(f"hpss|{file_key}",
                               lambda: remove_drums_hpss(y, sr, progress=st_h).astype(np.float16)).astype(np.float32)
        beat_scale, retempo_note = p["beat_scale"], None
        tcfg = None
        if tpl and topt.get("tempo", True):
            t_ = float(tpl["tempo"])
            tcfg = (t_ * .98, t_ * 1.02, t_)
        if tcfg or (p.get("retempo") and BEAT_STYLES[p["style_key"]].get("retempo")):
            src_bpm = analysis["tempo"]
            m, rate, target = retempo_plan(src_bpm, p["style_key"], None if tcfg else p.get("retempo_bpm"), cfg=tcfg)
            if abs(rate - 1) > .01:
                ctx.progress(.36, f"Меняем темп {src_bpm:.0f} → {target:.0f} BPM без изменения тона…", force=True)
                y = stretch_tempo(y, sr, rate)
                if hpss_bed is not None:
                    hpss_bed = stretch_tempo(hpss_bed, sr, rate)
                analysis = scale_analysis(analysis, rate)
            beat_scale = m
            retempo_note = f"{src_bpm:.0f} → {target:.0f} BPM"
        ctx.progress(.4, f"Пишем и сводим бит: {BEAT_STYLES[p['style_key']]['name']}…", force=True)
        y, beat_info = render_beat_mix(
            y, sr, analysis, p["style_key"], bed=None, vol=p["beat_vol"], filter_amt=p["beat_filter"],
            side=p["beat_side"], density=p["beat_density"], new_bass=p["beat_bass"], seed=int(p["beat_seed"]),
            scale=beat_scale, layers=p["layers"], hpss_bed=hpss_bed, template=tpl)
        beat_info["retempo"] = retempo_note
        if tpl:
            beat_info["template"] = tpl.get("name")
        s = p["settings"]
        beat_info["bpm_out"] = beat_info["bpm"] * (s["speed"] if s["link"] else s["tempo"])
        beat_info["sure"] = bool(analysis.get("sure", True))
        del hpss_bed

    s = p["settings"]
    fx_bpm = beat_info["bpm_out"] if beat_info else None
    if fx_bpm is None and any(p["fx"].get(k, 0) > 0 for k in SYNCED_FX):
        an = analysis or _cached(f"an|{file_key}|0", lambda: analyze(y_original, sr))
        fx_bpm = an["tempo"] * (s["speed"] if s["link"] else s["tempo"])
    amb, amb_name = _ambience_from(p, sr)
    ctx.progress(.5, "Эффекты, атмосфера и мастеринг…", force=True)
    tpl = p.get("template") if p["beat_on"] and p["style_key"] == "template" else None
    topt = p.get("tpl_opts") or {}
    match, master = None, p["master"]
    if tpl and topt.get("sound", True):
        match = {"curve": tpl["curve"], "width": tpl["width"], "amount": .8}
    if tpl and topt.get("loud", True) and master:
        master = {**master, "lufs": float(np.clip(tpl["lufs"], -14, -8))}
    y, master_report = process_audio(
        y, sr, s, p["fx"], fx_bpm or 120.0, master=master, match=match,
        fades=(max(p["fade_in"], .02), max(p["fade_out"], .05)),
        ambience={"loop": amb, "level_db": p["ambience"]["level_db"]} if amb is not None else None)
    return y, y_original, analysis, beat_info, fx_bpm, master_report, amb_name

def run_long_job(ctx, p):
    """Версия на 30 мин – 3 часа: плавные стыки, непрерывная атмосфера, быстрое видео, SEO «1 Hour»."""
    ctx.progress(.02, "Загружаем трек…", force=True)
    y, sr = load_audio(p["src_path"])
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    if p.get("master_on", True):
        ctx.progress(.06, "Мастеринг под YouTube…", force=True)
        y, _ = master_audio(y, sr, {"target": p.get("target", "youtube")})
    amb, amb_name = _ambience_from(p, sr)
    st_b = ctx.stage(.1, .55, f"Собираем версию на {p['minutes']} мин…")
    wav = ctx.path("long.wav")
    dur, reps = build_long_version(y, sr, wav, p["minutes"], crossfade=p["crossfade"], ambience=amb,
                                   amb_level_db=(p.get("ambience") or {}).get("level_db", -16), progress=st_b)
    del y
    ctx.progress(.56, "Кодируем MP3…", force=True)
    _wav_file_to_mp3(wav, ctx.path("long.mp3"), 192)
    files = [_file_entry(ctx, "long.mp3", "🎵 MP3 192 кбит/с", "audio", "audio/mpeg")]
    cover, genre = _cover_for(ctx, p)
    files.insert(0, _file_entry(ctx, "cover.jpg", "🖼️ Обложка JPG", "image", "image/jpeg"))
    vstage = ctx.stage(.62, .98, "Собираем видео (длинный ролик собирается быстро — фон не перекодируется)…")
    note = None
    try:
        vf = _video_for(ctx, p, cover, genre, wav, dur, stage=vstage)
        if vf:
            files.append(vf)
    except Exception as err:
        note = f"Видео не получилось: {err}"
    Path(wav).unlink(missing_ok=True)                       # WAV на часы слишком большой для хранения
    titles, description, tags = make_seo_long(p["title"], p["artist"], genre["short"], p["label"],
                                              p["minutes"], amb_name)
    return {"files": [f for f in files if f], "titles": titles, "description": description, "tags": tags,
            "duration": dur, "note": note, "repeats": reps}


def run_mix_job(ctx, p):
    """Микс из 2–20 треков: одинаковая обработка и громкость, плавные переходы, таймкоды."""
    sr = 44100
    n = len(p["tracks"])
    st_m = ctx.stage(.02, .6, f"Обрабатываем и сводим {n} треков…")
    s, fx, master = p["settings"], p["fx"], p["master"]

    def loader_for(path):
        def load():
            x, _ = librosa.load(path, sr=sr, mono=False)
            return (np.stack([x, x]) if x.ndim == 1 else x[:2]).astype(np.float32)
        return load

    def proc(x):
        bpm = 120.0
        if any(fx.get(k, 0) > 0 for k in SYNCED_FX):
            bpm = analyze(x, sr)["tempo"] * (s["speed"] if s["link"] else s["tempo"])
        out, _ = process_audio(x, sr, s, fx, bpm, master=master, fades=(.02, .05))
        return out

    wav = ctx.path("mix.wav")
    dur, stamps = build_mix([(loader_for(t["path"]), t["title"]) for t in p["tracks"]], sr, wav,
                            crossfade=p["crossfade"], process=proc, progress=st_m)
    amb, amb_name = _ambience_from(p, sr)
    if amb is not None:                                      # атмосфера поверх всего микса, кусками
        ctx.progress(.62, "Добавляем атмосферу…", force=True)
        data, _ = sf.read(wav, dtype="float32", always_2d=True)
        y = data.T
        del data
        y = mix_ambience(y, sr, amb, (p.get("ambience") or {}).get("level_db", -16))
        y = limiter(y, sr, -1.0)
        sf.write(wav, np.clip(_tpdf_dither(y, 16), -1, 1).T, sr, subtype="PCM_16")
        del y
    ctx.progress(.66, "Кодируем MP3…", force=True)
    _wav_file_to_mp3(wav, ctx.path("mix.mp3"), 320 if dur < 40 * 60 else 192)
    files = [_file_entry(ctx, "mix.mp3", "🎵 MP3 микса", "audio", "audio/mpeg")]
    cover, genre = _cover_for(ctx, p)
    files.insert(0, _file_entry(ctx, "cover.jpg", "🖼️ Обложка JPG", "image", "image/jpeg"))
    note = None
    vstage = ctx.stage(.7, .98, "Собираем видео…")
    try:
        vf = _video_for(ctx, p, cover, genre, wav, dur, stage=vstage)
        if vf:
            files.append(vf)
    except Exception as err:
        note = f"Видео не получилось: {err}"
    Path(wav).unlink(missing_ok=True)
    titles, description, tags = make_seo_mix(p["title"], genre["short"], p["label"], dur / 60, stamps)
    return {"files": [f for f in files if f], "titles": titles, "description": description, "tags": tags,
            "duration": dur, "timestamps": timestamps_text(stamps), "note": note}


# =====================================================================
# МОИ ПРЕСЕТЫ («фирменный звук»)
# =====================================================================


def save_preset(username, name, data):
    with closing(get_conn()) as conn, conn:
        conn.execute("INSERT OR REPLACE INTO presets (username, name, data, created_at) VALUES (?, ?, ?, datetime('now'))",
                     (username, name.strip()[:60], __import__("json").dumps(data, ensure_ascii=False)))


def list_presets(username):
    with closing(get_conn()) as conn:
        rows = conn.execute("SELECT name, data FROM presets WHERE username = ? ORDER BY created_at DESC",
                            (username,)).fetchall()
    return {n: __import__("json").loads(d) for n, d in rows}


def delete_preset(username, name):
    with closing(get_conn()) as conn, conn:
        conn.execute("DELETE FROM presets WHERE username = ? AND name = ?", (username, name))


# =====================================================================
# ГЕНЕРАТОР НАЗВАНИЙ КАНАЛОВ И ПЛЕЙЛИСТОВ
# =====================================================================

NAME_BANK = {
    "disco": {"a": ["Mirror", "Disco", "Golden", "Velvet", "Funky", "Neon", "Groove", "Glitter", "Midnight", "Roller"],
              "b": ["Ball", "Fever", "Nights", "Club", "Machine", "Party", "Groove", "Palace", "Lights", "Dancers"],
              "p": ["Nu-Disco Hits {year} 🪩", "Disco Remixes of Popular Songs", "Friday Night Disco Party",
                    "Disco House Mix — {n} Hour", "Funky Disco Grooves for {mood} Nights"]},
    "lofi": {"a": ["Rainy", "Sleepy", "Dusty", "Cozy", "Midnight", "Velvet", "Paper", "Quiet", "Warm", "Foggy"],
             "b": ["Tapes", "Beats", "Café", "Window", "Library", "Notebook", "Cassette", "Room", "Vinyl", "Lamp"],
             "p": ["Rainy {day} Lo-Fi to Study & Relax", "Lo-Fi Beats for {mood} Nights", "{n} Hours of Cozy Lo-Fi",
                   "Coffee Shop Lo-Fi ☕ {mood} Vibes", "Lo-Fi for Sleep 🌙 No Ads"]},
    "slowed": {"a": ["Faded", "Lonely", "Echo", "Blue", "Slow", "Hollow", "Velvet", "Lost", "Silent", "Moon"],
               "b": ["Waves", "Memories", "Reverb", "Hours", "Dreams", "Tapes", "Nights", "Hearts", "Rooms", "Skies"],
               "p": ["Sad Songs Slowed + Reverb 💔", "Slowed Songs to Cry to at 3AM", "{mood} Slowed Playlist",
                     "Songs That Hit Different Slowed", "Slowed & Reverb — {n} Hour Mix"]},
    "phonk": {"a": ["Drift", "Dark", "Night", "Venom", "Ghost", "Midnight", "Toxic", "Shadow", "Neon", "Street"],
              "b": ["Phonk", "Cartel", "Garage", "Tokyo", "Drift", "Rush", "Mafia", "Lane", "Clan", "Engine"],
              "p": ["Aggressive Drift Phonk 🔥 Gym Mix", "Phonk for Night Drive 🚗", "Brazilian Phonk Hits {year}",
                    "Phonk That Makes You Feel Like a Villain", "{n} Hour Phonk Mix — No Ads"]},
    "trap": {"a": ["Cloud", "Ice", "Moon", "Glow", "Wave", "Purple", "Sky", "Gold", "Frost", "Lunar"],
             "b": ["Trap", "Nation", "Beats", "Vibes", "City", "Records", "Zone", "Kingdom", "Club", "Garden"],
             "p": ["Melodic Trap Mix {year}", "Chill Trap for Late Nights", "Trap Beats to Vibe To",
                   "Atmospheric Trap 🌌 {mood}", "{n} Hour Trap Playlist"]},
    "house": {"a": ["Deep", "Sunset", "Ocean", "Golden", "Velvet", "Island", "Summer", "Club", "Urban", "Night"],
              "b": ["House", "Grooves", "Sessions", "Lounge", "Waves", "Terrace", "Rooftop", "Motion", "Pulse", "Club"],
              "p": ["Deep House Mix {year} 🌅", "Sunset Deep House Session", "Rooftop Lounge Mix",
                    "Summer Vibes Deep House", "{n} Hour Deep House Mix"]},
    "synth": {"a": ["Neon", "Retro", "Chrome", "Laser", "Night", "Cyber", "Outrun", "Electric", "Vapor", "Arcade"],
              "b": ["Drive", "Dreams", "Highway", "City", "Horizon", "Sunset", "Tapes", "Grid", "Wave", "Memory"],
              "p": ["Synthwave Night Drive 🌃", "Retrowave Mix — Back to the 80s", "Outrun Highway Mix",
                    "Cyberpunk Synthwave for Focus", "{n} Hour Synthwave Mix"]},
    "dance": {"a": ["Hyper", "Sped", "Rapid", "Bright", "Turbo", "Nova", "Pulse", "Sugar", "Star", "Flash"],
              "b": ["Up", "Songs", "Beats", "Edits", "Mode", "Rush", "Party", "Vibes", "Hits", "Club"],
              "p": ["Sped Up Songs TikTok {year}", "Nightcore Hits to Dance", "Sped Up Playlist ⚡",
                    "Songs Sped Up That Slap", "{n} Hour Sped Up Mix"]},
}
NAME_MOODS = ["Chill", "Dreamy", "Rainy", "Late", "Lonely", "Golden", "Cozy", "Dark", "Calm", "Night"]


def generate_names(style, seed, count=8):
    import datetime
    rng = random.Random(seed)
    bank = NAME_BANK[style]
    channels = set()
    patterns = ["{a} {b}", "{a}{b}", "{b} of {a}", "the {a} {b}", "{a} {b} Club", "{a}.{b}"]
    while len(channels) < count:
        a, b = rng.choice(bank["a"]), rng.choice(bank["b"])
        if a == b:
            continue
        channels.add(rng.choice(patterns).format(a=a, b=b).replace("the ", "The "))
    playlists = [t.format(day=rng.choice(["Day", "Night", "Morning"]), mood=rng.choice(NAME_MOODS),
                          n=rng.choice([1, 2, 3]), year=datetime.date.today().year)
                 for t in rng.sample(bank["p"], len(bank["p"]))]
    return sorted(channels), playlists


# =====================================================================
# АНАЛИЗ НИШИ YOUTUBE (YouTube Data API v3)
# =====================================================================

NICHE_QUERIES = {
    "disco": ["disco remix", "nu disco", "disco house mix", "disco remix {year}", "funky disco"],
    "lofi": ["lofi hip hop", "lofi rain", "lofi study music", "lofi sleep 1 hour", "lofi beats {year}"],
    "slowed": ["slowed reverb", "slowed and reverb 1 hour", "sad slowed songs", "slowed tiktok songs", "slowed {year}"],
    "phonk": ["phonk", "drift phonk", "brazilian phonk", "phonk 1 hour", "gym phonk {year}"],
    "trap": ["melodic trap", "chill trap mix", "trap beats {year}", "atmospheric trap"],
    "house": ["deep house mix", "deep house {year}", "melodic house", "sunset deep house"],
    "synth": ["synthwave", "retrowave mix", "night drive synthwave", "cyberpunk music"],
    "dance": ["sped up songs", "nightcore", "sped up tiktok {year}", "sped up playlist"],
}


def yt_niche(api_key, query, days=30):
    """Спрос по запросу: просмотры в день у самых популярных роликов последних days дней."""
    import datetime
    import json as _json
    import time as _t
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT ts, data FROM yt_cache WHERE q = ?", (f"{query}|{days}",)).fetchone()
    if row and _t.time() - row[0] < 12 * 3600:               # кэш 12 ч — бережём квоту API
        return _json.loads(row[1])
    after = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get("https://www.googleapis.com/youtube/v3/search", timeout=20, params={
        "part": "snippet", "q": query, "type": "video", "order": "viewCount", "publishedAfter": after,
        "maxResults": 25, "key": api_key})
    if r.status_code != 200:
        msg = r.json().get("error", {}).get("message", r.text[:200]) if r.headers.get("content-type", "").startswith(
            "application/json") else r.text[:200]
        raise RuntimeError(f"YouTube API {r.status_code}: {msg}")
    js = r.json()
    ids = [it["id"]["videoId"] for it in js.get("items", []) if it.get("id", {}).get("videoId")]
    total = int(js.get("pageInfo", {}).get("totalResults", 0))
    vids = []
    if ids:
        v = requests.get("https://www.googleapis.com/youtube/v3/videos", timeout=20, params={
            "part": "statistics,snippet", "id": ",".join(ids), "key": api_key})
        v.raise_for_status()
        now = datetime.datetime.now(datetime.timezone.utc)
        for it in v.json().get("items", []):
            pub = datetime.datetime.fromisoformat(it["snippet"]["publishedAt"].replace("Z", "+00:00"))
            age = max(1.0, (now - pub).total_seconds() / 86400)
            views = int(it.get("statistics", {}).get("viewCount", 0))
            vids.append({"title": it["snippet"]["title"], "channel": it["snippet"]["channelTitle"],
                         "views": views, "per_day": views / age, "url": f"https://youtu.be/{it['id']}"})
    vids.sort(key=lambda x: -x["per_day"])
    vpd = [x["per_day"] for x in vids]
    med = float(np.median(vpd)) if vpd else 0.0
    res = {"query": query, "total": total, "median_per_day": med, "top": vids[:5],
           "score": med / np.log10(total + 10) if vpd else 0.0}
    with closing(get_conn()) as conn, conn:
        conn.execute("INSERT OR REPLACE INTO yt_cache (q, ts, data) VALUES (?, ?, ?)",
                     (f"{query}|{days}", _t.time(), _json.dumps(res, ensure_ascii=False)))
    return res


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


def render_hero(text="Ремиксы с новым битом, атмосфера, студийный мастеринг, версии на 1–3 часа, миксы, видео с текстом песни, обложка и SEO для YouTube. Всё считается в фоне — можно закрыть вкладку."):
    st.markdown(f"""
<div class="nbs-hero">
  <div class="nbs-eq"><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
  <div class="nbs-logo">NBS<span>SOFT MUSIC</span></div>
  <p>{text}</p>
  <div class="nbs-tags"><b>Новый бит в такт</b><b>8D и 20+ эффектов</b><b>Мастеринг −14 LUFS</b><b>Версии на 1–3 часа</b><b>Миксы с таймкодами</b><b>Видео для YouTube</b></div>
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
                    token = create_login_token(user.strip())
                    st.session_state.update(logged_in=True, username=user.strip(), auth_token=token)
                    st.query_params["s"] = token
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
    "disco": "nu-disco remix, four on the floor kick, octave disco bassline, funky muted guitar, lush string stabs, "
             "open hi-hats, groovy dance floor energy",
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
GEN_BPM = {"disco": 120, "lofi": 80, "phonk": 130, "trap": 140, "house": 122, "synth": 105, "dance": 128}


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


# =====================================================================
# ИНТЕРФЕЙС
# =====================================================================

PAGES = ["🎛 Студия", "♾ Длинная версия", "🎚 Микс", "📁 Мой кабинет", "📈 Ниша YouTube", "🏷 Названия"]
ADMIN_PAGE = "🛠 Админ-панель"
MIX_SOUND = {
    "Без изменений": dict(link=False, speed=1.0, tempo=1.0, pitch=0.0, reverb=0.0, bass=0.0, vinyl=0.0),
    "Slowed + Reverb": dict(link=True, speed=.85, tempo=1.0, pitch=0.0, reverb=.5, bass=.1, vinyl=.15),
    "Sped Up": dict(link=True, speed=1.2, tempo=1.0, pitch=0.0, reverb=.1, bass=0.0, vinyl=0.0),
    "Lo-Fi": dict(link=True, speed=.93, tempo=1.0, pitch=0.0, reverb=.25, bass=.2, vinyl=.6),
}


# --- ключи виджетов с версией: так пресет применяется без предупреждений Streamlit ---

def wk(name):
    st.session_state.setdefault("_reg", set()).add(name)
    return f"{name}#{st.session_state.get('pv_nonce', 0)}"


def pv(name, default):
    return st.session_state.get("pv_vals", {}).get(name, default)


def pidx(options, name, default):
    val = pv(name, default)
    return options.index(val) if val in options else (options.index(default) if default in options else 0)


def collect_preset():
    out = {}
    for name in st.session_state.get("_reg", set()):
        key = f"{name}#{st.session_state.get('pv_nonce', 0)}"
        if key in st.session_state:
            v = st.session_state[key]
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[name] = v
    return out


def apply_preset(data):
    st.session_state["pv_vals"] = dict(data)
    st.session_state["pv_nonce"] = st.session_state.get("pv_nonce", 0) + 1


def go(page):
    st.session_state["nav"] = page


def save_upload(ctx, upl, name):
    if upl is None:
        return None
    ext = Path(upl.name).suffix.lower() or ".bin"
    path = ctx.path(name + ext)
    with open(path, "wb") as fh:
        fh.write(upl.getvalue())
    return path


def fmt_size(n):
    return f"{n / 1e6:.1f} МБ" if n < 1e9 else f"{n / 1e9:.2f} ГБ"


# --- общие блоки ---

def cover_block(prefix, default_title, show_artist=True):
    genre_label = st.selectbox("Ниша / стиль канала", list(GENRES), key=wk(f"{prefix}genre"),
                               index=pidx(list(GENRES), f"{prefix}genre", list(GENRES)[0]))
    genre = GENRES[genre_label]
    c1, c2 = st.columns(2)
    title = c1.text_input("Название", pv(f"{prefix}title", default_title), key=wk(f"{prefix}title"))
    artist = c2.text_input("Артист", pv(f"{prefix}artist", "Smagulov"), key=wk(f"{prefix}artist")) if show_artist else ""
    genre_text = st.text_input("Текст жанра на обложке", pv(f"{prefix}gtext_{genre_label}", genre["short"]),
                               key=wk(f"{prefix}gtext_{genre_label}"))
    bg = st.file_uploader("Свой фон для обложки (необязательно)", type=["jpg", "jpeg", "png", "webp"], key=f"{prefix}bg")
    return genre_label, genre, title, artist, genre_text, bg


def ambience_block(prefix, default_level=-16):
    opts = ["none"] + list(AMBIENCES) + ["custom"]
    names = {"none": "Без атмосферы", "custom": "📁 Свой файл (кафе, город, птицы…)", **AMBIENCES}
    kind = st.selectbox("Звук атмосферы поверх трека", opts, format_func=names.get, key=wk(f"{prefix}amb"),
                        index=pidx(opts, f"{prefix}amb", "none"))
    level, upl = default_level, None
    if kind != "none":
        level = st.slider("Громкость атмосферы относительно музыки, дБ", -30, -4, int(pv(f"{prefix}amb_lvl", default_level)),
                          1, key=wk(f"{prefix}amb_lvl"), help="−16 дБ — фоном, −8 дБ — атмосфера на первом плане.")
        if kind == "custom":
            upl = st.file_uploader("Файл атмосферы (MP3/WAV, лучше 1–3 минуты)", type=["mp3", "wav"], key=f"{prefix}amb_file")
    return kind, level, upl


def video_block(prefix, allow_wave=True, allow_lyrics=True, default_on=True):
    has_ffmpeg = shutil.which("ffmpeg") is not None
    on = st.checkbox("🎬 Сделать видео для YouTube (MP4 1080p)", value=has_ffmpeg and pv(f"{prefix}vid", default_on),
                     disabled=not has_ffmpeg, key=wk(f"{prefix}vid"))
    if not has_ffmpeg:
        st.caption("Видео собирается на сервере с ffmpeg (Render, свой сервер).")
    v = {"on": on}
    if on:
        c1, c2 = st.columns(2)
        v["animated"] = c1.checkbox("Анимированная обложка (наезд камеры, мерцание, пыль)",
                                    value=pv(f"{prefix}vid_anim", True), key=wk(f"{prefix}vid_anim"))
        v["waveform"] = c2.checkbox("Живая волна звука", value=pv(f"{prefix}vid_wave", True),
                                    key=wk(f"{prefix}vid_wave"), disabled=not allow_wave) and allow_wave
        v["wm_text"] = st.text_input("Водяной знак: ник канала (например @nbs_soft_music)",
                                     pv(f"{prefix}vid_wm", ""), key=wk(f"{prefix}vid_wm"))
        v["logo"] = st.file_uploader("…или логотип канала (PNG с прозрачным фоном)", type=["png", "jpg", "jpeg"],
                                     key=f"{prefix}vid_logo")
        if allow_lyrics:
            v["lyrics"] = st.text_area(
                "Текст песни на видео (необязательно)", key=f"{prefix}vid_lyr", height=120,
                placeholder="[00:12.5] Первая строка\n[00:18.0] Вторая строка\n\nили просто строки — распределятся по треку",
                help="Точно в такт — с метками времени в формате LRC: [мм:сс.сс] строка. Без меток строки идут равномерно.")
        if not allow_wave:
            st.caption("Для роликов длиннее 20 минут волна и текст отключены: фон зацикливается без перекодирования, "
                       "поэтому часовое видео собирается за минуты.")
    return v


@st.fragment(run_every=4)
def render_active_jobs(username):
    """Живой прогресс задач: обновляется сам каждые 4 с, не трогая остальную страницу."""
    jobs = list_jobs(username, limit=10)
    active = [j for j in jobs if j["status"] in ("queued", "running")]
    seen = st.session_state.setdefault("_seen_active", set())
    finished_now = False
    for j in jobs:
        if j["id"] in seen and j["status"] in ("done", "error"):
            seen.discard(j["id"])
            st.toast(("✅ Готово: " if j["status"] == "done" else "❌ Ошибка: ") + j["title"])
            st.session_state["_new_done"] = True
            finished_now = True
    for j in active:
        seen.add(j["id"])
        st.progress(min(1.0, float(j["progress"] or 0)),
                    text=f"{JOB_KINDS.get(j['kind'], '')} «{j['title']}» — {JOB_STATUS[j['status']]}: {j['message'] or ''}")
    on_cabinet = st.session_state.get("nav") == "📁 Мой кабинет"
    if finished_now and on_cabinet:
        st.session_state["_new_done"] = False
        st.rerun(scope="app")                      # обновить список результатов в кабинете
    if st.session_state.get("_new_done") and not on_cabinet:
        c1, c2 = st.columns([3, 2])
        c1.success("Задача готова — результат в «Моём кабинете».")
        if c2.button("📁 Открыть кабинет", key="open_cab_live"):
            st.session_state["_new_done"] = False
            go("📁 Мой кабинет")
            st.rerun(scope="app")


def submit_ui(username, kind, title, fn, params, prepare):
    try:
        submit_job(username, kind, title, fn, params, prepare=prepare)
        st.session_state["_seen_active"] = st.session_state.get("_seen_active", set())
        st.success("Задача запущена. Можно закрыть вкладку — результат появится в «Моём кабинете».")
        st.button("📁 Перейти в кабинет", on_click=go, args=("📁 Мой кабинет",), key=f"goto_cab_{kind}")
    except RuntimeError as err:
        st.error(str(err))


# --- страница: Студия ---

def render_studio(username, api_key, ai_mode):
    # Шаг 1. Музыка
    with st.container(border=True):
        step_title(1, "🎵 Музыка")
        mode = st.radio("Источник", ["upload", "generate"], horizontal=True, label_visibility="collapsed",
                        format_func={"upload": "🎵 Загрузить свой трек", "generate": "✨ Создать трек с нуля (ИИ)"}.get)
        source_bytes, source_name, source_bpm = None, None, None
        if mode == "upload":
            up = st.file_uploader("Загрузите трек (.mp3 или .wav)", type=["mp3", "wav"], key="studio_src")
            if up is not None:
                source_bytes, source_name = up.getvalue(), up.name
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
                            generated = True
                        except Exception as err:
                            refund_ai(username, ai_mode, "трек с нуля")
                            st.error(f"{err}. Генерация возвращена на баланс." if ai_mode == "credits" else str(err))
                    if generated:
                        st.rerun()
            gen = st.session_state.get("gen_track")
            if gen:
                source_bytes, source_name, source_bpm = gen["bytes"], gen["name"], gen["bpm"]
                st.success("Трек готов. Это ваша оригинальная музыка — дальше её можно обработать как обычно.")
                st.download_button("⬇️ Скачать сгенерированный трек (MP3)", gen["bytes"], gen["name"],
                                   "audio/mpeg", key="dl_gen")
        if source_bytes is not None:
            st.audio(source_bytes)
            src_hash = hashlib.md5(source_bytes).hexdigest()[:16]
            with st.expander("🩺 Проверка трека: перегруз, баланс, фаза, тишина"):
                diag = st.session_state.get("diag")
                if diag and diag[0] == src_hash:
                    render_diagnosis(*diag[1])
                elif st.button("Проверить трек", key="diag_btn"):
                    with st.spinner("Анализируем…"):
                        with tempfile.NamedTemporaryFile(suffix=Path(source_name).suffix or ".mp3") as tf:
                            tf.write(source_bytes)
                            tf.flush()
                            y0, sr0 = load_audio(tf.name)
                        st.session_state["diag"] = (src_hash, diagnose(y0, sr0))
                        del y0
                    st.rerun()
    if source_bytes is None:
        return

    # Мои пресеты
    with st.container(border=True):
        st.markdown("**⭐ Мои пресеты — «фирменный звук» в один клик**")
        presets = list_presets(username)
        c1, c2, c3 = st.columns([3, 1, 1])
        chosen = c1.selectbox("Сохранённые", ["—"] + list(presets), key="preset_pick", label_visibility="collapsed")
        if c2.button("Применить", disabled=chosen == "—", key="preset_apply"):
            apply_preset(presets[chosen])
            st.rerun()
        if c3.button("Удалить", disabled=chosen == "—", key="preset_del"):
            delete_preset(username, chosen)
            st.rerun()
        c1, c2 = st.columns([3, 2])
        new_name = c1.text_input("Название нового пресета", placeholder="Мой Lo-Fi для канала", key="preset_name",
                                 label_visibility="collapsed")
        save_clicked = c2.button("💾 Сохранить текущие настройки", key="preset_save")

    # Рецепты в одну кнопку
    with st.container(border=True):
        st.markdown("**⚡ Рецепты в одну кнопку** — сразу выставляют тональность, темп, зал, бит и аккорды")
        names = list(RECIPES)
        for row in (names[:4], names[4:]):
            cols = st.columns(4)
            for col, name in zip(cols, row):
                if col.button(name, key=f"recipe_{name}", help=RECIPES[name]["tip"], width="stretch"):
                    apply_preset({**collect_preset(), **recipe_values(name)})
                    st.session_state["recipe_done"] = name
                    st.rerun()
        done = st.session_state.pop("recipe_done", None)
        if done:
            st.success(f"Рецепт «{done}» применён: {RECIPES[done]['tip']}. Можно послушать 20 секунд внизу или "
                       "подправить ползунки.")

    # Шаг 2. Релиз и обложка
    with st.container(border=True):
        step_title(2, "🎨 Релиз и обложка")
        genre_label, genre, song_title, artist_name, genre_text, bg_file = cover_block("s_", "I WELCOME OCTOBER (Remix)")

    # Шаг 3. Скорость и звучание
    with st.container(border=True):
        step_title(3, "🎛️ Тональность, скорость и звучание")
        ki = st.session_state.get("keyinfo")
        if not ki or ki[0] != src_hash:
            with st.spinner("Определяем тональность и темп песни…"):
                try:
                    with tempfile.NamedTemporaryFile(suffix=Path(source_name).suffix or ".mp3") as tf:
                        tf.write(source_bytes)
                        tf.flush()
                        y0, sr0 = load_audio(tf.name)
                    ki = (src_hash, song_key_info(y0, sr0))
                    del y0
                except Exception:
                    ki = (src_hash, None)
            st.session_state["keyinfo"] = ki
        song_key = ki[1]
        preset_names = list(PRESETS)
        preset_name = st.selectbox("Пресет", preset_names, key=wk(f"preset_{genre_label}"),
                                   index=pidx(preset_names, f"preset_{genre_label}", genre["preset"]))
        p = PRESETS[preset_name]
        k = f"{genre_label}_{preset_name}"
        with st.expander("Тонкая настройка", expanded=preset_name == "Свои настройки"):
            link = st.checkbox("Эффект пластинки: тон меняется вместе со скоростью",
                               value=pv(f"link_{k}", p["link"]), key=wk(f"link_{k}"))
            if link:
                speed = st.slider("Скорость", 0.70, 1.40, float(pv(f"speed_{k}", p["speed"])), 0.01, key=wk(f"speed_{k}"))
                tempo, pitch = 1.0, 0.0
            else:
                speed = 1.0
                tempo = st.slider("Темп (без изменения тона)", 0.75, 1.30, float(pv(f"tempo_{k}", p["tempo"])), 0.01,
                                  key=wk(f"tempo_{k}"))
                pitch = st.slider("Тональность, полутоны", -6.0, 6.0, float(pv(f"pitch_{k}", p["pitch"])), 0.5,
                                  key=wk(f"pitch_{k}"))
            reverb = st.slider("Реверберация", 0.0, 1.0, float(pv(f"reverb_{k}", p["reverb"])), 0.05, key=wk(f"reverb_{k}"))
            bass = st.slider("Усиление баса", 0.0, 1.0, float(pv(f"bass_{k}", p["bass"])), 0.05, key=wk(f"bass_{k}"))
            vinyl = st.slider("Винил (тепло, шипение, щелчки)", 0.0, 1.0, float(pv(f"vinyl_{k}", p["vinyl"])), 0.05,
                              key=wk(f"vinyl_{k}"))
        c1, c2 = st.columns(2)
        formant = c1.checkbox("🗣 Живой голос при смене тона", value=pv("formant", True), key=wk("formant"),
                              help="Ноты сдвигаются, а тембр голоса остаётся своим: без «бурундука» при +3…+4 "
                                   "и без «великана» при −3…−4.")
        hz432 = c2.checkbox("🎼 Строй 432 Гц", value=pv("hz432", False), key=wk("hz432"),
                            help="Опускает строй с 440 до 432 Гц (на треть полутона) — популярно для музыки "
                                 "для сна, медитации и расслабления.")
        settings = dict(link=link, speed=speed, tempo=tempo, pitch=pitch, reverb=reverb, bass=bass, vinyl=vinyl,
                        formant=formant, hz432=hz432)
        if song_key:
            root, minor = song_key["root"], song_key["minor"]
            sh = total_shift(settings)
            shr = int(round(sh))
            st.markdown(f"🎼 **Тональность песни: {key_label(root, minor, True)}** · темп ≈ {song_key['bpm']:.0f} BPM")
            tips = " · ".join(f"**{d:+d}** → {key_label(root + d, minor)} — {t}" for d, t in KEY_SHIFT_TIPS.items()
                              if d in (-2, -1, 1, 2))
            st.caption("Куда сдвинуть: " + tips)
            if abs(sh) >= .05:
                approx = "≈ " if abs(sh - shr) > .1 else ""
                st.caption(f"С текущими настройками: {key_label(root, minor)} → **{approx}{key_label(root + shr, minor)}** "
                           f"({sh:+.1f} полутона)" + (" · строй 432 Гц" if hz432 else ""))
            st.markdown("**✨ Красивая тональность** — сайт сам подберёт сдвиг, темп и зал под настроение:")
            cols = st.columns(len(TONE_MOODS))
            for col, (mk, ml) in zip(cols, TONE_MOODS.items()):
                if col.button(ml, key=f"tone_{mk}", width="stretch"):
                    t = beautiful_tone(mk, minor, song_key["bpm"])
                    vals = {**collect_preset(), f"link_{k}": False, f"tempo_{k}": t["tempo"],
                            f"pitch_{k}": t["pitch"], f"reverb_{k}": t["reverb"], "formant": True}
                    apply_preset(vals)
                    st.session_state["tone_done"] = (ml, t)
                    st.rerun()
            done = st.session_state.pop("tone_done", None)
            if done:
                ml, t = done
                st.success(f"{ml}: тон {t['pitch']:+.0f} → {key_label(root + int(t['pitch']), minor)}, "
                           f"темп ×{t['tempo']:.2f}, зал {t['reverb']:.2f}. Голос остаётся живым.")
        else:
            st.caption("Тональность определить не получилось — настройте тон вручную в «Тонкой настройке».")

    # Шаг 4. Новый бит и слои
    with st.container(border=True):
        step_title(4, "🥁 Новый бит и музыкальные слои")
        beat_on = st.checkbox("Сделать новый бит", value=pv("beat_on", True), key=wk("beat_on"),
                              help="Находит доли трека с точностью до миллисекунд и пишет под них свой ритм.")
        my_styles = list_styles(username)
        style_labels = {"auto": f"Как у жанра — {BEAT_STYLES[genre['beat']]['name']}"}
        style_labels.update({f"tpl:{n}": f"🎯 Как образец: {n}" for n in my_styles})
        style_labels.update({key: s_["name"] for key, s_ in BEAT_STYLES.items() if key != "template"})
        opts = list(style_labels)
        style_choice = st.selectbox("Стиль бита", opts, format_func=style_labels.get, disabled=not beat_on,
                                    key=wk(f"beatstyle_{genre_label}"), index=pidx(opts, f"beatstyle_{genre_label}", "auto"))
        template, tpl_opts = None, {}
        if style_choice.startswith("tpl:") and style_choice[4:] in my_styles:
            style_key, template = "template", my_styles[style_choice[4:]]
        else:
            style_key = genre["beat"] if style_choice in ("auto",) or style_choice.startswith("tpl:") else style_choice
        bst = BEAT_STYLES[style_key]
        with st.expander("🎯 Ремикс по образцу — покажите готовый ремикс, и сайт сделает так же с любой песней",
                         expanded=template is not None):
            if template is not None and beat_on:
                st.caption(f"Образец «{template['name']}»: {template['tempo']:.0f} BPM, громкость "
                           f"{template['lufs']:.0f} LUFS, стерео {template['width']:.2f}")
                c1, c2 = st.columns(2)
                tpl_opts = dict(
                    tempo=c1.checkbox("Темп как у образца (голос не меняется)", value=True, key="tpl_tempo"),
                    sound=c1.checkbox("Тембр и ширина как у образца", value=True, key="tpl_sound",
                                      help="Сайт подгоняет баланс низов, середины и верхов и ширину стерео под образец."),
                    loud=c2.checkbox("Громкость как у образца", value=True, key="tpl_loud"),
                    chorus=c2.checkbox("Начать с припева", value=False, key="tpl_chorus",
                                       help="Как в клубных ремиксах: трек начинается с самой яркой части песни."))
                if st.button(f"🗑 Удалить образец «{template['name']}»", key="tpl_del"):
                    delete_style(username, template["name"])
                    st.rerun()
                st.divider()
            st.caption("Загрузите ремикс, который нравится: сайт изучит его темп, рисунок бочки, снейра, хэтов и баса, "
                       "тембр и громкость. Образец сохранится и появится в списке «Стиль бита». Сам образец в новый "
                       "трек не попадает — копируется только манера.")
            ref_file = st.file_uploader("Готовый ремикс-образец", type=["mp3", "wav", "ogg", "flac", "m4a"], key="tpl_file")
            c1, c2 = st.columns([3, 2])
            ref_name = c1.text_input("Название образца", key="tpl_name", placeholder="Клубный ремикс 110",
                                     label_visibility="collapsed")
            if c2.button("🧠 Изучить образец", key="tpl_learn", disabled=ref_file is None):
                name = (ref_name.strip() or Path(ref_file.name).stem)[:60]
                with st.spinner("Слушаем образец: темп, рисунок бита, тембр… (около минуты)"):
                    with tempfile.NamedTemporaryFile(suffix=Path(ref_file.name).suffix or ".mp3") as tf:
                        tf.write(ref_file.getvalue())
                        tf.flush()
                        yr, srr = load_audio(tf.name)
                    yr = yr[:, :int(240 * srr)]
                    try:
                        tpl_new = extract_style(yr, srr, name)
                    finally:
                        del yr
                save_style(username, tpl_new)
                st.session_state["tpl_learned"] = tpl_new
                st.rerun()
            learned = st.session_state.get("tpl_learned")
            if learned:
                bar = lambda v: "".join("█" if x >= .8 else "▓" if x >= .55 else "·" for x in v)
                st.success(f"Образец «{learned['name']}» изучен и сохранён: {learned['tempo']:.0f} BPM. "
                           "Выберите его в списке «Стиль бита».")
                st.code(f"бочка  {bar(learned['kick'])}\nснейр  {bar(learned['snare'])}\n"
                        f"хэты   {bar(learned['hat'])}\nбас    {bar(learned['bass'])}", language=None)
        kb = f"{genre_label}_{style_key}"
        rm_opts = ["hpss", "filter"]
        drum_removal = st.radio("Старые барабаны трека", rm_opts, horizontal=True, key=wk("drum_rm"),
                                index=pidx(rm_opts, "drum_rm", "hpss"), disabled=not beat_on,
                                format_func={"hpss": "Убрать (рекомендуется)", "filter": "Только приглушить"}.get,
                                help="«Убрать» вырезает удары и оставляет мелодию, аккорды и вокал — новый бит "
                                     "не спорит со старым. «Приглушить» — мягкий фильтр, звучит ближе к оригиналу.")
        retempo, retempo_bpm = False, None
        if bst.get("retempo") and beat_on:
            lo, hi, ideal = bst["retempo"]
            c1, c2 = st.columns([3, 2])
            retempo = c1.checkbox("Подогнать темп песни под стиль (голос не меняется)", value=pv(f"retempo_{style_key}", True),
                                  key=wk(f"retempo_{style_key}"),
                                  help="Медленную песню ускорит, быструю замедлит — до темпа танцпола. Тон голоса сохраняется.")
            retempo_bpm = c2.slider("Темп, BPM", lo, hi, int(pv(f"retempo_bpm_{style_key}", ideal)), 1,
                                    key=wk(f"retempo_bpm_{style_key}"), disabled=not retempo)
        with st.expander("Настройки бита"):
            beat_vol = st.slider("Громкость бита относительно трека", 0.3, 1.5, float(pv(f"bvol_{kb}", .9)), 0.05,
                                 key=wk(f"bvol_{kb}"), help="0,9 — сбалансировано: бит и музыка на одной громкости.")
            beat_filter = st.slider("Мягкость старого звука (фильтр верхов)", 0.0, 1.0, float(pv(f"bfilter_{kb}", bst["filter"])),
                                    0.05, key=wk(f"bfilter_{kb}"))
            beat_side = st.slider("Качание под бочку", 0.0, 1.0, float(pv(f"bside_{kb}", bst["side"])), 0.05, key=wk(f"bside_{kb}"))
            beat_density = st.slider("Сложность и сбивки", 0.0, 1.0, float(pv(f"bdens_{kb}", .5)), 0.05, key=wk(f"bdens_{kb}"))
            beat_bass = st.checkbox("Новый бас (808 / саб) по нотам трека", value=pv(f"bbass_{kb}", True), key=wk(f"bbass_{kb}"))
            c1, c2 = st.columns(2)
            t_opts = ["Как найден", "½× (в 2 раза медленнее)", "2× (в 2 раза быстрее)"]
            tempo_choice = c1.radio("Темп бита", t_opts, key=wk("btempo"), index=pidx(t_opts, "btempo", "Как найден"),
                                    help="Если бит спешит или тянется — переключите.")
            bpm_hint = c2.number_input("Темп трека вручную, BPM (0 — авто)", 0, 220, int(pv("bpm_hint", 0)), 1,
                                       key=wk("bpm_hint"), help="Для треков без барабанов или со сложным ритмом.")
            seed_key = wk("beat_seed")
            c1, c2 = st.columns([2, 1])
            if seed_key not in st.session_state:
                st.session_state[seed_key] = int(pv("beat_seed", 1))
            beat_seed = c1.number_input("Номер вариации бита", 1, 99999, key=seed_key,
                                        help="Тот же номер даёт тот же бит, другой — новый ритм.")
            c2.button("🎲 Другая", key="seed_btn",
                      on_click=lambda: st.session_state.update({seed_key: random.randint(1, 99999)}))
        beat_scale = {"Как найден": 1, "½× (в 2 раза медленнее)": 0.5}.get(tempo_choice, 2)
        keys_names = {"rhodes": "электропиано", "bell": "колокольчики", "organ": "хаус-орган",
                      "arp": "арпеджио-синт", "pluck": "плак-синт", "disco": "струнные и фанк-гитара",
                      "piano": "фортепиано", "strings_long": "струнные"}
        c1, c2 = st.columns([3, 2])
        layers_on = c1.checkbox("🎹 Красивые аккорды поверх песни (в её тональности)", value=pv("layers_on", True),
                                key=wk("layers_on"), disabled=not beat_on)
        mood_opts = list(CHORD_MOODS)
        chord_mood = c2.selectbox("Настроение аккордов", mood_opts, key=wk("chord_mood"),
                                  index=pidx(mood_opts, "chord_mood", "auto"), disabled=not (beat_on and layers_on),
                                  format_func=lambda m: "Как у стиля бита" if m == "auto" else CHORD_MOODS[m]["name"])
        lst = {**LAYER_STYLES[style_key], **(CHORD_MOODS[chord_mood] or {})}
        kb = f"{kb}_{chord_mood}"
        with st.expander("Слои, переходы и сведение"):
            layer_pad = st.slider("Аккорды (пэд)", 0.0, 1.0, float(pv(f"lpad_{kb}", lst["pad"])), 0.05, key=wk(f"lpad_{kb}"))
            layer_keys = st.slider(f"Клавиши: {keys_names[lst['keys_type']]}", 0.0, 1.0, float(pv(f"lkeys_{kb}", lst["keys"])),
                                   0.05, key=wk(f"lkeys_{kb}"))
            layer_echo = st.slider("Эхо клавиш в темп", 0.0, 1.0, float(pv(f"lecho_{kb}", lst["echo"])), 0.05, key=wk(f"lecho_{kb}"))
            layer_fx = st.checkbox("Переходы: брейк, райзер, реверс-тарелка и удар на дропе", value=pv(f"lfx_{kb}", True),
                                   key=wk(f"lfx_{kb}"))
            layer_polish = st.checkbox("Склейка и широкое стерео слоёв", value=pv(f"lpol_{kb}", True), key=wk(f"lpol_{kb}"))
        layers = dict(on=beat_on and layers_on, pad=layer_pad, keys=layer_keys, echo=layer_echo, fx=layer_fx,
                      polish=layer_polish, mood=chord_mood)

    # Шаг 5. Эффекты
    with st.container(border=True):
        step_title(5, "🎚️ Эффекты")
        fx_names = list(FX_PRESETS)
        fx_preset = st.selectbox("Набор эффектов", fx_names, key=wk("fx_preset"), index=pidx(fx_names, "fx_preset", fx_names[0]),
                                 help="Готовый набор выставляет ползунки ниже; дальше их можно менять.")
        base = {**FX_DEFAULTS, **FX_PRESETS[fx_preset]}
        kf = f"fx_{fx_preset}_"
        fx = {}

        def fs(col, label, name, lo, hi, step, help_=None):
            return col.slider(label, lo, hi, float(pv(kf + name, base[name])), step, key=wk(kf + name), help=help_)
        t1, t2, t3, t4, t5 = st.tabs(["Тембр", "Окраска", "Движение", "Пространство", "Ритм и финал"])
        with t1:
            c1, c2, c3 = st.columns(3)
            fx["eq_low"] = fs(c1, "Низы, дБ", "eq_low", -12.0, 12.0, .5)
            fx["eq_mid"] = fs(c2, "Середина, дБ", "eq_mid", -12.0, 12.0, .5)
            fx["eq_high"] = fs(c3, "Верха, дБ", "eq_high", -12.0, 12.0, .5)
            fx["hp"] = st.slider("Срез низов, Гц (20 — выключен)", 20, 1000, int(pv(kf + "hp", base["hp"])), 10, key=wk(kf + "hp"))
            fx["lp"] = st.slider("Срез верхов, Гц (20000 — выключен)", 500, 20000, int(pv(kf + "lp", base["lp"])), 100,
                                 key=wk(kf + "lp"), help="Низкие значения — эффект «из-за стены» или «под водой».")
        with t2:
            fx["saturation"] = fs(st, "Ламповая теплота (сатурация)", "saturation", 0.0, 1.0, .05)
            fx["distortion"] = fs(st, "Дисторшн", "distortion", 0.0, 1.0, .05)
            fx["bitcrush"] = fs(st, "Биткраш (звук ретро-приставки)", "bitcrush", 0.0, 1.0, .05)
            fx["radio"] = fs(st, "Старое радио / телефон", "radio", 0.0, 1.0, .05)
        with t3:
            fx["chorus"] = fs(st, "Хорус (объём, «плывущий» звук)", "chorus", 0.0, 1.0, .05)
            fx["flanger"] = fs(st, "Фленжер (реактивный «вжух»)", "flanger", 0.0, 1.0, .05)
            fx["phaser"] = fs(st, "Фэйзер (вращение)", "phaser", 0.0, 1.0, .05)
            fx["autofilter"] = fs(st, "Свип-фильтр в темп", "autofilter", 0.0, 1.0, .05)
            fx["tremolo"] = fs(st, "Тремоло (пульсация громкости)", "tremolo", 0.0, 1.0, .05)
        with t4:
            fx["eight_d"] = fs(st, "8D Audio (звук вокруг головы, в наушниках)", "eight_d", 0.0, 1.0, .05)
            fx["autopan"] = fs(st, "Автопанорама (лево–право в темп)", "autopan", 0.0, 1.0, .05)
            fx["width"] = fs(st, "Ширина стерео (1 — как есть, 0 — моно)", "width", 0.0, 2.0, .05)
            fx["room"] = fs(st, "Зал (дополнительная реверберация)", "room", 0.0, 1.0, .05)
            fx["echo"] = fs(st, "Эхо в темп", "echo", 0.0, 1.0, .05)
            c1, c2 = st.columns(2)
            echo_opts = [0.25, 0.5, 0.75, 1.0]
            echo_lbl = {0.25: "1/16", 0.5: "1/8", 0.75: "пунктирная 1/8", 1.0: "1/4"}
            fx["echo_beats"] = c1.selectbox("Шаг эха", echo_opts, format_func=echo_lbl.get, key=wk(kf + "eb"),
                                            index=pidx(echo_opts, kf + "eb", base["echo_beats"]))
            fx["echo_fb"] = fs(c2, "Повторы эха", "echo_fb", 0.1, 0.8, .05)
        with t5:
            fx["gate"] = fs(st, "Транс-гейт (рубленый ритм 1/16)", "gate", 0.0, 1.0, .05)
            fx["comp"] = fs(st, "Компрессор (плотнее и громче)", "comp", 0.0, 1.0, .05)
            fx["tapestop"] = fs(st, "Тейп-стоп в конце (плёнка останавливается)", "tapestop", 0.0, 1.0, .05)
        chosen_fx = active_fx(fx)
        st.caption("Включено: " + (", ".join(chosen_fx) if chosen_fx else "ничего"))

    # Шаг 6. Атмосфера
    with st.container(border=True):
        step_title(6, "🌧️ Атмосфера")
        amb_kind, amb_level, amb_file = ambience_block("s_")

    # Шаг 7. Профи ИИ-ремикс
    with st.container(border=True):
        step_title(7, "🤖 Профи ИИ-ремикс")
        ai_on = st.checkbox("Перегенерировать нейросетью Stable Audio", value=False, key="ai_on",
                            help="Нейросеть берёт трек как основу и заново исполняет его в выбранном стиле.")
        ai = None
        if ai_on:
            reason = ai_block_reason(api_key, ai_mode, username)
            if reason:
                st.warning(reason)
            elif ai_mode == "credits":
                st.caption("За ИИ-ремикс списывается 1 генерация (если нейросеть вернёт ошибку — вернётся).")
            ai_source = st.radio("Что отправить нейросети", ["remix", "original"], horizontal=True, key="ai_src",
                                 format_func={"remix": "Готовый трек со всеми эффектами", "original": "Исходный трек"}.get)
            ai_strength = st.slider("Насколько сильно переделать", 0.5, 0.95, 0.75, 0.05, key="ai_str",
                                    help="0,6–0,7 — близко к оригиналу; 0,8–0,9 — смелее.")
            ai_extra = st.text_input("Пожелания к звучанию (по-английски, необязательно)", key="ai_extra",
                                     placeholder="female vocal chops, rainy night mood")
            ai_model = st.selectbox("Модель", ["stable-audio-2.5", "stable-audio-2"], key="ai_model")
            if not reason:
                ai = dict(source=ai_source, strength=ai_strength, extra=ai_extra, model=ai_model, key=api_key, mode=ai_mode)

    # Шаг 8. Мастеринг и экспорт
    with st.container(border=True):
        step_title(8, "💎 Мастеринг и экспорт")
        master_on = st.checkbox("Профессиональный мастеринг", value=pv("m_on", True), key=wk("m_on"),
                                help="Ровная громкость по стандарту площадки, лимитер без искажений, плотный бас.")
        t_keys = list(MASTER_TARGETS)
        m_target = st.selectbox("Громкость под площадку", t_keys, format_func=lambda x: MASTER_TARGETS[x]["name"],
                                key=wk("m_target"), index=pidx(t_keys, "m_target", "youtube"), disabled=not master_on)
        with st.expander("Тонкая настройка мастеринга"):
            m_glue = st.slider("Склейка (шинный компрессор)", 0.0, 1.0, float(pv("m_glue", .3)), 0.05, key=wk("m_glue"))
            m_mono = st.checkbox("Плотный бас: всё ниже 120 Гц в центр", value=pv("m_mono", True), key=wk("m_mono"))
            m_warm = st.slider("Теплота (середина-низ)", 0.0, 1.0, float(pv("m_warm", 0.0)), 0.05, key=wk("m_warm"))
            m_air = st.slider("Воздух (блеск верхов)", 0.0, 1.0, float(pv("m_air", 0.0)), 0.05, key=wk("m_air"))
        st.markdown("**✂️ Обрезка и плавность**")
        c1, c2 = st.columns(2)
        trim_start = c1.number_input("Начать с, сек", 0.0, 900.0, 0.0, 0.5, key="trim_start")
        trim_end = c2.number_input("Закончить на, сек (0 — до конца)", 0.0, 900.0, 0.0, 0.5, key="trim_end")
        c1, c2 = st.columns(2)
        fade_in = c1.slider("Плавное начало, сек", 0.0, 10.0, float(pv("fade_in", 0.0)), 0.5, key=wk("fade_in"))
        fade_out = c2.slider("Плавный конец, сек", 0.0, 15.0, float(pv("fade_out", 3.0)), 0.5, key=wk("fade_out"))
        wav24 = st.checkbox("WAV 24 бит (студийный; иначе 16 бит с дизерингом)", value=pv("wav24", False), key=wk("wav24"))
        st.markdown("**🎬 Видео**")
        video = video_block("s_")

    if save_clicked:
        if not new_name.strip():
            st.warning("Введите название пресета.")
        else:
            save_preset(username, new_name, collect_preset())
            st.success(f"Пресет «{new_name.strip()}» сохранён — он появится в списке «Мои пресеты».")

    params = dict(
            genre_label=genre_label, title=song_title, artist=artist_name, genre_text=genre_text, label=p["label"],
            src_hash=hashlib.md5(source_bytes).hexdigest(), trim_start=trim_start, trim_end=trim_end,
            beat_on=beat_on, bpm_hint=bpm_hint or None, drum_removal=drum_removal, style_key=style_key,
            beat_vol=beat_vol, beat_filter=beat_filter, beat_side=beat_side, beat_density=beat_density,
            beat_bass=beat_bass, beat_seed=int(beat_seed), beat_scale=beat_scale, layers=layers,
            retempo=retempo, retempo_bpm=retempo_bpm, template=template, tpl_opts=tpl_opts,
            settings=settings, fx=fx, master=dict(on=master_on, target=m_target, glue=m_glue, mono_bass=m_mono,
                                                  warmth=m_warm, air=m_air),
            fade_in=fade_in, fade_out=fade_out, bits=24 if wav24 else 16,
            ambience={"kind": None if amb_kind == "none" else amb_kind, "level_db": amb_level},
            ai=ai, video={kk: vv for kk, vv in video.items() if kk != "logo"})

    with st.container(border=True):
        st.markdown("**🎧 Прослушать 20 секунд перед запуском** — тот же звук, что получится, на самом ярком куске песни")
        if st.button("🎧 Прослушать «было / стало»", key="preview_btn", width="stretch",
                     disabled=bool(trim_end and trim_end <= trim_start)):
            with st.spinner("Готовим 20 секунд с вашими настройками (обычно 10–30 секунд)…"):
                try:
                    st.session_state["preview"] = (src_hash,) + studio_preview(source_bytes, source_name, params)
                except Exception as err:
                    st.session_state.pop("preview", None)
                    st.error(f"Прослушка не получилась: {err}")
        pr = st.session_state.get("preview")
        if pr and pr[0] == src_hash:
            c1, c2 = st.columns(2)
            c1.caption(f"Было (с {fmt_ts(pr[3])}, громкость выровнена для честного сравнения)")
            c1.audio(pr[1], format="audio/mpeg")
            c2.caption("Стало")
            c2.audio(pr[2], format="audio/mpeg")
            st.caption("Поменяли настройки — нажмите «Прослушать» ещё раз.")

    if st.button("🚀 Создать трек, обложку, видео и SEO", type="primary", width="stretch"):
        if not song_title.strip():
            st.warning("Введите название трека.")
            return
        if trim_end and trim_end <= trim_start:
            st.warning("Конец обрезки должен быть позже начала.")
            return
        if amb_kind == "custom" and amb_file is None:
            st.warning("Загрузите файл атмосферы или выберите другую.")
            return

        def prepare(ctx):
            src = ctx.path("source" + (Path(source_name).suffix.lower() or ".mp3"))
            with open(src, "wb") as fh:
                fh.write(source_bytes)
            params["src_path"] = src
            params["bg_path"] = save_upload(ctx, bg_file, "bg")
            params["video"]["logo_path"] = save_upload(ctx, video.get("logo"), "logo")
            if amb_kind == "custom":
                params["ambience"]["path"] = save_upload(ctx, amb_file, "ambience")
        submit_ui(username, "studio", song_title, run_studio_job, params, prepare)


def render_diagnosis(m, issues):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Громкость", f"{m['lufs']:.1f} LUFS")
    c2.metric("Пик", f"{m['true_peak']:+.1f} dBTP")
    c3.metric("Бас к середине", f"{m['bass_db']:+.0f} дБ")
    c4.metric("Стерео", f"{m['correlation']:+.2f}")
    icon = {"bad": "🔴", "warn": "🟠", "info": "🔵", "ok": "🟢"}
    for lvl, title, advice in issues:
        st.markdown(f"{icon[lvl]} **{title}** — {advice}")


# --- страница: Длинная версия ---

def render_long(username):
    with st.container(border=True):
        step_title(1, "♾ Источник")
        done = [j for j in list_jobs(username, 50) if j["status"] == "done" and j["kind"] == "studio"
                and any(f["name"] == "track.wav" for f in j["result"].get("files", []))]
        opts = ["upload"] + [j["id"] for j in done]
        names = {"upload": "📁 Загрузить готовый трек", **{j["id"]: f"🎛 {j['title']} ({j['created_at'][:16]})" for j in done}}
        default = st.session_state.pop("long_src_job", None)
        if default in opts:
            st.session_state["long_src"] = default
        elif st.session_state.get("long_src") not in opts:
            st.session_state["long_src"] = opts[1] if len(opts) > 1 else opts[0]
        src = st.selectbox("Что зациклить", opts, format_func=names.get, key="long_src")
        up = None
        if src == "upload":
            up = st.file_uploader("Трек (.mp3 или .wav)", type=["mp3", "wav"], key="long_up")
        c1, c2 = st.columns(2)
        minutes = c1.selectbox("Длительность", [30, 60, 90, 120, 180], index=1, key="long_min",
                               format_func=lambda m: f"{m} мин" if m < 60 else f"{m // 60} ч" + (" 30 мин" if m % 60 else ""))
        crossfade = c2.slider("Плавность стыка, сек", 2.0, 12.0, 6.0, 0.5, key="long_xf")
        master_on = st.checkbox("Мастеринг под YouTube (−14 LUFS)", value=src == "upload", key="long_master",
                                help="Результаты студии уже прошли мастеринг — повторно не нужно.")
    with st.container(border=True):
        step_title(2, "🌧️ Атмосфера")
        amb_kind, amb_level, amb_file = ambience_block("l_", -16)
        st.caption("Атмосфера идёт непрерывно поверх всего ролика — без повторов на стыках трека.")
    with st.container(border=True):
        step_title(3, "🎨 Обложка, видео и SEO")
        default_title = next((j["title"] for j in done if j["id"] == src), "Night Rain")
        genre_label, genre, title, artist, genre_text, bg = cover_block("l_", default_title)
        video = video_block("l_", allow_wave=minutes <= 20, allow_lyrics=False)
    if st.button(f"♾ Собрать версию на {minutes} мин", type="primary", width="stretch"):
        if src == "upload" and up is None:
            st.warning("Загрузите трек или выберите результат из кабинета.")
            return
        if amb_kind == "custom" and amb_file is None:
            st.warning("Загрузите файл атмосферы.")
            return
        label = PRESETS[GENRES[genre_label]["preset"]]["label"]
        params = dict(genre_label=genre_label, title=title, artist=artist, genre_text=genre_text, label=label,
                      minutes=minutes, crossfade=crossfade, master_on=master_on,
                      ambience={"kind": None if amb_kind == "none" else amb_kind, "level_db": amb_level},
                      video={kk: vv for kk, vv in video.items() if kk != "logo"})

        def prepare(ctx):
            if src == "upload":
                params["src_path"] = save_upload(ctx, up, "source")
            else:
                params["src_path"] = str(JOBS_DIR / src / "track.wav")
            params["bg_path"] = save_upload(ctx, bg, "bg")
            params["video"]["logo_path"] = save_upload(ctx, video.get("logo"), "logo")
            if amb_kind == "custom":
                params["ambience"]["path"] = save_upload(ctx, amb_file, "ambience")
        submit_ui(username, "long", f"{title} · {minutes} мин", run_long_job, params, prepare)


# --- страница: Микс ---

def render_mix(username):
    with st.container(border=True):
        step_title(1, "🎚 Треки")
        files = st.file_uploader("Загрузите 2–20 треков (.mp3 или .wav) в нужном порядке", type=["mp3", "wav"],
                                 accept_multiple_files=True, key="mix_files")
        files = list(files or [])[:20]
        titles = []
        if files:
            sort_az = st.checkbox("Отсортировать по имени файла", value=False, key="mix_sort")
            if sort_az:
                files = sorted(files, key=lambda f: f.name.lower())
            default = "\n".join(re.sub(r"[_]+", " ", Path(f.name).stem) for f in files)
            txt = st.text_area("Названия для таймкодов (по одному на строку, в том же порядке)", default,
                               height=min(400, 30 + 24 * len(files)), key=f"mix_titles_{len(files)}_{sort_az}")
            titles = [t.strip() for t in txt.splitlines() if t.strip()]
            if len(titles) != len(files):
                st.warning(f"Названий {len(titles)}, а треков {len(files)} — поправьте список.")
        crossfade = st.slider("Плавный переход между треками, сек", 0.0, 12.0, 5.0, 0.5, key="mix_xf")
    with st.container(border=True):
        step_title(2, "🎛️ Звучание всего микса")
        c1, c2 = st.columns(2)
        sound = c1.selectbox("Обработка", list(MIX_SOUND), key="mix_sound")
        fx_name = c2.selectbox("Эффекты", list(FX_PRESETS), key="mix_fx")
        target = st.selectbox("Громкость", list(MASTER_TARGETS), format_func=lambda x: MASTER_TARGETS[x]["name"], key="mix_target")
        st.caption("Каждый трек выводится на одинаковую громкость — в миксе ничего не «прыгает».")
        amb_kind, amb_level, amb_file = ambience_block("m_", -18)
    with st.container(border=True):
        step_title(3, "🎨 Обложка, видео и SEO")
        genre_label, genre, mix_title, _, genre_text, bg = cover_block("m_", "Night Lo-Fi Mix", show_artist=False)
        video = video_block("m_", allow_wave=True, allow_lyrics=False)
        st.caption("Если микс длиннее 20 минут, видео собирается в быстром режиме: анимированный фон без волны.")
    if st.button("🎚 Собрать микс", type="primary", width="stretch"):
        if len(files) < 2:
            st.warning("Нужно минимум 2 трека.")
            return
        if len(titles) != len(files):
            st.warning("Количество названий должно совпадать с количеством треков.")
            return
        params = dict(genre_label=genre_label, title=mix_title, artist="", genre_text=genre_text,
                      label=sound if sound != "Без изменений" else genre["short"], crossfade=crossfade,
                      settings=MIX_SOUND[sound], fx={**FX_DEFAULTS, **FX_PRESETS[fx_name]},
                      master=dict(on=True, target=target),
                      ambience={"kind": None if amb_kind == "none" else amb_kind, "level_db": amb_level},
                      video={kk: vv for kk, vv in video.items() if kk != "logo"})

        def prepare(ctx):
            params["tracks"] = [{"path": save_upload(ctx, f, f"t{i:02d}"), "title": t} for i, (f, t) in enumerate(zip(files, titles))]
            params["bg_path"] = save_upload(ctx, bg, "bg")
            params["video"]["logo_path"] = save_upload(ctx, video.get("logo"), "logo")
            if amb_kind == "custom":
                params["ambience"]["path"] = save_upload(ctx, amb_file, "ambience")
        submit_ui(username, "mix", mix_title, run_mix_job, params, prepare)


# --- страница: Мой кабинет ---

def render_cabinet(username):
    st.subheader("📁 Мой кабинет")
    st.caption(f"Результаты хранятся {get_setting('JOB_KEEP_DAYS', '7')} дней. Скачайте нужное заранее.")
    jobs = list_jobs(username, 50)
    finished = [j for j in jobs if j["status"] in ("done", "error")]
    if not jobs:
        st.info("Пока пусто. Создайте трек в «Студии», длинную версию или микс — результаты появятся здесь.")
        return
    if not finished:
        st.info("Задачи в работе — прогресс выше. Результат появится здесь сам.")
        return
    names = {j["id"]: f"{JOB_STATUS[j['status']].split()[0]} {JOB_KINDS.get(j['kind'], '')} · {j['title']} · "
                      f"{(j['finished_at'] or j['created_at'])[:16]}" for j in finished}
    jid = st.selectbox("Результат", [j["id"] for j in finished], format_func=names.get, key="cab_pick")
    j = next(x for x in finished if x["id"] == jid)
    r = j["result"]
    with st.container(border=True):
        if j["status"] == "error":
            st.error(j["message"])
        if r.get("note"):
            st.warning(r["note"])
        files = r.get("files", [])
        folder = JOBS_DIR / jid
        img = next((f for f in files if f["kind"] == "image"), None)
        if img and (folder / img["name"]).exists():
            st.image(str(folder / img["name"]), caption="Обложка 1280×720")
        if j["kind"] == "studio" and r.get("duration"):
            rep = r.get("master")
            c1, c2, c3 = st.columns(3)
            if rep:
                c1.metric("Громкость", f"{rep['lufs_out']:.1f} LUFS", f"{rep['lufs_out'] - rep['lufs_in']:+.1f} дБ")
                c2.metric("Пик", f"{rep['tp_out']:.1f} dBTP")
            c3.metric("Длительность", fmt_ts(r["duration"]))
            beat = r.get("beat")
            if beat:
                timing = {2: " · half-time", 0.5: " · double-time"}.get(beat["mult"], "")
                if beat.get("retempo"):
                    st.markdown(f"**⏩ Темп песни:** {beat['retempo']} (тон голоса сохранён)")
                if beat.get("template"):
                    st.markdown(f"**🎯 По образцу:** {beat['template']}")
                st.markdown(f"**🥁 Новый бит:** {beat['style']}{timing} · {beat['bpm_out']:.1f} BPM · "
                            f"бас трека {beat['key']}" + (f" · аккорды {beat['chords']}" if beat.get("chords") else ""))
                if not beat.get("sure", True):
                    st.caption("Ритм трека выражен слабо. Если бит не попадает — задайте темп вручную в настройках бита.")
            if r.get("fx"):
                st.markdown("**🎚️ Эффекты:** " + ", ".join(r["fx"]))
            if r.get("ambience"):
                st.markdown(f"**🌧️ Атмосфера:** {r['ambience']}")
        elif r.get("duration"):
            st.metric("Длительность", fmt_ts(r["duration"]))
        for f in files:
            p_ = folder / f["name"]
            if not p_.exists() or f["kind"] not in ("audio", "video"):
                continue
            if f["size"] > 120e6:
                st.caption(f"{f['label']}: {fmt_size(f['size'])} — слишком большой для просмотра в браузере, скачайте.")
            elif f["kind"] == "audio":
                st.caption(f["label"])
                st.audio(str(p_))
            else:
                st.video(str(p_))
        if r.get("timestamps"):
            st.markdown("**⏱ Таймкоды для описания**")
            st.code(r["timestamps"], language="text")
        if r.get("titles"):
            with st.expander("🚀 YouTube SEO (English)", expanded=True):
                for t in r["titles"]:
                    st.code(t, language="text")
                st.code(r.get("description", ""), language="text")
                st.code(r.get("tags", ""), language="text")
        if files:
            st.markdown("**📥 Скачать**")
            cols = st.columns(min(4, len(files)))
            for i, f in enumerate(files):
                p_ = folder / f["name"]
                if not p_.exists():
                    continue
                label = f"{f['label']} · {fmt_size(f['size'])}"
                fname = f"{slugify(j['title'])}{Path(f['name']).suffix}"
                try:   # файл читается с диска только при нажатии — большие видео не занимают память
                    cols[i % len(cols)].download_button(label, functools.partial(Path.read_bytes, p_), fname, f["mime"],
                                                        key=f"dl_{jid}_{i}", on_click="ignore", width="stretch")
                except Exception:   # старая версия Streamlit без «ленивых» файлов
                    cols[i % len(cols)].download_button(label, p_.read_bytes(), fname, f["mime"],
                                                        key=f"dl_{jid}_{i}", width="stretch")
        c1, c2 = st.columns(2)
        if j["kind"] == "studio" and j["status"] == "done":
            c1.button("♾ Сделать версию на 1–3 часа", key=f"mk_long_{jid}",
                      on_click=lambda: st.session_state.update(long_src_job=jid, nav="♾ Длинная версия"))
        if c2.button("🗑 Удалить результат", key=f"del_{jid}"):
            delete_job(username, jid)
            st.rerun()


# --- страница: Ниша YouTube ---

def render_niche():
    st.subheader("📈 Анализ ниши YouTube")
    st.caption("Смотрим самые популярные ролики по запросу за последние дни: сколько просмотров в день они набирают "
               "и сколько конкурентов. Чем выше «оценка», тем выгоднее делать такой контент.")
    key = get_setting("YOUTUBE_API_KEY", "")
    if not key:
        key = st.text_input("Ключ YouTube Data API", type="password", key="yt_key",
                            help="console.cloud.google.com → создайте проект → включите «YouTube Data API v3» → "
                                 "Credentials → Create API key. Бесплатно ~100 запросов в день.")
        st.caption("Администратор может добавить ключ один раз на сервере (переменная YOUTUBE_API_KEY), тогда поле не нужно.")
    import datetime
    year = datetime.date.today().year
    styles = {k: BEAT_STYLES[k]["name"] for k in NICHE_QUERIES if k in BEAT_STYLES}
    styles["slowed"] = "Slowed + Reverb"
    c1, c2 = st.columns([2, 1])
    niche = c1.selectbox("Ниша", list(NICHE_QUERIES), format_func=lambda k: styles.get(k, k), key="niche")
    days = c2.selectbox("Период", [7, 30, 90], index=1, format_func=lambda d: f"{d} дней", key="niche_days")
    q_text = st.text_area("Запросы (по одному на строку)", "\n".join(q.format(year=year) for q in NICHE_QUERIES[niche]),
                          height=140, key=f"niche_q_{niche}")
    queries = [q.strip() for q in q_text.splitlines() if q.strip()][:10]
    if st.button("📈 Анализировать", type="primary", disabled=not key):
        rows = []
        with st.spinner("Запрашиваем YouTube…"):
            for q in queries:
                try:
                    rows.append(yt_niche(key, q, days))
                except Exception as err:
                    st.error(f"«{q}»: {err}")
                    break
        st.session_state["niche_rows"] = rows
    rows = st.session_state.get("niche_rows")
    if rows:
        rows = sorted(rows, key=lambda r: -r["score"])
        st.dataframe([{"Запрос": r["query"], "Просмотров в день (медиана топа)": int(r["median_per_day"]),
                       "Роликов по запросу (≈)": r["total"], "Оценка": round(r["score"])} for r in rows],
                     hide_index=True, width="stretch")
        best = rows[0]
        st.success(f"Лучшая возможность сейчас: «{best['query']}» — топ-ролики набирают около "
                   f"{int(best['median_per_day']):,} просмотров в день.".replace(",", " "))
        for r in rows:
            with st.expander(f"Топ роликов: {r['query']}"):
                for v in r["top"]:
                    st.markdown(f"[{v['title']}]({v['url']}) · {v['channel']} · {int(v['per_day']):,} в день".replace(",", " "))


# --- страница: Названия ---

def render_names():
    st.subheader("🏷 Названия каналов и плейлистов")
    styles = {k: BEAT_STYLES.get(k, {"name": "Slowed + Reverb"})["name"] for k in NAME_BANK}
    style = st.selectbox("Ниша", list(NAME_BANK), format_func=styles.get, key="names_style")
    st.session_state.setdefault("names_seed", 1)
    st.button("🎲 Ещё варианты", on_click=lambda: st.session_state.update(names_seed=random.randint(1, 10 ** 6)))
    channels, playlists = generate_names(style, st.session_state["names_seed"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Названия каналов**")
        for n in channels:
            st.code(n, language="text")
    with c2:
        st.markdown("**Названия плейлистов**")
        for n in playlists:
            st.code(n, language="text")
    st.caption("Перед выбором проверьте, что такого канала ещё нет в поиске YouTube.")


def render_admin_page(username):
    render_admin(username)
    st.markdown("---")
    st.markdown("**💽 Хранилище результатов**")
    total = sum(f.stat().st_size for f in JOBS_DIR.rglob("*") if f.is_file()) if JOBS_DIR.exists() else 0
    st.caption(f"Занято: {fmt_size(total)}. Старые результаты удаляются через {get_setting('JOB_KEEP_DAYS', '7')} дней.")
    if st.button("Очистить результаты старше 2 дней"):
        cleanup_jobs(2)
        st.success("Готово.")


def main():
    st.set_page_config(page_title="NBS SOFT MUSIC", page_icon="🎧", layout="centered")
    inject_design()
    init_db()
    job_executor()                      # запускает очередь задач один раз на сервер
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("username", "")

    if not st.session_state["logged_in"]:
        token = st.query_params.get("s")            # связь оборвалась или страница перезагрузилась — входим сами
        restored = user_from_token(token)
        if restored:
            st.session_state.update(logged_in=True, username=restored, auth_token=token)
        else:
            if token:
                del st.query_params["s"]
            render_auth()
            st.stop()
    elif st.session_state.get("auth_token") and st.query_params.get("s") != st.session_state["auth_token"]:
        st.query_params["s"] = st.session_state["auth_token"]   # ссылка в адресной строке всегда с входом

    username = st.session_state["username"]
    user = get_user(username)
    if not user or not user["is_active"]:
        delete_login_token(st.session_state.get("auth_token"))
        st.query_params.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.error("Аккаунт заблокирован или удалён. Обратитесь к администратору.")
        st.stop()
    admin = is_admin(username)
    pages = PAGES + ([ADMIN_PAGE] if admin else [])
    if st.session_state.get("nav") not in pages:
        st.session_state["nav"] = pages[0]

    with st.sidebar:
        st.markdown("<div class='nbs-side-logo'>NBS<span>SOFT MUSIC</span></div>", unsafe_allow_html=True)
        st.write("")
        st.write(f"👤 **{username}**" + (" · администратор" if admin else ""))
        st.radio("Раздел", pages, key="nav", label_visibility="collapsed")
        st.markdown("**🤖 ИИ-генерации**")
        if admin:
            st.caption("Безлимит (администратор)")
        else:
            st.metric("Баланс", int(user["ai_credits"] or 0))
            st.caption(get_setting("BUY_INFO", "Чтобы пополнить, напишите администратору."))
        with st.expander("Свой ключ Stability AI"):
            st.text_input("API-ключ", type="password", key="own_key", help="platform.stability.ai → API Keys")
            st.caption("Со своим ключом генерации не списываются с баланса: платит ваш счёт Stability.")
        if st.button("🚪 Выйти"):
            delete_login_token(st.session_state.get("auth_token"))
            st.query_params.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    api_key, ai_mode = ai_context(username)
    page = st.session_state["nav"]
    if page == "🎛 Студия":
        render_hero()
    render_active_jobs(username)
    if page == "🎛 Студия":
        render_studio(username, api_key, ai_mode)
    elif page == "♾ Длинная версия":
        render_long(username)
    elif page == "🎚 Микс":
        render_mix(username)
    elif page == "📁 Мой кабинет":
        render_cabinet(username)
    elif page == "📈 Ниша YouTube":
        render_niche()
    elif page == "🏷 Названия":
        render_names()
    elif page == ADMIN_PAGE and admin:
        render_admin_page(username)


if __name__ == "__main__":
    main()
