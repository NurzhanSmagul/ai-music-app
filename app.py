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
from scipy.signal import butter, oaconvolve, sosfilt

import beatgen

# =====================================================================
# НАСТРОЙКИ
# =====================================================================

DB_PATH = Path(__file__).resolve().with_name("users.db")
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
        "short": "Phonk", "preset": "Phonk Drift",
        "photos": [1078, 1062, 1048, 111], "accent": (255, 70, 90),
    },
    "NIGHTCORE / FAST VIBE (Энергичный высокий темп)": {
        "short": "Sped Up", "preset": "Sped Up / Nightcore",
        "photos": [1015, 1016, 1042], "accent": (120, 240, 255),
    },
    "DEEP HOUSE / CLUB (Клубный ритм)": {
        "short": "Deep House", "preset": "Deep House",
        "photos": [1060, 1068, 1074, 1081], "accent": (90, 255, 200),
    },
    "SYNTHWAVE / RETRO 80s (Космический ретро-вайб)": {
        "short": "Synthwave", "preset": "Synthwave",
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
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
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


def shift_pitch_and_tempo(y, sr, pitch, tempo):
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


def process_audio(y, sr, s):
    if s["link"]:
        y = change_speed(y, sr, s["speed"])
    else:
        y = shift_pitch_and_tempo(y, sr, s["pitch"], s["tempo"])
    y = bass_boost(y, sr, s["bass"])
    y = apply_reverb(y, sr, s["reverb"])
    y = normalize(y, -3.0)
    y = apply_vinyl(y, sr, s["vinyl"])
    y = normalize(y, -1.0)
    return apply_fades(y, sr).astype(np.float32)


def encode_audio(y, sr, fmt):
    buf = io.BytesIO()
    if fmt == "WAV":
        sf.write(buf, y.T, sr, format="WAV", subtype="PCM_16")
    else:
        sf.write(buf, y.T, sr, format="MP3")
    return buf.getvalue()


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


def render_auth():
    st.title("🔐 Вход в AI Music Suite")
    st.write("Войдите или зарегистрируйтесь, чтобы открыть студию ремиксов.")
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


def render_results(res):
    st.success("🎉 Готово! Ремикс, обложка и SEO созданы.")
    st.subheader("📥 Результаты")
    st.image(res["cover"], caption="Обложка для YouTube (1280×720)")
    st.audio(res["wav"], format="audio/wav")

    cols = st.columns(3 if res.get("mp3") else 2)
    cols[0].download_button("🖼️ Обложка (JPG)", res["cover"], f"{res['slug']}_cover.jpg",
                            "image/jpeg", key="dl_cover")
    cols[1].download_button("🎧 Ремикс (WAV)", res["wav"], f"{res['slug']}.wav",
                            "audio/wav", key="dl_wav")
    if res.get("mp3"):
        cols[2].download_button("🎵 Ремикс (MP3)", res["mp3"], f"{res['slug']}.mp3",
                                "audio/mpeg", key="dl_mp3")

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
    st.set_page_config(page_title="AI Music & Remix Studio", page_icon="🎛️", layout="centered")
    init_db()
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("username", "")

    if not st.session_state["logged_in"]:
        render_auth()
        st.stop()

    st.sidebar.write(f"👤 Аккаунт: **{st.session_state['username']}**")
    if st.sidebar.button("🚪 Выйти"):
        for key in ("logged_in", "username", "result", "file_key"):
            st.session_state.pop(key, None)
        st.rerun()

    st.title("🎛️ AI Remix Studio & YouTube SEO")
    st.write("Ремиксы ваших треков, обложка для YouTube и готовое английское SEO.")

    uploaded = st.file_uploader("Загрузите трек (.mp3 или .wav)", type=["mp3", "wav"])
    if uploaded is None:
        st.info("Загрузите свой трек или трек, на который у вас есть права.")
        return

    file_key = f"{uploaded.name}-{uploaded.size}"
    if st.session_state.get("file_key") != file_key:
        st.session_state["file_key"] = file_key
        st.session_state.pop("result", None)
    st.audio(uploaded)

    # --- Релиз и обложка ---
    st.subheader("🎨 Релиз и обложка")
    genre_label = st.selectbox("🎵 Ниша / стиль", list(GENRES))
    genre = GENRES[genre_label]
    song_title = st.text_input("Название трека", "I WELCOME OCTOBER (Remix)")
    artist_name = st.text_input("Артист", "Smagulov")
    genre_text = st.text_input("Текст жанра на обложке", genre["short"], key=f"gtext_{genre_label}")
    bg_file = st.file_uploader("Свой фон для обложки (необязательно)", type=["jpg", "jpeg", "png", "webp"])

    # --- Ремикс ---
    st.subheader("🎛️ Ремикс")
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

    settings = dict(link=link, speed=speed, tempo=tempo, pitch=pitch,
                    reverb=reverb, bass=bass, vinyl=vinyl)

    if st.button("🚀 Создать ремикс, обложку и SEO", type="primary"):
        if not song_title.strip():
            st.warning("Введите название трека.")
            st.stop()

        suffix = Path(uploaded.name).suffix.lower() or ".mp3"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(uploaded.getvalue())
            tmp.close()

            with st.status("Обработка…", expanded=True) as status:
                st.write("🔄 Загружаем аудио…")
                if librosa.get_duration(path=tmp.name) > MAX_TRACK_MINUTES * 60:
                    status.update(label="Трек слишком длинный", state="error")
                    st.error(f"Трек длиннее {MAX_TRACK_MINUTES} минут.")
                    st.stop()
                y, sr = load_audio(tmp.name)

                st.write("🎚️ Применяем эффекты…")
                y = process_audio(y, sr, settings)
                wav = encode_audio(y, sr, "WAV")
                try:
                    mp3 = encode_audio(y, sr, "MP3")
                except Exception:  # старая libsndfile без MP3 — отдаём только WAV
                    mp3 = None

                st.write("🎨 Рисуем обложку…")
                background = None
                if bg_file is not None:
                    try:
                        background = Image.open(bg_file).convert("RGB")
                    except OSError:
                        st.warning("Не удалось открыть ваш фон — использую стандартный.")
                background = background or fetch_background(genre["photos"]) or gradient_background(genre["accent"])
                cover = make_cover(song_title, artist_name, genre_text, genre["accent"], background)

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
            tags=tags, slug=slugify(f"{artist_name}_{song_title}"),
        )

    if "result" in st.session_state:
        render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
