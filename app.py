import streamlit as st
import os
import random
import time
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import librosa
import soundfile as sf
import numpy as np
import sqlite3
import hashlib

# --- БАЗА ДАННЫХ ДЛЯ ПОЛЬЗОВАТЕЛЕЙ ---
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            is_active INTEGER DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0] == hash_password(password):
        return True
    return False

# Настройка страницы
st.set_page_config(page_title="NBS SOFT — AI Music & FX Studio", page_icon="👑", layout="centered")

# Сессия для авторизации
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# --- ЭКРАН АВТОРИЗАЦИИ / РЕГИСТРАЦИИ ---
if not st.session_state["logged_in"]:
    st.title("🔐 NBS SOFT — Вход в платформу")
    st.write("Войдите в систему для доступа к студии эффектов, микшеру и вирусным роликам.")
    
    tab1, tab2 = st.tabs(["🔑 Вход", "📝 Регистрация"])
    
    with tab1:
        login_user = st.text_input("Логин (Email или имя)", key="login_u")
        login_pass = st.text_input("Пароль", type="password", key="login_p")
        if st.button("Войти в систему"):
            if verify_user(login_user, login_pass):
                st.session_state["logged_in"] = True
                st.session_state["username"] = login_user
                st.success("Успешный вход!")
                st.rerun()
            else:
                st.error("Неверный логин или пароль")
                
    with tab2:
        reg_user = st.text_input("Придумайте логин", key="reg_u")
        reg_pass = st.text_input("Придумайте пароль", type="password", key="reg_p")
        if st.button("Зарегистрироваться"):
            if reg_user and reg_pass:
                if register_user(reg_user, reg_pass):
                    st.success("Регистрация успешна! Перейдите во вкладку «Вход».")
                else:
                    st.error("Такой логин уже занят.")
            else:
                st.warning("Заполните все поля.")
    
    st.stop()

# --- БОКОВАЯ ПАНЕЛЬ С БРЕНДИНГОМ NBS SOFT ---
if os.path.exists("logo_cropped.png"):
    st.sidebar.image("logo_cropped.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.write(f"👤 Аккаунт: **{st.session_state['username']}**")
if st.sidebar.button("🚪 Выйти из аккаунта"):
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.rerun()

# --- ОСНОВНОЙ ФУНКЦИОНАЛ ПРИЛОЖЕНИЯ ---
st.title("👑 NBS SOFT — Профессиональная студия эффектов")
st.write("Тонкая настройка реверберации, эха, басов, винила и микширования битов!")

# Загрузка трека
uploaded_file = st.file_uploader("Загрузите исходный трек (.mp3 или .wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    audio_path = "input_track.mp3"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("Трек успешно загружен в студию!")
    
    # Настройки стиля и релиза
    st.subheader("🎨 Настройки релиза и обложки")
    
    genre_category = st.selectbox(
        "🎵 Музыкальный стиль",
        [
            "SLOWED & REVERB (Глубокий атмосферный вайб)",
            "PHONK / DRIFT PHONK (Качающий бас и темный звук)",
            "LO-FI / CHILL BEATS (Мягкий винтажный звук)",
            "NIGHTCORE / HIGH SPEED (Энергичный ускоренный вайб)",
            "SYNTHWAVE / RETRO 80s (Космическая атмосфера)"
        ]
    )
    
    song_title = st.text_input("Название трека", "I WELCOME OCTOBER")
    artist_name = st.text_input("Имя автора / Артиста", "Smagulov & Zhaken")
    
    default_genre_text = genre_category.split(" (")[0]
    genre_text = st.text_input("Текст жанра на обложке", default_genre_text)

    # --- ПАНЕЛЬ ТОЧНОЙ НАСТРОЙКИ ЭФФЕКТОВ (FX RACK) ---
    st.subheader("🎛️ Мастер эффектов и обработки звука (FX Rack)")
    
    col_fx1, col_fx2 = st.columns(2)
    with col_fx1:
        pitch_shift = st.slider("🔑 Сдвиг тональности (полутоны)", -5.0, 3.0, -1.5, 0.5)
        tempo_factor = st.slider("⏱️ Скорость (Темп)", 0.70, 1.20, 0.85, 0.01)
        bass_boost_gain = st.slider("🔊 Усиление баса (Bass Boost)", 0.0, 2.0, 0.6, 0.1)
    with col_fx2:
        reverb_mix = st.slider("🌊 Интенсивность эха / Реверберации", 0.0, 1.0, 0.45, 0.05)
        reverb_delay_ms = st.slider("⏳ Задержка эха (Delay, мс)", 50, 300, 120, 10)
        vinyl_noise = st.slider("📻 Плотность шума винила", 0.0, 0.02, 0.004, 0.001)

    # Выбор дополнительного фонового бита
    selected_beat_layer = st.selectbox(
        "🎶 Дополнительный фоновый ритм / бит",
        [
            "Нет",
            "Drift Phonk Drum Loop (Качающие ударные)",
            "Lo-Fi Rain & Vinyl Atmosphere (Дождь и винил)",
            "Cyberpunk Bass Pulse (Плотный басовый пульс)",
            "Chill Ambient Pad (Космический синтезатор)"
        ]
    )

    if st.button("🚀 Обработать трек и создать обложку"):
        
        # --- ОБРАБОТКА АУДИО С УЧЕТОМ ТОЧНЫХ НАСТРОЕК ---
        with st.spinner("🔄 Применяем студийные эффекты и микшируем аудио..."):
            try:
                y, sr = librosa.load(audio_path, sr=None, res_type='kaiser_best')
                
                # Тон и темп
                if pitch_shift != 0.0:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift, n_fft=2048, hop_length=512)
                if tempo_factor != 1.0:
                    y = librosa.effects.time_stretch(y, rate=tempo_factor)
                
                # Точный Bass Boost
                if bass_boost_gain > 0:
                    y_bass = librosa.effects.pitch_shift(y, sr=sr, n_steps=-12)
                    y = y + (y_bass * bass_boost_gain)

                # Точное Эхо / Реверберация (Reverb & Delay)
                if reverb_mix > 0:
                    delay_samples = int(sr * (reverb_delay_ms / 1000.0))
                    reverb_signal = np.zeros_like(y)
                    if len(y) > delay_samples:
                        reverb_signal[delay_samples:] = y[:-delay_samples] * reverb_mix
                        y = y + reverb_signal

                # Шум винила
                if vinyl_noise > 0:
                    noise = np.random.normal(0, vinyl_noise, len(y))
                    y = y + noise

                # Добавление фонового бита / атмосферы
                if selected_beat_layer != "Нет":
                    t = np.linspace(0, len(y)/sr, len(y))
                    if "Phonk" in selected_beat_layer:
                        beat_pulse = np.sin(2 * np.pi * 2.2 * t) * 0.18
                        y = y + (beat_pulse * y)
                    elif "Rain" in selected_beat_layer:
                        rain_noise = np.random.normal(0, 0.01, len(y))
                        y = y + rain_noise
                    elif "Cyberpunk" in selected_beat_layer:
                        synth_pulse = np.sin(2 * np.pi * 1.5 * t) * 0.15
                        y = y + synth_pulse
                    elif "Ambient" in selected_beat_layer:
                        pad_pulse = np.sin(2 * np.pi * 0.8 * t) * 0.12
                        y = y + pad_pulse

                # Нормализация громкости
                max_val = max(abs(y.min()), abs(y.max()))
                if max_val > 0:
                    y = y / max_val * 0.95
                
                safe_audio_path = "remixed_track.wav"
                sf.write(safe_audio_path, y, sr, subtype='PCM_16')
            except Exception as e:
                st.error(f"Ошибка при обработке аудио: {e}")
                st.stop()
        
        # --- ГЕНЕРАЦИЯ ОБЛОЖКИ ---
        with st.spinner("🎨 Создаем премиальную обложку NBS SOFT..."):
            bg_image = None
            try:
                random_photo_id = random.choice([1047, 1058, 1078, 1062, 1039])
                image_url = f"https://picsum.photos/id/{random_photo_id}/1280/720"
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    bg_image = Image.open(BytesIO(response.content)).convert("RGB")
                    bg_image = bg_image.filter(ImageFilter.GaussianBlur(radius=3))
            except:
                bg_image = None

            if bg_image is None:
                bg_image = Image.new("RGB", (1280, 720), (20, 20, 30))

            darken = Image.new("RGBA", bg_image.size, (0, 0, 0, 185))
            cover_image = Image.alpha_composite(bg_image.convert("RGBA"), darken)
            draw = ImageDraw.Draw(cover_image)

            font_path = "C:\\Windows\\Fonts\\arialbd.ttf"
            title_font = ImageFont.truetype(font_path, 85) if os.path.exists(font_path) else ImageFont.load_default()
            sub_font = ImageFont.truetype(font_path, 42) if os.path.exists(font_path) else ImageFont.load_default()

            for ax in range(-4, 5):
                for ay in range(-4, 5):
                    draw.text((90 + ax, 180 + ay), song_title.upper(), font=title_font, fill=(0,0,0,255))
            
            draw.text((90, 180), song_title.upper(), font=title_font, fill=(255,255,255,255))
            draw.text((90, 300), f"⚡ {genre_text.upper()} | NBS SOFT", font=sub_font, fill=(255,215,0,255))

            cover_filename = "youtube_cover.png"
            cover_image.convert("RGB").save(cover_filename, "PNG")

        st.success("🎉 Готово! Ваш трек настроен и сгенерирован.")

        # --- РЕЗУЛЬТАТЫ СКАЧИВАНИЯ ---
        st.subheader("📥 Скачать результаты:")
        st.image(cover_filename, caption="Обложка релиза (1280x720)", use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            with open(cover_filename, "rb") as img_file:
                st.download_button("📥 Скачать обложку (PNG)", img_file, file_name="youtube_cover.png", mime="image/png")
        with col2:
            if os.path.exists(safe_audio_path):
                with open(safe_audio_path, "rb") as aud_file:
                    st.download_button("📥 Скачать аудио (WAV)", aud_file, file_name="remixed_track.wav", mime="audio/wav")
