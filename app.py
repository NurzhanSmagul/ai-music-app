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
st.set_page_config(page_title="NBS SOFT — AI Music & Viral Studio", page_icon="👑", layout="centered")

# Сессия для авторизации
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# --- ЭКРАН АВТОРИЗАЦИИ / РЕГИСТРАЦИИ ---
if not st.session_state["logged_in"]:
    st.title("🔐 NBS SOFT — Вход в платформу")
    st.write("Войдите в систему, чтобы получить доступ к генератору треков, студии ремиксов и автогенератору вирусных Shorts/TikTok роликов.")
    
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
st.sidebar.image("logo_cropped.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.write(f"👤 Аккаунт: **{st.session_state['username']}**")
if st.sidebar.button("🚪 Выйти из аккаунта"):
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.rerun()

# --- ОСНОВНОЙ ФУНКЦИОНАЛ ПРИЛОЖЕНИЯ ---
st.title("👑 NBS SOFT — AI Music & Viral Studio")
st.write("Профессиональная студия ремиксов, защиты от Content ID и автогенерации контента для YouTube и TikTok!")

# Загрузка трека
uploaded_file = st.file_uploader("Загрузите исходный трек (.mp3 или .wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    audio_path = "input_track.mp3"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("Файл успешно загружен!")
    
    # Режим работы
    mode = st.radio(
        "🎯 Выберите режим работы:",
        ["🎚️ Профессиональный Ремикс & Защита", "🎬 Генератор вирусных Shorts / TikTok сценариев"]
    )
    
    # Настройки стиля и обложки
    st.subheader("🎨 Настройки релиза и визуального контента")
    
    genre_category = st.selectbox(
        "🎵 Музыкальная ниша / Стиль",
        [
            "LO-FI / CHILL MUSIC (Мягкий расслабленный вайб)",
            "SLOWED & REVERB (Глубокий атмосферный замедленный вайб)",
            "PHONK / DRIFT PHONK (Глубокий бас, темный кач)",
            "NIGHTCORE / FAST VIBE (Энергичный высокий темп)",
            "DEEP HOUSE / CLUB (Клубный ритм)",
            "SYNTHWAVE / RETRO 80s (Космический ретро-вайб)"
        ]
    )
    
    song_title = st.text_input("Название трека", "I WELCOME OCTOBER")
    artist_name = st.text_input("Имя автора / Артиста", "Smagulov & Zhaken")
    
    default_genre_text = genre_category.split(" (")[0]
    genre_text = st.text_input("Текст жанра на обложке", default_genre_text)

    # Аудио параметры
    pitch_shift, tempo_factor, add_reverb, add_vinyl = -1.0, 0.90, True, True

    if "Ремикс" in mode or "SLOWED" in genre_category:
        with st.expander("🎛️ Настройки студийных эффектов ремикса"):
            pitch_shift = st.slider("Сдвиг тональности", -4.0, 2.0, -2.0, 0.5)
            tempo_factor = st.slider("Скорость (Темп)", 0.75, 0.95, 0.82, 0.01)
            add_reverb = st.checkbox("Эффект эха (Reverb)", value=True)
            add_vinyl = st.checkbox("Шум винила (Vinyl Crackle)", value=True)

    if st.button("🚀 Запустить генерацию контента и SEO"):
        
        # --- ОБРАБОТКА АУДИО ---
        with st.spinner("🔄 Обрабатываем аудио и защищаем от Content ID..."):
            try:
                y, sr = librosa.load(audio_path, sr=None, res_type='kaiser_best')
                if pitch_shift != 0.0:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift, n_fft=2048, hop_length=512)
                if tempo_factor != 1.0:
                    y = librosa.effects.time_stretch(y, rate=tempo_factor)
                
                if add_reverb:
                    delay_samples = int(sr * 0.08)
                    reverb_signal = np.zeros_like(y)
                    if len(y) > delay_samples:
                        reverb_signal[delay_samples:] = y[:-delay_samples] * 0.35
                        y = y + reverb_signal

                if add_vinyl:
                    noise = np.random.normal(0, 0.004, len(y))
                    y = y + noise

                max_val = max(abs(y.min()), abs(y.max()))
                if max_val > 0:
                    y = y / max_val * 0.95
                
                safe_audio_path = "remixed_track.wav"
                sf.write(safe_audio_path, y, sr, subtype='PCM_16')
            except Exception as e:
                st.error(f"Ошибка при обработке аудио: {e}")
                st.stop()
        
        # --- ГЕНЕРАЦИЯ ОБЛОЖКИ (КРУПНЫЙ И ЧЕТКИЙ ТЕКСТ) ---
        with st.spinner("🎨 Создаем крупную обложку 1280x720..."):
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
            # Увеличенные размеры шрифтов для отличной читаемости
            title_font = ImageFont.truetype(font_path, 85) if os.path.exists(font_path) else ImageFont.load_default()
            sub_font = ImageFont.truetype(font_path, 45) if os.path.exists(font_path) else ImageFont.load_default()

            # Обводка для контраста
            for ax in range(-4, 5):
                for ay in range(-4, 5):
                    draw.text((90 + ax, 180 + ay), song_title.upper(), font=title_font, fill=(0,0,0,255))
            
            draw.text((90, 180), song_title.upper(), font=title_font, fill=(255,255,255,255))
            draw.text((90, 300), f"⚡ {genre_text.upper()} | NBS SOFT", font=sub_font, fill=(255,215,0,255))

            cover_filename = "youtube_cover.png"
            cover_image.convert("RGB").save(cover_filename, "PNG")

        st.success("🎉 Готово! Обложка и аудио успешно сгенерированы.")

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

        if os.path.exists(safe_audio_path):
            st.audio(safe_audio_path, format="audio/wav")

        # --- YOUTUBE SEO & ТЕГИ ---
        st.markdown("---")
        st.subheader("🚀 Англоязычное YouTube SEO и Теги")
        
        yt_titles = [
            f"{song_title} [{default_genre_text} Vibe / Audio] | NBS SOFT",
            f"{artist_name} - {song_title} (Slowed & Reverb / HQ Audio)",
            f"{song_title} — {default_genre_text} Remix"
        ]
        
        yt_description = f"""🎵 {artist_name} — {song_title}
🎧 Version: {default_genre_text}
⚡ Powered by NBS SOFT Studio

Immerse yourself in the ultimate atmosphere. Enjoy the vibe, drop a like, and subscribe for more daily releases!

📌 Follow & Support:
• Telegram Channel: Your Link Here
• Spotify / Apple Music: Your Link Here

#️⃣ Hashtags:
#{song_title.replace(' ', '')} #{default_genre_text.replace(' ', '')} #SlowedAndReverb #Vibes #Music #Audio #Remix #NBSSoft
"""
        st.markdown("**💡 Оптимизированные названия для YouTube:**")
        for t in yt_titles:
            st.code(t, language="text")

        st.markdown("**📝 SEO Описание для видео:**")
        st.text_area("Скопируйте описание:", yt_description, height=150)

        # --- СЦЕНАРИЙ ДЛЯ SHORTS / TIKTOK ---
        st.markdown("---")
        st.subheader("🎬 Авто-сценарий для вирусных Shorts / TikTok")
        hook_ideas = [
            f"«Когда включил этот {default_genre_text} трек и забыл обо всем...»",
            f"«Этот бит в стиле {default_genre_text} сносит крышу с первых секунд 🎧»",
            f"«Идеальный трек для ночных поездок и вайба: {song_title}»"
        ]
        selected_hook = random.choice(hook_ideas)
        st.markdown(f"**⚡ Хук (первые 3 секунды видео):**")
        st.code(selected_hook, language="text")
