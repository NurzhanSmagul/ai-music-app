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
st.set_page_config(page_title="NBS SOFT — AI Music & BPM Sync Studio", page_icon="👑", layout="centered")

# Сессия для авторизации
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# --- ЭКРАН АВТОРИЗАЦИИ / РЕГИСТРАЦИИ ---
if not st.session_state["logged_in"]:
    st.title("🔐 NBS SOFT — Вход в платформу")
    st.write("Войдите в систему для доступа к студии с синхронизацией BPM и SEO.")
    
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
st.title("👑 NBS SOFT — BPM Sync & Anti-ID Studio")
st.write("Автоматическое определение BPM, синхронизация ритмов, HQ обработка и генерация SEO!")

# Загрузка трека
uploaded_file = st.file_uploader("Загрузите исходный трек (.mp3 или .wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    audio_path = "input_track.mp3"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("✅ Трек успешно загружен в студию!")
    
    # Настройки релиза и обложки
    st.subheader("🎨 Настройки релиза и обложки")
    
    genre_category = st.selectbox(
        "🎵 Музыкальный стиль / Направление",
        [
            "SLOWED & REVERB (Глубокий атмосферный вайб + Анти-ID)",
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

    # --- ПАНЕЛЬ НАСТРОЙКИ КАЧЕСТВА И ЭФФЕКТОВ ---
    st.subheader("🎛️ Студийный FX Rack & BPM Синхронизация")
    
    col_fx1, col_fx2 = st.columns(2)
    with col_fx1:
        pitch_shift = st.slider("🔑 Сдвиг тональности (для обхода ID)", -4.0, 2.0, -1.5, 0.5)
        tempo_factor = st.slider("⏱️️ Скорость (Темп)", 0.75, 1.10, 0.88, 0.01)
        bass_boost_gain = st.slider("🔊 Усиление баса (Bass Boost)", 0.0, 2.0, 0.7, 0.1)
    with col_fx2:
        reverb_mix = st.slider("🌊 Интенсивность эха / Реверберации", 0.0, 1.0, 0.40, 0.05)
        reverb_delay_ms = st.slider("⏳ Задержка эха (Delay, мс)", 60, 250, 120, 10)
        vinyl_noise = st.slider("📻 Плотность шума винила / Анти-ID фактора", 0.0, 0.015, 0.003, 0.001)

    selected_beat_layer = st.selectbox(
        "🎶 Дополнительный синхронизированный бит / вайб",
        [
            "Нет (Чистая обработка исходника)",
            "Drift Phonk Beat (Строго по BPM трека)",
            "Lo-Fi Vinyl & Rain Rhythm (Мягкий грув)",
            "Cyberpunk Bass Pulse (Ритмичный пульс под BPM)"
        ]
    )

    if st.button("🚀 Запустить BPM-синхронизацию, HQ звук и SEO"):
        
        # --- ОБРАБОТКА АУДИО И СИНХРОНИЗАЦИЯ BPM ---
        with st.spinner("🔄 Анализируем BPM трека, синхронизируем ритм и обрабатываем звук..."):
            try:
                y, sr = librosa.load(audio_path, sr=None, mono=True)
                
                # 1. Автоматический расчет BPM исходного трека
                tempo_detected, _ = librosa.beat.beat_track(y=y, sr=sr)
                if isinstance(tempo_detected, np.ndarray):
                    track_bpm = float(tempo_detected[0])
                else:
                    track_bpm = float(tempo_detected)
                
                # Применяем сдвиг питча и темпа
                if pitch_shift != 0.0:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift, n_fft=2048, hop_length=512)
                
                if tempo_factor != 1.0:
                    y = librosa.effects.time_stretch(y, rate=tempo_factor)
                
                # Скорректированный BPM с учетом изменения скорости
                effective_bpm = track_bpm * tempo_factor

                # 2. Бас-буст
                if bass_boost_gain > 0:
                    y_bass = librosa.effects.pitch_shift(y, sr=sr, n_steps=-12)
                    y = y + (y_bass * bass_boost_gain * 0.4)

                # 3. Эхо и реверберация
                if reverb_mix > 0:
                    delay_samples = int(sr * (reverb_delay_ms / 1000.0))
                    reverb_signal = np.zeros_like(y)
                    if len(y) > delay_samples:
                        reverb_signal[delay_samples:] = y[:-delay_samples] * reverb_mix
                        y = y + reverb_signal

                # 4. Шум винила
                if vinyl_noise > 0:
                    noise = np.random.normal(0, vinyl_noise, len(y))
                    y = y + noise

                # 5. СТРОГО СИНХРОНИЗИРОВАННЫЙ ПО BPM ФОНОВЫЙ РИТМ
                if selected_beat_layer != "Нет":
                    t = np.linspace(0, len(y)/sr, len(y))
                    # Точная частота ударов в секунду (Hz) на основе измеренного BPM
                    beat_freq = effective_bpm / 60.0
                    
                    if "Phonk" in selected_beat_layer:
                        # Качающий саб-бас и акценты строго по долям BPM
                        bpm_pulse = np.abs(np.sin(2 * np.pi * beat_freq * t)) ** 3 * 0.22
                        y = y + (bpm_pulse * y)
                    elif "Rain" in selected_beat_layer:
                        rain_noise = np.random.normal(0, 0.008, len(y))
                        rhythmic_mod = (0.8 + 0.2 * np.sin(2 * np.pi * beat_freq * t))
                        y = y + (rain_noise * rhythmic_mod)
                    elif "Cyberpunk" in selected_beat_layer:
                        synth_pulse = np.sin(2 * np.pi * (beat_freq * 2) * t) * 0.14
                        y = y + synth_pulse

                # Нормализация громкости (HQ Limiter)
                max_val = max(abs(y.min()), abs(y.max()))
                if max_val > 0:
                    y = y / max_val * 0.95
                
                safe_audio_path = "remixed_track.wav"
                sf.write(safe_audio_path, y, sr, subtype='PCM_16')
                st.session_state["ready_audio"] = safe_audio_path
                st.session_state["detected_bpm"] = round(effective_bpm, 1)

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
            st.session_state["ready_cover"] = cover_filename

        st.success(f"🎉 Готово! Трек обработан. Определенный BPM: **{st.session_state.get('detected_bpm', 120)}**")

    # --- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ---
    if "ready_cover" in st.session_state and os.path.exists(st.session_state["ready_cover"]):
        st.markdown("---")
        st.subheader("📥 Готовые материалы релиза:")
        st.image(st.session_state["ready_cover"], caption="Обложка релиза (1280x720)", use_container_width=True)
        
        with open(st.session_state["ready_cover"], "rb") as img_file:
            st.download_button("📥 Скачать обложку (PNG)", img_file, file_name="youtube_cover.png", mime="image/png")

    if "ready_audio" in st.session_state and os.path.exists(st.session_state["ready_audio"]):
        st.markdown(f"🎵 **Прослушать обработанный трек (BPM: {st.session_state.get('detected_bpm', 'Auto')}):**")
        st.audio(st.session_state["ready_audio"], format="audio/wav")
        
        with open(st.session_state["ready_audio"], "rb") as aud_file:
            st.download_button("📥 Скачать трек в HQ (WAV)", aud_file, file_name="remixed_track.wav", mime="audio/wav")

        # --- YOUTUBE SEO ---
        st.markdown("---")
        st.subheader("🚀 YouTube SEO Оптимизация (Названия, Описание, Теги)")
        
        clean_title_slug = song_title.replace(" ", "")
        genre_slug = default_genre_text.split(" ")[0]

        yt_titles = [
            f"{song_title} [{default_genre_text} / HQ Audio] | NBS SOFT",
            f"{artist_name} - {song_title} (Slowed & Reverb / Bass Boosted)",
            f"{song_title} — {default_genre_text} (Vibe Edition)"
        ]
        
        st.markdown("**💡 1. Оптимизированные названия:**")
        for t in yt_titles:
            st.code(t, language="text")

        yt_description = f"""🎵 Artist: {artist_name}
🎧 Track: {song_title}
✨ Version: {default_genre_text}
⚡ Powered by NBS SOFT Studio

Immerse yourself in the ultimate atmosphere. Enjoy the vibe, drop a like, and subscribe for more daily releases!

📌 Support & Links:
• Telegram Channel: Link in Bio
• Stream / Download: Available on all platforms

#️⃣ Tags & Hashtags:
#{clean_title_slug} #{genre_slug} #SlowedAndReverb #Phonk #BassBoosted #MusicVibes #Audio #NBSSoft #TrendingMusic
"""
        st.markdown("**📝 2. Полное SEO-описание для видео:**")
        st.text_area("Скопируйте описание:", yt_description, height=160)

        youtube_tags = f"{song_title}, {artist_name}, {song_title} slowed, {song_title} reverb, {default_genre_text}, phonk, bass boosted, chill music, aesthetic music, slowed and reverb songs, nbs soft, audio edit, remix, tiktok music, youtube shorts music, 8d audio, nightcore"
        
        st.markdown("**🏷️ 3. Теги для YouTube Studio:**")
        st.code(youtube_tags, language="text")
