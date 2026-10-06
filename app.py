import streamlit as st
import os
import random
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import librosa
import soundfile as sf
import numpy as np
import sqlite3
import hashlib
import pandas as pd

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
st.set_page_config(page_title="NBS SOFT — Pro Music Studio & Timeline Editor", page_icon="👑", layout="centered")

# Сессия для авторизации
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# --- ЭКРАН АВТОРИЗАЦИИ / РЕГИСТРАЦИИ ---
if not st.session_state["logged_in"]:
    st.title("🔐 NBS SOFT — Вход в платформу")
    st.write("Войдите в систему для доступа к студии ремиксов и таймлайн-редактору.")
    
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
st.title("👑 NBS SOFT — Music Studio & Timeline Beats")
st.write("Профессиональная студия обработки звука, таймлайн-редактор ударных и генератор YouTube SEO!")

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

    # --- ПАНЕЛЬ НАСТРОЙКИ КАЧЕСТВА И УДАРНЫХ ---
    st.subheader("🎛️ Студийный FX Rack & Настройка дорожек")
    
    col_fx1, col_fx2 = st.columns(2)
    with col_fx1:
        pitch_shift = st.slider("🔑 Сдвиг тональности (полутоны)", -5.0, 3.0, -2.0, 0.5)
        tempo_factor = st.slider("⏱ Скорость (Темп)", 0.70, 1.20, 0.82, 0.01)
        trim_start_sec = st.slider("✂️ Обрезать начало трека (сек) [Удаление артефактов старта]", 0.0, 1.0, 0.15, 0.05)
    with col_fx2:
        reverb_mix = st.slider("🌊 Интенсивность эха / Реверберации", 0.0, 1.0, 0.40, 0.05)
        reverb_delay_ms = st.slider("⏳ Задержка эха (Delay, мс)", 80, 300, 150, 10)
        vinyl_noise = st.slider("📻 Плотность винилового шума", 0.0, 0.015, 0.002, 0.001)

    st.markdown("---")
    st.subheader("🥁 Настройка слоев ударных и баса (Timeline Beats)")
    
    col_beat1, col_beat2, col_beat3 = st.columns(3)
    with col_beat1:
        kick_volume = st.slider("🔊 Громкость Kick (Бочка)", 0.0, 2.0, 1.0, 0.1)
    with col_beat2:
        snare_volume = st.slider("🔊 Громкость Snare (Снейр)", 0.0, 2.0, 0.8, 0.1)
    with col_beat3:
        bass_volume = st.slider("🔊 Громкость 808 Bass (Саб)", 0.0, 2.0, 1.2, 0.1)

    selected_beat_layer = st.selectbox(
        "🎶 Ритмический паттерн ударных",
        [
            "Drift Phonk Heavy Kick & Snare (Мощный кач)",
            "Lo-Fi Chill Percussion (Мягкий ритм)",
            "Cyberpunk Electronic Beat (Плотный электронный бит)",
            "Нет (Чистая обработка без добавления ударных)"
        ]
    )

    if st.button("🚀 Обработать трек и применить мастеринг"):
        
        # --- ОБРАБОТКА АУДИО И СБОРКА ТАЙМЛАЙНА ---
        with st.spinner("🔄 Чистка спектра, сведение дорожек и мастеринг..."):
            try:
                # Загружаем трек
                y, sr = librosa.load(audio_path, sr=None, mono=True)
                
                # Анализ BPM
                tempo_detected, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                if isinstance(tempo_detected, np.ndarray):
                    track_bpm = float(tempo_detected[0])
                else:
                    track_bpm = float(tempo_detected)
                
                # Защита от артефактов через паддинг
                pad_len = int(sr * 1.0)
                y_padded = np.pad(y, (pad_len, pad_len), mode='constant')

                if pitch_shift != 0.0:
                    y_padded = librosa.effects.pitch_shift(y_padded, sr=sr, n_steps=pitch_shift, n_fft=2048, hop_length=512)
                
                if tempo_factor != 1.0:
                    y_padded = librosa.effects.time_stretch(y_padded, rate=tempo_factor)
                
                effective_pad = int(pad_len * (1.0 / tempo_factor))
                if len(y_padded) > 2 * effective_pad:
                    y = y_padded[effective_pad : -effective_pad]
                else:
                    y = y_padded

                effective_bpm = track_bpm * tempo_factor

                # Отрезаем указанное пользователем время от начала для полного удаления «тыыыз»
                trim_samples = int(trim_start_sec * sr)
                if len(y) > trim_samples:
                    y = y[trim_samples:]

                # Абсолютно плавный Fade-In на старте (50мс)
                fade_samples = int(sr * 0.05)
                if len(y) > fade_samples:
                    y[:fade_samples] = y[:fade_samples] * np.linspace(0.0, 1.0, fade_samples)

                y = y / (np.max(np.abs(y)) + 1e-6) * 0.7

                # --- ДОБАВЛЕНИЕ УДАРНЫХ И БАСА ПО ТАЙМЛАЙНУ ---
                timeline_events = []
                if selected_beat_layer != "Нет":
                    mixed = y.copy()
                    
                    # Генерация чистой бочки (Kick) без кликов
                    kick_dur = 0.12
                    t_kick = np.linspace(0, kick_dur, int(sr * kick_dur))
                    freq_sweep = np.linspace(140, 45, len(t_kick))
                    kick_wave = np.sin(2 * np.pi * freq_sweep * t_kick) * np.exp(-8 * t_kick) * 0.5 * kick_volume

                    # Генерация снейра (Snare)
                    snare_dur = 0.08
                    t_snare = np.linspace(0, snare_dur, int(sr * snare_dur))
                    snare_wave = (np.random.normal(0, 1, len(t_snare)) * np.exp(-18 * t_snare) + 
                                  np.sin(2 * np.pi * 220 * t_snare) * np.exp(-12 * t_snare)) * 0.3 * snare_volume

                    adjusted_frames = (beat_frames / tempo_factor).astype(int) - trim_samples

                    for i, frame in enumerate(adjusted_frames):
                        idx = int(frame)
                        if idx < int(sr * 0.1):
                            continue
                            
                        timestamp_sec = round(max(0, idx) / sr, 2)
                        
                        # Внедрение бочки
                        if idx + len(kick_wave) < len(mixed):
                            mixed[idx:idx + len(kick_wave)] += kick_wave
                            timeline_events.append({"Время (сек)": timestamp_sec, "Тип дорожки": "Kick (Бочка)", "Событие": f"Удар #{i+1}"})
                        
                        # Внедрение снейра на каждый нечетный такт
                        if i % 2 == 1:
                            if idx + len(snare_wave) < len(mixed):
                                mixed[idx:idx + len(snare_wave)] += snare_wave
                                timeline_events.append({"Время (сек)": timestamp_sec, "Тип дорожки": "Snare (Снейр)", "Событие": f"Акцент #{i+1}"})

                    y = mixed
                    st.session_state["timeline_events"] = timeline_events
                else:
                    st.session_state["timeline_events"] = []

                # --- 808 SUB-BASS ---
                if bass_volume > 0:
                    y_bass = librosa.effects.pitch_shift(y, sr=sr, n_steps=-12)
                    y_bass = y_bass / (np.max(np.abs(y_bass)) + 1e-6)
                    y = y + (y_bass * bass_volume * 0.3)

                # --- ЭХО И РЕВЕРБЕРАЦИЯ ---
                if reverb_mix > 0:
                    delay_samples = int(sr * (reverb_delay_ms / 1000.0))
                    reverb_signal = np.zeros_like(y)
                    if len(y) > delay_samples:
                        reverb_signal[delay_samples:] = y[:-delay_samples] * reverb_mix
                        y = y + reverb_signal

                # --- ВИНИЛОВЫЙ ШУМ ---
                if vinyl_noise > 0:
                    y = y + np.random.normal(0, vinyl_noise, len(y))

                # --- МАСТЕРИНГ И ЛИМИТЕР ---
                y = np.tanh(y * 1.15) / 1.15
                max_val = np.max(np.abs(y))
                if max_val > 0:
                    y = y / max_val * 0.92
                
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

        st.success(f"🎉 Готово! Новый BPM: **{st.session_state.get('detected_bpm', 120)}**. Артефакты отрезаны, дорожки сведены!")

    # --- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ И ТАЙМЛАЙНА ---
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

        # --- ТАЙМЛАЙН РЕДАКТОР ---
        if "timeline_events" in st.session_state and st.session_state["timeline_events"]:
            st.markdown("---")
            st.subheader("🎚️ Таймлайн-редактор: Дорожки ударных и баса")
            st.write("Синхронизированная таблица точек внедрения сэмплов по сетке трека:")
            
            df_timeline = pd.DataFrame(st.session_state["timeline_events"])
            st.dataframe(df_timeline, use_container_width=True)

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
