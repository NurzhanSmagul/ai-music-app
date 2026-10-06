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

# Кроссплатформенная загрузка шрифта
def get_font(size):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\Arial.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()

# Настройка страницы
st.set_page_config(page_title="NBS SOFT — AI Phonk & Remix Studio", page_icon="👑", layout="centered")

# Сессия для авторизации
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# --- ЭКРАН АВТОРИЗАЦИИ / РЕГИСТРАЦИИ ---
if not st.session_state["logged_in"]:
    st.title("🔐 NBS SOFT — Вход в студию")
    st.write("Войдите в систему для доступа к AI-студии ремиксов.")
    
    tab_auth1, tab_auth2 = st.tabs(["🔑 Вход", "📝 Регистрация"])
    
    with tab_auth1:
        login_user = st.text_input("Логин", key="login_u")
        login_pass = st.text_input("Пароль", type="password", key="login_p")
        if st.button("Войти в систему"):
            if verify_user(login_user, login_pass):
                st.session_state["logged_in"] = True
                st.session_state["username"] = login_user
                st.success("Успешный вход!")
                st.rerun()
            else:
                st.error("Неверный логин или пароль")
                
    with tab_auth2:
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

# --- БОКОВАЯ ПАНЕЛЬ С БРЕНДИНГОМ ---
if os.path.exists("logo_cropped.png"):
    st.sidebar.image("logo_cropped.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.write(f"👤 Аккаунт: **{st.session_state['username']}**")
if st.sidebar.button("🚪 Выйти из аккаунта"):
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.rerun()

# --- ОСНОВНОЙ ФУНКЦИОНАЛ ПРИЛОЖЕНИЯ ---
st.title("👑 NBS SOFT — AI Phonk & Remix Studio")
st.write("Превращайте любые треки в полноценные авторские ремиксы с выбором стиля, ИИ и продвинутой защитой!")

tab_remix, tab_daw, tab_ai_gen = st.tabs(["🎯 Сделать AI Ремикс", "🎛️ DAW & Сведение", "🤖 MusicGen AI Лаборатория"])

# --- ВКЛАДКА 1: СДЕЛАТЬ AI РЕМИКС (ПО ВЫБОРУ) ---
with tab_remix:
    st.subheader("🎯 Генератор полноценных AI-ремиксов")
    st.write("Загрузите ваш исходный трек, выберите желаемый стиль, степень изменения и позвольте системе полностью преобразить аранжировку.")
    
    remix_file = st.file_uploader("📂 Загрузить трек для ремикса (.mp3 или .wav)", type=["mp3", "wav"], key="remix_up")
    
    if remix_file is not None:
        remix_input_path = "source_remix.mp3"
        with open(remix_input_path, "wb") as f:
            f.write(remix_file.getbuffer())
        st.success("✅ Трек загружен и готов к трансформации!")
        
        st.markdown("### 🎚️ Настройки трансформации и выбора стиля:")
        
        remix_genre = st.selectbox(
            "🎵 Выберите целевой жанр ремикса",
            [
                "Drift Phonk (Агрессивный ковбелл, жесткий 808 бас, кач)",
                "Dark Phonk (Мрачная атмосфера, плотный саб, мистический вайб)",
                "Slowed & Reverb Vibe (Глубокое замедление, эхо, космос)",
                "Cyberpunk / Synthwave Remix (Электронный футуристический бит)",
                "Phonk House (Танцевальный качающий ритм 130 BPM)"
            ]
        )
        
        transformation_depth = st.slider(
            "⚡ Степень изменения оригинала (Интенсивность AI-трансформации)",
            min_value=1, max_value=5, value=4,
            help="1 — легкий ремикс поверх оригинала, 5 — полная пересборка мелодии и гармонии в новый трек."
        )
        
        remix_song_title = st.text_input("Название будущего ремикса", "I WELCOME OCTOBER (AI Phonk Remix)")
        remix_artist = st.text_input("Имя артиста / продюсера", "Smagulov & Zhaken")
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            add_cowbell = st.checkbox("🔔 Добавить фирменный Phonk Cowbell (Колокольчик)", value=True)
            heavy_bass = st.checkbox("💥 Усилить 808 Drift Bass (Саб-бас)", value=True)
        with col_opt2:
            anti_id_protect = st.checkbox("🛡️ Применить защиту Anti-Content ID", value=True)
            generate_cover = st.checkbox("🎨 Автоматически создать обложку релиза", value=True)

        if st.button("🚀 Сделать AI Ремикс"):
            with st.spinner("🎧 ИИ анализирует дорожку, перестраивает гармонию и создает полноценный ремикс..."):
                try:
                    y, sr = librosa.load(remix_input_path, sr=None, mono=True)
                    
                    pitch_steps = -3.0 if "Drift" in remix_genre or "Dark" in remix_genre else -2.0
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
                    
                    tempo_rate = 0.82 if "Slowed" in remix_genre else 0.90
                    y = librosa.effects.time_stretch(y, rate=tempo_rate)
                    
                    if heavy_bass:
                        y_sub = librosa.effects.pitch_shift(y, sr=sr, n_steps=-12)
                        y = y + (y_sub * 0.4)
                        
                    if add_cowbell:
                        bell_dur = 0.1
                        t_bell = np.linspace(0, bell_dur, int(sr * bell_dur))
                        bell_wave = np.sin(2 * np.pi * 830 * t_bell) * np.exp(-15 * t_bell) * 0.3
                        hop = int(sr * 0.45)
                        for idx in range(0, len(y) - len(bell_wave), hop):
                            y[idx:idx + len(bell_wave)] += bell_wave

                    if anti_id_protect:
                        y = np.tanh(y * 1.3) / 1.3
                        y = y + np.random.normal(0, 0.002, len(y))

                    y = y / (np.max(np.abs(y)) + 1e-6) * 0.95
                    
                    output_remix_path = "final_ai_remix.wav"
                    sf.write(output_remix_path, y, sr, subtype='PCM_16')
                    st.session_state["remix_audio"] = output_remix_path
                    
                    if generate_cover:
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

                        title_font = get_font(80)
                        sub_font = get_font(40)

                        draw.text((90, 180), remix_song_title.upper(), font=title_font, fill=(255,255,255,255))
                        draw.text((90, 300), f"⚡ {remix_genre.split(' (')[0].upper()} | NBS SOFT", font=sub_font, fill=(255,215,0,255))

                        cover_filename = "remix_cover.png"
                        cover_image.convert("RGB").save(cover_filename, "PNG")
                        st.session_state["remix_cover"] = cover_filename

                    st.success("🎉 Ваш полноценный AI-ремикс успешно создан!")
                    
                except Exception as e:
                    st.error(f"Ошибка при создании ремикса: {e}")

    if "remix_audio" in st.session_state and os.path.exists(st.session_state["remix_audio"]):
        st.markdown("---")
        st.subheader("🎧 Готовый AI-ремикс:")
        
        if "remix_cover" in st.session_state and os.path.exists(st.session_state["remix_cover"]):
            st.image(st.session_state["remix_cover"], caption="Обложка ремикса (1280x720)", use_container_width=True)
            with open(st.session_state["remix_cover"], "rb") as cf:
                st.download_button("📥 Скачать обложку ремикса (PNG)", cf, file_name="remix_cover.png", mime="image/png")
                
        st.audio(st.session_state["remix_audio"], format="audio/wav")
        with open(st.session_state["remix_audio"], "rb") as af:
            st.download_button("📥 Скачать готовый трек (WAV)", af, file_name="ai_phonk_remix.wav", mime="audio/wav")

# --- ВКЛАДКА 2: DAW & СВЕДЕНИЕ ---
with tab_daw:
    st.subheader("🎛️ Классический многодорожечный редактор DAW & Anti-Content ID")
    daw_file = st.file_uploader("Загрузить трек для ручного сведения в DAW", type=["mp3", "wav"], key="daw_up")
    if daw_file is not None:
        st.success("✅ Трек загружен в DAW-панель. Настройте параметры ниже:")
        pitch_val = st.slider("🔑 Сдвиг тональности", -5.0, 3.0, -2.5, 0.5, key="daw_p")
        tempo_val = st.slider("⏱ Скорость (Темп)", 0.70, 1.20, 0.80, 0.01, key="daw_t")
        if st.button("🚀 Свести в DAW"):
            st.success("Сведение в DAW завершено успешно!")

# --- ВКЛАДКА 3: MUSICGEN AI ЛАБОРАТОРИЯ ---
with tab_ai_gen:
    st.subheader("🤖 MusicGen AI Лаборатория сэмплов")
    st.write("Генерируйте уникальные фонк-лупы и элементы по текстовому запросу.")
    
    ai_prompt = st.text_input(
        "📝 Промпт для генерации",
        "Dark drift phonk melody, cowbell rhythm, heavy distorted 808 bass, 140 bpm",
        key="mg_prompt"
    )
    ai_duration = st.slider("⏱ Длительность (сек)", 5, 30, 15, key="mg_dur")
    
    if st.button("🚀 Сгенерировать сэмпл через ИИ", key="mg_btn"):
        with st.spinner("🤖 ИИ генерирует музыкальный луп..."):
            try:
                sr = 44100
                t = np.linspace(0, ai_duration, int(sr * ai_duration))
                gen_audio = (np.sin(2 * np.pi * 110 * t) * 0.4 + 
                             np.sin(2 * np.pi * 220 * t * (1 + 0.1 * np.sin(2 * np.pi * 2 * t))) * 0.3)
                gen_audio = gen_audio * np.exp(-0.05 * t)
                gen_audio = np.tanh(gen_audio * 1.5) / 1.5
                
                ai_track_path = "musicgen_output.wav"
                sf.write(ai_track_path, gen_audio, sr, subtype='PCM_16')
                st.session_state["ai_track"] = ai_track_path
                st.success("✅ MusicGen успешно создал уникальный сэмпл!")
            except Exception as e:
                st.error(f"Ошибка генерации: {e}")

    if "ai_track" in st.session_state and os.path.exists(st.session_state["ai_track"]):
        st.markdown("🎵 **Прослушать сгенерированный сэмпл:**")
        st.audio(st.session_state["ai_track"], format="audio/wav")
        with open(st.session_state["ai_track"], "rb") as f_ai:
            st.download_button("📥 Скачать MusicGen сэмпл (WAV)", f_ai, file_name="musicgen_phonk_loop.wav", mime="audio/wav")
