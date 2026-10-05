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
st.set_page_config(page_title="AI YouTube Music & Remix Studio", page_icon="🎛️", layout="centered")

# Сессия для авторизации
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# --- ЭКРАН АВТОРИЗАЦИИ / РЕГИСТРАЦИИ ---
if not st.session_state["logged_in"]:
    st.title("🔐 Вход в платформу AI Music Suite")
    st.write("Войдите в свой аккаунт или зарегистрируйтесь, чтобы получить доступ к генератору и студии ремиксов.")
    
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
                    st.success("Регистрация успешна! Теперь перейдите во вкладку «Вход».")
                else:
                    st.error("Такой логин уже занят.")
            else:
                st.warning("Заполните все поля.")
    
    st.stop()

# --- ОСНОВНОЙ ФУНКЦИОНАЛ ПРИЛОЖЕНИЯ ---
st.sidebar.write(f"👤 Аккаунт: **{st.session_state['username']}**")
if st.sidebar.button("🚪 Выйти из аккаунта"):
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.rerun()

st.title("🎛️ AI Музыкальный продюсер, Ремиксы & YouTube SEO")
st.write("Создавайте безопасные треки, профессиональные ремиксы, уникальные обложки и готовое английское SEO!")

# Загрузка трека
uploaded_file = st.file_uploader("Загрузите исходный трек (.mp3 или .wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    audio_path = "input_track.mp3"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("Файл успешно загружен!")
    
    # Режим работы
    mode = st.radio(
        "🎯 Выберите режим обработки:",
        ["🎵 Защита от Content ID (Стандарт)", "🎚️ Профессиональный Ремикс & Эффекты"]
    )
    
    # Настройки стиля и обложки
    st.subheader("🎨 Настройки релиза и обложки")
    
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
    
    song_title = st.text_input("Название трека", "I WELCOME OCTOBER (Remix)")
    artist_name = st.text_input("Имя автора / Артиста", "Smagulov")
    
    default_genre_text = genre_category.split(" (")[0]
    genre_text = st.text_input("Текст жанра на обложке", default_genre_text)

    # Параметры ремикса в зависимости от выбранного режима
    pitch_shift, tempo_factor, add_reverb, add_vinyl = 0.0, 1.0, False, False

    if "Ремикс" in mode or "SLOWED" in genre_category:
        with st.expander("🎛️ Настройки студийных эффектов ремикса"):
            remix_preset = st.selectbox(
                "Пресет ремикса:",
                ["Slowed & Reverb (Замедленный с эхом)", "Nightcore (Ускоренный + высокий тон)", "Phonk Drift (Качающий бас)", "Lo-Fi Vinyl (Шум винила + теплота)", "Кастомные настройки"]
            )
            
            if "Slowed" in remix_preset or "SLOWED" in genre_category:
                pitch_shift = st.slider("Сдвиг тональности", -4.0, 2.0, -2.0, 0.5)
                tempo_factor = st.slider("Скорость (Темп)", 0.75, 0.95, 0.82, 0.01)
                add_reverb = True
                add_vinyl = True
            elif "Nightcore" in remix_preset:
                pitch_shift = st.slider("Сдвиг тональности", 1.0, 5.0, 3.0, 0.5)
                tempo_factor = st.slider("Скорость (Темп)", 1.10, 1.35, 1.20, 0.01)
            elif "Phonk" in remix_preset:
                pitch_shift = st.slider("Сдвиг тональности", -3.0, 1.0, -1.5, 0.5)
                tempo_factor = st.slider("Скорость (Темп)", 0.90, 1.10, 1.05, 0.01)
                add_reverb = True
            elif "Lo-Fi" in remix_preset:
                pitch_shift = st.slider("Сдвиг тональности", -1.0, 1.0, -0.5, 0.5)
                tempo_factor = st.slider("Скорость (Темп)", 0.85, 0.98, 0.92, 0.01)
                add_vinyl = True
            else:
                pitch_shift = st.slider("Сдвиг тональности (полутоны)", -4.0, 4.0, -1.0, 0.5)
                tempo_factor = st.slider("Коэффициент темпа", 0.75, 1.25, 0.90, 0.01)
                add_reverb = st.checkbox("Добавить эффект эха (Reverb)")
                add_vinyl = st.checkbox("Добавить шум винила (Vinyl Crackle)")
    else:
        with st.expander("⚙️ Тонкая настройка аудио (Защита от Content ID)"):
            pitch_shift = st.slider("Сдвиг тональности (полутоны)", -4.0, 4.0, -0.5, 0.5)
            tempo_factor = st.slider("Коэффициент темпа (скорость)", 0.85, 1.15, 0.98, 0.01)

    if st.button("🚀 Запустить генерацию ремикса, обложки и SEO"):
        
        # --- ОБРАБОТКА АУДИО ---
        with st.spinner("🔄 Шаг 1/2: Применяем студийные эффекты и ремикс..."):
            try:
                y, sr = librosa.load(audio_path, sr=None, res_type='kaiser_best')
                
                # Тон и темп
                if pitch_shift != 0.0:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift, n_fft=2048, hop_length=512)
                if tempo_factor != 1.0:
                    y = librosa.effects.time_stretch(y, rate=tempo_factor)
                
                # Имитация реверберации (простое наложение задержанного сигнала с затуханием)
                if add_reverb:
                    delay_samples = int(sr * 0.08) # 80ms delay
                    reverb_signal = np.zeros_like(y)
                    if len(y) > delay_samples:
                        reverb_signal[delay_samples:] = y[:-delay_samples] * 0.35
                        y = y + reverb_signal

                # Имитация винилового шума / теплоты
                if add_vinyl:
                    noise = np.random.normal(0, 0.004, len(y))
                    y = y + noise

                # Нормализация звука
                max_val = max(abs(y.min()), abs(y.max()))
                if max_val > 0:
                    y = y / max_val * 0.95
                
                safe_audio_path = "remixed_track.wav"
                sf.write(safe_audio_path, y, sr, subtype='PCM_16')
            except Exception as e:
                st.error(f"Ошибка при обработке аудио: {e}")
                st.stop()
        
        # --- ГЕНЕРАЦИЯ ОБЛОЖКИ ---
        with st.spinner("🎨 Шаг 2/2: Создаем уникальную обложку 1280x720..."):
            bg_image = None
            try:
                if "SLOWED" in genre_category or "Reverb" in genre_category:
                    photo_ids = [1047, 1058, 1084, 1039]
                elif "PHONK" in genre_category:
                    photo_ids = [1078, 1062, 1048, 111]
                elif "NIGHTCORE" in genre_category:
                    photo_ids = [1015, 1016, 1042]
                else:
                    photo_ids = [1060, 1068, 1074, 1081]
                
                random_photo_id = random.choice(photo_ids)
                image_url = f"https://picsum.photos/id/{random_photo_id}/1280/720"
                
                response = requests.get(image_url, timeout=15)
                if response.status_code == 200 and len(response.content) > 2048:
                    bg_image = Image.open(BytesIO(response.content)).convert("RGB")
                    bg_image = bg_image.filter(ImageFilter.GaussianBlur(radius=3))
            except:
                bg_image = None

            if bg_image is None:
                width, height = 1280, 720
                bg_image = Image.new("RGB", (width, height))
                draw_bg = ImageDraw.Draw(bg_image)
                for y_px in range(height):
                    r = int(20 + (10 - 20) * y_px / height)
                    g = int(20 + (10 - 20) * y_px / height)
                    b = int(35 + (20 - 35) * y_px / height)
                    draw_bg.line([(0, y_px), (width, y_px)], fill=(r, g, b))

            darken = Image.new("RGBA", bg_image.size, (0, 0, 0, 175))
            cover_image = Image.alpha_composite(bg_image.convert("RGBA"), darken)
            draw = ImageDraw.Draw(cover_image)

            font_paths = ["C:\\Windows\\Fonts\\arialbd.ttf", "C:\\Windows\\Fonts\\segoeui.ttf"]
            title_font, subtitle_font = None, None
            for path in font_paths:
                if os.path.exists(path):
                    try:
                        title_font = ImageFont.truetype(path, 70)
                        subtitle_font = ImageFont.truetype(path, 34)
                        break
                    except:
                        continue
            
            if not title_font:
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()

            def draw_text_with_outline(draw_obj, xy, text, font, fill_color):
                x, y = xy
                for adj_x in range(-4, 5):
                    for adj_y in range(-4, 5):
                        if adj_x != 0 or adj_y != 0:
                            draw_obj.text((x + adj_x, y + adj_y), text, font=font, fill=(0,0,0,255))
                draw_obj.text((x, y), text, font=font, fill=fill_color)

            draw_text_with_outline(draw, (100, 70), song_title.upper(), title_font, (255, 255, 255, 255))
            draw_text_with_outline(draw, (100, 160), genre_text.upper(), subtitle_font, (255, 220, 100, 255))

            cover_filename = "youtube_cover.png"
            cover_image.convert("RGB").save(cover_filename, "PNG")

        st.success("🎉 Готово! Ваш ремикс и обложка успешно созданы.")

        # --- ВЫВОД РЕЗУЛЬТАТОВ ---
        st.subheader("📥 Скачать результаты:")
        st.image(cover_filename, caption="Обложка релиза для YouTube (1280x720)", use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            with open(cover_filename, "rb") as img_file:
                st.download_button("📥 Скачать обложку (PNG)", img_file, file_name="youtube_cover.png", mime="image/png")
        with col2:
            if os.path.exists(safe_audio_path):
                with open(safe_audio_path, "rb") as aud_file:
                    st.download_button("📥 Скачать ремикс (WAV)", aud_file, file_name="remixed_track.wav", mime="audio/wav")

        if os.path.exists(safe_audio_path):
            st.audio(safe_audio_path, format="audio/wav")

        # --- YOUTUBE SEO ---
        st.markdown("---")
        st.subheader("🚀 Англоязычное YouTube SEO для ремикса")
        clean_genre = default_genre_text.split("/")[0].strip()
        
        yt_titles = [
            f"{song_title} [{clean_genre} Vibe / Audio]",
            f"{artist_name} - {song_title} (Slowed & Reverb / Boosted)",
            f"{song_title} — {clean_genre} Remix (HQ Audio)"
        ]
        
        yt_description = f"""🎵 {artist_name} — {song_title}
🎧 Version: {clean_genre}

Immerse yourself in the ultimate atmosphere. Enjoy the vibe, drop a like, and subscribe for more daily releases!

📌 Follow & Support:
• Telegram Channel: Your Link Here
• Spotify / Apple Music: Your Link Here

#️⃣ Hashtags:
#{song_title.replace(' ', '')} #{clean_genre.replace(' ', '')} #SlowedAndReverb #Vibes #Music #Audio #Remix
"""
        st.markdown("**💡 Названия для видео:**")
        for t in yt_titles:
            st.code(t, language="text")

        st.markdown("**📝 SEO Описание:**")
        st.text_area("Скопируйте описание:", yt_description, height=160)