import streamlit as st
import os
import random
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
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
st.set_page_config(page_title="NBS SOFT — Pro Multi-Track DAW Session", page_icon="👑", layout="wide")

# Сессия для авторизации
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# --- ЭКРАН АВТОРИЗАЦИИ / РЕГИСТРАЦИИ ---
if not st.session_state["logged_in"]:
    st.title("🔐 NBS SOFT — Вход в многодорожечную студию")
    st.write("Войдите в систему для доступа к DAW-сессии.")
    
    col_l1, col_l2 = st.columns(2)
    with col_l1:
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
                
    with col_l2:
        reg_user = st.text_input("Придумайте логин", key="reg_u")
        reg_pass = st.text_input("Придумайте пароль", type="password", key="reg_p")
        if st.button("Зарегистрироваться"):
            if reg_user and reg_pass:
                if register_user(reg_user, reg_pass):
                    st.success("Регистрация успешна! Войдите слева.")
                else:
                    st.error("Логин уже занят.")
            else:
                st.warning("Заполните все поля.")
    
    st.stop()

# --- БОКОВАЯ ПАНЕЛЬ ---
if os.path.exists("logo_cropped.png"):
    st.sidebar.image("logo_cropped.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.write(f"👤 Аккаунт: **{st.session_state['username']}**")
if st.sidebar.button("🚪 Выйти"):
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.rerun()

# --- ОСНОВНОЙ ИНТЕРФЕЙС DAW ---
st.title("👑 NBS SOFT — Multi-Track DAW Session (Audition / FL Style)")
st.write("Многодорожечная сессия с визуализацией форм волны (Waveforms), таймлайном и независимым сдвигом треков.")

# Секция релизов
st.subheader("🎨 Настройки релиза и обложки")
col_meta1, col_meta2, col_meta3 = st.columns(3)
with col_meta1:
    song_title = st.text_input("Название трека", "I WELCOME OCTOBER")
with col_meta2:
    artist_name = st.text_input("Артист", "Smagulov & Zhaken")
with col_meta3:
    genre_category = st.selectbox(
        "Стиль",
        ["SLOWED & REVERB", "PHONK / DRIFT", "LO-FI BEATS", "NIGHTCORE", "SYNTHWAVE"]
    )

st.markdown("---")
st.subheader("🎚️ Редактор сессии: Дорожки (Tracks)")

# --- УПРАВЛЕНИЕ ДОРОЖКАМИ СЕССИИ (TRACK 1, TRACK 2, TRACK 3) ---
col_tr1, col_tr2, col_tr3 = st.columns(3)

with col_tr1:
    st.markdown("### 🟢 Track 1 (Основной трек)")
    file_t1 = st.file_uploader("Загрузить аудио для Track 1 (.mp3 / .wav)", type=["mp3", "wav"], key="f_t1")
    vol_t1 = st.slider("Громкость Trk 1", 0.0, 2.0, 1.0, 0.1, key="v_t1")
    shift_t1 = st.slider("Сдвиг Trk 1 (сек)", -2.0, 5.0, 0.0, 0.05, key="s_t1")
    pitch_t1 = st.slider("Питч Trk 1", -5.0, 3.0, -2.0, 0.5, key="p_t1")

with col_tr2:
    st.markdown("### 🔵 Track 2 (Бочка / Бас)")
    file_t2 = st.file_uploader("Загрузить аудио для Track 2 (.mp3 / .wav)", type=["mp3", "wav"], key="f_t2")
    vol_t2 = st.slider("Громкость Trk 2", 0.0, 2.0, 1.2, 0.1, key="v_t2")
    shift_t2 = st.slider("Сдвиг Trk 2 (сек)", -2.0, 5.0, 0.0, 0.05, key="s_t2")
    use_synth_kick = st.checkbox("Использовать синтезированный Kick/Bass", value=True, key="syn_k")

with col_tr3:
    st.markdown("### 🟣 Track 3 (Снейр / Перкуссия)")
    file_t3 = st.file_uploader("Загрузить аудио для Track 3 (.mp3 / .wav)", type=["mp3", "wav"], key="f_t3")
    vol_t3 = st.slider("Громкость Trk 3", 0.0, 2.0, 0.9, 0.1, key="v_t3")
    shift_t3 = st.slider("Сдвиг Trk 3 (сек)", -2.0, 5.0, 0.0, 0.05, key="s_t3")
    use_synth_snare = st.checkbox("Использовать синтезированный Snare", value=True, key="syn_s")

st.markdown("---")
st.subheader("🎛️ Мастер-эффекты сессии (Master FX)")
col_fx1, col_fx2 = st.columns(2)
with col_fx1:
    tempo_factor = st.slider("⏱ Общая скорость (Темп / Stretch)", 0.70, 1.20, 0.82, 0.01)
    trim_start_sec = st.slider("✂️ Обрезка старта (Анти-артефакт «тыыыз»)", 0.0, 1.0, 0.15, 0.05)
with col_fx2:
    reverb_mix = st.slider("🌊 Реверберация (Эхо)", 0.0, 1.0, 0.40, 0.05)
    vinyl_noise = st.slider("📻 Виниловый шум", 0.0, 0.015, 0.002, 0.001)

if st.button("🚀 Свести все дорожки в единую сессию (Master Mix)"):
    if file_t1 is None and not use_synth_kick:
        st.warning("Загрузите хотя бы аудиофайл для Track 1.")
    else:
        with st.spinner("🔄 Рендеринг многодорожечной сессии, выравнивание таймлайна и мастеринг..."):
            try:
                sr = 44100
                
                # Загрузка и обработка Track 1
                y1 = np.array([])
                if file_t1 is not None:
                    path1 = "temp_t1.mp3"
                    with open(path1, "wb") as f:
                        f.write(file_t1.getbuffer())
                    y1, sr = librosa.load(path1, sr=sr, mono=True)
                else:
                    y1 = np.zeros(sr * 10) # 10 секунд тишины если пустой
                
                # Защита от артефактов
                pad_len = int(sr * 1.0)
                y1_padded = np.pad(y1, (pad_len, pad_len), mode='constant')
                if pitch_t1 != 0.0:
                    y1_padded = librosa.effects.pitch_shift(y1_padded, sr=sr, n_steps=pitch_t1)
                if tempo_factor != 1.0:
                    y1_padded = librosa.effects.time_stretch(y1_padded, rate=tempo_factor)
                
                effective_pad = int(pad_len * (1.0 / tempo_factor))
                if len(y1_padded) > 2 * effective_pad:
                    y1 = y1_padded[effective_pad : -effective_pad]
                else:
                    y1 = y1_padded

                # Обрезка старта
                trim_samples = int(trim_start_sec * sr)
                if len(y1) > trim_samples:
                    y1 = y1[trim_samples:]
                
                # Нормализация
                if len(y1) > 0 and np.max(np.abs(y1)) > 0:
                    y1 = y1 / np.max(np.abs(y1)) * 0.7 * vol_t1

                # Анализ BPM для ритм-дорожек
                tempo_detected, beat_frames = librosa.beat.beat_track(y=y1 if len(y1)>0 else np.zeros(sr), sr=sr)
                track_bpm = float(tempo_detected[0]) if isinstance(tempo_detected, np.ndarray) else float(tempo_detected)
                effective_bpm = track_bpm * tempo_factor

                # Определяем максимальную длину микса
                max_len = len(y1) + int(5.0 * sr)
                master_mix = np.zeros(max_len)

                # Микшируем Track 1 с учетом сдвига
                shift1_samples = int(shift_t1 * sr)
                start_idx1 = max(0, shift1_samples)
                end_idx1 = start_idx1 + len(y1)
                if end_idx1 > len(master_mix):
                    master_mix = np.pad(master_mix, (0, end_idx1 - len(master_mix)), mode='constant')
                master_mix[start_idx1:end_idx1] += y1

                # Загрузка или генерация Track 2 (Kick / Bass)
                y2 = np.array([])
                if file_t2 is not None:
                    path2 = "temp_t2.mp3"
                    with open(path2, "wb") as f:
                        f.write(file_t2.getbuffer())
                    y2, _ = librosa.load(path2, sr=sr, mono=True)
                    y2 = y2 / (np.max(np.abs(y2)) + 1e-6) * 0.7 * vol_t2
                elif use_synth_kick and len(beat_frames) > 0:
                    kick_dur = 0.12
                    t_k = np.linspace(0, kick_dur, int(sr * kick_dur))
                    freq_sw = np.linspace(140, 45, len(t_k))
                    kick_wave = np.sin(2 * np.pi * freq_sw * t_k) * np.exp(-8 * t_k) * 0.6 * vol_t2
                    
                    y2 = np.zeros(len(master_mix))
                    adjusted_frames = (beat_frames / tempo_factor).astype(int) - trim_samples
                    for frame in adjusted_frames:
                        idx = int(frame)
                        if 0 <= idx < len(y2) - len(kick_wave):
                            y2[idx:idx + len(kick_wave)] += kick_wave

                if len(y2) > 0:
                    shift2_samples = int(shift_t2 * sr)
                    start_idx2 = max(0, shift2_samples)
                    end_idx2 = start_idx2 + len(y2)
                    if end_idx2 > len(master_mix):
                        master_mix = np.pad(master_mix, (0, end_idx2 - len(master_mix)), mode='constant')
                    master_mix[start_idx2:end_idx2] += y2[:len(master_mix)-start_idx2]

                # Загрузка или генерация Track 3 (Snare / Perc)
                y3 = np.array([])
                if file_t3 is not None:
                    path3 = "temp_t3.mp3"
                    with open(path3, "wb") as f:
                        f.write(file_t3.getbuffer())
                    y3, _ = librosa.load(path3, sr=sr, mono=True)
                    y3 = y3 / (np.max(np.abs(y3)) + 1e-6) * 0.6 * vol_t3
                elif use_synth_snare and len(beat_frames) > 0:
                    snare_dur = 0.08
                    t_s = np.linspace(0, snare_dur, int(sr * snare_dur))
                    snare_wave = (np.random.normal(0, 1, len(t_s)) * np.exp(-18 * t_s)) * 0.4 * vol_t3
                    
                    y3 = np.zeros(len(master_mix))
                    adjusted_frames = (beat_frames / tempo_factor).astype(int) - trim_samples
                    for i, frame in enumerate(adjusted_frames):
                        if i % 2 == 1:
                            idx = int(frame)
                            if 0 <= idx < len(y3) - len(snare_wave):
                                y3[idx:idx + len(snare_wave)] += snare_wave

                if len(y3) > 0:
                    shift3_samples = int(shift_t3 * sr)
                    start_idx3 = max(0, shift3_samples)
                    end_idx3 = start_idx3 + len(y3)
                    if end_idx3 > len(master_mix):
                        master_mix = np.pad(master_mix, (0, end_idx3 - len(master_mix)), mode='constant')
                    master_mix[start_idx3:end_idx3] += y3[:len(master_mix)-start_idx3]

                # --- ЭФФЕКТЫ МАСТЕРА ---
                if reverb_mix > 0:
                    delay_samples = int(sr * 0.15)
                    rev_sig = np.zeros_like(master_mix)
                    if len(master_mix) > delay_samples:
                        rev_sig[delay_samples:] = master_mix[:-delay_samples] * reverb_mix
                        master_mix = master_mix + rev_sig

                if vinyl_noise > 0:
                    master_mix = master_mix + np.random.normal(0, vinyl_noise, len(master_mix))

                # Мастеринг лимитер
                master_mix = np.tanh(master_mix * 1.15) / 1.15
                max_v = np.max(np.abs(master_mix))
                if max_v > 0:
                    master_mix = master_mix / max_v * 0.92

                output_path = "session_master.wav"
                sf.write(output_path, master_mix, sr, subtype='PCM_16')
                st.session_state["ready_audio"] = output_path
                st.session_state["detected_bpm"] = round(effective_bpm, 1)

                # --- ОТРИСОВКА ВИЗУАЛЬНЫХ ДОРОЖЕК WAVELINES (КАК В ADOBE AUDITION) ---
                fig, axes = plt.subplots(3, 1, figsize=(10, 6), facecolor="#1e1e1e")
                for ax in axes:
                    ax.set_facecolor("#121212")
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')

                # Track 1 plot
                t_axis1 = np.linspace(0, len(y1)/sr, len(y1)) + shift_t1
                axes[0].plot(t_axis1, y1, color="#00ffcc", linewidth=0.8)
                axes[0].set_title("Track 1: Main Audio Session", color="white", fontsize=10)
                axes[0].grid(True, color="#333333", linestyle="--")

                # Track 2 plot
                if len(y2) > 0:
                    t_axis2 = np.linspace(0, len(y2)/sr, len(y2)) + shift_t2
                    axes[1].plot(t_axis2, y2, color="#ff5555", linewidth=0.8)
                axes[1].set_title("Track 2: Kick / Bass Lane", color="white", fontsize=10)
                axes[1].grid(True, color="#333333", linestyle="--")

                # Track 3 plot
                if len(y3) > 0:
                    t_axis3 = np.linspace(0, len(y3)/sr, len(y3)) + shift_t3
                    axes[2].plot(t_axis3, y3, color="#aa55ff", linewidth=0.8)
                axes[2].set_title("Track 3: Snare / Percussion Lane", color="white", fontsize=10)
                axes[2].grid(True, color="#333333", linestyle="--")

                plt.tight_layout()
                waveform_filename = "session_waveforms.png"
                plt.savefig(waveform_filename, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close(fig)
                st.session_state["ready_waveforms"] = waveform_filename

            except Exception as e:
                st.error(f"Ошибка сведения сессии: {e}")
                st.stop()

        # --- ГЕНЕРАЦИЯ ОБЛОЖКИ ---
        with st.spinner("🎨 Создаем обложку релиза..."):
            try:
                bg_img = Image.open(BytesIO(requests.get(f"https://picsum.photos/id/{random.choice([1047,1058,1078,1062])}/1280/720").content)).convert("RGB")
                bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=3))
            except:
                bg_img = Image.new("RGB", (1280, 720), (20, 20, 30))

            cover = Image.alpha_composite(bg_img.convert("RGBA"), Image.new("RGBA", bg_img.size, (0, 0, 0, 185)))
            draw = ImageDraw.Draw(cover)
            font_path = "C:\\Windows\\Fonts\\arialbd.ttf"
            t_font = ImageFont.truetype(font_path, 85) if os.path.exists(font_path) else ImageFont.load_default()
            s_font = ImageFont.truetype(font_path, 42) if os.path.exists(font_path) else ImageFont.load_default()

            draw.text((90, 180), song_title.upper(), font=t_font, fill=(255,255,255,255))
            draw.text((90, 300), f"⚡ {genre_category} | NBS SOFT DAW", font=s_font, fill=(255,215,0,255))
            
            cov_path = "youtube_cover.png"
            cover.convert("RGB").save(cov_path, "PNG")
            st.session_state["ready_cover"] = cov_path

        st.success("🎉 Многодорожечная сессия успешно сведена!")

# --- ВЫВОД РЕЗУЛЬТАТОВ И ВИЗУАЛИЗАЦИИ СЕССИИ ---
if "ready_waveforms" in st.session_state and os.path.exists(st.session_state["ready_waveforms"]):
    st.markdown("---")
    st.subheader("🖥️ Визуализация сессии таймлайна (Waveform Tracks):")
    st.image(st.session_state["ready_waveforms"], caption="Многодорожечные аудиоволны сессии", use_container_width=True)

if "ready_audio" in st.session_state and os.path.exists(st.session_state["ready_audio"]):
    st.markdown("---")
    st.subheader(f"🎵 Мастер-микс сессии (BPM: {st.session_state.get('detected_bpm', 'Auto')}):")
    st.audio(st.session_state["ready_audio"], format="audio/wav")
    
    with open(st.session_state["ready_audio"], "rb") as af:
        st.download_button("📥 Скачать Master WAV", af, file_name="session_master.wav", mime="audio/wav")

if "ready_cover" in st.session_state and os.path.exists(st.session_state["ready_cover"]):
    st.markdown("---")
    st.subheader("📥 Обложка релиза:")
    st.image(st.session_state["ready_cover"], use_container_width=True)
    with open(st.session_state["ready_cover"], "rb") as cf:
        st.download_button("📥 Скачать обложку PNG", cf, file_name="youtube_cover.png", mime="image/png")

# --- YOUTUBE SEO ---
st.markdown("---")
st.subheader("🚀 YouTube SEO Оптимизация")
yt_title = f"{song_title} [{genre_category} / Multi-Track DAW Mix] | NBS SOFT"
st.code(yt_title, language="text")
yt_desc = f"""Artist: {artist_name}
Track: {song_title}
Session Mix produced via NBS SOFT Multi-Track DAW Studio.

#️⃣ Tags: #{song_title.replace(' ','')} #{genre_category.split()[0]} #AudioSession #NBSSoft #Remix #StudioMaster
"""
st.text_area("Описание для видео:", yt_desc, height=120)
