import streamlit as st
import os

st.set_page_config(page_title="NBS SOFT — Studio", page_icon="👑", layout="centered")

st.title("👑 NBS SOFT — Music & Audio Studio")
st.write("Студия успешно запущена и работает в стабильном режиме!")

# Простая вкладка для проверки
tab1, tab2 = st.tabs(["🎵 Аудио инструменты", "🎨 Информация"])

with tab1:
    st.subheader("Загрузка и обработка аудио")
    uploaded_file = st.file_uploader("Загрузите MP3 или WAV файл", type=["mp3", "wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        st.success("Аудиофайл успешно загружен в систему!")
        
        song_title = st.text_input("Название трека", "I WELCOME OCTOBER")
        artist_name = st.text_input("Имя артиста", "Smagulov & Zhaken")
        
        if st.button("Сгенерировать SEO для YouTube"):
            st.markdown("### 📋 Готовые теги и описание:")
            st.code(f"{song_title} - {artist_name} [Slowed & Reverb / HQ]", language="text")
            st.success("Готово!")

with tab2:
    st.subheader("О проекте")
    st.write("Платформа разработана для управления релизами, треками и обложками.")
    st.write("👤 Аккаунт: **Активен**")
