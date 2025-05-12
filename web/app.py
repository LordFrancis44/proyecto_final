import streamlit as st

st.set_page_config(page_title="Análisis de Conferencias", layout="wide")
st.title("🔍 Análisis de Conferencias TED")

# Entrada del usuario
url = st.text_input("🎥 Introduce el enlace del video de YouTube:")

if url:
    st.video(url)

    if st.button("📊 Analizar video"):
        st.success("Análisis completado.")  # Simulación de procesamiento

        # Pestañas para los diferentes análisis
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧠 Expresión no verbal", 
            "🗣 Expresión verbal", 
            "🔊 Análisis del habla", 
            "⭐ Score final"
        ])

        with tab1:
            st.subheader("Expresión no verbal")
            st.write("👁 Contacto visual: alto")
            st.write("😊 Emociones predominantes: alegría, sorpresa")

        with tab2:
            st.subheader("Expresión verbal")
            st.write("Palabras clave: innovación, impacto, sociedad")

        with tab3:
            st.subheader("Análisis del habla")
            st.write("Velocidad de habla: 135 palabras/minuto")
            st.write("Claridad fonética: buena")

        with tab4:
            st.subheader("Score total del discurso")
            st.metric("Puntuación global", "8.7 / 10")
