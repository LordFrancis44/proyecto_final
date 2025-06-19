import streamlit as st
import os
import sys
import json
import time
import uuid
import yt_dlp
import logging
import pandas as pd
import numpy as np
import altair as alt

# --- CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TASKS_DIR = os.path.join(DATA_DIR, "tasks")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(TASKS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - WEB_APP - %(levelname)s - %(message)s')

# --- ESCALA ---
def scale_metric_to_10(value, ideal_range, is_inverse=False):
    min_ideal, max_ideal = ideal_range
    if is_inverse:
        score = (max_ideal - value) / (max_ideal - min_ideal)
    else:
        score = (value - min_ideal) / (max_ideal - min_ideal)
    return np.clip(score, 0, 1) * 10

# --- VISUALIZACIÓN ---
def display_metric_gauge(title, score, help_text):
    st.markdown(f"**{title}**")
    st.markdown(f"<h3 style='text-align: center; color: #2E86C1;'>{score:.1f} / 10</h3>", unsafe_allow_html=True)
    st.progress(score / 10)
    st.caption(help_text)
    st.markdown("---")

# --- RESULTADOS ---
def display_results(results_data):
    st.success("¡Análisis completado! Aquí tienes tu feedback de comunicación.")

    metadata = results_data.get("processing_metadata", {})
    duration = metadata.get("duration_seconds")
    if duration:
        st.info(f"⏱️ Tiempo de análisis: {int(duration // 60)} min {int(duration % 60)} seg.")

    non_verbal = results_data.get("non_verbal_expression", {})
    emotions = results_data.get("emotion_analysis", {})
    speech = results_data.get("speech_analysis", {})
    verbal = results_data.get("verbal_analysis", {})

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Expresión No Verbal", "🗣️ Expresión Verbal", "🔊 Análisis del Habla", "⭐ Score Final"
    ])

    # --- NO VERBAL ---
    with tab1:
        st.header("Tu Perfil de Comunicador No Verbal")
        RANGES = {
            "gesticulation_height": (1.0, 0.5),
            "gesticulation_variability": (0.0, 0.15),
            "mouth_opening": (0.05, 0.4),
            "body_dynamism": (0.0, 0.05),
            "head_tilt": (0.0, 5.0),
            "posture_openness": (1.5, 2.5)
        }

        scores_nv = {
            "posture": scale_metric_to_10(non_verbal.get('posture_openness_avg', 0.0), RANGES["posture_openness"]),
            "mouth": scale_metric_to_10(non_verbal.get('mouth_opening_avg', 0.0), RANGES["mouth_opening"]),
            "gest_height": scale_metric_to_10(non_verbal.get('gesticulation_height_avg', 1.0), RANGES["gesticulation_height"], is_inverse=True),
            "dyn": scale_metric_to_10(non_verbal.get('body_dynamism', 0.0), RANGES["body_dynamism"]),
            "gest_var": scale_metric_to_10(non_verbal.get('gesticulation_variability', 0.0), RANGES["gesticulation_variability"]),
            "tilt": scale_metric_to_10(non_verbal.get('head_tilt_variability', 0.0), RANGES["head_tilt"]),
        }

        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_gauge("Postura de Confianza", scores_nv["posture"], "Mide qué tan abierta y segura es tu postura.")
            display_metric_gauge("Claridad Vocal", scores_nv["mouth"], "Basado en la apertura de tu boca.")
        with col2:
            display_metric_gauge("Presencia Escénica", scores_nv["gest_height"], "Evalúa si tus manos están activas.")
            display_metric_gauge("Dinamismo en Escena", scores_nv["dyn"], "Mide tu movimiento en el escenario.")
        with col3:
            display_metric_gauge("Lenguaje de Manos", scores_nv["gest_var"], "Puntúa la variedad y energía de tus gestos.")
            display_metric_gauge("Conexión Empática", scores_nv["tilt"], "Basado en la inclinación de tu cabeza.")

    # --- VERBAL ---
    with tab2:
        st.header("Análisis del Contenido del Discurso")

        if "error" in verbal:
            st.error(f"❌ Error en el análisis verbal: {verbal['error']}")
        else:
            st.markdown("### 📝 Resumen del Discurso")
            st.write(verbal.get("summary", "No disponible."))

            st.markdown("### 🔑 Palabras Clave Detectadas")
            st.write(", ".join(verbal.get("keywords", [])))

            st.markdown("### 📈 Métricas Verbales")
            scores_vb = verbal.get("verbal_scores", {})

            verbal_metric_map = {
                "Claridad del Mensaje": ("message_clarity", "¿El mensaje se entendió con facilidad?"),
                "Estructura del Contenido": ("structure", "¿Está bien organizado el discurso?"),
                "Diversidad Léxica": ("lexical_diversity", "¿Usó variedad de palabras?"),
                "Recursos Retóricos": ("rhetoric", "¿Utilizó recursos como metáforas o preguntas?"),
                "Fuerza del Cierre": ("ending", "¿Cerró con una idea potente o memorable?")
            }

            for title, (json_key, help_text) in verbal_metric_map.items():
                score = scores_vb.get(json_key, 0)
                display_metric_gauge(title, score, help_text)

            st.markdown("### 📜 Transcripción Completa")
            st.text(verbal.get("full_transcription", "No disponible."))

    # --- HABLA ---
    with tab3:
        st.header("Análisis de la Voz y el Habla")
        RANGES_SPEECH = {
            "pitch_variation": (15, 60),
            "silence_percentage": (40, 10),
            "volume_variability": (0.01, 0.1)
        }

        scores_sp = {
            "pitch": scale_metric_to_10(speech.get('pitch_variation', 0.0), RANGES_SPEECH["pitch_variation"]),
            "silence": scale_metric_to_10(speech.get('silence_percentage', 40.0), RANGES_SPEECH["silence_percentage"], is_inverse=True),
            "volume": scale_metric_to_10(speech.get('volume_variability', 0.0), RANGES_SPEECH["volume_variability"]),
        }

        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_gauge("Voz Carismática (Melodía)", scores_sp["pitch"], "Variación del tono.")
        with col2:
            display_metric_gauge("Ritmo y Pausa", scores_sp["silence"], "Manejo de silencios.")
        with col3:
            display_metric_gauge("Energía Vocal", scores_sp["volume"], "Dinamismo de volumen.")

    # --- SCORE FINAL ---
    with tab4:
        st.header("Tu Score Global de Comunicación")
        all_scores = list(scores_nv.values()) + list(scores_sp.values()) + list(verbal.get("verbal_scores", {}).values())
        final_score = np.mean([s for s in all_scores if not np.isnan(s)])
        st.markdown(f"<h1 style='text-align: center; color: #1E8449;'>{final_score:.1f} / 10</h1>", unsafe_allow_html=True)
        st.progress(final_score / 10)

        st.markdown("#### Recomendaciones Clave (Ejemplo)")
        st.markdown(f"- **Tu punto más fuerte:** {'Postura de Confianza' if final_score > 7 else 'Claridad Vocal'}.")
        st.markdown(f"- **Área a mejorar:** {'Dinamismo en Escena' if scores_nv['dyn'] < 6 else 'Lexicalidad'}.")

    if st.button("Analizar otro video"):
        if 'job_id' in st.session_state: del st.session_state['job_id']
        st.rerun()

# --- LÓGICA GENERAL ---
def display_main_interface():
    st.set_page_config(page_title="Análisis de Conferencias", layout="wide")
    st.title("🔍 Análisis de Discursos y Conferencias")
    st.markdown("Introduce la URL de un video de una charla TED o TEDx para obtener un feedback detallado sobre tu comunicación.")
    url = st.text_input("🎥 Introduce el enlace del video de YouTube:", key="url_input")
    if st.button("📊 Analizar video", key="analyze_button"):
        if url: start_analysis_process(url)
        else: st.warning("Por favor, introduce una URL.")

def start_analysis_process(url):
    with st.spinner("Validando URL..."):
        is_valid, message = validate_url_fast(url)
    if not is_valid:
        st.error(f"Error: {message}")
        return
    job_id = str(uuid.uuid4())
    with open(os.path.join(TASKS_DIR, f"{job_id}.json"), 'w') as f:
        json.dump({"url": url, "job_id": job_id}, f)
    st.session_state['job_id'] = job_id
    st.rerun()

def display_processing_status():
    job_id = st.session_state.get('job_id')
    st.info(f"✅ Tarea enviada. Analizando video...")
    st.spinner(f"Procesando análisis para ID: {job_id}")
    time.sleep(10)
    st.rerun()

def validate_url_fast(url):
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
        channel = info.get('channel', '').lower()
        title = info.get('title', '').lower()
        if "ted" in channel or "tedx" in channel or "ted talk" in title:
            return True, "URL válida."
        else:
            return False, "El video no parece ser una charla TED o TEDx."
    except Exception as e:
        return False, "Error al validar la URL."

def main_flow():
    if 'job_id' not in st.session_state:
        display_main_interface()
    else:
        result_fp = os.path.join(RESULTS_DIR, f"{st.session_state['job_id']}.json")
        error_fp = os.path.join(RESULTS_DIR, f"{st.session_state['job_id']}.error")
        if os.path.exists(result_fp):
            with open(result_fp, 'r') as f:
                display_results(json.load(f))
        elif os.path.exists(error_fp):
            with open(error_fp, 'r') as f:
                st.error(f"Ocurrió un error: {f.read()}")
                if st.button("Intentar otro video"):
                    del st.session_state['job_id']
                    st.rerun()
        else:
            display_processing_status()

if __name__ == "__main__":
    main_flow()
