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

# --- FUNCIONES DE VISUALIZACIÓN Y ESCALADO DE MÉTRICAS ---

def scale_metric_to_10(value, ideal_range, is_inverse=False):
    """
    Normaliza un valor a una escala de 0 a 10.
    """
    min_ideal, max_ideal = ideal_range
    
    if is_inverse:
        score = (max_ideal - value) / (max_ideal - min_ideal)
    else:
        score = (value - min_ideal) / (max_ideal - min_ideal)
        
    return np.clip(score, 0, 1) * 10

def display_metric_gauge(title, score, help_text):
    """Muestra una métrica con un título atractivo, un score y una barra de progreso."""
    st.markdown(f"**{title}**")
    # Asegurarse de que el score no sea None antes de formatear
    score_display = score if score is not None else 0.0
    st.markdown(f"<h3 style='text-align: center; color: #2E86C1;'>{score_display:.1f} / 10</h3>", unsafe_allow_html=True)
    st.progress(score_display / 10)
    st.caption(help_text)
    st.markdown("---")


# --- FUNCIONES PRINCIPALES DE LA INTERFAZ ---

def display_results(results_data):
    """Muestra los resultados finales en pestañas con las nuevas métricas y visualizaciones."""
    st.success("¡Análisis completado! Aquí tienes tu feedback de comunicación.")

    metadata = results_data.get("processing_metadata", {})
    duration = metadata.get("duration_seconds")

    if duration:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        st.info(f"⏱️ Tiempo total de análisis: {minutes} minutos y {seconds} segundos.")
    
    st.markdown("---") 

    # --- INICIO DE LA MODIFICACIÓN: Extraer también los resultados verbales ---
    non_verbal = results_data.get("non_verbal_expression", {})
    emotions = results_data.get("emotion_analysis", {})
    speech = results_data.get("speech_analysis", {})
    verbal = results_data.get("verbal_analysis", {}) # NUEVA LÍNEA
    # --- FIN DE LA MODIFICACIÓN ---
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Expresión No Verbal", "🗣️ Expresión Verbal", "🔊 Análisis del Habla", "⭐ Score Final"
    ])

    # --- PESTAÑA 1: EXPRESIÓN NO VERBAL (sin cambios) ---
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
        
        score_gest_height = scale_metric_to_10(non_verbal.get('gesticulation_height_avg', 1.0), RANGES["gesticulation_height"], is_inverse=True)
        score_gest_var = scale_metric_to_10(non_verbal.get('gesticulation_variability', 0.0), RANGES["gesticulation_variability"])
        score_mouth_open = scale_metric_to_10(non_verbal.get('mouth_opening_avg', 0.0), RANGES["mouth_opening"])
        score_body_dyn = scale_metric_to_10(non_verbal.get('body_dynamism', 0.0), RANGES["body_dynamism"])
        score_head_tilt = scale_metric_to_10(non_verbal.get('head_tilt_variability', 0.0), RANGES["head_tilt"])
        score_posture = scale_metric_to_10(non_verbal.get('posture_openness_avg', 0.0), RANGES["posture_openness"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_gauge("Postura de Confianza", score_posture, "Mide qué tan abierta y segura es tu postura. Una puntuación alta indica hombros erguidos y presencia.")
            display_metric_gauge("Claridad Vocal", score_mouth_open, "Basado en la apertura de tu boca. Una buena puntuación sugiere una articulación clara y potente.")
        with col2:
            display_metric_gauge("Presencia Escénica", score_gest_height, "Evalúa si tus manos están en una zona de gesticulación activa y visible, en lugar de caídas.")
            display_metric_gauge("Dinamismo en Escena", score_body_dyn, "Mide tu movimiento en el escenario. Una puntuación alta refleja un uso energético y seguro del espacio.")
        with col3:
            display_metric_gauge("Lenguaje de Manos", score_gest_var, "Puntúa la variedad y energía de tus gestos. Un buen score significa que usas tus manos para enfatizar tu mensaje.")
            display_metric_gauge("Conexión Empática", score_head_tilt, "Basado en la inclinación de tu cabeza. Un valor alto sugiere que conectas con la audiencia de forma natural.")
        
        st.header("Tu Paleta Emocional")
        if emotions and emotions.get('emotion_distribution'):
            emotion_translation = {'angry': 'Enfado','disgust': 'Asco','fear': 'Miedo','happy': 'Alegría','sad': 'Tristeza','surprise': 'Sorpresa','neutral': 'Neutralidad'}
            dominant_emotion_en = emotions.get('dominant_emotion', 'N/A')
            dominant_emotion_es = emotion_translation.get(dominant_emotion_en, dominant_emotion_en).capitalize()
            st.metric(label="Tono Emocional Dominante", value=dominant_emotion_es)
            emotion_dist = emotions.get('emotion_distribution', {})
            df_emotions = pd.DataFrame(list(emotion_dist.items()), columns=['Emoción', 'Porcentaje'])
            df_emotions['Porcentaje'] *= 100
            df_emotions['Emoción'] = df_emotions['Emoción'].map(emotion_translation).fillna(df_emotions['Emoción'])
            chart = alt.Chart(df_emotions).mark_bar().encode(x=alt.X('Emoción:N', title='Emoción', sort='-y'),y=alt.Y('Porcentaje:Q', title='Porcentaje (%)')).properties(title='Distribución Emocional')
            st.altair_chart(chart, use_container_width=True)
            st.caption("Esta gráfica muestra la variedad de emociones que se proyectan durante el discurso.")
        else:
            st.write("No se pudo realizar el análisis emocional.")

    # --- PESTAÑA 2: EXPRESIÓN VERBAL (AHORA CON CONTENIDO) ---
    with tab2:
        st.header("Análisis del Contenido de tu Discurso")
        
        # --- INICIO DE LA MODIFICACIÓN: Mostrar resultados verbales ---
        verbal_scores = verbal.get("verbal_scores", {})
        summary = verbal.get("summary", "No se pudo generar un resumen.")
        keywords = verbal.get("keywords", [])
        transcription = verbal.get("full_transcription", "No hay transcripción disponible.")
        
        if not verbal:
             st.warning("No se encontraron datos del análisis verbal.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                display_metric_gauge("Claridad del Mensaje", verbal_scores.get('message_clarity'), "Evalúa la simplicidad y longitud de tus frases. Un score alto indica un mensaje fácil de seguir.")
                display_metric_gauge("Riqueza Léxica", verbal_scores.get('lexical_diversity'), "Puntúa la variedad de tu vocabulario. Una puntuación alta evita la repetición y enriquece el mensaje.")
                display_metric_gauge("Fuerza del Cierre", verbal_scores.get('ending'), "Evalúa si el final del discurso contiene un llamado a la acción o palabras de cierre contundentes.")
            with col2:
                display_metric_gauge("Estructura Narrativa", verbal_scores.get('structure'), "Mide la cohesión y organización del discurso. Un buen score sugiere una estructura clara (inicio, nudo, desenlace).")
                display_metric_gauge("Uso de Retórica", verbal_scores.get('rhetoric'), "Detecta el uso de figuras retóricas (preguntas, repeticiones) para hacer el discurso más persuasivo.")

            st.subheader("Resumen del Mensaje")
            st.info(summary)

            st.subheader("Palabras Clave Detectadas")
            if keywords:
                st.success(", ".join(keywords).capitalize())
            else:
                st.caption("No se detectaron palabras clave significativas.")

            with st.expander("Ver transcripción completa"):
                st.text_area("Transcripción", transcription, height=300, disabled=True)
        # --- FIN DE LA MODIFICACIÓN ---

    # --- PESTAÑA 3: ANÁLISIS DEL HABLA (sin cambios) ---
    with tab3:
        st.header("Análisis de la Voz y el Habla")
        RANGES_SPEECH = {
            "pitch_variation": (15, 60),
            "silence_percentage": (40, 10),
            "volume_variability": (0.01, 0.1)
        }
        score_pitch = scale_metric_to_10(speech.get('pitch_variation', 0.0), RANGES_SPEECH["pitch_variation"])
        score_pauses = scale_metric_to_10(speech.get('silence_percentage', 40.0), RANGES_SPEECH["silence_percentage"], is_inverse=True)
        score_volume = scale_metric_to_10(speech.get('volume_variability', 0.0), RANGES_SPEECH["volume_variability"])

        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_gauge("Voz Carismática (Melodía)", score_pitch, "Mide la variación de tu tono. Una puntuación alta indica una voz expresiva y melódica que evita la monotonía.")
        with col2:
            display_metric_gauge("Ritmo y Pausa (Elocuencia)", score_pauses, "Evalúa tu uso de los silencios. Un buen score significa que usas las pausas para dar énfasis y claridad.")
        with col3:
            display_metric_gauge("Energía Vocal (Dinamismo)", score_volume, "Puntúa la variación en tu volumen. Una puntuación alta refleja una voz dinámica que usa la energía para captar la atención.")

    # --- PESTAÑA 4: SCORE FINAL (AHORA INCLUYE SCORES VERBALES) ---
    with tab4:
        st.header("Tu Score Global de Comunicación")
        st.info("Este es un score agregado basado en todas las métricas analizadas.")
        
        # --- INICIO DE LA MODIFICACIÓN: Añadir scores verbales al cálculo final ---
        verbal_scores_list = list(verbal.get("verbal_scores", {}).values())

        all_scores = [
            score_posture, score_mouth_open, score_gest_height, score_body_dyn, 
            score_gest_var, score_head_tilt, score_pitch, score_pauses, score_volume
        ]
        # Añadimos los scores verbales a la lista
        all_scores.extend(verbal_scores_list)
        
        # Filtramos posibles valores nulos o no numéricos antes de calcular la media
        valid_scores = [s for s in all_scores if isinstance(s, (int, float)) and not np.isnan(s)]
        
        final_score = np.mean(valid_scores) if valid_scores else 0.0
        # --- FIN DE LA MODIFICACIÓN ---
        
        st.markdown(f"<h1 style='text-align: center; color: #1E8449;'>{final_score:.1f} / 10</h1>", unsafe_allow_html=True)
        st.progress(final_score / 10)
        
        st.markdown("#### Recomendaciones Clave (Ejemplo)")
        st.markdown(f"- **Tu punto más fuerte:** {'Postura de Confianza' if final_score > 7 else 'Lenguaje de Manos'}.")
        st.markdown(f"- **Área a mejorar:** {'Dinamismo en Escena' if score_body_dyn < 6 else 'Claridad Vocal'}.")

    if st.button("Analizar otro video"):
        if 'job_id' in st.session_state: del st.session_state['job_id']
        st.rerun()

# --- LÓGICA DE FLUJO PRINCIPAL Y VALIDACIÓN (sin cambios) ---
def display_main_interface():
    st.set_page_config(page_title="Análisis de Conferencias", layout="wide")
    st.title("🔍 Análisis de Discursos y Conferencias")
    st.markdown("Introduce la URL de un video de una charla TED o TEDx para obtener un feedback detallado sobre tu comunicación.")
    url = st.text_input("🎥 Introduce el enlace del video de YouTube:", key="url_input")
    if st.button("📊 Analizar video", key="analyze_button"):
        if url: start_analysis_process(url)
        else: st.warning("Por favor, introduce una URL para analizar.")

def start_analysis_process(url):
    with st.spinner("Validando URL... Por favor, espera."):
        is_valid, message = validate_url_fast(url)
    if not is_valid:
        st.error(f"Error de validación: {message}")
        return
    job_id = str(uuid.uuid4())
    task_filepath = os.path.join(TASKS_DIR, f"{job_id}.json")
    with open(task_filepath, 'w') as f:
        json.dump({"url": url, "job_id": job_id}, f)
    st.session_state['job_id'] = job_id
    st.rerun()

def display_processing_status():
    job_id = st.session_state.get('job_id')
    st.info(f"✅ Tarea enviada. Estamos procesando tu video.")
    st.warning("Este proceso puede tardar varios minutos. La página se actualizará automáticamente.")
    st.spinner(f"Esperando resultados del análisis con ID: {job_id}")
    time.sleep(10)
    st.rerun()

def validate_url_fast(url):
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
        channel = info.get('channel', '').lower()
        title = info.get('title', '').lower()
        if "ted" in channel or "tedx" in channel or "ted talk" in title: return True, "URL válida."
        else: return False, "El video no parece ser una charla TED o TEDx."
    except Exception as e:
        logging.error(f"Error al validar la URL {url}: {e}")
        return False, "No se pudo validar la URL. Asegúrate de que es un enlace de YouTube válido y público."

def main_flow():
    job_id = st.session_state.get('job_id')
    if not job_id:
        display_main_interface()
    else:
        result_filepath = os.path.join(RESULTS_DIR, f"{job_id}.json")
        error_filepath = os.path.join(RESULTS_DIR, f"{job_id}.error")
        if os.path.exists(result_filepath):
            with open(result_filepath, 'r') as f: results_data = json.load(f)
            display_results(results_data)
        elif os.path.exists(error_filepath):
            with open(error_filepath, 'r') as f: error_message = f.read()
            st.error(f"Ocurrió un error durante el análisis: {error_message}")
            if st.button("Intentar con otro video"):
                del st.session_state['job_id']
                st.rerun()
        else:
            display_processing_status()

if __name__ == "__main__":
    main_flow()