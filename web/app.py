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

def scale_metric_gaussian(value, ideal_point, width, is_inverse=False):
    """
    Escala un valor usando una curva Gaussiana. La puntuación es 10 en el 'ideal_point'
    y disminuye a medida que el valor se aleja. 'width' controla la tolerancia.
    """
    if value is None or np.isnan(value):
        return 0.0
    
    if is_inverse:
        if value <= ideal_point: return 10.0
        # La penalización aumenta a medida que 'value' se hace MÁS GRANDE que el ideal
        score = np.exp(-0.5 * ((value - ideal_point) / width) ** 2)
    else:
        # La penalización aumenta a medida que 'value' se aleja del ideal en CUALQUIER dirección
        score = np.exp(-0.5 * ((value - ideal_point) / width) ** 2)

    return score * 10

def display_metric_gauge(title, score, help_text):
    """Muestra una métrica con un título atractivo, un score y una barra de progreso."""
    st.markdown(f"**{title}**")
    # Asegurarse de que el score no sea None antes de formatear
    score_display = score if score is not None else 0.0
    st.markdown(f"<h3 style='text-align: center; color: #2E86C1;'>{score_display:.1f} / 10</h3>", unsafe_allow_html=True)
    st.progress(score_display / 10)
    st.caption(help_text)
    st.markdown("---")

def display_timestamp_link(label, example_data, video_url):
    """Muestra un enlace a un momento específico del vídeo si el timestamp existe."""
    if example_data and example_data.get('timestamp') is not None:
        ts = example_data['timestamp']
        text = example_data.get('text', '')
        st.markdown(f"**{label}:**")
        # El enlace abre el vídeo de YouTube en el segundo exacto
        link = f"<a href='{video_url}&t={int(ts)}s' target='_blank'>Ver momento (Minuto {int(ts//60)}:{int(ts%60):02d})</a>"
        st.markdown(link, unsafe_allow_html=True)
        if text:
            st.info(f"> {text}")

def display_results(results_data, video_url):
    """Muestra los resultados finales en pestañas con las nuevas métricas y visualizaciones."""
   
    video_metadata = results_data.get("video_metadata", {})
    video_title = video_metadata.get('title')
    if video_title:
        st.header(f"Análisis para: *{video_title}*")

    st.success("¡Análisis completado! Aquí tienes tu feedback de comunicación.")

    
    metadata = results_data.get("processing_metadata", {})
    duration = metadata.get("duration_seconds")

    if duration:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        st.info(f"⏱️ Tiempo total de análisis: {minutes} minutos y {seconds} segundos.")
    
    st.markdown("---") 

    non_verbal = results_data.get("non_verbal_expression", {})
    non_verbal_scores = non_verbal.get("scores", {})
    non_verbal_examples = non_verbal.get("examples", {})

    emotions = results_data.get("emotion_analysis", {})

    speech = results_data.get("speech_analysis", {})
    speech_scores = speech.get("scores", {})
    speech_examples = speech.get("examples", {})

    verbal = results_data.get("verbal_analysis", {})
    verbal_scores = verbal.get("scores", {})

    language_map = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian'
    }
    detected_code = verbal.get('detected_language', 'N/D')
    display_language = language_map.get(detected_code, detected_code.capitalize())
    st.info(f"Idioma detectado: **{display_language}**")

    qualitative_feedback = verbal.get("qualitative_feedback", {})
    filler_feedback = qualitative_feedback.get("filler_words", {})
    sentence_feedback = qualitative_feedback.get("key_sentences", {})
    
    RANGES_NON_VERBAL = {
            "posture_openness": {'ideal_point': 2.2, 'width': 0.5},
            "mouth_opening": {'ideal_point': 0.35, 'width': 0.15},
            "gesticulation_height": {'ideal_point': 0.5, 'width': 0.2, 'is_inverse': True}, # Ideal a la altura del pecho
            "body_dynamism": {'ideal_point': 0.04, 'width': 0.02},
            "gesticulation_variability": {'ideal_point': 0.1, 'width': 0.05},
            "head_tilt": {'ideal_point': 4.0, 'width': 2.0}
        }
    
    score_gest_height = scale_metric_gaussian(non_verbal_scores.get('gesticulation_height_avg', 1.0), **RANGES_NON_VERBAL["gesticulation_height"])
    score_gest_height = smooth_score(score_gest_height, factor=0.2)
    score_gest_var = scale_metric_gaussian(non_verbal_scores.get('gesticulation_variability', 0.0), **RANGES_NON_VERBAL["gesticulation_variability"])
    score_gest_var = smooth_score(score_gest_var, factor=0.2)
    score_mouth_open = scale_metric_gaussian(non_verbal_scores.get('mouth_opening_avg', 0.0), **RANGES_NON_VERBAL["mouth_opening"])
    score_mouth_open = smooth_score(score_mouth_open, factor=0.2)
    score_body_dyn = scale_metric_gaussian(non_verbal_scores.get('body_dynamism', 0.0), **RANGES_NON_VERBAL["body_dynamism"])
    score_body_dyn = smooth_score(score_body_dyn, factor=0.2)
    score_head_tilt = scale_metric_gaussian(non_verbal_scores.get('head_tilt_variability', 0.0), **RANGES_NON_VERBAL["head_tilt"])
    score_head_tilt = smooth_score(score_head_tilt, factor=0.3)
    score_posture = scale_metric_gaussian(non_verbal_scores.get('posture_openness_avg', 0.0), **RANGES_NON_VERBAL["posture_openness"])
    score_posture = smooth_score(score_posture, factor=0.3)
    
    RANGES_SPEECH = {
        "pitch_variation":    {'ideal_point': 55, 'width': 20},
        "silence_percentage": {'ideal_point': 18, 'width': 10},
        "volume_variability": {'ideal_point': 0.08, 'width': 0.05}
    }
    score_pitch = scale_metric_gaussian(speech_scores.get('pitch_variation'), **RANGES_SPEECH["pitch_variation"])
    score_pitch = smooth_score(score_pitch, factor=0.2)
    score_pauses = scale_metric_gaussian(speech_scores.get('silence_percentage'), **RANGES_SPEECH["silence_percentage"])
    score_pauses = smooth_score(score_pauses, factor=0.2)
    score_volume = scale_metric_gaussian(speech_scores.get('volume_variability'), **RANGES_SPEECH["volume_variability"])
    score_volume = smooth_score(score_volume, factor=0.2)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Expresión No Verbal", "🗣️ Contenido y Claridad", "🔊 Voz y Prosodia", "⭐ Resumen y Score Final"
    ])

    # --- PESTAÑA 1: EXPRESIÓN NO VERBAL ---
    with tab1:
        st.header("Tu Perfil de Comunicador No Verbal")
        
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
        
        with st.expander("🔍 Ver ejemplos donde mejorar"):
            st.write("Hemos identificado los siguientes ejemplos de aspectos que puedes mejorar")
            # La función display_timestamp_link no es ideal aquí, lo hacemos manual
            def show_improvement_example(label, timestamp, video_url):
                if timestamp is not None:
                    st.markdown(f"**{label}:**")
                    link = f"<a href='{video_url}&t={int(timestamp)}s' target='_blank'>Ver Ejemplo ({int(timestamp//60)}:{int(timestamp%60):02d})</a>"
                    st.markdown(link, unsafe_allow_html=True)

            non_verbal_examples = non_verbal.get("examples_to_improve", {})
            show_improvement_example("Mala posición de manos (caídas)", non_verbal_examples.get('gesticulation_height'), video_url)
            show_improvement_example("Postura cerrada", non_verbal_examples.get('posture_openness'), video_url)
            show_improvement_example("Poca inclinación de cabeza", non_verbal_examples.get('head_tilt'), video_url)
           


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

    # --- PESTAÑA 2: EXPRESIÓN VERBAL ---
    with tab2:
        st.header("Análisis del Contenido de tu Discurso")

        if not verbal:
            st.warning("No se encontraron datos del análisis verbal.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                display_metric_gauge("Claridad del Mensaje", verbal_scores.get('message_clarity'), "Evalúa la simplicidad y longitud de tus frases. Un score alto indica un mensaje fácil de seguir.")
                display_metric_gauge("Riqueza Léxica", verbal_scores.get('lexical_diversity'), "Puntúa la variedad de tu vocabulario para evitar la repetición.")
            with col2:
                display_metric_gauge("Estructura Narrativa", verbal_scores.get('structure'), "Mide la cohesión y organización de tu discurso.")
                display_metric_gauge("Uso de Retórica", verbal_scores.get('rhetoric'), "Detecta figuras retóricas para hacer el discurso más persuasivo.")
            with col3:
                # El nuevo gauge para muletillas
                display_metric_gauge("Uso de Muletillas", verbal_scores.get('filler_words_usage'), "Una puntuación alta indica un discurso fluido y sin apenas muletillas (ej: 'um', 'este', 'like').")
                display_metric_gauge("Fuerza del Cierre", verbal_scores.get('ending'), "Evalúa si el final es contundente y claro.")
            
            st.markdown("---")
            st.subheader("💡 Feedback Cualitativo")

            # Sección de Muletillas
            with st.container():
                st.markdown("#### Análisis de Muletillas")
                if filler_feedback and filler_feedback.get('total_count', 0) > 0:
                    total_count = filler_feedback['total_count']
                    most_common = filler_feedback.get('most_common', 'N/A')
                    dist = filler_feedback.get('distribution', {})
                    
                    st.warning(f"Hemos detectado **{total_count}** muletillas en tu discurso.")
                    st.write(f"La más frecuente fue **'{most_common}'**.")
                    
                    # Mostrar enlace al primer uso de la muletilla más común
                    if filler_feedback.get('example_timestamp'):
                         ts = filler_feedback['example_timestamp']
                         st.markdown(f"Puedes escuchar un ejemplo <a href='{video_url}&t={int(ts)}s' target='_blank'>aquí</a>.", unsafe_allow_html=True)

                    with st.expander("Ver desglose de muletillas"):
                        df_fillers = pd.DataFrame(dist.items(), columns=['Muletilla', 'Frecuencia']).sort_values('Frecuencia', ascending=False)
                        st.dataframe(df_fillers)
                else:
                    st.success("¡Felicidades! Apenas hemos detectado muletillas en tu discurso. ¡Muy buen trabajo!")

            # Sección de Frases Clave
            with st.container():
                st.markdown("#### Oraciones Clave Identificadas")
                if sentence_feedback:
                    best_sentence = sentence_feedback.get('best_sentence')
                    confusing_sentence = sentence_feedback.get('confusing_sentence')

                    if best_sentence:
                        st.markdown("**✅ Ejemplo de Oración Clara y Potente:**")
                        st.info(f"> {best_sentence['text']}")
                        ts = best_sentence['timestamp']
                        st.markdown(f"Escúchala en el <a href='{video_url}&t={int(ts)}s' target='_blank'>minuto {int(ts//60)}:{int(ts%60):02d}</a>.", unsafe_allow_html=True)
                    
                    if confusing_sentence:
                        st.markdown("**⚠️ Ejemplo de Oración Potencialmente Confusa:**")
                        st.warning(f"> {confusing_sentence['text']}")
                        ts = confusing_sentence['timestamp']
                        st.markdown(f"Escúchala en el <a href='{video_url}&t={int(ts)}s' target='_blank'>minuto {int(ts//60)}:{int(ts%60):02d}</a>.", unsafe_allow_html=True)

            # Resumen y Transcripción
            st.markdown("---")
            with st.expander("Ver Resumen y Palabras clave"):
                st.subheader("Resumen")
                st.info(verbal.get("summary", "No se pudo generar un resumen."))
                st.subheader("Palabras Clave")
                st.success(", ".join(verbal.get("keywords", [])).capitalize())


    # --- PESTAÑA 3: ANÁLISIS DEL HABLA ---
    with tab3:
        st.header("Análisis de la Prosodia")

        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_gauge("Voz Carismática (Melodía)", score_pitch, "Mide la variación de tu tono. Una puntuación alta indica una voz expresiva y melódica que evita la monotonía.")
        with col2:
            display_metric_gauge("Ritmo y Pausa (Elocuencia)", score_pauses, "Evalúa tu uso de los silencios. Un buen score significa que usas las pausas para dar énfasis y claridad.")
        with col3:
            display_metric_gauge("Energía Vocal (Dinamismo)", score_volume, "Puntúa la variación en tu volumen. Una puntuación alta refleja una voz dinámica que usa la energía para captar la atención.")

        with st.expander("🔍 Ejemplos de mejora"):
            st.write("Hemos encontrado los siguientes ejemplos de aspectos a mejorar:")
            
            def show_improvement_example(label, timestamp, video_url):
                if timestamp is not None:
                    st.markdown(f"**{label}:**")
                    link = f"<a href='{video_url}&t={int(timestamp)}s' target='_blank'>Ver Ejemplo ({int(timestamp//60)}:{int(timestamp%60):02d})</a>"
                    st.markdown(link, unsafe_allow_html=True)

            speech_examples = speech.get("examples_to_improve", {})
            show_improvement_example("Momento de voz más monótona", speech_examples.get('monotony'), video_url)
            show_improvement_example("Ritmo muy rápido", speech_examples.get('fast_pace'), video_url)
            show_improvement_example("Ritmo muy lento", speech_examples.get('slow_pace'), video_url)
            show_improvement_example("Momento de tono excesivamente grave", speech_examples.get('low_pitch'), video_url)
            show_improvement_example("Momento de tono excesiamente agudo", speech_examples.get('high_pitch'), video_url) 

    # --- PESTAÑA 4: SCORE FINAL ---
    with tab4:
        st.header("Tu Score Global de Comunicación")
        st.info("Este es un score agregado basado en todas las métricas analizadas.")
        
        verbal_scores_list = list(verbal_scores.values()) if verbal_scores else []
        speech_scores_list = [score_pitch, score_pauses, score_volume]
        non_verbal_scores_list = [score_posture, score_mouth_open, score_gest_height, score_body_dyn, score_gest_var, score_head_tilt]

        all_scores = verbal_scores_list + non_verbal_scores_list + speech_scores_list
        valid_scores = [s for s in all_scores if isinstance(s, (int, float)) and not np.isnan(s)]
        final_score = np.mean(valid_scores) if valid_scores else 0.0
        
        st.markdown(f"<h1 style='text-align: center; color: #1E8449;'>{final_score:.1f} / 10</h1>", unsafe_allow_html=True)
        st.progress(final_score / 10)
        
        st.markdown("#### Recomendaciones Clave (Ejemplo)")
        st.markdown(f"- **Tu punto más fuerte:** {'Postura de Confianza' if final_score > 7 else 'Lenguaje de Manos'}.")
        st.markdown(f"- **Área a mejorar:** {'Dinamismo en Escena' if score_body_dyn < 6 else 'Claridad Vocal'}.")

    if st.button("Analizar otro video"):
        if 'job_id' in st.session_state: del st.session_state['job_id']
        st.rerun()

# --- LÓGICA DE FLUJO PRINCIPAL Y VALIDACIÓN ---
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

def smooth_score(score, factor=0.8):
    """
    Suaviza una puntuación para que no sea tan extrema (0 o 10).
    Mueve la puntuación hacia el 5. El 'factor' controla la intensidad.
    factor=0.0 -> no hay suavizado.
    factor=1.0 -> todas las notas son 5.
    """
    if score is None: return None
    # Interpola linealmente entre la puntuación original y el punto medio (5)
    return score * (1 - factor) + 5 * factor

def validate_url_fast(url):
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
        channel = info.get('channel', '').lower()
        title = info.get('title', '').lower()
        if "ted" in channel or "tedx" in channel or "ted talk" in title or "ted" in title or "tedx" in title: return True, "URL válida."
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
            video_url = st.session_state.get('last_url', '')
            if 'url_input' in st.session_state:
                video_url = st.session_state['url_input']
            display_results(results_data, video_url)
        elif os.path.exists(error_filepath):
            with open(error_filepath, 'r') as f: error_message = f.read()
            st.error(f"Ocurrió un error durante el análisis: {error_message}")
            if st.button("Intentar con otro video"):
                del st.session_state['job_id']
                st.rerun()
        else:
            display_processing_status()

if __name__ == "__main__":
    if 'url_input' in st.session_state and st.session_state['url_input']:
        st.session_state['last_url'] = st.session_state['url_input']
    main_flow()