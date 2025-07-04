import os
import cv2
import yt_dlp
import pandas as pd
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import logging
import math

# Configurar un logger simple para ver el progreso y errores
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FUNCIÓN 1: VALIDACIÓN Y DESCARGA ---
def download_and_validate_video(url, raw_dir="data/raw"):
    """
    Valida si la URL es de una TED Talk, descarga el video en 480p y devuelve la ruta.
    """
    logging.info(f"Iniciando validación y descarga para la URL: {url}")
    os.makedirs(raw_dir, exist_ok=True)

    ydl_opts = {
        'format': 'best[height<=480][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': os.path.join(raw_dir, '%(id)s.%(ext)s'),
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info['id']
            channel = info.get('channel', '').lower()
            title = info.get('title', '').lower()

            if "ted" not in channel and "tedx" not in channel and "ted talk" not in title:
                message = "El video no parece ser una charla TED o TEDx."
                logging.warning(message)
                return None, None, message

            logging.info(f"Video '{title}' validado. Descargando...")
            ydl.download([url])
            video_ext = info.get('ext', 'mp4')
            filepath = os.path.join(raw_dir, f"{video_id}.{video_ext}")
            
            if not os.path.exists(filepath):
                 raise FileNotFoundError(f"La descarga falló, no se encontró el archivo: {filepath}")

            logging.info(f"Descarga completa. Archivo en: {filepath}")
            return video_id, filepath, "Validación y descarga exitosa."

    except Exception as e:
        error_message = f"Error durante la descarga o validación: {e}"
        logging.error(error_message)
        return None, None, error_message

# --- FUNCIÓN 2: PROCESAMIENTO DE VIDEO EN UNA SOLA PASADA ---
def process_video_single_pass(video_path, video_id):
    """
    Procesa un archivo de video frame a frame (con muestreo), extrayendo
    landmarks de MediaPipe y análisis de emociones con DeepFace en una sola lectura.
    """
    logging.info(f"Iniciando procesamiento de video para: {video_id}")
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        model_complexity=1, 
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"No se pudo abrir el archivo de video: {video_path}")
        return pd.DataFrame(), pd.DataFrame(), None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps) if fps > 0 else 1

    landmarks_data, emotions_data, frame_count = [], [], 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Procesando {total_frames} frames con un intervalo de {frame_interval} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            
            speaker_is_present = results.pose_landmarks is not None
            row = {"frame": frame_count, "speaker_present": speaker_is_present}

            if speaker_is_present:
                for name, landmark_list in [("face", results.face_landmarks), 
                                            ("left_hand", results.left_hand_landmarks), 
                                            ("right_hand", results.right_hand_landmarks), 
                                            ("pose", results.pose_landmarks)]:
                    if landmark_list:
                        for i, lm in enumerate(landmark_list.landmark):
                            row[f"{name}_{i}_x"], row[f"{name}_{i}_y"], row[f"{name}_{i}_z"] = lm.x, lm.y, lm.z
                            if name == "pose": row[f"{name}_{i}_vis"] = lm.visibility
            
            landmarks_data.append(row)
            
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True, detector_backend='mediapipe')
                if isinstance(analysis, list) and len(analysis) > 0:
                    emotions_data.append({"frame": frame_count, "emotion": analysis[0]['dominant_emotion']})
            except Exception: pass
        
        frame_count += 1
    
    logging.info(f"Procesamiento de video completado. {len(landmarks_data)} frames analizados.")
    cap.release()
    holistic.close()
    
    return pd.DataFrame(landmarks_data), pd.DataFrame(emotions_data), fps

# --- FUNCIONES AUXILIARES PARA MÉTRICAS ---

def calculate_mar(row):
    """Calcula el Mouth Aspect Ratio para medir la apertura de la boca."""
    try:
        p_top = np.array([row["face_13_x"], row["face_13_y"]])
        p_bottom = np.array([row["face_14_x"], row["face_14_y"]])
        p_left = np.array([row["face_61_x"], row["face_61_y"]])
        p_right = np.array([row["face_291_x"], row["face_291_y"]])
        vertical_dist = np.linalg.norm(p_top - p_bottom)
        horizontal_dist = np.linalg.norm(p_left - p_right)
        return vertical_dist / horizontal_dist if horizontal_dist > 1e-6 else 0.0
    except KeyError: return np.nan

def average_wrist_y(row):
    """Calcula la altura media de las muñecas para medir la gesticulación."""
    y_coords = []
    if not pd.isna(row.get("left_hand_0_y")): y_coords.append(row["left_hand_0_y"])
    if not pd.isna(row.get("right_hand_0_y")): y_coords.append(row["right_hand_0_y"])
    return np.mean(y_coords) if y_coords else np.nan

def calculate_body_center_x(row):
    """Calcula el centro horizontal del cuerpo (entre los hombros)."""
    try:
        return (row["pose_11_x"] + row["pose_12_x"]) / 2
    except KeyError: return np.nan

def calculate_head_tilt_angle(row):
    """Calcula el ángulo de inclinación de la cabeza."""
    try:
        p_top = np.array([row["face_10_x"], row["face_10_y"]]) # Punto superior de la frente
        p_bottom = np.array([row["face_152_x"], row["face_152_y"]]) # Punta de la barbilla
        delta_x = p_top[0] - p_bottom[0]
        delta_y = p_top[1] - p_bottom[1]
        return math.degrees(math.atan2(delta_x, delta_y))
    except KeyError: return np.nan

def calculate_open_posture_ratio(row):
    """Calcula el ratio entre la anchura de hombros y la altura del torso."""
    try:
        p_shoulder_l = np.array([row["pose_11_x"], row["pose_11_y"]])
        p_shoulder_r = np.array([row["pose_12_x"], row["pose_12_y"]])
        p_hip_l = np.array([row["pose_23_x"], row["pose_23_y"]])
        p_hip_r = np.array([row["pose_24_x"], row["pose_24_y"]])
        
        shoulder_width = np.linalg.norm(p_shoulder_l - p_shoulder_r)
        
        shoulders_mid_y = (p_shoulder_l[1] + p_shoulder_r[1]) / 2
        hips_mid_y = (p_hip_l[1] + p_hip_r[1]) / 2
        torso_height = abs(shoulders_mid_y - hips_mid_y)
        
        return shoulder_width / torso_height if torso_height > 1e-6 else 0.0
    except KeyError: return np.nan

def find_noteworthy_timestamps(df, fps):
    """
    VERSIÓN FINAL: Encuentra el segmento más representativo a mejorar para
    las métricas clave de lenguaje no verbal.
    """
    if df.empty or not fps or fps <= 0:
        return {}

    examples_to_improve = {}
    
    def find_worst_segment_ts(metric_series, threshold, is_above_bad, min_duration_sec=2.0):
        # 1. SUAVIZADO: Se usa SOLO para identificar las regiones candidatas.
        # Ventana de 3 segundos (nuestro DataFrame es ~1 FPS)
        smoothed_metric = metric_series.rolling(window=3, center=True, min_periods=1).mean()

        # 2. BINARIZACIÓN: Se usa la serie suavizada para crear una máscara robusta contra el ruido.
        mask = smoothed_metric > threshold if is_above_bad else smoothed_metric < threshold
        if not mask.any(): return None
        
        # 3. IDENTIFICACIÓN DE SECUENCIAS
        sequences = mask.ne(mask.shift()).cumsum()
        problematic_groups = df[mask].groupby(sequences)
        
        worst_segment_ts = None
        max_score = -1

        for _, group in problematic_groups:
            # Duración aproximada en segundos del segmento
            duration_in_rows = len(group)
            # Factor de conversión de filas del DF a segundos reales
            rows_per_second_approx = 1 / (df['frame'].diff().mean() / fps) if len(df['frame']) > 1 else 1
            duration_in_sec = duration_in_rows / rows_per_second_approx

            if duration_in_sec >= min_duration_sec:
                start_frame = group.iloc[0]['frame']
                
                # 4. CÁLCULO DE GRAVEDAD SOBRE DATOS ORIGINALES 
                # Usamos los índices del grupo para seleccionar los datos de la serie original (no suavizada).
                original_values_in_segment = metric_series[group.index]
                severity = (original_values_in_segment - threshold).abs().mean()
                
                score = duration_in_sec * severity # Puntuamos por duración y gravedad real
                
                if score > max_score:
                    max_score = score
                    worst_segment_ts = round(start_frame / fps, 1)
        
        return worst_segment_ts


    # Manos Bajas (valor Y alto es malo)
    examples_to_improve['gesticulation_height'] = find_worst_segment_ts(df['avg_wrist_y'], threshold=0.7, is_above_bad=True, min_duration_sec=3.0)
    
    # Postura Cerrada (ratio bajo es malo)
    examples_to_improve['posture_openness'] = find_worst_segment_ts(df['open_posture_ratio'], threshold=1.8, is_above_bad=False, min_duration_sec=3.0)
    
    # Poca Inclinación de Cabeza (ángulo absoluto bajo es malo -> estático)
    df['head_tilt_abs'] = df['head_tilt_angle'].abs()
    # Para esta métrica, el problema no es que sea bajo, sino la falta de variación.
    # Así que medimos la desviación estándar en una ventana.
    df['head_tilt_std'] = df['head_tilt_abs'].rolling(window=5, min_periods=1).std().fillna(0)
    examples_to_improve['head_tilt'] = find_worst_segment_ts(df['head_tilt_std'], threshold=0.5, is_above_bad=False, min_duration_sec=4.0)
    
    return examples_to_improve

# --- FUNCIÓN 3: CÁLCULO DE MÉTRICAS FINALES ---

def calculate_final_metrics(landmarks_df, emotions_df, fps):
    """
    VERSIÓN FINAL Y ROBUSTA: Calcula métricas y ejemplos aplicando dos filtros en orden:
    1. Recorta los primeros y últimos 5 segundos del vídeo.
    2. De los datos restantes, usa solo los frames donde el ponente es visible.
    """
    logging.info("Calculando métricas finales y ejemplos...")
    if landmarks_df.empty or not fps:
        logging.warning("DataFrame de landmarks vacío o FPS no disponibles. No se pueden calcular métricas.")
        return {}

    # --- FILTRADO EN DOS ETAPAS ---

    # ETAPA 1: Recortar los primeros y últimos 5 segundos
    seconds_to_trim = 5
    total_duration = landmarks_df['frame'].max() / fps
    df_temp = landmarks_df
    emotions_temp = emotions_df

    if total_duration > (seconds_to_trim * 2):
        min_frame = seconds_to_trim * fps
        max_frame = landmarks_df['frame'].max() - (seconds_to_trim * fps)
        
        df_temp = landmarks_df[(landmarks_df['frame'] >= min_frame) & (landmarks_df['frame'] <= max_frame)]
        emotions_temp = emotions_df[emotions_df['frame'].isin(df_temp['frame'])]
        
        logging.info(f"Análisis no verbal: Se han eliminado los primeros y últimos {seconds_to_trim} segundos.")

    # ETAPA 2: Filtrar frames sin ponente de los datos ya recortados
    df_filtered = df_temp[df_temp['speaker_present'] == True].copy()

    if df_filtered.empty:
        logging.warning("No se detectó al ponente en el segmento de tiempo analizado.")
        return {"non_verbal_expression": {"scores": {}, "examples_to_improve": {}}, "emotion_analysis": {}}
    
    # Filtrar emociones correspondientes a los frames finales
    emotions_filtered = emotions_temp[emotions_temp['frame'].isin(df_filtered['frame'])]
    
    # --- FIN DEL FILTRADO ---

    # A partir de aquí, todo opera sobre el DataFrame final 'df_filtered'
    # La interpolación se aplica ahora a los datos limpios
    df = df_filtered.interpolate(method='linear', limit_direction='both', axis=0)
    
    # Calcular métricas por frame
    df["mar"] = df.apply(calculate_mar, axis=1)
    df["avg_wrist_y"] = df.apply(average_wrist_y, axis=1)
    df["body_center_x"] = df.apply(calculate_body_center_x, axis=1)
    df["head_tilt_angle"] = df.apply(calculate_head_tilt_angle, axis=1)
    df["open_posture_ratio"] = df.apply(calculate_open_posture_ratio, axis=1)

    # Calcular métricas agregadas
    mar_mean = df["mar"].mean()
    wrist_y_mean = df["avg_wrist_y"].mean()
    body_dynamism_std = df["body_center_x"].std()
    gesticulation_variability_std = df["avg_wrist_y"].std()
    head_tilt_variability_std = df["head_tilt_angle"].abs().std()
    posture_openness_avg = df["open_posture_ratio"].mean()

    # La búsqueda de ejemplos ahora se hace sobre el DataFrame ultra-limpio
    noteworthy_examples = find_noteworthy_timestamps(df, fps)
    
    # Análisis de emociones sobre el DataFrame de emociones filtrado
    if not emotions_filtered.empty:
        emotion_counts = emotions_filtered['emotion'].value_counts(normalize=True)
        dominant_emotion = emotion_counts.idxmax()
        emotion_distribution = emotion_counts.to_dict()
    else:
        dominant_emotion, emotion_distribution = "N/A", {}

    results = {
        "non_verbal_expression": {
            "scores": {
                "mouth_opening_avg": mar_mean,
                "gesticulation_height_avg": wrist_y_mean,
                "body_dynamism": body_dynamism_std,
                "gesticulation_variability": gesticulation_variability_std,
                "head_tilt_variability": head_tilt_variability_std,
                "posture_openness_avg": posture_openness_avg,
            },
            "examples_to_improve": noteworthy_examples
        },
        "emotion_analysis": {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_distribution
        }
    }
    # Limpieza de NaNs para la salida JSON
    for key, value in results["non_verbal_expression"]["scores"].items():
        if np.isnan(value):
            results["non_verbal_expression"]["scores"][key] = 0.0

    logging.info(f"Métricas no verbales calculadas sobre datos filtrados.")
    return results

# --- FUNCIÓN PRINCIPAL DEL WORKER ---
def run_full_analysis(url):
    video_id, video_path, message = download_and_validate_video(url)
    if not video_id:
        raise ValueError(message)

    try:
        landmarks_df, emotions_df, fps = process_video_single_pass(video_path, video_id)
        final_results = calculate_final_metrics(landmarks_df, emotions_df, fps)
    finally:
        logging.info(f"Limpiando archivo de video: {video_path}")
        os.remove(video_path)
    
    return final_results