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
            
            row = {"frame": frame_count}
            for name, landmark_list in [("face", results.face_landmarks), ("left_hand", results.left_hand_landmarks), 
                                        ("right_hand", results.right_hand_landmarks), ("pose", results.pose_landmarks)]:
                if landmark_list:
                    for i, lm in enumerate(landmark_list.landmark):
                        row[f"{name}_{i}_x"], row[f"{name}_{i}_y"], row[f"{name}_{i}_z"] = lm.x, lm.y, lm.z
                        if name == "pose": row[f"{name}_{i}_vis"] = lm.visibility
            landmarks_data.append(row)
            
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe')
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
    Analiza el DataFrame para encontrar los dos peores momentos
    para cada métrica, como ejemplos a mejorar.
    """
    if df.empty or not fps or fps <= 0:
        return {}

    examples = {}
    
    def get_two_worst_timestamps(metric_name, find_smallest=True):
        """
        Función interna para encontrar los 2 peores frames y convertirlos a timestamps.
        'find_smallest=True' significa que los valores más bajos son los peores.
        """
        try:
            if metric_name not in df.columns or df[metric_name].isnull().all():
                return []
            
            # Usar nsmallest/nlargest para encontrar los dos peores frames
            if find_smallest:
                worst_frames = df.nsmallest(2, metric_name)
            else:
                worst_frames = df.nlargest(2, metric_name)
            
            if worst_frames.empty:
                return []

            # Obtener los números de frame originales y convertir a timestamps
            timestamps = [round(row['frame'] / fps, 1) for _, row in worst_frames.iterrows()]
            return sorted(list(set(timestamps))) # Devuelve timestamps únicos y ordenados
        except (ValueError, KeyError):
            return []

    # 1. Postura (open_posture_ratio): los 2 valores más bajos (postura cerrada) son peores.
    examples['posture_openness'] = get_two_worst_timestamps('open_posture_ratio', find_smallest=True)

    # 2. Claridad Vocal (mar): los 2 valores más bajos (boca cerrada) son peores.
    examples['mouth_opening'] = get_two_worst_timestamps('mar', find_smallest=True)

    # 3. Presencia Escénica (avg_wrist_y): los 2 valores más altos (manos bajas) son peores.
    examples['gesticulation_height'] = get_two_worst_timestamps('avg_wrist_y', find_smallest=False)

    # 4. Dinamismo Corporal (body_center_x): buscamos los 2 momentos de menor movimiento.
    df['body_displacement'] = df['body_center_x'].diff().abs()
    examples['body_dynamism'] = get_two_worst_timestamps('body_displacement', find_smallest=True)

    # 5. Conexión Empática (head_tilt_angle): buscamos los 2 momentos de menor inclinación.
    df['head_tilt_abs'] = df['head_tilt_angle'].abs()
    examples['head_tilt'] = get_two_worst_timestamps('head_tilt_abs', find_smallest=True)

    return examples

# --- FUNCIÓN 3: CÁLCULO DE MÉTRICAS FINALES ---
def calculate_final_metrics(landmarks_df, emotions_df, fps):
    """
    Toma los DataFrames de landmarks y emociones y calcula las métricas finales agregadas,
    incluyendo las nuevas métricas de dinamismo corporal, inclinación de cabeza y postura.
    """
    logging.info("Calculando métricas finales y ejemplos ...")
    if landmarks_df.empty:
        logging.warning("El DataFrame de landmarks está vacío. No se pueden calcular métricas.")
        return {}

    df = landmarks_df.interpolate(method='linear', limit_direction='both', axis=0)

    # --- Calcular métricas por frame ---
    df["mar"] = df.apply(calculate_mar, axis=1)
    df["avg_wrist_y"] = df.apply(average_wrist_y, axis=1)
    df["body_center_x"] = df.apply(calculate_body_center_x, axis=1)
    df["head_tilt_angle"] = df.apply(calculate_head_tilt_angle, axis=1)
    df["open_posture_ratio"] = df.apply(calculate_open_posture_ratio, axis=1)

    # --- Calcular métricas agregadas ---
    # Métricas existentes
    mar_mean = df["mar"].mean()
    mar_std = df["mar"].std()
    wrist_y_mean = df["avg_wrist_y"].mean()
    wrist_y_std = df["avg_wrist_y"].std()
    body_dynamism = df["body_center_x"].std()
    head_tilt_variability = df["head_tilt_angle"].std()
    posture_openness_avg = df["open_posture_ratio"].mean()

    noteworthy_examples = find_noteworthy_timestamps(df.copy(), fps)
    
    # Análisis de emociones
    if not emotions_df.empty:
        emotion_counts = emotions_df['emotion'].value_counts(normalize=True)
        dominant_emotion = emotion_counts.idxmax()
        emotion_distribution = emotion_counts.to_dict()
    else:
        dominant_emotion, emotion_distribution = "N/A", {}

    results = {
        "non_verbal_expression": {
            "scores": {
                "mouth_opening_avg": mar_mean if not np.isnan(mar_mean) else 0.0,
                "mouth_opening_variability": mar_std if not np.isnan(mar_std) else 0.0,
                "gesticulation_height_avg": wrist_y_mean if not np.isnan(wrist_y_mean) else 1.0,
                "gesticulation_variability": wrist_y_std if not np.isnan(wrist_y_std) else 0.0,
                "body_dynamism": body_dynamism if not np.isnan(body_dynamism) else 0.0,
                "head_tilt_variability": head_tilt_variability if not np.isnan(head_tilt_variability) else 0.0,
                "posture_openness_avg": posture_openness_avg if not np.isnan(posture_openness_avg) else 0.0,
            },
            "examples_to_improve": noteworthy_examples
        },
        "emotion_analysis": {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_distribution
        }
    }
    logging.info(f"Métricas calculadas (con nuevas adiciones): {results}")
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