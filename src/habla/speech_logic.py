# ---- speech_logic.py -----

import os
import yt_dlp
import librosa
import numpy as np
import logging
import pandas as pd

# Configurar un logger simple para este módulo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SPEECH_LOGIC - %(levelname)s - %(message)s')

# --- FUNCIÓN 1: DESCARGA DE AUDIO ---
def download_audio_only(url, audio_dir="data/audio_raw"):
    """
    Descarga únicamente la pista de audio de una URL de YouTube y la convierte a formato .wav.
    """
    logging.info(f"Iniciando descarga de audio para la URL: {url}")
    os.makedirs(audio_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(audio_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav', 
            'preferredquality': '192',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            filepath = os.path.join(audio_dir, f"{video_id}.wav")
            
            if not os.path.exists(filepath):
                 raise FileNotFoundError(f"La descarga o conversión de audio falló: {filepath}")

            logging.info(f"Descarga de audio completa. Archivo en: {filepath}")
            return filepath

    except Exception as e:
        error_message = f"Error durante la descarga de audio: {e}"
        logging.error(error_message, exc_info=True)
        return None

def find_speech_examples(y, sr, f0, voiced_flag, non_silent_intervals, rms):
    examples_to_improve = {}
    hop_length = 512
    import pandas as pd

    # Crear un DataFrame base para el análisis
    df = pd.DataFrame({'f0': f0, 'rms': rms, 'frame': np.arange(len(f0))})
    df_voiced = df[voiced_flag].copy()

    if df_voiced.empty:
        return {}

    def find_worst_segment_ts(metric_series, threshold, is_above_bad, min_duration_sec=2.0):
        """
        Función auxiliar para encontrar el segmento más problemático.
        Usa suavizado para detectar y datos originales para evaluar.
        """
        if metric_series.empty or metric_series.isnull().all(): return None

        # 1. Suavizado para detección robusta
        window_size = int(1.5 * sr / hop_length)
        smoothed_metric = metric_series.rolling(window=window_size, center=True, min_periods=1).mean()

        # 2. Binarización con la serie suavizada
        mask = smoothed_metric > threshold if is_above_bad else smoothed_metric < threshold
        if not mask.any(): return None
        
        # 3. Identificación de secuencias
        sequences = mask.ne(mask.shift()).cumsum()
        problematic_groups = metric_series[mask].groupby(sequences)
        
        worst_segment_ts = None
        max_score = -1

        for _, group in problematic_groups:
            duration = len(group) * hop_length / sr
            if duration >= min_duration_sec:
                start_frame_index = group.index[0]
                
                # 4. Gravedad calculada sobre los datos ORIGINALES del segmento
                severity = (group - threshold).abs().mean()
                score = duration * severity
                
                if score > max_score:
                    max_score = score
                    start_frame_num = df.loc[start_frame_index, 'frame']
                    worst_segment_ts = round(librosa.frames_to_time(int(start_frame_num), sr=sr, hop_length=hop_length), 1)
        
        return worst_segment_ts

    # --- 1. Voz Monótona ---
    try:
        if len(df_voiced) > int(3 * sr / hop_length):
            # Métrica: Desviación estándar del tono en ventanas de 3 segundos
            f0_std_series = df_voiced['f0'].rolling(window=int(3 * sr / hop_length), center=True).std()
            # Umbral: Por debajo de 20 Hz se considera monótono
            examples_to_improve['monotony'] = find_worst_segment_ts(f0_std_series, threshold=20.0, is_above_bad=False, min_duration_sec=3.0)
    except Exception: pass

    # --- 2. Ritmo Rápido y Lento (La lógica original ya es correcta, pues trata con segmentos) ---
    try:
        if non_silent_intervals.size > 0:
            longest_speech = max(non_silent_intervals, key=lambda i: i[1] - i[0])
            examples_to_improve['fast_pace'] = round(longest_speech[0] / sr, 1)
        if len(non_silent_intervals) > 1:
            pauses = [(non_silent_intervals[i][0] - non_silent_intervals[i-1][1]) for i in range(1, len(non_silent_intervals))]
            if pauses and max(pauses) > (sr * 2.5):
                longest_pause_idx = np.argmax(pauses)
                pause_start_sample = non_silent_intervals[longest_pause_idx][1]
                examples_to_improve['slow_pace'] = round(pause_start_sample / sr, 1)
    except Exception: pass
        
    # --- 3. Tono Inadecuado (Aplicando la lógica de segmentos) ---
    try:
        mean_f0 = df_voiced['f0'].mean()
        if pd.notna(mean_f0):
            # Umbrales basados en la media del propio orador
            high_pitch_threshold = mean_f0 * 1.40 
            low_pitch_threshold = mean_f0 * 0.60
            
            # Buscamos segmentos donde el tono es consistentemente muy agudo o muy grave
            examples_to_improve['high_pitch'] = find_worst_segment_ts(df_voiced['f0'], high_pitch_threshold, is_above_bad=True)
            examples_to_improve['low_pitch'] = find_worst_segment_ts(df_voiced['f0'], low_pitch_threshold, is_above_bad=False)
    except Exception: pass

    return examples_to_improve

# --- FUNCIÓN 2: CÁLCULO DE MÉTRICAS DEL HABLA ---
def calculate_speech_metrics(audio_path):
    """
    Analiza un archivo de audio para extraer métricas de prosodia, pausas y energía.
    """
    logging.info(f"Iniciando análisis de habla para: {audio_path}")
    if not audio_path or not os.path.exists(audio_path):
        return {}

    try:
        # Cargar el audio
        y, sr = librosa.load(audio_path, sr=None) # sr=None para mantener el sample rate original
        seconds_to_trim = 5
        duration_total = librosa.get_duration(y=y, sr=sr)
        if duration_total > (seconds_to_trim * 2):
            start_sample = sr * seconds_to_trim
            end_sample = len(y) - (sr * seconds_to_trim)
            y = y[start_sample:end_sample]
            logging.info(f"Audio recortado. Se han eliminado los primeros y últimos {seconds_to_trim} segundos.")
        
        total_duration = librosa.get_duration(y=y, sr=sr)

        # 1. Variación Tonal (Prosodia)
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        voiced_f0 = f0[voiced_flag] # Quedarse solo con las partes donde hay voz
        pitch_variation = np.std(voiced_f0) if len(voiced_f0) > 0 else 0.0

        # 2. Uso de Pausas (Porcentaje de Silencio)
        # top_db=40 significa que cualquier cosa 40dB por debajo del máximo es silencio
        non_silent_intervals = librosa.effects.split(y, top_db=40)
        speech_duration = sum([end - start for start, end in non_silent_intervals]) / sr
        silence_duration = total_duration - speech_duration
        silence_percentage = (silence_duration / total_duration) * 100 if total_duration > 0 else 0.0
        
        # 3. Energía Vocal (Volumen y Dinamismo)
        rms = librosa.feature.rms(y=y)[0]
        volume_avg = np.mean(rms)
        volume_variability = np.std(rms)

        noteworthy_examples = find_speech_examples(y, sr, f0, voiced_flag, non_silent_intervals, rms)
        
        for category in noteworthy_examples:
            if noteworthy_examples[category] is not None:
                noteworthy_examples[category] += seconds_to_trim


        results = {
            "speech_analysis": {
                "scores":{
                    "pitch_variation": pitch_variation if not np.isnan(pitch_variation) else 0.0,
                    "silence_percentage": silence_percentage if not np.isnan(silence_percentage) else 0.0,
                    "volume_avg": volume_avg if not np.isnan(volume_avg) else 0.0,
                    "volume_variability": volume_variability if not np.isnan(volume_variability) else 0.0,
                },
                "examples_to_improve": noteworthy_examples
            }
        }
        logging.info(f"Métricas de habla calculadas: {results}")
        return results

    except Exception as e:
        logging.error(f"Error analizando el archivo de audio {audio_path}: {e}", exc_info=True)
        return {}


# --- FUNCIÓN PRINCIPAL (ORQUESTADOR) ---
def run_prosody_analysis(url):
    """
    Orquesta todo el proceso: descarga audio, lo analiza y limpia el archivo.
    """
    audio_path = download_audio_only(url)
    if not audio_path:
        raise ValueError("La descarga del audio falló.")

    try:
        final_results = calculate_speech_metrics(audio_path)
    finally:
        # Asegurarse de que el audio se borra incluso si el análisis falla
        logging.info(f"Limpiando archivo de audio: {audio_path}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    return final_results