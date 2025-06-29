# src/habla/speech_logic.py

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

# Encontrar ejemplos
def find_speech_examples(y, sr, f0, voiced_flag, non_silent_intervals, rms):
    """
    VERSIÓN ROBUSTA: Encuentra 2 ejemplos distintos de momentos a mejorar.
    """
    import pandas as pd
    examples_to_improve = {}
    hop_length = 512
    MIN_EXAMPLE_DISTANCE_SEC = 10.0 # Asegura que los ejemplos estén separados por al menos 10 segundos

    def get_distinct_timestamps(df, column_name, find_smallest=True, count=2):
        """Función auxiliar para obtener N timestamps distintos para una métrica."""
        if df.empty or column_name not in df.columns or df[column_name].isnull().all():
            return []
        
        # Obtenemos más candidatos de los que necesitamos para tener de dónde elegir
        candidates = df.nsmallest(count * 5, column_name) if find_smallest else df.nlargest(count * 5, column_name)
        
        selected_timestamps = []
        for _, row in candidates.iterrows():
            ts = round(librosa.frames_to_time(row['frame'], sr=sr, hop_length=hop_length), 1)
            # Comprobar si el nuevo timestamp está lo suficientemente lejos de los ya seleccionados
            is_far_enough = all(abs(ts - existing_ts) > MIN_EXAMPLE_DISTANCE_SEC for existing_ts in selected_timestamps)
            if is_far_enough:
                selected_timestamps.append(ts)
            if len(selected_timestamps) == count:
                break
        return sorted(selected_timestamps)

    # --- 1. Voz Monótona ---
    try:
        df_f0 = pd.DataFrame({'f0': f0, 'frame': np.arange(len(f0))})
        df_voiced = df_f0[voiced_flag]
        if len(df_voiced) > int(3 * sr / hop_length):
            df_voiced['f0_std'] = df_voiced['f0'].rolling(window=int(3 * sr / hop_length), center=True).std().fillna(100)
            examples_to_improve['monotony'] = get_distinct_timestamps(df_voiced, 'f0_std', find_smallest=True)
        else:
            examples_to_improve['monotony'] = []
    except Exception as e:
        print(f"Error en monotonía: {e}")
        examples_to_improve['monotony'] = []

    # --- 2. Ritmo Rápido ---
    try:
        if len(non_silent_intervals) > 2:
            speech_segments = [{'duration': (end - start), 'ts': round(start / sr, 1)} for start, end in non_silent_intervals]
            speech_segments.sort(key=lambda x: x['duration'], reverse=True)
            examples_to_improve['fast_pace'] = [s['ts'] for s in speech_segments[:2]]
        else:
            examples_to_improve['fast_pace'] = []
    except Exception:
        examples_to_improve['fast_pace'] = []

    # --- 3. Volumen Bajo y Alto ---
    try:
        df_rms = pd.DataFrame({'rms': rms, 'frame': np.arange(len(rms))})
        df_rms_voiced = df_rms[voiced_flag]
        if len(df_rms_voiced) > 5: # Necesitamos suficientes datos
            examples_to_improve['low_volume'] = get_distinct_timestamps(df_rms_voiced, 'rms', find_smallest=True)
            examples_to_improve['high_volume'] = get_distinct_timestamps(df_rms_voiced, 'rms', find_smallest=False)
        else:
            examples_to_improve['low_volume'] = []
            examples_to_improve['high_volume'] = []
    except Exception:
        examples_to_improve['low_volume'] = []
        examples_to_improve['high_volume'] = []
        
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