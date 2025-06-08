# src/habla/speech_logic.py

import os
import yt_dlp
import librosa
import numpy as np
import logging

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
            'preferredcodec': 'wav', # WAV es ideal para análisis
            'preferredquality': '192',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            # El archivo final será .wav por el post-procesador
            filepath = os.path.join(audio_dir, f"{video_id}.wav")
            
            if not os.path.exists(filepath):
                 raise FileNotFoundError(f"La descarga o conversión de audio falló: {filepath}")

            logging.info(f"Descarga de audio completa. Archivo en: {filepath}")
            return filepath

    except Exception as e:
        error_message = f"Error durante la descarga de audio: {e}"
        logging.error(error_message, exc_info=True)
        return None

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
        
        results = {
            "speech_analysis": {
                "pitch_variation": pitch_variation if not np.isnan(pitch_variation) else 0.0,
                "silence_percentage": silence_percentage if not np.isnan(silence_percentage) else 0.0,
                "volume_avg": volume_avg if not np.isnan(volume_avg) else 0.0,
                "volume_variability": volume_variability if not np.isnan(volume_variability) else 0.0,
            }
        }
        logging.info(f"Métricas de habla calculadas: {results}")
        return results

    except Exception as e:
        logging.error(f"Error analizando el archivo de audio {audio_path}: {e}", exc_info=True)
        return {}


# --- FUNCIÓN PRINCIPAL (ORQUESTADOR) ---
def run_speech_analysis(url):
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