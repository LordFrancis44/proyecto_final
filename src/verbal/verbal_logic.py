import os
import tempfile
#import torch
import whisper
import yt_dlp
from transformers import pipeline
from urllib.parse import urlparse, parse_qs
from collections import Counter
import re
import numpy as np
import traceback
import logging

RHETORICAL_PATTERNS = {
    'en': r'\b(like|as if|as though|as)\b'
}

ENDING_KEYWORDS = {
    'en': ["thank you", "final thought", "remember", "last message", "takeaway", "to sum up", "in conclusion", "to summaraize"]
}

FILLER_WORDS = {
    'en': ['um', 'uh', 'er', 'ah', 'like', 'okay', 'right', 'so', 'you know', 'well', 'basically', 'actually', 'literally', 'I mean']
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Utilidades ---
def download_audio(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        # COMENTADO: Es mejor no fijar la ruta de ffmpeg si está en el PATH del sistema.
        # 'ffmpeg_location': '/opt/homebrew/bin/ffmpeg',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Unificado a WAV
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        # La extensión ahora es .wav
        audio_path = os.path.join(tempfile.gettempdir(), f"{info['id']}.wav")
        return audio_path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    detected_language = result.get('language')
    full_text = result.get('text', '')
    
    word_data = [word for segment in result.get('segments', []) for word in segment.get('words', [])]
            
    return full_text, word_data, detected_language

def summarize_text(text, language):
    # Nos centramos en la máxima calidad para inglés
    if language != 'en':
        return "El resumen detallado solo está optimizado para discursos en inglés."
    
    model_name = "facebook/bart-large-cnn"
    
    try:
        summarizer = pipeline("summarization", model=model_name)
        
        # ETAPA 1: Crear "resúmenes intermedios" de todo el texto
        max_chunk_length_chars = 4000 # Unos 800-900 palabras para estar seguros
        chunks = [text[i:i+max_chunk_length_chars] for i in range(0, len(text), max_chunk_length_chars)]
        
        intermediate_summaries = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
        intermediate_text = " ".join([s['summary_text'] for s in intermediate_summaries])
        
        # Si el texto intermedio ya es corto, lo devolvemos
        if len(intermediate_text) < max_chunk_length_chars:
            return intermediate_text

        # ETAPA 2: Crear un resumen final a partir del texto de resúmenes intermedios
        final_summary = summarizer(intermediate_text, max_length=250, min_length=60, do_sample=False)
        
        return final_summary[0]['summary_text']

    except Exception as e:
        logging.error(f"Error durante el resumen avanzado: {e}", exc_info=True)
        return "El resumen no pudo ser generado debido a un error técnico."

def extract_keywords(text, base_path, language):
    stopwords_path = os.path.join(base_path, f'stopwords_en.txt')
    stopwords = set()
    if os.path.exists(stopwords_path):
        # Añadida codificación utf-8 para compatibilidad
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set(word.strip().lower() for word in f.readlines())
            
    words = [w.strip('.,!"()?¿¡').lower() for w in text.split() if w.lower() not in stopwords and len(w) > 3]
    freq = Counter(words)
    return [word for word, _ in freq.most_common(10)]

def lexical_diversity(text):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)

def count_rhetorical_devices(text, language):
    """
    MODIFICADO: Detecta símiles, preguntas y ANÁFORAS.
    Usa una puntuación no lineal para una evaluación más realista.
    """
    # 1. Conteo de figuras
    pattern = RHETORICAL_PATTERNS.get(language, '')
    questions = text.count('?')
    comparisons = len(re.findall(pattern, text.lower())) if pattern else 0
    
    # Detección de Anáforas (repetición al inicio de frases)
    anaphora_count = 0
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 1]
    if len(sentences) > 1:
        for i in range(len(sentences) - 1):
            # Comparamos las primeras 2-3 palabras
            first_phrase = " ".join(sentences[i].split()[:3])
            second_phrase = " ".join(sentences[i+1].split()[:3])
            if len(first_phrase.split()) > 1 and first_phrase.lower() == second_phrase.lower():
                anaphora_count += 1

    total_devices = anaphora_count + questions + comparisons
    
    # 2. Puntuación no lineal (sigmoide)
    # Centrado en 5 dispositivos para una buena nota. 'k=0.3' da una curva suave.
    score = 1 / (1 + np.exp(-0.3 * (total_devices - 5)))
    
    return score

def estimate_structure_score(text):
    """
    MODIFICADO: Usa una puntuación no lineal para la estructura.
    """
    segments = [s for s in re.split(r'\n\n+|\.|\?|!', text) if len(s.strip()) > 20]
    num_segments = len(segments)
    
    # Puntuación no lineal centrada en 15 segmentos/ideas para una nota alta.
    score = 1 / (1 + np.exp(-0.25 * (num_segments - 15)))
    return score

def ending_strength(text, language):
    keywords = ENDING_KEYWORDS.get(language, [])
    closing = text[-500:].lower()
    return float(any(k in closing for k in keywords))

def clarity_score(text):
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 5]
    if not sentences: return 1.0 # Si no hay frases, la claridad es máxima por defecto
    
    word_counts = [len(s.split()) for s in sentences]
    avg_len = np.mean(word_counts)
    std_len = np.std(word_counts)
    
    # Penalización por longitud media. Ideal en 15, penaliza más allá.
    len_score = np.exp(-0.01 * (max(0, avg_len - 15))**2)
    
    # Penalización por inconsistencia. Poca variación es bueno.
    consistency_score = np.exp(-0.01 * std_len**2)
    
    # La puntuación final de claridad es una media ponderada
    return (len_score * 0.7 + consistency_score * 0.3)

def rhetoric_score(text, language):
    pattern = RHETORICAL_PATTERNS.get(language, '')
    
    questions = text.count('?')
    comparisons = len(re.findall(pattern, text.lower())) if pattern else 0
    
    # Detección de Anáfora (repetición al inicio de frases)
    anaphora_count = 0
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 1]
    if len(sentences) > 1:
        for i in range(len(sentences) - 1):
            first_phrase = " ".join(sentences[i].split()[:3])
            second_phrase = " ".join(sentences[i+1].split()[:3])
            if len(first_phrase.split()) > 1 and first_phrase.lower() == second_phrase.lower():
                anaphora_count += 1
                
    # Ponderamos la importancia: las anáforas son más potentes.
    total_score = (anaphora_count * 3) + (questions * 1) + (comparisons * 0.5)
    
    # Normalización logística, centrada en una puntuación ponderada de 10
    return 1 / (1 + np.exp(-0.2 * (total_score - 10)))

def filler_words_score(word_data):
    # ... (la lógica de `analyze_filler_words` es la misma)
    filler_analysis = analyze_filler_words(word_data, 'en') # Forzamos inglés
    total_words = len(word_data) if word_data else 1
    
    # La puntuación ahora es más sensible al principio.
    # 1 muletilla cada 100 palabras (~1% ratio) ya baja la nota a 8.
    filler_ratio = (filler_analysis['total_count'] / total_words) * 100
    score = 10 * np.exp(-0.15 * filler_ratio)
    
    return score, filler_analysis

def structure_score(text):
    # Contar párrafos (separados por saltos de línea dobles)
    paragraphs = [p for p in text.split('\n\n') if len(p.strip().split()) > 20]
    num_paragraphs = len(paragraphs)
    
    # Puntuación logística. Ideal alrededor de 12 párrafos para un discurso de 15-20 min.
    return 1 / (1 + np.exp(-0.3 * (num_paragraphs - 12)))

def analyze_filler_words(word_data, language):
    filler_list = FILLER_WORDS.get(language, [])
    if not filler_list:
        return {"total_count": 0, "most_common": None, "example_timestamp": None, "distribution": {}}

    filler_counts = Counter()
    example_timestamps = {}

    for word_info in word_data:
        normalized_word = re.sub(r'[^\w\s]', '', word_info.get('word', '')).lower()
        if normalized_word in filler_list:
            filler_counts[normalized_word] += 1
            if normalized_word not in example_timestamps:
                example_timestamps[normalized_word] = round(word_info.get('start', 0), 1)
    if not filler_counts:
        return {"total_count": 0, "most_common": None, "example_timestamp": None, "distribution": {}}
    most_common_filler, _ = filler_counts.most_common(1)[0]
    return {"total_count": sum(filler_counts.values()), "most_common": most_common_filler.capitalize(), 
            "example_timestamp": example_timestamps.get(most_common_filler), "distribution": dict(filler_counts)}

def extract_key_sentences(full_text, word_data, keywords):
    """
    Extrae ejemplos de una oración clara y potente, y una potencialmente confusa.
    """
    if not word_data: return {"best_sentence": None, "confusing_sentence": None}
    sentences_with_ts = []
    current_sentence = {'text': '', 'start_ts': word_data[0]['start']}
    for i, word_info in enumerate(word_data):
        current_sentence['text'] += word_info.get('word', '') + ' '
        if word_info.get('word', '').strip().endswith(('.', '?', '!')):
            sentences_with_ts.append(current_sentence)
            if i + 1 < len(word_data):
                current_sentence = {'text': '', 'start_ts': word_data[i+1]['start']}
            else:
                current_sentence = None; break
    if current_sentence: sentences_with_ts.append(current_sentence)
    if not sentences_with_ts: return {"best_sentence": None, "confusing_sentence": None}

    best_sentence_info = None
    max_score = -float('inf')
    for sentence in sentences_with_ts:
        text_lower = sentence['text'].lower()
        word_count = len(text_lower.split())
        score = sum(1 for kw in keywords if kw in text_lower) - (word_count / 10)
        if 5 < word_count < 30 and score > max_score:
            max_score = score
            best_sentence_info = {"text": sentence['text'].strip(), "timestamp": round(sentence['start_ts'], 1)}
            
    longest = max(sentences_with_ts, key=lambda s: len(s['text'].split()))
    confusing_sentence_info = {"text": longest['text'].strip(), "timestamp": round(longest['start_ts'], 1)}

    return {"best_sentence": best_sentence_info, "confusing_sentence": confusing_sentence_info}


def smooth_score(score, factor=0.1):
    if score is None: return None
    return score * (1 - factor) + 5 * factor

def run_speech_analysis(video_url):
    audio_path = None
    try:
        # 1. Descarga
        audio_path = download_audio(video_url)
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
            raise ValueError(f"El archivo de audio parece vacío o corrupto: {audio_path}")

        # 2. Transcripción
        full_text, word_data, language = transcribe_audio(audio_path)

        if not full_text or not language:
            raise ValueError("La transcripción falló o no detectó el idioma.")

        language = 'en'

        # 3. Análisis de Contenido (ahora la lógica es secuencial y clara)
        summary = summarize_text(full_text, language)
        stopwords_base_path = os.path.dirname(__file__)
        keywords = extract_keywords(full_text, stopwords_base_path, language)
        filler_analysis = analyze_filler_words(word_data, language)
        key_sentences = extract_key_sentences(full_text, word_data, keywords)
        
        # --- Cálculo de Puntuaciones ---
        total_words = len(word_data)
        filler_ratio = (filler_analysis['total_count'] / total_words) * 100 if total_words > 0 else 0
        filler_score = 10 * np.exp(-0.2 * filler_ratio)
        
        verbal_scores = {
            "message_clarity": round(smooth_score(clarity_score(full_text) * 10), 2),
            "structure": round(smooth_score(structure_score(full_text) * 10), 2),
            "lexical_diversity": round(smooth_score(lexical_diversity(full_text) * 10), 2),
            "rhetoric": round(smooth_score(rhetoric_score(full_text, language) * 10), 2),
            "ending": round(smooth_score(ending_strength(full_text, language) * 10), 2),
            "filler_words_usage": round(smooth_score(filler_score), 2)
        }

        # --- Ensamblaje del Resultado Final ---
        return {
            "verbal_analysis": {
                "summary": summary,
                "keywords": keywords,
                "scores": verbal_scores,
                "qualitative_feedback": {
                    "filler_words": filler_analysis,
                    "key_sentences": key_sentences
                },
                "detected_language": language
            }
        }
    except Exception as e:
        logging.error(f"ERROR en run_speech_analysis: {e}", exc_info=True)
        return {"verbal_analysis": {"error": str(e)}}
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)