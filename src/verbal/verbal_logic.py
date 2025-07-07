# src/verbal/verbal_logic.py 

import os
import tempfile
import whisper
import yt_dlp
from transformers import pipeline, logging as hf_logging
from collections import Counter
import re
import numpy as np
import logging

# --- CONFIGURACIÓN Y CONSTANTES ---
# Reducir los logs de la librería transformers para no saturar la consola
hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - VERBAL_LOGIC - %(levelname)s - %(message)s')

# Directorio para la caché de modelos de Hugging Face
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# Listas de palabras y patrones multilingües
FILLER_WORDS = {
    'en': ['um', 'uh', 'er', 'ah', 'like', 'okay', 'right', 'so', 'you know', 'well', 'basically', 'actually', 'literally', 'i mean'],
    'es': ['eh', 'este', 'pues', 'bueno', 'o sea', 'sabes', 'en plan', 'vale', 'es que', 'a ver', 'digamos']
}

ENDING_KEYWORDS = {
    'en': ["thank you", "in conclusion", "to sum up", "my final thought", "to summarize", "last message", "takeaway"],
    'es': ["muchas gracias", "en conclusión", "para resumir", "para concluir", "mi última reflexión", "y con esto termino"]
}

RHETORICAL_PATTERNS = {
    'en': r'\b(like|as if|as though|as)\b',
    'es': r'\b(como si|tal como|así como)\b'
}

# --- FUNCIONES DE ANÁLISIS ---

def download_audio(video_url):
    """Descarga el audio de una URL de YouTube y lo guarda como .wav en un directorio temporal."""
    logging.info(f"Iniciando descarga de audio para: {video_url}")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        audio_path = os.path.join(tempfile.gettempdir(), f"{info['id']}.wav")
        logging.info(f"Audio descargado en: {audio_path}")
        return audio_path

def transcribe_audio(audio_path):
    """Transcribe el audio usando Whisper, detectando el idioma y devolviendo timestamps por palabra."""
    logging.info("Iniciando transcripción ...")
    model = whisper.load_model("medium")
    
    result = model.transcribe(audio_path, word_timestamps=True)
    
    detected_language = result.get('language')
    full_text = result.get('text', '')
    word_data = [word for segment in result.get('segments', []) for word in segment.get('words', [])]
            
    logging.info(f"Transcripción completada. Idioma detectado: {detected_language}")
    return full_text, word_data, detected_language

def summarize_text(text, language):
    """Genera un resumen del texto, seleccionando el modelo apropiado según el idioma."""
    if language == 'es':
        model_name = "Milos/t5-base-spanish-summarization"
        logging.info(f"Usando modelo de resumen para español: {model_name}")
    elif language == 'en':
        model_name = "facebook/bart-large-cnn"
        logging.info(f"Usando modelo de resumen para inglés: {model_name}")
    else:
        logging.warning(f"No hay un modelo de resumen optimizado para '{language}'.")
        return f"El resumen detallado no está disponible para el idioma '{language}'."
    
    try:
        summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
       
       # Crear "resúmenes intermedios" de todo el texto
        max_chunk_length_chars = 4000 
        chunks = [text[i:i+max_chunk_length_chars] for i in range(0, len(text), max_chunk_length_chars)]
        
        intermediate_summaries = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
        intermediate_text = " ".join([s['summary_text'] for s in intermediate_summaries])
        
        # Si el texto intermedio ya es corto, lo devolvemos
        if len(intermediate_text) < max_chunk_length_chars:
            return intermediate_text

        # Crear un resumen final a partir del texto de resúmenes intermedios
        final_summary = summarizer(intermediate_text, max_length=250, min_length=60, do_sample=False)

        return final_summary[0]['summary_text']
    
    except Exception as e:
        logging.error(f"Error durante el resumen con el modelo {model_name}: {e}", exc_info=True)
        return "El resumen no pudo ser generado debido a un error técnico."

def extract_keywords(text, base_path, language):
    """Extrae las palabras clave del texto, usando el archivo de stopwords del idioma correspondiente."""
    stopwords_path = os.path.join(base_path, f'stopwords_{language}.txt')
    stopwords = set()
    
    if os.path.exists(stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set(word.strip().lower() for word in f.readlines())
        logging.info(f"Stopwords para '{language}' cargadas correctamente.")
    else:
        logging.warning(f"No se encontró el archivo de stopwords: {stopwords_path}. El análisis de palabras clave puede ser menos preciso.")
            
    words = [w.strip('.,!"()?¿¡').lower() for w in text.split() if w.lower() not in stopwords and len(w) > 3]
    freq = Counter(words)
    return [word for word, _ in freq.most_common(10)]

# --- FUNCIONES DE PUNTUACIÓN ---

def clarity_score(text):
    """Puntúa la claridad del mensaje basándose en la longitud media de las frases."""
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip().split()) > 3]
    if not sentences: return 5.0 
    
    avg_len = np.mean([len(s.split()) for s in sentences])
    # Puntuación Gaussiana: ideal en 15 palabras por frase.
    score = 10 * np.exp(-0.5 * ((avg_len - 15) / 8) ** 2)
    return score

def structure_score(text):
    """Puntúa la estructura basándose en el número de párrafos."""
    paragraphs = [p for p in text.splitlines() if len(p.strip().split()) > 15]
    num_paragraphs = len(paragraphs)
    # Puntuación logística: ideal cerca de 10-15 párrafos.
    score = 10 * (1 / (1 + np.exp(-0.3 * (num_paragraphs - 10))))
    return score

def lexical_diversity_score(text):
    """Puntúa la riqueza del vocabulario (ratio de palabras únicas)."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words: return 0.0
    diversity = len(set(words)) / len(words)
    # Puntuación logística: un ratio de 0.5 es excelente.
    score = 10 * (1 / (1 + np.exp(-15 * (diversity - 0.4))))
    return score

def rhetoric_score(text, language):
    """Puntúa el uso de figuras retóricas."""
    pattern = RHETORICAL_PATTERNS.get(language, '')
    questions = text.count('?')
    comparisons = len(re.findall(pattern, text.lower())) if pattern else 0
    
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 1]
    anaphora_count = 0
    if len(sentences) > 1:
        for i in range(len(sentences) - 1):
            first_phrase = " ".join(sentences[i].split()[:3])
            second_phrase = " ".join(sentences[i+1].split()[:3])
            if len(first_phrase.split()) > 1 and first_phrase.lower() == second_phrase.lower():
                anaphora_count += 1
                
    total_score = (anaphora_count * 3) + (questions * 1) + (comparisons * 0.5)
    # Puntuación logística: centrada en una puntuación ponderada de 8.
    score = 10 * (1 / (1 + np.exp(-0.3 * (total_score - 8))))
    return score

def ending_strength_score(text, language):
    """Puntúa la fuerza del cierre del discurso."""
    keywords = ENDING_KEYWORDS.get(language, [])
    closing_len = max(300, int(len(text) * 0.15)) # Último 15% del texto
    closing = text[-closing_len:].lower()
    return 9.5 if any(k in closing for k in keywords) else 3.0

def filler_words_score(word_data, language):
    """Puntúa el uso de muletillas. Una puntuación alta significa pocas muletillas."""
    filler_analysis = analyze_filler_words(word_data, language)
    total_words = len(word_data) if word_data else 1
    filler_ratio = (filler_analysis['total_count'] / total_words) * 100 # Muletillas por 100 palabras
    # Puntuación exponencial: cae rápidamente con más muletillas.
    score = 10 * np.exp(-0.15 * filler_ratio)
    return score, filler_analysis

# --- FUNCIONES DE FEEDBACK CUALITATIVO ---

def analyze_filler_words(word_data, language):
    """Encuentra y cuenta las muletillas, devolviendo estadísticas y un ejemplo."""
    filler_list = FILLER_WORDS.get(language, [])
    if not filler_list: return {"total_count": 0, "most_common": None, "example_timestamp": None, "distribution": {}}

    filler_counts = Counter()
    example_timestamps = {}
    for word_info in word_data:
        normalized_word = re.sub(r'[^\w\s]', '', word_info.get('word', '')).lower()
        if normalized_word in filler_list:
            filler_counts[normalized_word] += 1
            if normalized_word not in example_timestamps:
                example_timestamps[normalized_word] = round(word_info.get('start', 0), 1)
    
    if not filler_counts: return {"total_count": 0, "most_common": "Ninguna", "example_timestamp": None, "distribution": {}}
    
    most_common_filler, _ = filler_counts.most_common(1)[0]
    return {
        "total_count": sum(filler_counts.values()),
        "most_common": most_common_filler.capitalize(),
        "example_timestamp": example_timestamps.get(most_common_filler),
        "distribution": dict(filler_counts)
    }

def extract_key_sentences(full_text, word_data, keywords):
    """Extrae una oración clara y potente, y una potencialmente confusa."""
    if not word_data: return {"best_sentence": None, "confusing_sentence": None}

    # Reconstruir frases con sus timestamps de inicio
    sentences_with_ts = []
    current_sentence_text = []
    current_start_ts = word_data[0]['start']
    for word_info in word_data:
        word = word_info.get('word', '')
        current_sentence_text.append(word)
        if word.strip().endswith(('.', '?', '!')):
            sentences_with_ts.append({
                "text": "".join(current_sentence_text).strip(),
                "timestamp": current_start_ts
            })
            current_sentence_text = []
            # El timestamp de la siguiente frase es el 'start' de la palabra actual.
            current_start_ts = word_info.get('start')

    if not sentences_with_ts: return {"best_sentence": None, "confusing_sentence": None}

    # Encontrar la mejor frase (corta, con palabras clave)
    best_sentence_info, max_score = None, -float('inf')
    for sentence in sentences_with_ts:
        text_lower, word_count = sentence['text'].lower(), len(sentence['text'].split())
        score = sum(1 for kw in keywords if kw in text_lower) - (abs(word_count - 15) / 5)
        if 8 < word_count < 25 and score > max_score:
            max_score = score
            best_sentence_info = {"text": sentence['text'], "timestamp": round(sentence['timestamp'], 1)}

    # Encontrar la frase más larga como potencialmente confusa
    longest = max(sentences_with_ts, key=lambda s: len(s['text'].split()))
    confusing_sentence_info = {"text": longest['text'], "timestamp": round(longest['timestamp'], 1)}

    return {"best_sentence": best_sentence_info, "confusing_sentence": confusing_sentence_info}


# --- ORQUESTADOR PRINCIPAL ---

def run_verbal_analysis(video_url):
    """
    Función principal que orquesta todo el proceso de análisis verbal:
    descarga, transcripción, análisis de contenido y puntuación.
    """
    audio_path = None
    try:
        # 1. Descarga y Transcripción
        audio_path = download_audio(video_url)
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
            raise ValueError(f"El archivo de audio parece vacío o corrupto: {audio_path}")

        full_text, word_data, language = transcribe_audio(audio_path)
        if not full_text or not language:
            raise ValueError("La transcripción falló o no detectó el idioma. El video podría no tener audio.")

        # 2. Análisis de Contenido (ahora es multilingüe)
        logging.info(f"Iniciando análisis de contenido para el idioma: '{language}'")
        summary = summarize_text(full_text, language)
        stopwords_base_path = os.path.dirname(__file__)
        keywords = extract_keywords(full_text, stopwords_base_path, language)
        key_sentences = extract_key_sentences(full_text, word_data, keywords)
        
        # 3. Cálculo de Puntuaciones
        score_fillers, filler_analysis = filler_words_score(word_data, language)
        
        verbal_scores = {
            "message_clarity": clarity_score(full_text),
            "structure": structure_score(full_text),
            "lexical_diversity": lexical_diversity_score(full_text),
            "rhetoric": rhetoric_score(full_text, language),
            "ending": ending_strength_score(full_text, language),
            "filler_words_usage": score_fillers
        }

        # 4. Ensamblaje del Resultado Final
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
        logging.error(f"ERROR CRÍTICO en run_verbal_analysis: {e}", exc_info=True)
        return {"verbal_analysis": {"error": f"El análisis verbal ha fallado: {e}"}}
    finally:
        if audio_path and os.path.exists(audio_path):
            logging.info(f"Limpiando archivo de audio temporal: {audio_path}")
            os.remove(audio_path)