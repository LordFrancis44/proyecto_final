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

RHETORICAL_PATTERNS = {
    'en': r'\b(like|as if|as though|as)\b',
    'es': r'\b(como si|tal como|es como|así como|como)\b'
}

ENDING_KEYWORDS = {
    'en': ["thank you", "final thought", "remember", "last message", "takeaway", "to sum up", "in conclusion"],
    'es': ["gracias", "muchas gracias", "para concluir", "en resumen", "reflexión final", "para finalizar", "en conclusión", "para terminar"]
}

FILLER_WORDS = {
    'en': ['um', 'uh', 'er', 'ah', 'like', 'okay', 'right', 'so', 'you know', 'well', 'basically', 'actually', 'literally'],
    'es': ['eh', 'este', 'bueno', 'o sea', 'en plan', 'pues', 'vale', 'a ver', 'digamos', 'en fin', 'esto']
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Utilidades ---
def download_audio(video_url):
    """
    MODIFICADO: Descarga el audio en formato WAV para unificar con speech_logic.
    """
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
    # Sin cambios, ya estaba bien.
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    detected_language = result.get('language')
    full_text = result.get('text', '')
    word_data = []
    for segment in result.get('segments', []):
        for word_info in segment.get('words', []):
            word_data.append(word_info)
    return full_text, word_data, detected_language

def summarize_text(text, language):
    # Sin cambios, ya estaba bien.
    if language == 'en':
        model_name = "facebook/bart-large-cnn"
    else:
        model_name = "sshleifer/distilbart-cnn-12-6"
    try:
        summarizer = pipeline("summarization", model=model_name)
        max_chunk_length = 1024
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        summarized_chunks = summarizer(chunks[:5], max_length=150, min_length=40, do_sample=False)
        return " ".join([s['summary_text'] for s in summarized_chunks])
    except Exception as e:
        print(f"Error durante el resumen: {e}")
        return "El resumen no pudo ser generado para este idioma o texto."

def extract_keywords(text, base_path, language):
    """
    CORREGIDO: Construye la ruta al archivo de stopwords correctamente.
    """
    stopwords_path = os.path.join(base_path, f'stopwords_{language}.txt')
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
    pattern = RHETORICAL_PATTERNS.get(language, '')
    repetitions = len(re.findall(r'\b(\w+)\b(?:\W+\1\b)+', text.lower()))
    questions = text.count('?')
    comparison = len(re.findall(pattern, text.lower())) if pattern else 0
    return repetitions + questions + comparison

def estimate_structure_score(text):
    segments = [s for s in re.split(r'\n|\.|\?|!', text) if len(s.strip()) > 10]
    return min(len(segments) / 10, 1.0)

def ending_strength(text, language):
    keywords = ENDING_KEYWORDS.get(language, [])
    closing = text[-500:].lower()
    return float(any(k in closing for k in keywords))

def clarity_score(text):
    sentences = [s for s in text.split('.') if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    return max(0.0, min(1.0, 1 - (avg_sentence_length - 10) / 20))

def analyze_filler_words(word_data, language):
    """
    Cuenta la frecuencia de las muletillas y encuentra la más común con un timestamp de ejemplo.
    """
    filler_list = FILLER_WORDS.get(language, [])
    if not filler_list:
        return {"total_count": 0, "most_common": None, "example_timestamp": None, "distribution": {}}

    filler_counts = Counter()
    example_timestamps = {}

    for word_info in word_data:
        # Normalizamos la palabra para la comparación
        word = word_info['word'].strip('.,!?').lower()
        if word in filler_list:
            filler_counts[word] += 1
            # Guardamos el timestamp de la primera vez que vemos cada muletilla
            if word not in example_timestamps:
                example_timestamps[word] = round(word_info['start'], 1)
    
    total_fillers = sum(filler_counts.values())
    
    if not filler_counts:
        return {"total_count": 0, "most_common": None, "example_timestamp": None, "distribution": {}}
        
    most_common_filler, _ = filler_counts.most_common(1)[0]
    
    return {
        "total_count": total_fillers,
        "most_common": most_common_filler,
        "example_timestamp": example_timestamps.get(most_common_filler),
        "distribution": dict(filler_counts)
    }

def extract_key_sentences(full_text, word_data, keywords, filler_analysis):
    """
    Extrae ejemplos de una oración clara y potente, y una potencialmente confusa.
    """
    sentences_with_ts = []
    # Primero, dividimos el texto en frases y les asignamos un timestamp de inicio
    current_ts = 0
    for word_info in word_data:
        if not sentences_with_ts:
            current_ts = round(word_info['start'], 1)
            sentences_with_ts.append({'text': '', 'start_ts': current_ts})
        
        sentences_with_ts[-1]['text'] += word_info['word'] + ' '
        
        if '.' in word_info['word'] or '?' in word_info['word'] or '!' in word_info['word']:
             # Preparamos para la siguiente oración
            next_word_index = word_data.index(word_info) + 1
            if next_word_index < len(word_data):
                current_ts = round(word_data[next_word_index]['start'], 1)
                sentences_with_ts.append({'text': '', 'start_ts': current_ts})

    if not sentences_with_ts:
        return {"best_sentence": None, "confusing_sentence": None}

    # Encontrar la "Mejor Oración"
    best_sentence = None
    max_score = -1
    for sentence in sentences_with_ts:
        text = sentence['text'].lower()
        word_count = len(text.split())
        
        # Puntuamos la oración: +1 por cada palabra clave, -1 por cada 10 palabras de longitud
        score = sum(1 for kw in keywords if kw in text) - (word_count / 10)
        
        if score > max_score and 5 < word_count < 25: # Buscamos frases de longitud ideal
            max_score = score
            best_sentence = {
                "text": sentence['text'].strip(),
                "timestamp": sentence['start_ts']
            }

    # Encontrar el "Momento Confuso" (la frase más larga que ya teníamos)
    longest_sentence_text = max([s['text'] for s in sentences_with_ts], key=len)
    confusing_sentence = None
    for sentence in sentences_with_ts:
        if sentence['text'] == longest_sentence_text:
            confusing_sentence = {
                "text": sentence['text'].strip(),
                "timestamp": sentence['start_ts']
            }
            break

    return {
        "best_sentence": best_sentence,
        "confusing_sentence": confusing_sentence
    }

def find_verbal_examples(full_text, word_data, language):
    """
    Analiza el texto y los datos de palabras para encontrar timestamps de momentos clave.
    """
    if not word_data: return {}
    examples = {}

    # 1. Fuerza del Cierre
    try:
        closing_keywords = ENDING_KEYWORDS.get(language, [])
        for word_info in reversed(word_data[-70:]): # Analizamos las últimas 70 palabras
            if any(keyword in word_info['word'].lower() for keyword in closing_keywords):
                examples['ending_strength'] = {'timestamp': round(word_info['start'], 1)}
                break
        else:
             examples['ending_strength'] = {'timestamp': None}
    except Exception:
        examples['ending_strength'] = {'timestamp': None}

    # 2. Uso de Retórica (Preguntas)
    try:
        for word_info in word_data:
            if '?' in word_info['word']:
                examples['rhetoric_question'] = {'timestamp': round(word_info['start'], 1)}
                break 
        else:
            examples['rhetoric_question'] = {'timestamp': None}
    except Exception:
        examples['rhetoric_question'] = {'timestamp': None}

    # 3. Claridad (la frase más larga, como ejemplo a mejorar)
    try:
        sentences = [s.strip() for s in re.split(r'[.!?]', full_text) if s.strip()]
        if sentences:
            longest_sentence = max(sentences, key=lambda s: len(s.split()))
            first_word_of_longest = longest_sentence.split(' ')[0]
            
            for word_info in word_data:
                if first_word_of_longest.lower() in word_info['word'].lower():
                    examples['clarity_longest_sentence'] = {'timestamp': round(word_info['start'], 1)}
                    break
            else:
                examples['clarity_longest_sentence'] = {'timestamp': None}
        else:
            examples['clarity_longest_sentence'] = {'timestamp': None}
    except Exception:
        examples['clarity_longest_sentence'] = {'timestamp': None}

    return examples

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

        # 3. Análisis de Contenido (ahora la lógica es secuencial y clara)
        summary = summarize_text(full_text, language)
        stopwords_base_path = os.path.dirname(__file__)
        keywords = extract_keywords(full_text, stopwords_base_path, language)
        filler_analysis = analyze_filler_words(word_data, language)
        key_sentences = extract_key_sentences(full_text, word_data, keywords, filler_analysis)
        
        total_words = len(word_data)
        filler_ratio = (filler_analysis['total_count'] / total_words) * 100 if total_words > 0 else 0
        filler_score = max(0, 10 - (filler_ratio / 0.5))

        verbal_scores = {
            "message_clarity": round(clarity_score(full_text) * 10, 2),
            "structure": round(estimate_structure_score(full_text) * 10, 2),
            "lexical_diversity": round(lexical_diversity(full_text) * 10, 2),
            "rhetoric": round(min(count_rhetorical_devices(full_text, language) / 10, 1.0) * 10, 2),
            "ending": round(ending_strength(full_text, language) * 10, 2),
            "filler_words_usage": round(filler_score, 2)
        }

        # 4. Ensamblaje del Resultado
        return {
            "verbal_analysis": {
                "summary": summary,
                "keywords": keywords,
                "full_transcription": full_text[:5000],
                "scores": verbal_scores,
                "qualitative_feedback": {
                    "filler_words": filler_analysis,
                    "key_sentences": key_sentences
                },
                "detected_language": language
            }
        }
    except Exception as e:
        print(f"ERROR en run_speech_analysis: {e}")
        traceback.print_exc() # Imprime el traceback completo en la consola del worker
        return {"verbal_analysis": {"error": str(e)}}
    finally:
        # Limpieza del archivo de audio
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)