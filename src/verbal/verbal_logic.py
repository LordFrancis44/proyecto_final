import os
import tempfile
#import torch
import whisper
import yt_dlp
from transformers import pipeline
from urllib.parse import urlparse, parse_qs
from collections import Counter
import re

# --- Utilidades ---
def download_audio(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'ffmpeg_location': '/opt/homebrew/bin/ffmpeg',  # Asegúrate de que este path sea válido para tu sistema
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        audio_path = os.path.join(tempfile.gettempdir(), f"{info['id']}.mp3")
        return audio_path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language='en')
    return result['text']

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summarized = summarizer(chunks, max_length=130, min_length=30, do_sample=False)
    return " ".join([s['summary_text'] for s in summarized])

def extract_keywords(text, stopwords_path):
    with open(stopwords_path, 'r') as f:
        stopwords = set(word.strip().lower() for word in f.readlines())
    words = [w.strip('.,!"()?').lower() for w in text.split() if w.lower() not in stopwords and len(w) > 3]
    freq = Counter(words)
    return [word for word, _ in freq.most_common(10)]

def lexical_diversity(text):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)

def count_rhetorical_devices(text):
    repetitions = len(re.findall(r'\b(\w+)\b(?:\W+\1\b)+', text.lower()))
    questions = text.count('?')
    metaphors = len(re.findall(r'\b(like|as if|as though)\b', text.lower()))
    return repetitions + questions + metaphors

def estimate_structure_score(text):
    segments = [s for s in re.split(r'\n|\.|\?|!', text) if len(s.strip()) > 10]
    return min(len(segments) / 10, 1.0)

def ending_strength(text):
    closing = text[-300:].lower()
    keywords = ["thank you", "final thought", "remember", "last message", "takeaway"]
    return float(any(k in closing for k in keywords))

def clarity_score(text):
    sentences = [s for s in text.split('.') if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    return max(0.0, min(1.0, 1 - (avg_sentence_length - 10) / 20))

def run_speech_analysis(video_url):
    try:
        audio_path = download_audio(video_url)
        full_text = transcribe_audio(audio_path)
        summary = summarize_text(full_text)
        stopwords_file = os.path.join(os.path.dirname(__file__), 'stopwords_en.txt')
        keywords = extract_keywords(full_text, stopwords_file)

        verbal_scores = {
            "message_clarity": round(clarity_score(full_text) * 10, 2),
            "structure": round(estimate_structure_score(full_text) * 10, 2),
            "lexical_diversity": round(lexical_diversity(full_text) * 10, 2),
            "rhetoric": round(min(count_rhetorical_devices(full_text) / 10, 1.0) * 10, 2),
            "ending": round(ending_strength(full_text) * 10, 2)
        }

        return {
            "verbal_analysis": {
                "summary": summary,
                "keywords": keywords,
                "full_transcription": full_text[:5000],
                "verbal_scores": verbal_scores
            }
        }
    except Exception as e:
        return {"verbal_analysis": {"error": str(e)}}
