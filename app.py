import gradio as gr
import numpy as np
import whisper
import torch
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS  # Coqui TTS
import tempfile
import docx
import ebooklib
from ebooklib import epub
import pysrt
import moviepy.editor as mp
import speech_recognition as sr

# Multilingual TTS
def text_to_speech(text, lang_code="en"):
    try:
        model_map = {
            "en": "tts_models/en/ljspeech/tacotron2-DDC",
            "fr": "tts_models/fr/mai/tacotron2-DDC",
            "es": "tts_models/es/mai/tacotron2-DDC",
            "de": "tts_models/de/thorsten/tacotron2-DCA",
            "zh": "tts_models/zh-CN/baker/tacotron2-DDC-GST",
            "hi": "tts_models/multilingual/multi-dataset/your_tts"
        }
        model_name = model_map.get(lang_code, model_map["en"])
        tts = TTS(model_name=model_name)
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        tts.tts_to_file(text=text, file_path=output_path)
        return output_path
    except Exception as e:
        print(f"TTS error: {e}")
        return None

class RobustRAG:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            min_df=1,
            stop_words='english',
            token_pattern=r'(?u)\b\w\w+\b'
        )
        self.docs = ["This is a placeholder document."]
        self.stt_model = whisper.load_model("tiny")
        self.initialize_embeddings()

    def initialize_embeddings(self):
        self.doc_embeddings = self.vectorizer.fit_transform(self.docs).toarray()

    def load_documents_from_texts(self, texts):
        self.docs = texts
        self.initialize_embeddings()

    def search(self, query):
        try:
            query_vec = self.vectorizer.transform([query]).toarray()
            scores = np.dot(self.doc_embeddings, query_vec.T)
            best_idx = np.argmax(scores)
            return self.docs[best_idx]
        except Exception as e:
            return f"Search error: {str(e)}"

    def transcribe(self, audio_path, language=None):
        try:
            result = self.stt_model.transcribe(audio_path, language=language)
            return result["text"]
        except Exception as e:
            return f"Transcription error: {str(e)}"

class LLMWrapper:
    def __init__(self):
        model_name = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to("cpu")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Helpers

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def extract_text(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "txt":
        return open(file_path, "r", encoding="utf-8").read()
    elif ext == "docx":
        return "\n".join([p.text for p in docx.Document(file_path).paragraphs])
    elif ext == "epub":
        book = epub.read_epub(file_path)
        return "\n".join([item.get_content().decode("utf-8") for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT])
    elif ext in ["srt", "vtt"]:
        return pysrt.open(file_path).text
    elif ext in ["mp3", "mp4"]:
        audio = mp.AudioFileClip(file_path)
        audio.write_audiofile("temp.wav")
        recognizer = sr.Recognizer()
        with sr.AudioFile("temp.wav") as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_whisper(audio_data)
    else:
        return "Unsupported file type."

rag = RobustRAG()
llm = LLMWrapper()

def process(audio_file, language):
    try:
        audio = AudioSegment.from_file(audio_file)
        wav_path = "input.wav"
        audio.export(wav_path, format="wav")

        lang_code = None if language == "auto" else language
        query = rag.transcribe(wav_path, language=lang_code)
        if "error" in query.lower():
            return query, "Could not process audio", None

        context = rag.search(query)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        answer = llm.generate(prompt)
        tts_path = text_to_speech(answer, lang_code=(language if language != "auto" else "en"))
        return query, answer, tts_path
    except Exception as e:
        return f"System error: {str(e)}", "", None

def update_knowledge(file):
    try:
        text = extract_text(file.name)
        rag.load_documents_from_texts([text])
        return "Knowledge source loaded."
    except Exception as e:
        return f"Failed to load knowledge source: {str(e)}"

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ♿ Multilingual Voice RAG Assistant")

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak your question",
            waveform_options={"sample_rate": 16000}
        )
        lang_dropdown = gr.Dropdown(
            label="Select Language",
            choices=["auto", "en", "fr", "es", "de", "zh", "hi"],
            value="auto"
        )

    with gr.Row():
        with gr.Column():
            query_output = gr.Textbox(label="Transcribed Question")
        with gr.Column():
            response_output = gr.Textbox(label="System Response", lines=4)

    audio_response = gr.Audio(label="Spoken Answer")

    with gr.Row():
        knowledge_upload = gr.File(label="Upload Knowledge Source (PDF, TXT, EPUB, MP3, MP4, DOCX, SRT)", file_types=[".pdf", ".txt", ".epub", ".mp3", ".mp4", ".docx", ".srt"])
        status_output = gr.Textbox(label="Knowledge Upload Status")

    audio_input.change(
        fn=process,
        inputs=[audio_input, lang_dropdown],
        outputs=[query_output, response_output, audio_response]
    )

    knowledge_upload.change(fn=update_knowledge, inputs=knowledge_upload, outputs=status_output)

if __name__ == "__main__":
    demo.launch()
