# ♿ Multilingual Voice RAG Assistant

A multilingual, speech-enabled assistant that answers user queries based on uploaded documents and media using a Retrieval-Augmented Generation (RAG) architecture. Ask a question in your voice, and get a spoken answer in your language!

![1](https://github.com/user-attachments/assets/98236f37-739e-4e4b-9305-66a15f3f5854)

---

Gradio App hosted on Huggingface Spaces : https://huggingface.co/spaces/dreamboat26/Demo

## 🚀 What This Project Does

This assistant can:
- Accept spoken questions in multiple languages
- Transcribe your voice to text using Whisper
- Search uploaded documents for relevant content
- Use a small LLM to generate an answer
- Convert the answer into natural speech in your selected language

---

## 🎯 Key Features

- 🎤 **Voice-based question input**
- 🌐 **Multilingual support** (`en`, `fr`, `es`, `de`, `zh`, `hi`)
- 📚 **Upload custom knowledge** from files like PDFs, EPUBs, TXT, DOCX, MP3, MP4, SRT
- 🔍 **Document search** using TF-IDF vectorization
- 🧠 **Answer generation** via `FLAN-T5`
- 🔊 **Text-to-speech** response using Coqui TTS
- 🖥️ **Gradio interface** for seamless interaction

---

### 📦 All in One:
A **single app** that combines **speech-to-text**, **retrieval-augmented generation**, and **text-to-speech**, fully customizable and open-source.

---

## 📁 Supported File Types

- `.pdf`, `.txt`, `.docx`, `.epub`
- `.mp3`, `.mp4` (audio auto-extracted)
- `.srt` (subtitle files)

---

## 🧠 Architecture Overview

### 🔁 Pipeline Flow

1. **Voice Input**  
   - User speaks into the microphone
   - Audio captured via Gradio interface

2. **Speech-to-Text (Whisper)**  
   - Audio is transcribed using OpenAI Whisper
   - Supports automatic or manual language selection

3. **Context Retrieval (TF-IDF)**  
   - TF-IDF vectorizer is used to encode the uploaded text documents
   - Transcribed question is encoded and compared to corpus
   - Top-matching text snippet is selected as context

4. **Answer Generation (FLAN-T5)**  
   - Context + question is passed to `google/flan-t5-small`
   - A concise answer is generated using prompt-based generation

5. **Text-to-Speech (Coqui TTS)**  
   - The generated answer is synthesized to audio in the input language
   - Multilingual models are used for language-specific TTS

6. **Gradio Output**  
   - Transcribed question shown
   - Generated answer displayed
   - Spoken answer played as audio

---

## ⚙️ Implementation Notes

This version was **deployed on Hugging Face Spaces**, so it:

- Uses **lightweight models** (`whisper-tiny`, `flan-t5-small`) to stay within resource limits.
- Limits **generation length and context size** for performance (responses in ~150–200s max).
- Works best with **short files** (1–2 pages of PDF or ~2000 words).

⚠️ **Limitations** in this version:
- Since this was implemented through Huggingface Spaces I was using limited to using CPU free tier for the model, but this can be further optimized for GPU acceleration.
- Long documents or large media files may cause slowdowns as I am using lighter model, for full proper implementation would use a much better model to handle larger documents.
- Retrieval is TF-IDF based (non-semantic, brittle to paraphrasing).
- Text-to-speech output time grows with response length.

✅ **Planned Enhancements** for a full-scale version:
- Upgrade to `flan-t5-base` or `mistral-7B` with GPU support.
- Use `sentence-transformers` + `FAISS` for faster, smarter semantic search.
- Add context summarization and chunking for large documents.
- Use streaming TTS or more natural synthesis (e.g. Bark, ElevenLabs).

---

## 🧰 Tools and Libraries

| Task | Tool |
|------|------|
| UI | Gradio |
| Transcription | Whisper (`tiny` model) |
| Retrieval | `TfidfVectorizer` from Scikit-learn |
| LLM | `google/flan-t5-small` |
| TTS | Coqui TTS |
| File Parsing | PyMuPDF, `python-docx`, `ebooklib`, `pysrt`, `moviepy`, `speechrecognition` |

---

## 🌍 TTS Language Support

| Language | Model |
|----------|-----------------------------------------------------|
| English  | `tts_models/en/ljspeech/tacotron2-DDC`              |
| French   | `tts_models/fr/mai/tacotron2-DDC`                   |
| Spanish  | `tts_models/es/mai/tacotron2-DDC`                   |
| German   | `tts_models/de/thorsten/tacotron2-DCA`             |
| Chinese  | `tts_models/zh-CN/baker/tacotron2-DDC-GST`         |
| Hindi    | `tts_models/multilingual/multi-dataset/your_tts`   |

---

## 🧪 Example Workflow

1. Upload a PDF book or MP3 podcast.
2. Speak your question into the microphone.
3. System transcribes the audio, searches your document, answers, and reads the answer aloud.
4. All in your selected language!

---

## 🧪 Sample Usage

1. Upload `book.pdf` or `lecture.mp3`.
2. Speak your question:  
   _"What is the central theme of the document?"_
3. App transcribes → retrieves → generates → synthesizes
4. You see:
   - 📝 Transcription
   - 💬 Answer text
   - 🔊 Spoken response (in your selected language)

---

## 📦 Installation & Setup

### 1. Clone the Repo

```bash
git clone https://github.com/dreamboat26/Flickdone-Assessment.git
cd Flickdone-Assessment
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run The App
```bash
python app.py
```

## License

This project is licensed under the MIT License.
