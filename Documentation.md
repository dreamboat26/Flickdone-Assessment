# 🧠 Multilingual Voice RAG Assistant - Pipeline Documentation

## Overview

This document outlines the design, toolchain, and retrieval-augmented generation (RAG) logic powering the Multilingual Voice RAG Assistant. The system combines voice input, document-based information retrieval, and multilingual response generation into an interactive AI assistant accessible via a web interface.

---

## 1. System Pipeline Architecture

```plaintext
[User Voice Input]
        ↓
[Speech-to-Text: Whisper Tiny]
        ↓
[Document Search: TF-IDF Retriever]
        ↓
[Answer Generation: FLAN-T5 Small]
        ↓
[Text-to-Speech: Coqui TTS]
        ↓
[Voice Output to User]
```

### Key Steps:

* **Input:** The user speaks into a microphone.
* **Transcription:** Whisper STT converts voice to text.
* **Retrieval:** A TF-IDF search finds the most relevant document snippet.
* **Answer Generation:** FLAN-T5 small generates a language-based response.
* **Speech Synthesis:** Coqui TTS speaks the response in the original language.

---

## 2. Tools & Models Used

| Component           | Tool / Model                            | Purpose                                   |
| ------------------- | --------------------------------------- | ----------------------------------------- |
| STT (Transcription) | OpenAI Whisper (tiny)                   | Converts speech to text                   |
| Retrieval           | TF-IDF (Scikit-learn)                   | Matches query to document segments        |
| LLM                 | `google/flan-t5-small` (Hugging Face)   | Generates natural-language answers        |
| TTS (Voice Output)  | Coqui TTS models                        | Speaks back answers in multiple languages |
| UI                  | Gradio                                  | Interactive frontend                      |
| File Parsing        | PyMuPDF, python-docx, ebooklib, moviepy | Extracts content from uploaded files      |

---

## 3. File Handling & Content Ingestion

The assistant accepts various file types as knowledge sources:

* **PDF:** Extracted via PyMuPDF (fitz)
* **TXT, DOCX:** Simple text extraction via standard libraries
* **EPUB:** Parsed via `ebooklib`
* **Audio/Video (MP3/MP4):** Transcribed via Whisper using `moviepy` and `speech_recognition`
* **SRT (Subtitles):** Parsed for conversational or temporal context

After extraction, documents are stored and embedded using a TF-IDF vectorizer.

---

## 4. Retrieval Logic (TF-IDF)

* Input query is vectorized using TF-IDF
* Dot product similarity is computed with all document embeddings
* The top match is selected as context for the LLM prompt

> Note: This approach is lightweight but may miss semantic matches — it is designed for **CPU-friendly environments**.

---

## 5. Generation Logic (FLAN-T5)

* The model receives the following prompt:

  ```
  Context: <most relevant text>

  Question: <user’s question>
  Answer:
  ```
* FLAN-T5 then generates an answer, limited to \~128 tokens

Due to performance constraints, `flan-t5-small` is used on CPU. In a GPU-enabled setting, larger models like `flan-t5-base`, `mistral-7B`, or `phi-2` can significantly improve fluency and accuracy.

---

## 6. Text-to-Speech (TTS)

Coqui TTS handles multilingual output. A dynamic language → model mapping allows:

* English, French, Spanish, German, Chinese, Hindi
* Auto-detection fallback defaults to English if unspecified

TTS output is written to a temporary WAV file and streamed to the user.

---

## 7. Design Constraints

This demo is deployed on Hugging Face Spaces **free tier**, which imposes:

* **CPU-only execution (no GPU)**
* **Longer latency (150–200s typical)**
* **Smaller model choices (to fit RAM limits)**
* **Best suited for 1–2 page documents or short media files**

> For production deployment, GPU support and smarter chunking strategies will be implemented.

---

## 8. Planned Improvements

* Switch from TF-IDF → semantic retrieval (e.g., Sentence Transformers + FAISS)
* Support for multi-document merging & chunking
* Expand document loader with summary + memory capabilities
* Upgrade LLM and TTS based on user preferences
* Add real-time microphone streaming (browser-compatible)

---

## Conclusion

This work showcases a **modular and extensible voice-based AI assistant** architecture that integrates speech, retrieval, language generation, and speech synthesis across multiple languages. While limited by current deployment resources, the design is intentionally lightweight, open, and adaptable to GPU-ready environments for real-world scalability.
