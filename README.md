# MeetingMind 🎙️
### AI-Powered Meeting Intelligence & Knowledge Assistant

MeetingMind transforms meeting recordings into structured, searchable knowledge. It uses local speech recognition and AI-powered analysis to generate meeting summaries, extract decisions, identify action items, and retrieve insights from previous meetings.

## ✨ Key Features

- **Audio Transcription:** Converts meeting recordings into text using locally running Faster-Whisper.
- **AI Meeting Summaries:** Generates concise summaries of meeting discussions.
- **Decision Extraction:** Automatically identifies important decisions made during meetings.
- **Action Item Tracking:** Extracts tasks and commitments for better follow-up.
- **Meeting Intelligence (RAG):** Searches across previous meeting transcripts to retrieve relevant information.
- **AI Agent:** Uses tools to investigate meeting history and answer context-aware questions.
- **Telegram Integration:** Optionally delivers action-item digests through Telegram.

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Speech Recognition | Faster-Whisper |
| LLM | Google Gemini / Groq |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Database | SQLite |
| Language | Python |
| Integration | Telegram Bot API |

## ⚙️ Architecture

1. Upload meeting audio.
2. Faster-Whisper locally converts speech into text.
3. The LLM processes the transcript to generate summaries, decisions and action items.
4. Transcript chunks are embedded and indexed for semantic search.
5. The RAG-powered agent retrieves relevant information from previous meetings.
6. Users receive contextual answers based on their meeting history.

## 🚀 Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/Shreya21052004/meetingmind-ai.git
cd meetingmind-ai
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment (Windows)**

```powershell
.venv\Scripts\activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Configure environment variables**

```bash
Copy-Item .env.example .env
```

Add your Gemini or Groq API key to `.env` and configure your preferred LLM provider.

**6. Run the application**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 🔒 Privacy & Data Handling

- Audio transcription runs locally using Faster-Whisper.
- Raw uploaded audio is deleted after processing.
- Transcript text is sent to the selected Gemini or Groq API for AI analysis.
- Meeting data and derived information are stored locally using SQLite.

Avoid uploading confidential meeting recordings to hosted AI services without appropriate permission.

## ⚠️ Current Limitations

- Processes uploaded recordings, not live meetings.
- Speaker diarization is not currently supported.
- Task completion requires explicit confirmation.
- Designed as an MVP for small-scale meeting history and retrieval.

## 👩‍💻 Author

**Shreya Ammalajeri**  
B.Tech – Computer Science and Engineering

[GitHub](https://github.com/Shreya21052004)

---

*MeetingMind — Turning conversations into actionable knowledge.*
