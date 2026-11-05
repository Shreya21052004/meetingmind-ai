# MeetingMind — local Whisper + free-tier LLM MVP

MeetingMind transcribes uploaded meeting audio locally, extracts summaries/decisions/action items using a selectable free-tier text API, stores transcript chunks for RAG, and uses a single tool-using agent to investigate previous meetings.

## Architecture

- **ASR:** Faster-Whisper running locally on your computer (`WHISPER_MODEL=base` by default). Audio is not uploaded to an ASR provider.
- **LLM:** Choose either Google Gemini API free tier or Groq developer/free tier using `LLM_PROVIDER`.
- **Embeddings:** local `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector search:** FAISS (with NumPy fallback).
- **Storage:** SQLite.
- **UI:** Streamlit.
- **Optional integration:** Telegram Bot API.

The LLM provider receives transcript text for summarization, extraction and agent answers. Do not send confidential meeting content to a hosted API unless your organization and participants permit it. Free-tier availability and quotas can change. Do not enable billing or add a payment method if you want to keep this project at zero API cost; when a free quota is exhausted, wait for reset or switch provider.

## 1. Requirements

Use Python 3.11 or 3.12. On Windows PowerShell, from the folder containing `app.py`:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Faster-Whisper downloads the selected model weights the first time you process audio. This is a one-time model download (not a paid inference API call). The default `base` model is a reasonable CPU starting point; `small` may improve accuracy but uses more memory and time. The first run can take a while.

## 2. Configure a free-tier text provider

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
notepad .env
```

Choose **one** provider. The supplied `.env.example` defaults to Groq Free Plan. Do not upgrade or configure paid usage.

### Option A — Gemini free tier

1. Create an API key in [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Set `LLM_PROVIDER=gemini`.
3. Set `GEMINI_API_KEY=your_key` in `.env`.
4. Leave `GEMINI_MODEL=gemini-2.5-flash` unless Google AI Studio shows a different free-tier model available to your account.
5. Do not set up billing. Gemini free-tier data may be used by Google to improve its products; avoid confidential transcripts.

### Option B — Groq free tier (recommended default)

1. Create an API key in [Groq Console](https://console.groq.com/keys).
2. Set `LLM_PROVIDER=groq`.
3. Set `GROQ_API_KEY=your_key` in `.env`.
4. Keep `GROQ_MODEL=openai/gpt-oss-120b` or use a currently available model shown in your Groq console.
5. Do not upgrade to a paid plan. Free-tier limits apply.

You do not need a Hugging Face token, Ollama, or a paid API for this setup.

## 3. Local Whisper settings

Defaults in `.env`:

```env
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

For a lower-memory/faster test, use `tiny`. For better accuracy (with more RAM/time), use `small`. Faster-Whisper runs on CPU by default. No separate FFmpeg installation is normally needed because Faster-Whisper uses PyAV for audio decoding.

## 4. Run

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## 5. First demo

1. Start with a short, clear 1–3 minute MP3 or WAV recording.
2. Upload it in **New meeting**.
3. Wait for local Whisper to load its model and transcribe the audio.
4. Review the transcript, summary, decisions and action items.
5. Upload a second meeting recording that refers to an earlier commitment.
6. Open **Ask MeetingMind** and ask: `What previous commitments are mentioned in this meeting history?`
7. Ask: `Which action items have no completion confirmation?`

## 6. Optional Telegram integration

1. Create a bot using Telegram's BotFather.
2. Obtain the bot token and chat ID.
3. Add both values to `.env`.
4. Restart Streamlit and use **Send action-item digest to Telegram**.

## MVP limitations

- Version 1 processes uploaded audio; it does not monitor live meetings.
- Speaker diarization is not implemented. Whisper does not reliably identify who said each sentence.
- Task status remains `unconfirmed` until evidence of completion is explicitly found; the MVP does not automatically mark tasks complete.
- Audio is processed locally, but transcript text is sent to the selected Gemini/Groq API for LLM analysis and agent responses.
- Free-tier API quotas, model availability and policies can change. The app does not automatically switch to a paid provider or enable billing.
- The MVP stores transcripts and derived meeting information in a local SQLite database. Raw uploaded audio is deleted after processing.
- The RAG index is rebuilt from stored transcript chunks for MVP simplicity. This is suitable for a small demo, not a large production system.

## Troubleshooting

### Gemini/Groq quota reached
Wait until the provider's free quota resets, or switch `LLM_PROVIDER` and configure the other provider's key. Do not enable billing if you want to stay free.

### Invalid API key/model
Check the selected provider, key and model name in `.env`. Restart Streamlit after editing `.env`.

### First transcription is slow
The first run downloads the Faster-Whisper model. Try `WHISPER_MODEL=tiny` for a quicker initial test, then restart Streamlit.

### Out of memory / slow transcription
Use `WHISPER_MODEL=tiny` or `base`, and close other memory-heavy applications. Long recordings can take a long time on CPU; test with a short recording first.

### FAISS install issue
The application has a NumPy similarity-search fallback if FAISS is unavailable. You can temporarily remove `faiss-cpu` from requirements and run without FAISS.

### Audio upload issue
Start with MP3 or WAV. Convert video recordings to audio before upload.
