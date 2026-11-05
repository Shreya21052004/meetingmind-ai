import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "meetingmind.db"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embedder = None
_whisper_model = None


def get_embedder():
    """Load the local embedding model only when RAG is first used."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_whisper_model():
    """Load Faster-Whisper locally; model weights are downloaded once on first use."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Local Whisper is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc
        model_size = os.getenv("WHISPER_MODEL", "base").strip() or "base"
        device = os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu"
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
        _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _whisper_model


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                transcript TEXT NOT NULL,
                summary TEXT NOT NULL,
                decisions_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                FOREIGN KEY(meeting_id) REFERENCES meetings(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                task TEXT NOT NULL,
                owner TEXT,
                due_date TEXT,
                status TEXT NOT NULL DEFAULT 'unconfirmed',
                evidence TEXT,
                FOREIGN KEY(meeting_id) REFERENCES meetings(id)
            )
        """)


def llm(messages, max_tokens=1200, temperature=0.2):
    """Call the selected free-tier text API. Audio transcription stays local."""
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add a free Gemini API key to your .env file."
            )
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        system_text = "\n\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        ).strip()
        contents = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                continue
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": str(message.get("content", ""))}],
            })
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            response = requests.post(
                url, params={"key": api_key}, json=payload, timeout=90
            )
            response.raise_for_status()
            data = response.json()
            return "".join(
                part.get("text", "")
                for part in data["candidates"][0]["content"]["parts"]
            ).strip()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            if status == 429:
                raise RuntimeError(
                    "Gemini free-tier request limit reached. Wait for the quota to reset or switch to Groq; no payment is required."
                ) from exc
            if status in (400, 403):
                raise RuntimeError(
                    "Gemini rejected the request. Check your API key, model name and AI Studio free-tier access."
                ) from exc
            raise RuntimeError(f"Gemini API request failed (HTTP {status}).") from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Gemini returned an unexpected response. Please retry.") from exc

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add a Groq API key to your .env file."
            )
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=90,
            )
            response.raise_for_status()
            return (response.json()["choices"][0]["message"].get("content") or "").strip()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            if status == 429:
                raise RuntimeError(
                    "Groq free-tier rate limit reached. Wait for the quota to reset or switch to Gemini; no payment is required."
                ) from exc
            if status in (400, 401, 403):
                raise RuntimeError(
                    "Groq rejected the request. Check your API key and configured model name."
                ) from exc
            raise RuntimeError(f"Groq API request failed (HTTP {status}).") from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Groq returned an unexpected response. Please retry.") from exc

    raise RuntimeError("Unsupported LLM_PROVIDER. Choose 'gemini' or 'groq' in your .env file.")


def transcribe_audio(audio_path):
    """Transcribe audio locally with Faster-Whisper; no audio is sent to an API."""
    model = get_whisper_model()
    segments, _info = model.transcribe(
        str(audio_path), beam_size=5, vad_filter=True
    )
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    return transcript


def split_text(text, size=1000, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def extract_meeting_details(transcript):
    prompt = f"""
You are MeetingMind, a careful meeting analyst.
Extract information ONLY when supported by the transcript.
Do not invent owners, deadlines, decisions, or completion status.
If owner or due date is not stated, use null.
Return ONLY valid JSON with this exact shape:
{{
  "title": "short descriptive meeting title",
  "summary": " concise summary in 3-5 sentences",
  "decisions": ["decision 1"],
  "action_items": [
    {{
      "task": "specific action",
      "owner": null,
      "due_date": null,
      "evidence": "short exact supporting quote"
    }}
  ]
}}
A mentioned task is not proof that it was completed.
TRANSCRIPT:
{transcript[:24000]}
"""
    raw = llm(
        [
            {"role": "system", "content": "Return valid JSON only. Never fabricate meeting facts."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1600,
        temperature=0.1,
    )
    return parse_json(raw)


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    raise ValueError("The model did not return valid JSON. Please retry or use a shorter recording.")


def save_meeting(title, transcript, details):
    created_at = datetime.now(timezone.utc).isoformat()
    chunks = split_text(transcript)
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO meetings(title, created_at, transcript, summary, decisions_json) VALUES(?,?,?,?,?)",
            (
                title or details.get("title") or "Untitled meeting",
                created_at,
                transcript,
                details.get("summary", ""),
                json.dumps(details.get("decisions", []), ensure_ascii=False),
            ),
        )
        meeting_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO chunks(meeting_id, chunk_text) VALUES(?,?)",
            [(meeting_id, chunk) for chunk in chunks],
        )
        for item in details.get("action_items", []):
            if not isinstance(item, dict) or not item.get("task"):
                continue
            conn.execute(
                """INSERT INTO action_items(meeting_id, task, owner, due_date, status, evidence)
                   VALUES(?,?,?,?,?,?)""",
                (
                    meeting_id,
                    str(item["task"]),
                    item.get("owner"),
                    item.get("due_date"),
                    "unconfirmed",
                    item.get("evidence"),
                ),
            )
    return meeting_id


def process_recording(audio_path, title=""):
    transcript = transcribe_audio(audio_path)
    if not transcript:
        raise ValueError("No speech was detected in this recording.")
    details = extract_meeting_details(transcript)
    meeting_id = save_meeting(title or details.get("title", "Meeting"), transcript, details)
    return meeting_id, transcript, details


def search_meetings(query, top_k=5, exclude_meeting_id=None):
    """Small RAG retriever. Rebuilds a flat FAISS index from saved chunks for MVP simplicity."""
    with db() as conn:
        rows = conn.execute(
            """SELECT c.id, c.meeting_id, c.chunk_text, m.title, m.created_at
               FROM chunks c JOIN meetings m ON c.meeting_id=m.id
               ORDER BY c.id"""
        ).fetchall()

    if exclude_meeting_id is not None:
        rows = [r for r in rows if r["meeting_id"] != exclude_meeting_id]
    if not rows:
        return []

    embedder = get_embedder()
    texts = [r["chunk_text"] for r in rows]
    vectors = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)

    try:
        import faiss
        vectors = np.asarray(vectors, dtype="float32")
        q = np.asarray(q, dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        scores, indices = index.search(q, min(top_k, len(rows)))
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                row = rows[int(idx)]
                hits.append({
                    "score": float(score),
                    "meeting_id": row["meeting_id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "text": row["chunk_text"],
                })
        return hits
    except ImportError:
        scores = np.dot(vectors, q[0])
        best = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "score": float(scores[i]),
                "meeting_id": rows[int(i)]["meeting_id"],
                "title": rows[int(i)]["title"],
                "created_at": rows[int(i)]["created_at"],
                "text": rows[int(i)]["chunk_text"],
            }
            for i in best
        ]


def list_action_items():
    with db() as conn:
        rows = conn.execute("""
            SELECT a.*, m.title AS meeting_title, m.created_at
            FROM action_items a JOIN meetings m ON a.meeting_id=m.id
            ORDER BY a.id DESC LIMIT 50
        """).fetchall()
    return [dict(r) for r in rows]


def list_recent_meetings():
    with db() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, summary FROM meetings ORDER BY id DESC LIMIT 10"
        ).fetchall()
    return [dict(r) for r in rows]


def run_agent(user_question, max_steps=3):
    """
    A simple single-agent tool loop:
    model chooses a tool -> Python executes it -> model sees result -> final answer.
    Tool choice is JSON-based for compatibility across hosted inference providers.
    """
    tool_specs = {
        "search_meetings": "Search previous meeting transcripts by semantic meaning. Args: {\"query\": \"...\"}",
        "list_action_items": "List extracted commitments and their current evidence status. Args: {}",
        "list_recent_meetings": "List recent meetings and their summaries. Args: {}",
    }
    observations = []
    for _ in range(max_steps):
        prompt = f"""
You are MeetingMind, an assistant that investigates meeting history.
Available tools:
{json.dumps(tool_specs)}
User question: {user_question}
Previous tool observations:
{json.dumps(observations, ensure_ascii=False)}
Choose exactly one next step. Return ONLY JSON:
{{"action":"tool","tool_name":"search_meetings","arguments":{{"query":"..."}}}}
OR, when you have enough evidence:
{{"action":"final","answer":"..."}}
Never claim a task is completed unless the meeting evidence explicitly confirms completion.
If no evidence exists, say "not confirmed in the available meeting records".
"""
        raw = llm(
            [
                {"role": "system", "content": "You are a careful tool-using meeting research agent. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=0.1,
        )
        decision = parse_json(raw)

        if decision.get("action") == "final":
            return decision.get("answer", "I could not form a final answer.")

        if decision.get("action") != "tool":
            break

        name = decision.get("tool_name")
        args = decision.get("arguments") or {}
        if name == "search_meetings":
            result = search_meetings(str(args.get("query", user_question)), top_k=5)
        elif name == "list_action_items":
            result = list_action_items()
        elif name == "list_recent_meetings":
            result = list_recent_meetings()
        else:
            result = {"error": "Unknown tool"}
        observations.append({"tool": name, "result": result})

    # Safe fallback if the model fails to follow the JSON tool protocol.
    if not observations:
        hits = search_meetings(user_question, top_k=5)
        observations.append({"tool": "search_meetings", "result": hits})
    return llm(
        [
            {"role": "system", "content": "Answer only from the supplied meeting evidence. Clearly state uncertainty."},
            {"role": "user", "content": f"Question: {user_question}\nEvidence: {json.dumps(observations, ensure_ascii=False)}"},
        ],
        max_tokens=900,
        temperature=0.1,
    )


def send_telegram_message(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env first.")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()
