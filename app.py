import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core import (
    init_db, process_recording, list_recent_meetings,
    list_action_items, run_agent, send_telegram_message, db
)

load_dotenv()
st.set_page_config(page_title="MeetingMind", page_icon="🎙️", layout="wide")
init_db()

st.title("🎙️ MeetingMind")
st.caption("Agentic meeting intelligence — transcribe, remember, investigate and follow up.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Upload a meeting recording\n"
        "2. Local Whisper transcribes it on this laptop\n"
        "3. Gemini or Groq extracts decisions and tasks\n"
        "4. Local MiniLM + FAISS remember meeting history\n"
        "5. The agent investigates your questions"
    )
    st.divider()
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    st.caption("Audio transcription runs locally with Faster-Whisper. Meeting transcript text is sent to your selected Gemini/Groq API for analysis. Obtain participant consent and avoid confidential content.")
    st.caption(f"Text provider: {provider.title()} · Whisper: local CPU")

tab_upload, tab_history, tab_agent = st.tabs(["➕ New meeting", "🗂️ Meeting history", "🤖 Ask MeetingMind"])

with tab_upload:
    st.subheader("Upload a meeting recording")
    title = st.text_input("Meeting title (optional)", placeholder="Weekly project sync")
    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "wav", "m4a", "flac", "ogg"],
        help="For the first MVP, upload audio-only files. Convert video recordings to MP3/WAV first."
    )
    if uploaded:
        st.audio(uploaded)
    if st.button("Transcribe and analyze", type="primary", disabled=uploaded is None):
        suffix = Path(uploaded.name).suffix or ".wav"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(uploaded.getvalue())
                temp_path = temp.name
            with st.spinner("Transcribing audio and analyzing the meeting..."):
                meeting_id, transcript, details = process_recording(temp_path, title)
            st.success(f"Meeting processed successfully. Meeting ID: {meeting_id}")
            st.session_state["last_meeting_id"] = meeting_id
            st.subheader("Summary")
            st.write(details.get("summary", "No summary returned."))
            st.subheader("Decisions")
            decisions = details.get("decisions", [])
            if decisions:
                for decision in decisions:
                    st.markdown(f"- {decision}")
            else:
                st.info("No explicit decisions were extracted.")
            st.subheader("Action items")
            items = details.get("action_items", [])
            if items:
                for item in items:
                    owner = item.get("owner") or "Owner not specified"
                    due = item.get("due_date") or "Deadline not specified"
                    st.markdown(f"- **{item.get('task')}** — {owner}; {due}")
                    if item.get("evidence"):
                        st.caption(f"Evidence: “{item['evidence']}”")
            else:
                st.info("No explicit action items were extracted.")
            with st.expander("View full transcript"):
                st.write(transcript)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            st.info("Check your selected Gemini/Groq API key and free-tier quota. Audio transcription is local; only transcript text is sent to the selected LLM provider.")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

with tab_history:
    st.subheader("Recent meetings")
    meetings = list_recent_meetings()
    if not meetings:
        st.info("No meetings yet. Upload your first recording.")
    else:
        for meeting in meetings:
            with st.expander(f"#{meeting['id']} · {meeting['title']} · {meeting['created_at'][:10]}"):
                st.write(meeting["summary"])
                with db() as conn:
                    row = conn.execute(
                        "SELECT transcript, decisions_json FROM meetings WHERE id=?",
                        (meeting["id"],)
                    ).fetchone()
                if row:
                    st.markdown("**Decisions**")
                    for decision in json.loads(row["decisions_json"] or "[]"):
                        st.markdown(f"- {decision}")
                    with st.expander("Transcript"):
                        st.write(row["transcript"])

    st.subheader("Extracted action items")
    tasks = list_action_items()
    if tasks:
        for task in tasks:
            st.markdown(
                f"- **{task['task']}** — {task.get('owner') or 'Owner unknown'} "
                f"· Due: {task.get('due_date') or 'Not specified'} "
                f"· Status: `{task['status']}`"
            )
    else:
        st.info("No action items have been extracted yet.")

    if tasks and st.button("Send action-item digest to Telegram"):
        digest = "MeetingMind — action-item digest\n\n" + "\n".join(
            f"• {t['task']} | Owner: {t.get('owner') or 'Unknown'} | Due: {t.get('due_date') or 'Not specified'}"
            for t in tasks[:20]
        )
        try:
            send_telegram_message(digest)
            st.success("Digest sent to Telegram.")
        except Exception as exc:
            st.error(str(exc))

with tab_agent:
    st.subheader("Ask questions across your meetings")
    st.caption("Examples: What commitments are still unconfirmed? What did we decide about model evaluation? Who owns the presentation?")
    question = st.text_input("Ask MeetingMind", key="agent_question")
    if st.button("Investigate", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("The agent is selecting tools and searching meeting memory..."):
                answer = run_agent(question.strip())
            st.markdown("### Agent response")
            st.write(answer)
        except Exception as exc:
            st.error(f"Agent failed: {exc}")

st.divider()
st.caption("MeetingMind MVP · Local Faster-Whisper + Gemini/Groq free tier + MiniLM + FAISS + SQLite")
