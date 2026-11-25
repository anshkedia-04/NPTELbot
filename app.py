import json
import os
import random
import re
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

st.set_page_config(
    page_title="NPTEL IoT Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

VECTORSTORE_PATH = "faiss_index_iot"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STUDY_GUIDE_PATH = "study_guide.json"
QUESTIONS_PATH = "questions.json"

GROQ_MODELS = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
GROQ_KEY_ENV = "GROQ_API_KEY"
GROQ_CONSOLE_URL = "https://console.groq.com/keys"

CHAT_MODES: dict[str, dict[str, str]] = {
    "Q&A": {
        "description": "Direct answers grounded in course transcripts",
        "template": (
            "You are a helpful NPTEL teaching assistant for 'Introduction to IoT'.\n"
            "Answer the student's question based ONLY on the provided lecture context.\n"
            "If the context does not contain enough information, reply clearly:\n"
            "\"I couldn't find information on that topic in the selected course transcripts.\"\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        ),
    },
    "Summarize Topic": {
        "description": "Structured exam-ready notes on any topic",
        "template": (
            "You are an expert NPTEL professor for 'Introduction to IoT'.\n"
            "Create a structured, exam-focused summary of \"{query}\" using ONLY the provided context.\n"
            "Start with a brief overview, then use clear headings and bullet points for definitions, "
            "protocols, architectures, and comparisons.\n"
            "If the context does not contain relevant information, reply clearly:\n"
            "\"I couldn't find information on that topic in the selected course transcripts.\"\n\n"
            "Context:\n{context}\n\n"
            "Study Summary:"
        ),
    },
    "Explain Simply": {
        "description": "Plain-English explanation with analogies",
        "template": (
            "You are a friendly, patient tutor helping an NPTEL student prepare for their IoT exam.\n"
            "Explain \"{query}\" simply using plain language and clear analogies based ONLY on the provided context.\n"
            "If the context does not contain relevant information, reply clearly:\n"
            "\"I couldn't find information on that topic in the selected course transcripts.\"\n\n"
            "Context:\n{context}\n\n"
            "Simple Explanation:"
        ),
    },
    "Quiz Me": {
        "description": "3 practice MCQs with answer key",
        "template": (
            "You are an NPTEL professor for 'Introduction to IoT'.\n"
            "Based ONLY on the provided context, generate 3 multiple-choice questions (a, b, c, d) on \"{query}\".\n"
            "Include an **Answer Key & Explanations** section at the bottom.\n"
            "If the context does not contain enough information, reply clearly:\n"
            "\"I couldn't find enough information on that topic in the selected course transcripts to create a quiz.\"\n\n"
            "Context:\n{context}\n\n"
            "Practice Quiz:"
        ),
    },
}

STARTER_PROMPTS = [
    "Compare MQTT and CoAP protocols",
    "Explain 6LoWPAN header compression",
    "What are the main challenges in IoT networking?",
    "Quiz me on Zigbee network topologies",
]


def inject_styles() -> None:
    """Apply clean, lightweight styling for cards, headers, and badges."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.75rem;
                padding-bottom: 2.5rem;
                max-width: 1100px;
            }
            .app-hero {
                margin-bottom: 1rem;
            }
            .app-hero h1 {
                font-size: 1.85rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                margin: 0 0 0.25rem 0;
                color: #0F172A;
            }
            .app-hero p {
                font-size: 0.95rem;
                color: #475569;
                margin: 0;
            }
            .status-pill {
                display: inline-block;
                padding: 0.2rem 0.65rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            .status-ok {
                background-color: #ECFDF5;
                color: #047857;
                border: 1px solid #A7F3D0;
            }
            .status-missing {
                background-color: #FEF3C7;
                color: #92400E;
                border: 1px solid #FDE68A;
            }
            .week-badge {
                display: inline-block;
                padding: 0.15rem 0.55rem;
                border-radius: 6px;
                background-color: #EEF2FF;
                color: #4338CA;
                font-size: 0.78rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            .question-title {
                font-size: 1.12rem;
                font-weight: 600;
                line-height: 1.5;
                color: #0F172A;
                margin: 0.25rem 0 0.75rem 0;
            }
            div[data-testid="stMetric"] {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                padding: 0.85rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def natural_lecture_key(lecture_name: str) -> tuple[int, int]:
    """Extract (week, lecture) integers so W2 sorts before W10."""
    numbers = re.findall(r"\d+", lecture_name)
    if len(numbers) >= 2:
        return (int(numbers[0]), int(numbers[1]))
    if len(numbers) == 1:
        return (int(numbers[0]), 0)
    return (999, 999)


def format_lecture_title(raw_name: str) -> str:
    """Format 'W1 L1 Intro to IOT' into 'Week 1 · Lecture 1 — Intro to IOT'."""
    match = re.match(r"^W(\d+)\s+L(\d+)\s+(.+)$", raw_name.strip())
    if match:
        week, lecture, title = match.groups()
        return f"Week {week} · Lecture {lecture} — {title}"
    return raw_name


def extract_week_number(raw_name: str) -> int | None:
    """Return the week number from a lecture key like 'W1 L1 Intro to IOT'."""
    match = re.match(r"^W(\d+)\b", raw_name.strip())
    return int(match.group(1)) if match else None


def resolve_api_key() -> tuple[str, str]:
    """Return (groq_api_key, source_label) from session state, Streamlit secrets, or .env."""
    session_val = st.session_state.get("groq_api_key", "").strip()
    if session_val:
        return session_val, "Session sidebar"

    try:
        secret_val = str(st.secrets.get(GROQ_KEY_ENV, "")).strip()
        if secret_val and not secret_val.startswith("your_"):
            return secret_val, "Streamlit Secrets"
    except Exception:
        pass

    env_val = os.getenv(GROQ_KEY_ENV, "").strip()
    if env_val and not env_val.startswith("your_"):
        return env_val, ".env file"

    return "", "Not configured"


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings() -> HuggingFaceEmbeddings:
    """Load the HuggingFace sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"token": False},
    )


@st.cache_resource(show_spinner="Loading lecture index...")
def load_vector_store(_embeddings: HuggingFaceEmbeddings) -> FAISS | None:
    """Load the local FAISS vector store from disk."""
    if not os.path.exists(VECTORSTORE_PATH):
        return None
    return FAISS.load_local(
        VECTORSTORE_PATH,
        _embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_data
def load_study_guide() -> dict[str, str]:
    """Load the pre-generated lecture study guide JSON."""
    if not os.path.exists(STUDY_GUIDE_PATH):
        return {}
    with open(STUDY_GUIDE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_questions() -> list[dict[str, Any]]:
    """Load the pre-generated weekly assignment MCQs."""
    if not os.path.exists(QUESTIONS_PATH):
        return []
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_llm(model_name: str, api_key: str) -> ChatGroq:
    """Instantiate the Groq LangChain chat model."""
    return ChatGroq(
        model_name=model_name,
        temperature=0.3,
        groq_api_key=api_key,
    )


def init_session_state() -> None:
    """Initialize required keys in Streamlit session state."""
    defaults: dict[str, Any] = {
        "groq_api_key": "",
        "messages": [],
        "pending_prompt": None,
        "test_questions": [],
        "current_q_index": 0,
        "user_score": 0,
        "answered_current_q": False,
        "last_answer_correct": False,
        "test_history": [],
        "celebrated_test": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_test_state() -> None:
    """Clear current mock test progress."""
    st.session_state.test_questions = []
    st.session_state.current_q_index = 0
    st.session_state.user_score = 0
    st.session_state.answered_current_q = False
    st.session_state.last_answer_correct = False
    st.session_state.test_history = []
    st.session_state.celebrated_test = False


def render_sidebar(
    selected_tab: str, study_guide_data: dict[str, str]
) -> tuple[str, str, str]:
    """Render sidebar configuration and return (model_name, chat_mode, selected_lecture)."""
    with st.sidebar:
        st.markdown("### ⚙️ AI Tutor Settings")
        st.caption("Powered by **Groq** · Mock Test & Study Guide work offline")

        model_name = st.selectbox("Groq Model", GROQ_MODELS)

        api_key, key_source = resolve_api_key()
        if api_key:
            st.markdown(
                f'<span class="status-pill status-ok">● Key ready ({key_source})</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill status-missing">○ Groq API key needed for AI Tutor</span>',
                unsafe_allow_html=True,
            )

        with st.expander("Update or Enter Groq API Key", expanded=not bool(api_key)):
            with st.form(key="groq_api_key_form", clear_on_submit=True):
                entered_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    placeholder="Paste GROQ_API_KEY (gsk_...)...",
                )
                col_save, col_clear = st.columns(2)
                save_clicked = col_save.form_submit_button("Save", use_container_width=True)
                clear_clicked = col_clear.form_submit_button("Clear", use_container_width=True)

            if save_clicked:
                if entered_key.strip():
                    st.session_state.groq_api_key = entered_key.strip()
                    st.success("Groq API key saved for this session.")
                    st.rerun()
                else:
                    st.warning("Please paste a non-empty key before saving.")

            if clear_clicked:
                st.session_state.groq_api_key = ""
                st.info("Cleared session key.")
                st.rerun()

            st.caption(
                f"Get a free key at [Groq Console]({GROQ_CONSOLE_URL}). "
                f"For persistent setup, set `{GROQ_KEY_ENV}` in `.env` (local) "
                f"or Streamlit **App Settings → Secrets** (cloud)."
            )

        chat_mode = "Q&A"
        selected_lecture = "Search All Lectures"

        if selected_tab == "🧠 AI Tutor":
            st.divider()
            st.markdown("### 🎯 Tutor Mode")
            chat_mode = st.radio(
                "Response style",
                list(CHAT_MODES.keys()),
            )
            st.caption(CHAT_MODES[chat_mode]["description"])

            sorted_lectures = sorted(study_guide_data.keys(), key=natural_lecture_key)
            lecture_options = ["Search All Lectures"] + sorted_lectures
            selected_lecture = st.selectbox(
                "Lecture Scope",
                options=lecture_options,
                format_func=lambda x: x if x == "Search All Lectures" else format_lecture_title(x),
                help="Restrict retrieval to a single lecture or search the whole course.",
            )

            if st.session_state.messages:
                if st.button("Clear Conversation", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

        st.divider()
        st.caption(
            "Content sourced from the NPTEL *Introduction to Internet of Things* course. "
            "Non-commercial educational study aid."
        )

    return model_name, chat_mode, selected_lecture


def render_mock_test(questions: list[dict[str, Any]]) -> None:
    """Render the configurable Mock Test simulator."""
    if not questions:
        st.error(f"No questions found in `{QUESTIONS_PATH}`.")
        return

    # Setup screen
    if not st.session_state.test_questions:
        col1, col2, col3 = st.columns(3)
        col1.metric("Question Bank", f"{len(questions)} MCQs")
        col2.metric("Course Coverage", "Weeks 1 – 12")
        col3.metric("API Key Required", "None")

        st.write("")
        with st.container(border=True):
            st.subheader("Configure Your Mock Test")
            st.caption("Practice with official weekly assignment questions and instant solutions.")

            available_weeks = sorted({int(q["week"]) for q in questions if "week" in q})
            c1, c2 = st.columns(2)
            with c1:
                week_choice = st.selectbox(
                    "Topic Scope",
                    ["All Weeks (1–12)"] + [f"Week {w}" for w in available_weeks],
                )
            if week_choice == "All Weeks (1–12)":
                pool = questions
            else:
                target_week = int(week_choice.split()[-1])
                pool = [q for q in questions if q.get("week") == target_week]

            max_q = len(pool)
            count_options = [n for n in (10, 25, 50) if n <= max_q]
            if not count_options or max_q not in count_options and max_q < 50:
                count_options = sorted(set(count_options + [max_q]))

            with c2:
                default_idx = count_options.index(50) if 50 in count_options else len(count_options) - 1
                num_questions = st.selectbox(
                    "Number of Questions",
                    options=count_options,
                    index=default_idx,
                )

            if st.button("Start Mock Test", type="primary", use_container_width=True):
                reset_test_state()
                st.session_state.test_questions = random.sample(pool, num_questions)
                st.rerun()
        return

    total_q = len(st.session_state.test_questions)
    idx = st.session_state.current_q_index

    # Active test screen
    if idx < total_q:
        question = st.session_state.test_questions[idx]
        q_num = idx + 1
        answered_count = idx + (1 if st.session_state.answered_current_q else 0)

        top_left, top_mid, top_right = st.columns([3, 2, 1])
        top_left.markdown(f"**Question {q_num} of {total_q}**")
        accuracy = (st.session_state.user_score / answered_count * 100) if answered_count else 0.0
        top_mid.markdown(
            f"**Score:** {st.session_state.user_score} / {answered_count} ({accuracy:.0f}%)"
        )
        if top_right.button("End Test", use_container_width=True):
            st.session_state.current_q_index = total_q
            st.rerun()

        st.progress(q_num / total_q)

        with st.container(border=True):
            week_num = question.get("week", "?")
            st.markdown(f'<span class="week-badge">Week {week_num}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="question-title">{question["question_text"]}</div>',
                unsafe_allow_html=True,
            )

            options = [f"{k}. {v}" for k, v in question["options"].items()]
            user_answer = st.radio(
                "Select your answer:",
                options,
                key=f"q_radio_{idx}_{len(st.session_state.test_questions)}",
                index=None,
                disabled=st.session_state.answered_current_q,
                label_visibility="collapsed",
            )

            if not st.session_state.answered_current_q:
                if st.button("Check Answer", type="primary"):
                    if user_answer is None:
                        st.warning("Please select an option before checking.")
                    else:
                        selected_key = user_answer.split(".", 1)[0].strip().lower()
                        correct_key = str(question.get("correct_key", "")).strip().lower()
                        is_correct = (
                            selected_key == correct_key
                            if correct_key
                            else user_answer == question.get("correct_text")
                        )

                        if is_correct:
                            st.session_state.user_score += 1
                        st.session_state.last_answer_correct = is_correct
                        st.session_state.answered_current_q = True
                        st.session_state.test_history.append(
                            {
                                "week": week_num,
                                "question_text": question["question_text"],
                                "user_answer": user_answer,
                                "correct_text": question["correct_text"],
                                "solution": question.get("solution", ""),
                                "is_correct": is_correct,
                            }
                        )
                        st.rerun()

        if st.session_state.answered_current_q:
            if st.session_state.last_answer_correct:
                st.success("Correct!")
            else:
                st.error(f"Incorrect — Correct answer: **{question['correct_text']}**")

            if question.get("solution"):
                st.info(f"**Explanation:** {question['solution']}")

            btn_label = "Finish & View Results" if q_num == total_q else "Next Question →"
            if st.button(btn_label, type="primary", use_container_width=True):
                st.session_state.current_q_index += 1
                st.session_state.answered_current_q = False
                st.rerun()
        return

    # Results screen
    attempted = len(st.session_state.test_history)
    score = st.session_state.user_score
    percentage = (score / attempted * 100) if attempted else 0.0

    st.subheader("Mock Test Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Final Score", f"{score} / {attempted}")
    m2.metric("Accuracy", f"{percentage:.1f}%")
    m3.metric("Incorrect", f"{attempted - score}")

    if attempted and percentage >= 60:
        if not st.session_state.celebrated_test:
            st.balloons()
            st.session_state.celebrated_test = True
        st.success("Strong performance! You are well prepared for the exam.")
    elif attempted:
        st.warning("Good practice run. Review the missed questions and Study Guide below.")
    else:
        st.info("Test ended before any questions were answered.")

    missed = [item for item in st.session_state.test_history if not item["is_correct"]]
    if missed:
        with st.expander(f"Review Missed Questions ({len(missed)})", expanded=True):
            for i, item in enumerate(missed, start=1):
                st.markdown(f"**{i}. [Week {item['week']}] {item['question_text']}**")
                st.markdown(f"- Your answer: `{item['user_answer']}`")
                st.markdown(f"- Correct answer: **{item['correct_text']}**")
                if item["solution"]:
                    st.caption(f"Explanation: {item['solution']}")
                if i < len(missed):
                    st.divider()

    if st.button("Start New Mock Test", type="primary"):
        reset_test_state()
        st.rerun()


def execute_rag_query(
    prompt: str,
    chat_mode: str,
    selected_lecture: str,
    model_name: str,
    api_key: str,
) -> tuple[str, list[str]]:
    """Retrieve relevant transcript chunks and stream the LLM answer."""
    embeddings = load_embeddings()
    vectorstore = load_vector_store(embeddings)
    if vectorstore is None:
        raise FileNotFoundError(
            f"Vector store not found at `{VECTORSTORE_PATH}`. Run `python build_vectorstore.py` first."
        )

    search_kwargs: dict[str, Any] = {"k": 5}
    if selected_lecture != "Search All Lectures":
        search_kwargs["filter"] = {"source": selected_lecture}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(prompt)
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = sorted(
        {doc.metadata.get("source", "Unknown") for doc in docs},
        key=natural_lecture_key,
    )

    llm = build_llm(model_name, api_key)
    prompt_tmpl = ChatPromptTemplate.from_template(CHAT_MODES[chat_mode]["template"])
    chain = prompt_tmpl | llm | StrOutputParser()
    payload = {"context": context, "query": prompt}

    try:
        answer = st.write_stream(chain.stream(payload))
        if not isinstance(answer, str):
            answer = str(answer)
    except Exception:
        answer = chain.invoke(payload)
        st.markdown(answer)

    return answer, sources


def render_chatbot(
    model_name: str,
    chat_mode: str,
    selected_lecture: str,
) -> None:
    """Render the RAG-powered AI Tutor tab."""
    api_key, _ = resolve_api_key()

    if not api_key:
        st.info(
            "**Groq API key required for AI Tutor.**\n\n"
            f"- **Local run:** Add `{GROQ_KEY_ENV}=your_key` to `.env` in the project root.\n"
            f"- **Streamlit Cloud:** Add `{GROQ_KEY_ENV} = \"your_key\"` in **App Settings → Secrets**.\n"
            "- **Quick test:** Paste your key in the sidebar under **Update or Enter Groq API Key**."
        )

    scope_label = (
        "All 60 lectures"
        if selected_lecture == "Search All Lectures"
        else format_lecture_title(selected_lecture)
    )
    st.caption(f"Active configuration: **Groq** (`{model_name}`) · Mode: **{chat_mode}** · Scope: **{scope_label}**")

    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("#### Ask anything about NPTEL *Introduction to IoT*")
            st.caption("Answers are grounded directly in the 60 official lecture transcripts. Try a starter prompt:")
            cols = st.columns(2)
            for idx, starter in enumerate(STARTER_PROMPTS):
                if cols[idx % 2].button(starter, key=f"starter_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = starter
                    st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                formatted_sources = [format_lecture_title(s) for s in message["sources"]]
                st.caption("Sources: " + " · ".join(formatted_sources))

    user_prompt = st.chat_input("Ask a question about IoT protocols, sensors, cloud, or architecture...")
    if not user_prompt and st.session_state.pending_prompt:
        user_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if not user_prompt:
        return

    if not api_key:
        st.warning(
            f"Please add your **Groq** API key in the sidebar or `.env` (`{GROQ_KEY_ENV}`) before chatting."
        )
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching lecture transcripts..."):
                answer, sources = execute_rag_query(
                    prompt=user_prompt,
                    chat_mode=chat_mode,
                    selected_lecture=selected_lecture,
                    model_name=model_name,
                    api_key=api_key,
                )
            if sources:
                formatted_sources = [format_lecture_title(s) for s in sources]
                st.caption("Sources: " + " · ".join(formatted_sources))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as exc:
            st.error(
                "Could not generate a response from **Groq**. "
                f"Please verify your API key and network connection.\n\nDetails: `{exc}`"
            )


def render_study_guide(study_guide_data: dict[str, str]) -> None:
    """Render the searchable, week-filtered Study Guide."""
    if not study_guide_data:
        st.error(f"Study guide not found at `{STUDY_GUIDE_PATH}`.")
        return

    sorted_lectures = sorted(study_guide_data.items(), key=lambda item: natural_lecture_key(item[0]))
    weeks = sorted({w for name, _ in sorted_lectures if (w := extract_week_number(name)) is not None})

    c1, c2 = st.columns([3, 1])
    search_query = c1.text_input(
        "Search lectures",
        placeholder="Filter by keyword or topic (e.g., MQTT, Zigbee, Arduino, Fog)...",
        label_visibility="collapsed",
    ).strip()
    week_filter = c2.selectbox(
        "Filter by week",
        ["All Weeks"] + [f"Week {w}" for w in weeks],
        label_visibility="collapsed",
    )

    selected_week = int(week_filter.split()[-1]) if week_filter != "All Weeks" else None
    filtered = []
    for raw_name, summary in sorted_lectures:
        if selected_week is not None and extract_week_number(raw_name) != selected_week:
            continue
        if search_query:
            haystack = f"{raw_name} {summary}".lower()
            if search_query.lower() not in haystack:
                continue
        filtered.append((raw_name, summary))

    st.caption(f"Showing **{len(filtered)}** of {len(sorted_lectures)} lectures · No API key required")

    if not filtered:
        st.info("No lectures matched your search filter. Try another keyword or clear the week filter.")
        return

    for idx, (raw_name, summary) in enumerate(filtered):
        expand_first = bool(search_query) and idx == 0
        with st.expander(format_lecture_title(raw_name), expanded=expand_first):
            st.markdown(summary)


def main() -> None:
    """Application entrypoint."""
    inject_styles()
    init_session_state()

    study_guide_data = load_study_guide()
    question_data = load_questions()

    st.markdown(
        """
        <div class="app-hero">
            <h1>🎓 NPTEL IoT Exam-Prep Assistant</h1>
            <p>Introduction to Internet of Things · Weeks 1–12 · 60 Lecture Guides &amp; 180 Official Assignment MCQs</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_options = ["🎯 Mock Test", "🧠 AI Tutor", "📖 Study Guide"]
    if hasattr(st, "segmented_control"):
        selected_tab = st.segmented_control(
            "Navigation",
            options=nav_options,
            default=nav_options[0],
            label_visibility="collapsed",
        )
        if not selected_tab:
            selected_tab = nav_options[0]
    else:
        selected_tab = st.radio(
            "Navigation",
            options=nav_options,
            horizontal=True,
            label_visibility="collapsed",
        )

    st.write("")

    model_name, chat_mode, selected_lecture = render_sidebar(
        selected_tab, study_guide_data
    )

    if selected_tab == "🎯 Mock Test":
        render_mock_test(question_data)
    elif selected_tab == "🧠 AI Tutor":
        render_chatbot(model_name, chat_mode, selected_lecture)
    else:
        render_study_guide(study_guide_data)


if __name__ == "__main__":
    main()