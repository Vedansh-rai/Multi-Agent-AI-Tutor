"""
JEE Algebra Solver — Streamlit UI
Interactive interface with step-by-step explanations, LaTeX rendering,
confidence display, and agent transparency.
"""

import streamlit as st
import requests
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page Config ──
st.set_page_config(
    page_title="🧭 JEE Algebra Solver",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API Config ──
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8001")

# ── Streamlit Cloud Compatibility (Auto-Start Backend) ──
# Removed: Backend is now started via start.sh to prevent double-booting.
# ── Custom Styling ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .trace-step {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .source-card {
        background: #f0f4ff;
        border: 1px solid #d0d7ff;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ──
st.markdown('<div class="main-header">🧭 JEE Algebra Solver</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Agent AI Tutor for JEE Advanced/Mains • SymPy Verified • RAG-Powered</div>',
    unsafe_allow_html=True,
)


# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")

    api_url = st.text_input("API URL", value=API_BASE, help="FastAPI backend URL")

    st.divider()
    st.header("📊 Quick Actions")

    if st.button("🔄 Ingest KB Files", use_container_width=True):
        try:
            resp = requests.post(f"{api_url}/ingest", timeout=120)
            if resp.status_code == 200:
                st.success(f"✅ Files: {resp.json().get('result', {})}")
            else:
                st.error(f"Failed: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

    if st.button("📚 Ingest JEE Formulas", use_container_width=True):
        try:
            resp = requests.post(f"{api_url}/ingest/jee", timeout=60)
            if resp.status_code == 200:
                st.success(f"✅ Chunks: {resp.json().get('result', {})}")
            else:
                st.error(f"Failed: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

    if st.button("📜 View History", use_container_width=True):
        try:
            resp = requests.get(f"{api_url}/history", timeout=10)
            if resp.status_code == 200:
                history = resp.json().get("history", [])
                if history:
                    for h in history[:10]:
                        with st.expander(f"#{h['id']} — {h.get('topic', '?')} ({h.get('timestamp', '')[:10]})"):
                            st.write(f"**Problem:** {h.get('parsed_problem', '')[:100]}")
                            st.write(f"**Answer:** {h.get('final_answer', '')[:100]}")
                            st.write(f"**Confidence:** {h.get('confidence', 0)}")
                            feedback = h.get('user_feedback', '')
                            if feedback:
                                st.write(f"**Feedback:** {feedback}")
                else:
                    st.info("No history yet.")
        except Exception as e:
            st.error(f"Connection error: {e}")

    st.divider()
    st.markdown("**Built with:**")
    st.markdown("🤖 LangGraph Agents")
    st.markdown("📚 ChromaDB RAG")
    st.markdown("🧠 Self-Learning Memory")
    st.markdown("🔧 SymPy + Calculator")


# ── Main Input Area ──
input_mode = st.radio(
    "Choose input method:",
    ["📝 Text", "📷 Image", "🎤 Audio"],
    horizontal=True,
)

result = None
edited_text = None

if input_mode == "📝 Text":
    user_text = st.text_area(
        "Enter your math problem:",
        placeholder="e.g., Solve x² - 5x + 6 = 0\ne.g., Find the derivative of sin(x) · eˣ\ne.g., What is the probability of getting 3 heads in 5 coin tosses?",
        height=120,
    )

    if st.button("🚀 Solve", type="primary", use_container_width=True):
        if user_text.strip():
            with st.spinner("🧠 Agents are working on your JEE problem..."):
                try:
                    resp = requests.post(
                        f"{api_url}/solve",
                        json={"text": user_text},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                    else:
                        st.error(f"API Error: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Make sure FastAPI is running on " + api_url)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a math problem.")

elif input_mode == "📷 Image":
    uploaded_image = st.file_uploader(
        "Upload a math problem image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear photo of a math problem",
    )

    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # State management for the 2-step process
        if "extracted_text" not in st.session_state:
            st.session_state.extracted_text = None
        if "editing_text" not in st.session_state:
            st.session_state.editing_text = False

        # Step 1: Extract
        if not st.session_state.editing_text:
            if st.button("🔍 Extract Text", type="primary", use_container_width=True):
                with st.spinner("🔎 Analyzing image..."):
                    try:
                        files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                        resp = requests.post(f"{api_url}/extract/image", files=files, timeout=120)
                        if resp.status_code == 200:
                            st.session_state.extracted_text = resp.json().get('problem_text', '')
                            st.session_state.editing_text = True
                            st.rerun()
                        else:
                            st.error(f"API Error: {resp.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Step 2: Edit & Solve
        if st.session_state.editing_text:
            st.info("💡 Review the extracted text below. Make any necessary corrections before solving!")
            
            edited_text = st.text_area(
                "Extracted Problem Text (Edit if needed):",
                value=st.session_state.extracted_text,
                height=150
            )
            
            col_solve, col_cancel = st.columns([3, 1])
            with col_solve:
                if st.button("🚀 Confirm & Solve", type="primary", use_container_width=True):
                    with st.spinner("🧠 Agents are working..."):
                        try:
                            # Send the edited text to the standard text solver
                            resp = requests.post(f"{api_url}/solve/text", json={"text": edited_text}, timeout=120)
                            if resp.status_code == 200:
                                result = resp.json()
                                # Reset state
                                st.session_state.editing_text = False
                                st.session_state.extracted_text = None
                            else:
                                st.error(f"API Error: {resp.text}")
                        except Exception as e:
                            st.error(f"Error: {e}")
                            
            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.editing_text = False
                    st.session_state.extracted_text = None
                    st.rerun()

elif input_mode == "🎤 Audio":
    recorded_audio = st.audio_input("🎙️ Record a math problem")
    
    uploaded_audio = st.file_uploader(
        "Or upload an existing audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        help="Upload audio of you reading a math problem",
    )

    audio_to_process = recorded_audio or uploaded_audio

    if audio_to_process:
        if st.button("🎙️ Transcribe & Solve", type="primary", use_container_width=True):
            with st.spinner("🎙️ Transcribing and solving..."):
                try:
                    files = {"file": (audio_to_process.name if hasattr(audio_to_process, 'name') else "recording.wav", audio_to_process.getvalue(), audio_to_process.type if hasattr(audio_to_process, 'type') else "audio/wav")}
                    resp = requests.post(f"{api_url}/solve/audio", files=files, timeout=180)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.info(f"📋 Transcript: {result.get('problem_text', 'N/A')}")
                    else:
                        st.error(f"API Error: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend.")
                except Exception as e:
                    st.error(f"Error: {e}")


def _format_math(text: str) -> str:
    """Fix LaTeX delimiters for Streamlit markdown rendering."""
    if not text:
        return ""
    text = text.replace("\\(", "$").replace("\\)", "$")
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    return text


# ── Display Results ──
if result:
    st.divider()

    # ── Error Check ──
    if result.get("error") and not result.get("final_answer") and not result.get("solution"):
        st.error(f"❌ {result['error']}")
    else:
        # Normalise: /solve returns JEESolveResponse; /solve/text returns PipelineResponse
        is_jee_schema = "solution" in result  # JEESolveResponse has nested `solution`
        if is_jee_schema:
            sol = result.get("solution") or {}
            final_ans = sol.get("final_answer", "")
            conf = (result.get("verification") or {}).get("confidence_score", 0.0)
            steps = sol.get("steps", [])
            expl_var = result.get("explanation", "")
            if isinstance(expl_var, dict):
                explanation_text = expl_var.get("formatted_explanation", "")
                key_insight = expl_var.get("key_insight", result.get("key_insight", ""))
            else:
                explanation_text = expl_var
                key_insight = result.get("key_insight", "")
            topic = result.get("topic", "N/A")
            problem_type = result.get("problem_type", "")
            status = result.get("status", "PARTIAL")
            sources = result.get("rag_sources", [])
            breakdown = {}
        else:
            final_ans = result.get("final_answer", "")
            conf = result.get("confidence", 0.0)
            steps = result.get("solution_steps", [])
            expl_dict = result.get("explanation", {})
            explanation_text = expl_dict.get("formatted_explanation", "") if isinstance(expl_dict, dict) else ""
            key_insight = expl_dict.get("key_insight", "") if isinstance(expl_dict, dict) else ""
            topic = result.get("topic", "N/A")
            problem_type = result.get("problem_type", "")
            status = result.get("status", "PARTIAL")
            sources = result.get("rag_sources", [])
            breakdown = result.get("score_breakdown", {})

        # ── Status badge ──
        _STATUS_COLOR = {"SOLVED": "🟢", "UNSOLVED": "🔴", "PARTIAL": "🟡", "BLOCKED": "⛔"}
        status_icon = _STATUS_COLOR.get(status, "🟡")
        st.markdown(
            f"<div style='text-align:right;font-size:0.95rem;color:#555;'>"
            f"Status: {status_icon} <strong>{status}</strong> "
            f"| Retries: {result.get('retry_count', 0)}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Metrics Row ──
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("📐 Topic", (topic or "N/A").replace("_", " ").title())
        with m2:
            conf_pct = f"{conf:.1%}"
            conf_color = "normal" if conf >= 0.8 else ("off" if conf >= 0.5 else "inverse")
            st.metric("🎯 Confidence", conf_pct, delta=None)
        with m3:
            st.metric("📝 Steps", len(steps))
        with m4:
            st.metric("📚 KB Sources", len(sources))

        # ── Confidence Meter ──
        _bar_color = "#28a745" if conf >= 0.8 else ("#ffc107" if conf >= 0.5 else "#dc3545")
        st.markdown(
            f"""<div style="background:#eee;border-radius:8px;height:12px;margin:4px 0 12px 0;">
            <div style="width:{min(conf,1.0)*100:.1f}%;background:{_bar_color};height:12px;border-radius:8px;
            transition:width 0.4s ease;"></div></div>
            <div style="font-size:0.8rem;color:#666;text-align:right;margin-top:-8px;">
            Verification confidence: {conf:.1%}</div>""",
            unsafe_allow_html=True,
        )

        if result.get("needs_human_review"):
            st.warning(
                f"⚠️ **Human Review Requested** — confidence below threshold. "
                f"Please verify the solution manually."
            )

        # ── Tabs ──
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📝 Solution", "🔍 Agent Trace", "📚 Sources", "📊 Verification", "💬 Feedback"
        ])

        # ── Tab 1: Solution & Explanation ──
        with tab1:
            # Final Answer
            st.markdown("### ✅ Final Answer")
            
            # If the backend returned the default placeholder, try to extract it from the explanation
            if "Refer to the step-by-step" in final_ans and explanation_text:
                import re
                # Look for "Step 4: Interpret result - solutions are x = 2 and x = 3" or similar
                match = re.search(r"solutions are([^<]+?)(?:Step|✅ Verification|💡 Key Insight|$)", explanation_text, re.IGNORECASE)
                if match:
                    final_ans = match.group(1).strip()
            
            if final_ans.strip().startswith("{"):
                try:
                    import json as _json
                    _parsed = _json.loads(final_ans)
                    final_ans = _parsed.get("final_answer", final_ans)
                except Exception:
                    pass
            st.success(f"**{_format_math(str(final_ans))}**" if final_ans else "N/A")

            # Problem type badge
            if problem_type:
                st.markdown(
                    f"<span style='background:#EEF2FF;color:#4338ca;padding:2px 10px;"
                    f"border-radius:12px;font-size:0.82rem;font-weight:600;'>"
                    f"{problem_type.replace('_',' ').title()}</span>",
                    unsafe_allow_html=True,
                )

            # ── Solver Steps ──
            if steps:
                st.markdown("---")
                st.markdown("### 🔢 Step-by-Step Solution")
                for step in steps:
                    step_num  = step.get("step", "?")
                    desc      = step.get("description", "")
                    work      = step.get("work", "")
                    tool_res  = step.get("tool_result", "")
                    tool_call = step.get("tool_call") or {}

                    with st.container():
                        st.markdown(
                            f"""<div style="background:#fff;border-left:5px solid #4F46E5;
                                border-radius:8px;padding:14px;margin-bottom:8px;
                                box-shadow:0 2px 4px rgba(0,0,0,0.06);">
                                <span style="background:#EEF2FF;color:#4F46E5;font-weight:800;
                                    padding:3px 10px;border-radius:9999px;font-size:0.82rem;">
                                    STEP {step_num}</span>
                                <h4 style="margin:8px 0 4px;color:#1F2937;">{desc}</h4>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        if work:
                            st.markdown(_format_math(work))
                        if tool_call:
                            st.markdown(
                                f"<div style='background:#F8FAFC;padding:8px 12px;border-radius:6px;"
                                f"font-family:monospace;font-size:0.85rem;border:1px solid #E2E8F0;'>"
                                f"🔧 <strong>Tool:</strong> <code>{tool_call.get('type','?')}</code> — "
                                f"<code>{str(tool_call.get('input', tool_call.get('args','')))[:120]}</code>"
                                f"</div>", unsafe_allow_html=True)
                        if tool_res:
                            st.markdown(
                                f"<div style='background:#F0FDF4;padding:8px 12px;border-radius:6px;"
                                f"font-family:monospace;font-size:0.85rem;border:1px solid #BBF7D0;color:#166534;'>"
                                f"✅ <strong>Result:</strong> <code>{tool_res}</code>"
                                f"</div>", unsafe_allow_html=True)
                        st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

            # ── JEE Explanation (formatted_explanation or dict fallback) ──
            st.markdown("---")
            if explanation_text:
                # New JEE schema: formatted_explanation is a markdown string with emoji headers
                st.markdown("### 📖 Explanation")
                st.markdown(_format_math(explanation_text))
                if key_insight:
                    st.info(f"💡 **Key Insight:** {_format_math(key_insight)}")
            else:
                # Legacy dict format from /solve/text
                expl_dict = result.get("explanation", {})
                if isinstance(expl_dict, dict) and expl_dict:
                    st.markdown("### 📖 Explanation")
                    if expl_dict.get("step_by_step") and not steps:
                        st.markdown(_format_math(str(expl_dict["step_by_step"])))
                    if expl_dict.get("key_insight"):
                        st.info(f"💡 **Key Insight:** {_format_math(expl_dict['key_insight'])}")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if expl_dict.get("intuitive_explanation"):
                            st.markdown("#### 🧠 Intuitive Explanation")
                            st.markdown(_format_math(expl_dict["intuitive_explanation"]))
                        if expl_dict.get("common_pitfall"):
                            st.warning(f"⚠️ {_format_math(expl_dict['common_pitfall'])}")
                    with col_b:
                        if expl_dict.get("real_world_analogy"):
                            st.markdown("#### 🌍 Real-World Analogy")
                            st.markdown(_format_math(expl_dict["real_world_analogy"]))
                        if expl_dict.get("quick_tip"):
                            st.success(f"⚡ {_format_math(expl_dict['quick_tip'])}")
                    if expl_dict.get("related_topics"):
                        tags = " ".join([f"`{t}`" for t in expl_dict["related_topics"]])
                        st.markdown(f"**Related Topics:** {tags}")

        # ── Tab 2: Agent Trace ──
        with tab2:
            st.subheader("🔍 Agent Execution Trace")
            trace = result.get("trace", [])
            if trace:
                for step in trace:
                    with st.container():
                        st.markdown(
                            f'<div class="trace-step" style="color: #1F2937;">'
                            f'<strong style="color: #4F46E5;">Step {step.get("step", "?")}:</strong> '
                            f'<span style="font-weight: 600;">{step.get("agent_name", "Unknown Agent")}</span><br>'
                            f'<div style="margin-top: 8px;">'
                            f'<em style="color: #4B5563;">Action:</em> <span>{step.get("action", "N/A")}</span><br>'
                            f'<em style="color: #4B5563;">Input:</em> <span style="font-family: monospace; font-size: 0.9em;">{step.get("input_summary", "")[:120]}</span><br>'
                            f'<em style="color: #4B5563;">Output:</em> <span style="font-family: monospace; font-size: 0.9em; color: #065F46;">{step.get("output_summary", "")[:200]}</span>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                        meta = step.get("metadata", {})
                        if meta:
                            with st.expander("Metadata"):
                                st.json(meta)
            else:
                st.info("No trace available.")

        # ── Tab 3: Retrieved Sources ──
        with tab3:
            st.subheader("📚 Retrieved Knowledge Base Sources")
            if sources:
                for s in sources:
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <strong>{s.get("source", "unknown")}</strong> '
                        f'(relevance: {s.get("score", 0):.2f})'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No RAG sources were used.")  

            st.subheader("📝 Full RAG Context")
            rag_ctx = result.get("rag_context", "")
            if rag_ctx and rag_ctx != "No relevant knowledge base documents found.":
                with st.expander("View full retrieved context"):
                    st.text(rag_ctx)

        # ── Tab 4: Verification Details ──
        with tab4:
            st.subheader("📊 Verification Score Breakdown")
            # JEE schema: verification is a nested object
            if is_jee_schema:
                ver = result.get("verification") or {}
                st.metric("Verified", "✅ Yes" if ver.get("verified") else "❌ No")
                st.metric("Confidence Score", f"{ver.get('confidence_score', conf):.1%}")
                st.metric("Method", ver.get("method", "hybrid").title())
                st.progress(
                    min(ver.get("confidence_score", conf), 1.0),
                    text=f"Confidence: {ver.get('confidence_score', conf):.1%}",
                )
            elif breakdown:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("SymPy Checks", f"{breakdown.get('sympy_confidence', 0):.1%}")
                    st.metric("Numerical Score", f"{breakdown.get('llm_numerical', 0):.1%}")
                with col2:
                    st.metric("Domain Score", f"{breakdown.get('llm_domain', 0):.1%}")
                    st.metric("Reasoning Score", f"{breakdown.get('llm_reasoning', 0):.1%}")
                st.progress(
                    min(conf, 1.0),
                    text=f"Overall Confidence: {conf:.1%}",
                )
            else:
                st.info("No verification data available.")

        # ── Tab 5: Feedback ──
        with tab5:
            st.subheader("💬 Your Feedback")
            st.markdown("Help the system learn! Your feedback is stored and used to improve future answers.")

            interaction_id = result.get("interaction_id", 0)
            fb_col1, fb_col2 = st.columns(2)

            with fb_col1:
                if st.button("✅ Correct", use_container_width=True, type="primary"):
                    try:
                        resp = requests.post(
                            f"{api_url}/feedback",
                            json={
                                "interaction_id": interaction_id,
                                "feedback": "correct",
                            },
                            timeout=10,
                        )
                        if resp.status_code == 200:
                            st.success("✅ Thank you! Positive feedback recorded.")
                        else:
                            st.error("Failed to save feedback.")
                    except Exception as e:
                        st.error(f"Error: {e}")

            with fb_col2:
                if st.button("❌ Incorrect", use_container_width=True):
                    st.session_state["show_correction"] = True

            if st.session_state.get("show_correction"):
                comment = st.text_input("What was wrong?", key="fb_comment")
                corrected = st.text_input("Correct answer (optional):", key="fb_corrected")

                if st.button("Submit Correction", type="primary"):
                    try:
                        resp = requests.post(
                            f"{api_url}/feedback",
                            json={
                                "interaction_id": interaction_id,
                                "feedback": "incorrect",
                                "comment": comment,
                                "corrected_answer": corrected,
                            },
                            timeout=10,
                        )
                        if resp.status_code == 200:
                            st.success("📝 Correction saved! The system will learn from this.")
                            st.session_state["show_correction"] = False
                        else:
                            st.error("Failed to save correction.")
                    except Exception as e:
                        st.error(f"Error: {e}")


# ── Footer ──
st.divider()
st.markdown(
    '<div style="text-align:center;color:#888;font-size:0.85rem;">'
    "🧭 JEE Algebra Solver v2.0 • FastAPI + LangGraph + SymPy + ChromaDB + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
