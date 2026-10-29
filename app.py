import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.common import (
    format_prompt,
    is_sql_output,
    load_config,
)

st.set_page_config(
    page_title="SchemaSQL | Text to SQL",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #17243b;
        --muted: #78869d;
        --line: #e6ebf2;
        --blue: #4263eb;
        --teal: #0e9f91;
        --canvas: #f5f7fb;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(87, 113, 244, .08), transparent 34rem),
            var(--canvas);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1.5rem; }
    [data-testid="stMainBlockContainer"] { max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem; }
    [data-testid="stSidebar"] {
        background: #111a2e;
        border-right: 1px solid rgba(255,255,255,.07);
    }
    [data-testid="stSidebar"] * { color: #e8edf7; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aab6cb; }
    h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif; color: var(--ink); }
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 3px 12px rgba(27, 44, 75, .035);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .79rem; }
    [data-testid="stMetricValue"] { font-size: 1.35rem; }
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        border: 1px solid #dfe5ef;
        border-radius: 10px;
        background: #fbfcfe;
    }
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: #8398f5;
        box-shadow: 0 0 0 3px rgba(66, 99, 235, .10);
    }
    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] button {
        min-height: 2.85rem;
        border: 0;
        border-radius: 10px;
        background: linear-gradient(110deg, #4263eb, #5b6ff0);
        color: #fff;
        font-weight: 700;
        box-shadow: 0 7px 16px rgba(66, 99, 235, .19);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 20px rgba(66, 99, 235, .25);
        color: #fff;
    }
    [data-testid="stCode"] {
        border: 1px solid #253652;
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stDownloadButton"] button {
        border-radius: 9px;
        border: 1px solid var(--line);
    }
    .brand-lockup {
        display: flex; align-items: center; gap: 11px;
        padding: 4px 0 22px; color: #f5f7ff;
        font: 800 1.13rem 'Manrope', sans-serif; letter-spacing: -.035em;
    }
    .brand-mark {
        display: grid; place-items: center; width: 36px; height: 36px;
        border-radius: 11px; color: white; font-size: 18px;
        background: linear-gradient(145deg, #5473ff, #2e49bd);
        box-shadow: 0 5px 15px rgba(66, 99, 235, .32);
    }
    .hero {
        position: relative; overflow: hidden;
        padding: 30px 34px 32px; margin-bottom: 22px;
        border: 1px solid #e6eaf4; border-radius: 20px;
        background: linear-gradient(115deg, #fff 30%, #f0f3ff 100%);
        box-shadow: 0 7px 24px rgba(27, 44, 75, .045);
    }
    .hero:after {
        content: ''; position: absolute; width: 230px; height: 230px;
        right: 8%; top: -145px; border-radius: 50%;
        border: 1px solid rgba(66,99,235,.10);
        box-shadow: 0 0 0 25px rgba(66,99,235,.025), 0 0 0 55px rgba(66,99,235,.02);
    }
    .eyebrow {
        color: #5169d8; font: 700 .72rem 'DM Sans', sans-serif;
        letter-spacing: .13em; text-transform: uppercase;
    }
    .hero h1 {
        margin: 8px 0 7px; color: #17243b;
        font: 800 clamp(1.8rem, 3vw, 2.5rem)/1.16 'Manrope', sans-serif;
        letter-spacing: -.055em;
    }
    .hero p { margin: 0; max-width: 660px; color: #73819a; font-size: .98rem; }
    .section-heading {
        margin: 8px 0 4px; color: #192944;
        font: 700 1.12rem 'Manrope', sans-serif; letter-spacing: -.025em;
    }
    .section-hint { color: #8290a5; font-size: .85rem; margin-bottom: 15px; }
    .sidebar-note {
        margin-top: 1rem; padding: 12px 13px;
        border: 1px solid rgba(255,255,255,.1); border-radius: 11px;
        background: rgba(255,255,255,.045); color: #aab6cb;
        font-size: .78rem; line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

cfg = load_config()


@st.cache_resource
def load_generator(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, device


with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><span class="brand-mark">S</span>SchemaSQL</div>',
        unsafe_allow_html=True,
    )
    st.caption("AI / ML TRACK  ·  TEXT TO SQL")
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-note">Generated SQL is a suggestion, not an execution. '
        'Review every query before using it with a database.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Schema-aware query workspace</div>
      <h1>Ask your data in plain English.</h1>
      <p>Turn natural-language questions into SQL, grounded in your schema.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">Build a query</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-hint">Provide the schema and describe the result you need. '
    'Your query is generated locally by the base model.</div>',
    unsafe_allow_html=True,
)

with st.form("sql_generation_form", border=False):
    schema_column, question_column = st.columns([1.05, 0.95], gap="large")
    with schema_column:
        schema = st.text_area(
            "Database schema",
            value="""CREATE TABLE employees (
  id INTEGER,
  name TEXT,
  department TEXT,
  salary REAL
);""",
            height=210,
            help="Paste CREATE TABLE statements or a concise description of your tables and columns.",
        )
    with question_column:
        question = st.text_area(
            "What should the query do?",
            value="List employee names in Engineering ordered by salary from highest to lowest.",
            height=150,
            help="Describe the information you want in everyday language.",
        )
        st.caption(
            "Tip: Include filters, sorting, grouping, or limits in your request for a more specific query."
        )
    submitted = st.form_submit_button("✦  Generate SQL", type="primary", use_container_width=True)

if submitted:
    if not schema.strip() or not question.strip():
        st.error("Please provide both a schema and a question.")
    else:
        prompt = format_prompt(schema, question)
        try:
            with st.spinner("Loading model and generating your query..."):
                tokenizer, model, device = load_generator(cfg["base_model"])
                tokens = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(cfg["max_input_length"]),
                ).to(device)
                with torch.no_grad():
                    output = model.generate(
                        **tokens,
                        max_new_tokens=128,
                        num_beams=10,
                        early_stopping=True,
                    )
                generated_sql = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            if not is_sql_output(generated_sql):
                st.session_state.pop("generated_sql", None)
                st.error(
                    "The model did not return SQL. Try a more specific question and check that "
                    "the requested columns exist in the schema."
                )
            else:
                st.session_state["generated_sql"] = generated_sql
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

if st.session_state.get("generated_sql"):
    st.markdown('<div class="section-heading">Generated query</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-hint">Review the SQL before running it against any database.</div>',
        unsafe_allow_html=True,
    )
    st.code(st.session_state["generated_sql"], language="sql")
    st.download_button(
        "Download SQL",
        st.session_state["generated_sql"],
        file_name="generated_query.sql",
        mime="text/sql",
    )
