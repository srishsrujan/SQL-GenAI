import json

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.common import (
    REPORT_DIR,
    ROOT,
    adapter_matches_base,
    format_prompt,
    is_sql_output,
    load_config,
)
from src.retrieval import ExampleRetriever
from src.upload_hub import upload_model

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
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 12px;
    }
    [data-testid="stSidebar"] input { color: #17243b !important; }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
    [data-testid="stSidebar"] [data-testid="stToggle"] label { color: #e8edf7; }
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
model_dir = ROOT / cfg["final_model_dir"]
adapter_compatible = adapter_matches_base(model_dir, cfg["base_model"])
index_dir = ROOT / "artifacts" / "retrieval"
retrieval_ready = (index_dir / "rows.json").exists() and (index_dir / "vectors.npy").exists()
evaluation_path = REPORT_DIR / "evaluation.json"


@st.cache_resource
def load_generator(base_model, adapter_dir, use_finetuned):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    if use_finetuned:
        model = PeftModel.from_pretrained(model, adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, device


@st.cache_resource
def load_retriever(model_name):
    return ExampleRetriever(model_name=model_name)


with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><span class="brand-mark">S</span>SchemaSQL</div>',
        unsafe_allow_html=True,
    )
    st.caption("AI / ML TRACK  ·  TEXT TO SQL")
    st.markdown("---")
    st.markdown("#### Workspace settings")
    use_finetuned = st.toggle(
        "Fine-tuned LoRA adapter",
        value=adapter_compatible,
        disabled=not adapter_compatible,
        help="Train a LoRA adapter from the currently configured base model before enabling this option.",
    )
    if model_dir.exists() and not adapter_compatible:
        st.caption("The local adapter was trained from a different base model and cannot be used here.")
    use_retrieval = st.toggle(
        "Retrieve similar examples",
        value=retrieval_ready,
        disabled=not retrieval_ready,
        help="Add semantically similar training examples as context.",
    )
    top_k = st.slider("Examples to retrieve", 1, 5, 3, disabled=not use_retrieval)
    st.markdown(
        '<div class="sidebar-note">Generated SQL is a suggestion, not an execution. '
        'Review every query before using it with a database.</div>',
        unsafe_allow_html=True,
    )
    if adapter_compatible:
        with st.expander("Publish LoRA adapter"):
            st.caption("Uses your local Hugging Face CLI login or HF_TOKEN.")
            repo_id = st.text_input(
                "Hugging Face model repository",
                placeholder="username/schemasql-lora",
            )
            private_repo = st.checkbox("Create a private repository", value=False)
            if st.button("Upload model and model card", use_container_width=True):
                if not repo_id.strip():
                    st.error("Enter a Hugging Face repository ID.")
                else:
                    try:
                        with st.spinner("Uploading adapter and model card..."):
                            upload_model(model_dir, repo_id.strip(), private=private_repo)
                        st.success(f"Uploaded to https://huggingface.co/{repo_id.strip()}")
                    except Exception as exc:
                        st.error(f"Upload failed: {exc}")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Schema-aware query workspace</div>
      <h1>Ask your data in plain English.</h1>
      <p>Turn natural-language questions into SQL, grounded in your schema and enhanced with relevant examples.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics = st.columns(4)
metrics[0].metric("Base model", cfg["base_model"].split("/")[-1])
metrics[1].metric("Fine-tuned adapter", "Ready" if model_dir.exists() else "Not trained")
metrics[2].metric("Example retrieval", "Ready" if retrieval_ready else "Index not built")
metrics[3].metric("Evaluation", "Available" if evaluation_path.exists() else "Not run")

st.markdown('<div class="section-heading">Build a query</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-hint">Provide the schema and describe the result you need. '
    'Your query is generated locally by the selected model.</div>',
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
    elif use_finetuned and not model_dir.exists():
        st.error("Fine-tuned model not found. Run `python -m src.train` first.")
    else:
        examples = []
        if use_retrieval:
            try:
                retriever = load_retriever(cfg["embedding_model"])
                examples = retriever.search(schema, question, top_k)
            except Exception as exc:
                st.info(f"Retrieval unavailable ({exc}). Continuing without retrieved examples.")
        example_text = "\n\n".join(
            f"Schema: {example['schema']}\nQuestion: {example['question']}\nSQL: {example['sql']}"
            for example in examples
        )
        prompt = format_prompt(schema, question, example_text)
        try:
            with st.spinner("Loading model and generating your query..."):
                tokenizer, model, device = load_generator(
                    cfg["base_model"], str(model_dir), use_finetuned
                )
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
                st.session_state.pop("generated_examples", None)
                st.error(
                    "The model did not return SQL. Try a more specific question and check that "
                    "the requested columns exist in the schema."
                )
            else:
                st.session_state["generated_sql"] = generated_sql
                st.session_state["generated_examples"] = examples
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
    generated_examples = st.session_state.get("generated_examples", [])
    if generated_examples:
        with st.expander(f"Retrieved examples · {len(generated_examples)}"):
            for number, example in enumerate(generated_examples, 1):
                st.markdown(f"**Example {number} · similarity {example['score']:.3f}**")
                st.write(example["question"])
                st.code(example["sql"], language="sql")

st.markdown("---")
st.markdown('<div class="section-heading">Model evaluation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-hint">Held-out test results help compare the base model with its fine-tuned version.</div>',
    unsafe_allow_html=True,
)
if evaluation_path.exists():
    try:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        st.caption(f"Evaluated on {evaluation['test_samples']} held-out examples.")
        evaluation_columns = st.columns(
            3 if "finetuned_exact_match_accuracy" in evaluation else 2
        )
        evaluation_columns[0].metric(
            "Base · exact match",
            f"{evaluation['base_exact_match_accuracy']:.1%}",
        )
        evaluation_columns[1].metric(
            "Base · normalized",
            f"{evaluation['base_normalized_exact_match_accuracy']:.1%}",
        )
        if "finetuned_exact_match_accuracy" in evaluation:
            evaluation_columns[2].metric(
                "Fine-tuned · exact match",
                f"{evaluation['finetuned_exact_match_accuracy']:.1%}",
            )
        with st.expander("Example benchmark outputs"):
            for number, example in enumerate(evaluation.get("examples", []), 1):
                st.markdown(f"**{number}. {example['question']}**")
                st.caption(f"Expected: {example['gold_sql']}")
                st.markdown("Base model")
                st.code(example["base_prediction"], language="sql")
                if "finetuned_prediction" in example:
                    st.markdown("Fine-tuned model")
                    st.code(example["finetuned_prediction"], language="sql")
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not read the evaluation report: {exc}")
else:
    st.info("Run `python -m src.evaluate` to show benchmark metrics and example outputs here.")
