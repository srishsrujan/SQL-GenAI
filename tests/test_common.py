from src.common import adapter_matches_base, format_prompt, is_sql_output, normalize_sql

def test_normalize_sql_ignores_case_whitespace_and_final_semicolon():
    assert normalize_sql(" SELECT  * FROM T; ") == "select * from t"

def test_is_sql_output_rejects_question_echo_and_accepts_sql_statements():
    assert not is_sql_output("List employee names")
    assert is_sql_output("SELECT name FROM employees")
    assert is_sql_output("WITH active AS (SELECT 1) SELECT * FROM active")

def test_adapter_must_match_configured_base_model(tmp_path):
    adapter_config = tmp_path / "adapter_config.json"
    adapter_config.write_text(
        '{"base_model_name_or_path": "google/flan-t5-small"}',
        encoding="utf-8",
    )
    assert adapter_matches_base(tmp_path, "google/flan-t5-small")
    assert not adapter_matches_base(tmp_path, "juierror/flan-t5-text2sql-with-schema-v2")
    assert not adapter_matches_base(tmp_path / "missing", "google/flan-t5-small")

def test_format_prompt_contains_schema_and_question():
    prompt = format_prompt("CREATE TABLE t (id INT);", "show all rows")
    assert prompt == (
        "convert question and table into SQL query. "
        "tables: t(id). question: show all rows"
    )

def test_schema_to_tables_handles_multiple_tables_and_nested_types():
    schema = '''
    CREATE TABLE "employees" (
        id INTEGER PRIMARY KEY,
        name VARCHAR(100),
        department TEXT,
        CONSTRAINT salary_check CHECK (salary >= 0)
    );
    CREATE TABLE teams (id INT, "team name" TEXT);
    '''
    assert format_prompt(schema, "list employees") == (
        "convert question and table into SQL query. "
        "tables: employees(id,name,department), teams(id,team name). "
        "question: list employees"
    )

def test_format_prompt_preserves_non_ddl_schema():
    prompt = format_prompt("employees(id, name, department)", "list employees")
    assert "tables: employees(id, name, department)" in prompt
