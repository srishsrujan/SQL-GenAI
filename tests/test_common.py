from src.common import normalize_sql, format_prompt

def test_normalize_sql_ignores_case_whitespace_and_final_semicolon():
    assert normalize_sql(" SELECT  * FROM T; ") == "select * from t"

def test_format_prompt_contains_schema_and_question():
    prompt = format_prompt("CREATE TABLE t (id INT);", "show all rows")
    assert "CREATE TABLE t" in prompt
    assert "show all rows" in prompt
    assert prompt.endswith("SQL:")
