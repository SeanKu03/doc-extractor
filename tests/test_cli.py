from extractor.__main__ import main


def test_cli_returns_2_when_ollama_is_down(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # порт, на котором никого нет
    doc = tmp_path / "doc.txt"
    doc.write_text("Счёт № 1", encoding="utf-8")
    assert main([str(doc)]) == 2
    assert "LLM недоступна" in capsys.readouterr().err


def test_cli_rejects_empty_document(tmp_path):
    doc = tmp_path / "empty.txt"
    doc.write_text("   ", encoding="utf-8")
    assert main([str(doc)]) == 2
