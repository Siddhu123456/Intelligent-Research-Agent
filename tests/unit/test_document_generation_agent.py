import os

from agents.document_generation_agent import DocumentGenerationAgent
from tools.document_tools import DocumentTools


def test_document_generation_agent_handles_invalid_report_sections(monkeypatch, tmp_path):
    dummy_pdf = tmp_path / "report.pdf"

    def fake_generate_research_pdf(output_path, metadata=None, sections=None, references=None):
        assert isinstance(sections, list)
        assert sections[0]["content"] == "A report"
        with open(output_path, "wb") as f:
            f.write(b"%PDF-1.4")
        return output_path

    monkeypatch.setattr(DocumentTools, "generate_research_pdf", staticmethod(fake_generate_research_pdf))

    state = {
        "active_report": "A report",
        "report_sections": "invalid-string",
        "report_section_order": "invalid-order",
        "citations": [],
        "report_title": "Test Report",
    }

    result = DocumentGenerationAgent.run(state)


def test_document_generation_agent_wraps_string_state(monkeypatch, tmp_path):
    def fake_generate_research_pdf(output_path, metadata=None, sections=None, references=None):
        assert isinstance(sections, list)
        assert sections[0]["content"] == "A string report"
        with open(output_path, "wb") as f:
            f.write(b"%PDF-1.4")
        return output_path

    monkeypatch.setattr(DocumentTools, "generate_research_pdf", staticmethod(fake_generate_research_pdf))

    result = DocumentGenerationAgent.run("A string report")

    assert result["current_step"] == "done"
    assert result["generated_pdf"] == b"%PDF-1.4"
    assert result["run_metadata"]["generated_pdf_filename"]

    assert result["current_step"] == "done"
    assert result["generated_pdf"] == b"%PDF-1.4"
    assert result["run_metadata"]["generated_pdf_filename"] == os.path.basename(result["run_metadata"]["generated_pdf_filename"])
