import os
import tempfile

from reportlab.platypus import PageBreak

from tools.document_tools import (
    build_styles,
    build_report,
    build_cover,
    DocumentTools,
)


def test_build_styles_returns_dict():
    styles = build_styles()
    assert isinstance(styles, dict)
    # check a couple of expected style keys
    for key in ("body", "cv_title", "sec_title"):
        assert key in styles


def test_build_report_with_sections():
    st = build_styles()
    sections = [
        {"title": "Introduction", "content": "This is an intro paragraph."},
        {"title": "Findings", "content": [{"heading": None, "text": "Item 1"}]},
    ]
    story = build_report(st, sections=sections, references=[{"title": "ref", "url": "http://example.com"}])
    assert isinstance(story, list)
    assert len(story) > 0


def test_build_cover_returns_pagebreak():
    st = build_styles()
    metadata = {"topic": "Test", "title": "T"}
    cover = build_cover(st, metadata)
    assert isinstance(cover, list)
    assert any(isinstance(item, PageBreak) for item in cover)


def test_documenttools_writes_pdf(tmp_path):
    out = os.path.join(tmp_path, "test_report.pdf")
    sections = [{"title": "Intro", "content": "A short paragraph."}]
    ret = DocumentTools.generate_research_pdf(output_path=out, metadata={"topic": "T", "title": "Title"}, sections=sections, references=[])
    assert isinstance(ret, str)
    assert os.path.exists(ret)
    assert os.path.getsize(ret) > 0