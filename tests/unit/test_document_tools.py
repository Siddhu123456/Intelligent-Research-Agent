import os
import tempfile

from reportlab.platypus import PageBreak, Paragraph

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


def test_build_report_parses_markdown_lists_and_headings():
    st = build_styles()
    content = (
        "Governance Frameworks and Legal Challenges\n\n"
        "NIST’s Technical Governance Model\n\n"
        "- Securing AI Systems: Requires adversarial robustness testing and secure-by-design principles for AI models.\n"
        "- AI-Driven Threat Detection: Mandates continuous validation of detection accuracy and bias mitigation in training data.\n"
        "- Offensive Risk Mitigation: Calls for red-team exercises to simulate AI-enabled attacks and assess system resilience.\n\n"
        "Ethical Governance and Liability Frameworks\n\n"
        "- Inclusivity: Requires stakeholder participation in AI design, ensuring affected parties influence risk thresholds and decision rules.\n"
        "- Visibility: Demands transparent audit trails for AI decisions, aligning with agency law principles to assign liability in cases of harm.\n"
    )
    sections = [{"title": "Governance", "content": content}]
    story = build_report(st, sections=sections, references=[])

    assert isinstance(story, list)
    assert any(isinstance(item, Paragraph) for item in story)
    assert any(getattr(item, 'bulletText', None) == '•' for item in story if isinstance(item, Paragraph))


def test_build_cover_returns_pagebreak():
    st = build_styles()
    metadata = {"topic": "Test", "title": "T"}
    cover = build_cover(st, metadata)
    assert isinstance(cover, list)
    assert any(isinstance(item, PageBreak) for item in cover)


def test_build_report_handles_list_of_text_blocks():
    st = build_styles()
    sections = [
        {
            "title": "Governance",
            "content": [
                "The governance of autonomous AI in cybersecurity requires reconciling technical constraints with legal accountability.",
                "NIST’s Technical Governance Model",
                "- Securing AI Systems: Requires adversarial robustness testing and secure-by-design principles for AI models.",
                "- AI-Driven Threat Detection: Mandates continuous validation of detection accuracy and bias mitigation in training data.",
            ],
        }
    ]
    story = build_report(st, sections=sections, references=[])
    assert isinstance(story, list)
    assert any(isinstance(item, Paragraph) for item in story)


def test_build_report_handles_first_section_list_strings():
    st = build_styles()
    sections = [
        {
            "title": "Executive Summary",
            "content": [
                "This is the first paragraph.",
                "This is the second paragraph.",
                "- A markdown list item.",
            ],
        }
    ]
    story = build_report(st, sections=sections, references=[])
    assert isinstance(story, list)
    assert any(isinstance(item, Paragraph) for item in story)


def test_build_report_handles_string_content_in_non_first_section():
    st = build_styles()
    sections = [
        {
            "title": "Intro",
            "content": "A short intro paragraph.",
        },
        {
            "title": "Conclusion",
            "content": "A concise conclusion paragraph.",
        },
    ]
    story = build_report(st, sections=sections, references=[])
    assert isinstance(story, list)
    assert any(isinstance(item, Paragraph) for item in story)


def test_build_report_parses_markdown_subsections_in_non_first_section():
    st = build_styles()
    sections = [
        {
            "title": "Propulsion",
            "content": (
                "Interstellar travel hinges on propulsion systems capable of achieving significant fractions of light speed.\n\n"
                "### Nuclear Pulse Propulsion\n\n"
                "This method involves detonating nuclear charges behind a spacecraft, pushing it forward via a shock absorber system.\n\n"
                "### Fusion Rockets\n\n"
                "Fusion propulsion requires sustained plasma confinement and ignition, which remains technically elusive."
            ),
        }
    ]
    story = build_report(st, sections=sections, references=[])
    assert isinstance(story, list)
    assert any(
        isinstance(item, Paragraph)
        and getattr(item.style, 'name', '') == 'subhead'
        and 'Nuclear Pulse Propulsion' in item.getPlainText()
        for item in story
    )
    assert any(
        isinstance(item, Paragraph)
        and getattr(item.style, 'name', '') == 'subhead'
        and 'Fusion Rockets' in item.getPlainText()
        for item in story
    )


def test_documenttools_writes_pdf(tmp_path):
    out = os.path.join(tmp_path, "test_report.pdf")
    sections = [{"title": "Intro", "content": "A short paragraph."}]
    ret = DocumentTools.generate_research_pdf(output_path=out, metadata={"topic": "T", "title": "Title"}, sections=sections, references=[])
    assert isinstance(ret, str)
    assert os.path.exists(ret)
    assert os.path.getsize(ret) > 0