import pytest

from tools.document_tools import (
    _classify,
    _inline_fmt,
    _build_styles,
    TitleBanner,
    DocumentTools,
)


class TestDocumentTools:
    """Unit tests for DocumentTools."""

    def test_classify_title(self):
        """Test title classification."""

        kind, text = _classify(
            "# Research Paper"
        )

        assert kind == "title"

        assert text == "Research Paper"

    def test_classify_h1(self):
        """Test H1 classification."""

        kind, text = _classify(
            "## Introduction"
        )

        assert kind == "h1"

        assert text == "Introduction"

    def test_classify_h2(self):
        """Test H2 classification."""

        kind, text = _classify(
            "### Background"
        )

        assert kind == "h2"

        assert text == "Background"

    def test_classify_bullet(self):
        """Test bullet classification."""

        kind, text = _classify(
            "* Bullet item"
        )

        assert kind == "bullet"

        assert text == "Bullet item"

    def test_classify_numbered(self):
        """Test numbered classification."""

        kind, text = _classify(
            "1. First item"
        )

        assert kind == "numbered"

        assert text == "First item"

    def test_classify_body(self):
        """Test body classification."""

        kind, text = _classify(
            "Regular paragraph text"
        )

        assert kind == "body"

        assert text == (
            "Regular paragraph text"
        )

    def test_inline_fmt_bold(self):
        """Test bold inline formatting."""

        result = _inline_fmt(
            "**Bold Text**"
        )

        assert (
            result
            == "<b>Bold Text</b>"
        )

    def test_inline_fmt_italic(self):
        """Test italic inline formatting."""

        result = _inline_fmt(
            "*Italic Text*"
        )

        assert (
            result
            == "<i>Italic Text</i>"
        )

    def test_inline_fmt_combined(self):
        """Test combined formatting."""

        result = _inline_fmt(
            "**Bold** and *Italic*"
        )

        assert (
            "<b>Bold</b>"
            in result
        )

        assert (
            "<i>Italic</i>"
            in result
        )

    def test_build_styles(self):
        """Test style dictionary creation."""

        styles = _build_styles()

        assert isinstance(
            styles,
            dict,
        )

        expected_styles = [
            "body",
            "h1",
            "h2",
            "bullet",
            "numbered",
        ]

        for style_name in (
            expected_styles
        ):
            assert (
                style_name
                in styles
            )

    def test_title_banner_creation(self):
        """Test TitleBanner initialization."""

        banner = TitleBanner(
            title="Research Report",
            subtitle="May 2026",
        )

        assert (
            banner.title
            == "Research Report"
        )

        assert (
            banner.subtitle
            == "May 2026"
        )

    def test_generate_pdf_returns_bytes(self):
        """Test PDF generation."""

        sample_report = """
# Test Research Report

## Introduction

This is a sample report.

## Conclusion

Testing PDF generation.
"""

        pdf_bytes = (
            DocumentTools.generate_pdf(
                sample_report
            )
        )

        assert isinstance(
            pdf_bytes,
            bytes,
        )

        assert (
            len(pdf_bytes) > 0
        )

    def test_generate_pdf_with_bullets(self):
        """Test PDF generation with bullets."""

        sample_report = """
# Research Notes

## Findings

* First finding
* Second finding
* Third finding
"""

        pdf_bytes = (
            DocumentTools.generate_pdf(
                sample_report
            )
        )

        assert isinstance(
            pdf_bytes,
            bytes,
        )

        assert (
            len(pdf_bytes) > 0
        )

    def test_generate_pdf_with_numbered_list(self):
        """Test PDF generation with numbered list."""

        sample_report = """
# Methodology

1. Collect data
2. Process data
3. Generate report
"""

        pdf_bytes = (
            DocumentTools.generate_pdf(
                sample_report
            )
        )

        assert isinstance(
            pdf_bytes,
            bytes,
        )

        assert (
            len(pdf_bytes) > 0
        )

    def test_generate_pdf_with_abstract(self):
        """Test PDF generation with abstract."""

        sample_report = """
# AI Research

Abstract
This is the abstract section.

## Introduction

Introduction content.
"""

        pdf_bytes = (
            DocumentTools.generate_pdf(
                sample_report
            )
        )

        assert isinstance(
            pdf_bytes,
            bytes,
        )

        assert (
            len(pdf_bytes) > 0
        )