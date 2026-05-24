import pytest

from tools.report_section_tools import (
    ReportSectionTools,
)


class TestReportSectionTools:
    """Unit tests for ReportSectionTools."""

    def test_extract_sections_success(
        self,
    ):
        """Test successful section extraction."""

        report = """
# Research Report

## Introduction

This is introduction content.

## Conclusion

This is conclusion content.
"""

        result = (
            ReportSectionTools
            .extract_sections(
                report
            )
        )

        assert (
            "introduction"
            in result["sections"]
        )

        assert (
            "conclusion"
            in result["sections"]
        )

        assert (
            result["sections"][
                "introduction"
            ]
            == (
                "This is "
                "introduction content."
            )
        )

    def test_extract_sections_empty_report(
        self,
    ):
        """Test extraction with empty report."""

        result = (
            ReportSectionTools
            .extract_sections(
                ""
            )
        )

        assert (
            result["sections"]
            == {}
        )

        assert (
            result["section_order"]
            == []
        )

    def test_extract_sections_order_preserved(
        self,
    ):
        """Test section order preservation."""

        report = """
## Abstract

Abstract content.

## Introduction

Introduction content.

## Conclusion

Conclusion content.
"""

        result = (
            ReportSectionTools
            .extract_sections(
                report
            )
        )

        assert (
            result["section_order"]
            == [
                "abstract",
                "introduction",
                "conclusion",
            ]
        )

    def test_reconstruct_report_success(
        self,
    ):
        """Test report reconstruction."""

        sections = {
            "introduction":
            "Introduction content.",
            "conclusion":
            "Conclusion content.",
        }

        section_order = [
            "introduction",
            "conclusion",
        ]

        report = (
            ReportSectionTools
            .reconstruct_report(
                "Research Report",
                sections,
                section_order,
            )
        )

        assert (
            "# Research Report"
            in report
        )

        assert (
            "## Introduction"
            in report
        )

        assert (
            "## Conclusion"
            in report
        )

    def test_reconstruct_report_empty_sections(
        self,
    ):
        """Test reconstruction with empty sections."""

        report = (
            ReportSectionTools
            .reconstruct_report(
                "Research Report",
                {},
                [],
            )
        )

        assert report == ""

    def test_get_section_order(
        self,
    ):
        """Test fallback section ordering."""

        sections = {
            "conclusion":
            "Conclusion content",
            "introduction":
            "Introduction content",
            "analysis":
            "Analysis content",
        }

        result = (
            ReportSectionTools
            .get_section_order(
                sections
            )
        )

        assert (
            result[0]
            == "introduction"
        )

        assert (
            "analysis"
            in result
        )

        assert (
            "conclusion"
            in result
        )

    def test_section_exists_true(
        self,
    ):
        """Test existing section detection."""

        sections = {
            "introduction":
            "Content"
        }

        result = (
            ReportSectionTools
            .section_exists(
                sections,
                "Introduction",
            )
        )

        assert result is True

    def test_section_exists_false(
        self,
    ):
        """Test non-existing section detection."""

        sections = {
            "introduction":
            "Content"
        }

        result = (
            ReportSectionTools
            .section_exists(
                sections,
                "Conclusion",
            )
        )

        assert result is False

    def test_normalize_section_name(
        self,
    ):
        """Test section name normalization."""

        result = (
            ReportSectionTools
            .normalize_section_name(
                "Background & Context"
            )
        )

        assert (
            result
            == (
                "background_and_context"
            )
        )

    def test_normalize_section_name_hyphen(
        self,
    ):
        """Test normalization with hyphen."""

        result = (
            ReportSectionTools
            .normalize_section_name(
                "Future-Trends"
            )
        )

        assert (
            result
            == "future_trends"
        )

    def test_normalize_section_name_multiple_spaces(
        self,
    ):
        """Test normalization cleanup."""

        result = (
            ReportSectionTools
            .normalize_section_name(
                "Analysis  &  Insights"
            )
        )

        assert (
            result
            == (
                "analysis_and_insights"
            )
        )