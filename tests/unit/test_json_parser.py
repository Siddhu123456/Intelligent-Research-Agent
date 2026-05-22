import pytest

from utils.json_parser import (
    JSONParser,
)


class TestJSONParser:
    """Unit tests for JSONParser."""

    def test_clean_response_plain_json(
        self,
    ):
        """Test cleaning plain JSON."""

        content = (
            '{"answer": "AI response"}'
        )

        result = (
            JSONParser.clean_response(
                content
            )
        )

        assert (
            result
            == '{"answer": "AI response"}'
        )

    def test_clean_response_markdown_json(
        self,
    ):
        """Test markdown JSON cleanup."""

        content = (
            "```json\n"
            "{\n"
            '    "answer": "AI response"\n'
            "}\n"
            "```"
        )

        result = (
            JSONParser.clean_response(
                content
            )
        )

        assert (
            "```"
            not in result
        )

        assert (
            '"answer": "AI response"'
            in result
        )

    def test_clean_response_think_tags(
        self,
    ):
        """Test removal of think tags."""

        content = (
            "<think>\n"
            "Internal reasoning\n"
            "</think>\n\n"
            "{\n"
            '    "answer": "Visible response"\n'
            "}"
        )

        result = (
            JSONParser.clean_response(
                content
            )
        )

        assert (
            "<think>"
            not in result
        )

        assert (
            "Internal reasoning"
            not in result
        )

        assert (
            '"answer": "Visible response"'
            in result
        )

    def test_extract_json_valid(
        self,
    ):
        """Test valid JSON extraction."""

        content = (
            '{"summary": '
            '"AI summary"}'
        )

        result = (
            JSONParser.extract_json(
                content
            )
        )

        assert (
            result["summary"]
            == "AI summary"
        )

    def test_extract_json_markdown(
        self,
    ):
        """Test extraction from markdown."""

        content = (
            "```json\n"
            "{\n"
            '    "title": "AI Research"\n'
            "}\n"
            "```"
        )

        result = (
            JSONParser.extract_json(
                content
            )
        )

        assert (
            result["title"]
            == "AI Research"
        )

    def test_extract_json_embedded_text(
        self,
    ):
        """Test extraction from mixed text."""

        content = (
            "Some explanation text.\n\n"
            "{\n"
            '    "answer": "Final result"\n'
            "}\n\n"
            "Additional notes."
        )

        result = (
            JSONParser.extract_json(
                content
            )
        )

        assert (
            result["answer"]
            == "Final result"
        )

    def test_extract_json_invalid(
        self,
    ):
        """Test invalid JSON raises ValueError."""

        content = (
            "No valid JSON here"
        )

        with pytest.raises(
            ValueError
        ):
            JSONParser.extract_json(
                content
            )

    def test_safe_extract_success(
        self,
    ):
        """Test safe extraction success."""

        content = (
            '{"answer": '
            '"Safe result"}'
        )

        fallback = {
            "answer":
            "Fallback response"
        }

        result = (
            JSONParser.safe_extract(
                content=content,
                fallback=fallback,
            )
        )

        assert (
            result["answer"]
            == "Safe result"
        )

    def test_safe_extract_fallback(
        self,
    ):
        """Test safe extraction fallback."""

        content = (
            "invalid response"
        )

        fallback = {
            "answer":
            "Fallback response"
        }

        result = (
            JSONParser.safe_extract(
                content=content,
                fallback=fallback,
            )
        )

        assert (
            result
            == fallback
        )

    def test_safe_extract_nested_json(
        self,
    ):
        """Test nested JSON extraction."""

        content = (
            "```json\n"
            "{\n"
            '    "metadata": {\n'
            '        "title": "AI",\n'
            '        "year": 2026\n'
            "    }\n"
            "}\n"
            "```"
        )

        result = (
            JSONParser.safe_extract(
                content=content,
                fallback={},
            )
        )

        assert (
            result["metadata"][
                "title"
            ]
            == "AI"
        )

        assert (
            result["metadata"][
                "year"
            ]
            == 2026
        )