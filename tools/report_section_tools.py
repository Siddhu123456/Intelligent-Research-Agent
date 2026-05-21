import re


class ReportSectionTools:
    """Tools for structured report sections."""

    DEFAULT_SECTIONS = [
        "Title",
        "Executive Summary",
        "Background & Context",
        "Key Findings",
        "Analysis & Insights",
        "Conclusion",
        "References",
    ]

    @staticmethod
    def extract_sections(
        report: str,
    ) -> dict:
        """
        Extract structured sections
        and preserve original order.
        """

        if not report.strip():

            return {
                "sections": {},
                "section_order": [],
            }

        sections = {}

        section_order = []

        pattern = (
            r"##\s+(.*?)\n(.*?)(?=\n##|\Z)"
        )

        matches = re.findall(
            pattern,
            report,
            flags=re.DOTALL,
        )

        for (
            section_name,
            section_content,
        ) in matches:

            cleaned_name = (
                ReportSectionTools
                .normalize_section_name(
                    section_name
                )
            )

            sections[
                cleaned_name
            ] = (
                section_content
                .strip()
            )

            section_order.append(
                cleaned_name
            )

        return {
            "sections": sections,
            "section_order": (
                section_order
            ),
        }

    @staticmethod
    def reconstruct_report(
        sections: dict[str, str],
        section_order: list[str],
    ) -> str:
        """
        Rebuild full report using
        persistent section order.
        """

        if not sections:

            return ""

        report_parts = [
            "# Research Report\n"
        ]

        for section_name in (
            section_order
        ):

            if (
                section_name
                not in sections
            ):

                continue

            content = sections.get(
                section_name,
                "",
            )

            if not content.strip():

                continue

            formatted_title = (
                section_name
                .replace("_", " ")
                .title()
            )

            report_parts.append(
                (
                    f"\n## "
                    f"{formatted_title}\n\n"
                    f"{content}\n"
                )
            )

        return "".join(
            report_parts
        ).strip()

    @staticmethod
    def get_section_order(
        sections: dict[str, str],
    ) -> list[str]:
        """
        Determine fallback section ordering.
        """

        preferred_order = [
            "title",
            "executive_summary",
            "background_and_context",
            "background",
            "key_findings",
            "analysis_and_insights",
            "analysis",
            "future_trends",
            "security_considerations",
            "limitations",
            "deployment_strategy",
            "recommendations",
            "conclusion",
            "references",
        ]

        ordered_sections = []

        # Add preferred sections first

        for section in (
            preferred_order
        ):

            if section in sections:

                ordered_sections.append(
                    section
                )

        # Add remaining sections

        for section in sections:

            if (
                section
                not in ordered_sections
            ):

                ordered_sections.append(
                    section
                )

        return ordered_sections

    @staticmethod
    def section_exists(
        sections: dict[str, str],
        section_name: str,
    ) -> bool:
        """
        Check whether section exists.
        """

        normalized_name = (
            ReportSectionTools
            .normalize_section_name(
                section_name
            )
        )

        return (
            normalized_name
            in sections
        )

    @staticmethod
    def normalize_section_name(
        section_name: str,
    ) -> str:
        """
        Normalize section names.
        """

        normalized = (
            section_name
            .strip()
            .lower()
            .replace("&", "and")
            .replace("-", "_")
            .replace(" ", "_")
        )

        while "__" in normalized:

            normalized = (
                normalized.replace(
                    "__",
                    "_",
                )
            )

        return normalized