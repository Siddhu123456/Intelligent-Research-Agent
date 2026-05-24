import re


class ReportSectionTools:
    """Tools for structured report sections."""

    DEFAULT_SECTIONS = [
        "Title",
        "Abstract",
        "Introduction",
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
        while preserving markdown
        hierarchy safely.
        """

        if not isinstance(
            report,
            str,
        ):

            report = str(
                report
            )

        report = report.replace(
            "\r\n",
            "\n",
        )

        report = report.strip()

        if not report:

            return {
                "sections": {},
                "section_order": [],
            }

        sections = {}

        section_order = []

        # Only extract TRUE top-level
        # markdown sections (##).
        #
        # Nested subsections (###)
        # remain inside section content.

        pattern = (
            r"(?m)^##\s+(.*?)\n(.*?)(?=^##\s|\Z)"
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

            if not isinstance(
                section_content,
                str,
            ):

                section_content = str(
                    section_content
                )

            cleaned_content = (
                section_content
                .strip()
            )
            
            # Remove duplicated inline titles
            # like:
            #
            # ## Conclusion
            # Conclusion
            #
            # caused by LLM generation.

            first_line_split = (
                cleaned_content.split(
                    "\n",
                    1,
                )
            )

            if (
                len(first_line_split)
                > 1
            ):

                first_line = (
                    first_line_split[0]
                    .strip()
                    .lower()
                )

                normalized_title = (
                    section_name
                    .strip()
                    .lower()
                )

                if (
                    first_line
                    == normalized_title
                ):

                    cleaned_content = (
                        first_line_split[1]
                        .strip()
                    )

            if not cleaned_content:

                continue

            sections[
                cleaned_name
            ] = cleaned_content

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
        Reconstruct full report
        while preserving markdown
        hierarchy safely.
        """

        if not sections:

            return ""

        report_parts = []

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

            if not isinstance(
                content,
                str,
            ):

                content = str(
                    content
                )

            content = content.strip()

            if not content:

                continue

            formatted_title = (
                section_name
                .replace("_", " ")
                .title()
            )

            report_parts.append(
                (
                    f"## "
                    f"{formatted_title}\n\n"
                    f"{content}\n"
                )
            )

        return "\n".join(
            report_parts
        ).strip()

    @staticmethod
    def get_section_order(
        sections: dict[str, str],
    ) -> list[str]:
        """
        Determine intelligent
        dynamic section ordering.
        """

        if not sections:

            return []

        preferred_order = [
            "abstract",
            "introduction",
            "background",
            "technical_foundations",
            "methodology",
            "applications",
            "analysis",
            "ethical_considerations",
            "security_considerations",
            "limitations",
            "future_trends",
            "recommendations",
            "conclusion",
            "references",
        ]

        ordered_sections = []

        for section in (
            preferred_order
        ):

            if section in sections:

                ordered_sections.append(
                    section
                )

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
        Normalize section names
        safely for dynamic reports.
        """

        if not isinstance(
            section_name,
            str,
        ):

            section_name = str(
                section_name
            )

        normalized = (
            section_name
            .strip()
            .lower()
            .replace("&", "and")
            .replace("-", "_")
            .replace(" ", "_")
        )

        normalized = re.sub(
            r"[^a-z0-9_]",
            "",
            normalized,
        )

        while "__" in normalized:

            normalized = (
                normalized.replace(
                    "__",
                    "_",
                )
            )

        return normalized.strip("_")