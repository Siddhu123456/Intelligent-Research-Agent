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
        
        # Remove persistent title
        # before section extraction.

        report = re.sub(
            r"(?m)^#\s+.*?\n+",
            "",
            report,
            count=1,
        )

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
        report_title: str,
        sections: dict[str, str],
        section_order: list[str],
    ) -> str:
        """
        Reconstruct full report
        while preserving markdown
        hierarchy safely.

        Final structure:

        # Report Title

        ## Section

        ### Subsection
        """

        if not sections:

            return ""

        report_parts = []

        # Persistent document title

        if (
            report_title
            and isinstance(
                report_title,
                str,
            )
            and report_title.strip()
        ):

            cleaned_title = (
                report_title.strip()
            )

            # Prevent accidental duplicate
            # markdown heading markers.

            cleaned_title = re.sub(
                r"^#+\s*",
                "",
                cleaned_title,
            )

            report_parts.append(
                f"# {cleaned_title}\n"
            )

        # Reconstruct sections

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
                .replace(
                    "_",
                    " ",
                )
                .title()
            )

            # Remove accidental
            # duplicated inline titles
            #
            # Example:
            #
            # ## Conclusion
            # Conclusion
            #
            # content...

            lines = (
                content.splitlines()
            )

            if lines:

                first_line = (
                    lines[0]
                    .strip()
                    .lower()
                )

                normalized_title = (
                    formatted_title
                    .strip()
                    .lower()
                )

                if (
                    first_line
                    == normalized_title
                ):

                    content = "\n".join(
                        lines[1:]
                    ).strip()
                    
            # Prevent accidental
            # nested ## headings
            #
            # Convert:
            # ## Something
            #
            # INTO:
            # ### Something
            #
            # INSIDE section content.

            content = re.sub(
                r"(?m)^##\s+",
                "### ",
                content,
            )

            # Ensure spacing consistency

            content = re.sub(
                r"\n{3,}",
                "\n\n",
                content,
            ).strip()

            report_parts.append(
                (
                    f"## "
                    f"{formatted_title}\n\n"
                    f"{content}\n"
                )
            )

        reconstructed_report = (
            "\n".join(
                report_parts
            )
            .strip()
        )

        # Final spacing cleanup

        reconstructed_report = re.sub(
            r"\n{3,}",
            "\n\n",
            reconstructed_report,
        )

        return reconstructed_report

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

        intro_sections = []

        body_sections = []

        ending_sections = []

        for section in sections:

            normalized = (
                section
                .strip()
                .lower()
            )

            # INTRODUCTION-TYPE

            if normalized in {
                "abstract",
                "introduction",
                "executive_summary",
                "background",
                "background_and_context",
            }:

                intro_sections.append(
                    section
                )

            # ENDING-TYPE

            elif normalized in {
                "conclusion",
                "future_directions",
                "future_trends",
                "recommendations",
                "limitations",
                "references",
            }:

                ending_sections.append(
                    section
                )

            # EVERYTHING ELSE
            # becomes BODY section

            else:

                body_sections.append(
                    section
                )

        # References always last

        references = [
            s
            for s in ending_sections
            if s == "references"
        ]

        ending_sections = [
            s
            for s in ending_sections
            if s != "references"
        ]

        ordered_sections = (
            intro_sections
            + body_sections
            + ending_sections
            + references
        )

        return ordered_sections

    @staticmethod
    def delete_section(
        sections: dict[str, str],
        section_order: list[str],
        section_name: str,
    ) -> tuple[dict[str, str], list[str]]:
        """
        Remove a section and its content from the sections
        dict and update the section order list.
        Returns updated (sections, section_order).
        """

        normalized = (
            ReportSectionTools
            .normalize_section_name(
                section_name
            )
        )

        if normalized in sections:
            try:
                # Remove the section
                sections.pop(normalized, None)

                # Remove from ordering if present
                if normalized in section_order:
                    section_order = [s for s in section_order if s != normalized]
            except Exception:
                pass

        return sections, section_order

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