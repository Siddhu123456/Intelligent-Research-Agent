from io import (
    BytesIO,
)

from reportlab.lib.pagesizes import (
    letter,
)

from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
)


class DocumentTools:
    """Tools for professional document generation."""

    @staticmethod
    def generate_pdf(
        report: str,
    ) -> bytes:
        """Generate PDF from report."""

        pdf_buffer = BytesIO()

        document = (
            SimpleDocTemplate(
                pdf_buffer,
                pagesize=letter,
            )
        )

        styles = (
            getSampleStyleSheet()
        )

        elements = []

        for line in report.split(
            "\n",
        ):

            stripped_line = (
                line.strip()
            )

            if not stripped_line:

                elements.append(
                    Spacer(
                        1,
                        12,
                    )
                )

                continue

            paragraph = (
                Paragraph(
                    stripped_line,
                    styles["BodyText"],
                )
            )

            elements.append(
                paragraph
            )

            elements.append(
                Spacer(
                    1,
                    8,
                )
            )

        document.build(
            elements,
        )

        pdf_buffer.seek(
            0,
        )

        return pdf_buffer.getvalue()