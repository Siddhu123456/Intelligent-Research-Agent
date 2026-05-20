from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    HRFlowable,
    PageBreak,
)
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re


# ── Colour palette ──────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#1A2744")   # headings / rule lines
CHARCOAL= colors.HexColor("#2C2C2C")   # body text
MID     = colors.HexColor("#4A5568")   # secondary text / captions
RULE    = colors.HexColor("#CBD5E0")   # decorative rules
WHITE   = colors.white


def _build_styles() -> dict:
    """Return a dict of named ParagraphStyles for a research document."""
    base = getSampleStyleSheet()

    styles = {}

    # ── Document title (first H1 found) ─────────────────────────────────────
    styles["doc_title"] = ParagraphStyle(
        "doc_title",
        fontName="Times-Bold",
        fontSize=22,
        leading=28,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    # ── Authors / date line ──────────────────────────────────────────────────
    styles["byline"] = ParagraphStyle(
        "byline",
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        textColor=MID,
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    # ── Abstract label ───────────────────────────────────────────────────────
    styles["abstract_heading"] = ParagraphStyle(
        "abstract_heading",
        fontName="Times-Bold",
        fontSize=10,
        leading=14,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceBefore=10,
        spaceAfter=2,
    )

    # ── Abstract body ────────────────────────────────────────────────────────
    styles["abstract_body"] = ParagraphStyle(
        "abstract_body",
        fontName="Times-Roman",
        fontSize=9.5,
        leading=14,
        textColor=CHARCOAL,
        alignment=TA_JUSTIFY,
        leftIndent=0.6 * inch,
        rightIndent=0.6 * inch,
        spaceAfter=12,
    )

    # ── Section heading (##) ─────────────────────────────────────────────────
    styles["h1"] = ParagraphStyle(
        "h1",
        fontName="Times-Bold",
        fontSize=13,
        leading=17,
        textColor=NAVY,
        spaceBefore=16,
        spaceAfter=4,
        keepWithNext=1,
    )

    # ── Sub-section heading (###) ────────────────────────────────────────────
    styles["h2"] = ParagraphStyle(
        "h2",
        fontName="Times-BoldItalic",
        fontSize=11,
        leading=15,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=3,
        keepWithNext=1,
    )

    # ── Body text ────────────────────────────────────────────────────────────
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Times-Roman",
        fontSize=11,
        leading=16,
        textColor=CHARCOAL,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        firstLineIndent=0.25 * inch,
    )

    # ── First paragraph after a heading (no indent) ──────────────────────────
    styles["body_first"] = ParagraphStyle(
        "body_first",
        parent=styles["body"],
        firstLineIndent=0,
    )

    # ── Bulleted / list items ─────────────────────────────────────────────────
    styles["bullet"] = ParagraphStyle(
        "bullet",
        fontName="Times-Roman",
        fontSize=11,
        leading=16,
        textColor=CHARCOAL,
        leftIndent=0.35 * inch,
        spaceAfter=3,
        bulletIndent=0.15 * inch,
    )

    # ── Numbered list ────────────────────────────────────────────────────────
    styles["numbered"] = ParagraphStyle(
        "numbered",
        parent=styles["bullet"],
    )

    # ── Caption / note ───────────────────────────────────────────────────────
    styles["caption"] = ParagraphStyle(
        "caption",
        fontName="Times-Italic",
        fontSize=9,
        leading=12,
        textColor=MID,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    return styles


# ── Header / footer canvas callbacks ────────────────────────────────────────

def _make_header_footer(title: str):
    """Return onFirstPage / onLaterPages callbacks for headers and footers."""

    def _header_footer(canvas, doc):
        canvas.saveState()
        w, h = letter

        # ── Header rule (skip page 1 title area) ───────────────────────────
        if doc.page > 1:
            canvas.setStrokeColor(NAVY)
            canvas.setLineWidth(0.75)
            canvas.line(0.85 * inch, h - 0.65 * inch,
                        w - 0.85 * inch, h - 0.65 * inch)

            # Running title
            canvas.setFont("Times-Italic", 8.5)
            canvas.setFillColor(MID)
            short = title[:72] + "…" if len(title) > 72 else title
            canvas.drawString(0.85 * inch, h - 0.52 * inch, short)

        # ── Footer rule ──────────────────────────────────────────────────────
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(0.85 * inch, 0.65 * inch,
                    w - 0.85 * inch, 0.65 * inch)

        # Page number
        canvas.setFont("Times-Roman", 9)
        canvas.setFillColor(MID)
        canvas.drawCentredString(w / 2, 0.45 * inch, str(doc.page))

        canvas.restoreState()

    return _header_footer, _header_footer


# ── Line classifier ──────────────────────────────────────────────────────────

def _classify(line: str):
    """Return (kind, text) for a stripped line."""
    if line.startswith("### "):
        return "h2", line[4:].strip()
    if line.startswith("## "):
        return "h1", line[3:].strip()
    if line.startswith("# "):
        return "title", line[2:].strip()
    if re.match(r"^\*\s+|^-\s+|^•\s+", line):
        return "bullet", re.sub(r"^[\*\-•]\s+", "", line)
    if re.match(r"^\d+\.\s+", line):
        return "numbered", re.sub(r"^\d+\.\s+", "", line)
    if line.lower().startswith("abstract"):
        return "abstract_heading", line
    return "body", line


def _inline_fmt(text: str) -> str:
    """Convert **bold** and *italic* markdown to ReportLab XML."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    return text


# ── Main class ───────────────────────────────────────────────────────────────

class DocumentTools:
    """Tools for professional research-document generation."""

    @staticmethod
    def generate_pdf(report: str) -> bytes:
        """
        Generate a polished research-document PDF from *report*.

        Supported markdown-lite syntax
        ──────────────────────────────
        # Title          → document title (centred, large)
        ## Section       → numbered section heading
        ### Subsection   → italic sub-heading
        * / - item       → bullet list
        1. item          → numbered list
        Abstract …       → styled abstract block
        **bold**         → bold inline
        *italic*         → italic inline
        blank line       → paragraph break
        """
        styles = _build_styles()
        pdf_buffer = BytesIO()

        # Detect document title for running header
        running_title = "Research Document"
        for line in report.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                running_title = stripped[2:].strip()
                break

        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            leftMargin=0.85 * inch,
            rightMargin=0.85 * inch,
            topMargin=1.05 * inch,
            bottomMargin=0.85 * inch,
            title=running_title,
        )

        on_first, on_later = _make_header_footer(running_title)

        elements = []
        lines = report.splitlines()

        title_done       = False
        in_abstract      = False
        abstract_lines   = []
        prev_was_heading = False
        section_counter  = 0

        def flush_abstract():
            nonlocal in_abstract
            if abstract_lines:
                elements.append(
                    Paragraph("Abstract", styles["abstract_heading"])
                )
                elements.append(
                    HRFlowable(
                        width="55%",
                        thickness=0.5,
                        color=RULE,
                        spaceAfter=6,
                        hAlign="CENTER",
                    )
                )
                elements.append(
                    Paragraph(
                        _inline_fmt(" ".join(abstract_lines)),
                        styles["abstract_body"],
                    )
                )
                abstract_lines.clear()
            in_abstract = False

        for raw in lines:
            stripped = raw.strip()

            # Blank line → paragraph break / flush abstract
            if not stripped:
                if in_abstract:
                    flush_abstract()
                prev_was_heading = False
                continue

            kind, text = _classify(stripped)

            # ── Document title ───────────────────────────────────────────────
            if kind == "title" and not title_done:
                elements.append(Spacer(1, 0.15 * inch))
                elements.append(Paragraph(_inline_fmt(text),
                                           styles["doc_title"]))
                elements.append(
                    HRFlowable(
                        width="70%",
                        thickness=1.5,
                        color=NAVY,
                        spaceBefore=6,
                        spaceAfter=10,
                        hAlign="CENTER",
                    )
                )
                title_done = True
                prev_was_heading = True
                continue

            # ── Subsequent # lines treated as section headings ───────────────
            if kind == "title" and title_done:
                kind = "h1"

            # ── Abstract detection ───────────────────────────────────────────
            if kind == "abstract_heading":
                in_abstract = True
                prev_was_heading = True
                continue

            if in_abstract:
                abstract_lines.append(text)
                continue

            # ── Section headings ─────────────────────────────────────────────
            if kind == "h1":
                flush_abstract()
                section_counter += 1
                numbered_text = f"{section_counter}. {_inline_fmt(text).upper()}"
                elements.append(Paragraph(numbered_text, styles["h1"]))
                elements.append(
                    HRFlowable(
                        width="100%",
                        thickness=0.5,
                        color=RULE,
                        spaceAfter=4,
                    )
                )
                prev_was_heading = True
                continue

            if kind == "h2":
                flush_abstract()
                elements.append(
                    Paragraph(_inline_fmt(text), styles["h2"])
                )
                prev_was_heading = True
                continue

            # ── Bullet list ──────────────────────────────────────────────────
            if kind == "bullet":
                flush_abstract()
                elements.append(
                    Paragraph(
                        f"<bullet>\u2022</bullet>{_inline_fmt(text)}",
                        styles["bullet"],
                    )
                )
                prev_was_heading = False
                continue

            # ── Numbered list ────────────────────────────────────────────────
            if kind == "numbered":
                flush_abstract()
                # preserve original number
                m = re.match(r"^(\d+)\.\s+(.*)", stripped)
                num  = m.group(1) if m else "1"
                body_text = m.group(2) if m else text
                elements.append(
                    Paragraph(
                        f"<bullet>{num}.</bullet>{_inline_fmt(body_text)}",
                        styles["numbered"],
                    )
                )
                prev_was_heading = False
                continue

            # ── Body paragraph ───────────────────────────────────────────────
            flush_abstract()
            style = styles["body_first"] if prev_was_heading else styles["body"]
            elements.append(Paragraph(_inline_fmt(text), style))
            prev_was_heading = False

        # Flush any trailing abstract block
        flush_abstract()

        # Build
        doc.build(
            elements,
            onFirstPage=on_first,
            onLaterPages=on_later,
        )
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE = """
# The Role of Large Language Models in Scientific Discovery

*Dr. A. Rahman, Dr. P. Singh, Dr. L. Chen — May 2026*

Abstract
Large language models (LLMs) have rapidly moved from novelty to indispensable
tool in many scientific domains. This paper surveys the mechanisms by which
LLMs accelerate hypothesis generation, literature synthesis, and experimental
design across disciplines including biology, chemistry, and materials science.
We identify key limitations and propose a research agenda to address them.

## Introduction

The emergence of transformer-based language models has fundamentally altered
the landscape of computational science. Researchers now routinely employ these
systems for tasks that once required months of expert effort.

This paper makes three primary contributions to the literature. First, we
provide a **systematic taxonomy** of LLM use-cases in scientific research.
Second, we present a *meta-analysis* of 142 empirical studies. Third, we
outline a forward-looking research agenda.

## Background

### Transformer Architecture

The transformer architecture, introduced by Vaswani et al. (2017), relies on
self-attention mechanisms to model long-range dependencies in sequential data.
Modern LLMs scale this approach to hundreds of billions of parameters.

### Prior Work

Several survey articles have examined AI in science broadly. Our work differs
by focusing specifically on language-mediated discovery workflows.

## Methodology

We conducted a systematic review of peer-reviewed articles published between
2019 and 2025. Inclusion criteria were:

* Primary research studies involving an LLM and a scientific task
* At least one quantitative outcome measure
* Published in a peer-reviewed venue

Studies were retrieved from three databases:

1. Semantic Scholar
2. PubMed Central
3. arXiv (cs.AI, q-bio, cond-mat)

## Results

Our analysis reveals a clear trend toward **multi-modal** systems that combine
language understanding with structured data retrieval. In chemistry, LLM-guided
retrosynthesis planning reduced expert iteration cycles by an average of 38%.

### Biology Sub-domain

In genomics, sequence-aware LLMs demonstrated strong zero-shot performance on
functional annotation tasks, approaching fine-tuned specialist models.

### Materials Science Sub-domain

Crystal structure prediction benefited most from LLM-generated hypothesis
priors, cutting the search space explored by DFT calculations significantly.

## Discussion

These results suggest that LLMs are most valuable as *reasoning augmenters*
rather than autonomous discovery engines. Human–AI collaboration frameworks
consistently outperform either party in isolation.

## Conclusion

Large language models represent a transformative but **not unlimited** tool in
scientific discovery. Continued progress requires investment in grounding,
uncertainty quantification, and reproducibility standards.

## References

Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS 30*.
Brown, T. et al. (2020). Language models are few-shot learners. *NeurIPS 33*.
"""

    pdf_bytes = DocumentTools.generate_pdf(SAMPLE)
    with open("/mnt/user-data/outputs/research_document.pdf", "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF written ({len(pdf_bytes):,} bytes)")