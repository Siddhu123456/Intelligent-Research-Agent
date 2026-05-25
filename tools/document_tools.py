import os
import re

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    PageBreak, Table, TableStyle, KeepTogether
)
from reportlab.lib.colors import HexColor, black, white

# ── Palette (white pages, refined editorial) ──────────────────────────────────
INK          = HexColor("#111111")
INK_LIGHT    = HexColor("#444444")
MUTED        = HexColor("#888888")
RULE_DARK    = HexColor("#111111")
RULE_LIGHT   = HexColor("#DDDDDD")
ACCENT_RED   = HexColor("#C0392B")   # single bold accent colour
BG_TAG       = HexColor("#F4F4F4")
CODE_BG      = HexColor("#F0F0F0")
CODE_INK     = HexColor("#222222")
PAGE_W, PAGE_H = A4
ML = 22 * mm   # left margin (wider for feel)
MR = 18 * mm
MT = 24 * mm
MB = 20 * mm


# ── Page decorator ─────────────────────────────────────────────────────────────
def make_page_decorator(metadata):
    """Return a page callback that uses the provided metadata."""
    header_left = "RESEARCH REPORT"
    header_right = metadata.get("date", "")

    def make_page(canvas, doc):
        canvas.saveState()

        if doc.page == 1:
            draw_cover(canvas, metadata)
            canvas.restoreState()
            return

        canvas.setFillColor(white)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

        canvas.setFillColor(ACCENT_RED)
        canvas.rect(0, 0, 3.5, PAGE_H, fill=1, stroke=0)

        canvas.setStrokeColor(RULE_DARK)
        canvas.setLineWidth(0.75)
        canvas.line(ML, PAGE_H - MT + 4, PAGE_W - MR, PAGE_H - MT + 4)

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(ML, PAGE_H - MT + 7, header_left)
        if header_right:
            canvas.drawRightString(PAGE_W - MR, PAGE_H - MT + 7, header_right)

        canvas.setStrokeColor(RULE_LIGHT)
        canvas.setLineWidth(0.5)
        canvas.line(ML, MB - 4, PAGE_W - MR, MB - 4)

        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(INK)
        canvas.drawRightString(PAGE_W - MR, MB - 11, str(doc.page))

        canvas.restoreState()

    return make_page


# ── Styles ─────────────────────────────────────────────────────────────────────
def build_styles():
    def s(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        # ---- Cover ----
        "cv_eyebrow": s("cv_eyebrow",
            fontName="Helvetica", fontSize=8, textColor=ACCENT_RED,
            spaceAfter=10, alignment=TA_LEFT, leading=11, letterSpacing=2,
        ),
        "cv_title": s("cv_title",
            fontName="Helvetica-Bold", fontSize=38, textColor=INK,
            spaceAfter=0, alignment=TA_LEFT, leading=44, wordWrap='CJK',
        ),
        "cv_subtitle": s("cv_subtitle",
            fontName="Helvetica", fontSize=13, textColor=INK_LIGHT,
            spaceAfter=6, alignment=TA_LEFT, leading=18,
        ),
        "cv_meta_label": s("cv_meta_label",
            fontName="Helvetica-Bold", fontSize=7.5, textColor=MUTED,
            leading=11, letterSpacing=1,
        ),
        "cv_meta_value": s("cv_meta_value",
            fontName="Helvetica", fontSize=9, textColor=INK,
            leading=13,
        ),
        # ---- Section headers ----
        "sec_number": s("sec_number",
            fontName="Helvetica-Bold", fontSize=8, textColor=ACCENT_RED,
            spaceBefore=18, spaceAfter=1, leading=11, letterSpacing=1,
        ),
        "sec_title": s("sec_title",
            fontName="Helvetica-Bold", fontSize=16, textColor=INK,
            spaceAfter=2, leading=20, wordWrap='CJK',
        ),
        # ---- Sub-heading ----
        "subhead": s("subhead",
            fontName="Helvetica-Bold", fontSize=10, textColor=INK,
            spaceBefore=10, spaceAfter=3, leading=14,
        ),
        # ---- Body ----
        "body": s("body",
            fontName="Helvetica", fontSize=9.5, textColor=INK_LIGHT,
            spaceAfter=8, leading=15.5, alignment=TA_JUSTIFY,
        ),
        # ---- Inline code ----
        "code": s("code",
            fontName="Courier", fontSize=8.5, textColor=CODE_INK,
            spaceAfter=6, leading=13, backColor=CODE_BG,
            leftIndent=6, rightIndent=6, borderPadding=(3,3,3,3),
        ),
        # ---- References ----
        "ref_title": s("ref_title",
            fontName="Helvetica-Bold", fontSize=9, textColor=INK,
            spaceAfter=1, leading=13,
        ),
        "ref_url": s("ref_url",
            fontName="Courier", fontSize=7.5, textColor=MUTED,
            spaceAfter=9, leading=11,
        ),
        # ---- Abstract callout ----
        "abstract": s("abstract",
            fontName="Helvetica", fontSize=10, textColor=INK,
            spaceAfter=10, leading=16.5, alignment=TA_JUSTIFY,
            leftIndent=0, rightIndent=0,
        ),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────
def hrule(color=RULE_LIGHT, t=0.5, above=4, below=4):
    return HRFlowable(width="100%", thickness=t, color=color,
                      spaceBefore=above, spaceAfter=below)

def section_header(num, title, st):
    return KeepTogether([
        Spacer(1, 2*mm),
        Paragraph(f"— {num:02d} —", st["sec_number"]),
        Paragraph(title, st["sec_title"]),
        hrule(RULE_DARK, t=1.0, above=4, below=10),
    ])

def body(text, st):
    return Paragraph(text, st["body"])

def subhead(text, st):
    return Paragraph(text, st["subhead"])

def tag_row(items, st):
    """Render small pill-style tags in a single paragraph."""
    pills = "  ".join(
        f'<font color="#C0392B">▪</font> {i}' for i in items
    )
    return Paragraph(pills, ParagraphStyle("tags",
        fontName="Helvetica", fontSize=8, textColor=INK_LIGHT,
        leading=13, spaceAfter=10,
    ))


# ── Cover page (drawn in onFirstPage callback) ────────────────────────────────
def draw_cover(canvas, metadata):
    """Called from onFirstPage — draws directly onto the canvas."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    c = canvas
    md = metadata
    topic    = md.get("topic", "Research")
    title    = md.get("title",  topic).replace("&amp;", "&").replace("\n", " ")
    subtitle = md.get("subtitle", "").replace("&amp;", "&")
    W, H = PAGE_W, PAGE_H

    # ── White background ───────────────────────────────────────────────────
    c.setFillColor(white)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # ── Left black editorial band ──────────────────────────────────────────
    BAND = 62 * mm
    c.setFillColor(INK)
    c.rect(0, 0, BAND, H, fill=1, stroke=0)

    # ── Red accent stripe on right edge of band ────────────────────────────
    c.setFillColor(ACCENT_RED)
    c.rect(BAND, 0, 3.5, H, fill=1, stroke=0)

    # ── Vertical "RESEARCH REPORT" rotated text in band ───────────────────
    c.saveState()
    c.setFillColor(HexColor("#3a3a3a"))
    c.setFont("Helvetica", 7)
    c.translate(11 * mm, H / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "INTELLIGENT RESEARCH  ·  TECHNICAL  ANALYSIS")
    c.restoreState()

    # ── Giant faint topic word (watermark in band) ─────────────────────────
    c.saveState()
    c.setFillColor(HexColor("#1c1c1c"))
    c.setFont("Helvetica-Bold", 64)
    c.translate(BAND / 2, H * 0.5)
    c.rotate(90)
    c.drawCentredString(0, -20, "RESEARCH REPORT")
    c.restoreState()

    # ── Right content area ─────────────────────────────────────────────────
    CX = BAND + 3.5 + 14 * mm
    CW = W - CX - 14 * mm

    # ── Top red eyebrow label ──────────────────────────────────────────────
    TOP_Y = H - 26 * mm
    c.setFillColor(ACCENT_RED)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(CX, TOP_Y, "RESEARCH REPORT")

    # ── Hairline red rule ──────────────────────────────────────────────────
    c.setStrokeColor(ACCENT_RED)
    c.setLineWidth(0.8)
    c.line(CX, TOP_Y - 5, CX + CW, TOP_Y - 5)

    # ── Main title — bold, word-wrapped ────────────────────────────────────
    TITLE_SIZE = 34
    TITLE_LEAD = 40
    c.setFont("Helvetica-Bold", TITLE_SIZE)
    c.setFillColor(INK)

    words = title.split()
    lines, line = [], ""
    for w in words:
        test = (line + " " + w).strip()
        if stringWidth(test, "Helvetica-Bold", TITLE_SIZE) <= CW:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)

    TITLE_ANCHOR_Y = H * 0.60
    for i, ln in enumerate(lines):
        c.drawString(CX, TITLE_ANCHOR_Y - i * TITLE_LEAD, ln)

    title_bottom = TITLE_ANCHOR_Y - (len(lines) - 1) * TITLE_LEAD

    # ── Bold rule under title ──────────────────────────────────────────────
    rule_y = title_bottom - 10
    c.setStrokeColor(INK)
    c.setLineWidth(1.5)
    c.line(CX, rule_y, CX + CW, rule_y)

    # Light secondary rule
    c.setStrokeColor(RULE_LIGHT)
    c.setLineWidth(0.5)
    c.line(CX, rule_y - 3, CX + CW, rule_y - 3)

    # ── Subtitle ───────────────────────────────────────────────────────────
    if subtitle:
        SUB_SIZE = 9.5
        c.setFont("Helvetica", SUB_SIZE)
        c.setFillColor(INK_LIGHT)
        sub_words = subtitle.split()
        sub_lines, sub_line = [], ""
        for w in sub_words:
            test = (sub_line + " " + w).strip()
            if stringWidth(test, "Helvetica", SUB_SIZE) <= CW:
                sub_line = test
            else:
                if sub_line:
                    sub_lines.append(sub_line)
                sub_line = w
        if sub_line:
            sub_lines.append(sub_line)
        for i, ln in enumerate(sub_lines):
            c.drawString(CX, rule_y - 18 - i * 14, ln)

    # ── Bottom footer bar ──────────────────────────────────────────────────
    FOOT_H = 17 * mm
    c.setFillColor(BG_TAG)
    c.rect(BAND + 3.5, 0, W - BAND - 3.5, FOOT_H, fill=1, stroke=0)
    c.setStrokeColor(RULE_LIGHT)
    c.setLineWidth(0.5)
    c.line(CX, FOOT_H, W - 14*mm, FOOT_H)
    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    c.drawString(CX, FOOT_H / 2 - 2,
                 "Generated by Research Engine  ·  All sources cited in references")

    # ── Page 01 label in band ──────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(HexColor("#555555"))
    c.drawCentredString(BAND / 2, 10 * mm, "01")


def build_cover(st, metadata):
    """Cover is drawn in the page callback; just emit a PageBreak to start page 2."""
    return [PageBreak()]


# ── Report body ────────────────────────────────────────────────────────────────
def build_report(st, sections=None, references=None):
    """
    Build the report body.

    Parameters
    ----------
    sections : list of dicts, optional
        Each dict: { "title": str, "content": str | list[{"heading": str|None, "text": str}] }
        The first section is rendered as an abstract callout box.
        If omitted, falls back to the built-in Groq demo content.
    references : list of dicts, optional
        Each dict: { "title": str, "url": str }
        If omitted, falls back to the built-in Groq demo references.
    """
    if sections is None:
        sections = []

    if references is None:
        references = []

    story = []

    for i, sec in enumerate(sections):
        story.append(section_header(i + 1, sec["title"], st))
        content = sec["content"]

        if i == 0:
            # First section → abstract callout box
            text = content if isinstance(content, str) else " ".join(
                b.get("text", "") for b in content
            )
            abstract_table = Table(
                [[Paragraph(text, st["abstract"])]],
                colWidths=[PAGE_W - ML - MR]
            )
            abstract_table.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), BG_TAG),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
                ("TOPPADDING",    (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LINERIGHT",     (0, 0), (0,  -1),  3, ACCENT_RED),
            ]))
            story.append(abstract_table)
            story.append(Spacer(1, 4 * mm))

        elif isinstance(
            content,
            str,
        ):

            lines = (
                content.split(
                    "\n"
                )
            )

            paragraph_buffer = []

            def flush_paragraph():

                if (
                    paragraph_buffer
                ):

                    paragraph_text = (
                        " ".join(
                            paragraph_buffer
                        )
                        .strip()
                    )

                    if paragraph_text:

                        story.append(
                            body(
                                paragraph_text,
                                st,
                            )
                        )

                    paragraph_buffer.clear()

            for line in lines:

                stripped = (
                    line.strip()
                )

                if not stripped:

                    flush_paragraph()

                    continue

                # ### Markdown subheads

                if re.match(
                    r"^#{3,4}\s+",
                    stripped,
                ):

                    flush_paragraph()

                    heading_text = re.sub(
                        r"^#{3,4}\s+",
                        "",
                        stripped,
                    ).strip()

                    if heading_text:

                        story.append(
                            subhead(
                                heading_text,
                                st,
                            )
                        )

                    continue

                paragraph_buffer.append(
                    stripped
                )

            flush_paragraph()

        else:
            for block in content:
                if block.get("heading"):
                    story.append(subhead(block["heading"], st))
                story.append(body(block["text"], st))

    # References

    if references:

        story.append(
            PageBreak()
        )

        story.append(
            section_header(
                len(sections) + 1,
                "References",
                st,
            )
        )

        for ref in references:

            story.append(
                Paragraph(
                    ref["title"],
                    st["ref_title"],
                )
            )

            story.append(
                Paragraph(
                    ref["url"],
                    st["ref_url"],
                )
            )

    return story



# ── Build ──────────────────────────────────────────────────────────────────────
def generate(out, metadata=None, sections=None, references=None):
    """
    Generate the research report PDF.

    Parameters
    ----------
    out : str
        Output file path — any valid path on your filesystem, e.g.:
            "C:/Users/you/Documents/report.pdf"   (Windows)
            "/home/you/report.pdf"                 (Linux/Mac)

    metadata : dict, optional
        Cover page data. Only keys you pass are rendered.
        Recognised keys:
          topic    – short subject label (also used in running header)
          title    – full cover title (falls back to topic)
          subtitle – tagline beneath the title
          date     – publication / generation date
        Example:
          {
              "topic":    "Groq",
              "title":    "Groq AI Architecture & Edge Inference",
              "subtitle": "Low-Latency Computing · LPU Design",
          }

    sections : list of dicts, optional
        Report body content. Each dict:
          { "title": str,
            "content": str  OR  list[{"heading": str|None, "text": str}] }
        The first section is rendered as an abstract callout box.
        If omitted, built-in Groq demo content is used.

    references : list of dicts, optional
        Each dict: { "title": str, "url": str }
        If omitted, built-in Groq demo references are used.
    """
    if metadata is None:
        metadata = {}

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    page_fn = make_page_decorator(metadata)

    doc = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=MT, bottomMargin=MB,
        title=metadata.get("title", metadata.get("topic", "Research Report")),
        author="Research Engine",
    )
    st = build_styles()
    story = build_cover(st, metadata) + build_report(st, sections, references)
    doc.build(story, onFirstPage=page_fn, onLaterPages=page_fn)
    print(f"Done → {out}")


class DocumentTools:
    """Tools for document generation."""

    @staticmethod
    def generate_research_pdf(
        output_path: str,
        metadata: dict | None = None,
        sections: list | None = None,
        references: list | None = None,
    ) -> str:
        """
        Generate research PDF document.
        """

        generate(
            out=output_path,
            metadata=metadata,
            sections=sections,
            references=references,
        )

        return output_path


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":

    import tempfile

    out = os.path.join(
        tempfile.gettempdir(),
        "research_report.pdf",
    )

    DocumentTools.generate_research_pdf(
        output_path=out,
        metadata={
            "topic": "Groq",
            "title": (
                "Groq AI Architecture "
                "& Edge Inference"
            ),
            "subtitle": (
                "Low-Latency Computing "
                "· LPU Design"
            ),
        },
    )

    print(
        f"Saved to {out}"
    )