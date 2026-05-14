"""Build the thesis as a single PDF file from per-chapter Markdown.

Output: thesis.pdf, thesis.docx (intermediate).

Pipeline:
1. Read 01_abstract.md ... 07_references.md (skip 00_title.md — title page is
   built separately by `make_styles.py` for full control over layout).
2. Replace `[@bibkey]` / `[@k1; @k2]` citation markers with numeric `[N]`
   refs based on the CITATIONS mapping.
3. Concatenate into a single Markdown stream.
4. Convert to DOCX via pypandoc, using `reference.docx` to apply AITU thesis
   styles (Times New Roman 12pt, 1.5 spacing, A4, 30/25/30/20 mm margins,
   centered page numbers).
5. Prepend `title_page.docx` to the pandoc output via docxcompose so the
   final thesis.docx starts with a proper AITU title page.
6. Convert thesis.docx → thesis.pdf via Word COM (docx2pdf).

If reference.docx or title_page.docx are missing, `make_styles.py` is invoked
to regenerate them.

Usage:
    venv\\Scripts\\python.exe thesis\\build.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pypandoc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent

# Map of @bibkey -> reference number (matches order in 07_references.md).
CITATIONS: dict[str, int] = {
    "DeepForest2019": 1,
    "AbbasYOLO2025": 2,
    "Ventura2024": 3,
    "VelasquezCamacho2023": 4,
    "VelasquezCamacho2025": 5,
    "SofiaDeepForest2024": 6,
    "Dakov2024": 6,  # alias
    "Sun2025": 7,
    "Lv2023": 8,
    "LvMCAN2023": 8,  # alias
    "Wang2021": 9,
    "Martins2021": 10,
    "Martins2021Species": 10,
    "dosSantos2019": 11,
    "Zheng2022": 12,
    "Xia2021": 13,
    "He2022": 14,
    "He2020": 15,
    "Xu2025": 16,
    "Awad2021": 17,
    "Branson2019": 18,
    "Zhang2022": 19,
    "Zhang2024": 20,
    "Chen2022": 21,
    "Chen2023": 22,
    "Schmohl2022": 23,
    "Huerta2021": 24,
    "SAM2023": 25,
    "WBF2021": 26,
    "FasterRCNN2015": 27,
    "MaskRCNN2017": 28,
    "RetinaNet2017": 29,
    "YOLOv1": 30,
    "UNet2015": 31,
    "UltralyticsYOLO2023": 32,
    "Timilsina2020": 33,
    "Chen2021": 34,
    "Dong2019": 35,
    "SAM2_2024": 36,
    "Ravi2024": 36,  # alias
    "MaskRCNN2017_b": 28,  # alias of MaskRCNN2017 for Mask R-CNN section
}

CITE_GROUP_RE = re.compile(r"\[((?:@[A-Za-z0-9_]+(?:\s*;\s*)?)+)\]")
SINGLE_CITE_RE = re.compile(r"@([A-Za-z0-9_]+)")


def replace_citations(text: str) -> str:
    def sub_group(match: re.Match) -> str:
        keys = [m.group(1) for m in SINGLE_CITE_RE.finditer(match.group(1))]
        nums = []
        for k in keys:
            if k not in CITATIONS:
                print(f"WARN: unknown citation key '@{k}'", file=sys.stderr)
                nums.append("?")
            else:
                nums.append(str(CITATIONS[k]))
        return "[" + ", ".join(nums) + "]"
    return CITE_GROUP_RE.sub(sub_group, text)


def ensure_styles() -> None:
    """Re-run make_styles.py if reference.docx or title_page.docx missing."""
    need = not (HERE / "reference.docx").exists() \
        or not (HERE / "title_page.docx").exists()
    if need:
        print("Generating reference.docx and title_page.docx ...")
        subprocess.run(
            [sys.executable, str(HERE / "make_styles.py")], check=True
        )


def combine_markdown() -> tuple[Path, str]:
    """Read 01..07 .md, replace citations, write thesis_full.md."""
    files = sorted(
        p for p in HERE.iterdir()
        if (p.suffix == ".md"
            and not p.name.startswith(("thesis_", "thesis."))
            and not p.name.startswith("00_"))
    )
    print(f"Reading {len(files)} content files:")
    for p in files:
        print(f"  - {p.name}")
    chunks = [replace_citations(p.read_text(encoding="utf-8")) for p in files]
    combined = "\n\n".join(chunks)
    out = HERE / "thesis_full.md"
    out.write_text(combined, encoding="utf-8")
    print(f"Wrote combined markdown: {out} "
          f"({len(combined):,} chars, {combined.count(chr(10)) + 1:,} lines)")
    return out, combined


def render_docx(combined: str, out: Path) -> None:
    """Pandoc markdown → docx using reference.docx for styles."""
    extra_args = [
        "--reference-doc", str(HERE / "reference.docx"),
        "--standalone",
        "--toc",
        "--toc-depth=2",
        "-M", "lang=en-US",
    ]
    pypandoc.convert_text(
        combined, "docx", format="md",
        outputfile=str(out),
        extra_args=extra_args,
    )
    print(f"Wrote DOCX (content only): {out} "
          f"({out.stat().st_size:,} bytes)")


def prepend_title(content_docx: Path, final_docx: Path) -> None:
    """Use docxcompose to prepend title_page.docx onto content_docx."""
    from docxcompose.composer import Composer
    from docx import Document
    title_page = Document(str(HERE / "title_page.docx"))
    composer = Composer(title_page)
    composer.append(Document(str(content_docx)))
    composer.save(str(final_docx))
    print(f"Wrote final DOCX (title + content): {final_docx} "
          f"({final_docx.stat().st_size:,} bytes)")


def render_pdf(docx: Path, pdf: Path) -> None:
    from docx2pdf import convert
    print(f"Converting {docx.name} -> {pdf.name} via Word COM ...")
    convert(str(docx), str(pdf))
    if pdf.exists():
        print(f"Wrote PDF: {pdf} ({pdf.stat().st_size:,} bytes)")
    else:
        raise RuntimeError("docx2pdf produced no PDF; open DOCX in Word manually.")


def main() -> None:
    ensure_styles()
    _, combined = combine_markdown()
    content_docx = HERE / "thesis_content.docx"
    render_docx(combined, content_docx)
    final_docx = HERE / "thesis.docx"
    prepend_title(content_docx, final_docx)
    render_pdf(final_docx, HERE / "thesis.pdf")
    # Also keep an HTML for browser preview
    pypandoc.convert_text(
        combined, "html", format="md",
        outputfile=str(HERE / "thesis.html"),
        extra_args=["--standalone", "--toc", "--toc-depth=2",
                    "--number-sections",
                    "--metadata", "title=Development of a Deep Learning Model for Automated Tree Recognition and Green Space Mapping in Urban Environments"],
    )
    print(f"Wrote HTML: {HERE / 'thesis.html'}")


if __name__ == "__main__":
    main()
