"""Build the thesis as LaTeX via the AITU memoir template + tectonic.

Pipeline:
1. Convert each `0N_*.md` source to a LaTeX fragment via pypandoc with
   `--top-level-division=chapter`.
2. Wire the fragments into the AITU template (`thesis/latex/`) by overwriting:
   - `frontmatter/title.tex`
   - `frontmatter/abstract.tex`
   - `frontmatter/intro.tex` (the standalone Introduction chapter)
   - `frontmatter/concl.tex` (Conclusion)
   - `chapters/chapter01/introduction.tex`
   - `chapters/chapter02/main.tex`
   - `chapters/chapter03/conclusion.tex`
   - `chapters/appendices/references.tex`
3. Write a simplified driver `thesis_main.tex` that uses the AITU memoir
   preamble but only imports our content (skips placeholder dedication /
   declaration / definitions / dda blocks).
4. Run `tectonic` on `thesis_main.tex` to produce `thesis_main.pdf`.

Usage:
    venv\\Scripts\\python.exe thesis\\build_latex.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pypandoc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
LATEX = HERE / "latex"
TECTONIC = Path.home() / ".local" / "bin" / "tectonic.exe"

# Map of @bibkey -> reference number (mirrors build.py CITATIONS).
CITATIONS: dict[str, int] = {
    "DeepForest2019": 1, "AbbasYOLO2025": 2, "Ventura2024": 3,
    "VelasquezCamacho2023": 4, "VelasquezCamacho2025": 5,
    "SofiaDeepForest2024": 6, "Dakov2024": 6,
    "Sun2025": 7, "Lv2023": 8, "LvMCAN2023": 8,
    "Wang2021": 9, "Martins2021": 10, "Martins2021Species": 10,
    "dosSantos2019": 11, "Zheng2022": 12, "Xia2021": 13,
    "He2022": 14, "He2020": 15, "Xu2025": 16, "Awad2021": 17,
    "Branson2019": 18, "Zhang2022": 19, "Zhang2024": 20,
    "Chen2022": 21, "Chen2023": 22, "Schmohl2022": 23,
    "Huerta2021": 24, "SAM2023": 25, "SAM2024": 25, "WBF2021": 26,
    "FasterRCNN2015": 27, "MaskRCNN2017": 28, "RetinaNet2017": 29,
    "YOLOv1": 30, "UNet2015": 31, "UltralyticsYOLO2023": 32,
    "Timilsina2020": 33, "Chen2021": 34, "Dong2019": 35,
}

CITE_GROUP_RE = re.compile(r"\[((?:@[A-Za-z0-9_]+(?:\s*;\s*)?)+)\]")
SINGLE_CITE_RE = re.compile(r"@([A-Za-z0-9_]+)")


def replace_citations(text: str) -> str:
    def sub(m: re.Match) -> str:
        nums = []
        for key in (sk.group(1) for sk in SINGLE_CITE_RE.finditer(m.group(1))):
            nums.append(str(CITATIONS.get(key, "?")))
        return "[" + ", ".join(nums) + "]"
    return CITE_GROUP_RE.sub(sub, text)


def md_to_latex(md_text: str) -> str:
    """Run pypandoc, return LaTeX body.

    `--wrap=none` is required: pandoc's default wrapping can split inline
    `\\textbf{...}` spans across paragraph breaks when the bold text contains
    multi-byte Unicode (e.g. Cyrillic / Kazakh), which makes the output
    LaTeX-invalid (a `\\text@command` argument cannot cross a paragraph end).

    After conversion we post-process the result to collapse blank lines
    inside `\\begin{longtable}...\\end{longtable}` column specs — pandoc still
    emits blank paragraphs between column lines in the table preamble, which
    breaks LaTeX (the longtable column-spec argument cannot contain `\\par`).
    """
    tex = pypandoc.convert_text(
        md_text, "latex", format="md",
        extra_args=["--top-level-division=chapter", "--wrap=none"],
    )

    # Collapse blank lines inside every longtable environment. Pandoc still
    # emits blank paragraphs between column-spec lines in the table preamble,
    # which breaks LaTeX. We do a line-by-line scan instead of a single regex
    # for robustness against nested braces.
    out_lines: list[str] = []
    inside = 0
    for line in tex.splitlines():
        stripped = line.strip()
        if stripped.startswith("\\begin{longtable}"):
            inside += 1
        # drop empty lines while inside a longtable
        if inside > 0 and stripped == "":
            continue
        out_lines.append(line)
        if stripped.startswith("\\end{longtable}"):
            inside = max(0, inside - 1)
    tex = "\n".join(out_lines) + "\n"

    # Strip the explicit `Chapter N.` / `N.M[.K]` numbering that we ship in the
    # Markdown headings. LaTeX/memoir auto-numbers sections within each chapter,
    # so leaving the explicit numbers in produces double-numbering like
    # "Chapter 3. Chapter 3." or "3.3.5 3.3.5". The reference text in the body
    # ("see Section 3.2.3") is intentionally left as plain text.
    tex = re.sub(
        r"\\chapter\{Chapter\s+\d+\.\s*(.+?)\}",
        r"\\chapter{\1}", tex,
    )
    tex = re.sub(
        r"\\(section|subsection|subsubsection|paragraph)\{(?:\d+(?:\.\d+)*\.?)\s+(.+?)\}",
        r"\\\1{\2}", tex,
    )
    return tex


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {path.relative_to(HERE)}  ({len(content):,} chars)")


def _strip_chapter_header(latex: str, expected_title: str | None = None) -> str:
    """Pandoc emits `\\chapter{Title}\\label{...}` at the top. We sometimes
    want only the body when we control the chapter wrapper ourselves."""
    return re.sub(r"^\\chapter\{[^}]*\}\\label\{[^}]*\}\s*", "", latex,
                  count=1, flags=re.MULTILINE)


# ============ Frontmatter ============

def make_title_tex() -> str:
    """AITU-style title page using the same primitives as the supplied
    title.tex placeholder."""
    return r"""%
% File: title.tex (generated by build_latex.py)
%
\begin{titlingpage}
\begin{SingleSpace}
\calccentering{\unitlength}
\begin{adjustwidth*}{\unitlength}{-\unitlength}
\vspace*{10mm}
\begin{center}
{\large LIMITED LIABILITY PARTNERSHIP}\\[1mm]
{\large ASTANA IT UNIVERSITY}\\[10mm]

\rule[0.5ex]{\linewidth}{2pt}\vspace*{-\baselineskip}\vspace*{3.2pt}
\rule[0.5ex]{\linewidth}{1pt}\\[\baselineskip]

{\linespread{1.2}\selectfont
{\HUGE\bfseries DIPLOMA PROJECT}\\[6mm]
}

\vspace{3mm}
{\large\bfseries Topic:}\\[2mm]
{\linespread{1.2}\selectfont
{\Large\bfseries Tree Detection for Astana ---\\
Deep Learning for Urban Green Space Mapping}\\[6mm]
}

\rule[0.5ex]{\linewidth}{1pt}\vspace*{-\baselineskip}\vspace{3.2pt}
\rule[0.5ex]{\linewidth}{2pt}\\[\baselineskip]

\vspace{4mm}
\includegraphics[scale=0.9]{logos/AITU.png}\\[6mm]

{\large\itshape By}\\[3mm]
{\large\bfseries Totin Anuar}\\
{\large\bfseries Aidarkhanov Rasul}\\
{\large\bfseries Sharipov Berik}\\[10mm]

\begin{minipage}{12cm}
\centering
Educational Program: 6B06103 --- Information Technologies\\[2mm]
Scientific Supervisor: Syndar Satbayev\\[2mm]
\itshape School of Information Technologies, Astana IT University
\end{minipage}\\[12mm]

\vfill
{\large\textsc{Astana, 2026}}
\vspace{6mm}
\end{center}
\end{adjustwidth*}
\end{SingleSpace}
\end{titlingpage}
"""


def make_abstract_tex(md: str) -> str:
    """Three-language abstract. Pandoc converts cleanly; we just wrap."""
    body = md_to_latex(replace_citations(md))
    # Pandoc emits three \chapter{Abstract}, \chapter{Аннотация},
    # \chapter{Аңдатпа} — we keep them, memoir handles non-numbered chapters
    # via \chapter* if needed, but a regular \chapter is fine for frontmatter.
    return body


def make_intro_tex(md: str) -> str:
    return md_to_latex(replace_citations(md))


def make_concl_tex(md: str) -> str:
    return md_to_latex(replace_citations(md))


# ============ Main chapters ============

def make_chapter_tex(md_path: Path) -> str:
    return md_to_latex(replace_citations(md_path.read_text(encoding="utf-8")))


# ============ References ============

def make_references_tex(md: str) -> str:
    """References from 07_references.md. We do NOT use bibtex — we just emit
    a 'References' chapter with numbered entries."""
    body = md_to_latex(md)
    # Replace the auto-generated \chapter{References} (Pandoc gives it) with
    # \chapter*{References} so it doesn't count in the table of contents
    # numbering; we still want it listed though.
    body = re.sub(r"\\chapter\{References\}\\label\{references\}",
                  r"\\chapter*{References}\\addcontentsline{toc}{chapter}{References}",
                  body, count=1)
    return body


# ============ Main driver ============

THESIS_MAIN_TEX = r"""% Auto-generated driver for the AITU memoir thesis template.
% Built by thesis/build_latex.py from the per-chapter Markdown sources.

\RequirePackage[l2tabu]{nag}
\let\ordinal\relax
\documentclass[a4paper,12pt,oneside,openbib,oldfontcommands]{memoir}

\usepackage{datetime}
\usepackage{ifpdf}
\ifpdf
\pdfinfo{
   /Author (Totin A., Aidarkhanov R., Sharipov B.)
   /Title (Tree Detection for Astana)
   /Keywords (Astana; Tree Detection; YOLO; DeepForest; SAM; Deep Learning; Remote Sensing)
}
\fi

\usepackage{microtype}
\sloppy

% A4 page layout (memoir-style, matches AITU template)
\settrimmedsize{297mm}{210mm}{*}
\setlength{\trimtop}{0pt}
\setlength{\trimedge}{\stockwidth}
\addtolength{\trimedge}{-\paperwidth}
\settypeblocksize{634pt}{448.13pt}{*}
\setulmargins{4cm}{*}{*}
\setlrmargins{*}{*}{1.5}
\setmarginnotes{17pt}{51pt}{\onelineskip}
\setheadfoot{\onelineskip}{2\onelineskip}
\setheaderspaces{*}{2\onelineskip}{*}
\checkandfixthelayout

\frenchspacing
\OnehalfSpacing

\setsecnumdepth{subsection}
\maxsecnumdepth{subsubsection}

\makepagestyle{myvf}
\makeoddfoot{myvf}{}{\thepage}{}
\makeevenfoot{myvf}{}{\thepage}{}
\makeevenhead{myvf}{\small\textsc{\leftmark}}{}{}
\makeoddhead{myvf}{}{}{\small\textsc{\rightmark}}
\pagestyle{myvf}

\newcommand{\clearemptydoublepage}{\newpage{\thispagestyle{empty}\cleardoublepage}}

% Use fontspec (xelatex) with a Unicode font that covers Latin + Cyrillic
% (including Kazakh-specific glyphs ғ ө ұ ң қ һ і) instead of T2A/T1 which
% miss several Kazakh letters.
\usepackage{fontspec}
\setmainfont{Times New Roman}
% Hyphenation in three languages via babel (no character mapping done by babel
% when fontspec is in use).
\usepackage[main=english, russian]{babel}
\babelprovide[import]{kazakh}

\usepackage{import}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{newlfont}
\usepackage{graphicx}
\usepackage{float}
\usepackage{url}
\usepackage[colorlinks=true,allcolors=black]{hyperref}
\usepackage{memhfixc}
\usepackage{enumerate}
\usepackage{enumitem}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{calc}
\usepackage{etoolbox}

% Tighter list spacing (pandoc-friendly defaults are too loose)
\setlist[itemize]{itemsep=2pt, topsep=4pt}
\setlist[enumerate]{itemsep=2pt, topsep=4pt}

% Pandoc helpers
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
% Pandoc emits `{\def\LTcaptype{none} ... \end{longtable}}` to suppress
% auto-captioning of longtables. The trick requires a `none` counter to exist;
% memoir doesn't define one, so we create a dummy.
\newcounter{none}
% Code-block environment expected by pandoc's LaTeX output (Shaded wraps
% Highlighting; we define a minimal shaded background).
\usepackage{framed}
\usepackage{fancyvrb}
\definecolor{shadecolor}{RGB}{248,248,248}
\newenvironment{Shaded}{\begin{snugshade}}{\end{snugshade}}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\}, fontsize=\small}
% Pandoc highlight tokens (NormalTok, KeywordTok, etc.) — define as no-ops
% so syntax-highlighted code compiles even without a colour scheme.
\providecommand{\NormalTok}[1]{#1}
\providecommand{\KeywordTok}[1]{\textbf{#1}}
\providecommand{\DataTypeTok}[1]{#1}
\providecommand{\DecValTok}[1]{#1}
\providecommand{\StringTok}[1]{#1}
\providecommand{\CommentTok}[1]{\textit{#1}}
\providecommand{\OperatorTok}[1]{#1}
\providecommand{\AttributeTok}[1]{#1}
\providecommand{\FunctionTok}[1]{#1}
\providecommand{\ControlFlowTok}[1]{\textbf{#1}}
\providecommand{\BuiltInTok}[1]{#1}
\providecommand{\OtherTok}[1]{#1}
\providecommand{\VariableTok}[1]{#1}
\providecommand{\SpecialCharTok}[1]{#1}
\providecommand{\PreprocessorTok}[1]{#1}
\providecommand{\WarningTok}[1]{\textbf{#1}}
\providecommand{\AlertTok}[1]{\textbf{#1}}
\providecommand{\ErrorTok}[1]{\textbf{#1}}
\providecommand{\InformationTok}[1]{#1}
\providecommand{\AnnotationTok}[1]{\textit{#1}}
\providecommand{\ConstantTok}[1]{#1}
\providecommand{\FloatTok}[1]{#1}
\providecommand{\BaseNTok}[1]{#1}
\providecommand{\CharTok}[1]{#1}
\providecommand{\SpecialStringTok}[1]{#1}
\providecommand{\VerbatimStringTok}[1]{#1}
\providecommand{\ImportTok}[1]{#1}
\providecommand{\DocumentationTok}[1]{#1}
\providecommand{\CommentVarTok}[1]{#1}
\providecommand{\ExtensionTok}[1]{#1}
\providecommand{\RegionMarkerTok}[1]{#1}

\widowpenalty=1000
\clubpenalty=1000

\begin{document}

\frontmatter
\pagenumbering{roman}

\input{frontmatter/title}

\input{frontmatter/abstract}

\renewcommand{\contentsname}{Table of Contents}
\maxtocdepth{subsection}
\tableofcontents*
\clearpage

\input{frontmatter/intro.tex}

\mainmatter

\import{chapters/chapter01/}{introduction.tex}

\import{chapters/chapter02/}{main.tex}

\import{chapters/chapter03/}{conclusion.tex}

\input{frontmatter/concl.tex}

\backmatter
\input{chapters/appendices/references.tex}

\end{document}
"""


def main() -> None:
    if not TECTONIC.exists():
        sys.exit(f"tectonic not found at {TECTONIC}; install it first.")
    if not LATEX.exists():
        sys.exit(f"AITU template not unpacked at {LATEX}; "
                 f"copy from 'критериии и примеры/AITU_Diploma_2025-2026' first.")

    print("=== Reading Markdown sources ===")
    md = {p.stem: p.read_text(encoding="utf-8")
          for p in HERE.glob("0?_*.md")}
    for k in sorted(md):
        print(f"  {k}: {len(md[k]):,} chars")

    print("\n=== Generating LaTeX fragments ===")
    write(LATEX / "frontmatter" / "title.tex", make_title_tex())
    write(LATEX / "frontmatter" / "abstract.tex",
          make_abstract_tex(md["01_abstract"]))
    write(LATEX / "frontmatter" / "intro.tex",
          make_intro_tex(md["02_intro"]))
    write(LATEX / "chapters" / "chapter01" / "introduction.tex",
          make_chapter_tex(HERE / "03_chapter1.md"))
    write(LATEX / "chapters" / "chapter02" / "main.tex",
          make_chapter_tex(HERE / "04_chapter2.md"))
    write(LATEX / "chapters" / "chapter03" / "conclusion.tex",
          make_chapter_tex(HERE / "05_chapter3.md"))
    write(LATEX / "frontmatter" / "concl.tex",
          make_concl_tex(md["06_conclusion"]))
    write(LATEX / "chapters" / "appendices" / "references.tex",
          make_references_tex(md["07_references"]))
    write(LATEX / "thesis_main.tex", THESIS_MAIN_TEX)

    print("\n=== Compiling with tectonic ===")
    cmd = [str(TECTONIC), "--keep-logs", "--print",
           "--outdir", str(LATEX),
           str(LATEX / "thesis_main.tex")]
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(LATEX), capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    print(result.stdout[-3000:] if result.stdout else "")
    if result.returncode != 0:
        print("STDERR (last 4000 chars):", file=sys.stderr)
        print((result.stderr or "")[-4000:], file=sys.stderr)
        sys.exit(f"tectonic failed with code {result.returncode}")

    pdf = LATEX / "thesis_main.pdf"
    if pdf.exists():
        # Copy to thesis/thesis_latex.pdf for convenient access
        dst = HERE / "thesis_latex.pdf"
        shutil.copy(str(pdf), str(dst))
        print(f"\nWrote PDF: {dst} ({dst.stat().st_size:,} bytes)")
    else:
        sys.exit("tectonic produced no PDF")


if __name__ == "__main__":
    main()
