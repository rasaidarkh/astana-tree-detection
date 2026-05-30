"""Build the standalone three-language ABSTRACT (RU / EN / KZ) as its own PDF.

AITU often requires the abstract as a separate submission document. This script
renders thesis/ABSTRACT.md to thesis/ABSTRACT.pdf via pypandoc (body only) wrapped
in a hand-written xelatex preamble (Times New Roman through fontspec, so Kazakh
glyphs render), compiled with tectonic.

Usage:  venv\\Scripts\\python.exe thesis\\build_abstract.py
"""
from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path

import pypandoc

HERE = Path(__file__).resolve().parent
LATEX = HERE / "latex"           # tectonic compiles here (working env / perms)
SRC = HERE / "ABSTRACT.md"
TEX = HERE / "ABSTRACT.tex"      # source-of-truth copy kept next to the md
PDF = HERE / "ABSTRACT.pdf"

TECTONIC = Path(os.path.expanduser(r"~\.local\bin\tectonic.exe"))
if not TECTONIC.exists():
    TECTONIC = Path(os.path.expanduser(r"~\.local\bin\tectonic"))

PREAMBLE = r"""\documentclass[12pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage[main=english,russian]{babel}
\babelprovide[import]{kazakh}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage[colorlinks=true,allcolors=black]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
% headings: section* used by pandoc for '#'
\usepackage{titlesec}
\titleformat{\section}{\centering\large\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{0pt}{10pt}
\begin{document}
"""


def main() -> None:
    assert SRC.exists(), SRC

    body = pypandoc.convert_file(
        str(SRC), "latex", extra_args=["--wrap=none"]
    )

    tex_content = PREAMBLE + body + "\n\\end{document}\n"
    TEX.write_text(tex_content, encoding="utf-8")
    print(f"wrote {TEX.name} ({len(body):,} chars body)")

    # tectonic must run inside latex/ (its working env + writable cache);
    # compiling elsewhere fails with "could not open format file latex".
    build_tex = LATEX / "_abstract_build.tex"
    build_tex.write_text(tex_content, encoding="utf-8")
    subprocess.run([str(TECTONIC), "--outdir", str(LATEX), str(build_tex)],
                   cwd=str(LATEX), check=True)
    built_pdf = LATEX / "_abstract_build.pdf"
    shutil.copy(str(built_pdf), str(PDF))
    build_tex.unlink(missing_ok=True)
    built_pdf.unlink(missing_ok=True)
    print(f"\nWrote {PDF}")


if __name__ == "__main__":
    main()
