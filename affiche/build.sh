#!/bin/sh
# Compile l'affiche (et l'annexe si presente) avec XeLaTeX + BibTeX.
for f in affiche annexe; do
  [ -f $f.tex ] || continue
  xelatex -interaction=nonstopmode $f.tex > /dev/null
  bibtex $f > /dev/null 2>&1
  xelatex -interaction=nonstopmode $f.tex > /dev/null
  xelatex -interaction=nonstopmode $f.tex > $f.compile.log
done
# Les deux pages dans un seul PDF
xelatex -interaction=nonstopmode affiche_complete.tex > /dev/null
