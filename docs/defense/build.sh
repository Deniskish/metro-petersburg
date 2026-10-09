#!/usr/bin/env bash
# Сборка PDF для защиты: cheatsheet.md + slides_5min.md → defense.pdf.
# Запуск: bash docs/defense/build.sh [--pages КАТАЛОГ]   (--pages — ещё и PNG страниц для просмотра)
# Движок: xelatex, если есть, иначе tectonic (тоже XeTeX); шрифты PT Serif / PT Sans / PT Mono, кириллица.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
if command -v xelatex >/dev/null; then ENGINE=xelatex; else ENGINE=tectonic; fi
echo "== PDF ($ENGINE)"
pandoc cheatsheet.md slides_5min.md --metadata-file=metadata.yaml --toc --resource-path=.:"$ROOT" \
  --pdf-engine="$ENGINE" -H preamble.tex -o defense.pdf 2> >(grep -v "accessing absolute path" >&2)
echo "готово: docs/defense/defense.pdf"
if [ "${1:-}" = "--pages" ] && [ -n "${2:-}" ]; then
  mkdir -p "$2"
  pdftoppm -r 70 -png defense.pdf "$2/p"
  echo "страниц: $(ls "$2"/p-*.png | wc -l) → $2"
fi
