#!/usr/bin/env bash
# Fetch the public-domain corpora used by the examples into ./data (git-ignored).
#   data/alice.txt         Lewis Carroll, Alice's Adventures in Wonderland (Project Gutenberg via NLTK)
#   data/bryant.txt        Sara Cone Bryant, How to Tell Stories to Children (Project Gutenberg via NLTK)
#   data/brown_tagged.txt  Brown corpus, one sentence per line as word/TAG tokens (NLTK)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
base=https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora
curl -sSfL -o "$tmp/gutenberg.zip" "$base/gutenberg.zip"
curl -sSfL -o "$tmp/brown.zip" "$base/brown.zip"
python3 - "$tmp" <<'PY'
import sys, zipfile, re
tmp = sys.argv[1]
g = zipfile.ZipFile(f"{tmp}/gutenberg.zip")
open("data/alice.txt", "wb").write(g.read("gutenberg/carroll-alice.txt"))
open("data/bryant.txt", "wb").write(g.read("gutenberg/bryant-stories.txt"))
b = zipfile.ZipFile(f"{tmp}/brown.zip")
with open("data/brown_tagged.txt", "w") as out:
    for name in sorted(n for n in b.namelist() if re.fullmatch(r"brown/c[a-r]\d\d", n)):
        for line in b.read(name).decode("latin-1").splitlines():
            line = line.strip()
            if line:
                out.write(line + "\n")
PY
wc -c data/*
