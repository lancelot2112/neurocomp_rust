#!/usr/bin/env python3
"""A deliberately dumb baseline: keep the raw text and answer by searching it.

Input: a story dump from the episodic harness (DUMP_STORIES=dir), one story per line:
    train|test <TAB> held_out <TAB> answer_at <TAB> words...

Every story read so far is kept as text (training stories, and each test story once it
has been answered). At a test question (the story up to `answer_at`), two searches:

- grep: find the most recent occurrence in the text of the longest suffix of the
  question (up to 8 words) that occurs, and answer with the word that followed it.
- grep + context: among all occurrences of that suffix, take the one whose story shares
  the most words with the current story so far (each shared word weighted by its rarity,
  log(stories / stories containing it)), ties to the most recent; answer with the word
  that followed.

Reports accuracy on trained (held_out = 0) and held-out (1) questions, and the size of
the text: raw bytes, gzip-compressed bytes, and one byte per word id.

    python3 scripts/text_search_baseline.py /path/to/season_0_0.txt
"""
import gzip
import math
import sys
from collections import defaultdict

MAX_N = 8


def main(path):
    stories = []
    for line in open(path):
        kind, held, at, text = line.rstrip("\n").split("\t")
        stories.append((kind, held == "1", int(at), text.split(" ")))

    tokens = []          # all text read so far
    story_of = []        # story index of each token
    story_words = []     # set of words in each stored story
    df = defaultdict(int)
    grams = defaultdict(list)  # n-gram -> positions of its last word

    def add_story(words):
        sid = len(story_words)
        story_words.append(set(words))
        for w in set(words):
            df[w] += 1
        for w in words:
            tokens.append(w)
            story_of.append(sid)
            end = len(tokens) - 1
            for n in range(1, MAX_N + 1):
                if end - n + 1 < 0:
                    break
                grams[tuple(tokens[end - n + 1 : end + 1])].append(end)

    train_text = []
    tally = {"grep": [[0, 0], [0, 0]], "grep + context": [[0, 0], [0, 0]]}
    for kind, held, at, words in stories:
        if kind == "train":
            add_story(words)
            train_text.extend(words)
            continue
        prefix, answer = words[:at], words[at]
        here = set(prefix)
        hits = None
        for n in range(min(MAX_N, len(prefix)), 0, -1):
            occ = [p for p in grams.get(tuple(prefix[-n:]), []) if p + 1 < len(tokens) and story_of[p + 1] == story_of[p]]
            if occ:
                hits = occ
                break
        g = c = None
        if hits:
            g = tokens[hits[-1] + 1]
            stories_n = len(story_words)
            def score(p):
                shared = story_words[story_of[p]] & here
                return sum(math.log(stories_n / df[w]) for w in shared)
            best = max(hits, key=lambda p: (score(p), p))
            c = tokens[best + 1]
        for name, guess in (("grep", g), ("grep + context", c)):
            t = tally[name][held]
            t[0] += guess == answer
            t[1] += 1
        add_story(words)

    raw = " ".join(train_text).encode()
    print(f"training text: {len(train_text)} words, {len(raw)} bytes raw, {len(gzip.compress(raw, 9))} bytes gzip -9, {len(train_text)} bytes as one-byte word ids")
    print(f"index (n-grams up to {MAX_N}, positions as u32): {sum(len(v) for v in grams.values()) * 4} bytes")
    for name, t in tally.items():
        seen, held = t[0], t[1]
        print(f"{name:16s} trained {100 * seen[0] / max(1, seen[1]):5.1f}% of {seen[1]}   held out {100 * held[0] / max(1, held[1]):5.1f}% of {held[1]}")


if __name__ == "__main__":
    main(sys.argv[1])
