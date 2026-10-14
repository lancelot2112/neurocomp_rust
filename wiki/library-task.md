# The library task (design)

A general-purpose test: read a small corpus of books and answer questions about them. The
program must first learn to use the screen: move the cursor, click a book open, page through
it. Then it must find the right book and the right page for a question, and say the answer.
Nothing is given as symbols: every step is bit vectors through the world's channels.

## The world's side (an extension of `src/world`)

- **Screen** (eye channel): a row of places, each empty or holding a book's cover code. The
  program's cursor is seen at its place. An open book shows one word at a time.
- **Hand**: `LEFT`, `RIGHT`, `CLICK`.
  - On the shelf, left and right move the cursor, and a click opens the book under it.
  - In a book, right shows the next word, left the one before, and a click on the last word
    closes the book.
  - A move that does nothing is sensed as nothing (lesson 29).
- **Ear**: questions arrive as spoken words ("where did tom go"). One's own voice is heard too.
- **Voice**: saying a word aloud is the answer.
- **Affect**: a stroke after a right answer said aloud, from a teacher who is present only in
  question scenes. It is a sense, not a reward signal; whether the program values it is
  evolution's business.
- **Books**: generated from a small grammar (the earlier branch's story generator: names,
  places, animals, things). Each book has a cover code and 3–8 sentences. The physics is varied
  per lifetime: covers, word codes and motor primitives all change. The corpus is varied too, so
  answers cannot be wired in.

## The observer's side (fitness)

Scored from events, in stages that each count on their own, so evolution has a gradient long
before the final answer is right:
1. **Opened** a book (any click on a cover).
2. **Read**: distinct words of an open book seen in order.
3. **Found**: on a question scene, the book containing the answer was opened.
4. **Reached**: its sentence holding the answer was shown.
5. **Answered**: the answer word was said aloud before the scene ended.

A lifetime's fitness is a weighted sum, with weights rising by stage. Held-out lifetimes use new
books and new physics.

## Curriculum within a lifetime

- Free play: only the shelf, no teacher.
- Reading: a teacher opens a book with their own cursor (seen, tagged as theirs) and pages
  through it.
- Questions: the teacher asks, then waits.

The same scenes run in every lifetime; only the order of books and questions varies.

## What the network will need (and should evolve, not be given)

- Act on what is seen (the reflexes of experiment 01 already do this).
- Sequence prediction and residue, to learn the books' words (experiment 02).
- A store that holds which book holds which words, and cueing it from a question. On the
  earlier branch that needed a one-shot store (lesson 22).
- Holding the question while navigating.

Each is a sub-task with its own scene, so it can be evolved and checked alone before the whole.
