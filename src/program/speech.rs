//! Output: a buffer the cortex writes words to, and its efference copy.
//!
//! - **Speaking is committing.** At a step where the network speaks, it writes the word
//!   with the most evidence (the source mix's choice) with its confidence, or abstains
//!   ("unknown") when the confidence is under a threshold.
//! - **Hearing yourself.** What was spoken comes back as the next input in place of the
//!   page (the answer is not read), so memory and the areas see the network's own word.
//!   The buffer remembers which steps were self-generated, so the caller can mark them as
//!   its own (no learning from them as if they were the world's).
//! - **Self-supervision.** When the world's word is known afterwards (training, or the
//!   page kept for scoring), each spoken word is compared with it: right, wrong, or
//!   withheld. These outcomes are what an abstention gate or the speaking areas learn
//!   from, and what the report scores.

use crate::fixed::Q16;
#[cfg(test)]
use crate::fixed::ONE;

/// One utterance: the word spoken (None: "unknown"), the confidence it was spoken with,
/// and the word the world had there, if known.
#[derive(Clone, Debug, PartialEq)]
pub struct Spoken {
    pub word: Option<usize>,
    pub confidence: Q16,
    pub truth: Option<usize>,
    /// A tag for the report (e.g. held-out vs trained question).
    pub tag: u8,
}

#[derive(Default)]
pub struct OutputBuffer {
    said: Vec<Spoken>,
    /// Abstain under this confidence (0: always speak).
    pub threshold: Q16,
}

impl OutputBuffer {
    pub fn new(threshold: Q16) -> Self {
        Self { said: Vec::new(), threshold }
    }

    /// Speak `word` (the network's best choice) with `confidence`, or abstain under the
    /// threshold. Returns what was spoken.
    pub fn speak(&mut self, word: Option<usize>, confidence: Q16, truth: Option<usize>, tag: u8) -> Option<usize> {
        let word = word.filter(|_| confidence >= self.threshold);
        self.said.push(Spoken { word, confidence, truth, tag });
        word
    }

    pub fn said(&self) -> &[Spoken] {
        &self.said
    }

    pub fn last(&self) -> Option<&Spoken> {
        self.said.last()
    }

    /// For utterances with this tag (None: all): (spoken, right among spoken, all).
    pub fn score(&self, tag: Option<u8>) -> (usize, usize, usize) {
        let it = self.said.iter().filter(|s| tag.map_or(true, |t| s.tag == t));
        let (mut spoken, mut right, mut all) = (0, 0, 0);
        for s in it {
            all += 1;
            if let Some(w) = s.word {
                spoken += 1;
                right += (Some(w) == s.truth) as usize;
            }
        }
        (spoken, right, all)
    }

    /// What the score would have been at another threshold `th` (the confidence and the
    /// best word are kept for withheld answers too only if they were spoken; withheld ones
    /// stay withheld): (spoken, right among spoken, all). Useful at threshold 0, where
    /// every answer was spoken, to draw the accuracy–coverage curve.
    pub fn score_at(&self, th: Q16, tag: Option<u8>) -> (usize, usize, usize) {
        let it = self.said.iter().filter(|s| tag.map_or(true, |t| s.tag == t));
        let (mut spoken, mut right, mut all) = (0, 0, 0);
        for s in it {
            all += 1;
            if let (Some(w), true) = (s.word, s.confidence >= th) {
                spoken += 1;
                right += (Some(w) == s.truth) as usize;
            }
        }
        (spoken, right, all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaks_abstains_and_scores() {
        let mut b = OutputBuffer::new(ONE / 2);
        assert_eq!(b.speak(Some(3), ONE * 3 / 4, Some(3), 0), Some(3));
        assert_eq!(b.speak(Some(4), ONE / 4, Some(5), 0), None); // unsure: "unknown"
        assert_eq!(b.speak(Some(6), ONE, Some(7), 1), Some(6));
        assert_eq!(b.score(None), (2, 1, 3));
        assert_eq!(b.score(Some(0)), (1, 1, 2));
        assert_eq!(b.score_at(ONE * 9 / 10, None), (1, 0, 3));
    }
}
