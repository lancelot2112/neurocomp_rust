//! The observer: scores from the world's events, outside both the world and the program.
//! This is what evolution reads as fitness; nothing here is sent to the program.
//!
//! All scores are integer counts; a caller forms rates as it likes.

use super::{Conditioning, Event, SceneKind, CLICK};

/// Conditioning: per kind of trial (paying or not), trials and trials with a click in the
/// response window (the light's steps and the step after; for Pavlov, any click).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Responding {
    pub paying_trials: u32,
    pub paying_clicked: u32,
    pub other_trials: u32,
    pub other_clicked: u32,
    pub strokes: u32,
}

impl Responding {
    /// Discrimination in parts per 1024: click rate on paying trials minus on the others.
    pub fn discrimination(&self) -> i32 {
        let r = |c: u32, n: u32| if n == 0 { 0 } else { (c as i64 * 1024 / n as i64) as i32 };
        r(self.paying_clicked, self.paying_trials) - r(self.other_clicked, self.other_trials)
    }
}

fn window(kind: Conditioning, t: usize) -> bool {
    match kind {
        Conditioning::Pavlov => true,
        Conditioning::Instrumental | Conditioning::Reversal => (3..=5).contains(&t),
        Conditioning::Delayed => (5..=7).contains(&t),
    }
}

/// Conditioning responding over `events` (pass a slice to score a stretch, e.g. one reversal
/// phase).
pub fn responding(events: &[Event]) -> Responding {
    let mut r = Responding::default();
    let mut cur: Option<(usize, bool, bool)> = None;
    let mut close = |r: &mut Responding, c: Option<(usize, bool, bool)>| {
        if let Some((_, pays, clicked)) = c {
            if pays {
                r.paying_trials += 1;
                r.paying_clicked += clicked as u32;
            } else {
                r.other_trials += 1;
                r.other_clicked += clicked as u32;
            }
        }
    };
    for e in events {
        r.strokes += e.stroked as u32;
        let (SceneKind::Conditioning(kind), Some((trial, t, _, pays))) = (e.scene, e.trial) else { continue };
        if cur.map_or(true, |c| c.0 != trial) {
            close(&mut r, cur.take());
            cur = Some((trial, pays, false));
        }
        if e.moved == Some(CLICK) && window(kind, t) {
            if let Some(c) = cur.as_mut() {
                c.2 = true;
            }
        }
    }
    close(&mut r, cur);
    r
}

/// Sequence: steps where a next token was due, and those where the program said it on the
/// step before it was shown.
pub fn anticipation(events: &[Event]) -> (u32, u32) {
    let (mut due, mut hit) = (0, 0);
    for pair in events.windows(2) {
        if pair[1].scene == SceneKind::Sequence && pair[0].scene == SceneKind::Sequence {
            if let Some(next) = pair[1].shown {
                due += 1;
                hit += (pair[0].said == Some(next)) as u32;
            }
        }
    }
    (due, hit)
}

/// Echo play: sounds made aloud, and those answered by the caregiver.
pub fn echoes(events: &[Event]) -> (u32, u32) {
    let said = events.iter().filter(|e| e.scene == SceneKind::Echo && e.said.is_some()).count() as u32;
    let echoed = events.iter().filter(|e| e.echoed).count() as u32;
    (said, echoed)
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    #[test]
    fn a_program_that_clicks_on_the_light_discriminates() {
        let mut w = World::new(5, 4);
        w.conditioning(Conditioning::Instrumental, 60);
        let mut prev = BitVector::new(SENSE_BITS, Some(0));
        while !w.done() {
            let sees_light = channel(&prev, EYE).count_ones() > 0;
            let a = w.act(None, if sees_light { Some(CLICK) } else { None });
            prev = w.step(&a);
        }
        let r = responding(&w.events);
        assert_eq!(r.paying_clicked, r.paying_trials);
        assert_eq!(r.other_clicked, 0);
        assert_eq!(r.discrimination(), 1024);
        assert_eq!(r.strokes, r.paying_trials);
    }

    #[test]
    fn saying_the_next_token_is_counted() {
        let mut w = World::new(6, 4);
        let seq = [0, 1, 2, 3, 0, 1];
        w.sequence(&seq);
        let mut i = 0;
        while !w.done() {
            let a = w.act(seq.get(i + 1).copied(), None);
            w.step(&a);
            i += 1;
        }
        assert_eq!(anticipation(&w.events), (5, 5));
    }
}
