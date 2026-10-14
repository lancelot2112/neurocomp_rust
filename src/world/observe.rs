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

/// The score of a genome that never clicked in a response window in any of its lifetimes: below
/// any discrimination (which runs from -1024 to 1024). Never acting is death; clicking at all,
/// even at the wrong times, beats it; clicking at the right times is best.
pub const SILENT: i32 = -4096;

impl Responding {
    /// Discrimination, or `SILENT` if there was no click in any trial's response window.
    pub fn score(&self) -> i32 {
        if self.paying_clicked + self.other_clicked == 0 { SILENT } else { self.discrimination() }
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

/// Discrimination within each phase of a reversal (trials `k*period` to `(k+1)*period`), the
/// worst of them: a genome scores well only if it does well both before and after every swap.
/// A phase with no click counts 0.
pub fn worst_phase(events: &[Event], period: usize) -> i32 {
    let period = period.max(1);
    let phases = events.iter().filter_map(|e| e.trial.map(|t| t.0 / period)).max().map_or(0, |m| m + 1);
    (0..phases)
        .map(|ph| {
            let evs: Vec<Event> = events.iter().filter(|e| e.trial.map_or(false, |t| t.0 / period == ph)).cloned().collect();
            responding(&evs).discrimination()
        })
        .min()
        .unwrap_or(0)
}

/// The graded probes: small capacities, each scored on a smooth scale in parts per 1024, so
/// evolution can climb them one at a time (none is fed to the program).
pub const PROBES: [&str; NPROBES] = ["respond", "tell", "discriminate", "reverse", "remember", "win-stay", "copy", "echo", "babble"];
/// How many probes there are.
pub const NPROBES: usize = 9;
/// One lifetime pair's probes, in the order of `PROBES`.
pub type ProbeRow = [i32; NPROBES];

/// Per trial of a conditioning scene: (trial index, cue trial, paying trial, a click in steps
/// `win`, a click in steps `off`).
fn trials(events: &[Event], win: (usize, usize), off: (usize, usize)) -> Vec<(usize, bool, bool, bool, bool)> {
    let mut out: Vec<(usize, bool, bool, bool, bool)> = Vec::new();
    let mut cur: Option<usize> = None;
    for e in events {
        let Some((trial, t, cue, pays)) = e.trial else { continue };
        if cur != Some(trial) {
            out.push((trial, cue, pays, false, false));
            cur = Some(trial);
        }
        let last = out.last_mut().unwrap();
        if e.moved == Some(CLICK) {
            last.3 |= (win.0..=win.1).contains(&t);
            last.4 |= (off.0..=off.1).contains(&t);
        }
    }
    out
}

type Trial = (usize, bool, bool, bool, bool);

/// Click rate (parts per 1024) over the trials picked by `pick`.
fn rate(tr: &[Trial], pick: impl Fn(&Trial) -> bool, click: impl Fn(&Trial) -> bool) -> i32 {
    let n = tr.iter().filter(|x| pick(x)).count();
    if n == 0 { return 0; }
    (tr.iter().filter(|x| pick(x) && click(x)).count() as i64 * 1024 / n as i64) as i32
}

/// The probes from one serial-reversal lifetime (both lights shown, steps 2-4; clicks counted in
/// steps 3-5) and one delayed lifetime (a light at step 2 on cue trials only; clicks in 5-7).
/// Each averages 0 for a network that clicks at random, and for one that never clicks:
/// - respond: clicks while a light is up minus clicks in a matched stretch with none (steps 7-9);
/// - tell: clicks differently to the two lights, consistently: the difference (A - B) in the
///   lifetime's second half, signed by its direction in the first half;
/// - discriminate: more clicks on the paying light than the other (the task);
/// - reverse: discrimination in the second half of each reversal phase (after it has had time to
///   switch), averaged over the phases; clicking one light is right in half the phases and wrong
///   in the rest, so only following the swaps scores;
/// - remember: after a light that is already gone, more clicks than after no light;
/// - win-stay: after a click on a light, the click rate the next time that light comes, when the
///   click was stroked minus when it was not. A network that ignores strokes scores 0 (so does one
///   that always clicks one light: the same rate after either); one whose clicks follow what was
///   stroked scores above 0. The step between telling the lights apart and discriminating.
///
/// And from a sequence lifetime (a random token shown on the eye each step):
/// - copy: says the token it saw a few steps ago (see `copy`).
pub fn probes(reversal: &[Event], period: usize, delayed: &[Event], sequence: &[Event]) -> ProbeRow {
    let tr = trials(reversal, (3, 5), (7, 9));
    let all = |_: &Trial| true;
    let respond = rate(&tr, all, |x| x.3) - rate(&tr, all, |x| x.4);
    let half = tr.len() / 2;
    let diff = |part: &[Trial]| rate(part, |x| x.1, |x| x.3) - rate(part, |x| !x.1, |x| x.3);
    let first = diff(&tr[..half]);
    let tell = if first == 0 { 0 } else { diff(&tr[half..]) * first.signum() };
    let discriminate = rate(&tr, |x| x.2, |x| x.3) - rate(&tr, |x| !x.2, |x| x.3);
    let period = period.max(1);
    let phases = tr.last().map_or(0, |x| x.0 / period + 1);
    let late: Vec<i32> = (0..phases)
        .map(|ph| {
            let part: Vec<Trial> = tr.iter().filter(|x| x.0 / period == ph && x.0 % period >= period / 2).copied().collect();
            rate(&part, |x| x.2, |x| x.3) - rate(&part, |x| !x.2, |x| x.3)
        })
        .collect();
    let reverse = if late.is_empty() { 0 } else { late.iter().sum::<i32>() / late.len() as i32 };
    let td = trials(delayed, (5, 7), (5, 7));
    let remember = rate(&td, |x| x.1, |x| x.3) - rate(&td, |x| !x.1, |x| x.3);
    // win-stay: each click (in the response window) on a light is stroked if the trial paid;
    // the next trial showing the same light is clicked or not
    let (mut won, mut won_n, mut lost, mut lost_n) = (0i64, 0i64, 0i64, 0i64);
    for i in 0..tr.len() {
        if !tr[i].3 {
            continue;
        }
        let Some(next) = tr[i + 1..].iter().find(|x| x.1 == tr[i].1) else { continue };
        if tr[i].2 {
            won_n += 1;
            won += next.3 as i64;
        } else {
            lost_n += 1;
            lost += next.3 as i64;
        }
    }
    // (a side with no clicks counts 0: one that only ever clicks where it is stroked scores its
    // repeat rate; one that never clicks scores 0)
    let r = |c: i64, n: i64| if n == 0 { 0 } else { c * 1024 / n };
    let win_stay = (r(won, won_n) - r(lost, lost_n)) as i32;
    [respond, tell, discriminate, reverse, remember, win_stay, copy(sequence), echo(sequence), babble(sequence)]
}

/// Reading aloud: of the tokens said while tokens are shown one a step, how many match a token
/// shown 1 to 4 steps before, minus how many match one shown 9 to 12 steps before (chance: the
/// shown tokens are independent), per token said. Only tokens shown without a teacher saying them
/// count (with a teacher, echoing the ear would be copying the sound, not reading). Saying
/// nothing, or saying tokens at random, scores 0; reading each untaught token aloud two steps
/// later scores about 900 (half that with a teacher on half the tokens).
pub fn copy(events: &[Event]) -> i32 {
    let ev: Vec<&Event> = events.iter().filter(|e| e.scene == SceneKind::Sequence).collect();
    let said = ev.iter().filter(|e| e.said.is_some()).count() as i64;
    if said == 0 {
        return 0;
    }
    let hits = |lags: std::ops::RangeInclusive<usize>| -> i64 {
        lags.map(|l| (l..ev.len()).filter(|&t| ev[t].said.is_some() && !ev[t - l].taught && ev[t].said == ev[t - l].shown).count() as i64).sum()
    };
    ((hits(1..=4) - hits(9..=12)) * 1024 / said) as i32
}

/// Echoing: of the tokens said while tokens are shown one a step, how many match a token the
/// teacher said 1 to 4 steps before, minus how many match one it said 9 to 12 steps before, per
/// token said. The stepping stone to reading: the network hears a sound and must find the motor
/// code that makes it (its own voice is heard as the same sound), with no glyph involved. Saying
/// nothing or at random scores 0; reading every token aloud scores about as much as echoing (it
/// says the taught ones too); echoing every taught token two steps later scores about 900.
pub fn echo(events: &[Event]) -> i32 {
    let ev: Vec<&Event> = events.iter().filter(|e| e.scene == SceneKind::Sequence).collect();
    let said = ev.iter().filter(|e| e.said.is_some()).count() as i64;
    if said == 0 {
        return 0;
    }
    let hits = |lags: std::ops::RangeInclusive<usize>| -> i64 {
        lags.map(|l| (l..ev.len()).filter(|&t| ev[t].said.is_some() && ev[t - l].taught && ev[t].said == ev[t - l].shown).count() as i64).sum()
    };
    ((hits(1..=4) - hits(9..=12)) * 1024 / said) as i32
}

/// Babbling: how many of the tokens shown while reading the network has said at least once, in
/// parts per 1024. The first stone on the way to speech: a voice that makes many different sounds
/// (its own are heard) gives echoing something to learn from. It needs no sense at all; it only
/// asks for a varied voice. Saying nothing scores 0, one token over and over 1024 / tokens.
pub fn babble(events: &[Event]) -> i32 {
    let ev = events.iter().filter(|e| e.scene == SceneKind::Sequence);
    let (mut shown, mut said) = (0u64, 0u64);
    for e in ev {
        if let Some(t) = e.shown {
            shown |= 1 << t.min(63);
        }
        if let Some(t) = e.said {
            said |= 1 << t.min(63);
        }
    }
    if shown == 0 {
        return 0;
    }
    ((said & shown).count_ones() as i64 * 1024 / shown.count_ones() as i64) as i32
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
    fn the_delayed_light_flashes_delay_steps_before_the_window() {
        for d in 1..=5 {
            let mut w = World::new(5, 4);
            w.delay = d;
            w.conditioning(Conditioning::Delayed, 20);
            let none = BitVector::new(ACT_BITS, Some(0));
            let mut lit = vec![];
            while !w.done() {
                let senses = w.step(&none);
                let e = w.events.last().unwrap();
                if channel(&senses, EYE).count_ones() > 0 {
                    lit.push(e.trial.unwrap().1);
                }
            }
            assert!(!lit.is_empty() && lit.iter().all(|&t| t + d == 5), "delay {d}: {lit:?}");
        }
    }

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
    fn worst_phase_punishes_clicking_one_light_through_a_reversal() {
        // click whenever light A is on: right before the swap, wrong after it
        let mut w = World::new(7, 4);
        w.reversal_period = 20;
        w.conditioning(Conditioning::Reversal, 40);
        let mut prev = BitVector::new(SENSE_BITS, Some(0));
        let light_a = {
            let mut probe = World::new(7, 4);
            probe.conditioning(Conditioning::Reversal, 1);
            probe.lights_for_test()[0].clone()
        };
        while !w.done() {
            let eye = channel(&prev, EYE);
            let sees_a = eye.as_words().iter().zip(light_a.as_words()).any(|(a, b)| a & b != 0);
            let a = w.act(None, if sees_a { Some(CLICK) } else { None });
            prev = w.step(&a);
        }
        assert!(responding(&w.events).discrimination().abs() < 400, "pooled, it looks like half right");
        assert!(worst_phase(&w.events, 20) < -500, "its worst phase shows it never switched");
    }

    #[test]
    fn silence_is_worse_than_any_clicking() {
        let run = |click: bool| {
            let mut w = World::new(5, 4);
            w.conditioning(Conditioning::Instrumental, 20);
            while !w.done() {
                let a = w.act(None, if click { Some(CLICK) } else { None });
                w.step(&a);
            }
            responding(&w.events)
        };
        let (silent, always) = (run(false), run(true));
        assert_eq!(silent.score(), SILENT);
        assert_eq!(always.score(), 0);
        assert!(silent.score() < -1024);
    }

    #[test]
    fn the_probes_grade_each_capacity_on_its_own() {
        // hand-written policies acting on what they saw a step ago (a harness test of the
        // observer, not a network)
        let lights = {
            let mut probe = World::new(7, 4);
            probe.conditioning(Conditioning::Reversal, 1);
            probe.lights_for_test().to_vec()
        };
        let sees = |prev: &BitVector, l: &BitVector| channel(prev, EYE).as_words().iter().zip(l.as_words()).any(|(a, b)| a & b != 0);
        let run = |policy: &dyn Fn(&BitVector, usize) -> bool| -> ProbeRow {
            let mut w = World::new(7, 4);
            w.reversal_period = 50;
            w.conditioning(Conditioning::Reversal, 200);
            let mut prev = BitVector::new(SENSE_BITS, Some(0));
            let mut t = 0;
            while !w.done() {
                let a = w.act(None, if policy(&prev, t) { Some(CLICK) } else { None });
                prev = w.step(&a);
                t += 1;
            }
            let mut d = World::new(8, 4);
            d.conditioning(Conditioning::Delayed, 200);
            let mut prev = BitVector::new(SENSE_BITS, Some(0));
            let mut since = 99;
            while !d.done() {
                since = if channel(&prev, EYE).count_ones() > 0 { 1 } else { since + 1 };
                let a = d.act(None, if (4..=6).contains(&since) { Some(CLICK) } else { None });
                prev = d.step(&a);
            }
            probes(&w.events, 50, &d.events, &[])
        };
        let blind = run(&|_, t| t % 3 == 0);
        // a test-only policy that knows the phase (an oracle for checking the observer): clicks
        // on the paying light; reverse rewards it
        let follows = run(&|p, t| { let phase_a = (t / 10) / 50 % 2 == 0; sees(p, &lights[if phase_a { 0 } else { 1 }]) });
        let any_light = run(&|p, _| channel(p, EYE).count_ones() > 0);
        let light_a = run(&|p, _| sees(p, &lights[0]));
        // a blind clicker: no probe above noise except remember, which the delayed policy adds
        assert!(blind[0].abs() < 80 && blind[2].abs() < 80, "{blind:?}");
        // clicking on any light: responds, cannot tell them apart
        assert!(any_light[0] > 800 && any_light[1].abs() < 80 && any_light[2].abs() < 80, "{any_light:?}");
        // clicking on light A: responds, tells them apart; right half the time, so no discrimination
        // pooled and none after the swaps either
        assert!(light_a[0] > 300 && light_a[1] > 800 && light_a[2].abs() < 150 && light_a[3].abs() < 150, "{light_a:?}");
        // the blind and any-light clickers average 0 on tell and reverse (no noise bias)
        assert!(blind[1].abs() < 80 && blind[3].abs() < 120 && any_light[3].abs() < 120, "{blind:?} {any_light:?}");
        assert!(follows[2] > 800 && follows[3] > 800, "{follows:?}");
        // win-stay: 0 for clickers that ignore strokes (blind, any light, light A only); high for
        // the one whose clicks follow what pays
        assert!(blind[5].abs() < 80 && any_light[5].abs() < 80 && light_a[5].abs() < 80, "{blind:?} {any_light:?} {light_a:?}");
        assert!(follows[5] > 500, "{follows:?}");
        // the delayed lifetime clicks 3-5 steps after a light is gone: remembered
        assert!(blind[4] > 500, "{blind:?}");
    }

    #[test]
    fn gaps_between_trials_stop_a_clock_from_passing_for_responding() {
        // a blind policy that clicks on steps 3-5 of every 10, counted from the start: with
        // trials back to back it lines up with every light; with gaps it cannot
        let run = |gap: usize| -> (i32, usize) {
            let mut w = World::new(7, 4);
            w.gap_max = gap;
            w.reversal_period = 50;
            w.conditioning(Conditioning::Reversal, 200);
            let mut t = 0;
            while !w.done() {
                let a = w.act(None, if (3..=5).contains(&(t % 10)) { Some(CLICK) } else { None });
                w.step(&a);
                t += 1;
            }
            let d = World::new(8, 4);
            (probes(&w.events, 50, &d.events, &[])[0], w.events.len())
        };
        let (locked, steps) = run(0);
        let (jittered, longer) = run(6);
        assert!(locked > 900, "back to back, the clock looks like responding: {locked}");
        assert!(jittered.abs() < 120, "with gaps it does not: {jittered}");
        assert!(steps == 2000 && longer > 2400, "{steps} {longer}");
    }

    #[test]
    fn copy_scores_reading_aloud_and_not_chance() {
        let run = |say: &dyn Fn(&[usize], usize) -> Option<usize>| -> i32 {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(4);
            let toks: Vec<usize> = (0..400).map(|_| rand::Rng::gen_range(&mut rng, 0..8)).collect();
            let mut w = World::new(6, 8);
            w.sequence(&toks);
            let mut t = 0;
            while !w.done() {
                let a = w.act(say(&toks, t), None);
                w.step(&a);
                t += 1;
            }
            copy(&w.events)
        };
        let reads = run(&|toks, t| if t >= 2 { Some(toks[t - 2]) } else { None });
        let random = run(&|_, t| Some((t * 5 + 3) % 8));
        let silent = run(&|_, _| None);
        assert!(reads > 800, "{reads}");
        assert!(random.abs() < 100, "{random}");
        assert_eq!(silent, 0);
    }

    #[test]
    fn with_a_teacher_echoing_does_not_count_as_reading() {
        // a test-only policy that sees the world's events: it repeats a token two steps later
        // either always (reading) or only when the teacher said it (echoing)
        let run = |echo_only: bool| -> i32 {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(4);
            let toks: Vec<usize> = (0..400).map(|_| rand::Rng::gen_range(&mut rng, 0..8)).collect();
            let mut w = World::new(6, 8);
            w.teacher = true;
            w.sequence(&toks);
            let mut t = 0;
            while !w.done() {
                let say = if t >= 2 && (!echo_only || w.events[t - 2].taught) { Some(toks[t - 2]) } else { None };
                let a = w.act(say, None);
                w.step(&a);
                t += 1;
            }
            assert!(w.events.iter().filter(|e| e.taught).count() > 150);
            copy(&w.events)
        };
        let (reads, echoes) = (run(false), run(true));
        assert!(reads > 350, "{reads}");
        assert!(echoes.abs() < 100, "{echoes}");
        // and the echo probe the other way round: echoing scores, and so does reading everything
        // (it says the taught tokens too), while a blind random speaker does not
        let echo_of = |echo_only: bool| -> i32 {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(4);
            let toks: Vec<usize> = (0..400).map(|_| rand::Rng::gen_range(&mut rng, 0..8)).collect();
            let mut w = World::new(6, 8);
            w.teacher = true;
            w.sequence(&toks);
            let mut t = 0;
            while !w.done() {
                let say = if t >= 2 && (!echo_only || w.events[t - 2].taught) { Some(toks[t - 2]) } else { None };
                let a = w.act(say, None);
                w.step(&a);
                t += 1;
            }
            echo(&w.events)
        };
        let (e_echo, e_read) = (echo_of(true), echo_of(false));
        assert!(e_echo > 800, "{e_echo}");
        assert!(e_read > 350, "{e_read}");
        let random = {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(9);
            let toks: Vec<usize> = (0..400).map(|_| rand::Rng::gen_range(&mut rng, 0..8)).collect();
            let mut w = World::new(6, 8);
            w.teacher = true;
            w.sequence(&toks);
            while !w.done() {
                let a = w.act(Some(rand::Rng::gen_range(&mut rng, 0..8)), None);
                w.step(&a);
            }
            echo(&w.events)
        };
        assert!(random.abs() < 150, "{random}");
        // babbling: how many of the shown tokens were ever said
        let babbled = |say: &dyn Fn(usize) -> Option<usize>| -> i32 {
            let mut w = World::new(6, 8);
            w.sequence(&(0..64).map(|i| i % 8).collect::<Vec<_>>());
            let mut t = 0;
            while !w.done() {
                let a = w.act(say(t), None);
                w.step(&a);
                t += 1;
            }
            babble(&w.events)
        };
        assert_eq!(babbled(&|_| None), 0);
        assert_eq!(babbled(&|_| Some(3)), 128);
        assert_eq!(babbled(&|t| Some(t % 8)), 1024);
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
