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
pub const PROBES: [&str; NPROBES] = ["respond", "tell", "discriminate", "reverse", "remember", "win-stay", "copy", "echo", "babble", "surprise", "maze", "screen", "repeat", "select", "feel", "echo~", "surprise~", "feel~", "remember~", "select~", "mixclick", "b-click-cond", "b-click-seq", "b-voice-seq", "b-voice-cond", "b-stroke", "b-moves", "b-tokens", "b-odd", "novelty", "screen~", "surprise+", "flag", "feel-gap"];
/// How many probes there are.
pub const NPROBES: usize = 34;
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

/// Trials by the world's own window mark (a delayed trial's window moves with its delay): each
/// trial's index, cue, whether it pays, and whether it was clicked in its window.
fn trials_in_window(events: &[Event]) -> Vec<Trial> {
    let mut out: Vec<Trial> = Vec::new();
    let mut cur: Option<usize> = None;
    for e in events {
        let Some((trial, _, cue, pays)) = e.trial else { continue };
        if cur != Some(trial) {
            out.push((trial, cue, pays, false, false));
            cur = Some(trial);
        }
        if e.moved == Some(CLICK) && e.window {
            out.last_mut().unwrap().3 = true;
        }
    }
    out
}

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
pub fn probes(reversal: &[Event], period: usize, delayed: &[Event], sequence: &[Event], pattern: &[Event], maze_events: &[Event], screen_events: &[Event]) -> ProbeRow {
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
    // (`set_stray_cost`) clicks away from any light's window, per trial, taken off
    let stray = if stray_cost() { stray_rate(reversal) } else { 0 };
    let (respond, discriminate, reverse) = ((respond - stray).max(-1024), (discriminate - stray).max(-1024), (reverse - stray).max(-1024));
    // (`set_miss_cost`) the lights that called for a click and got none
    let (respond, discriminate, reverse) = if miss_cost() && !tr.is_empty() {
        let miss_all = 1024 - rate(&tr, all, |x| x.3);
        let miss_pay = 1024 - rate(&tr, |x| x.2, |x| x.3);
        ((respond - miss_all).max(-1024), (discriminate - miss_pay).max(-1024), (reverse - miss_pay).max(-1024))
    } else {
        (respond, discriminate, reverse)
    };
    // (the window marked by the world: steps 5-7 for a delay up to 5, as always)
    let td = trials_in_window(delayed);
    let remember = rate(&td, |x| x.1, |x| x.3) - rate(&td, |x| !x.1, |x| x.3);
    let remember = if miss_cost() && !td.is_empty() { (remember - (1024 - rate(&td, |x| x.1, |x| x.3))).max(-1024) } else { remember };
    // win-stay: each click (in the response window) on a light is stroked if the trial paid;
    // the next trial showing the same light is clicked or not
    // (a lifetime with no stroked click to compare against an unstroked one scores 0: with no
    // swaps a light either always pays or never does, so a clicker that ignores strokes and keeps
    // to one light had only one side, and scored +1024 or -1024 by which light it kept to, a
    // spread of about 300 in a 12-world mean. A learner has both sides: it clicks the light that
    // did not pay once, and stops)
    let win_stay = win_stay_of(&tr, fair_win_stay());
    [respond, tell, discriminate, reverse, remember, win_stay, copy(sequence), echo(sequence), babble(sequence), surprise(pattern), maze(maze_events), screen(screen_events), repeat(maze_events), 0, 0, echo_graded(sequence), surprise_graded(pattern), 0, remember_graded(delayed), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, screen_graded(screen_events), surprise_shaped(pattern), 0, feel_in_gaps(reversal)]
}

/// The farthest lag copy and echo compare against: both count only steps with that many steps
/// behind them, so near and far lags have the same number of chances to match (counting from each
/// lag's own start gave the near lags more, which a blind speaker turned into a score: about 80
/// with 2 tokens).
const FAR: usize = 12;

/// Reading aloud: of the tokens said while tokens are shown one a step, how many match a token
/// shown 1 to 4 steps before, minus how many match one shown 9 to 12 steps before (chance: the
/// shown tokens are independent), per token said. Only tokens shown without a teacher saying them
/// count (with a teacher, echoing the ear would be copying the sound, not reading). Saying
/// nothing, or saying tokens at random, scores 0; reading each untaught token aloud two steps
/// later scores about 900 (half that with a teacher on half the tokens).
/// The tokens an event said: its sentence (with sentences), else its one token.
fn says(e: &Event, tok: Option<usize>) -> bool {
    match tok {
        None => false,
        Some(t) => if e.sentence.is_empty() { e.said == Some(t) } else { e.sentence.contains(&t) },
    }
}

/// How many tokens an event said.
fn n_said(e: &Event) -> i64 {
    if e.sentence.is_empty() { e.said.is_some() as i64 } else { e.sentence.len() as i64 }
}

/// (`set_fair_said`) echo and copy per token they could answer, not only per token said: the
/// count divided by is the tokens said or the tokens shown that count (taught for echo, untaught
/// for copy), whichever is more, so saying one token in a lifetime cannot score by luck. Off by
/// default.
static FAIR_SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_fair_said(on: bool) {
    FAIR_SAID.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn fair_said() -> bool {
    FAIR_SAID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Stray clicks cost (`set_stray_cost`): in the conditioning world (switching's and shapes'
/// ladders), the share of clicks made away from any light's window, per trial, is taken off
/// respond, discriminate and reverse: a light is a prompt, and clicking without one is a wrong
/// answer. Off by default.
static STRAY_COST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_stray_cost(on: bool) {
    STRAY_COST.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn stray_cost() -> bool {
    STRAY_COST.load(std::sync::atomic::Ordering::Relaxed)
}

/// Clicks in a conditioning lifetime made away from any light's window (steps 3 to 5 of a
/// trial), per trial, in parts per 1024 (`set_stray_cost`).
pub fn stray_rate(events: &[Event]) -> i32 {
    let tr = trials(events, (3, 5), (7, 9));
    if tr.is_empty() {
        return 0;
    }
    // (a click within three steps of a stroke answers it: `World::gap_strokes`)
    let n = events.iter().enumerate().filter(|(i, e)| matches!(e.scene, SceneKind::Conditioning(_)) && e.moved == Some(CLICK) && !e.trial.is_some_and(|t| (3..=5).contains(&t.1)) && !(i.saturating_sub(3)..*i).any(|u| events[u].stroked)).count();
    (n * 1024 / tr.len()) as i32
}

/// Feeling in the lights world (stacked rungs, `World::gap_strokes`): clicks in the three steps
/// after a stroke that came in a gap between trials, against clicks at the other gap steps, in
/// parts per 1024, as `feel` is in its own world. The lights' trials go on around it, so a network
/// keeps telling lights apart while it learns to notice the stroke.
pub fn feel_in_gaps(events: &[Event]) -> i32 {
    let ev: Vec<&Event> = events.iter().filter(|e| matches!(e.scene, SceneKind::Conditioning(_))).collect();
    let gap = |e: &Event| e.trial.is_none();
    let (mut after, mut after_c, mut other, mut other_c) = (0i64, 0i64, 0i64, 0i64);
    for t in 0..ev.len() {
        let near = (t.saturating_sub(3)..t).any(|u| ev[u].stroked && gap(ev[u]));
        let click = (ev[t].moved == Some(CLICK)) as i64;
        if near {
            after += 1;
            after_c += click;
        } else if gap(ev[t]) {
            other += 1;
            other_c += click;
        }
    }
    if after == 0 || other == 0 {
        return 0;
    }
    (after_c * 1024 / after - other_c * 1024 / other) as i32
}

/// Misses cost (`set_miss_cost`): a prompt that called for an act and got none counts as a wrong
/// answer, so doing nothing scores -1024, not 0: respond, discriminate and reverse less the share of
/// paying lights not clicked, remember and flag less the cued trials or questions not answered,
/// maze less the trials with no choice, the screen less the trials with no hit, selection by
/// contrast -1024 for never clicking nor speaking. Off by default.
static MISS_COST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

thread_local! {
    /// (tests) the switch for this thread only
    static MISS_COST_HERE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

pub fn set_miss_cost(on: bool) {
    MISS_COST.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn miss_cost() -> bool {
    MISS_COST_HERE.with(|c| c.get()).unwrap_or_else(|| MISS_COST.load(std::sync::atomic::Ordering::Relaxed))
}

/// (tests) `set_miss_cost` for this thread only.
pub fn set_miss_cost_here(on: Option<bool>) {
    MISS_COST_HERE.with(|c| c.set(on));
}

/// Selection by contrast (`set_select_contrast`): 1, select and select~ are `select_contrast` (the
/// rates' contrast); 2, `select_performance` (the tasks' performance contrasted). 0 off, the
/// default.
static SELECT_CONTRAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn set_select_contrast(mode: u8) {
    SELECT_CONTRAST.store(mode, std::sync::atomic::Ordering::Relaxed);
}

pub fn select_contrast_on() -> bool {
    SELECT_CONTRAST.load(std::sync::atomic::Ordering::Relaxed) > 0
}

/// The selection score in use under `set_select_contrast` (1 the rates', 2 the performance's).
fn contrast_score(events: &[Event]) -> i32 {
    if SELECT_CONTRAST.load(std::sync::atomic::Ordering::Relaxed) == 2 { select_performance(events) } else { select_contrast(events) }
}

/// Selection as the tasks' performance contrasted (`set_select_contrast(2)`): responding (paying
/// lights clicked in their window, less clicks in the control window after) in respond blocks less
/// the same in babble blocks, and the voice rate in babble blocks less in respond blocks, averaged;
/// with still blocks, a third for not acting there (what is done elsewhere less what is done
/// there). Unlike the rates' contrast, a few clicks score little: run 37's short passer scored 814 on
/// the rates' contrast with respond at 30. A network blind to the task scores 0, a selector 1024.
pub fn select_performance(events: &[Event]) -> i32 {
    let of = |task: usize| -> Vec<Event> { events.iter().filter(|e| e.task == Some(task)).cloned().collect() };
    let (resp, bab) = (of(0), of(1));
    if resp.is_empty() || bab.is_empty() {
        return 0;
    }
    let respond = |ev: &[Event]| -> i64 {
        let tr = trials(ev, (3, 5), (7, 9));
        if tr.is_empty() { 0 } else { (rate(&tr, |_| true, |x| x.3) - rate(&tr, |_| true, |x| x.4)) as i64 }
    };
    let voice = |ev: &[Event]| -> i64 { ev.iter().filter(|e| e.said.is_some() || !e.sentence.is_empty()).count() as i64 * 1024 / ev.len() as i64 };
    let acts = |ev: &[Event]| -> i64 { ev.iter().filter(|e| e.moved == Some(CLICK) || e.said.is_some() || !e.sentence.is_empty()).count() as i64 * 1024 / ev.len() as i64 };
    if miss_cost() && acts(&resp) + acts(&bab) == 0 {
        return -1024;
    }
    let two = ((respond(&resp) - respond(&bab)) + (voice(&bab) - voice(&resp))) / 2;
    let still = of(2);
    if still.is_empty() {
        return two as i32;
    }
    let here = acts(&still);
    let elsewhere = (respond(&resp).max(0) + voice(&bab)) / 2;
    ((2 * two + (elsewhere - here)) / 3) as i32
}

/// Selection as the contrast between tasks (`set_select_contrast`): the click rate in respond
/// blocks against babble blocks and the voice rate in babble blocks against respond blocks, each
/// as (a - b) / (a + b), averaged; with still blocks, also acting elsewhere against acting there
/// (a third). A network that does the same whatever the task scores 0, one that clicks only in
/// respond blocks and speaks only in babble blocks 1024. Run 36's selection passed held by
/// clicking at every light and speaking now and then in every block (select 722).
pub fn select_contrast(events: &[Event]) -> i32 {
    let rate = |task: usize, act: &dyn Fn(&Event) -> bool| -> Option<i64> {
        let v: Vec<&Event> = events.iter().filter(|e| e.task == Some(task)).collect();
        (!v.is_empty()).then(|| v.iter().filter(|e| act(e)).count() as i64 * 1024 / v.len() as i64)
    };
    let click = |e: &Event| e.moved == Some(CLICK);
    let voice = |e: &Event| e.said.is_some() || !e.sentence.is_empty();
    let rel = |a: i64, b: i64| if a + b == 0 { 0 } else { (a - b) * 1024 / (a + b) };
    let (Some(cr), Some(cb), Some(vr), Some(vb)) = (rate(0, &click), rate(1, &click), rate(0, &voice), rate(1, &voice)) else { return 0 };
    // (`set_miss_cost`) never clicking nor speaking
    if miss_cost() && cr + cb + vr + vb == 0 {
        return -1024;
    }
    let two = (rel(cr, cb) + rel(vb, vr)) / 2;
    match (rate(2, &click), rate(2, &voice)) {
        (Some(cs), Some(vs)) => ((2 * two + rel((cr + cb + vr + vb) / 2, cs + vs)) / 3) as i32,
        _ => two as i32,
    }
}

/// Win-stay from a lifetime's trials: after a click on a light, the click rate on that light's next
/// trial when the click was stroked less when it was not; with `fair` (`set_fair_win_stay`), less
/// the same on the other light's next trial, halved. 0 when a side has no data.
fn win_stay_of(tr: &[Trial], fair: bool) -> i32 {
    // (`set_fair_win_stay`: the same, less the same rates on the next trial of the other light,
    // halved: a click that follows any recent stroke, whichever light, cancels out, and only a
    // click that follows what was stroked on that light scores)
    let stay = |same: bool| -> Option<i64> {
        let (mut won, mut won_n, mut lost, mut lost_n) = (0i64, 0i64, 0i64, 0i64);
        for i in 0..tr.len() {
            if !tr[i].3 {
                continue;
            }
            let Some(next) = tr[i + 1..].iter().find(|x| (x.1 == tr[i].1) == same) else { continue };
            if tr[i].2 {
                won_n += 1;
                won += next.3 as i64;
            } else {
                lost_n += 1;
                lost += next.3 as i64;
            }
        }
        if won_n == 0 || lost_n == 0 { None } else { Some(won * 1024 / won_n - lost * 1024 / lost_n) }
    };
    if fair {
        match (stay(true), stay(false)) { (Some(s), Some(o)) => ((s - o) / 2) as i32, _ => 0 }
    } else {
        stay(true).unwrap_or(0) as i32
    }
}

/// (`set_fair_win_stay`) win-stay against the other light: the stroked-minus-unstroked rate of
/// clicking the next trial of the same light, less that of the next trial of the other light,
/// halved. A clicker driven by any recent stroke, whichever light, scores about 0 or less (the
/// audit: 205 on the old probe). Off by default.
static FAIR_WIN_STAY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_fair_win_stay(on: bool) {
    FAIR_WIN_STAY.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn fair_win_stay() -> bool {
    FAIR_WIN_STAY.load(std::sync::atomic::Ordering::Relaxed)
}

/// What echo or copy divides by: the tokens said, or (`set_fair_said`) the tokens said or the ones
/// shown that count (`taught` for echo), whichever is more.
fn per(ev: &[&Event], taught: bool) -> i64 {
    let said: i64 = ev.iter().map(|e| n_said(e)).sum();
    if fair_said() { said.max(ev.iter().filter(|e| e.shown.is_some() && e.taught == taught).count() as i64) } else { said }
}

pub fn copy(events: &[Event]) -> i32 {
    let ev: Vec<&Event> = events.iter().filter(|e| e.scene == SceneKind::Sequence).collect();
    let said: i64 = per(&ev, false);
    if said == 0 {
        return 0;
    }
    let hits = |lags: std::ops::RangeInclusive<usize>| -> i64 {
        lags.map(|l| (FAR..ev.len()).filter(|&t| !ev[t - l].taught && says(ev[t], ev[t - l].shown)).count() as i64).sum()
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
    let said: i64 = per(&ev, true);
    if said == 0 {
        return 0;
    }
    let hits = |lags: std::ops::RangeInclusive<usize>| -> i64 {
        lags.map(|l| (FAR..ev.len()).filter(|&t| ev[t - l].taught && says(ev[t], ev[t - l].shown)).count() as i64).sum()
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
        for &t in &e.sentence {
            said |= 1 << t.min(63);
        }
    }
    if shown == 0 {
        return 0;
    }
    ((said & shown).count_ones() as i64 * 1024 / shown.count_ones() as i64) as i32
}

/// Surprise: on a stream of tokens with a pattern (a cycle) that some tokens break, how much
/// more often the program clicks in the 3 steps after a token that broke it than after one that
/// kept it, in parts per 1024. A click needs no code learned for any token, so this asks only
/// whether the program predicted the stream: a blind clicker, or one that clicks on every token,
/// scores 0. (Tokens kept within 3 steps after a break are left out of the comparison.)
pub fn surprise(events: &[Event]) -> i32 {
    let ev: Vec<&Event> = events.iter().filter(|e| e.scene == SceneKind::Sequence).collect();
    let clicked = |t: usize| (t + 1..=(t + 3).min(ev.len().saturating_sub(1))).any(|u| ev[u].moved == Some(CLICK));
    let (mut odd, mut odd_c, mut kept, mut kept_c) = (0i64, 0i64, 0i64, 0i64);
    for t in 0..ev.len().saturating_sub(3) {
        if ev[t].shown.is_none() {
            continue;
        }
        if ev[t].odd {
            odd += 1;
            odd_c += clicked(t) as i64;
        } else if !(t.saturating_sub(3)..t).any(|u| ev[u].odd) {
            kept += 1;
            kept_c += clicked(t) as i64;
        }
    }
    if odd == 0 || kept == 0 {
        return 0;
    }
    (odd_c * 1024 / odd - kept_c * 1024 / kept) as i32
}

/// Feeling reward: in a lifetime of strokes that come at random (`World::feel`), how much more often
/// the program clicks in the 3 steps after a stroke than at other steps, in parts per 1024. Nothing
/// is earned: it asks only that the stroke reach the hand, the circuit that learning from reward
/// builds on. A clicker that ignores the stroke, clicking always, never or at random, scores 0.
/// (Set by `evolve::probe_scores_in` from a feeling lifetime: `probes` leaves it 0.)
pub fn feel(events: &[Event]) -> i32 {
    let ev: Vec<&Event> = events.iter().filter(|e| e.scene == SceneKind::Feel).collect();
    let (mut after, mut after_c, mut other, mut other_c) = (0i64, 0i64, 0i64, 0i64);
    for t in 0..ev.len() {
        let near = (t.saturating_sub(3)..t).any(|u| ev[u].stroked);
        let click = (ev[t].moved == Some(CLICK)) as i64;
        if near { after += 1; after_c += click } else { other += 1; other_c += click }
    }
    if after == 0 || other == 0 {
        return 0;
    }
    (after_c * 1024 / after - other_c * 1024 / other) as i32
}

/// Echo with partial credit (`evolve::set_partial`): echo's matches, with lags 5 to 8 counted at
/// half (against 13 to 16), per taught token shown rather than per token said, so a network that
/// says few tokens is not scored on a handful. Echoing every taught token two steps later scores
/// about 1024; saying nothing or at random about 0.
pub fn echo_graded(events: &[Event]) -> i32 {
    let ev: Vec<&Event> = events.iter().filter(|e| e.scene == SceneKind::Sequence).collect();
    let far = 16;
    let taught = ev.iter().skip(far).filter(|e| e.taught && e.shown.is_some()).count() as i64;
    if taught == 0 {
        return 0;
    }
    let hits = |lags: std::ops::RangeInclusive<usize>| -> i64 {
        lags.map(|l| (far..ev.len()).filter(|&t| ev[t - l].taught && says(ev[t], ev[t - l].shown)).count() as i64).sum()
    };
    ((2 * (hits(1..=4) - hits(9..=12)) + hits(5..=8) - hits(13..=16)) * 512 / taught) as i32
}

/// Surprise with partial credit (`evolve::set_partial`): as `surprise`, and clicks 4 to 6 steps
/// after a token count at half (kept tokens: none broken in the 6 steps before).
pub fn surprise_graded(events: &[Event]) -> i32 {
    let ev: Vec<&Event> = events.iter().filter(|e| e.scene == SceneKind::Sequence).collect();
    let clicked = |t: usize, a: usize, b: usize| (t + a..=(t + b).min(ev.len().saturating_sub(1))).any(|u| ev[u].moved == Some(CLICK));
    let (mut odd, mut kept) = (0i64, 0i64);
    let (mut odd_near, mut odd_late, mut kept_near, mut kept_late) = (0i64, 0i64, 0i64, 0i64);
    for t in 0..ev.len().saturating_sub(6) {
        if ev[t].shown.is_none() {
            continue;
        }
        if ev[t].odd {
            odd += 1;
            odd_near += clicked(t, 1, 3) as i64;
            odd_late += clicked(t, 4, 6) as i64;
        } else if !(t.saturating_sub(6)..t).any(|u| ev[u].odd) {
            kept += 1;
            kept_near += clicked(t, 1, 3) as i64;
            kept_late += clicked(t, 4, 6) as i64;
        }
    }
    if odd == 0 || kept == 0 {
        return 0;
    }
    ((odd_near * 1024 / odd - kept_near * 1024 / kept) + (odd_late * 512 / odd - kept_late * 512 / kept)) as i32
}

/// Surprise shaped from clicking at all (`evolve::set_partial_surprise`): a quarter for clicking after
/// breaks at all (hit, the share of breaks with a click in the 3 steps after), three quarters for
/// lining up with them (hit less the same share after steps with no break in the 3 before), a
/// click 4 to 6 steps late counting half in both. Never clicking scores 0, clicking always 256,
/// and each false alarm dropped while the breaks are still clicked raises it (a click on every
/// break about 740: a late window also catches the next break's click; test
/// `shaped_surprise_pays_clicking_then_clicking_on_breaks`). For selection only: a rung's goal stays on `surprise`.
pub fn surprise_shaped(events: &[Event]) -> i32 {
    let ev: Vec<&Event> = events.iter().filter(|e| e.scene == SceneKind::Sequence).collect();
    let clicked = |t: usize, a: usize, b: usize| (t + a..=(t + b).min(ev.len().saturating_sub(1))).any(|u| ev[u].moved == Some(CLICK));
    // each step's credit: 2 a click in the 3 steps after, else 1 one 4 to 6 steps after, else 0
    let credit = |t: usize| if clicked(t, 1, 3) { 2 } else if clicked(t, 4, 6) { 1 } else { 0 };
    let (mut odd, mut odd_c, mut kept, mut kept_c) = (0i64, 0i64, 0i64, 0i64);
    for t in 0..ev.len().saturating_sub(6) {
        if ev[t].shown.is_none() {
            continue;
        }
        if ev[t].odd {
            odd += 1;
            odd_c += credit(t);
        } else if !(t.saturating_sub(6)..t).any(|u| ev[u].odd) {
            kept += 1;
            kept_c += credit(t);
        }
    }
    if odd == 0 || kept == 0 {
        return 0;
    }
    let hit = odd_c * 512 / odd;
    let fa = kept_c * 512 / kept;
    (hit / 4 + (hit - fa) * 3 / 4) as i32
}

/// Cues and questions (`World::flag`): clicking in the three steps after a question that had a cue
/// since the last one, against after one that had none, less every click anywhere else counted as
/// a wrong answer: 1024 x (answered after a cue / such questions - answered after none / such
/// questions - stray clicks / questions). Holding a bit from the cue and reading and clearing it at
/// the question scores 1024; clicking at every question 0; a latch clicking from the cue to the
/// question, or clicking at random, below 0.
pub fn flag(events: &[Event]) -> i32 {
    let ev: Vec<&Event> = events.iter().filter(|e| e.scene == SceneKind::Flag).collect();
    let mut in_window = vec![false; ev.len()];
    let (mut armed, mut armed_hit, mut plain, mut plain_hit) = (0i64, 0i64, 0i64, 0i64);
    for (t, e) in ev.iter().enumerate() {
        if let Some(a) = e.question {
            let w = t + 1..(t + 4).min(ev.len());
            let hit = w.clone().any(|u| ev[u].moved == Some(CLICK));
            for u in w {
                in_window[u] = true;
            }
            if a {
                armed += 1;
                armed_hit += hit as i64;
            } else {
                plain += 1;
                plain_hit += hit as i64;
            }
        }
    }
    if armed == 0 || plain == 0 {
        return 0;
    }
    let stray = ev.iter().enumerate().filter(|(t, e)| !in_window[*t] && e.moved == Some(CLICK)).count() as i64;
    let mut score = armed_hit * 1024 / armed - plain_hit * 1024 / plain - stray * 1024 / (armed + plain);
    // (`set_miss_cost`) cued questions not answered
    if miss_cost() {
        score -= (armed - armed_hit) * 1024 / armed;
    }
    score.max(-1024) as i32
}

/// Remember with partial credit (`evolve::set_partial_remember`): a cue trial clicked in its window
/// counts in full, one clicked only in the two steps after its window at half, against the same on
/// no-cue trials. A late answer to the light scores something; clicking at random or always 0. A
/// trial whose window is not marked (spent by an early click, `World::go_tone`) scores nothing.
pub fn remember_graded(events: &[Event]) -> i32 {
    // each trial: (cue, its credit), and steps since its window closed
    let mut tr: Vec<(bool, i64)> = Vec::new();
    let mut cur = None;
    let mut after = usize::MAX;
    for e in events {
        let Some((trial, _, cue, _)) = e.trial else { continue };
        if cur != Some(trial) {
            tr.push((cue, 0));
            cur = Some(trial);
            after = usize::MAX;
        }
        let last = tr.last_mut().unwrap();
        if e.window {
            after = 0;
            if e.moved == Some(CLICK) {
                last.1 = 1024;
            }
        } else if after != usize::MAX {
            after += 1;
            if after <= 2 && e.moved == Some(CLICK) && last.1 == 0 {
                last.1 = 512;
            }
        }
    }
    let mean = |c: bool| {
        let v: Vec<i64> = tr.iter().filter(|x| x.0 == c).map(|x| x.1).collect();
        if v.is_empty() { 0 } else { v.iter().sum::<i64>() / v.len() as i64 }
    };
    if tr.iter().all(|x| x.0) || tr.iter().all(|x| !x.0) {
        return 0;
    }
    (mean(true) - mean(false)) as i32
}

/// Feel with partial credit (`evolve::set_partial`): clicks 1 to 3 steps after a stroke count in
/// full, 4 to 6 at half and 7 to 9 at a quarter, each against the click rate at steps with no
/// stroke in the 9 before. A late answer to the stroke scores something; ignoring it scores 0.
pub fn feel_graded(events: &[Event]) -> i32 {
    let ev: Vec<&Event> = events.iter().filter(|e| e.scene == SceneKind::Feel).collect();
    let mut n = [0i64; 4];
    let mut c = [0i64; 4];
    let mut since = usize::MAX;
    for t in 0..ev.len() {
        if t > 0 && ev[t - 1].stroked {
            since = 1;
        } else if since != usize::MAX {
            since += 1;
        }
        let b = match since { 1..=3 => 0, 4..=6 => 1, 7..=9 => 2, _ => 3 };
        n[b] += 1;
        c[b] += (ev[t].moved == Some(CLICK)) as i64;
    }
    if n[3] == 0 {
        return 0;
    }
    let other = c[3] * 1024 / n[3];
    let rate = |b: usize| if n[b] == 0 { other } else { c[b] * 1024 / n[b] };
    ((rate(0) - other) + (rate(1) - other) / 2 + (rate(2) - other) / 4) as i32
}

/// Task selection, in a lifetime of blocks that alternate between two tasks (`evolve::Task::Mixed`):
/// responding (clicks while a light is up minus clicks in a matched stretch with none, in the
/// conditioning blocks) and babbling (the shown tokens said at least once, in the reading blocks),
/// their mean, minus doing the other task's thing: the voice used during conditioning blocks and
/// clicks during reading blocks (each per 1024 of those blocks' steps). A selector scores high;
/// one that does both all the time scores far below 0; one that does only one task about half.
/// (Set by `evolve::probe_scores_in` from a mixed lifetime: `probes` leaves it 0.)
pub fn select(events: &[Event]) -> i32 {
    if select_contrast_on() {
        return contrast_score(events);
    }
    let (respond, babbled, voice, clicks) = select_parts(events);
    match still(events) {
        // (three tasks) the still blocks too: neither click nor voice there
        Some(s) => (respond + babbled + s) / 3 - voice - clicks,
        None => (respond + babbled) / 2 - voice - clicks,
    }
}

/// (a mixed lifetime with still blocks, `Lesson::mixed_three`) 1024 less the rate of steps with a
/// click or a voice in the still blocks; None if there are none.
pub fn still(events: &[Event]) -> Option<i32> {
    let st: Vec<&Event> = events.iter().filter(|e| e.task == Some(2)).collect();
    if st.is_empty() {
        return None;
    }
    let acts = st.iter().filter(|e| e.moved == Some(CLICK) || e.said.is_some() || !e.sentence.is_empty()).count();
    Some(1024 - (acts * 1024 / st.len()) as i32)
}

/// What a network does, for the novelty island (`evolve::set_novelty`; probes 21 to 28, never a
/// goal): over every event of the lifetimes given, in parts per 1024, its click rate in conditioning
/// scenes and in token streams, its voice rate in token streams and in conditioning, its click rate
/// in the three steps after a stroke, its share of steps with another move (a turn, up or down), how
/// many of eight tokens it said, and its click rate in the three steps after an odd token. Acts in
/// their context, read from the log alone.
pub fn behaviour(events: &[&[Event]]) -> [i32; 8] {
    let all: Vec<&Event> = events.iter().flat_map(|e| e.iter()).collect();
    let rate = |pick: &dyn Fn(&Event) -> bool, act: &dyn Fn(&Event) -> bool| -> i32 {
        let n = all.iter().filter(|e| pick(e)).count();
        if n == 0 { 0 } else { (all.iter().filter(|e| pick(e) && act(e)).count() * 1024 / n) as i32 }
    };
    let cond = |e: &Event| matches!(e.scene, SceneKind::Conditioning(_));
    let seq = |e: &Event| e.scene == SceneKind::Sequence;
    let click = |e: &Event| e.moved == Some(CLICK);
    let voice = |e: &Event| e.said.is_some() || !e.sentence.is_empty();
    let after = |flag: &dyn Fn(&Event) -> bool| -> i32 {
        let (mut n, mut c) = (0usize, 0usize);
        for (t, e) in all.iter().enumerate() {
            if (t.saturating_sub(3)..t).any(|u| flag(all[u])) {
                n += 1;
                c += click(e) as usize;
            }
        }
        if n == 0 { 0 } else { (c * 1024 / n) as i32 }
    };
    let mut said = 0u64;
    for e in &all {
        if let Some(t) = e.said {
            said |= 1 << t.min(7);
        }
        for &t in &e.sentence {
            said |= 1 << t.min(7);
        }
    }
    [
        rate(&cond, &click),
        rate(&seq, &click),
        rate(&seq, &voice),
        rate(&cond, &voice),
        after(&|e: &Event| e.stroked),
        rate(&|_| true, &|e: &Event| e.moved.is_some() && e.moved != Some(CLICK)),
        (said.count_ones() * 128) as i32,
        after(&|e: &Event| e.odd),
    ]
}

/// Task selection with partial credit (`evolve::set_partial_select`): the costs, the voice in
/// conditioning and clicks in reading, counted at a quarter. Clicking at whatever the eye shows
/// then scores above babbling alone ((1024 + 1024) / 2 - 1024 / 4 = 768 against about 500), where
/// `select` gives it 0, so a gradient leads to responding before it is gated by the task.
pub fn select_graded(events: &[Event]) -> i32 {
    if select_contrast_on() {
        return contrast_score(events);
    }
    let (respond, babbled, voice, clicks) = select_parts(events);
    match still(events) {
        Some(s) => (respond + babbled + s) / 3 - (voice + clicks) / 4,
        None => (respond + babbled) / 2 - (voice + clicks) / 4,
    }
}

/// The parts of `select`: responding in the conditioning blocks, babbling in the reading blocks,
/// and the costs, the voice in conditioning and clicks in reading (each per 1024 steps); all 0 if
/// a lifetime lacks either kind of block.
pub fn select_parts(events: &[Event]) -> (i32, i32, i32, i32) {
    // (the same stimuli in every block, `Lesson::mixed_same`: a babble block is a conditioning
    // scene) the parts by each step's task: responding in respond blocks, the voice's variety in
    // babble blocks (each of two tokens said at least once, half each), the voice in respond blocks
    // and clicks in babble blocks as the costs
    if events.iter().any(|e| e.task == Some(1) && matches!(e.scene, SceneKind::Conditioning(_))) {
        let resp: Vec<Event> = events.iter().filter(|e| e.task == Some(0)).cloned().collect();
        let bab: Vec<&Event> = events.iter().filter(|e| e.task == Some(1)).collect();
        if resp.is_empty() || bab.is_empty() {
            return (0, 0, 0, 0);
        }
        let tr = trials(&resp, (3, 5), (7, 9));
        let all = |_: &Trial| true;
        let respond = rate(&tr, all, |x| x.3) - rate(&tr, all, |x| x.4);
        let voice = (resp.iter().filter(|e| e.said.is_some() || !e.sentence.is_empty()).count() * 1024 / resp.len()) as i32;
        let clicks = (bab.iter().filter(|e| e.moved == Some(CLICK)).count() * 1024 / bab.len()) as i32;
        let mut said = 0u64;
        for e in &bab {
            if let Some(t) = e.said {
                said |= 1 << t.min(63);
            }
            for &t in &e.sentence {
                said |= 1 << t.min(63);
            }
        }
        return (respond, (said.count_ones() as i32 * 512).min(1024), voice, clicks);
    }
    let cond: Vec<Event> = events.iter().filter(|e| matches!(e.scene, SceneKind::Conditioning(_))).cloned().collect();
    // (the still blocks are not reading blocks: `still` scores them)
    let seq: Vec<&Event> = events.iter().filter(|e| e.scene == SceneKind::Sequence && e.task != Some(2)).collect();
    if cond.is_empty() || seq.is_empty() {
        return (0, 0, 0, 0);
    }
    let tr = trials(&cond, (3, 5), (7, 9));
    let all = |_: &Trial| true;
    let respond = rate(&tr, all, |x| x.3) - rate(&tr, all, |x| x.4);
    let voice = (cond.iter().filter(|e| e.said.is_some()).count() * 1024 / cond.len()) as i32;
    let clicks = (seq.iter().filter(|e| e.moved == Some(CLICK)).count() * 1024 / seq.len()) as i32;
    let reading: Vec<Event> = events.iter().filter(|e| e.task != Some(2)).cloned().collect();
    (respond, babble(&reading), voice, clicks)
}

/// Repeating what paid: over a maze lifetime's choices in order, how often the next turn is the
/// same as this one when this one was stroked, minus when it was not, in parts per 1024. With
/// the food on one side all lifetime and no cue (`maze_fixed`, `maze_nocue`), it is learning from
/// reward alone: a learner turns the other way after a miss and keeps the paying turn. A lifetime
/// with no stroked choice to compare against an unstroked one scores 0, so a turner that ignores
/// strokes and keeps to one side scores 0 whichever side pays, as do random turners on average.
pub fn repeat(events: &[Event]) -> i32 {
    let turns: Vec<(Option<usize>, bool)> = events.iter().filter_map(|e| e.chose.filter(|_| e.scene == SceneKind::Maze).map(|c| (e.moved, c))).collect();
    let (mut won, mut won_n, mut lost, mut lost_n) = (0i64, 0i64, 0i64, 0i64);
    for w in turns.windows(2) {
        let same = (w[0].0 == w[1].0) as i64;
        if w[0].1 { won_n += 1; won += same } else { lost_n += 1; lost += same }
    }
    if won_n == 0 || lost_n == 0 {
        return 0;
    }
    (won * 1024 / won_n - lost * 1024 / lost_n) as i32
}

/// The T-maze: choices at the junction toward the food minus away from it, per trial, in parts
/// per 1024. A trial that never reaches a choice counts 0, so walking the stem matters; a blind
/// walker turns either way as often whatever the cue and scores 0.
pub fn maze(events: &[Event]) -> i32 {
    let trials = events.iter().filter_map(|e| e.maze.map(|m| m.0)).collect::<std::collections::BTreeSet<_>>().len() as i64;
    if trials == 0 {
        return 0;
    }
    let (right, wrong) = events.iter().fold((0i64, 0i64), |(r, w), e| match e.chose {
        Some(true) => (r + 1, w),
        Some(false) => (r, w + 1),
        None => (r, w),
    });
    // (`set_miss_cost`) a trial with no choice counts as a wrong one
    let none = if miss_cost() { (trials - right - wrong).max(0) } else { 0 };
    ((right - wrong - none) * 1024 / trials).max(-1024) as i32
}

/// The screen: clicks on the target above chance, per trial, in parts per 1024: (hits x cells -
/// clicks) / (trials x (cells - 1)). A click where the cursor happens to be hits 1 in `cells`
/// (the target and the cursor start on random cells), so a blind clicker scores about 0 and
/// moving the cursor to the target and clicking it on every trial scores 1024.
pub fn screen(events: &[Event]) -> i32 {
    let mut trials = std::collections::BTreeSet::new();
    let (mut cells, mut clicks, mut hits) = (2i64, 0i64, 0i64);
    for e in events {
        if let Some((t, c)) = e.screen {
            trials.insert(t);
            cells = c as i64;
        }
        if e.scene == SceneKind::Screen {
            if let Some(h) = e.chose {
                clicks += 1;
                hits += h as i64;
            }
        }
    }
    if trials.is_empty() {
        return 0;
    }
    // (at least -1024: with the screen going on, `World::screen_stream`, clicks are not one a trial)
    let n = trials.len() as i64;
    // (`set_miss_cost`) a trial with no hit counts against
    let miss = if miss_cost() { (n - hits.min(n)) * 1024 / n } else { 0 };
    ((hits * cells - clicks) * 1024 / (n * (cells - 1)) - miss).max(-1024) as i32
}

/// The screen with partial credit (`evolve::set_partial_screen`): as `screen`, and a trial in which
/// the cursor reached the target but no click hit it counts half a hit. A searcher that does not
/// yet click scores; one that also clicks on what it finds scores more. (With the cursor starting
/// off the target, `World::screen_away`, reaching it takes a move.)
pub fn screen_graded(events: &[Event]) -> i32 {
    let mut trials = std::collections::BTreeMap::new();
    let mut cells = 2i64;
    let (mut clicks, mut hits) = (0i64, 0i64);
    for e in events {
        if let Some((t, c)) = e.screen {
            cells = c as i64;
            let r = trials.entry(t).or_insert((false, false));
            if e.on_target {
                r.0 = true;
            }
            if e.chose == Some(true) {
                r.1 = true;
            }
        }
        if e.scene == SceneKind::Screen {
            if let Some(h) = e.chose {
                clicks += 1;
                hits += h as i64;
            }
        }
    }
    if trials.is_empty() {
        return 0;
    }
    let reached = trials.values().filter(|r| r.0 && !r.1).count() as i64;
    // (a hit is worth cells - 1 here: its click less; a trial that only reached the target, half that)
    ((((hits * cells - clicks) * 1024 + reached * (cells - 1) * 512) / (trials.len() as i64 * (cells - 1))).max(-1024)) as i32
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
    fn surprise_scores_a_clicker_that_predicts_and_not_a_blind_one() {
        // a cycle of three tokens, one in eight replaced by another; test-only policies: one that
        // sees the observer's marks clicks the step after a break, one clicks at random, one on
        // every token
        let run = |click: &dyn Fn(bool, &mut rand::rngs::StdRng) -> bool| -> i32 {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(2);
            let mut odd = vec![];
            let toks: Vec<usize> = (0..400).map(|i| {
                let c = [5, 1, 6][i % 3];
                let t = if rand::Rng::gen_ratio(&mut rng, 1, 8) { rand::Rng::gen_range(&mut rng, 0..8) } else { c };
                odd.push(t != c);
                t
            }).collect();
            let mut w = World::new(6, 8);
            w.sequence_marked(&toks, &odd);
            let mut last_odd = false;
            while !w.done() {
                let a = w.act(None, click(last_odd, &mut rng).then_some(CLICK));
                w.step(&a);
                last_odd = w.events.last().unwrap().odd;
            }
            surprise(&w.events)
        };
        let predicts = run(&|o, _| o);
        let random = run(&|_, r| rand::Rng::gen_ratio(r, 1, 3));
        let always = run(&|_, _| true);
        let never = run(&|_, _| false);
        assert!(predicts > 800, "{predicts}");
        assert!(random.abs() < 150, "{random}");
        assert_eq!(always, 0);
        assert_eq!(never, 0);
    }

    #[test]
    fn partial_feel_credits_a_late_answer_that_feel_does_not() {
        // a clicker that answers each stroke 5 steps later: outside feel's 3 steps, inside the
        // graded probe's half-credit window
        let run = |delay: usize| -> (i32, i32) {
            let mut w = World::new(6, 8);
            w.feel(3000);
            let mut since = usize::MAX;
            while !w.done() {
                let a = w.act(None, (since == delay).then_some(CLICK));
                w.step(&a);
                since = if w.events.last().is_some_and(|e| e.stroked) { 0 } else { since.saturating_add(1) };
            }
            (feel(&w.events), feel_graded(&w.events))
        };
        let (f, g) = run(5);
        assert!(f < 100, "{f}");
        assert!(g > 100, "{g}");
        let (f, g) = run(1);
        assert!(f > 300 && g >= f, "{f} {g}");
    }

    #[test]
    fn shapes_are_fresh_exemplars_of_the_lights() {
        // with blur and shift, each trial's light is a distorted copy: the same number of lines,
        // different from trial to trial, still overlapping its own light more than the other one
        let mut w = World::new(5, 4);
        w.blur = 2;
        w.shift = 0;
        w.conditioning(Conditioning::Reversal, 60);
        let none = BitVector::new(ACT_BITS, Some(0));
        let mut seen: Vec<(bool, BitVector)> = vec![];
        while !w.done() {
            let senses = w.step(&none);
            let e = w.events.last().unwrap();
            if let Some((_, 2, cue, _)) = e.trial {
                seen.push((cue, channel(&senses, EYE)));
            }
        }
        assert!(seen.iter().all(|(_, x)| x.count_ones() == ACTIVE));
        let same = |a: &BitVector, b: &BitVector| a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum::<u32>();
        let a: Vec<&BitVector> = seen.iter().filter(|x| x.0).map(|x| &x.1).collect();
        let b: Vec<&BitVector> = seen.iter().filter(|x| !x.0).map(|x| &x.1).collect();
        assert!(a.windows(2).any(|p| p[0].as_words() != p[1].as_words()), "fresh each trial");
        let within: u32 = a.windows(2).map(|p| same(p[0], p[1])).sum::<u32>() / (a.len() as u32 - 1);
        let across: u32 = a.iter().zip(&b).map(|(x, y)| same(x, y)).sum::<u32>() / a.len().min(b.len()) as u32;
        assert!(within > across + 4, "within {within} across {across}");
    }

    #[test]
    fn a_mirrored_maze_cue_stops_a_wired_turn_and_not_a_learner() {
        // policies that see the eye and the affect channel: a reflex turns to the half of the eye
        // the cue lit on the stem; a learner does the same but flips its rule after a choice that
        // brought no stroke; on 24 worlds, the cue as it was and mirrored
        let run = |mirror: bool, learner: bool, seed: u64| -> i32 {
            let mut w = World::new(seed, 4);
            w.maze_len = 2;
            w.gap_max = 2;
            w.maze_mirror = mirror;
            w.maze(100);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            let (mut right, mut flip, mut chose, mut at, mut decided) = (false, false, false, None, usize::MAX);
            while !w.done() {
                let eye = channel(&senses, EYE);
                let ws = eye.as_words();
                let half = ws.len() / 2;
                let (l, r): (u32, u32) = (ws[..half].iter().map(|x| x.count_ones()).sum(), ws[half..].iter().map(|x| x.count_ones()).sum());
                if channel(&senses, AFFECT).count_ones() > 0 {
                    chose = false;
                }
                let hand = match at {
                    Some((trial, p, _)) if p >= 2 => {
                        // (one choice a trial: its first step at the junction)
                        if trial != decided {
                            if learner && chose {
                                flip = !flip;
                            }
                            chose = true;
                            decided = trial;
                        }
                        Some(if right != flip { RIGHT } else { LEFT })
                    }
                    Some(_) => {
                        if l + r > 0 {
                            right = r > l;
                        }
                        Some(CLICK)
                    }
                    None => None,
                };
                let a = w.act(None, hand);
                senses = w.step(&a);
                at = w.events.last().unwrap().maze;
            }
            maze(&w.events)
        };
        let mean = |mirror: bool, learner: bool| (0..24u64).map(|s| run(mirror, learner, 500 + s)).sum::<i32>() / 24;
        let (reflex, reflex_m, learn, learn_m) = (mean(false, false), mean(true, false), mean(false, true), mean(true, true));
        eprintln!("maze: reflex {reflex}, mirrored {reflex_m}; learner {learn}, mirrored {learn_m}");
        assert!(reflex > 600 && reflex_m < -600, "a wired turn is right as it was and wrong mirrored");
        assert!(learn > 300 && learn_m > 300, "a learner is right either way: {learn} {learn_m}");
    }

    #[test]
    fn a_location_detector_tells_blurred_lights_apart_only_in_different_words() {
        // a fixed detector clicks when the eye's first half holds lines; under blur 2, 48 worlds
        let run = |same: bool, seed: u64| -> i32 {
            let mut w = World::new(seed, 4);
            (w.gap_max, w.reversal_period, w.blur) = (6, 500, 2);
            if same {
                w.same_word_lights();
            }
            w.conditioning(Conditioning::Reversal, 200);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            while !w.done() {
                let ws = channel(&senses, EYE).as_words().to_vec();
                let lit = ws[..ws.len() / 2].iter().any(|&x| x != 0);
                let a = w.act(None, if lit { Some(CLICK) } else { None });
                senses = w.step(&a);
            }
            probes(&w.events, 500, &[], &[], &[], &[], &[])[1]
        };
        let mean = |same: bool| (0..48u64).map(|s| run(same, 800 + s)).sum::<i32>() / 48;
        let (apart, one) = (mean(false), mean(true));
        eprintln!("tell, location detector: lights in their own words {apart}, in one word {one}");
        assert!(apart > 200 && one.abs() < 60, "{apart} {one}");
    }

    #[test]
    fn the_one_word_voice_is_the_same_in_a_conditioning_and_a_reading_lifetime() {
        let (mut a, mut b) = (World::new(3, 8), World::new(4, 8));
        a.fix_body(77);
        b.fix_body(77);
        a.one_word(11, 8);
        b.one_word_voice(12, 8);
        for k in 0..8 {
            assert_eq!(a.voice.command(k).as_words(), b.voice.command(k).as_words(), "token {k}");
            assert_eq!(a.sounds[k].as_words(), b.sounds[k].as_words());
            assert_eq!(b.voice.command(k).count_ones(), 8);
        }
    }

    #[test]
    fn the_same_word_lights_differ_only_by_pattern() {
        let mut w = World::new(9, 4);
        w.same_word_lights();
        let (a, b) = (w.lights[0].as_words().to_vec(), w.lights[1].as_words().to_vec());
        let wa: Vec<usize> = (0..a.len()).filter(|&i| a[i] != 0).collect();
        let wb: Vec<usize> = (0..b.len()).filter(|&i| b[i] != 0).collect();
        assert_eq!(wa, wb, "one word");
        assert_eq!(wa.len(), 1);
        assert_eq!(a[wa[0]].count_ones(), b[wb[0]].count_ones());
        assert_ne!(a, b);
    }

    #[test]
    fn the_maze_scores_turning_to_the_food_and_not_a_blind_walker() {
        // test-only policies that see the world's state: one walks the stem and turns to the
        // food's side; one walks and turns at random; one turns at once (never walks the stem)
        let run = |len: usize, cue_gone: bool, policy: &dyn Fn(Option<(usize, usize, bool)>, &mut rand::rngs::StdRng) -> Option<usize>| -> (i32, usize) {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
            let mut w = World::new(3, 4);
            w.maze_len = len;
            w.cue_gone = cue_gone;
            w.gap_max = 2;
            w.maze(100);
            let mut at = None;
            while !w.done() {
                let a = w.act(None, policy(at, &mut rng));
                w.step(&a);
                at = w.events.last().unwrap().maze;
            }
            (maze(&w.events), w.events.iter().filter(|e| e.stroked).count())
        };
        let knows = |at: Option<(usize, usize, bool)>, _: &mut rand::rngs::StdRng| match at {
            Some((_, p, side)) if p >= 2 => Some(if side { RIGHT } else { LEFT }),
            Some(_) => Some(CLICK),
            None => None,
        };
        let guesses = |at: Option<(usize, usize, bool)>, r: &mut rand::rngs::StdRng| match at {
            Some((_, p, _)) if p >= 2 => Some(if rand::Rng::gen_bool(r, 0.5) { RIGHT } else { LEFT }),
            Some(_) => Some(CLICK),
            None => None,
        };
        let (k, strokes) = run(2, false, &knows);
        assert!(k > 900, "{k}");
        assert!(strokes > 90, "the food is felt: {strokes}");
        let (g, _) = run(2, false, &guesses);
        assert!(g.abs() < 250, "{g}");
        // turning at once never reaches the junction of a stem of 2: no choices, 0
        let (t, _) = run(2, false, &|_, _| Some(RIGHT));
        assert_eq!(t, 0);
        // the cue: on the eye all along the stem, or only at its foot
        let mut w = World::new(3, 4);
        w.cue_gone = true;
        w.maze(3);
        let mut seen = vec![];
        while !w.done() {
            let senses = w.step(&w.act(None, Some(CLICK)));
            if let Some((_, p, _)) = w.events.last().unwrap().maze {
                seen.push((p, channel(&senses, EYE).count_ones()));
            }
        }
        // (each trial: the foot with the cue, then the stem dark, then the junction's code)
        assert_eq!(seen.iter().filter(|&&(p, n)| p == 0 && n > 0).count(), 3, "{seen:?}");
        assert!(seen.iter().all(|&(p, n)| p == 0 || p >= 2 || n == 0), "{seen:?}");
    }

    #[test]
    fn the_screen_scores_pointing_at_the_target_and_not_a_blind_clicker() {
        // test-only policies that read the world's state: one moves the cursor to the target and
        // clicks; one moves at random and clicks at random; one clicks at once
        let run = |(w_, h_, lit, cell, walls): (usize, usize, usize, usize, usize), policy: &dyn Fn(&World, &mut rand::rngs::StdRng) -> Option<usize>| -> (i32, usize) {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
            let mut w = World::new(3, 4);
            w.screen_w = w_;
            w.screen_h = h_;
            w.screen_lit = lit;
            w.screen_cell = cell;
            w.screen_walls = walls;
            w.gap_max = 2;
            w.screen(200);
            while !w.done() {
                let a = w.act(None, policy(&w, &mut rng));
                w.step(&a);
            }
            (screen(&w.events), w.events.iter().filter(|e| e.stroked).count())
        };
        // (the first move of a shortest path around the walls, by a flood back from the target)
        let points = |w: &World, _: &mut rand::rngs::StdRng| -> Option<usize> {
            let (cursor, target, walls, cols, cells) = w.screen_state_for_test()?;
            if cursor == target {
                return Some(CLICK);
            }
            let near = |c: usize| [(LEFT, (c % cols > 0).then(|| c - 1)), (RIGHT, (c % cols + 1 < cols && c + 1 < cells).then(|| c + 1)), (UP, (c >= cols).then(|| c - cols)), (DOWN, (c + cols < cells).then(|| c + cols))];
            let mut dist = vec![usize::MAX; cells];
            dist[target] = 0;
            let mut q = std::collections::VecDeque::from([target]);
            while let Some(c) = q.pop_front() {
                for (_, n) in near(c) {
                    if let Some(n) = n.filter(|&n| walls >> n & 1 == 0 && dist[n] == usize::MAX) {
                        dist[n] = dist[c] + 1;
                        q.push_back(n);
                    }
                }
            }
            near(cursor).into_iter().filter_map(|(m, n)| n.filter(|&n| walls >> n & 1 == 0).map(|n| (dist[n], m))).min().map(|x| x.1)
        };
        let wanders = |_: &World, r: &mut rand::rngs::StdRng| Some([LEFT, RIGHT, UP, DOWN, CLICK][rand::Rng::gen_range(r, 0..5)]);
        let at_once = |_: &World, _: &mut rand::rngs::StdRng| Some(CLICK);
        // (columns, rows, glyphs, cell bits, walls): up to a 2-D maze on 32-bit cells
        for set in [(2, 1, 1, 64, 0), (2, 2, 1, 64, 0), (4, 2, 3, 64, 0), (4, 4, 1, 32, 5), (8, 4, 2, 16, 10)] {
            let (p, strokes) = run(set, &points);
            assert!(p > 1000, "{set:?}: {p}");
            assert!(strokes > 190, "the target is felt: {strokes}");
            let (r, _) = run(set, &wanders);
            assert!(r.abs() < 150, "{set:?} wanders: {r}");
            let (o, _) = run(set, &at_once);
            assert!(o.abs() < 150, "{set:?} at once: {o}");
        }
        // a torch (a glyph shows only under the cursor): blind clickers still score about 0, and a
        // policy that reads only the eye (click when it holds more lines than the fewest seen,
        // the cursor alone; otherwise step aside) scores, where on a plain screen it does not
        let eye_only = |torch: bool| -> (i32, i32) {
            let mut w = World::new(3, 4);
            (w.screen_w, w.screen_h, w.screen_torch, w.gap_max) = (2, 1, torch, 2);
            w.screen(200);
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(2);
            let (mut fewest, mut senses, mut k) = (u32::MAX, BitVector::from_words(vec![0; SENSE_BITS / 64]), 0usize);
            while !w.done() {
                let ch = CH_BITS / 64;
                let lines: u32 = senses.as_words()[EYE * ch..(EYE + 1) * ch].iter().map(|x| x.count_ones()).sum();
                k += 1;
                let hand = if lines == 0 { None } else if lines > fewest { Some(CLICK) } else { Some(if k % 2 == 0 { LEFT } else { RIGHT }) };
                if lines > 0 { fewest = fewest.min(lines) }
                let a = w.act(None, hand);
                senses = w.step(&a);
            }
            let blind = {
                let mut w = World::new(3, 4);
                (w.screen_w, w.screen_h, w.screen_torch, w.gap_max) = (2, 1, torch, 2);
                w.screen(200);
                while !w.done() {
                    let a = w.act(None, Some([LEFT, RIGHT, CLICK][rand::Rng::gen_range(&mut rng, 0..3)]));
                    w.step(&a);
                }
                screen(&w.events)
            };
            (screen(&w.events), blind)
        };
        let (lit, blind) = eye_only(true);
        assert!(lit > 600 && blind.abs() < 150, "torch: eye-only {lit}, blind {blind}");
        let (plain, _) = eye_only(false);
        assert!(plain < 300, "plain: eye-only {plain}");
        // dark (no cursor mark) with the torch: the eye is empty unless the cursor is on the
        // target, so "click at anything seen, else step aside" scores: a respond. With the
        // cursor's mark shown it clicks at the cursor and scores about 0: the torch alone asks to
        // tell glyph-and-cursor from cursor
        let anything = |dark: bool| -> i32 {
            let mut w = World::new(3, 4);
            (w.screen_w, w.screen_h, w.screen_torch, w.screen_dark, w.gap_max) = (2, 1, true, dark, 2);
            w.screen(200);
            let (mut senses, mut k) = (BitVector::from_words(vec![0; SENSE_BITS / 64]), 0usize);
            while !w.done() {
                let ch = CH_BITS / 64;
                let seen = senses.as_words()[EYE * ch..(EYE + 1) * ch].iter().any(|&x| x != 0);
                k += 1;
                let hand = if seen { CLICK } else if k % 2 == 0 { LEFT } else { RIGHT };
                let a = w.act(None, Some(hand));
                senses = w.step(&a);
            }
            screen(&w.events)
        };
        let (dark, lit) = (anything(true), anything(false));
        assert!(dark > 600 && lit.abs() < 150, "dark {dark}, cursor shown {lit}");
        // a wall stops the cursor: on a 4 x 4 screen with walls the cursor never stands on one
        let mut w = World::new(3, 4);
        (w.screen_w, w.screen_h, w.screen_cell, w.screen_walls) = (4, 4, 32, 6);
        w.screen(50);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(5);
        while !w.done() {
            if let Some((cursor, _, walls, _, _)) = w.screen_state_for_test() {
                assert_eq!(walls >> cursor & 1, 0);
                assert_eq!(walls.count_ones(), 6);
            }
            let a = w.act(None, Some([LEFT, RIGHT, UP, DOWN][rand::Rng::gen_range(&mut rng, 0..4)]));
            w.step(&a);
        }
    }

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
            probes(&w.events, 50, &d.events, &[], &[], &[], &[])
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
        // win-stay: 0 for clickers that ignore strokes (blind, any light, light A only); the
        // oracle, which almost never clicks the light that does not pay, has little to compare
        // (a learner that does is `win_stay_scores_a_learner_and_not_a_stubborn_clicker`)
        assert!(blind[5].abs() < 80 && any_light[5].abs() < 80 && light_a[5].abs() < 80, "{blind:?} {any_light:?} {light_a:?}");
        assert!(follows[5] > -80, "{follows:?}");
        // the delayed lifetime clicks 3-5 steps after a light is gone: remembered
        assert!(blind[4] > 500, "{blind:?}");
    }

    #[test]
    fn remember_on_long_delays_distractors_varied_delays_and_two_lights() {
        // a policy that sees only the eye: on light A it remembers, and clicks in steps `from` to
        // `to` after it (an oracle for the window's timing on fixed delays), or (`until`) on every
        // step after it until the trial's light comes again; "any" remembers any light
        let run = |set: &dyn Fn(&mut World), seed: u64, policy: &dyn Fn(usize, bool) -> bool, any: bool| -> i32 {
            let mut d = World::new(seed, 4);
            set(&mut d);
            d.conditioning(Conditioning::Delayed, 200);
            let a = d.lights[0].clone();
            let mut prev = BitVector::new(SENSE_BITS, Some(0));
            let mut since = 999usize;
            let mut was_a = false;
            let mut trial = usize::MAX;
            while !d.done() {
                let eye = channel(&prev, EYE);
                if eye.count_ones() > 0 && (any || eye.as_words() == a.as_words()) {
                    since = 1;
                    was_a = eye.as_words() == a.as_words();
                } else {
                    since = since.saturating_add(1);
                }
                if let Scene::Conditioning(_, left, _, _, _) = d.scene {
                    if left != trial {
                        trial = left;
                    }
                }
                let act = d.act(None, if policy(since, was_a) { Some(CLICK) } else { None });
                prev = d.step(&act);
            }
            probes(&[], 50, &d.events, &[], &[], &[], &[])[4]
        };
        let fixed = |delay: usize| move |w: &mut World| w.delay = delay;
        for delay in [5usize, 8, 16] {
            // clicks in the window (the window opens `delay` steps after the light) score; a blind
            // clicker and one that never clicks score about 0
            let holder = run(&fixed(delay), 3, &|s, _| (delay..=delay + 1).contains(&s), false);
            let blind = run(&fixed(delay), 3, &|s, _| s % 2 == 0, false);
            assert!(holder > 600, "delay {delay}: {holder}");
            assert!(blind.abs() < 120, "delay {delay}: {blind}");
        }
        // the delay drawn each trial: holding until the window (clicking every step after the light)
        // scores; the old fixed-timing holder does not
        let vary = |w: &mut World| w.vary_delay = 8;
        let until = run(&vary, 5, &|s, _| s <= 12, false);
        let at3 = run(&vary, 5, &|s, _| (3..=4).contains(&s), false);
        assert!(until > 600, "{until}");
        assert!(at3 < until - 300, "{at3} {until}");
        // a distractor: remembering light A scores; remembering any light (the distractor too) less
        let dis = |w: &mut World| { w.delay = 8; w.distract = true; };
        let a_only = run(&dis, 7, &|s, a| a && (8..=9).contains(&s), false);
        let any_light = run(&dis, 7, &|s, _| s <= 9, true);
        assert!(a_only > 600, "{a_only}");
        assert!(any_light < a_only - 300, "{any_light} {a_only}");
        // two lights: remembering which one came scores; remembering that one came does not
        let two = |w: &mut World| { w.delay = 8; w.two_lights = true; };
        let which = run(&two, 9, &|s, a| a && (8..=9).contains(&s), true);
        let that = run(&two, 9, &|s, _| (8..=9).contains(&s), true);
        assert!(which > 600, "{which}");
        assert!(that.abs() < 120, "{that}");
        // the fixed rungs (two lights, a distractor of the other light, a tone at the window):
        // timing from any light scores about 0; holding which light came first scores
        let ear_run = |set: &dyn Fn(&mut World), seed: u64, policy: &dyn Fn(usize, bool, usize) -> bool| -> i32 {
            // policy(steps since the trial's first light, was it A, steps since the tone)
            let mut d = World::new(seed, 4);
            set(&mut d);
            d.conditioning(Conditioning::Delayed, 200);
            let a = d.lights[0].clone();
            let tone = d.tones[0].clone();
            let mut prev = BitVector::new(SENSE_BITS, Some(0));
            let (mut since, mut first_a, mut tone_since, mut any_since) = (999usize, false, 999usize, 999usize);
            let mut trial = usize::MAX;
            while !d.done() {
                if let Scene::Conditioning(_, left, t, _, _) = d.scene {
                    if left != trial || t == 0 {
                        trial = left;
                        since = 999;
                    }
                }
                let eye = channel(&prev, EYE);
                let lit = eye.count_ones() > 0;
                if lit && since >= 999 {
                    since = 1;
                    first_a = eye.as_words() == a.as_words();
                } else {
                    since = since.saturating_add(1);
                }
                any_since = if lit { 1 } else { any_since.saturating_add(1) };
                tone_since = if channel(&prev, EAR).as_words() == tone.as_words() { 1 } else { tone_since.saturating_add(1) };
                let _ = any_since;
                let act = d.act(None, if policy(since, first_a, tone_since) { Some(CLICK) } else { None });
                prev = d.step(&act);
            }
            probes(&[], 50, &d.events, &[], &[], &[], &[])[4]
        };
        let fair_distract = |w: &mut World| { w.delay = 8; w.distract = true; w.two_lights = true; };
        let held = ear_run(&fair_distract, 7, &|s, a, _| a && (8..=9).contains(&s));
        assert!(held > 600, "{held}");
        // a clicker 8 steps after any light (the distractor too) and a latch on any light
        let any8 = run(&fair_distract, 7, &|s, _| (8..=9).contains(&s), true);
        let latch = run(&fair_distract, 7, &|s, _| s <= 10, true);
        let latest_a = run(&fair_distract, 7, &|s, a| a && s <= 10, true);
        assert!(any8.abs() < 120 && latch.abs() < 120, "{any8} {latch}");
        assert!(latest_a < held - 300, "{latest_a} {held}");
        // a delay drawn each trial with a tone at the window: holding A until the tone scores;
        // delay lines from the light, a latch on any light, and clicking on the tone do not
        let fair_vary = |w: &mut World| { w.vary_delay = 8; w.go_tone = true; w.two_lights = true; };
        let hold = ear_run(&fair_vary, 5, &|s, a, g| a && s < 999 && g <= 2);
        assert!(hold > 600, "{hold}");
        let lines = ear_run(&fair_vary, 5, &|s, a, _| a && (s == 3 || s == 6));
        let on_tone = ear_run(&fair_vary, 5, &|_, _, g| g <= 2);
        let any_hold = ear_run(&fair_vary, 5, &|s, _, g| s < 999 && g <= 2);
        assert!(lines < 450 && lines < hold - 400, "{lines} {hold}");
        eprintln!("fair rungs: distract held {held}, any light at 8 {any8}, latch {latch}, latest A {latest_a}; vary held to the tone {hold}, delay lines {lines}, on the tone {on_tone}, any light held {any_hold}");
        assert!(on_tone.abs() < 120 && any_hold.abs() < 120, "{on_tone} {any_hold}");
        // graded: one step late scores about half, in the window in full, random clicks about 0
        let graded = |set: &dyn Fn(&mut World), seed: u64, policy: &dyn Fn(usize, bool) -> bool| -> (i32, i32) {
            let mut d = World::new(seed, 4);
            set(&mut d);
            d.conditioning(Conditioning::Delayed, 200);
            let a = d.lights[0].clone();
            let mut prev = BitVector::new(SENSE_BITS, Some(0));
            let (mut since, mut was_a) = (999usize, false);
            while !d.done() {
                let eye = channel(&prev, EYE);
                if eye.count_ones() > 0 {
                    since = 1;
                    was_a = eye.as_words() == a.as_words();
                } else {
                    since = since.saturating_add(1);
                }
                let act = d.act(None, if policy(since, was_a) { Some(CLICK) } else { None });
                prev = d.step(&act);
            }
            let r = probes(&[], 50, &d.events, &[], &[], &[], &[]);
            (r[4], r[18])
        };
        let lag8 = |w: &mut World| w.delay = 8;
        let (on_time, on_time_g) = graded(&lag8, 3, &|s, a| a && s == 8);
        let (late, late_g) = graded(&lag8, 3, &|s, a| a && s == 11);
        let (_, noise_g) = graded(&lag8, 3, &|s, _| s % 3 == 0);
        assert!(on_time > 900 && on_time_g > 900, "{on_time} {on_time_g}");
        assert!(late.abs() < 60 && (400..=620).contains(&late_g), "{late} {late_g}");
        assert!(noise_g.abs() < 150, "{noise_g}");
        // up to a delay of 5 the window is where it always was (steps 5-7)
        let mut w = World::new(1, 4);
        w.delay = 3;
        w.conditioning(Conditioning::Delayed, 5);
        while !w.done() {
            let a = w.act(None, None);
            w.step(&a);
        }
        assert!(w.events.iter().all(|e| e.trial.is_none() || e.window == (5..=7).contains(&e.trial.unwrap().1)));
    }

    #[test]
    fn feel_scores_clicking_after_a_stroke_and_not_clicking_regardless() {
        // policies that feel only the affect channel: one clicks the step after a stroke; one
        // clicks every step; one at random; one never
        let run = |kind: u8, seed: u64| -> i32 {
            let mut w = World::new(seed, 4);
            w.feel(1000);
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            while !w.done() {
                let touched = channel(&senses, AFFECT).count_ones() > 0;
                let click = match kind { 0 => touched, 1 => true, 2 => rand::Rng::gen_bool(&mut rng, 0.3), _ => false };
                let a = w.act(None, click.then_some(CLICK));
                senses = w.step(&a);
            }
            feel(&w.events)
        };
        let mean = |kind: u8| (0..12u64).map(|s| run(kind, 500 + s)).sum::<i32>() / 12;
        let (feels, always, random, never) = (mean(0), mean(1), mean(2), mean(3));
        assert!(feels > 250 && always == 0 && random.abs() < 60 && never == 0, "feels {feels}, always {always}, random {random}, never {never}");
    }

    #[test]
    fn select_scores_doing_each_task_in_its_block_and_not_both_at_once() {
        // blocks alternating a conditioning block (40 trials, no swaps) and a reading block (400
        // tokens, two of them), the block's task as `task_cue`; test-only policies: a selector
        // clicks at a light in conditioning blocks and says a random token in reading blocks; a
        // doer of both clicks at anything seen and says a token every step; a silent one
        let run = |kind: u8, seed: u64| -> i32 {
            let mut w = World::new(seed, 2);
            w.gap_max = 4;
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            let mut before = false;
            for b in 0..4 {
                w.task_cue = Some(b % 2);
                if b % 2 == 0 {
                    w.reversal_period = 40;
                    w.conditioning(Conditioning::Reversal, 40);
                } else {
                    let toks: Vec<usize> = (0..400).map(|_| rand::Rng::gen_range(&mut rng, 0..2)).collect();
                    w.sequence_marked(&toks, &vec![false; 400]);
                }
                while !w.done() {
                    let seen = channel(&senses, EYE).count_ones() > 0;
                    let second = seen && before;
                    before = seen;
                    let token = rand::Rng::gen_range(&mut rng, 0..2);
                    let (say, hand) = match (kind, w.task_cue) {
                        (0, Some(0)) => (None, second.then_some(CLICK)),
                        (0, _) => (Some(token), None),
                        (1, _) => (Some(token), seen.then_some(CLICK)),
                        _ => (None, None),
                    };
                    let a = w.act(say, hand);
                    senses = w.step(&a);
                }
            }
            select(&w.events)
        };
        let mean = |kind: u8| (0..12u64).map(|s| run(kind, 400 + s)).sum::<i32>() / 12;
        let (selector, both, silent) = (mean(0), mean(1), mean(2));
        assert!(selector > 600 && both < 0 && silent == 0, "selector {selector}, both {both}, silent {silent}");
    }

    #[test]
    fn feeling_in_the_lights_world_pays_a_stroke_follower_and_keeps_tell() {
        // the lights world with strokes in the gaps: a network that clicks at light A and after
        // any stroke keeps its tell and scores on feel-gap; one that clicks at light A alone scores
        // 0 on feel-gap; one that clicks every step scores 0 on both
        let run = |kind: u8, seed: u64| -> (i32, i32) {
            let mut w = World::new(seed, 2);
            w.gap_max = 6;
            w.gap_strokes = true;
            w.conditioning(Conditioning::Reversal, 60);
            let light_a = w.lights[0].as_words().to_vec();
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            while !w.done() {
                let a_seen = channel(&senses, EYE).as_words() == light_a.as_slice();
                let stroked = channel(&senses, AFFECT).count_ones() > 0;
                let click = match kind { 0 => a_seen || stroked, 1 => a_seen, _ => true };
                let a = w.act(None, click.then_some(CLICK));
                senses = w.step(&a);
            }
            let row = probes(&w.events, 60, &[], &[], &[], &[], &[]);
            (row[1], row[33])
        };
        let mean = |k: u8| { let v: Vec<(i32, i32)> = (0..10).map(|s| run(k, 70 + s)).collect(); (v.iter().map(|x| x.0).sum::<i32>() / 10, v.iter().map(|x| x.1).sum::<i32>() / 10) };
        let (both, tell_only, every) = (mean(0), mean(1), mean(2));
        eprintln!("(tell, feel-gap): light A and strokes {both:?}, light A alone {tell_only:?}, every step {every:?}");
        // (one click a stroke: feel-gap counts each of the three steps after it, as feel does)
        assert!(both.0 > 900 && both.1 > 300 && tell_only.1.abs() < 80 && every.1.abs() < 80, "{both:?} {tell_only:?} {every:?}");
    }

    #[test]
    fn misses_cost_so_doing_nothing_is_no_refuge() {
        // with the miss cost, a network that never acts scores -1024 on respond, flag and the
        // maze; a responder keeps its score
        set_miss_cost_here(Some(true));
        let cond = |act: bool, seed: u64| -> i32 {
            let mut w = World::new(seed, 2);
            w.gap_max = 4;
            w.conditioning(Conditioning::Reversal, 40);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            while !w.done() {
                let seen = channel(&senses, EYE).count_ones() > 0;
                let a = w.act(None, (act && seen).then_some(CLICK));
                senses = w.step(&a);
            }
            probes(&w.events, 40, &[], &[], &[], &[], &[])[0]
        };
        let fl = |seed: u64| -> i32 {
            let mut w = World::new(seed, 2);
            w.flag(400, 4, false, false);
            let senses_none = w.act(None, None);
            while !matches!(w.scene, Scene::Idle) {
                w.step(&senses_none);
            }
            flag(&w.events)
        };
        let (resp, idle, flag_idle) = (cond(true, 3), cond(false, 3), fl(3));
        set_miss_cost_here(None);
        eprintln!("miss cost: responder {resp}, never acting {idle}, flag never acting {flag_idle}");
        assert!(resp > 900 && idle == -1024 && flag_idle == -1024, "{resp} {idle} {flag_idle}");
    }

    #[test]
    fn contrast_scores_task_blind_acting_zero_and_selection_full() {
        // the same-stimulus mixed lifetime (held): a selector clicks at lights in respond blocks
        // and speaks in babble blocks, holding the cue; run 36's passer clicked at every light and
        // spoke now and then everywhere
        let run = |kind: u8, seed: u64| -> (i32, i32) {
            let mut w = World::new(seed, 2);
            w.gap_max = 4;
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            let cues: Vec<Vec<u64>> = (0..3).map(|k| w.cues[k % w.cues.len()].as_words().to_vec()).collect();
            let mut held: Option<usize> = None;
            for b in 0..10 {
                let task = if b < 2 { b } else { rand::Rng::gen_range(&mut rng, 0..2) };
                w.task = Some(task);
                w.task_cue = Some(task);
                w.task_cue_left = Some(3);
                w.no_pay = task != 0;
                w.reversal_period = 20;
                w.conditioning(Conditioning::Reversal, 20);
                while !w.done() {
                    let touch = channel(&senses, TOUCH).as_words().to_vec();
                    if let Some(f) = cues.iter().position(|c| *c == touch) {
                        held = Some(f);
                    }
                    let seen = channel(&senses, EYE).count_ones() > 0;
                    let token = rand::Rng::gen_range(&mut rng, 0..2);
                    let (say, hand) = match kind {
                        0 => match held { Some(0) => (None, seen.then_some(CLICK)), Some(1) => (Some(token), None), _ => (None, None) },
                        _ => (rand::Rng::gen_ratio(&mut rng, 1, 5).then_some(token), seen.then_some(CLICK)),
                    };
                    let a = w.act(say, hand);
                    senses = w.step(&a);
                }
            }
            (select(&w.events), select_contrast(&w.events))
        };
        let mean = |k: u8| { let v: Vec<(i32, i32)> = (0..12).map(|s| run(k, 950 + s)).collect(); (v.iter().map(|x| x.0).sum::<i32>() / 12, v.iter().map(|x| x.1).sum::<i32>() / 12) };
        let (sel, blind) = (mean(0), mean(1));
        // the performance contrast: the selector, the blind one, and a timid selector that clicks
        // at one light in ten in respond blocks and says a token one step in twenty in babble blocks
        let perf = |kind: u8| -> i32 {
            (0..12u64).map(|s| {
                let seed = 950 + s;
                let mut w = World::new(seed, 2);
                w.gap_max = 4;
                let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
                let mut senses = BitVector::new(SENSE_BITS, Some(0));
                let cues: Vec<Vec<u64>> = (0..3).map(|k| w.cues[k % w.cues.len()].as_words().to_vec()).collect();
                let mut held: Option<usize> = None;
                for b in 0..10 {
                    let task = if b < 2 { b } else { rand::Rng::gen_range(&mut rng, 0..2) };
                    w.task = Some(task);
                    w.task_cue = Some(task);
                    w.task_cue_left = Some(3);
                    w.no_pay = task != 0;
                    w.reversal_period = 20;
                    w.conditioning(Conditioning::Reversal, 20);
                    while !w.done() {
                        let touch = channel(&senses, TOUCH).as_words().to_vec();
                        if let Some(f) = cues.iter().position(|c| *c == touch) {
                            held = Some(f);
                        }
                        let seen = channel(&senses, EYE).count_ones() > 0;
                        let token = rand::Rng::gen_range(&mut rng, 0..2);
                        let (say, hand) = match (kind, held) {
                            (0, Some(0)) => (None, seen.then_some(CLICK)),
                            (0, Some(1)) => (Some(token), None),
                            (1, _) => (rand::Rng::gen_ratio(&mut rng, 1, 5).then_some(token), seen.then_some(CLICK)),
                            (2, Some(0)) => (None, (seen && rand::Rng::gen_ratio(&mut rng, 1, 10)).then_some(CLICK)),
                            (2, Some(1)) => (rand::Rng::gen_ratio(&mut rng, 1, 20).then_some(token), None),
                            _ => (None, None),
                        };
                        let a = w.act(say, hand);
                        senses = w.step(&a);
                    }
                }
                (select_performance(&w.events), select_contrast(&w.events))
            }).fold((0, 0), |a, x| (a.0 + x.0, a.1 + x.1)).0 / 12
        };
        let (p_sel, p_blind, p_timid) = (perf(0), perf(1), perf(2));
        eprintln!("select / contrast: selector {sel:?}, task-blind clicker and speaker {blind:?}; performance contrast: selector {p_sel}, blind {p_blind}, timid {p_timid}");
        assert!(p_sel > 700 && p_blind.abs() < 150 && p_timid < 200, "{p_sel} {p_blind} {p_timid}");
        assert!(sel.1 > 800 && blind.1.abs() < 150, "{sel:?} {blind:?}");
        assert!(blind.0 > 500, "the old score passed it: {blind:?}");
    }

    #[test]
    fn stray_clicks_cost_on_the_conditioning_rungs() {
        // reversal trials: a responder clicks at a light's window, a constant clicker every step
        // (respond less `stray_rate`, as `set_stray_cost` takes it; the switch itself is global)
        let run = |every: bool, seed: u64| -> i32 {
            let mut w = World::new(seed, 2);
            w.gap_max = 4;
            w.conditioning(Conditioning::Reversal, 40);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            while !w.done() {
                let seen = channel(&senses, EYE).count_ones() > 0;
                let a = w.act(None, (every || seen).then_some(CLICK));
                senses = w.step(&a);
            }
            (probes(&w.events, 40, &[], &[], &[], &[], &[])[0] - stray_rate(&w.events)).max(-1024)
        };
        let (resp, every) = ((0..8).map(|s| run(false, 30 + s)).sum::<i32>() / 8, (0..8).map(|s| run(true, 30 + s)).sum::<i32>() / 8);
        eprintln!("respond with stray cost: responder {resp}, constant clicker {every}");
        assert!(resp > 500 && every < -500, "{resp} {every}");
    }

    #[test]
    fn the_screen_goes_on_and_a_miss_is_felt() {
        // a searcher moves at random until the target is under the cursor, then clicks; a sitter
        // clicks at whatever it sees; with the stream a hit moves the target, a miss is felt
        let run = |search: bool, seed: u64| -> (i32, usize, usize) {
            let mut w = World::new(seed, 2);
            w.screen_stream = true;
            w.wrong_signal = true;
            w.screen_away = true;
            w.screen_w = 4;
            w.screen_h = 1;
            w.screen(20);
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let mut wrongs = 0;
            while !w.done() {
                let on = w.events.last().is_some_and(|e| e.on_target);
                let hand = if search { if on { Some(CLICK) } else { Some([LEFT, RIGHT][rand::Rng::gen_range(&mut rng, 0..2)]) } } else { Some(CLICK) };
                let a = w.act(None, hand);
                let senses = w.step(&a);
                wrongs += (channel(&senses, AFFECT).as_words()[0] == 0xAAAA_AAAA_AAAA_AAAA) as usize;
            }
            let hits = w.events.iter().filter(|e| e.chose == Some(true)).count();
            (screen(&w.events), hits, wrongs)
        };
        let s = run(true, 5);
        let c = run(false, 5);
        eprintln!("screen stream: searcher {s:?}, constant clicker {c:?} (score, hits, wrong signals)");
        assert!(s.1 >= 15 && s.0 > 0, "{s:?}");
        assert!(c.0 < 0 && c.2 > 10, "{c:?}");
    }

    #[test]
    fn flag_pays_a_bit_held_from_the_cue_to_the_question() {
        // strategies on the senses: a holder (set at the cue, answer and clear at the question),
        // a clicker at every question, a latch (from the cue, every step, to the question), a
        // fixed-delay clicker (3 steps after a cue), clicking at random, never
        let run = |kind: u8, seed: u64, gap: usize| -> i32 {
            let mut w = World::new(seed, 2);
            w.flag(600, gap, false, false);
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            let (mut held, mut since) = (false, usize::MAX);
            while !matches!(w.scene, Scene::Idle) {
                let seen = channel(&senses, EYE).count_ones() > 0;
                let heard = channel(&senses, EAR).count_ones() > 0;
                if seen {
                    since = 0;
                } else if since < usize::MAX {
                    since += 1;
                }
                let click = match kind {
                    0 => { let c = heard && held; if seen { held = true } if heard { held = false } c }
                    1 => heard,
                    2 => { if seen { held = true } let c = held; if heard { held = false } c }
                    3 => since == 3,
                    4 => rand::Rng::gen_ratio(&mut rng, 1, 6),
                    _ => false,
                };
                let a = w.act(None, click.then_some(CLICK));
                senses = w.step(&a);
            }
            flag(&w.events)
        };
        let mean = |k: u8, gap: usize| (0..12u64).map(|s| run(k, 700 + s, gap)).sum::<i32>() / 12;
        let m: Vec<i32> = (0..6).map(|k| mean(k, 8)).collect();
        eprintln!("flag (gaps to 8): holder {}, every question {}, latch {}, fixed delay {}, random {}, never {}", m[0], m[1], m[2], m[3], m[4], m[5]);
        assert!(m[0] > 900, "{m:?}");
        assert!(m[1].abs() < 60 && m[5] == 0, "{m:?}");
        assert!(m[2] < 0 && m[3] < 200 && m[4] < 0, "{m:?}");
    }

    #[test]
    fn shaped_surprise_pays_clicking_then_clicking_on_breaks() {
        // a stream with breaks 1 in 8; strategies that see the break flags, clicking the step after
        let mut w = World::new(1, 2);
        w.step(&w.act(None, None));
        let mut template = w.events[0].clone();
        template.scene = SceneKind::Sequence;
        let run = |policy: u8, seed: u64| -> (i32, i32) {
            let mut r = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let odd: Vec<bool> = (0..600).map(|_| rand::Rng::gen_ratio(&mut r, 1, 8)).collect();
            let mut ev = Vec::new();
            for t in 0..odd.len() {
                let mut e = template.clone();
                e.shown = Some(t % 3);
                e.odd = odd[t];
                let seen = t > 0 && odd[t - 1];
                let click = match policy {
                    0 => false,                                                // never
                    1 => true,                                                 // always
                    2 => rand::Rng::gen_ratio(&mut r, 1, 4),                   // at random
                    3 => seen || rand::Rng::gen_ratio(&mut r, 1, 4),           // breaks, and noise
                    4 => seen && rand::Rng::gen_ratio(&mut r, 1, 3),           // a third of the breaks
                    _ => seen,                                                 // every break
                };
                e.moved = click.then_some(CLICK);
                ev.push(e);
            }
            (surprise(&ev), surprise_shaped(&ev))
        };
        let mean = |p: u8| { let v: Vec<(i32, i32)> = (0..12).map(|s| run(p, 40 + s)).collect(); (v.iter().map(|x| x.0).sum::<i32>() / 12, v.iter().map(|x| x.1).sum::<i32>() / 12) };
        let m: Vec<(i32, i32)> = (0..6).map(mean).collect();
        eprintln!("surprise / surprise+: never {:?}, always {:?}, random {:?}, breaks and noise {:?}, a third of the breaks {:?}, every break {:?}", m[0], m[1], m[2], m[3], m[4], m[5]);
        // the old probe gives nothing to clicking at all; the shaped one does, and more as the
        // clicks line up with the breaks
        assert!(m[0].1 == 0 && m[1].0.abs() < 40 && m[2].0.abs() < 60);
        assert!(m[1].1 > 200 && m[2].1 > 50);
        assert!(m[3].1 > m[1].1 && m[5].1 > m[3].1 && m[5].1 > 700);
        assert!(m[4].1 > m[2].1);
    }

    #[test]
    fn graded_screen_pays_search_before_the_click_on_what_it_finds() {
        // a torch screen of two cells, the cursor starting off the target; policies that see the
        // eye: a sitter clicks when it sees a glyph and never moves; a wanderer moves at random and
        // never clicks; a searcher moves until it sees a glyph and then clicks
        let run = |kind: u8, seed: u64| -> (i32, i32) {
            let mut w = World::new(seed, 4);
            (w.screen_w, w.screen_h, w.screen_lit, w.screen_torch, w.screen_away, w.gap_max) = (2, 1, 1, true, true, 2);
            w.screen(100);
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            let mark = w.cursor_mark.count_ones();
            while !w.done() {
                let eye = channel(&senses, EYE);
                let glyph = eye.as_words().iter().any(|x| x.count_ones() > mark);
                let walk = if rand::Rng::gen_bool(&mut rng, 0.5) { LEFT } else { RIGHT };
                let hand = match kind {
                    0 => glyph.then_some(CLICK),
                    1 => Some(walk),
                    _ => Some(if glyph { CLICK } else { walk }),
                };
                let a = w.act(None, hand);
                senses = w.step(&a);
            }
            (screen(&w.events), screen_graded(&w.events))
        };
        let mean = |kind: u8| {
            let v: Vec<(i32, i32)> = (0..12u64).map(|s| run(kind, 600 + s)).collect();
            (v.iter().map(|x| x.0).sum::<i32>() / 12, v.iter().map(|x| x.1).sum::<i32>() / 12)
        };
        let (sit, wander, search) = (mean(0), mean(1), mean(2));
        eprintln!("screen (graded): sitter {sit:?}, wanderer {wander:?}, searcher {search:?}");
        assert!(wander.1 > sit.1 && search.1 > wander.1 && search.0 > 600 && wander.0 <= 0, "{sit:?} {wander:?} {search:?}");
    }

    #[test]
    fn graded_select_climbs_from_babbling_to_clicking_before_the_task_gates_it() {
        // a babbler says a token in reading blocks only (cued) and never clicks; a reflex also
        // clicks whenever the eye is lit, in either block; a selector clicks only in conditioning
        let run = |kind: u8, seed: u64| -> (i32, i32) {
            let mut w = World::new(seed, 2);
            w.gap_max = 4;
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            for b in 0..4 {
                w.task_cue = Some(b % 2);
                if b % 2 == 0 {
                    w.reversal_period = 40;
                    w.conditioning(Conditioning::Reversal, 40);
                } else {
                    let toks: Vec<usize> = (0..400).map(|_| rand::Rng::gen_range(&mut rng, 0..2)).collect();
                    w.sequence_marked(&toks, &vec![false; 400]);
                }
                while !w.done() {
                    let seen = channel(&senses, EYE).count_ones() > 0;
                    let reading = w.task_cue == Some(1);
                    let token = rand::Rng::gen_range(&mut rng, 0..2);
                    let say = if reading { Some(token) } else { None };
                    let hand = match kind { 1 => seen.then_some(CLICK), 2 => (seen && !reading).then_some(CLICK), _ => None };
                    let a = w.act(say, hand);
                    senses = w.step(&a);
                }
            }
            (select(&w.events), select_graded(&w.events))
        };
        let mean = |kind: u8| {
            let v: Vec<(i32, i32)> = (0..12u64).map(|s| run(kind, 900 + s)).collect();
            (v.iter().map(|x| x.0).sum::<i32>() / 12, v.iter().map(|x| x.1).sum::<i32>() / 12)
        };
        let (bab, refl, sel) = (mean(0), mean(1), mean(2));
        eprintln!("select (graded): babbler {bab:?}, reflex {refl:?}, selector {sel:?}");
        assert!(refl.0 < bab.0 && refl.1 > bab.1 && sel.1 > refl.1 && sel.0 > 600, "{bab:?} {refl:?} {sel:?}");
    }

    #[test]
    fn with_the_same_stimuli_only_the_held_cue_tells_the_task() {
        // every block a conditioning block (lights), its task drawn, the cue felt for 3 steps, clicks
        // paying only in respond blocks; a holder keeps the last cue felt; a timing reflex clicks at
        // lights and babbles always (what passed run 33's cued); a reactor acts only while the cue
        // is felt
        let run = |kind: u8, seed: u64| -> i32 {
            let mut w = World::new(seed, 2);
            w.gap_max = 4;
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            let cues: Vec<Vec<u64>> = (0..3).map(|k| w.cues[k % w.cues.len()].as_words().to_vec()).collect();
            let mut held: Option<usize> = None;
            for b in 0..10 {
                let task = if b < 2 { b } else { rand::Rng::gen_range(&mut rng, 0..2) };
                w.task = Some(task);
                w.task_cue = Some(task);
                w.task_cue_left = Some(3);
                w.no_pay = task != 0;
                w.reversal_period = 20;
                w.conditioning(Conditioning::Reversal, 20);
                while !w.done() {
                    let touch = channel(&senses, TOUCH).as_words().to_vec();
                    let felt = cues.iter().position(|c| *c == touch);
                    if felt.is_some() {
                        held = felt;
                    }
                    let seen = channel(&senses, EYE).count_ones() > 0;
                    let token = rand::Rng::gen_range(&mut rng, 0..2);
                    let now = match kind { 1 => None, 2 => felt, _ => held };
                    let (say, hand) = match (kind, now) {
                        (1, _) => (Some(token), seen.then_some(CLICK)),
                        (_, Some(0)) => (None, seen.then_some(CLICK)),
                        (_, Some(1)) => (Some(token), None),
                        _ => (None, None),
                    };
                    let a = w.act(say, hand);
                    senses = w.step(&a);
                }
            }
            select(&w.events)
        };
        let mean = |kind: u8| (0..12u64).map(|s| run(kind, 900 + s)).sum::<i32>() / 12;
        let (holder, timing, reactor) = (mean(0), mean(1), mean(2));
        eprintln!("same stimuli: holder {holder}, timing reflex {timing}, reactor {reactor}");
        assert!(holder > 600 && timing < 300 && reactor < holder - 300, "{holder} {timing} {reactor}");
    }

    #[test]
    fn held_three_tasks_need_the_cue_held_through_the_block() {
        // blocks of a drawn task (respond, babble, still), the cue felt for 3 steps; policies that
        // read the touch channel: a holder keeps the last cue it felt and does that task; a
        // reactor acts only while a cue is felt; a two-task selector (holds the cue, but babbles in
        // still blocks too)
        let run = |kind: u8, seed: u64| -> i32 {
            let mut w = World::new(seed, 2);
            w.gap_max = 4;
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            let cues: Vec<Vec<u64>> = (0..3).map(|k| w.cues[k % w.cues.len()].as_words().to_vec()).collect();
            let mut held: Option<usize> = None;
            for b in 0..9 {
                let task = if b < 3 { b } else { rand::Rng::gen_range(&mut rng, 0..3) };
                w.task = Some(task);
                w.task_cue = Some(task);
                w.task_cue_left = Some(3);
                if task == 0 {
                    w.reversal_period = 40;
                    w.conditioning(Conditioning::Reversal, 20);
                } else {
                    let toks: Vec<usize> = (0..200).map(|_| rand::Rng::gen_range(&mut rng, 0..2)).collect();
                    w.sequence_marked(&toks, &vec![false; 200]);
                }
                while !w.done() {
                    let touch = channel(&senses, TOUCH).as_words().to_vec();
                    let felt = cues.iter().position(|c| *c == touch);
                    if felt.is_some() {
                        held = felt;
                    }
                    let now = if kind == 1 { felt } else { held };
                    let seen = channel(&senses, EYE).count_ones() > 0;
                    let token = rand::Rng::gen_range(&mut rng, 0..2);
                    let (say, hand) = match now {
                        Some(0) => (None, seen.then_some(CLICK)),
                        Some(1) => (Some(token), None),
                        Some(_) if kind == 2 => (Some(token), None),
                        _ => (None, None),
                    };
                    let a = w.act(say, hand);
                    senses = w.step(&a);
                }
            }
            select(&w.events)
        };
        let mean = |kind: u8| (0..12u64).map(|s| run(kind, 700 + s)).sum::<i32>() / 12;
        let (holder, reactor, two) = (mean(0), mean(1), mean(2));
        eprintln!("held three: holder {holder}, reactor {reactor}, two-task selector {two}");
        assert!(holder > 600 && reactor < holder - 300 && two < holder - 150, "{holder} {reactor} {two}");
    }

    #[test]
    fn repeat_scores_learning_which_turn_pays_from_the_stroke_alone() {
        // a maze with no stem, the food on one side all lifetime and no cue: policies that feel
        // only the affect channel. A learner turns the other way after a miss and keeps a turn
        // that paid; a stubborn one always turns left; a random one turns either way
        let run = |kind: u8, seed: u64| -> (i32, i32) {
            let mut w = World::new(seed, 4);
            (w.maze_len, w.maze_fixed, w.maze_nocue, w.gap_max) = (0, true, true, 2);
            w.maze(100);
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let (mut senses, mut turn, mut pending) = (BitVector::new(SENSE_BITS, Some(0)), LEFT, false);
            while !w.done() {
                // a stroke after a turn: it paid, keep it; a new trial with none: turn the other way
                let at_junction = channel(&senses, EYE).count_ones() > 0;
                if channel(&senses, AFFECT).count_ones() > 0 { pending = false }
                if at_junction && pending && kind == 0 {
                    turn = if turn == LEFT { RIGHT } else { LEFT };
                    pending = false;
                }
                let hand = if !at_junction { None } else {
                    match kind { 0 => Some(turn), 1 => Some(LEFT), _ => Some([LEFT, RIGHT][rand::Rng::gen_range(&mut rng, 0..2)]) }
                };
                if hand.is_some() { pending = true }
                let a = w.act(None, hand);
                senses = w.step(&a);
            }
            (repeat(&w.events), maze(&w.events))
        };
        let mean = |kind: u8| { let v: Vec<(i32, i32)> = (0..12u64).map(|s| run(kind, 300 + s)).collect(); (v.iter().map(|x| x.0).sum::<i32>() / 12, v.iter().map(|x| x.1).sum::<i32>() / 12) };
        let (learner, stubborn, random) = (mean(0), mean(1), mean(2));
        assert!(learner.0 > 500 && learner.1 > 500, "learner {learner:?}");
        assert!(stubborn.0.abs() < 80 && random.0.abs() < 120, "stubborn {stubborn:?}, random {random:?}");
    }

    #[test]
    fn win_stay_scores_a_learner_and_not_a_stubborn_clicker() {
        // policies that see only the eye and the affect channel: a learner clicks a light (on its
        // second step, inside the window) unless its last click went unstroked; a stubborn one
        // always clicks the first light it saw. With no swaps and with swaps every 50 trials, on
        // 12 worlds each
        let run = |learner: bool, period: usize, seed: u64| -> i32 {
            let mut w = World::new(seed, 4);
            (w.gap_max, w.reversal_period) = (6, period);
            w.conditioning(Conditioning::Reversal, 200);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            let (mut lights, mut first, mut clicked, mut before): (Vec<(Vec<u64>, Option<bool>)>, Option<Vec<u64>>, Option<usize>, bool) = (vec![], None, None, false);
            while !w.done() {
                let eye = channel(&senses, EYE).as_words().to_vec();
                let shown = eye.iter().any(|&x| x != 0);
                if channel(&senses, AFFECT).count_ones() > 0 {
                    if let Some(k) = clicked.take() { lights[k].1 = Some(true) }
                }
                if shown && !before {
                    if let Some(k) = clicked.take() { lights[k].1 = Some(false) }
                }
                let second = shown && before;
                before = shown;
                let mut hand = None;
                if second && clicked.is_none() {
                    let k = lights.iter().position(|l| l.0 == eye).unwrap_or_else(|| { lights.push((eye.clone(), None)); lights.len() - 1 });
                    first.get_or_insert(eye.clone());
                    if if learner { lights[k].1 != Some(false) } else { first.as_ref() == Some(&eye) } {
                        hand = Some(CLICK);
                        clicked = Some(k);
                    }
                }
                let a = w.act(None, hand);
                senses = w.step(&a);
            }
            probes(&w.events, period, &[], &[], &[], &[], &[])[5]
        };
        for period in [200, 50] {
            let mean = |learner: bool| (0..12u64).map(|s| run(learner, period, 100 + s)).sum::<i32>() / 12;
            let (l, s) = (mean(true), mean(false));
            assert!(l > 500 && s.abs() < 80, "period {period}: learner {l}, stubborn {s}");
        }
    }

    #[test]
    fn fair_win_stay_cancels_a_clicker_driven_by_any_recent_stroke() {
        // a learner clicks a light unless its last click on that light went unstroked; a recency
        // clicker clicks any light within `k` steps of a stroke (else one trial in ten); 24
        // worlds, no swaps and swaps every 50 trials
        let run = |learner: bool, k: usize, period: usize, seed: u64| -> (i32, i32) {
            let mut w = World::new(seed, 4);
            (w.gap_max, w.reversal_period) = (6, period);
            w.conditioning(Conditioning::Reversal, 200);
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let mut senses = BitVector::new(SENSE_BITS, Some(0));
            let (mut lights, mut clicked, mut before, mut since): (Vec<(Vec<u64>, Option<bool>)>, Option<usize>, bool, usize) = (vec![], None, false, 999);
            let mut coin = false;
            while !w.done() {
                let eye = channel(&senses, EYE).as_words().to_vec();
                let shown = eye.iter().any(|&x| x != 0);
                if channel(&senses, AFFECT).count_ones() > 0 {
                    since = 0;
                    if let Some(c) = clicked.take() { lights[c].1 = Some(true) }
                } else {
                    since = since.saturating_add(1);
                }
                if shown && !before {
                    if let Some(c) = clicked.take() { lights[c].1 = Some(false) }
                    coin = rand::Rng::gen_ratio(&mut rng, 1, 10);
                }
                let second = shown && before;
                before = shown;
                let mut hand = None;
                if second && clicked.is_none() {
                    let c = lights.iter().position(|l| l.0 == eye).unwrap_or_else(|| { lights.push((eye.clone(), None)); lights.len() - 1 });
                    if if learner { lights[c].1 != Some(false) } else { since <= k || coin } {
                        hand = Some(CLICK);
                        clicked = Some(c);
                    }
                }
                let a = w.act(None, hand);
                senses = w.step(&a);
            }
            let tr = trials(&w.events, (3, 5), (7, 9));
            (win_stay_of(&tr, false), win_stay_of(&tr, true))
        };
        for period in [200, 50] {
            let mean = |learner: bool, k: usize| {
                let v: Vec<(i32, i32)> = (0..24u64).map(|s| run(learner, k, period, 300 + s)).collect();
                (v.iter().map(|x| x.0).sum::<i32>() / 24, v.iter().map(|x| x.1).sum::<i32>() / 24)
            };
            let (learn_old, learn_fair) = mean(true, 0);
            let (rec_old, rec_fair) = mean(false, 16);
            eprintln!("period {period}: learner old {learn_old} fair {learn_fair}; recency clicker old {rec_old} fair {rec_fair}");
            assert!(learn_fair > 300, "period {period}: learner {learn_fair}");
            assert!(rec_fair < 80 && rec_fair < learn_fair - 300, "period {period}: recency {rec_fair} (old {rec_old})");
        }
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
            (probes(&w.events, 50, &d.events, &[], &[], &[], &[])[0], w.events.len())
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
        // (echoing earns no copy: it never says an untaught token at its own lag, so it scores at
        // or below 0)
        assert!(echoes < 100, "{echoes}");
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
        // a blind speaker with two tokens and the teacher on every one: near and far lags have
        // the same chances, so it scores about 0 (the old count gave it about 80)
        let blind2: i32 = (0..16u64).map(|seed| {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let toks: Vec<usize> = (0..200).map(|_| rand::Rng::gen_range(&mut rng, 0..2)).collect();
            let mut w = World::new(seed, 2);
            w.teacher = true;
            w.teacher_rate = 8;
            w.sequence(&toks);
            while !w.done() {
                let a = w.act(Some(rand::Rng::gen_range(&mut rng, 0..2)), None);
                w.step(&a);
            }
            echo(&w.events)
        }).sum::<i32>() / 16;
        assert!(blind2.abs() < 40, "{blind2}");
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
