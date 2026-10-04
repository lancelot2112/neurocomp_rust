//! A thalamus-like relay: content-addressed routing of earlier cortical states.
//!
//! The relay keeps a short history of the frames a cortical node held. Each relay
//! *channel* is a fixed routing rule:
//! - a **query**: the frame `query_lag` steps back from now (0 = the current frame);
//! - a **key match**: the most recent earlier frame that overlaps the query by at
//!   least `match_fraction` of the query's active bits (content addressing);
//! - a **value**: the frame `value_offset` steps after that match, relayed to the
//!   channel's output.
//!
//! This is hard attention with one fixed query/key/value pattern per channel, the
//! operation an induction head performs ("find where this happened before and copy
//! what followed"). Biologically it plays the role of higher-order thalamic relay
//! between cortical areas (pulvinar / mediodorsal), with the limited number of
//! channels standing in for reticular-nucleus gating. Which channels are worth
//! keeping is left to the caller (e.g. ablation credit, see research/experiments/08).

use crate::bitvec::BitVector;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelayChannel {
    pub query_lag: usize,
    pub value_offset: usize,
}

pub struct Thalamus {
    bits: usize,
    capacity: usize,
    history: Vec<BitVector>, // oldest first
    pub channels: Vec<RelayChannel>,
    pub match_fraction: f32,
}

impl Thalamus {
    pub fn new(bits: usize, capacity: usize, channels: Vec<RelayChannel>) -> Self {
        Self { bits, capacity, history: Vec::new(), channels, match_fraction: 0.8 }
    }

    /// Record the cortical frame for this tick (call once per tick, before `relay`).
    pub fn observe(&mut self, frame: &BitVector) {
        self.history.push(frame.clone());
        if self.history.len() > self.capacity {
            self.history.remove(0);
        }
    }

    /// Forget everything observed so far (e.g. between independent episodes).
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// The value frame relayed by one channel now, if its query found a match.
    pub fn relay_channel(&self, c: RelayChannel) -> Option<&BitVector> {
        let n = self.history.len();
        let q_pos = n.checked_sub(1 + c.query_lag)?;
        let query = &self.history[q_pos];
        let q_bits = query.count_ones();
        if q_bits == 0 {
            return None;
        }
        let need = (q_bits as f32 * self.match_fraction).ceil() as u32;
        // most recent earlier frame matching the query whose value is already in the past
        (0..q_pos)
            .rev()
            .filter(|&p| p + c.value_offset < n)
            .find(|&p| overlap(query, &self.history[p]) >= need)
            .map(|p| &self.history[p + c.value_offset])
    }

    /// Hindsight route proposal: of the `candidates`, the routes that would
    /// relay `target` right now (overlap >= `match_fraction` of its bits). Call
    /// it when the cortex was surprised by `target`, before observing it, to
    /// find which routes would have carried the missing information.
    pub fn routes_that_would_relay(&self, target: &BitVector, candidates: &[RelayChannel]) -> Vec<RelayChannel> {
        let need = (target.count_ones() as f32 * self.match_fraction).ceil() as u32;
        if need == 0 {
            return Vec::new();
        }
        candidates
            .iter()
            .copied()
            .filter(|&c| self.relay_channel(c).map_or(false, |v| overlap(v, target) >= need))
            .collect()
    }

    /// Open-ended route discovery: on a surprise by `target` (call before
    /// observing it), find routes from the data instead of a fixed menu. For each
    /// earlier frame matching `target`, look at the frames up to `max_value_offset`
    /// before it (candidate keys); wherever one of them matches a frame in the
    /// current context, up to `max_query_lag` back, that pair defines a route
    /// (query lag, value offset). Routes are kept only if the relay, with its
    /// "most recent match" rule, would actually deliver `target` now.
    pub fn discover_routes(&self, target: &BitVector, max_query_lag: usize, max_value_offset: usize) -> Vec<RelayChannel> {
        let n = self.history.len();
        let need = |a: &BitVector| (a.count_ones() as f32 * self.match_fraction).ceil() as u32;
        let t_need = need(target);
        if t_need == 0 || n == 0 {
            return Vec::new();
        }
        let mut found: Vec<RelayChannel> = Vec::new();
        for p in 0..n {
            if overlap(&self.history[p], target) < t_need {
                continue;
            }
            for v in 1..=max_value_offset.min(p) {
                let key = &self.history[p - v];
                let k_need = need(key);
                if k_need == 0 {
                    continue;
                }
                for q in 0..=max_query_lag {
                    let Some(q_pos) = n.checked_sub(1 + q) else { break };
                    if q_pos <= p - v {
                        break; // the query must come after the key it matches
                    }
                    let c = RelayChannel { query_lag: q, value_offset: v };
                    if !found.contains(&c) && overlap(&self.history[q_pos], key) >= k_need {
                        found.push(c);
                    }
                }
            }
        }
        found.retain(|&c| self.relay_channel(c).map_or(false, |v| overlap(v, target) >= t_need));
        found
    }

    /// All channels' relayed frames concatenated, channel 0 first; an empty
    /// frame where a channel found nothing.
    pub fn relay(&self) -> BitVector {
        let mut words = Vec::with_capacity(self.channels.len() * self.bits.div_ceil(64));
        for &c in &self.channels {
            match self.relay_channel(c) {
                Some(v) => words.extend_from_slice(v.as_words()),
                None => words.extend(std::iter::repeat(0).take(self.bits.div_ceil(64))),
            }
        }
        BitVector::from_words(words)
    }
}

fn overlap(a: &BitVector, b: &BitVector) -> u32 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(i: usize) -> BitVector {
        BitVector::from_bits(&[i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3], 64)
    }

    #[test]
    fn induction_channel_copies_what_followed_the_earlier_match() {
        // mary(1) went(2) to(3) the(4) kitchen(5) .(6) where(7) is(8) mary(1) ?(9)
        let ch = RelayChannel { query_lag: 1, value_offset: 4 };
        let mut th = Thalamus::new(64, 32, vec![ch]);
        for s in [1, 2, 3, 4, 5, 6, 7, 8, 1, 9] {
            th.observe(&sym(s));
        }
        // at "?": query = "mary" one step back; earlier "mary" + 4 = "kitchen"
        assert_eq!(th.relay_channel(ch).unwrap().as_words(), sym(5).as_words());
        assert_eq!(th.relay().as_words(), sym(5).as_words());
    }

    #[test]
    fn no_match_relays_an_empty_frame() {
        let ch = RelayChannel { query_lag: 0, value_offset: 1 };
        let mut th = Thalamus::new(64, 8, vec![ch]);
        for s in [1, 2, 3] {
            th.observe(&sym(s));
        }
        assert!(th.relay_channel(ch).is_none());
        assert_eq!(th.relay().count_ones(), 0);
    }

    #[test]
    fn most_recent_match_wins_and_history_is_bounded() {
        let ch = RelayChannel { query_lag: 0, value_offset: 1 };
        let mut th = Thalamus::new(64, 5, vec![ch]);
        // 1 2 | 1 3 | 1  -> most recent earlier "1" was followed by 3
        for s in [1, 2, 1, 3, 1] {
            th.observe(&sym(s));
        }
        assert_eq!(th.relay_channel(ch).unwrap().as_words(), sym(3).as_words());
        th.observe(&sym(7));
        th.observe(&sym(7)); // capacity 5: the first "1 2" pair is gone
        assert_eq!(th.history.len(), 5);
    }

    #[test]
    fn hindsight_finds_the_route_that_would_have_relayed_the_answer() {
        let mut th = Thalamus::new(64, 32, vec![]);
        for s in [1, 2, 3, 4, 5, 6, 7, 8, 1, 9] {
            th.observe(&sym(s));
        }
        let all: Vec<RelayChannel> = (0..=2)
            .flat_map(|q| (1..=6).map(move |v| RelayChannel { query_lag: q, value_offset: v }))
            .collect();
        // the answer after "?" is "kitchen" (5)
        let routes = th.routes_that_would_relay(&sym(5), &all);
        assert_eq!(routes, vec![RelayChannel { query_lag: 1, value_offset: 4 }]);
    }

    #[test]
    fn discovery_finds_routes_outside_any_menu() {
        // mary(1) went(2) all(3) the(4) way(5) over(6) to(7) the(4) kitchen(9) .(10)
        // where(11) is(12) mary(1) right(13) now(14) ?(15)  -> answer kitchen
        let mut th = Thalamus::new(64, 40, vec![]);
        for s in [1, 2, 3, 4, 5, 6, 7, 4, 9, 10, 11, 12, 1, 13, 14, 15] {
            th.observe(&sym(s));
        }
        let routes = th.discover_routes(&sym(9), 8, 12);
        // query "mary" 3 back, kitchen 8 after the earlier "mary"
        assert!(routes.contains(&RelayChannel { query_lag: 3, value_offset: 8 }));
        // and every proposed route really relays the answer
        for r in &routes {
            assert_eq!(th.relay_channel(*r).unwrap().as_words(), sym(9).as_words());
        }
    }
}