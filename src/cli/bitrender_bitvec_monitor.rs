use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{self, stdout, Write};
use std::time::{Duration, Instant};

use crate::bitvec::BitVector;
use crate::cli::bitrender_bitvec_router::{render_bitvector_to, BitOrder, RenderConfig};

/// Configuration for monitoring/redrawing a BitVector over time.
#[derive(Clone, Debug)]
pub struct MonitorConfig {
    /// How often to poll and attempt a redraw.
    pub interval: Duration,
    /// Stop after at most this many frames (None = infinite).
    pub max_frames: Option<usize>,
    /// Stop after this duration (None = infinite).
    pub max_duration: Option<Duration>,
    /// If false, only redraw when content changes. If true, redraw every tick.
    pub redraw_on_no_change: bool,
    /// Optional title printed above the render.
    pub title: Option<String>,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_millis(100),
            max_frames: None,
            max_duration: None,
            redraw_on_no_change: false,
            title: Some("BitVector Monitor".to_string()),
        }
    }
}

/// Monitors a BitVector-producing closure and re-renders the CLI view when it changes.
/// Uses ANSI escape codes to clear the screen and update in-place.
/// Returns when max_frames or max_duration (if provided) is reached.
pub fn monitor_bitvector<F>(
    mut make_bv: F,
    render: &RenderConfig,
    monitor: &MonitorConfig,
) -> io::Result<()>
where
    F: FnMut() -> BitVector,
{
    let mut out = stdout();
    // Enter alternate screen and hide cursor
    write!(out, "\x1b[?1049h\x1b[?25l")?;
    out.flush()?;

    let start = Instant::now();
    let mut prev_hash: Option<u64> = None;
    let mut frames = 0usize;
    let mut changes = 0usize;
    let mut last_change = Instant::now();

    let spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let mut spin_idx = 0usize;

    let deadline = monitor.max_duration.map(|d| start + d);

    let res = loop {
        // Produce the current BitVector snapshot
        let bv = make_bv();

        // Hash the content to detect change (by words)
        let mut hasher = DefaultHasher::new();
        // Include bit length and both word orders to be robust to iter order choices.
        bv.len().hash(&mut hasher);
        for w in bv.as_lswords() {
            w.hash(&mut hasher);
        }
        let h = hasher.finish();

        let changed = prev_hash.map_or(true, |ph| ph != h);
        prev_hash = Some(h);

        if changed || monitor.redraw_on_no_change {
            // Clear screen and home cursor
            write!(out, "\x1b[2J\x1b[H")?;

            if let Some(title) = &monitor.title {
                writeln!(out, "{} ({} order)", title, match render.bit_order {
                    BitOrder::LeastSignificantFirst => "LSB",
                    BitOrder::MostSignificantFirst => "MSB",
                })?;
                writeln!(out)?;
            }

            // Render the bitvector using the chosen renderer config
            render_bitvector_to(&bv, render, &mut out)?;

            if changed {
                changes += 1;
                last_change = Instant::now();
            }
        }

        // Status line (always update)
        let elapsed = Instant::now().duration_since(start);
        let since_change = Instant::now().duration_since(last_change);
        let sp = spinner[spin_idx % spinner.len()];
        spin_idx = (spin_idx + 1) % spinner.len();

        writeln!(
            out,
            "\n{} frames: {} | changes: {} | elapsed: {:.1?} | since change: {:.1?}",
            sp, frames, changes, elapsed, since_change
        )?;
        out.flush()?;

        frames += 1;

        // Termination conditions
        if monitor
            .max_frames
            .map_or(false, |max| frames >= max)
        {
            break Ok(());
        }
        if let Some(dl) = deadline {
            if Instant::now() >= dl {
                break Ok(());
            }
        }

        // Sleep until next frame
        std::thread::sleep(monitor.interval);
    };

    // Leave alternate screen and show cursor
    write!(out, "\x1b[?25h\x1b[?1049l")?;
    out.flush()?;

    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;

    // NOTE: This is an integration-like test that runs briefly and renders to a TTY.
    // Marked as ignored to avoid CI noise. Run with:
    // cargo test monitor_demo -- --ignored --nocapture
    #[test]
    #[ignore]
    fn monitor_demo() {
        // Simulated evolving BitVector in a background thread.
        let state = Arc::new(Mutex::new(BitVector::from_words(vec![0; 4])));
        let st_clone = Arc::clone(&state);

        let producer = thread::spawn(move || {
            // Flip a few bits over time
            for i in 0..200 {
                let mut bv = st_clone.lock().unwrap().clone();
                // Toggle a bit based on time
                let bit_idx = (i % (bv.len().max(1))) as usize;
                bv.toggle(bit_idx); // assume BitVector has toggle(idx); otherwise implement your update here
                *st_clone.lock().unwrap() = bv;
                thread::sleep(Duration::from_millis(50));
            }
        });

        let make_bv = {
            let state = Arc::clone(&state);
            move || state.lock().unwrap().clone()
        };

        let render_cfg = RenderConfig {
            // Try other modes: HexDump, Braille, Rainbow, Grayscale, etc.
            mode: crate::cli::bitrender_bitvec_router::RenderMode::Rainbow,
            cols: 64,
            cell_bits: 8,
            bit_order: BitOrder::LeastSignificantFirst,
        };

        let mon_cfg = MonitorConfig {
            interval: Duration::from_millis(100),
            max_frames: Some(100),
            max_duration: Some(Duration::from_secs(10)),
            redraw_on_no_change: false,
            title: Some("Monitoring neurocomp BitVector".to_string()),
        };

        // Run the monitor loop (renders to stdout alternate screen)
        let _ = monitor_bitvector(make_bv, &render_cfg, &mon_cfg);

        let _ = producer.join();
    }
}