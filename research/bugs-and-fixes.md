# Bugs found and fixed

| Found in | Bug | Effect | Fix (commit) |
|---|---|---|---|
| [01](experiments/01-stock-network-reading.md) | `Stay()`/`Pop()` self-loops used `read_back = history_depth`, but the ring held `history_depth` frames, so the read wrapped onto the frame being written | the "previous frame" loop read the same tick | read `history_depth − 1`; runtime rings sized from what edges read; `get_frame` asserts the frame exists (`e89b76d`) |
| [01](experiments/01-stock-network-reading.md) | Inhibition was set per **output** word but checked at the kernel's **input** word index | unrelated kernels inhibited each other | check the kernel's own output words; inhibit vector sized by the output (`e89b76d`) |
| [01](experiments/01-stock-network-reading.md) | Kernels visited in list order, so the first kernel always won inhibition | order bias, not best match | visit by excitation, strongest first (`e89b76d`) |
| [01](experiments/01-stock-network-reading.md) | `KernelGroup::default()` always adds a kernel writing output bit 0 | stray bit in every node | `KernelGroup::new()` for empty groups (`e89b76d`) |
| [03](experiments/03-book-scale-char-prediction.md) | Grown-kernel threshold = 80% of *all* sampled bits | deep kernels fired with a whole frame wrong (−4 points) | tolerance of one frame's noise (`ac60b9e`) |
| [04](experiments/04-word-segmentation.md) | (design hazard) recognized-word feedback into segmentation | snowballed into letter-by-letter cuts | only words of 3+ letters can trigger cuts (`ac60b9e`) |
