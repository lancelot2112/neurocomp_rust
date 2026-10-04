# Related work and reading list

Grouped by topic, with notes on how each connects to this project. Links go to
arXiv or publisher pages where they are stable. Otherwise the citation is enough to find it.

## Sequence memory
- **Cleary, J. & Witten, I. (1984).** *Data compression using adaptive coding and
  partial string matching.* IEEE Trans. Communications 32(4). PPM: predict from the
  longest context seen before, back off when it's new. This is what
  [surprise-driven growth](concepts/surprise-driven-growth.md) converges to, and the
  "back-off n-gram" baseline in [03](experiments/03-book-scale-char-prediction.md).
- **Hawkins, J. & Ahmad, S. (2016).** *Why neurons have thousands of synapses, a theory
  of sequence memory in neocortex.* Frontiers in Neural Circuits.
  [arXiv:1511.00083](https://arxiv.org/abs/1511.00083). HTM temporal memory: grow
  dendritic segments sampled from previously active cells when input is unpredicted.
  The closest neural relative of our growth rule.
- **Cui, Y., Ahmad, S. & Hawkins, J. (2016).** *Continuous online sequence learning with
  an unsupervised neural network model.* Neural Computation.
  [arXiv:1512.05463](https://arxiv.org/abs/1512.05463). Online (prequential)
  evaluation of HTM on sequence prediction, the same protocol we use.

## Word segmentation
- **Saffran, J., Aslin, R. & Newport, E. (1996).** *Statistical learning by 8-month-old
  infants.* Science 274. Infants segment speech using transitional probabilities.
  This is our baseline in [04](experiments/04-word-segmentation.md).
- **Elman, J. (1990).** *Finding structure in time.* Cognitive Science 14(2). A simple
  recurrent network's prediction error peaks at word boundaries, the same signal as
  our `target_probability()` dips.
- **Brent, M. (1999).** *An efficient, probabilistically sound algorithm for
  segmentation and word discovery.* Machine Learning 34.
- **Goldwater, S., Griffiths, T. & Johnson, M. (2009).** *A Bayesian framework for word
  segmentation: exploring the effects of context.* Cognition 112. A strong baseline
  for unsupervised segmentation with lexicon feedback.

## Distributional syntax and semantics
- **Harris, Z. (1954).** *Distributional structure.* Word 10. Words in similar contexts
  have similar functions and meanings, the premise of [05](experiments/05-syntax.md)
  and [06A](experiments/06-meaning.md).
- **Brown, P. et al. (1992).** *Class-based n-gram models of natural language.*
  Computational Linguistics 18(4). Brown clustering.
- **Schütze, H. (1995).** *Distributional part-of-speech tagging.* EACL. POS induction
  from left/right context vectors, i.e. our count-vector reference.
- **Christodoulopoulos, C., Goldwater, S. & Steedman, M. (2010).** *Two decades of
  unsupervised POS induction: how far have we come?* EMNLP. A survey with standard metrics
  (many-to-one, V-measure) we could adopt instead of nearest-neighbor agreement.
- **Sahlgren, M. (2005).** *An introduction to random indexing.* Random sparse context
  vectors summed into word vectors. Our sparse-code Hebbian masks are a sampled version.
- **Levy, O. & Goldberg, Y. (2014).** *Neural word embedding as implicit matrix
  factorization.* NeurIPS. word2vec ≈ factorized PMI; why count-based references are strong.

## Sparse distributed representations
- **Kanerva, P. (1988).** *Sparse Distributed Memory.* MIT Press.
- **Kanerva, P. (2009).** *Hyperdimensional computing: an introduction to computing in
  distributed representation with high-dimensional random vectors.* Cognitive
  Computation 1. Binding by XOR/permutation, a candidate for
  [variable binding](concepts/variable-binding.md).
- **Plate, T. (1995).** *Holographic reduced representations.* IEEE Trans. Neural Networks 6(3).
- **Ahmad, S. & Hawkins, J. (2016).** *How do neurons operate on sparse distributed
  representations? A mathematical theory of sparsity, neurons and active dendrites.*
  [arXiv:1601.00720](https://arxiv.org/abs/1601.00720). Collision and matching-error
  math behind [sparse codes and collisions](concepts/sparse-codes-and-collisions.md).
- **Olshausen, B. & Field, D. (1996).** *Emergence of simple-cell receptive field
  properties by learning a sparse code for natural images.* Nature 381.

## Binding and attention
- **Vaswani, A. et al. (2017).** *Attention is all you need.*
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
- **Olsson, C. et al. (2022).** *In-context learning and induction heads.* Transformer
  Circuits Thread. Induction heads copy what followed an earlier match, the
  mechanism our network lacks ([06B](experiments/06-meaning.md)).
- **Ramsauer, H. et al. (2020).** *Hopfield networks is all you need.*
  [arXiv:2008.02217](https://arxiv.org/abs/2008.02217). Attention = modern Hopfield
  associative retrieval.
- **Ba, J. et al. (2016).** *Using fast weights to attend to the recent past.*
  [arXiv:1610.06258](https://arxiv.org/abs/1610.06258). Fast, temporary associative
  memory as an alternative to attention.
- **Smolensky, P. (1990).** *Tensor product variable binding and the representation of
  symbolic structures in connectionist systems.* Artificial Intelligence 46.
- **Weston, J. et al. (2015).** *Towards AI-complete question answering: a set of
  prerequisite toy tasks.* [arXiv:1502.05698](https://arxiv.org/abs/1502.05698). bAbI.
  Our [06B](experiments/06-meaning.md) is a variant of task 1 with held-out pairs.
- **Lake, B. & Baroni, M. (2018).** *Generalization without systematicity.*
  [arXiv:1711.00350](https://arxiv.org/abs/1711.00350). Held-out combinations as the
  test of compositional generalization, the same design principle as our held-out pairs.

## Top-down feedback and predictive coding
- **McClelland, J. & Rumelhart, D. (1981).** *An interactive activation model of context
  effects in letter perception: Part 1.* Psychological Review 88(5). Word-level units
  feed back to letter units (the word-superiority effect). This is the architecture
  tested in [07](experiments/07-top-down-bias.md).
- **Rao, R. & Ballard, D. (1999).** *Predictive coding in the visual cortex: a functional
  interpretation of some extra-classical receptive-field effects.* Nature Neuroscience
  2(1). Higher levels send predictions down and lower levels send residual errors up.
- **Friston, K. (2005).** *A theory of cortical responses.* Phil. Trans. R. Soc. B 360.
  Precision-weighted prediction errors, i.e. weighting top-down by its reliability
  ([top-down bias](concepts/top-down-bias.md)).
- **Hawkins, J., Ahmad, S. & Cui, Y. (2017).** *A theory of how columns in the neocortex
  enable learning the structure of the world.* Frontiers in Neural Circuits. HTM's use
  of apical/feedback context to disambiguate input.

## Thalamus and attention
- **Crick, F. (1984).** *Function of the thalamic reticular complex: the searchlight
  hypothesis.* PNAS 81. The reticular nucleus as an attentional gate on thalamic relay.
  The limited, competing relay channels in [09](experiments/09-thalamic-attention.md).
- **Sherman, S. M. & Guillery, R. W. (2006).** *Exploring the Thalamus and Its Role in
  Cortical Function* (2nd ed.). MIT Press. Higher-order thalamic nuclei relay
  cortex → thalamus → cortex "driver" signals: the relay path in `Thalamus`.
- **Saalmann, Y. et al. (2012).** *The pulvinar regulates information transmission
  between cortical areas based on attention demands.* Science 337. Attention-dependent
  routing between cortical areas through the thalamus.
- **Wimmer, R. et al. (2015).** *Thalamic control of sensory selection in divided
  attention.* Nature 526. Prefrontal control of the reticular nucleus gating sensory relay.
- **Schmitt, L. et al. (2017).** *Thalamic amplification of cortical connectivity
  sustains attentional control.* Nature 545. Mediodorsal thalamus sustaining
  task-relevant cortical representations.
- **Halassa, M. & Kastner, S. (2017).** *Thalamic functions in distributed cognitive
  control.* Nature Neuroscience 20. A review of the thalamus as a controller of
  cortical communication.

## Credit assignment
- **Lillicrap, T., Santoro, A., Marris, L., Akerman, C. & Hinton, G. (2020).**
  *Backpropagation and the brain.* Nature Reviews Neuroscience 21. A survey of how
  brains might assign credit across layers.
- **Lillicrap, T., Cownden, D., Tweed, D. & Akerman, C. (2016).** *Random synaptic
  feedback weights support error backpropagation for deep learning.* Nature
  Communications 7. Feedback alignment.
- **Bengio, Y. (2014).** *How auto-encoders could provide credit assignment in deep
  networks via target propagation.* [arXiv:1407.7906](https://arxiv.org/abs/1407.7906).
- **Frémaux, N. & Gerstner, W. (2016).** *Neuromodulated spike-timing-dependent
  plasticity, and theory of three-factor learning rules.* Frontiers in Neural
  Circuits 9. The "three-factor" policy in [08](experiments/08-credit-assignment.md).
- **Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D. & Brea, J. (2018).**
  *Eligibility traces and plasticity on behavioral time scales.* Frontiers in Neural
  Circuits 12. Bridging the delay between a cause and its reward.
- **Williams, R. (1992).** *Simple statistical gradient-following algorithms for
  connectionist reinforcement learning.* Machine Learning 8. REINFORCE: reward minus
  baseline times eligibility.
- **Wolpert, D. & Tumer, K. (2002).** *Optimal payoff functions for members of
  collectives.* Advances in Complex Systems 4. "Difference rewards": credit each agent
  with the system's reward minus the reward had it been absent. This is the
  counterfactual behind ablation credit in [08](experiments/08-credit-assignment.md).
- **Foerster, J. et al. (2018).** *Counterfactual multi-agent policy gradients.* AAAI.
  [arXiv:1705.08926](https://arxiv.org/abs/1705.08926). The same idea with a learned
  counterfactual baseline.
- **Hochreiter, S. & Schmidhuber, J. (1997).** *Long short-term memory.* Neural
  Computation 9(8). Gated memory cells that learn what to hold across long gaps, which
  is the job of the memory units in [08](experiments/08-credit-assignment.md), learned
  there by backprop through time.
- **Bengio, Y., Simard, P. & Frasconi, P. (1994).** *Learning long-term dependencies
  with gradient descent is difficult.* IEEE Trans. Neural Networks 5(2).

## Learning rules
- **Hebb, D. (1949).** *The Organization of Behavior.* Wiley.
- **Turrigiano, G. (2008).** *The self-tuning neuron: synaptic scaling of excitatory
  synapses.* Cell 135. Homeostatic plasticity, the idea behind
  `KernelClass::with_target_active`.

## Data
- Project Gutenberg texts via the NLTK data repository (*Alice*, Bryant's stories), and the
  Brown corpus (Francis & Kučera, 1979) with tags, via NLTK. Fetched by
  [`scripts/fetch_corpora.sh`](../scripts/fetch_corpora.sh).
