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
- **Sukhbaatar, S., Szlam, A., Weston, J. & Fergus, R. (2015).** *End-to-end memory
  networks.* [arXiv:1503.08895](https://arxiv.org/abs/1503.08895). Learned multi-hop
  attention over stored sentences, on bAbI; the learned version of our
  [13](experiments/13-big-loop.md) big loop.
- **Graves, A. (2016).** *Adaptive computation time for recurrent neural networks.*
  [arXiv:1603.08983](https://arxiv.org/abs/1603.08983). Learning how many steps to take
  (halting), cf. learned hop counts.

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

## Hippocampus and entorhinal cortex
- **Marr, D. (1971).** *Simple memory: a theory for archicortex.* Phil. Trans. R. Soc. B 262.
  Sparse expansion and recurrent completion as the basis of episodic memory.
- **Treves, A. & Rolls, E. (1994).** *Computational analysis of the role of the
  hippocampus in memory.* Hippocampus 4(3). Dentate-gyrus pattern separation, CA3
  autoassociative completion, capacity.
- **O'Reilly, R. & McClelland, J. (1994).** *Hippocampal conjunctive encoding, storage,
  and recall: avoiding a trade-off.* Hippocampus 4(6).
- **Yassa, M. & Stark, C. (2011).** *Pattern separation in the hippocampus.* Trends in
  Neurosciences 34(10).
- **O'Keefe, J. & Dostrovsky, J. (1971).** *The hippocampus as a spatial map.* Brain
  Research 34. Place cells.
- **Hafting, T. et al. (2005).** *Microstructure of a spatial map in the entorhinal
  cortex.* Nature 436. Grid cells.
- **Whittington, J. et al. (2020).** *The Tolman-Eichenbaum Machine: unifying space and
  relational memory through generalization in the hippocampal formation.* Cell 183. Structure
  (EC) vs content (sensory), bound in hippocampus.
- **Whittington, J., Warren, J. & Behrens, T. (2022).** *Relating transformers to models and
  neural representations of the hippocampal formation.* ICLR.
  [arXiv:2112.04035](https://arxiv.org/abs/2112.04035). Transformers with position encodings
  ≈ TEM; the bridge between [concepts/hippocampal-functions](concepts/hippocampal-functions.md)
  and attention.
- **Stachenfeld, K., Botvinick, M. & Gershman, S. (2017).** *The hippocampus as a predictive
  map.* Nature Neuroscience 20. Successor representations: graph codes from prediction.
- **Teyler, T. & DiScenna, P. (1986).** *The hippocampal memory indexing theory.* Behavioral
  Neuroscience 100.
- **McClelland, J., McNaughton, B. & O'Reilly, R. (1995).** *Why there are complementary
  learning systems in the hippocampus and neocortex.* Psychological Review 102. Fast
  episodic store plus slow cortical learning via replay.
- **Hasselmo, M. (2005).** *What is the function of hippocampal theta rhythm?* Hippocampus
  15. Encoding vs retrieval phases.
- **Lisman, J. & Jensen, O. (2013).** *The theta-gamma neural code.* Neuron 77. Items
  held at different gamma sub-cycles within a theta cycle: multiplexing instead of blending.
- **Frady, E. P., Kent, S., Olshausen, B. & Sommer, F. (2020).** *Resonator networks.*
  Neural Computation 32. Iteratively factoring superposed vector-symbolic bindings.
- **Zacks, J. et al. (2007).** *Event perception: a mind-brain perspective.* Psychological
  Bulletin 133. Event boundaries at prediction errors.
- **Mattar, M. & Daw, N. (2018).** *Prioritized memory access explains planning and
  hippocampal replay.* Nature Neuroscience 21. Replay favours memories whose update is most
  useful; cf. question-tagged replay in [17](experiments/17-consolidation.md).
- **Lisman, J. & Grace, A. (2005).** *The hippocampal-VTA loop: controlling the entry of
  information into long-term memory.* Neuron 46. CA1/subiculum novelty → dopamine →
  encoding: novelty as a computed mismatch, not a frequency count.
- **O'Mara, S. (2005).** *The subiculum: what it does, what it might do, and what
  neuroanatomy has yet to tell us.* Journal of Anatomy 207. The main hippocampal output,
  incl. via mammillary bodies to the anterior thalamus.

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
- **Doya, K. (2000).** *Complementary roles of basal ganglia and cerebellum in learning
  and motor control.* Current Opinion in Neurobiology 10. Cortex unsupervised,
  cerebellum supervised, basal ganglia reinforcement. See
  [basal ganglia and cerebellum](concepts/basal-ganglia-and-cerebellum.md).
- **O'Reilly, R. & Frank, M. (2006).** *Making working memory work: a computational model
  of learning in the prefrontal cortex and basal ganglia.* Neural Computation 18. PBWM:
  basal ganglia learn when to gate information into working memory.
  Used in [18](experiments/18-prefrontal-working-memory.md).
- **Schultz, W., Dayan, P. & Montague, P. R. (1997).** *A neural substrate of prediction
  and reward.* Science 275. Dopamine neurons signal reward-prediction error (outcome
  better or worse than expected), not reward. The advantage credit of
  [19](experiments/19-l5-shared-reward.md).
- **Marr, D. (1969).** *A theory of cerebellar cortex.* J. Physiology 202; **Albus, J.
  (1971).** *A theory of cerebellar function.* Mathematical Biosciences 10; **Ito, M.
  (2001).** *Cerebellar long-term depression.* Physiological Reviews 81. Granule
  expansion + climbing-fibre error = a per-unit supervised learner.
- **Bell, C., Han, V., Sugawara, Y. & Grant, K. (1997).** *Synaptic plasticity in a
  cerebellum-like structure depends on temporal order.* Nature 387. Learned cancellation
  of predictable input: store/transmit only what was not predicted.
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
- **Amit, D. & Fusi, S. (1994).** *Learning in neural networks with material synapses.*
  Neural Computation 6. Binary synapses with stochastic transitions; memory as a
  palimpsest. **Fusi, S., Drew, P. & Abbott, L. (2005).** *Cascade models of synaptically
  stored memories.* Neuron 45.
- **Gaines, B. (1969).** *Stochastic computing systems.* Advances in Information Systems
  Science 2; **Alaghi, A. & Hayes, J. (2013).** *Survey of stochastic computing.* ACM TECS 12.
  Values as densities of ones in bit streams; AND multiplies. See
  [probability in bits](concepts/probability-in-bits.md).

## Data
- Project Gutenberg texts via the NLTK data repository (*Alice*, Bryant's stories), and the
  Brown corpus (Francis & Kučera, 1979) with tags, via NLTK. Fetched by
  [`scripts/fetch_corpora.sh`](../scripts/fetch_corpora.sh).
