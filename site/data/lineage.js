/* the lineage track: what has fed the substrate, and what has come out of it.
   hand-maintained against each source card and docs/LOG.md -- the corpora
   are named only where the card reads `binding: bound`, and every checkpoint link
   is one that scripts/export_releases.py wrote into releases.js.

   `done` is what has run. everything after `now` is unchecked: planned, not claimed. */
window.IBM_LINEAGE = {
  now: 8,          /* index of the last completed node; everything past it is future */
  nodes: [
    { kind: "corpus", label: "film + audio",
      note: "Next-frame and next-sample continuation over film. The primer the substrate was first trained through.",
      detail: "Row 16 of the corrections ledger belongs to this corpus: video→EEG transfer of 91.5% was withdrawn because pairing each frame with a RANDOM frame from the same film still transferred 83–86%. Only ~5 of the 35.5 points depend on prediction; the rest is gradient flow through the dynamics.",
      href: "#corpus" },

    { kind: "model", label: "av_predictor",
      note: "19 checkpoints, 68.9M parameters, best at step 19,000.",
      detail: "obj-av. The audio-video continuation arm. Its own video prediction was measured at −0.25 skill against persistence — it matched appearance rather than predicting.",
      href: "https://huggingface.co/jacob-valdez/ibm-1/tree/main/checkpoints" },

    { kind: "corpus", label: "LibriBrain MEG",
      note: "Deep within-person MEG during audiobook listening. card: binding bound.",
      detail: "The corpus behind the MEG encoder. Ledger row 11: an assumed-constant clock offset was really 4,300–5,300 ppm of drift, ±3.4 s across a chapter, which smeared a 1–8 Hz effect across 3–27 cycles and averaged it away. Resampling onto the fitted line took three windows from p=0.171/0.463/0.902 to p=0.024.",
      href: "#corpus" },

    { kind: "model", label: "meg_encoder",
      note: "36 checkpoints, 26.4M parameters, best at step 22,000.",
      detail: "obj-meg. The most-iterated model in the programme.",
      href: "https://huggingface.co/jacob-valdez/ibm-1/tree/main/checkpoints" },

    { kind: "corpus", label: "THINGS-EEG2",
      note: "Large image/EEG pairing over the THINGS concept set. card: binding bound.",
      detail: "Ledger row 8: the image order was rebuilt by walking a directory and matched on 10 of 16,540 pairs, while the count check passed because the counts were equal. Three hours of confident negative results followed. image_metadata.npy had the answer.",
      href: "#corpus" },

    { kind: "model", label: "visual_evoked",
      note: "2 checkpoints, 7.2M parameters.",
      detail: "obj-evoked. The visual evoked-response arm.",
      href: "https://huggingface.co/jacob-valdez/ibm-1/tree/main/checkpoints" },

    { kind: "model", label: "eeg_to_image",
      note: "2 checkpoints, 10.3M parameters. 63.5% top-1 retrieval, 127× chance.",
      detail: "obj-contrastive. Retrieval, not generation: nearest bank entries ranked by the group-mean measured evoked EEG of the image, on the designated test set.",
      href: "https://huggingface.co/jacob-valdez/ibm-1/tree/main/checkpoints" },

    { kind: "model", label: "av_meg_predictor",
      note: "9 checkpoints, 93.7M parameters. The first arm to carry two objectives at once.",
      detail: "obj-av+meg. One schedule over both likelihood terms — the first materialization trained on more than one stream.",
      href: "https://huggingface.co/jacob-valdez/ibm-1/tree/main/checkpoints" },

    { kind: "note", label: "four controls, one answer",
      note: "Permuted kernels match trained ones on every pathway tested so far.",
      detail: "Ablation, disjoint-region retrieval, the motor task and the proprioceptive readout each compared a trained kernel against one with its site rows permuted. The largest difference is −0.0065. The dynamics are load-bearing as a filter; what they LEARNED has not yet been shown to carry anything. Everything below is an attempt to change that.",
      href: "#programme" },

    /* ---------------- everything past here is future, not claimed ---------------- */

    { kind: "corpus", label: "CMU mocap",
      note: "2,600+ trials, 144 subjects, 120 Hz, raw markers. Fetching.",
      detail: "Chosen because this body has two motions and neither is admissible: all 67 stored trajectories are outside the model's own declared joint ranges, and crawl-best's left knee is past its limit for 95.6% of 1,600 frames. Recorded human motion is admissible by construction, so it sidesteps the parameter search rather than solving it.",
      href: "http://mocap.cs.cmu.edu/" },

    { kind: "corpus", label: "proprioceptive corpus",
      note: "markers → inverse kinematics → muscle path lengths → spindle channels.",
      detail: "Gated before any byte was fetched: a ridge must beat zero, mean and persistence before any cortex claim; every trial must respect the declared joint ranges; and every motor arm carries a permuted-kernel control in the same run. Golgi-tendon force is deferred and tactile is excluded — nothing consumes the skin contact bundle yet.",
      href: "#corpus" },

    { kind: "model", label: "multi-objective materializations",
      note: "One substrate, many objectives at once, initialised from the arms above.",
      detail: "Sensory→cortex→motor, sensory→loop→sensory prediction, long motor rollout, and the existing prediction and retrieval terms, trained together. Each objective carries its own permuted arm, or a positive result is not diagnosable. Any checkpoint joining the initialisation must first reproduce its own recorded number under current code — ledger row 25 is a factor of 112 for skipping that.",
      href: "#programme" }
  ]
};
