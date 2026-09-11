/* the lineage highway.  tributaries join a trunk, checkpoints leave it as exits.
   hand-maintained against each source card and docs/LOG.md: a corpus is named only
   where its card reads `binding: bound`, and every checkpoint figure comes from
   releases.js as export_releases.py wrote it.

   width is CHECKPOINTS RELEASED, which is the one honest common unit here -- a
   corpus enters at the width of what it has produced, and the trunk is the sum.
   `done` strands are solid; everything past `now` is drawn open, because it is
   planned rather than run. */
window.IBM_LINEAGE = {
  now: 4,                       /* the last stage that has actually run */
  stages: [
    { y: 0.00, label: "primer" },
    { y: 0.17, label: "paired MEG" },
    { y: 0.34, label: "evoked + retrieval" },
    { y: 0.52, label: "two objectives at once" },
    { y: 0.66, label: "four controls" },
    { y: 0.82, label: "motion capture" },
    { y: 1.00, label: "many objectives" }
  ],

  /* a strand runs from stage `from` to stage `to`, entering from a side. */
  strands: [
    { id: "av", label: "film + audio", kind: "corpus", w: 19, from: 0, to: 6, side: -1,
      note: "next-frame and next-sample continuation over public-domain film",
      detail: "The primer the substrate was first trained through. Ledger row 16 belongs to it: a 91.5% video→EEG transfer was withdrawn because pairing each frame with a RANDOM frame from the same film still transferred 83–86%. Only ~5 of the 35.5 points depend on prediction; the rest is gradient flow through the dynamics.",
      href: "#corpus" },

    { id: "meg", label: "LibriBrain MEG", kind: "corpus", w: 36, from: 1, to: 6, side: 1,
      note: "deep within-person MEG during audiobook listening · binding bound",
      detail: "Ledger row 11: an assumed-constant clock offset was really 4,300–5,300 ppm of drift, ±3.4 s across a chapter, which smeared a 1–8 Hz effect across 3–27 cycles. Resampling onto the fitted line took three windows from p=0.171/0.463/0.902 to p=0.024. v1 and v2 are void; v3 is the rebuild everything since has used.",
      href: "#corpus" },

    { id: "eeg", label: "THINGS-EEG2", kind: "corpus", w: 4, from: 2, to: 6, side: -1,
      note: "image/EEG pairing over the THINGS concept set · binding bound",
      detail: "Ledger row 8: the image order was rebuilt by walking a directory and matched on 10 of 16,540 pairs, while the count check passed because the counts were equal. Three hours of confident negative results followed. image_metadata.npy had the answer.",
      href: "#corpus" },

    { id: "mocap", label: "CMU mocap", kind: "corpus", w: 8, from: 5, to: 6, side: 1,
      future: true,
      note: "2,514 trials, 112 subjects, 120 Hz, raw markers · fetched",
      detail: "Chosen because this body has two motions and neither is admissible: all 67 stored trajectories are outside the model's own declared joint ranges, and crawl-best's left knee is past its limit for 95.6% of 1,600 frames. Recorded human motion is admissible by construction. The archive's 112 subjects and 2,514 trials do NOT match the card's pre-fetch claim of 144 and 2,600+, and that mismatch is unresolved.",
      href: "http://mocap.cs.cmu.edu/" }
  ],

  /* an exit leaves the trunk at a stage: a released model, or a control. */
  exits: [
    { at: 1, side: -1, kind: "model", label: "av_predictor", w: 19,
      note: "19 checkpoints · 68.9M · obj-av",
      detail: "The audio-video continuation arm. Its own video prediction measured −0.25 skill against persistence: it matched appearance rather than predicting.",
      href: "https://huggingface.co/jacob-valdez/ibm-1/tree/main/checkpoints" },

    { at: 2, side: 1, kind: "model", label: "meg_encoder", w: 36,
      note: "36 checkpoints · 26.4M · obj-meg",
      detail: "The most-iterated model in the programme, and the one whose head carries the low-rank lead field. Its docstring records why: a free readout let the paired head reach skill +0.94 with an effective cortical rank of 1.03 — the readout doing the work and the cortex a scalar.",
      href: "https://huggingface.co/jacob-valdez/ibm-1/tree/main/checkpoints" },

    { at: 3, side: -1, kind: "model", label: "visual_evoked · eeg_to_image", w: 4,
      note: "2 + 2 checkpoints · 63.5% top-1, 127× chance",
      detail: "Retrieval, not generation: nearest bank entries ranked by the group-mean measured evoked EEG of the image, on the corpus's designated test set.",
      href: "https://huggingface.co/jacob-valdez/ibm-1/tree/main/checkpoints" },

    { at: 3, side: 1, kind: "model", label: "av_meg_predictor", w: 9,
      note: "9 checkpoints · 93.7M · obj-av+meg",
      detail: "The first arm to carry two objectives at once — one schedule over both likelihood terms.",
      href: "https://huggingface.co/jacob-valdez/ibm-1/tree/main/checkpoints" },

    { at: 4, side: -1, kind: "control", label: "four permuted-kernel controls", w: 0,
      note: "largest difference −0.0065",
      detail: "Ablation, disjoint-region retrieval, the motor task and the proprioceptive readout each compared a trained kernel against one with its site rows permuted. The dynamics are load-bearing as a filter; what they LEARNED has not been shown to carry anything. Everything past this point is an attempt to change that.",
      href: "#programme" },

    { at: 6, side: 1, kind: "model", label: "multi-objective materializations", w: 12,
      future: true,
      note: "planned · first attempt stopped at step 3,675",
      detail: "One substrate, many objectives at once. The first run collapsed into the failure PairedNeuralLoop's own docstring names: effective cortical rank fell to 1.01 within 1,000 steps and stayed there. Stopped rather than run to 20,000 — rightly, but the reason given at the time was wrong in the opposite direction. Against the correct per-batch zero baseline the MEG head is WORSE than emitting nothing, and a lead-rank sweep at 64, 16 and 4 then refuted the readout as the cause: the rank collapses at every setting. A later check found the trainer had no train/test split at all, so those figures are in-sample; it now reserves a held-out tail.",
      href: "#programme" }
  ]
};
