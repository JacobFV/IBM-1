/* the lineage highway.  tributaries join a trunk, checkpoints leave it as exits.
   hand-maintained against each source card and docs/LOG.md: a corpus is named only
   where its card reads `binding: bound`, and every checkpoint figure is read from
   releases.js (as export_releases.py wrote it) by lineage.js, not typed here.

   width is CHECKPOINTS RELEASED, which is the one honest common unit here -- a
   corpus enters at the width of what it has produced, and the trunk is the sum.
   `done` strands are solid; everything past `now` is drawn open, because it is
   planned rather than run. */
window.IBM_LINEAGE = {
  now: 4,                       /* the last stage that has actually run */
  stages: [
    { y: 0.00, label: "primer" },
    { y: 0.17, label: "paired MEG" },
    { y: 0.41, label: "evoked + retrieval" },
    { y: 0.58, label: "two objectives at once" },
    { y: 0.72, label: "four controls" },
    { y: 0.85, label: "motion capture" },
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
      detail: "Chosen because this body has two motions and neither is admissible: all 67 stored trajectories are outside the model's own declared joint ranges, and crawl-best's left knee is past its limit for 95.6% of 1,600 frames. Recorded human motion is admissible by construction. The fetch gate flagged 112 subjects and 2,514 trials against the card's pre-fetch claim of 144 and 2,600+; the archive turned out to be intact — all 2,740 members pass CRC and the 31 absent subject IDs are scattered through the range rather than forming a truncated tail. 144 was the highest subject ID mistaken for a count. The card now carries the measured figures.",
      href: "http://mocap.cs.cmu.edu/" }
  ],

  /* an exit leaves the trunk at a stage: a released checkpoint, or a control.
     a checkpoint names its models and nothing else -- its width, its 🤗 links and
     their sizes are read from releases.js at render time. */
  exits: [
    { at: 1, side: -1, kind: "model", label: "audio-visual predictor", models: ["av_predictor"] },

    { at: 2, side: 1, kind: "model", label: "speech to MEG encoder", models: ["meg_encoder"] },

    { at: 3, side: -1, kind: "model", label: "visual evoked + EEG to image", models: ["visual_evoked", "eeg_to_image"] },

    { at: 3, side: 1, kind: "model", label: "audio-visual + MEG predictor", models: ["av_meg_predictor"] },

    { at: 4, side: -1, kind: "control", label: "four permuted-kernel controls", w: 0,
      note: "largest difference −0.0065",
      detail: "Ablation, disjoint-region retrieval, the motor task and the proprioceptive readout each compared a trained kernel against one with its site rows permuted. The dynamics are load-bearing as a filter; what they LEARNED has not been shown to carry anything. Everything past this point is an attempt to change that.",
      href: "#programme" },

    /* planned: nothing released yet, so it has no link and keeps a hand-set width */
    { at: 6, side: 1, kind: "model", label: "multi-objective materializations", models: [], w: 12, future: true }
  ]
};
