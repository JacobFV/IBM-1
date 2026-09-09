# what goes into a body learning to be itself

Pain is one component. It is not the list. This is the list — generously, so that
no future agent has to rediscover that the thing they are building sits beside
thirty others, and so that "we have nociception now" is never mistaken for "we
have the drive system".

Each entry says what it is, why a developing body needs it, and **what exists
here versus what does not**. Status is one of:

- **built** — a component computes it and is registered
- **partial** — some of it exists, named gap
- **declared** — named in the ontology, nothing computes it
- **absent** — not represented at all

The honest summary before the detail: of the ~40 components below, a handful are
built, rather more are partial or declared, and most are absent. That is the
expected shape at this stage and the point of writing it down is to keep it
visible rather than flattering.

---

## 1. Homeostatic drives — the signals that make some states worse than others

Without these a body has no reason to prefer any state to any other, and
reinforcement learning has nothing to ground a cost in.

| component | why it is needed | status |
|---|---|---|
| **nociception** | damage signal; the cost term for protective learning. High threshold, A-delta and C, sensitising | **built** — `nociceptor_polymodal`, gated |
| **hunger / satiety** | drives seeking; the first thing an infant acts on | **partial** — gastric distension, nutrient load, hepatoportal glucose transduced; no appetite state integrating them |
| **thirst** | osmolality and volume depletion | **absent** — the fluid model exists, the drive does not |
| **fatigue / exertion** | limits effort, forces rest, shapes gait economy | **partial** — `endurance_h` derived from substrate against demand; not a drive |
| **thermal comfort / discomfort** | drives shelter, posture, huddling | **partial** — warm and cold thermoreceptors built; comfort as a valued state is not |
| **air hunger (dyspnea)** | the most urgent drive there is | **partial** — pulmonary stretch and chemoreception transduced; no dyspnea state |
| **bladder / bowel urgency** | continence is learned, and it is learned from this | **partial** — renal and bladder channels exist; urgency does not |
| **nausea** | protects against ingested harm; a distinct affect | **absent** |
| **itch (pruriception)** | a separate channel from pain with its own fibres and its own behaviour (scratch, not withdraw) | **absent** |
| **sleep pressure / circadian phase** | consolidation, and the developmental schedule itself | **partial** — the slow oscillation is fitted to real sleep; no homeostat |

## 2. Affect and reward — what makes learning go somewhere

| component | why | status |
|---|---|---|
| **pleasure / appetitive reward** | the counterweight to pain; without it a body only avoids | **absent** |
| **curiosity / information gain** | drives exploration when nothing hurts and nothing is wanted | **absent** — named as a target in `DIRECTION.md` |
| **surprise / prediction error** | the substrate of self-supervised learning, and a reward in its own right | **partial** — predictive losses exist per modality; no unified error signal |
| **effort cost** | why a body finds an economical gait rather than any gait | **partial** — metabolic power is computed per muscle; not used as a cost |
| **fear / threat appraisal** | fast defensive responses that precede understanding | **absent** |
| **soothing / C-tactile affective touch** | a real, separate afferent class; the basis of comfort and of social contact | **absent** — CT afferents are not declared |
| **social affect / attachment** | the frame most human learning happens inside | **absent** |
| **disgust** | contamination avoidance | **absent** |

## 3. Self-supervised prediction — learning without a teacher

| component | why | status |
|---|---|---|
| **forward model / efference copy** | distinguishes self-caused from world-caused sensation. Arguably the single most important thing on this page: without it, every movement is an unexplained sensory event | **absent** |
| **sensory attenuation** | why you cannot tickle yourself; falls out of a working forward model | **absent** |
| **cross-modal prediction** | vision predicting touch, sound predicting sight | **partial** — audio-visual objective exists and is video-dominated |
| **proprioceptive prediction** | predicting limb state before feedback arrives, which is what makes fast movement possible given conduction delay | **absent** |
| **temporal prediction of the world** | what happens next, independent of what things look like now | **measured absent** — the video term learns appearance matching, not prediction |
| **inter-modal binding** | that this sight and this sound are one event | **absent** |

## 4. Motor development — how control is bootstrapped

| component | why | status |
|---|---|---|
| **spinal reflex arcs** | the scaffold everything else is built on | **built** — stretch, reciprocal, autogenic, Renshaw; 192 of 214 muscles under arcs |
| **spontaneous activity / motor babbling** | how a body discovers what its muscles do. Prenatal and essential | **absent** |
| **primitive reflexes** | Moro, palmar grasp, rooting, stepping, ATNR — developmental scaffolds that appear, do work, and are inhibited | **absent** |
| **postural control / righting** | prerequisite for everything upright | **partial** — a stance controller holds against a push; not learned |
| **supervised imitation of trajectories** | the current bootstrap route | **built** — 68 motions, 17,622 frames |
| **reaching and grasping** | the first goal-directed act | **absent** |
| **sequence and rhythm generation (CPGs)** | locomotion is rhythmic and the rhythm is not planned centrally | **absent** |
| **muscle synergies** | dimensionality reduction the body actually uses | **absent** |

## 5. Body schema — knowing what you are

| component | why | status |
|---|---|---|
| **somatotopic map formation** | learned, not given; the map is shaped by use | **partial** — cortical regions assigned geometrically, not learned |
| **body ownership / self-recognition** | the sense that this limb is mine | **absent** |
| **peripersonal space** | the region where the world is actionable | **absent** |
| **tool incorporation** | extension of the schema | **absent** |
| **interoceptive self-model** | the body as felt from inside | **partial** — 15 visceral channels reach cortex; the cortex adds nothing to them yet |

## 6. Social and communicative

| component | why | status |
|---|---|---|
| **face preference and detection** | present from birth, orients everything social | **absent** |
| **gaze following / joint attention** | the channel most word learning arrives through | **absent** |
| **imitation** | how a body learns acts it has never performed | **absent** |
| **vocal turn-taking and babbling** | speech is motor before it is linguistic | **absent** — speech out is not started |
| **attachment** | the regulatory relationship learning happens inside | **absent** |

## 7. Regulatory and attentional

| component | why | status |
|---|---|---|
| **arousal / vigilance** | gates whether learning happens at all | **partial** — thalamo-cortical loop oscillates; not a controller |
| **selective attention** | what gets learned from | **absent** |
| **autonomic regulation** | the body's own control loop | **partial** — vagal and splanchnic afference declared; efferent limb absent (nothing drives `b_preganglionic` / `c_postganglionic`) |
| **memory consolidation / replay** | what makes experience durable | **absent** — declared as blocked in the curriculum |

---

## How to use this list

**Do not build these in the order they are written.** They are grouped by kind,
not by priority. The ordering that matters is what unblocks what:

1. **Efference copy** is the highest-leverage absent item. Prediction, sensory
   attenuation, body ownership and agency all sit on it, and none of them can be
   built first.
2. **A unified prediction-error signal** turns the per-modality losses that
   already exist into a learning signal the whole body can share.
3. **Spontaneous motor activity** is how a body with 214 innervated muscles finds
   out what they do. It is cheap and it is the natural companion to the supervised
   corpus.
4. **Pleasure** — because a body with only nociception can learn to avoid and
   cannot learn to seek, and half of development is seeking.

**Keep the status column honest.** The temptation is to move a row to *built*
when a field is declared or a channel is routed. A component is built when
something computes it and a case whose answer is known prints that answer. The
nociceptor row says *built* because it prints exactly zero at 5 N of firm touch
and 51.2 Hz at a 40 N crush.

**And record what is deliberately deferred.** Several of these are absent by
choice at this stage rather than by oversight — social components in particular
are downstream of a body that can act at all. Deferred is a status; unrecorded is
a bug.
