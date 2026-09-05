# datasets

## what a dataset is

a dataset is not $p(y\mid x)$. that framing presumes a designated direction, and most of
these sources do not have one — a simultaneous EEG-fMRI recording constrains
electromagnetic state and haemodynamic state through two different processes, and
neither conditions the other.

it is closer to $p(x,y)$, but not quite that either. a dataset is a finite set of
**jointly realized values of ibm state variables**, and those variables are not
epistemically alike. some were *imposed* by the experimenter and some were *measured*:

$$
\text{dataset}
=
\left\{
\text{realizations of }
\left(
\underbrace{x_{\text{imposed}}}_{\text{intervention}},\
\underbrace{x_{\text{measured}}}_{\text{evidence}}
\right)
\right\}
$$

both are ordinary ibm state (§6). the difference is what they are permitted to do: an
imposed variable is conditioned on, a measured one contributes a likelihood term.

$$
p(\theta\mid D)
\propto
p(\theta)
\prod_d
p\!\left(x_{\text{measured}}^{(d)} \;\middle|\; x_{\text{imposed}}^{(d)},\theta\right)
$$

the distinction is load-bearing. a stimulus was chosen by an experimenter, not drawn
from the brain's joint distribution; treating a designed stimulus as jointly sampled
lets a model learn the experimental design as though it were biology. that is the most
common way a corpus this heterogeneous goes wrong.

every stream of every source below therefore declares which of the two it is, and which
observation process connects it to brain state.

## bandwidth is part of what a dataset is

a source's sampling rate determines which temporal laplacian components it can
constrain at all (§1). an fMRI run contributes precision below $0.25\ \mathrm{Hz}$ and
none above it; a spike-band recording contributes nothing below its high-pass. this is
not a preprocessing detail — it is the reason heterogeneous sources compose without
conflict, and it is recorded per stream.

## inventory

**601 source cards** in `~/Documents/win/registry/cards`, plus 38 proposals pending:

| kind | count |
|---|---|
| dataset | 251 |
| atlas | 136 |
| tool | 125 |
| model | 46 |
| standard | 30 |
| derived_geometry | 13 |

the active training configuration in `sc-wbd`
(`integrated-whole-brain-modeling-across-modalities-scales-and-dynamics/configs/source_cards`)
currently binds nine: `anatomical_prior`, `ds000113_real`, `ds002336_real`,
`eegmmidb_real`, `montage_calibration`, `negative_control_shuffled`, `sim_wholebrain`,
`sleepedf_real`, `tribe_v2_teacher`.

## roles

a role is not a description of what a source *is*. it is a statement of what it is
permitted to do to the posterior.

| role | permitted to |
|---|---|
| observation | contribute a likelihood term about measured state |
| boundary | fix an exogenous input — a stimulus, a forcing, an intervention |
| prior | shape $p(\theta)$ before individual evidence |
| calibration | fix an instrument, geometry or coordinate frame, never the biology |
| oracle | supply synthetic ground truth for a surrogate fit |
| teacher | supply distillation targets, never an observation likelihood |
| evaluation | score a materialization; never updates $\theta$ |
| negative_control | must produce a null; if it does not, the pipeline is wrong |

## access status

| status | meaning |
|---|---|
| `repo-held` | bytes acquired and audited |
| `p0` / `p1` | prioritised for acquisition |
| `restricted` | requires a data use agreement or application |
| `watch` | announced, not yet released, or terms unresolved |
| `ref` | reference only, never a training source |

## 1. datasets, by the state they constrain

251 sources. the grouping is by which ibm state a source's observation process
terminates on, not by acquisition modality — a simultaneous EEG-fMRI study constrains
electromagnetic *and* haemodynamic state through two different processes and is grouped
by that fact rather than by either instrument.

### scalp electromagnetic (31)

| source | streams | role | status | n |
|---|---|---|---|---|
| `armeni-audiobook-meg` | meg, audio_stimulus, eyetracking, pupillometry, t1w | observation/evaluation | restricted | 3 |
| `bbbd` | eeg_64ch, eog, ecg, respiration, pupil | observation/boundary/calibration | repo-held | 178 |
| `brantley-full-body-mobi` | eeg_60ch, eog, emg_bilateral_lower_limb, imu_full_body_17_sensors | observation/boundary | p1 | 10 |
| `broderick-natural-speech-eeg` | eeg_128ch_natural_speech, eeg_reverse_speech, eeg_cocktail_party, eeg_speech_in_noise, eeg_n400 | observation/evaluation | p0 | 19 |
| `chbmp` | eeg_rest_four_conditions, t1w_mprage, dwi_12dir, field_maps, mmse | observation/prior | restricted | 282 |
| `chennu-propofol-sedation-eeg` | eeg_preprocessed, sedation_condition_labels | observation | p0 | 20 |
| `deap` | eeg_32ch, peripheral_physiology, music_video_stimuli, affect_ratings, frontal_face_video_subset | observation/boundary | p1 | 32 |
| `ds001971` | eeg_108ch, emg_bilateral_tibialis, heel_pressure, goniometry, auditory_cue_events | observation/boundary | p0 | 20 |
| `ds004256` | eeg, spatialised_audio_stimuli, elevation_conditions, localisation_responses | observation/boundary/calibration | p1 | — |
| `eegmmidb` | eeg_64ch, task_annotations, baseline_runs | observation/boundary/evaluation | repo-held | 109 |
| `erp-core` | eeg_raw, eeg_bids, event_codes, behavioural_responses, eeglab_erplab_pipelines | observation/calibration/evaluation | p0 | 40 |
| `hcp-meg` | meg_rest, meg_task, individual_anatomy, derived_source_estimates, connectivity_products | observation/prior | p1 | 95 |
| `kymata-soto` | meg, eeg, transform_expression_maps, latency_maps, stimulus_audio | observation/prior | watch | — |
| `libribrain` | meg, audio_stimuli, transcripts, alignments, held_out_splits | observation/boundary/evaluation | repo-held | 1 |
| `mahnob-hci` | eeg_32ch, peripheral_physiology, eye_tracking, multi_camera_video, audio | observation/boundary | p1 | 27 |
| `mass` | polysomnography, high_density_eeg, expert_sleep_staging, spindle_annotations, k_complex_annotations | observation/evaluation | p1 | 200 |
| `meg-masc` | meg_208ch, audio_stimuli, word_and_phoneme_alignments, story_transcripts, head_position | observation/boundary | repo-held | 27 |
| `meg-scans` | meg, graded_audio_conditions, transcripts | observation/boundary | p0 | — |
| `mne-sample` | meg, eeg, t1w, freesurfer_reconstruction, bem_surfaces | calibration/evaluation | repo-held | 1 |
| `mne-somato` | meg, t1w, freesurfer_reconstruction, stimulation_events | calibration/evaluation | repo-held | 1 |
| `mne-spm-face` | meg_ctf, t1w, stimulus_events | calibration/evaluation | repo-held | 1 |
| `omega` | meg_rest, t1w, eog, ecg, demographics | observation/prior | restricted | 644 |
| `openbmi-lee2019` | eeg, emg, task_labels_motor_imagery, task_labels_erp, task_labels_ssvep | observation/evaluation | p1 | 54 |
| `physionet` | eeg_databases, psg_databases, ecg_databases, waveform_databases, annotation_sets | observation/prior | p1 | — |
| `sleep-edfx` | eeg_2ch, eog, emg_chin, respiration, body_temperature | observation/boundary/evaluation | repo-held | 78 |
| `studyforrest-meg-ds003633` | meg, audio_movie_stimulus, head_position | observation/boundary | p0 | — |
| `tdbrain` | eeg_rest_eyes_open, eeg_rest_eyes_closed, events, task_responses, clinical_data | observation/prior | restricted | 1274 |
| `things-eeg` | eeg_64ch, stimulus_sequence, behaviour | observation/boundary | p1 | 50 |
| `things-eeg2` | eeg_64ch, stimulus_images, behaviour | observation/boundary/evaluation | p0 | 10 |
| `tuh-eeg` | clinical_eeg, physician_reports, abnormality_labels, seizure_annotations, artifact_annotations | observation/prior/negative_control | restricted | 14000 |
| `zuco` | eeg_128ch, eye_tracking, fixation_related_potentials, sentence_materials, relation_annotations | observation/boundary | p0 | 30 |

### intracranial electromagnetic (22)

| source | streams | role | status | n |
|---|---|---|---|---|
| `allen-visual-coding-neuropixels` | spikes, lfp, running_speed, eye_and_pupil_tracking, optotagging | observation/prior | p0 | 58 |
| `aneux` | angiography, vessel_meshes, morphological_descriptors | observation/calibration | watch | — |
| `bci-competition-datasets` | eeg, ecog, emg, eog, task_labels | observation/evaluation | p1 | — |
| `brain-treebank` | ieeg, electrode_localisation, word_timings, transcripts, dependency_parses | observation/boundary | p0 | 10 |
| `chang-lab-ecog-cv-syllables` | ecog_256_channel, syllable_annotations, rest_baseline | observation | p1 | 3 |
| `clinical-ieeg-archives` | continuous_ieeg, seizure_annotations, electrode_localisation, clinical_metadata | observation/prior | restricted | — |
| `ds004194` | ecog, electrode_localisation, visual_stimuli, t1w, retinotopic_estimates | observation/boundary | p1 | — |
| `ds005574` | ecog_raw, ecog_high_gamma_derived, audio_stimulus, word_aligned_transcript, acoustic_features | observation/evaluation | p0 | 9 |
| `ds006234` | ieeg, electrode_localisation, auditory_stimuli, spoken_responses, response_timing | observation/boundary | p1 | — |
| `epilepsy-ecosystem` | ieeg_continuous_ambulatory, seizure_annotations, wearable_biosignals | observation/evaluation | restricted | 3 |
| `ibl-brain-wide-map` | spikes, raw_ephys, video_three_cameras, pose_tracking, wheel_position | observation/prior/evaluation | p0 | 139 |
| `intra` | complete_vessel_models, annotated_vessel_segments, generated_negative_samples | observation/calibration/evaluation | p1 | 103 |
| `invasive-human-array-bci` | threshold_crossings, sorted_units, lfp, attempted_movement_labels, decoder_outputs | observation/boundary | restricted | — |
| `kai-miller-ecog-library` | ecog, electrode_positions, task_events, analysis_scripts | observation/evaluation | p0 | 34 |
| `neurotycho` | ecog, simultaneous_eeg_and_ecog, behaviour, anaesthesia_state, sleep | observation/prior | p1 | — |
| `ram-intracranial` | ieeg, electrode_localisation, direct_stimulation_records, memory_task_behaviour, t1w | observation/boundary | restricted | 250 |
| `single-word-production-dutch-ieeg` | seeg, synchronised_audio, word_annotations, electrode_localisation | observation/boundary | p0 | 10 |
| `steinmetz-neuropixels-2019` | spikes, lfp_2500hz, behavioural_video, pupil_tracking, wheel_position | observation/prior | p0 | 10 |
| `swec-ethz-ieeg` | ieeg_continuous, seizure_annotations | observation/evaluation | p1 | 68 |
| `topbrain` | mra_volumes, cta_volumes, vessel_class_annotations, arterial_topology_labels | observation/calibration | p0 | — |
| `willett-handwriting-bci` | intracortical_threshold_crossings, spike_band_power, character_cues, sentence_labels | observation/boundary | p0 | 1 |
| `willett-speech-neuroprosthesis` | intracortical_threshold_crossings, spike_band_power, sentence_cues, diagnostic_blocks, tuning_tasks | observation/boundary | p0 | 1 |

### simultaneous multimodal (23)

| source | streams | role | status | n |
|---|---|---|---|---|
| `cam-can` | meg_rest, meg_task, eeg, t1w, t2w | observation/prior | restricted | 700 |
| `cinebrain` | eeg, bold, audiovisual_stimulus | observation/boundary | watch | — |
| `ds000116` | eeg, bold, auditory_stimuli, visual_stimuli, button_responses | observation/boundary/calibration | p1 | 17 |
| `ds000117` | meg_306ch, eeg_70ch, bold_task, t1w, dwi | observation/boundary/calibration | repo-held | 16 |
| `ds002336` | eeg_64ch, bold, neurofeedback_signal, t1w, task_events | observation/boundary | repo-held | 10 |
| `ds002338` | eeg_raw, eeg_gradient_corrected, bold, t1w, neurofeedback_scores | observation | p0 | — |
| `ds002725` | eeg, bold, music_stimuli, affect_ratings, t1w | observation/boundary | p1 | 21 |
| `ds003688` | ieeg, electrode_localisation, bold_movie, t1w, film_stimulus | observation/boundary | repo-held | 63 |
| `ds003768` | eeg, bold_fmri, sleep_staging | observation | p1 | 33 |
| `ds004024` | eeg, fmri, dwi, t1w, coil_pose | observation/calibration | repo-held | 13 |
| `ds004514` | eeg, fnirs, imagery_cues, stimulus_images, optode_montage | observation/boundary | repo-held | 12 |
| `ds006033` | eeg, bold, inner_speech_cues, t1w, behaviour | observation/boundary | p1 | — |
| `ds006110` | multi_echo_bold, structural_mri, dwi, eeg, behavioural_battery | observation | p0 | — |
| `ds007554` | eeg, fnirs, ecg, emg, task_events | observation/boundary | p0 | — |
| `hansen-many-networks` | structural_connectivity, functional_connectivity, meg_connectivity, pet_metabolic, gene_expression | prior/observation | repo-held | — |
| `hbcd` | t1w, t2w, dwi, bold_rest, eeg | observation/prior | restricted | — |
| `hcp-young-adult` | t1w, t2w, dwi, bold_rest, bold_task | observation/prior/calibration | p0 | 1200 |
| `healthy-brain-network` | eeg_128ch, eye_tracking, t1w, dwi, bold_rest | observation/prior | restricted | 10000 |
| `mous` | meg, bold, t1w, dwi, audio_stimuli | observation/boundary | p1 | 204 |
| `mpi-leipzig-mind-brain-body` | qt1_mp2rage, t1w, t2w, flair, dwi | observation/prior | p0 | — |
| `nm000254-natview` | eeg_64ch, bold_fmri, t1w, eye_tracking, ecg | observation/calibration | p0 | 22 |
| `single-neuron-ieeg-fmri-movie` | single_units, lfp, ieeg_macro, bold_movie, eye_tracking | observation/boundary | p0 | — |
| `wand-cubric` | dwi_ultrastrong_gradient, qmt, mcdespot, axcaliber, charmed | observation/calibration | p0 | 170 |

### hemodynamic and metabolic (52)

| source | streams | role | status | n |
|---|---|---|---|---|
| `abcd` | t1w, t2w, dwi, bold_rest, bold_task | observation/prior | restricted | 11800 |
| `abide` | bold_rest, t1w, dwi_subset, phenotype, diagnostic_instruments | observation/prior/evaluation | p1 | — |
| `adhd-200` | bold_rest, t1w, diagnostic_status, dimensional_symptom_measures, age | observation/prior/evaluation | p1 | 776 |
| `adni` | t1w, t2flair, dwi, asl_perfusion, bold_rest | observation/prior/negative_control | restricted | 2000 |
| `algonauts-2025` | bold_movie_responses, stimulus_video, stimulus_audio, transcripts, held_out_test_stimuli | evaluation/boundary | p0 | 4 |
| `aomic` | t1w, dwi, bold_rest, bold_task, fieldmaps | observation/prior | p0 | 442 |
| `aomic-id1000` | bold_task, cardiac, demographics, dwi, psychometrics | observation/prior | repo-held | 928 |
| `baby-connectome-project` | t1w, t2w, bold_rest, dwi, behavioural_assessment | observation/prior | restricted | — |
| `bold5000` | bold_images, localisers, t1w, dwi, behaviour_ratings | observation/boundary | p1 | 4 |
| `cneuromod-friends` | bold_movie, audio, video, physiology, transcripts | observation/boundary/evaluation | p0 | 6 |
| `cneuromod-things` | bold_images, behaviour, eye_tracking, anatomical | observation/boundary | p1 | 4 |
| `corr-reliability` | bold_rest, t1w, dwi, asl_cbf | observation/calibration/evaluation | p0 | 1629 |
| `courtois-neuromod` | bold_movie, bold_video_game, bold_things_images, bold_audio, eye_tracking | observation/boundary/evaluation | p0 | 6 |
| `crhd` | bold, dwi, structural_mri, clinical_and_behavioural_assessments | observation/prior | restricted | — |
| `dhcp` | t1w, t2w, bold_rest, dwi, physiology | observation/prior | restricted | — |
| `ds000158` | bold_localiser, vocal_and_nonvocal_stimuli, t1w | observation/calibration | p1 | 218 |
| `ds003059` | derivative_fmri | observation | p1 | — |
| `ds003192` | multi_echo_bold, end_tidal_co2, respiration, cardiac_trace, breath_hold_task | observation/calibration | p0 | 7 |
| `ds003787` | bold_retinotopy, t1w, freesurfer_surfaces, fmriprep_derivatives, vistasoft_prf_parameters | observation/calibration | repo-held | 44 |
| `ds004873` | bold, t2_mapping, t2star_mapping, asl_cbf, cbv | observation/calibration | repo-held | — |
| `ds005917` | t1w, t2w, dwi, bold_rest, bold_task_dot_probe | observation/prior | p0 | 58 |
| `ds006072` | multi_echo_bold_rest, multi_echo_bold_task, t1w, t2w, dbsi | observation/evaluation | p0 | 7 |
| `ds006623` | bold_rest, bold_task_mental_imagery, bold_task_motor_response, t1w | observation | p0 | 26 |
| `ds007738` | fnirs_whole_head, eye_tracking, multi_talker_audio, attention_instruction, behaviour | observation/boundary | p0 | — |
| `gallant-crcns-natural-movies` | bold_natural_movie, bold_natural_image, stimulus_frames, retinotopic_localiser | observation/evaluation | p1 | — |
| `haxby-hyperalignment-corpora` | bold_movie, bold_task_localiser, t1w | observation/evaluation | p1 | — |
| `hcp-7t-retinotopy` | bold_retinotopy, prf_parameters, individual_surfaces, stimulus_apertures, variance_explained | observation/prior/calibration | p0 | 181 |
| `hcp-aging` | t1w, t2w, dwi, bold_rest, bold_task | observation/prior | restricted | 1200 |
| `hcp-development` | t1w, t2w, dwi, bold_rest, bold_task | observation/prior | restricted | 1300 |
| `hcp-movie-7t` | bold_movie, bold_retinotopy, bold_rest, dwi_7t, stimulus_clips | observation/boundary | p0 | 184 |
| `hcp-task-vision-maps` | task_contrast_maps, individual_glm_outputs, task_paradigms | prior/evaluation | p1 | 1200 |
| `ibc-ds000244` | bold_task, t1w, dwi, behaviour, task_paradigms | observation/evaluation | p1 | 12 |
| `kamitani-perception-and-imagery-fmri` | bold_seen_images, bold_imagined_categories, stimulus_image_features, behavioural_vividness_ratings, space_template | observation/evaluation | repo-held | 5 |
| `le-petit-prince` | bold_story_listening, audio_stimuli, word_alignments, syntactic_annotations, t1w | observation/boundary | p1 | 112 |
| `lifespan-motor-somatosensory-fmri` | bold_motor, bold_somatosensory, t1w, age, behaviour | observation/prior | p1 | 155 |
| `magic-memory-curiosity` | bold_movie, magic_trick_videos, curiosity_ratings, memory_test | observation/boundary | p1 | — |
| `midnight-scan-club` | bold_rest, bold_task, t1w, t2w, field_maps | observation/prior/evaluation | p1 | 10 |
| `monash-fdg-pet-fmri` | bold, dynamic_fdg_pet, plasma_glucose_samples, visual_stimuli, t1w | observation/calibration | p0 | 10 |
| `myconnectome` | bold_rest, bold_task, dwi, t1w, metabolomics | observation/prior/evaluation | p1 | 1 |
| `narratives` | bold_story_listening, audio_stimuli, time_aligned_transcripts, t1w, comprehension_scores | observation/boundary | p0 | 345 |
| `natural-scenes-dataset` | bold_images, prf_mapping, category_localisers, t1w, t2w | observation/boundary/evaluation | p1 | 8 |
| `naturalistic-neuroimaging-database` | bold_movie, film_stimuli, annotations, t1w | observation/boundary/prior | p1 | 86 |
| `nki-rockland` | t1w, dwi, bold_rest, bold_task, asl_perfusion | observation/prior | p1 | 1000 |
| `oasis-3` | t1w, t2w, bold_rest, asl_perfusion, dwi | observation/prior/negative_control | restricted | 1098 |
| `prime-de` | bold_rest, t1w, dwi, anaesthesia_metadata | observation/prior | p1 | — |
| `sald` | t1w, bold_rest, age, sex, handedness | observation/prior | p1 | 494 |
| `sherlock-merlin-princeton` | bold_movie, bold_free_recall, recall_transcripts, scene_annotations | observation/evaluation | p1 | — |
| `studyforrest-ds000113` | bold_movie, bold_audio_movie, t1w, t2w, dwi | observation/boundary | repo-held | 4 |
| `studyforrest-retinotopy` | bold_retinotopy, higher_visual_localisers, individual_surfaces, stimulus_apertures | observation/calibration | p0 | 15 |
| `tcia` | ct, mr, pet, digital_pathology, ultrasound | observation/prior | p1 | — |
| `uk-biobank-imaging` | t1w, t2flair, swi, dwi, bold_rest | observation/prior/evaluation | restricted | 100000 |
| `whole-body-somatotopy-fmri` | bold_somatotopy, movement_cues, t1w, derived_somatotopic_maps | observation/calibration | p1 | 62 |

### stimulation response (17)

| source | streams | role | status | n |
|---|---|---|---|---|
| `ds002094` | eeg, tms | observation | p1 | 20 |
| `ds003037` | tms_eeg, emg_mep, t1w, coil_positions, stimulation_parameters | observation/calibration | p1 | — |
| `ds003670` | eeg, ecg, eog, stimulation_parameters, montage_definitions | observation/calibration | repo-held | 19 |
| `ds005498` | bold_tms_concurrent, bold_rest, t1w | observation | repo-held | 152 |
| `ds005620` | eeg, tms_evoked_eeg_subset, awakening_reports, exclusion_flags | observation | p1 | — |
| `ds005779` | eeg, tms | observation | p1 | 19 |
| `ds008037` | eeg, tms | observation/negative_control | repo-held | — |
| `fastmri-brain` | multicoil_kspace, single_coil_emulated_kspace, reconstructed_volumes, acquisition_parameters, pathology_labels_subset | observation/calibration | p0 | 6970 |
| `fieldtrip-tms-eeg-tutorial-data` | tms_eeg, pulse_triggers | calibration/evaluation | ref | 1 |
| `gp-tms-hsh` | mep_emg, coil_pose, e_field_simulation, structural_mri | observation/evaluation | p1 | 8 |
| `multi-site-tep-data` | tms_eeg, site_metadata, stimulation_parameters, coil_targeting | observation/evaluation | p1 | — |
| `tesa-reference-data` | tms_eeg, reference_processed_outputs, artifact_component_examples | calibration/evaluation | ref | — |
| `tfus-esi-resting-eeg` | eeg, tfus_parameters, tes_parameters, t1w, targeting | observation/calibration | p0 | — |
| `tfus-mvep-speller` | eeg, tfus_parameters, transducer_targeting, visual_stimuli, bci_performance | observation/calibration | p0 | 25 |
| `tms-e-field-direction-mapping-osf-9f3bc` | mep_emg_raw, regression_data | observation | watch | — |
| `tms-localization-example-osf-myrqn` | mep_emg_raw, per_pulse_coil_pose, structural_mri, head_mesh, simulated_e_field | observation/calibration | watch | 1 |
| `tms-pulsewise-coil-displacement` | per_pulse_coil_displacement_scalar, simulated_cortical_e_field, mep_amplitude | observation | watch | — |

### structural and microstructural (11)

| source | streams | role | status | n |
|---|---|---|---|---|
| `calgary-campinas` | raw_kspace_single_channel, raw_kspace_multi_channel, t1w, skull_strip_masks, hippocampus_masks | observation/calibration/evaluation | p0 | — |
| `ds003949` | vascular_imaging, structural_mri | observation | p1 | — |
| `fiber-data-hub` | tractograms, bundle_segmentations, parent_dataset_lineage, algorithm_parameters | prior/evaluation | p1 | — |
| `hcp-7t-diffusion` | dwi_7t, field_maps, gradient_tables, preprocessed_volumes | observation | p1 | 184 |
| `m4raw` | raw_kspace_t1w, raw_kspace_t2w, raw_kspace_flair, raw_kspace_gre, reconstructed_volumes | observation/calibration | repo-held | 183 |
| `massive-brain-dataset` | dwi_multishell, dwi_cartesian_grid, b0_field_maps, noise_maps, flair | observation/calibration | watch | 1 |
| `ppmi` | datscan_spect, mri, clinical_motor, neurobehavioural, autonomic | observation/prior | restricted | — |
| `skm-tea` | raw_kspace_qdess, sensitivity_maps, dicom_echo1, dicom_echo2, tissue_segmentations | observation/evaluation | restricted | 155 |
| `switchboard` | conversational_audio, transcripts, turn_annotations, disfluency_annotations, speaker_metadata | boundary | p1 | 543 |
| `synthrad2023` | ct, mr_t1w, body_mask | observation/calibration/prior | repo-held | 180 |
| `tractoinferno` | dwi, t1w, preprocessed_derivatives, reference_tractograms, bundle_segmentations | observation/evaluation | repo-held | 284 |

### vascular geometry (8)

| source | streams | role | status | n |
|---|---|---|---|---|
| `adam` | tof_mra, structural_mri, aneurysm_annotations, clinical_metadata | observation/evaluation | p1 | 254 |
| `aneurisk` | vessel_surfaces, centerlines_with_radius, curvature_and_torsion, aneurysm_annotations, clinical_metadata | observation/calibration | p1 | 100 |
| `fmost-mouse-vasculature-blocks` | fmost_vasculature_volumes, voxel_annotations | observation/prior | watch | 4 |
| `ixi` | t1w, t2w, pd, mra, dwi_15dir | observation/prior/calibration | p0 | — |
| `retinal-fundus-vessel-datasets` | colour_fundus, manual_vessel_masks, fov_masks_partial, second_observer_masks_partial, optic_disc_annotations_partial | observation/evaluation | p1 | — |
| `topcow` | mra_volumes, cta_volumes, multiclass_vessel_labels, cow_variant_annotations, topology_evaluation_metrics | observation/evaluation/calibration | p0 | 200 |
| `tubetk-mra` | mra, structural_mr | observation/prior | watch | — |
| `vessap` | lightsheet_vasculature, vessel_segmentation, centerlines, bifurcation_points, vessel_radii | observation/prior | p1 | — |

### effector, autonomic and interoceptive (36)

| source | streams | role | status | n |
|---|---|---|---|---|
| `amass` | smpl_pose_sequences, body_shape_parameters, surface_meshes, source_dataset_labels | boundary/calibration | p1 | 300 |
| `ami-meeting-corpus` | multi_microphone_audio, video, transcripts, dialogue_acts, head_pose | boundary/calibration | p1 | 189 |
| `bidmc` | ppg, impedance_respiration, ecg, manual_breath_annotations, numerics | observation/boundary | p1 | 53 |
| `biovid-heat-pain` | thermal_stimulus_log, electrodermal_activity, ecg, facial_emg, frontal_video | observation/boundary | p1 | 90 |
| `body-surface-gastric-mapping` | high_resolution_egg, propagation_maps, normative_reference_values, meal_response | observation/boundary | watch | — |
| `bp4d-plus` | dynamic_3d_face, 2d_video, thermal_video, eda, heart_rate | observation/boundary | p1 | 140 |
| `capnobase` | capnography, ppg, ecg, expert_breath_annotations, expert_beat_annotations | observation/boundary | p1 | 42 |
| `cmu-mocap` | marker_trajectories, joint_angle_sequences, skeleton_definitions, motion_category_labels | boundary | p1 | 144 |
| `egg-three-channel` | surface_egg, gastric_frequency_measures, meal_conditions | observation/boundary | watch | — |
| `ego4d` | egocentric_video, audio, gaze_subset, imu, narrations | boundary | p1 | 931 |
| `extrasensory` | accelerometry, gyroscope, audio_features, location, self_reported_context_labels | boundary | p1 | 60 |
| `fantasia` | ecg, respiration, beat_annotations, age_group | observation/boundary | p1 | 40 |
| `gi-cine-mri` | cine_mri, gastric_volumes, small_bowel_motility, colonic_volumes, transit_times | observation/boundary | watch | — |
| `hd-epic` | egocentric_video, gaze, audio, 3d_scene_scans, hand_pose | boundary/calibration | watch | 9 |
| `human36m` | 3d_joint_positions, joint_angles, multi_view_video, depth, body_scans | boundary | p1 | 11 |
| `iemocap` | audio, video, facial_motion_capture, categorical_emotion_labels, valence_arousal_dominance_ratings | boundary | p1 | 10 |
| `isruc` | polysomnography, dual_expert_staging, respiratory_events, patient_and_control_subgroups | observation/evaluation | p1 | 118 |
| `mesa-sleep` | polysomnography, actigraphy, sleep_staging, respiratory_events, parent_cohort_vascular_measures | observation/boundary | restricted | 2237 |
| `mimic-waveforms` | ecg_waveforms, ppg, arterial_blood_pressure, respiration, numeric_trends | observation/boundary | restricted | — |
| `mit-bih-arrhythmia` | ecg_two_lead, beat_annotations, rhythm_annotations | observation/evaluation | p1 | 47 |
| `motion-x` | smplx_motion, hand_pose, facial_expression, text_annotations, source_video_ids | boundary | p1 | — |
| `nhanes` | laboratory_biomarkers, physical_examination, accelerometry, dietary_recall, questionnaires | prior/boundary | p1 | — |
| `ninapro` | surface_emg, hand_kinematics, force, movement_labels, participant_clinical_data | observation/boundary | p0 | 100 |
| `novel-view-synthesis-benchmark-scenes` | multi_view_images, sfm_camera_poses, laser_scanned_geometry_subset, held_out_test_views | boundary/evaluation | p0 | — |
| `open-humans` | genomics, wearables, microbiome, continuous_glucose, self_report | boundary/prior | watch | — |
| `openaps` | continuous_glucose, insulin_delivery, carbohydrate_entries, algorithm_state, device_metadata | observation/boundary | watch | — |
| `pamap2` | imu_streams, heart_rate, activity_labels, participant_characteristics | boundary/observation | p1 | 9 |
| `project-aria` | rgb_video, slam_cameras, eye_tracking, imu, audio | boundary/calibration | p1 | — |
| `ptb-xl` | ecg_12_lead, diagnostic_labels, demographics | observation/prior | p1 | — |
| `recola` | audio, video, ecg, electrodermal_activity, continuous_affect_annotations | boundary/observation | p1 | 46 |
| `scannet` | rgb, depth, camera_poses, reconstructed_meshes, semantic_labels | boundary/calibration | p1 | — |
| `shhs` | home_polysomnography, sleep_staging, respiratory_events, oximetry, cardiovascular_outcomes | observation/boundary | restricted | 6441 |
| `studentlife` | accelerometry, audio_inference, location, conversation_inference, sleep_inference | boundary/prior | p1 | 48 |
| `totalcapture` | marker_mocap, imu_streams, multi_view_video, camera_calibration, ground_truth_pose | boundary/calibration | p1 | 5 |
| `wesad` | chest_ecg, chest_eda, chest_emg, respiration, skin_temperature | observation/boundary | p1 | 15 |
| `x-ite-pain` | thermal_stimulus, electrical_stimulus, eda, ecg, facial_emg | observation/boundary | p1 | 134 |

### sensory boundary and stimuli (36)

| source | streams | role | status | n |
|---|---|---|---|---|
| `air-impulse-responses` | binaural_room_impulse_responses, monaural_impulse_responses, room_dimensions, source_receiver_geometry, reverberation_times | calibration/boundary | p1 | — |
| `allen-visual-coding-2p` | df_over_f_traces, roi_masks, event_detected_traces, max_projections, running_speed | observation/prior | p1 | 243 |
| `audiocaps` | captions, audioset_clip_ids | boundary | p1 | — |
| `audioset` | video_ids, weak_event_labels, vggish_embeddings, ontology | boundary/prior | p1 | — |
| `ava-avd` | diarisation_annotations, active_speaker_face_boxes, ava_clip_ids | boundary | p1 | — |
| `but-reverbdb` | room_impulse_responses, background_noise, room_geometry, microphone_positions | calibration/boundary | p1 | — |
| `cmu-mosei` | audio, video, transcripts, sentiment_annotations, emotion_intensity_annotations | boundary | p1 | 1000 |
| `coco-search18` | fixation_sequences, target_categories, coco_image_ids, target_present_absent | boundary | p1 | 10 |
| `common-voice` | read_speech_audio, transcripts, self_reported_demographics, validation_votes | boundary | p1 | — |
| `demand` | multichannel_noise_recordings, array_geometry, environment_metadata | boundary/calibration | p1 | — |
| `dundee` | fixation_measures, text_materials | boundary | p1 | 20 |
| `epic-kitchens` | egocentric_video, audio, action_annotations, object_masks_subset, hand_object_annotations | boundary | p1 | 45 |
| `fisher-english` | conversational_audio, quick_transcripts, topic_assignments, speaker_metadata | boundary | p1 | 11000 |
| `fsd50k` | audio, event_labels, ontology_mapping, metadata | boundary | p1 | — |
| `gazebase` | gaze_1000hz, task_labels, calibration_records, participant_ids | boundary/calibration | p1 | 322 |
| `geco` | fixation_measures, word_level_reading_times, text_materials, language_condition, participant_language_profile | boundary | p1 | 33 |
| `howto100m` | video_ids, asr_transcripts, task_labels | boundary | p1 | — |
| `librispeech` | read_speech_audio, transcripts, speaker_metadata, clean_and_other_partitions | boundary/calibration | p1 | 2484 |
| `meld` | audio, video, transcripts, emotion_labels, sentiment_labels | boundary | p1 | — |
| `msp-podcast` | speech_segments, categorical_emotion_labels, valence_arousal_dominance_ratings, speaker_ids | boundary | p1 | — |
| `musan` | music, speech, noise | boundary/calibration | p1 | — |
| `objaverse` | meshes, textures, materials, metadata_tags, animations_subset | boundary | p1 | — |
| `onestop-eye-movements` | fixation_sequences, reading_measures, text_materials, comprehension_questions, difficulty_conditions | boundary | p1 | — |
| `openeds` | eye_images, pixel_annotations, eye_region_segmentation, point_clouds_of_eye_region | boundary/calibration | p1 | 152 |
| `openeds2020` | eye_image_sequences, 3d_gaze_vectors, sparse_segmentation | boundary/calibration | p1 | 90 |
| `provo` | fixation_measures, cloze_predictability_norms, text_materials, part_of_speech_annotations | boundary | p1 | 84 |
| `psychencode` | bulk_rna_seq, single_cell_rna_seq, single_nucleus_rna_seq, atac_seq, genotyping | prior | restricted | — |
| `scatterbrains` | head_surface_mesh | calibration/prior | repo-held | 16 |
| `spoken-coco` | spoken_captions, text_captions, coco_image_ids, forced_alignments | boundary | p1 | — |
| `studyforrest-eyegaze` | gaze, pupil, saccades, fixations, blinks | boundary/calibration | p0 | 15 |
| `teyed` | eye_images, 2d_3d_landmarks, eyeball_models, gaze_vectors, eye_movement_type_labels | boundary/calibration | p1 | 132 |
| `touch-and-go` | tactile_images, visual_video, material_labels, touch_annotations | boundary | p1 | — |
| `vggsound` | video_ids, class_labels, audio_visual_correspondence | boundary | p1 | — |
| `voxceleb` | audio_segments, face_tracks, speaker_identities, nationality_metadata | boundary | p1 | 7000 |
| `voxconverse` | diarisation_annotations, overlap_annotations, video_ids | boundary | p1 | — |
| `youcook2` | video_ids, procedure_segment_annotations, recipe_labels | boundary | p1 | — |

### population and clinical priors (5)

| source | streams | role | status | n |
|---|---|---|---|---|
| `babel` | frame_level_action_labels, sequence_level_labels, action_taxonomy | boundary | p1 | — |
| `bridge2ai-voice` | protocolised_voice_tasks, clinical_diagnoses, questionnaires, demographics | boundary | watch | — |
| `human-microbiome-project` | 16s_sequencing, shotgun_metagenomics, metatranscriptomics, metabolomics, host_measurements | prior | p1 | 300 |
| `sg-mind` | — | observation/boundary | watch | — |
| `supercognition-longitudinal` | — | observation/boundary/calibration | watch | — |

### oracle, synthetic and negative control (10)

| source | streams | role | status | n |
|---|---|---|---|---|
| `aneumo` | synthetic_geometries, cfd_velocity_fields, pressure_fields, wall_shear_stress, boundary_conditions | oracle/evaluation | p1 | — |
| `brainweb` | simulated_t1, simulated_t2, simulated_pd, ground_truth_tissue_maps, ms_lesion_phantoms | oracle/calibration/evaluation | ref | — |
| `bwin-000-synthetic-scaffold` | synthetic_eeg | negative_control | ref | 0 |
| `bwin-oracle-corpora` | — | oracle | repo-held | — |
| `enigma` | case_control_effect_size_maps, structural_connectomes, cortical_thickness_summaries, surface_area_summaries | prior/evaluation/negative_control | p1 | — |
| `example-published-study` | coordinate_table | negative_control | ref | 0 |
| `ismrm2015-submissions` | tractography | negative_control/evaluation | repo-held | 0 |
| `ismrm2015-tractography-challenge` | simulated_dwi, ground_truth_bundles, submitted_tractograms, scoring_results | oracle/evaluation/negative_control | repo-held | 1 |
| `mit-tuebingen-saliency-benchmark` | fixation_maps, images, viewing_parameters, held_out_test_set | boundary/evaluation/negative_control | p1 | — |
| `salicon` | mouse_derived_saliency_maps, coco_image_ids | boundary/negative_control | p1 | — |



## 2. models we can distill from

46 model cards. a teacher supplies a distillation target for $p(\theta)$ — never an
observation likelihood. distilling a teacher's output as though it were measurement is
the single most tempting error here: a teacher's agreement with itself is not evidence,
and a materialization fitted to one inherits its biases as though they were biology.

teacher outputs enter as **infill**: where a dataset ships one stream and not another, a
teacher supplies the missing stream — and supplies it *with a precision*, calibrated from
the model's own reported accuracy on that stream. if a teacher explains a fraction $r^2$
of a variable's variance it may contribute at most

$$
\Delta J_{\text{distilled}} = \frac{1}{(1-r^2)\operatorname{Var}[x]}
$$

so an encoder reporting $r^2=0.1$ contributes about a tenth of what a perfect measurement
would. distilling at unit precision is the fastest way to make a model hold a teacher's
biases as firmly as its own measurements.

two corrections are mandatory beyond the nominal figure. reported accuracy holds on the
benchmark distribution, so off-distribution use inflates the variance by an amount that is
itself a fitted parameter. and a teacher's residuals covary across everything it writes, so
its residual covariance is diagonal plus low-rank and the precision it contributes is
therefore, by woodbury, a diagonal **minus** a low-rank correction. the sign matters: a
teacher's shared error does not add a constraint, it subtracts the confidence correlated
values would otherwise appear to supply. at a shared-error fraction of 0.9, a model
writing $10^4$ cortical positions supplies the precision of about **one** independent
measurement.

every card with `use: distil` carries a `distillation` block — `reported_accuracy` and
`precision_model`. those figures have to come out of the papers, and a card whose figure
is unverified says so. six of the EEG teachers are now filled in from the published
tables — `labram`, `biot`, `cbramod` and `eegpt` on TUAB, `bendr` on P300, `neuro-gpt` on
BCI Competition IV 2a — each with the table and the URL it was read from.

converting one of those figures into the $r^2$ the precision formula needs is a real
inferential step and each card documents it. an AUROC becomes a discriminability through
$\mathrm{auc} = \Phi(d'/\sqrt2)$ and then $r^2 = d'^2/(d'^2+4)$; the same table's balanced
accuracy read through $\mathrm{bac} = \Phi(d'/2)$ agrees to about 0.01, which is the only
available check that the latent-gaussian model behind the conversion holds. what comes out
is an $r^2$ on the *benchmark's latent label* and therefore an upper bound on any $r^2$ on
a state variable — "is this recording abnormal" is not a field value. a raw multi-class
accuracy cannot be converted at all, because the probit route needs the chance level and
the schema records no arity, so `neuro-gpt` carries a verified figure and still supplies no
precision. that is the correct outcome and the card says so.

the sharing fraction is the harder half, because no paper reports one. `bendr` is the
exception and the only card here whose figure is measured rather than declared: we hold the
weights, so `scripts/measure_bendr_error_correlation.py` runs the real encoder over
eegmmidb, perturbs its *input* with four named deployment shifts — amplifier gain, a
swapped electrode label, a dropped electrode, mains interference — and measures the
correlation of the induced representation error across the $8192$ values it writes. that
error is a real sample under a real distribution shift, so no ground truth is needed. it
comes out at $\rho \approx 0.05$, an order of magnitude below the $0.8$ the other cards
declare as a prior, and its leading eigenvalue holds only 3–7% of the error variance — so
a rank-1 model is a conservative simplification for this teacher and not a fitted rank.
`scripts/verify_distillation_precision.py` checks the woodbury path against a dense inverse
and reproduces the effective-constraints curve.

### brain-signal foundation models — infill unobserved channels, segments and modalities of recorded brain state

| model | provider | role | status | distillable streams |
|---|---|---|---|---|
| `bendr` | University of Toronto and collaborators | teacher/prior | repo-held | contrastive_pretraining, tuh_pretraining_corpus |
| `biot` | University of Illinois and collaborators | teacher/evaluation | p1 | heterogeneous_channel_tokenisation, cross_signal_type_transfer |
| `brainbert` | MIT and collaborators | teacher/prior | p1 | per_electrode_representation, brain_treebank_pretraining |
| `brainlm` | Yale University and collaborators | teacher/evaluation | p1 | masked_autoencoding_objective, scaling_and_transfer_results |
| `brainomni` | Shanghai Artificial Intelligence Laboratory; Departm | teacher | watch | cross_modality_representation |
| `brain-jepa` | National University of Singapore and collaborators | teacher/prior | p1 | gradient_positional_encoding, predictive_embedding_objective |
| `brain-harmony` | Zijian Dong, Ruilin Li, Joanna Su Xian Chong, Nioush | prior/teacher | watch | heterogeneous_tr_handling, modality_specific_pretraining_then_hub_token_fusion, unified_morphology_function_representation |
| `cbramod` | Zhejiang University and collaborators | teacher/evaluation | p1 | criss_cross_attention, cross_dataset_transfer |
| `csbrain` | Shanghai Artificial Intelligence Laboratory; Sun Yat | teacher | watch | cross_scale_representation |
| `eegpt` | Shanghai Jiao Tong University and collaborators | teacher/prior | p1 | channel_name_embeddings, dual_self_supervised_objective |
| `labram` | Shanghai Jiao Tong University and collaborators | teacher/evaluation | p1 | neural_tokenisation, multi_dataset_pretraining |
| `meg-gpt` | Oxford Centre for Human Brain Activity (OHBA) and Ox | teacher/evaluation | watch | autoregressive_generation, free_rollout_behaviour |
| `neuro-gpt` | University of Southern California and collaborators | teacher/evaluation | p1 | causal_chunk_prediction, transfer_to_small_datasets |
| `neurolm` | Shanghai Jiao Tong University and collaborators | teacher | p1 | eeg_as_language_tokens |
| `brain-of` | RWTH Aachen University (Computer Science 3 -- Softwa | teacher | watch | representation |

### stimulus-to-brain encoders and brain world models — infill brain state given a stimulus, or roll it forward

| model | provider | role | status | distillable streams |
|---|---|---|---|---|
| `tribe` | Meta AI and collaborators | teacher/evaluation | p1 | trimodal_encoding, modality_ablation_results |
| `tribe-v2` | Meta AI and collaborators | teacher | watch | architectural_revision |
| `mirage` | Gokce, AlKhamissi and Schrimpf (EPFL) | prior | watch | architecture_and_results |
| `neuroworld` | arXiv:2608.01773 | prior/teacher/evaluation | p0 | causal_transition_learning_result, strict_past_only_conditioning, free_rollout_evaluation, architectural_choices |
| `cinesync` | unresolved -- named in PROMPT.md §53.1 | teacher | watch | model_dataset_pairing |
| `brainvista` | unresolved -- named in PROMPT.md §53.1 | prior | watch | architecture_and_results |
| `brainworld` | unresolved -- named in PROMPT.md §53.1 | prior | watch | architecture_and_results |
| `neurostorm` | unresolved -- named in PROMPT.md §53.1 | prior | watch | architecture_and_results |
| `brainjanus` | unresolved -- named in PROMPT.md §53.1 | prior | watch | architecture_and_results |
| `brain-dit` | unresolved -- named in PROMPT.md §53.1 | prior | watch | generative_diffusion_architecture |

### stimulus encoders — infill the sensory boundary when a dataset ships raw media but no annotation

| model | provider | role | status | distillable streams |
|---|---|---|---|---|
| `audio-spectrotemporal-encoders` | multiple -- HuBERT, WavLM, Whisper encoders, spectro | teacher | p0 | spectrotemporal_representations, forcing_precision |
| `wav2vec-bert-2.0` | Meta AI | teacher | p0 | layerwise_representations, forcing_precision |
| `llama-3.2` | Meta AI | teacher | p0 | layerwise_representations, token_surprisal, forcing_precision |
| `language-models-multiscale` | multiple -- the language model family and structured | teacher | p1 | multiscale_linguistic_structure, discourse_and_coreference |
| `multiscale-visual-encoders` | multiple -- hierarchical vision transformers, featur | teacher | p1 | multiscale_feature_maps |
| `multimodal-audiovisual-models` | multiple -- audiovisual contrastive and fusion model | teacher | p1 | joint_audiovisual_representation, audiovisual_correspondence |
| `phonetic-acoustic-models` | multiple -- forced aligners, phonetic classifiers, a | teacher | p1 | phone_alignments, articulatory_inversion |
| `depth-scene-geometry-models` | multiple -- MiDaS, Depth Anything, DUSt3R and succes | teacher | p1 | estimated_scene_geometry |
| `optical-flow-point-tracking-models` | multiple -- RAFT, TAP-Vid/TAPIR, CoTracker and succe | teacher | p1 | motion_fields, point_tracks_and_occlusion |

### generative video and world models — infill a stimulus that was never recorded, or extend one that was

| model | provider | role | status | distillable streams |
|---|---|---|---|---|
| `wan2.2-i2v-a14b` | Alibaba Wan team | teacher | p0 | video_prediction, video_latents |
| `wan2.2-ti2v-5b` | Alibaba Wan team | teacher | repo-held | video_prediction, video_latents |
| `vjepa2` | Meta AI | teacher | ref | multiscale_visual_features, latent_prediction |
| `video-jepa-family` | multiple -- V-JEPA, V-JEPA 2 and successors | teacher | p0 | spatially_structured_video_representations, forcing_precision |
| `causal-video-diffusion-models` | multiple -- causal and autoregressive video generati | teacher | p1 | causal_generative_dynamics, forcing_precision |
| `leworldmodel` | Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann Le | prior | ref | surprise_signal, stated_distributional_constraint, small_end_to_end_from_pixels |

### body and effector models — infill pose, shape and kinematics

| model | provider | role | status | distillable streams |
|---|---|---|---|---|
| `smpl-family` | Max Planck Institute for Intelligent Systems | calibration/prior | p1 | parametric_body_model, shape_space, mesh_topology |

### baselines and evaluation harnesses — never a distillation target; they score a materialization

| model | provider | role | status | distillable streams |
|---|---|---|---|---|
| `braindecode` | University of Freiburg and contributors | evaluation/calibration | p0 | architecture_zoo, reproducible_implementations |
| `moabb` | MOABB contributors / Inria and collaborators | evaluation | p0 | standardised_evaluation_protocol, within_versus_cross_session_evaluation |
| `riemannian-baselines` | multiple -- pyRiemann and the Riemannian BCI literat | evaluation/calibration | p0 | covariance_manifold_methods, transfer_via_manifold_alignment |
| `channel-adaptation-baselines` | multiple -- the EEG transfer-learning literature | calibration/evaluation | p0 | spherical_interpolation, source_space_adaptation, convolutional_and_token_adaptation |

### predecessor models — prior and negative control

| model | provider | role | status | distillable streams |
|---|---|---|---|---|
| `scwbd-001-004-predecessor-models` | This programme's predecessor repository, integrated- | evaluation/negative_control/prior | ref | runnable_demoted_assumptions, shared_trunk_failure, contributed_source_accounting |



## 3. atlases, priors and derived geometry

149 sources. these shape $p(\theta)$ and supply anatomical partitioning systems (§2),
interaction topologies (§3), and the supports and coordinate frames fields are indexed
over (§1). a calibration-role source fixes an instrument or a geometry and is never
permitted to update the biology.

### atlases and reference maps (136)

| source | title | role | status |
|---|---|---|---|
| `7t-qsm-venograms` | 7T QSM and SWI venograms | prior/calibration | watch |
| `aal3v2` | AAL3v2 -- Automated Anatomical Labelling atlas, version 3 | prior | repo-held |
| `allen-brain-cell-atlas` | Allen Brain Cell Atlas -- human 10x single-cell and multimodal cell-type products | prior/calibration | p1 |
| `allen-cell-types-patchseq` | Allen Cell Types Database and Patch-seq multimodal cell characterisation | prior/calibration | p1 |
| `allen-human-brain-atlas` | Allen Human Brain Atlas -- microarray gene expression with spatial sampling | observation/prior | p1 |
| `allen-mouse-connectivity` | Allen Mouse Brain Connectivity Atlas -- mesoscale directed projections | prior/calibration | p1 |
| `arterial-territory-atlas-liu2023` | Digital 3D brain MRI arterial territories atlas (Liu 2023) | prior/evaluation | repo-held |
| `ashs` | ASHS -- Automatic Segmentation of Hippocampal Subfields | prior/calibration | p1 |
| `ashs-oap` | ASHS-OAP -- Optimised Atlas Package for hippocampal subfield segmentation | prior/calibration | p1 |
| `asl-bold-breath-hold-datasets` | Simultaneous ASL and BOLD breath-hold datasets | calibration/observation | p1 |
| `ba-exvivo` | Ex vivo Brodmann area labels from ten histologically sectioned hemispheres | prior | repo-held |
| `babelbrain-examples` | BabelBrain example cases | oracle | p0 |
| `benson-neuropythy-retinotopic-atlas` | Benson/neuropythy anatomically-defined retinotopic atlas | prior/calibration | p0 |
| `bican` | BICAN -- BRAIN Initiative Cell Atlas Network human cell census products | prior/calibration | watch |
| `bigbrain` | BigBrain -- 20 micron whole-brain histological reconstruction, complete volumetric produ | observation/prior/calibration | p1 |
| `bigbrain-hippocampal-cerebellar-derivatives` | BigBrain hippocampal and cerebellar derivative products | prior/calibration | watch |
| `braingraph-hcp-connectomes` | Braingraph.org HCP connectome collection | prior/calibration | p1 |
| `brainmap` | BrainMap -- curated coordinate database of functional and structural neuroimaging | prior/calibration | p1 |
| `brainspan` | BrainSpan -- Atlas of the Developing Human Brain | prior | p1 |
| `brainstem-navigator` | Brainstem Navigator -- in-vivo probabilistic brainstem nuclei atlas | prior/calibration | p1 |
| `breath-hold-hypercapnia-cvr-datasets` | Breath-hold and hypercapnia cerebrovascular reactivity datasets | prior/calibration | p1 |
| `buckner2011` | Buckner cerebellar functional-connectivity atlas | prior/calibration | p1 |
| `capillary-density-statistics` | Statistical capillary-density and vessel-orientation sources | prior/calibration | p1 |
| `cerebellar-atlases` | Diedrichsen lab cerebellar atlas collection (SUIT / Buckner / MDTB / Nettekoven) | prior/calibration | repo-held |
| `cerebral-artery-atlas-mouches2019` | Statistical atlas of cerebral arteries from multi-centre MRA (Mouches 2019) | prior | repo-held |
| `circle-of-willis-centerline-resources` | Circle of Willis centreline and mesh resources | prior/calibration | p1 |
| `cit168-pauli` | CIT168 / Pauli high-resolution probabilistic subcortical atlas | prior/calibration | p0 |
| `cocomac` | CoCoMac -- collated macaque tract-tracing connectivity database | prior | p1 |
| `cohort-angiographic-components` | Angiographic components of UKB, HCP, ADNI and clinical archives | prior/calibration | p1 |
| `cohort-perfusion-components` | Perfusion and vascular-pathology components of the large cohorts | calibration/prior | p1 |
| `coil-probe-measurements` | TMS coil field probe measurements | calibration | p1 |
| `conte69` | Conte69 population-average cortical surface atlas | prior/calibration | repo-held |
| `dbs-ieeg-stimulation-datasets` | DBS and intracranial stimulation datasets | prior/calibration | restricted |
| `desikan2006` | Desikan-Killiany gyral cortical parcellation | prior/calibration | repo-held |
| `destrieux2010` | Destrieux sulco-gyral cortical parcellation | prior/calibration | repo-held |
| `diff5t` | diff5T -- diffusion MRI raw data at 5 tesla | calibration | watch |
| `diffusion-thalamic-connectivity-atlases` | Diffusion-derived thalamic connectivity atlases | prior/calibration | p1 |
| `distal-lead-dbs` | DISTAL atlas and Lead-DBS subcortical resources | prior/calibration | p1 |
| `dkt-atlas` | Desikan-Killiany-Tourville cortical parcellation (DKT and DKT40) | prior | repo-held |
| `dream-olfaction-challenge` | DREAM Olfaction Prediction Challenge dataset | evaluation/boundary/prior | p1 |
| `dwi-phantoms` | Diffusion MRI physical phantoms | oracle/calibration | p1 |
| `electrode-impedance-contact-models` | Electrode impedance and contact models | calibration | p1 |
| `emg-kinematic-rehabilitation-datasets` | EMG and kinematic rehabilitation datasets | prior/boundary | p1 |
| `enigma-hcp-structural-connectome` | ENIGMA-HCP group structural connectome | prior/calibration | p1 |
| `focus-subcortical` | FOCUS 100 micron subcortical atlas | prior/calibration | watch |
| `food-101` | Food-101 -- food image classification dataset | boundary/negative_control | p1 |
| `freesurfer-thalamic-histological` | FreeSurfer ex-vivo histological thalamic nuclei atlas | prior/calibration | p1 |
| `frequency-resolved-source-association` | Frequency-resolved EEG/MEG source-space association maps | prior/calibration | p1 |
| `fsaverage` | fsaverage -- FreeSurfer population-average surface template | prior/calibration | repo-held |
| `fsaverage6` | fsaverage6 -- FreeSurfer average surface at ico6 (40,962 vertices per hemisphere) | prior/calibration | repo-held |
| `fusi-datasets` | Functional ultrasound imaging (fUSI) datasets | prior/calibration | p1 |
| `gaze-social-egocentric-datasets` | Gaze and social egocentric interaction datasets | boundary | p1 |
| `glasser2016` | Glasser HCP-MMP1 multi-modal cortical parcellation | prior/calibration | repo-held |
| `goulas-autoradiography` | Goulas receptor autoradiography cortical profiles | prior/calibration | p1 |
| `gtex` | GTEx -- Genotype-Tissue Expression project | prior/calibration | p1 |
| `h01` | H01 -- petavoxel electron-microscopy reconstruction of a human cortical fragment | observation/prior | p1 |
| `hansen-lausanne-sc` | Hansen structural connectome in the Lausanne parcellation | prior/calibration | p1 |
| `hansen-receptors` | Hansen neurotransmitter receptor and transporter density maps | prior/calibration | repo-held |
| `hansen-schaefer-sc` | Hansen structural connectome in the Schaefer parcellation | prior/calibration | p1 |
| `harvard-oxford` | Harvard-Oxford cortical and subcortical structural atlases | prior/calibration | repo-held |
| `hcp-7t-retinotopy-benson-maps` | The HCP 7T Retinotopy Dataset (Benson et al. derived pRF solutions) | prior/calibration | p0 |
| `hcp-functional-connectivity` | HCP group and individual resting functional connectivity | prior/calibration | p1 |
| `hcp-meg-maps` | HCP MEG derived power, connectivity and band-limited maps | prior/calibration | p1 |
| `hcp-s1200-maps` | HCP S1200 group-average maps -- myelin, thickness, curvature and function | prior/calibration | p1 |
| `high-resolution-vascular-atlases` | High-resolution arterial and venous atlases | prior/calibration | p1 |
| `hill2010-evolutionary-expansion` | Hill cortical evolutionary and developmental expansion maps | prior/calibration | p1 |
| `hippomaps` | HippoMaps -- multimodal hippocampal map repository in unfolded space | prior/calibration | p1 |
| `histological-tract-atlases` | Histological and dissection-based tract atlases | prior/oracle | p1 |
| `hubmap` | HuBMAP -- Human BioMolecular Atlas Program | prior/calibration | p1 |
| `human-cell-atlas` | Human Cell Atlas | prior/calibration | p1 |
| `human-protein-atlas` | Human Protein Atlas | prior/calibration | p1 |
| `hydrophone-phantoms` | Hydrophone measurements and acoustic phantoms | calibration | p0 |
| `iavs` | IAVS -- intracranial arterial and venous structure resource (unreleased) | prior/calibration | watch |
| `iit-human-brain-atlas` | IIT Human Brain Atlas v5.0 -- DTI/HARDI templates, probabilistic labels and connectome | prior | restricted |
| `intersubject-correlation-maps` | Naturalistic intersubject correlation maps | prior/calibration | p1 |
| `intrinsic-timescale-maps` | Intrinsic neural timescale maps | prior/calibration | p1 |
| `ismrmrd-example-data` | ISMRMRD / MRD example datasets | calibration | p0 |
| `julich-brain` | Julich-Brain cytoarchitectonic probabilistic maps | prior/calibration | p0 |
| `kwave-reference-problems` | k-Wave reference and benchmark problems | oracle/negative_control | p0 |
| `manufacturer-coil-geometry` | Manufacturer TMS coil and waveform specifications | calibration | p1 |
| `margulies2016-gradients` | Margulies principal cortical connectivity gradients | prior/calibration | p1 |
| `markov2014` | Markov macaque directed cortical connectivity | prior/calibration | p1 |
| `marmoset-brain-mapping` | Marmoset Brain Mapping and Brain/MINDS marmoset atlases and connectivity | prior | p1 |
| `mdtb-cerebellar` | MDTB -- Multi-Domain Task Battery cerebellar functional atlas | prior/calibration | p1 |
| `metabolomics-resources` | Human metabolomics resources | prior/calibration | p1 |
| `microns` | MICrONS -- millimetre-scale mouse visual cortex EM connectome with functional imaging | prior/calibration | p1 |
| `microscopy-microvascular-networks` | Microscopy-derived microvascular network reconstructions, with a flow solution | prior/calibration | held |
| `mida-head-model` | MIDA v1.0 -- multimodal imaging-based detailed anatomical model of the human head and ne | prior/calibration | restricted |
| `mni-open-ieeg-atlas` | MNI Open iEEG Atlas -- normative intracranial EEG | prior/calibration | watch |
| `motor-threshold-calibration-studies` | Motor threshold calibration studies | calibration | p1 |
| `mr-arfi-displacement-data` | MR-ARFI and focal displacement measurements | calibration/evaluation | p1 |
| `mri-navigated-pulse-pose-datasets` | MRI-navigated TMS pulse and pose datasets | calibration | p1 |
| `mridata-org` | mridata.org -- open raw MR datasets | calibration/observation | p1 |
| `netneuro-lausanne-sc` | NetNeuroTools Lausanne structural connectomes | prior/calibration | p1 |
| `neuromorpho` | NeuroMorpho.Org -- curated digital neuronal morphology reconstructions | prior/calibration | p1 |
| `neuroquery` | NeuroQuery -- predictive meta-analytic mapping from arbitrary text | prior/calibration | p1 |
| `neurosynth` | Neurosynth -- automated coordinate-based meta-analysis of the fMRI literature | prior/calibration | p1 |
| `neurovault` | NeuroVault -- repository of unthresholded statistical brain maps | prior/calibration | p1 |
| `nist-mri-phantoms` | NIST and ADNI quantitative MRI system phantoms | calibration | p0 |
| `ocmr` | OCMR -- open-access multi-coil k-space dataset for cardiovascular MRI | calibration | p1 |
| `pals-b12` | PALS-B12 surface atlas: Brodmann, lobes, visuotopic and orbitofrontal label sets, on fsa | prior | repo-held |
| `pcasl-datasets` | Pseudo-continuous arterial spin labelling datasets | observation/calibration | p1 |
| `physiological-etco2-recordings` | Physiological recordings of end-tidal CO2, respiration, pulse and blood pressure | calibration/observation | p0 |
| `population-and-individual-prf-maps` | Published population and individual pRF map collections | prior/evaluation | p1 |
| `ppg-benchmark-collections` | PPG benchmark collections | calibration/observation | p1 |
| `pyrfume` | Pyrfume -- curated olfactory psychophysics and molecular data | boundary/prior/calibration | p1 |
| `qbold-mqbold-datasets` | qBOLD and mqBOLD oxygenation datasets | observation/calibration | p1 |
| `qsm-oef-datasets` | QSM-based oxygen extraction fraction datasets | observation/calibration | p1 |
| `qsm-reconstruction-challenge` | QSM Reconstruction Challenge datasets | oracle/calibration/negative_control | p0 |
| `quantitative-relaxometry-datasets` | Quantitative relaxometry datasets | observation/calibration/prior | p1 |
| `raichle-metabolism` | Raichle aerobic glycolysis and resting metabolism maps | prior/calibration | p1 |
| `scanner-field-maps-and-gradient-phantoms` | Scanner field maps and gradient-nonlinearity phantoms | calibration | p0 |
| `schaefer2018` | Schaefer 2018 local-global functional parcellation | prior/calibration | repo-held |
| `sea-hero-quest` | Sea Hero Quest -- global spatial navigation behavioural dataset | boundary/prior | watch |
| `skull-acoustics-ct-mri-datasets` | CT and MRI skull datasets for transcranial acoustics | observation/calibration/prior | p0 |
| `speech-rt-mri` | Real-time speech MRI with synchronised audio and raw multicoil data | observation/calibration | p1 |
| `suit-diedrichsen` | SUIT / Diedrichsen cerebellar atlas and normalisation template | prior/calibration | p1 |
| `surveillance-tof-mra-datasets` | Surveillance TOF-MRA vessel datasets | prior/calibration | watch |
| `sydnor2021-hierarchy` | Sydnor sensorimotor-association cortical axis | prior/calibration | p1 |
| `task-ieeg-propagation-fields` | Task-specific intracranial propagation fields | prior/calibration | p1 |
| `temporal-receptive-window-maps` | Cortical temporal receptive window maps | prior/calibration | p1 |
| `tes-montage-field-validation-data` | tES montage and field validation measurements | calibration | p1 |
| `tes-solver-reference-fixtures` | ROAST and SimNIBS distributed head models -- NOT precomputed field solutions | oracle/calibration | p0 |
| `thomas-thalamic` | THOMAS -- thalamic nuclei segmentation from white-matter-nulled MRI | prior/calibration | p1 |
| `tian2020` | Tian subcortical functional parcellation (Melbourne Subcortex Atlas) | prior/calibration | repo-held |
| `tms-artifact-phantoms` | TMS artifact phantoms | negative_control/calibration | p0 |
| `tms-fmri-studies` | Concurrent TMS-fMRI studies | prior/calibration | watch |
| `tonotopy-maps` | Human auditory cortical tonotopy maps | prior/calibration | p1 |
| `transducer-calibration-datasets` | Focused ultrasound transducer calibration datasets | calibration | p1 |
| `usda-fooddata-central` | USDA FoodData Central | calibration | p1 |
| `vascular-model-repository` | Vascular Model Repository | oracle/prior | p1 |
| `venat` | VENAT -- high-resolution 7T QSM venous atlas | prior/calibration | held |
| `vesselgraph-mouse` | VesselGraph -- mouse whole-brain microvascular graph dataset | prior/calibration | held |
| `vestibular-proprioceptive-datasets` | Vestibular and proprioceptive task datasets | prior/boundary/calibration | p1 |
| `von-economo` | von Economo and Koskinas cytoarchitectonic atlas | prior/calibration | repo-held |
| `yeo2011` | Yeo 2011 resting-state cortical networks (7 and 17), on fsaverage | prior | repo-held |

### derived geometries (13)

| source | title | role | status |
|---|---|---|---|
| `aseg-subcortical-supports` | aseg subcortical supports -- one subject's own nuclei as crisp occupancy fields, in the  |  | None |
| `cortical-support-bank` | Cortical support bank -- oriented cortical g-splats, laminar supports and cover overlays |  | None |
| `device-geometry` | Sensor and device geometry -- electrodes, coils, transducers, cameras, displays and room |  | None |
| `example-cortical-patch-from-coordinates` | EXAMPLE — a cortical patch support reconstructed from a published coordinate table |  | None |
| `literature-derived-geometry` | Literature-derived geometry -- patches, supports, laminar profiles and devices reconstru |  | None |
| `mr-acquisition-objects` | MR acquisition objects -- scanner frame, sequence timeline, trajectory, coils and recons |  | None |
| `reference-cortical-patch-on-fsaverage` | Reference cortical patch: a published MNI152 peak reconstructed onto the real fsaverage  |  | None |
| `retinal-world-geometry` | Retinal and world geometry per episode -- display, eye, gaze, and the retinalized featur |  | None |
| `subcortical-eeg-identifiability` | Subcortical versus cortical EEG identifiability -- posterior width per source under one  |  | None |
| `tract-path-products` | Tract path products -- endpoint-density splats, path ensembles and arc-length tractometr |  | None |
| `unfolded-hippocampus` | Unfolded hippocampal coordinates -- one subject's hippocampus as a harmonic long-axis co |  | None |
| `vascular-products` | Vascular products -- resolved vessel graphs and stochastic microvascular ensembles |  | None |
| `vectorview-306-array-geometry` | Elekta Neuromag VectorView 306 sensor geometry, read from a real array and completed fro |  | None |

## 4. standards and tools

155 sources. neither is evidence. standards fix formats, frames and conventions; tools
are solvers and pipelines whose output is derived data with the tool's own error
attached. a tool's output is never an observation.

### standards and specifications (30)

| source | title |
|---|---|
| `arrow-parquet` | Apache Arrow and Parquet -- columnar in-memory and on-disk tables |
| `bids` | BIDS -- Brain Imaging Data Structure |
| `bids-derivatives` | BIDS Derivatives -- conventions for generated products |
| `cellml` | CellML -- markup language for mathematical models of biological processes |
| `cellml-physiome-body-models` | CellML and Physiome cardiac, respiratory, endocrine and gastrointestinal models |
| `cellml-physiome-circulation` | CellML and Physiome cerebral circulation and oxygen-transport models |
| `datalad` | DataLad and git-annex -- versioned management of large files |
| `duncan1995-forehead-dpf` | Adult forehead differential pathlength factor, Duncan et al. 1995 |
| `electrophysiology-container-formats` | EDF/EDF+, BDF, GDF, BrainVision, CTF, FIF and Curry electrophysiology containers |
| `haemodynamic-model-families` | Published haemodynamic and oxygen-transport model families |
| `hed` | HED -- Hierarchical Event Descriptors |
| `imaging-container-formats` | NIfTI, GIfTI, CIfTI, MGH/MGZ, NRRD and DICOM imaging containers |
| `ismrmrd` | ISMRMRD / MRD -- ISMRM Raw Data format |
| `lems` | LEMS -- Low Entropy Model Specification language |
| `lsl` | Lab Streaming Layer and hardware triggers |
| `neural-mass-model-families` | Neural mass and population model families: Wilson-Cowan, Jansen-Rit, Wong-Wang, Hopf, AdEx/a |
| `neurobagel` | Neurobagel -- federated cohort discovery across neuroimaging datasets |
| `neuroml` | NeuroML -- declarative model description language for computational neuroscience |
| `nibs-bids` | NIBS-BIDS (BEP037) -- BIDS extension for non-invasive brain stimulation |
| `nidm` | NIDM -- Neuroimaging Data Model |
| `nwb` | NWB -- Neurodata Without Borders |
| `ome-ngff` | OME-NGFF / OME-Zarr -- next-generation file format for bioimaging |
| `openminds` | openMINDS and SANDS -- metadata models for research products and spatial anchoring |
| `prahl-haemoglobin-extinction` | Haemoglobin molar extinction, Gratzer/Kollias as distributed by Prahl |
| `pulseq` | Pulseq -- open vendor-neutral pulse sequence format and toolchain |
| `reprolake` | ReproLake or an equivalent immutable object and data-lineage layer |
| `snirf` | SNIRF -- Shared Near Infrared Spectroscopy Format |
| `templateflow` | TemplateFlow -- versioned archive of neuroimaging templates |
| `tractogram-formats` | TCK, TRK and TRX tractogram formats |
| `zarr-xarray` | Zarr and xarray -- chunked, labelled N-dimensional arrays |

### tools and solvers (125)

| source | title |
|---|---|
| `3d-slicer` | 3D Slicer -- medical image computing platform |
| `acoustic-closed-forms` | O'Neil, Rayleigh and the layered-fluid transfer matrix: exact acoustic solutions |
| `afni` | AFNI -- Analysis of Functional NeuroImages |
| `afq-insight` | AFQ-Insight -- statistical learning on tractometry profiles |
| `ants` | ANTs -- Advanced Normalization Tools |
| `arbor` | Arbor -- performance-portable multi-compartment neuron simulator |
| `babelbrain` | BabelBrain -- prospective transcranial focused ultrasound modelling application |
| `babelviscofdtd` | BabelViscoFDTD -- viscoelastic finite-difference time-domain solver |
| `bart` | BART -- Berkeley Advanced Reconstruction Toolbox |
| `bigbrainwarp` | BigBrainWarp -- toolbox for transforming between BigBrain and standard surfaces |
| `biomodels` | BioModels -- repository of curated quantitative biological models |
| `bossdb` | BossDB -- Brain Observatory Storage Service and Database |
| `brain-code` | Brain-CODE -- Ontario Brain Institute informatics platform |
| `brain-image-library` | Brain Image Library (BIL) |
| `brainglobe` | BrainGlobe Atlas API -- programmatic cross-species atlas access |
| `brainpy` | BrainPy -- differentiable brain dynamics programming framework |
| `brainstorm` | Brainstorm -- MEG/EEG/iEEG analysis application |
| `brian2` | Brian 2 -- equation-oriented spiking neural network simulator |
| `cgal` | CGAL -- Computational Geometry Algorithms Library |
| `coins` | COINS -- Collaborative Informatics and Neuroimaging Suite |
| `connectome-workbench` | Connectome Workbench and the CIFTI grayordinate convention |
| `crcns` | CRCNS.org -- Collaborative Research in Computational Neuroscience data sharing |
| `cvrmap` | CVRmap -- BIDS post-processing toolbox for cerebrovascular reactivity mapping |
| `dabi` | DABI -- Data Archive for the BRAIN Initiative |
| `dandi` | DANDI -- Distributed Archives for Neurophysiology Data Integration |
| `dcm-spm` | Dynamic Causal Modelling and SPM |
| `dcm2niix-heudiconv` | dcm2niix and HeuDiConv -- DICOM to NIfTI/BIDS conversion |
| `dipy` | DIPY -- Diffusion Imaging in Python |
| `dryad` | Dryad |
| `dsi-studio` | DSI Studio -- diffusion MRI analysis and tractography |
| `duneuro` | DUNEuro -- finite element forward modelling for EEG, MEG and tES |
| `ebrains-knowledge-graph` | EBRAINS Knowledge Graph -- the openMINDS metadata index behind the EBRAINS data catalogue |
| `ebrains-siibra` | EBRAINS siibra and openMINDS spatial anchoring |
| `eegdash` | EEGDash -- metadata-first programmatic access to 700+ BIDS electrophysiology datasets |
| `eeglab` | EEGLAB -- EEG signal processing toolbox |
| `enigma-toolbox` | ENIGMA Toolbox -- harmonised atlases, summary maps and connectome utilities |
| `fastsurfer` | FastSurfer -- deep-learning cortical surface pipeline |
| `fcp-indi` | FCP/INDI -- 1000 Functional Connectomes Project and the International Neuroimaging Data-shar |
| `fem-bem-analytic-reference-problems` | Finite-element and boundary-element analytic reference problems |
| `fenics-dolfinx` | FEniCSx / DOLFINx -- general finite-element PDE solver |
| `fieldtrip` | FieldTrip -- MATLAB toolbox for MEG, EEG and iEEG analysis |
| `figshare` | figshare |
| `fmrib-eeg-fmri-artifact-references` | FMRIB and TESA EEG-fMRI and TMS-EEG artifact reference methods |
| `fmriprep` | fMRIPrep -- robust preprocessing pipeline for functional MRI |
| `freesurfer` | FreeSurfer -- cortical surface reconstruction and segmentation |
| `freesurfer-tracts` | FreeSurfer TRACULA and tract priors |
| `fsl` | FSL -- FMRIB Software Library |
| `fsl-probtrackx` | FSL ProbtrackX -- probabilistic tractography |
| `g-node-gin` | GIN -- G-Node Infrastructure |
| `gadgetron` | Gadgetron -- streaming medical image reconstruction framework |
| `getdp` | GetDP -- general environment for the treatment of discrete problems |
| `gmsh` | Gmsh -- finite element mesh generator |
| `gpu-jemris` | GPU-JEMRIS -- GPU-accelerated JEMRIS |
| `hippunfold` | HippUnfold -- topological unfolding of the hippocampus |
| `ieeg-org` | IEEG.org -- International Epilepsy Electrophysiology Portal |
| `itis-database` | IT'IS Foundation database of tissue material properties |
| `itk-simpleitk` | ITK and SimpleITK -- Insight Toolkit for image analysis |
| `jemris` | JEMRIS -- general-purpose MRI simulation framework |
| `k-wave` | k-Wave -- acoustic and ultrasound simulation toolbox |
| `komamri` | KomaMRI -- GPU-accelerated Bloch simulator in Julia |
| `loris` | LORIS -- Longitudinal Online Research and Imaging System |
| `meep` | Meep -- finite-difference time-domain electromagnetics |
| `meshio` | meshio -- mesh format conversion library |
| `mne-python` | MNE-Python -- MEG and EEG analysis and forward/inverse modelling |
| `modeldb` | ModelDB -- database of published computational neuroscience models |
| `monai` | MONAI -- Medical Open Network for AI |
| `mriqc` | MRIQC -- automated MRI quality control |
| `mrtrix3` | MRtrix3 -- diffusion MRI processing and tractography |
| `mrzero` | MRzero -- differentiable MR sequence simulation and automated sequence discovery |
| `mujoco` | MuJoCo -- Multi-Joint dynamics with Contact |
| `nemar` | NEMAR -- NeuroElectroMagnetic data Archive and tools Resource |
| `nemo-archive` | NeMO Archive -- Neuroscience Multi-Omic Archive |
| `nest` | NEST -- large-scale spiking neuronal network simulator |
| `netneurotools` | netneurotools -- network neuroscience analysis utilities |
| `netpyne` | NetPyNE -- high-level interface to NEURON for network modelling |
| `neurolib` | neurolib -- whole-brain neural mass modelling framework |
| `neuromaps` | neuromaps -- curated brain-map collection and transformation toolbox |
| `neuron` | NEURON -- simulation environment for morphologically detailed neurons |
| `nibabel` | NiBabel -- read/write access to neuroimaging file formats |
| `nilearn` | Nilearn -- statistical learning on neuroimaging volumes and surfaces |
| `nimare` | NiMARE -- Neuroimaging Meta-Analysis Research Environment |
| `nipype` | Nipype -- uniform interfaces to neuroimaging software |
| `nitrc` | NITRC -- NeuroImaging Tools and Resources Collaboratory |
| `nnunet` | nnU-Net -- self-configuring segmentation framework |
| `open-source-brain` | Open Source Brain -- platform for sharing and running neural models |
| `open3d` | Open3D -- library for 3D data processing |
| `opencor` | OpenCOR -- CellML modelling environment |
| `openfoam` | OpenFOAM -- open-source computational fluid dynamics |
| `openmeeg` | OpenMEEG -- symmetric boundary element forward solver for EEG and MEG |
| `openmrf` | OpenMRF -- open magnetic resonance fingerprinting resources |
| `openneuro` | OpenNeuro -- BIDS-native open neuroimaging archive |
| `opensim` | OpenSim -- musculoskeletal modelling and simulation |
| `opensim-moco` | OpenSim Moco -- direct collocation optimal control for musculoskeletal models |
| `osf` | OSF -- Open Science Framework |
| `osl-ephys` | OSL-ephys -- OHBA Software Library for electrophysiology |
| `popeye` | popeye -- population receptive field estimation toolbox |
| `prfpy` | prfpy -- population receptive field fitting library |
| `pyafq` | pyAFQ -- Automated Fiber Quantification in Python |
| `pynn` | PyNN -- simulator-independent neuronal network modelling API |
| `pyvista` | PyVista -- Pythonic interface to VTK |
| `qmrlab` | qMRLab -- quantitative MRI analysis and simulation toolbox |
| `qsiprep` | QSIPrep -- preprocessing and reconstruction pipeline for diffusion MRI |
| `qsm-reconstruction-tools` | QSM reconstruction tool family |
| `recobundles` | RecoBundles -- model-based bundle recognition from tractograms |
| `roast` | ROAST -- Realistic vOlumetric Approach to Simulate Transcranial electric stimulation |
| `sarvas-closed-forms` | Sarvas and Heller-van Hulsteyn closed-form solutions |
| `scilpy` | scilpy -- SCIL diffusion MRI processing toolbox |
| `simnibs` | SimNIBS — transcranial electric and magnetic field simulation |
| `simvascular` | SimVascular -- image-based cardiovascular modelling and blood-flow simulation |
| `sparc-portal` | SPARC Portal -- Stimulating Peripheral Activity to Relieve Conditions |
| `synthseg` | SynthSeg -- contrast-agnostic brain segmentation |
| `tetgen` | TetGen -- quality tetrahedral mesh generator |
| `the-virtual-brain` | The Virtual Brain (TVB) -- whole-brain network simulation platform |
| `tractoflow` | TractoFlow -- reproducible Nextflow diffusion processing pipeline |
| `tractometer` | Tractometer -- tractography evaluation system |
| `tractseg` | TractSeg -- convolutional bundle segmentation from fibre orientation maps |
| `trimesh` | trimesh -- Python triangular mesh library |
| `veinseg` | VeinSeg -- QSM vein probability segmentation |
| `vmtk` | VMTK -- Vascular Modeling Toolkit |
| `vtk` | VTK -- Visualization Toolkit |
| `webknossos` | webKnossos -- browser-based annotation and exploration of large microscopy volumes |
| `xcat-mrxcat-phantoms` | XCAT and MRXCAT -- 4D anatomical phantom and its MR simulation extension |
| `xnat-central` | XNAT Central -- DECOMMISSIONED 2024-05-01 |
| `yarra` | Yarra -- clinical MR raw-data processing and workflow framework |
| `zenodo` | Zenodo |

## 5. the neuro2.ai catalogue

`datasets.neuro2.ai` indexes **26,399 open neuroscience datasets** across
44 providers — 2.55 PB, 47,792 authors,
2,316 tasks, 49 modalities. **13,862** are human, together
covering 688,979 subject-records.

this is a catalogue, not a commitment. the full index is at
`data/index/neuro2/datasets.jsonl.gz`; a filtered promotion shortlist — human, openly
licensed, in a modality that binds to ibm state — is at
`data/index/neuro2/candidates.jsonl` and holds **7,459** entries. a row becomes a source
only when it is promoted into `data/sources/<id>/` with a written card (see
`data/README.md`).

for scale: the 251 curated datasets in §1 are roughly 1.8% of the human subset here. the
curated set is not a sample of this one — it was selected for having a resolvable
observation process, which most rows here do not have.

### by provider

| provider | datasets |
|---|---|
| `datacite` | 4,829 |
| `zenodo` | 3,488 |
| `figshare` | 3,021 |
| `osf` | 2,097 |
| `openneuro` | 1,815 |
| `synapse` | 1,485 |
| `brainlife` | 1,153 |
| `ebrains` | 1,142 |
| `huggingface` | 1,131 |
| `brainimagelibrary` | 959 |
| `dandi` | 840 |
| `dataverse` | 835 |
| `pennsieve` | 582 |
| `neurovault` | 501 |
| `neuromorpho` | 493 |
| `sciencedb` | 345 |
| `dabi` | 284 |
| `physionet` | 244 |
| `conp` | 182 |
| `crcns` | 153 |

...and 24 smaller providers.

### by primary modality

| modality | datasets |
|---|---|
| `eeg` | 6,901 |
| `mri` | 6,593 |
| `microscopy` | 1,843 |
| `meg` | 1,532 |
| `emg` | 1,503 |
| `ephys` | 1,494 |
| `ieeg` | 647 |
| `nirs` | 631 |
| `histology` | 627 |
| `pet` | 463 |
| `ophys` | 458 |
| `signals` | 453 |
| `dwi` | 445 |
| `eyetracking` | 407 |
| `tes` | 360 |
| `tms` | 297 |
| `dbs` | 254 |
| `pns` | 229 |
| `atlas` | 210 |
| `bold` | 187 |
| `genomics` | 142 |
| `ecg` | 137 |
| `fus` | 109 |
| `ecog` | 99 |
| `beh` | 88 |
| `cerebrovascular` | 54 |
| `clinical` | 40 |
| `motion` | 29 |
| `mrs` | 27 |
| `t1w` | 21 |

### human subset by modality

| modality | datasets |
|---|---|
| `eeg` | 4,344 |
| `mri` | 4,147 |
| `emg` | 717 |
| `microscopy` | 515 |
| `meg` | 503 |
| `ieeg` | 406 |
| `histology` | 362 |
| `nirs` | 299 |
| `signals` | 291 |
| `ephys` | 269 |
| `eyetracking` | 246 |
| `dwi` | 216 |
| `tes` | 186 |
| `pet` | 171 |
| `atlas` | 156 |
| `bold` | 154 |
| `tms` | 149 |
| `dbs` | 138 |
| `genomics` | 115 |
| `ecg` | 68 |
| `pns` | 61 |
| `ophys` | 51 |
| `ecog` | 49 |
| `beh` | 45 |
| `clinical` | 38 |
| `cerebrovascular` | 27 |
| `motion` | 22 |
| `t1w` | 21 |
| `fus` | 17 |
| `mrs` | 13 |

### licence

| licence | datasets |
|---|---|
| `unstated` | 10,236 |
| `CC-BY-4.0` | 8,048 |
| `CC0-1.0` | 4,709 |
| `CC-BY-NC-SA-4.0` | 924 |
| `restricted` | 722 |
| `CC-BY-NC-4.0` | 678 |
| `CC-BY-SA-4.0` | 282 |
| `CC-BY-NC-ND-4.0` | 195 |
| `PDDL-1.0` | 144 |
| `ODC-BY-1.0` | 144 |
| `Apache-2.0` | 101 |
| `MIT` | 87 |

open: 13,627 · closed or restricted: 2,536 · unstated: 10,236

an unstated licence is not an open one. 10,236 rows — about
38% of the catalogue — cannot be used
until their terms are resolved, and resolving them is per-source work.

### species

| species | datasets |
|---|---|
| `human` | 13,862 |
| `unknown` | 8,395 |
| `mouse` | 2,198 |
| `rat` | 792 |
| `macaque` | 319 |
| `drosophila` | 225 |
| `pig` | 94 |
| `non-human primate` | 78 |
| `zebrafish` | 60 |
| `dog` | 45 |
| `monkey` | 45 |
| `cat` | 39 |
| `c. elegans` | 33 |
| `marmoset` | 27 |
| `cow` | 22 |
| `songbird` | 21 |

non-human sources are not off-topic. they are the only sources constraining state that
ibm-1 declares and no human method reaches — laminar and columnar population state, ionic
and interstitial dynamics, and the microscale structure that transduction and the local
processes depend on. they enter as priors on $\theta$, never as observations of human
state.
