# Sandeha: Test Plan

## Strategy

- `uv run pytest` runs the suite from the project root. The layout under `tests/` follows `src/sandeha/`.
- Tests run on tiny data, built in a temporary directory, and never read the real dataset or real runs. The one exception is `test_real_file_shapes`, which checks the shape and counts of the downloaded public dataset, and is skipped when it is absent.
- The solver and scene-run tests use the GPU, because that is where they run for real, and `test_cuda_runs_a_matmul` has no CPU fallback. Data generation needs the GPU, so the `tests/data/` and `tests/train/` fixtures use it to make their tiny data. The training itself in those tests, and the hybrid, experiment and demo tests, run on the CPU. A few GPU tests are skipped when there is no GPU, and `test_peak_memory_under_budget` when under 2 GB is free.
- Where a figure can be worked out by hand, the test compares against that known answer. Where the code guards against something, the test breaks that thing on purpose: a flipped bit, a NaN, a real SIGKILL.
- `select_arch.py`, `train_ensemble.py`, `select_handoff.py` and `run_experiments.py` each have a `--dry-run` on tiny data that writes nothing in the repo. Tests run these, so each command's path is exercised before the real run.
- A passing test shows the code does what it says. It shows nothing about how good the trained models are. That comes only from the experiments.

## Test groups

### Foundations

| Tests | What they prove |
|---|---|
| `tests/test_env.py` | The installed PyTorch can multiply matrices on this GPU |
| `tests/test_repo.py` | Data, runs and checkpoints are ignored by git, while source and results are not |
| `tests/test_long_job.py` | The wrapper refuses without mains power, holds the inhibitor, logs start and end, reports real exit codes including signals, forwards signals to the job and its children, and waits for clean-up. It uses a fake power-supply directory and a recording `systemd-inhibit` |

### Solver: `tests/physics/test_solver.py`

These must pass before any data is generated. They run on the GPU.

| What is proved | Example tests |
|---|---|
| Divergence and projection are right | `test_divergence_known_answer`, `test_projection_removes_divergence`, `test_preconditioner_is_symmetric`, `test_pressure_solve_needs_few_iterations` |
| Obstacles hold | `test_no_flow_into_obstacles`, `test_smoke_does_not_cross_a_thin_wall`, `test_obstacle_change_mid_run_is_respected`, `test_new_wall_stops_smoke_on_its_first_step` |
| The physics is right, each against a known behaviour | `test_buoyancy_lifts_smoke`, `test_vortex_pair_carries_itself`, `test_steady_vortex_stays_centred`, `test_advection_follows_a_rotation_without_spiralling`, `test_smoke_matches_the_mass_budget`, `test_converges_with_refinement`, `test_thin_sheets_have_no_cell_scale_artefacts` |
| Odd input does no damage | `test_sealed_cavity_stays_finite`, `test_fully_solid_domain_stays_finite`, `test_non_finite_item_survives_step`, `test_bad_velocity_in_a_moving_flow_never_heals`, `test_huge_item_survives_step`, `test_iteration_limit_is_reported_as_not_converged` |
| Runs repeat, and batches match single runs | `test_deterministic`, `test_batch_matches_single_runs` |

### Scenes: `tests/physics/test_scenes.py`

| What is proved | Example tests |
|---|---|
| Each split draws only its allowed shapes and physics ranges | `test_train_never_uses_unseen_shapes`, `test_test_b_only_uses_unseen_shapes`, `test_test_c_outside_train_ranges`, `test_other_splits_stay_inside_train_ranges` |
| Unseen shapes differ in form and not just in size | `test_families_really_are_different_shapes`, `test_unseen_families_are_not_simply_larger` |
| `test_c` pairings are the right ones, because the flow depends only on buoyancy times strength | `test_flow_depends_only_on_buoyancy_times_strength`, `test_test_c_pairs_low_with_low_and_high_with_high` |
| Geometry rules hold | `test_every_scene_puts_an_obstacle_above_a_source`, `test_obstacles_are_thick_enough_not_to_be_jumped`, `test_smoke_always_has_a_way_up`, `test_inflow_edges_are_smooth` |
| Sampling repeats, across processes too | `test_sampling_is_seeded`, `test_sampling_is_the_same_in_a_new_process` |
| Scenes really run | `test_scenes_run_through_the_solver` |

### Data: `tests/data/`

| What is proved | Example tests |
|---|---|
| Test splits are locked, and the lock comes before any read | `test_open_split_refuses_test_without_final_eval`, `test_lock_is_checked_before_the_files_are_touched` |
| Changed data is caught | `test_manifest_detects_tampering`, `test_replacing_a_split_and_its_manifest_together_is_detected`, `test_an_edited_normalizer_is_refused` |
| Splits share no scene | `test_no_scene_shared_between_splits`, `test_seeds_are_disjoint_at_the_planned_size` |
| A stopped run ends with the same bytes | `test_killed_process_resumes_to_the_same_bytes`, `test_resume_refuses_a_changed_config`, `test_finished_data_is_never_regenerated` |
| Stored frames are what the solver made | `test_fields_match_a_direct_solver_run`, `test_fine_state_pools_to_the_stored_frame` |
| Masks and targets agree through a late obstacle | `test_every_target_is_clear_of_the_mask_in_its_input`, `test_obstacle_seq_follows_a_late_obstacle` |
| The normaliser uses train only, and items are in the planned order | `test_normalizer_uses_train_only`, `test_normalizer_refuses_to_fit_on_any_other_split`, `test_item_channels_are_in_the_planned_order` |
| Bad states never reach the store | `test_non_finite_state_stops_the_run_and_stores_nothing`, `test_float16_overflow_stops_the_run` |
| The travel rule works, and generation can't start before its controls are measured | `test_named_scene_against_twice_the_largest_control`, `test_generation_needs_the_controls_to_be_measured` |
| The spot-check catches problems | `tests/data/test_spotcheck.py`: a NaN, a wrong shape, a changed file or normaliser all fail it |

### Models and training: `tests/models/`, `tests/train/`

| What is proved | Example tests |
|---|---|
| Models have the right shapes and solids stay zero | `test_model_output_shape`, `test_solid_cells_zero_after_apply_step`, `test_a_non_finite_prediction_stays_out_of_solid_cells_only` |
| The loss is what it says | `test_divergence_known_answer_and_axes`, `test_step_loss_is_mse_on_fluid_cells`, `test_step_loss_measures_divergence_after_undoing_channel_scales` |
| Training can learn: loss falls 10× on one batch | `test_overfits_one_batch` |
| Pushforward gives gradients only to the last step | `test_only_the_last_pushforward_step_gets_gradients`, `test_pushforward_waits_for_the_warm_up` |
| A kill loses nothing, even inside a save. On the CPU, a resumed run matches an unbroken one | `test_checkpoint_atomic`, `test_resume_continues`, `test_resume_restores_optimizer_and_rng` |
| Memory stays below 5.5 GB for the U-Net at batch 16 | `test_peak_memory_under_budget` |
| The best checkpoint is picked correctly and a blow-up can't win | `test_train_keeps_the_lowest_validation_error_and_the_schedule_ends`, `test_rollout_error_known_answer_and_capped_on_blow_up` |
| Selection, member checks, retraining and ablations follow the plan | `test_selection_takes_the_lower_error_and_a_tie_goes_to_unet`, `test_check_rejects_a_member_no_better_than_persistence`, `test_a_failing_member_is_retrained_once_with_seed_plus_100`, `test_ablations_change_only_what_they_name` |
| Shipped configs hold the planned values | `test_configs_hold_the_planned_values`, `test_shipped_settings_match_the_plan` |

### Ensemble and hybrid: `tests/test_ensemble.py`, `tests/test_hybrid.py`

| What is proved | Example tests |
|---|---|
| Doubt is zero when members agree, rises when they don't, and has known values | `test_identical_members_zero_doubt`, `test_disagreement_raises_doubt`, `test_doubt_is_the_mean_over_channels_of_each_normalised_std` |
| Every member steps from one input | `test_every_member_steps_from_the_same_input` |
| Scores skip solids, and a non-finite fluid cell can't hide | `test_score_ignores_solid_cells`, `test_one_non_finite_fluid_cell_makes_the_score_infinite` |
| Hand-off follows the rules: start frame, frames emitted, low divergence | `test_handoff_starts_from_the_frame_before_and_emits_each_frame`, `test_handoff_returns_low_divergence_state` |
| Policies behave as specified | `test_fixed_schedule_period`, `test_random_budget_rate_within_tolerance`, `test_threshold_triggers_only_above_tau`, `test_never_candidate_does_not_hand_off` |
| Non-finite output is rescued and counted, and a failing solver stays failed | `test_non_finite_output_forces_a_handoff_under_every_policy`, `test_failing_solver_ends_handoffs_and_later_frames_stay_blown_up` |
| The oracle reads no later frame | `test_oracle_never_reads_future` |
| Edits and late obstacles apply on the right frame | `test_obstacle_change_applies_next_frame`, `test_obstacle_change_during_a_handoff_applies_after_it`, `test_late_obstacle_joins_inside_a_handoff_as_in_the_data` |
| Timing counts every member | `test_time_accounting_includes_every_member` |

### Selection and experiments: `tests/test_handoff_selection.py`, `tests/eval/`

| What is proved | Example tests |
|---|---|
| The per-frame error matches hand calculation and caps blow-ups | `test_frame_error_is_per_channel_rms_over_reference_std`, `test_frame_error_caps_blow_ups`, `test_nrmse_counts_fluid_cells_only_and_caps_non_finite` |
| Ranking measures have known answers, and non-finite doubt ranks highest | `test_spearman_known`, `test_auroc_known`, `test_non_finite_doubt_ranks_highest`, `test_undefined_time_is_left_out_and_counted` |
| Selection follows the preregistered steps and ties | `test_aggregations_left_in_keeps_the_leader_and_those_not_shown_worse`, `test_choice_counts_a_share_of_exactly_a_quarter`, `test_a_tie_with_never_goes_to_never`, `test_selection_refuses_any_split_but_validation` |
| Blown-up rollouts stay in | `test_blown_up_rollout_counts_as_max_error` |
| Baselines match the budget by the rules, and "beats" needs the rule's interval | `test_baseline_schedule_nearest_the_budget`, `test_unmatched_budget_runs_both_neighbours`, `test_comparison_needs_an_interval_below_zero_and_enough_budget` |
| The oracle's search keeps the rule's candidate | `test_oracle_tau_search_keeps_the_nearest_share`, `test_oracle_is_not_run_on_a_budget_of_zero` |
| Only `val` opens without `--final` | `test_experiments_use_val_unless_final` |
| Results carry provenance, and the renderer refuses those without | `test_results_carry_provenance`, `test_render_results_refuses_missing_provenance` |
| Stale rollouts rerun, and a docs-only commit doesn't force that | `test_rollouts_from_other_code_are_run_again`, `test_rollouts_survive_a_commit_that_leaves_the_code_alone` |
| The whole path runs, from experiments to figures | `test_dry_run_writes_every_result_with_provenance`, `test_figures_and_results_come_from_a_dry_run` |

### Public track: `tests/well/`

| What is proved | Example tests |
|---|---|
| The loader reads the right channels and shapes from a small synthetic HDF5 file, and counts are confirmed, not assumed | `test_channels_are_density_pressure_and_velocity`, `test_counts_are_read_and_confirmed`, `test_files_that_disagree_are_refused` |
| The test split is guarded, and the final run needs a selection from the same members | `test_test_split_guarded`, `test_final_needs_a_selection_from_the_same_members` |
| Rollouts and measures have known answers | `test_shared_mean_rollout_has_known_doubt_and_error`, `test_blown_up_rollout_counts_as_max_error_and_stays` |
| Results are labelled as a benchmark | `test_selection_opens_valid_only_and_final_is_labelled` |
| The real download has the expected counts, once present | `test_real_file_shapes` |

### Demo: `tests/demo/`

The server tests use small stand-in models on the CPU. The page tests start the real server with `--random-weights --device cpu --members 2` and drive it in Chromium through Playwright. Neither says anything about trained models.

| What is proved | Example tests |
|---|---|
| Frames stream, and old ones are dropped | `test_ws_streams_frames`, `test_clients_get_the_latest_frame_and_older_ones_are_dropped`, `test_handoff_frames_are_published_as_they_are_made` |
| Edits apply on the next frame, or after a hand-off | `test_brush_changes_obstacle_next_frame`, `test_edit_during_handoff_applies_after` |
| The demo survives misuse | `test_reconnect_keeps_running`, `test_fills_domain_stays_finite`, `test_brush_seals_off_and_covers_the_inflow_and_keeps_running`, `test_bad_messages_do_not_end_the_connection` |
| It recovers from blow-ups | `test_blown_up_state_recovers`, `test_a_failing_solver_falls_back_and_the_simulation_keeps_going` |
| Mode and threshold control the source | `test_threshold_and_mode_control_the_source` |
| Start-up failures give a clear message and a non-zero exit | `test_no_gpu_exits_with_message`, `test_oom_exits_with_message`, `test_missing_ensemble_exits_with_message` |
| Untrained models are disclosed, and can't be written as a result | `test_discloses_untrained_models_and_member_count`, `test_bench_refuses_to_write_results_from_random_weights` |
| The page updates, draws and shows who is in control, using no outside requests, within 1280×720 | `test_canvas_updates_over_time`, `test_drawing_changes_obstacle_layer`, `test_indicator_switches_in_solver_mode`, `test_no_external_requests`, `test_layout_fits_1280x720`, `test_touch_draws`, `test_keyboard_reaches_controls_and_draws` |

## How results are validated

### Rules

- Tuning uses train and validation only. Test splits are read by the single final run, and by the spot-check's integrity exception in `PREREGISTRATION.md`.
- The hand-off choice is committed before any test split opens, and `run_experiments.py --final` runs once, on committed code.
- Rollouts that blow up count at the error cap and are never dropped.
- Any bug fix after the final run is logged with the numbers before and after, and the first numbers stay on record.
- A hoped-for result that fails is reported as a finding.

### Checks on each result

| Check | How it works |
|---|---|
| Provenance | Each main-track experiment file carries `git_commit`, `data_manifest_sha256`, `checkpoint_sha256` and `config`. `check_provenance` and `render_results.py` refuse a file without them |
| Right data | `open_split` checks every file's SHA-256 against the manifest, and the manifest against its tracked copy, `results/data_manifest.json` |
| Right models | `results/ensemble.json` holds each checkpoint's SHA-256. The selection, experiments and demo all compare it before loading |
| Right code | Experiment rollouts record `code_sha256`, and rerun if the code that makes them has changed. Old sets are kept under `superseded/` |
| Numbers not typed | Figures and the numbers in `RESULTS.md` come from scripts reading saved JSON. A dry-run result is labelled as measuring nothing |
| By eye | `solver_check.png`, `scene_check.png`, `scene_check_flow.png` and `spot_check.png` in `results/figures/` are inspected by a person |

### What the existing result files record

| File | Provenance in the file | Limits |
|---|---|---|
| `results/solver_check.json` | `git.head` `2b2df8aa83d19198a1b8337844511f54463989ed`, `working_tree_changed` `false` | The script's own check flows, not dataset scenes. No grid finer than 256×256 was compared |
| `results/spot_check.json` | `git_commit` `890c6ffac970c00ca5099156b2905a7d567274f5`, `git_dirty` `true`, `manifest_git_commit` `fb27f486bad5643324dc27d33f6dcb0fc583def2`, `manifest_git_dirty` `false` | An integrity check. It says nothing about any model. It lists `scenes_over_thinnest_wall` of `[12]` in `val`, which `scripts/check_travel.py` then judges |
| `results/throughput.json` | `git_commit` `7955a5ae0f6c4eb68a31cd3b8bdbd96f2105f374` | 100 training steps per model and batch size. Not the demo |
| `results/arch_selection.json` | `git_commit` `2529b2be141941064aa0de3383ee459ae8ca51aa`, `git_dirty` `false`, `data_manifest_sha256` | One seed per model, on `val`. It chooses the architecture, and isn't a result about the ensemble |

## Before submission

- **Clean-clone check:** done on a fresh clone at `9c4bac7`: install, release weights, the trained demo served its page, and 573 tests passed with 1 skipped, without `tests/data/test_travel.py`. That file was run separately with `tests/demo` in the main checkout at the final docs commit: 84 passed.
