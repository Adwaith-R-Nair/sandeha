# Sandeha: Test Plan

## Strategy

- Tests are written first and seen to fail before the implementation is written.
- `uv run pytest` runs the suite from the project root. Test layout mirrors `src/`.
- Tests run on the real GPU where the behaviour depends on it. There is no CPU fallback for the environment check.
- A result counts as validated only when it comes from a saved run. Summaries of a run are not evidence.
- Each phase gets an independent check, separate from its build, before it is committed.

## Environment and repository (Task 1)

| Test | Checks |
|---|---|
| `tests/test_env.py::test_cuda_runs_a_matmul` | The installed PyTorch multiplies two 1024×1024 matrices on the GPU, and the result is finite |
| `tests/test_long_job.py` | Wrapper refuses on offline or unknown mains with exit 3, runs the job under the inhibitor (job command in the inhibitor's arguments), logs output and start and end, passes on the exit code, reports 128+n for a job killed by a signal, exits 4 and logs `exit=unknown` when the job never reports a status, forwards SIGTERM, SIGINT and SIGHUP to the job and its children, waits for job cleanup with the inhibitor held, handles a signal during inhibitor start-up, runs from the project root, rejects bad usage and the reserved name `jobs` with exit 2. The fake inhibitor copies the real one's exit 1 for a job killed by a signal |
| `tests/test_repo.py` | Root `data/`, `runs/` and `checkpoints/` are ignored, while `src/sandeha/data/` and `results/` are not |

The wrapper tests use a fake power-supply directory and a recording `systemd-inhibit`. The real inhibitor and real power state are exercised by hand, as logged in `BUILD_LOG.md`.

## Solver checks (Task 2)

Required to pass before any data is generated, all in `tests/physics/test_solver.py`, on the GPU:

| Test | Checks |
|---|---|
| `test_divergence_known_answer` | `u = x` gives divergence 1 away from walls, independently of `project` |
| `test_projection_removes_divergence` | Random velocity at 64×64, with and without an obstacle: largest divergence in fluid cells below `1e-3`, and the flow is not simply erased |
| `test_no_flow_into_obstacles` | After 50 steps, with inflow partly drawn over the obstacle: zero velocity and density in solid cells, and density non-negative |
| `test_deterministic` | Two runs in one process give identical tensors |
| `test_buoyancy_lifts_smoke` | The density centroid rises, and stays at the inflow with zero buoyancy |
| `test_converges_with_refinement` | 128×128 error below 0.8 of 64×64 error against 256×256, on each grid and on a common grid. Smooth inflow to `t=1`, hard inflow to `t=0.5` (`DECISIONS.md` B12) |
| `test_sealed_cavity_stays_finite` | A closed ring stays finite and converged, and nothing crosses it |
| `test_fully_solid_domain_stays_finite` | All-zero output and no failure of the pressure solve |
| `test_smoke_does_not_cross_a_thin_wall` | A one-cell wall across the box keeps all smoke below it |
| `test_obstacle_change_mid_run_is_respected` | A late obstacle gets a new pressure setup, and after exactly 1 step its cells hold no velocity or density |
| `test_new_wall_stops_smoke_on_its_first_step` | Flow from before an obstacle change does not carry smoke through a new one-cell wall |
| `test_batch_matches_single_runs` | Batched items match single runs, including iteration counts |
| `test_non_finite_item_does_not_stall_or_poison_the_batch` | In `project`, a NaN or infinite item uses no iterations, stays non-finite and is reported, and the other item is unaffected |
| `test_infinite_face_is_not_reported_as_converged` | One infinite face, which also makes the tolerance infinite, is not accepted |
| `test_non_finite_item_survives_step` | On GPU and CPU, over 3 batched steps, NaN or infinity in velocity, faces or density: no crash, item reported as not converged, the other item matches a single run, and the device still works |
| `test_bad_velocity_in_a_moving_flow_never_heals` | NaN or infinity in `u` or `v` of a moving flow reaches density, and stays non-finite for 10 steps |
| `test_huge_item_survives_step` | Velocity of `1e30` does not crash the solver or disturb the other item |
| `test_iteration_limit_is_reported_as_not_converged` | `cg_max_iter` is respected and reported |
| `test_pressure_solve_needs_few_iterations` | Below 60 iterations at both 64×64 and 256×256 |
| `test_preconditioner_is_symmetric` | Required for conjugate gradients to be valid |
| `test_steady_vortex_stays_centred` | A steady solution does not drift. Catches fields advected on the wrong lattice. A frozen velocity field also passes, so it says nothing about momentum advection |
| `test_vortex_pair_carries_itself` | A vortex pair travels 0.6 to 1.1 of the point-vortex distance, along either axis, without drifting sideways. Fails if either velocity component is not advected |
| `test_advection_follows_a_rotation_without_spiralling` | A rotating blob keeps its radius. Catches a first-order backtrace |
| `test_thin_sheets_have_no_cell_scale_artefacts` | Laplacian-to-density ratio on a thin-sheet plume stays below 0.25 (`DECISIONS.md` B10) |
| `test_smoke_matches_the_mass_budget` | Smoke in the box within 5% of inflow minus decay at `t=1.75` |

A rendered 256×256 run, `results/figures/solver_check.png`, is also inspected by eye. `uv run python scripts/check_solver.py` regenerates it, along with the speed, convergence and long-run divergence numbers.

## Data integrity (Tasks 3 and 4)

Shape families and physics ranges match each split. Test splits refuse to open without `final_eval`, tampered files fail the checksum, seeds are disjoint between splits, an interrupted generation resumes identically, and the normaliser uses train only. Sample scenes from each split are rendered and checked for zero NaNs before use.

Scene checks, in `tests/physics/test_scenes.py`:

| Test | What it establishes |
|---|---|
| `test_train_never_uses_unseen_shapes`, `test_test_b_only_uses_unseen_shapes` | Records which families the sampler really draws, including rejected attempts, and requires exactly the allowed set |
| `test_families_really_are_different_shapes` | Each family's overlap with a disc of equal area. Catches a builder that draws the wrong thing |
| `test_unseen_families_are_not_simply_larger` | Unseen families cover the same area as training families, within 10% on average |
| `test_test_c_outside_train_ranges`, `test_test_c_pairs_low_with_low_and_high_with_high`, `test_test_c_flow_parameter_is_outside_the_train_range`, `test_other_splits_stay_inside_train_ranges` | Physics ranges per split. `test_c` has only the two tied pairings, about equally often, and its buoyancy times strength is outside the training range |
| `test_flow_depends_only_on_buoyancy_times_strength` | Same product gives the same velocity and rescaled density, within `1e-5`. A different product differs by over 0.1. This is the reason for the tied pairings (`DECISIONS.md` B20) |
| `test_inflow_not_blocked`, `test_obstacles_keep_off_sources_even_when_placed_right_on_them` | Sources stay clear, including when placement is forced onto them |
| `test_inflow_edges_are_smooth` | No hard source edges (`DECISIONS.md` B12) |
| `test_inflow_sits_below_obstacles` | Catches a flipped axis |
| `test_every_scene_puts_an_obstacle_above_a_source`, `test_in_path_reach_cannot_miss_a_source` | Every one of 200 scenes per split at 256×256 has an obstacle above a source, and the reach constant makes a miss impossible (`DECISIONS.md` B21) |
| `test_placed_families_are_equally_likely`, `test_a_family_is_kept_when_its_first_pose_is_rejected` | Each family is placed a third of the time, within four binomial standard deviations (`DECISIONS.md` B22) |
| `test_obstacles_are_thick_enough_not_to_be_jumped`, `test_thickness_check_*` | No solid part is thinner than about 9 cells at 256×256, and the check itself detects thin walls and spikes |
| `test_smoke_always_has_a_way_up`, `test_sealed_draws_are_thrown_away`, `test_way_up_*`, `test_sampler_*` | A plume is never sealed in, judged with the late obstacle in place |
| `test_a_third_have_a_late_obstacle`, `test_no_late_obstacle_in_the_unfamiliar_splits`, `test_late_obstacle_is_valid_when_present` | Late obstacles per split, with a binomial limit on the share |
| `test_sampling_is_seeded`, `test_sampling_is_the_same_in_a_new_process` | Determinism within a process and across processes with different hash seeds |
| `test_scenes_run_through_the_solver` | Batched scenes from each split stay finite and converged on the GPU, with a late obstacle applied |

`uv run python scripts/check_scenes.py` runs 56 scenes for a full 480 solver steps, measures family shares and the flow parameter, and renders `results/figures/scene_check.png` and `scene_check_flow.png`, which are inspected by eye.

Data checks, in `tests/data/`, on tiny datasets: 64×64 reference grid and a few frames, and 256×256 for shapes only.

| Test | What it establishes |
|---|---|
| `test_open_split_refuses_test_without_final_eval`, `test_lock_is_checked_before_the_files_are_touched`, `test_final_eval_opens_a_test_split`, `test_train_and_val_open_without_final_eval` | The lock, and that it comes before any read |
| `test_manifest_detects_tampering`, `test_a_missing_file_counts_as_tampering`, `test_other_splits_still_open_when_one_is_tampered`, `test_no_manifest_means_no_data` | One flipped bit in a split's file fails its checksum, and no manifest means no data |
| `test_no_scene_shared_between_splits`, `test_seeds_are_disjoint_at_the_planned_size`, `test_a_scene_seed_does_not_depend_on_split_sizes`, `test_scene_uses_its_own_seed` | Seeds and scene content are disjoint. Resizing one split rerolls no other |
| `test_resume_generation`, `test_killed_process_resumes_to_the_same_bytes`, `test_resume_does_not_redo_finished_scenes`, `test_resume_ignores_a_torn_temporary_file` | Interrupted at a boundary, inside a batch, or by a real SIGKILL across processes, a run ends with the same checksums as an uninterrupted one. This also exercises determinism across processes, an open item from Task 2 |
| `test_resume_refuses_a_changed_config`, `test_resume_refuses_a_changed_commit`, `test_finished_data_is_never_regenerated`, `test_dirty_tree_is_refused_unless_allowed` | The set can't change silently |
| `test_fields_match_a_direct_solver_run` | Stored frames match a separate solver loop to the byte, which pins down late-obstacle timing, downsampling and solid clearing. Each scene's largest travel and its frame match the same loop |
| `test_resume_survives_a_torn_timing_line`, `test_an_unreadable_line_inside_the_timing_file_is_counted`, `test_a_missing_timing_file_costs_only_the_timing`, `test_a_timing_line_is_on_disk_before_the_next_batch` | Damaged timing costs only timing: a resume finishes with the same checksums, and the manifest counts what it couldn't read |
| `test_manifest_records_each_scene_s_travel`, `test_manifest_names_the_scenes_above_the_thinnest_wall`, `test_travel_is_a_distance_across_both_directions`, `test_travel_counts_only_the_faces_of_one_cell` | The spot-check's rule for travel above the thinnest wall has its figures, and they measure a distance |
| `test_restart_works_on_a_scene_with_no_late_obstacle`, `test_restart_applies_a_late_obstacle_at_its_frame`, `test_named_scene_against_twice_the_largest_control`, `test_a_named_scene_is_not_its_own_control`, `test_a_split_with_no_control_is_not_accepted`, `test_a_figure_that_is_not_a_number_does_not_pass`, `test_the_rule_runs_on_a_dataset_s_named_scene`, `test_control_sample_covers_every_split_and_all_kinds_of_scene` | The rule runs on any named scene of a dataset from its manifest alone, with no split's files present, and a doubtful result doesn't pass |
| `test_restart_reproduces_the_frames_generation_stores`, `test_both_step_sizes_cover_the_same_physical_time`, `test_the_rule_moves_a_restart_back_from_the_scene_s_end`, `test_the_difference_is_measured_over_the_fluid_of_each_frame`, `test_a_figure_is_the_largest_of_its_frames_wherever_that_falls` | The restart at the data's step matches stored frames to the byte on both sides of a late obstacle, both step sizes span the same time, and the rule itself moves a restart back |
| `test_generation_needs_the_controls_to_be_measured`, `test_measured_controls_open_the_gate`, `test_controls_from_other_physics_do_not_open_the_gate`, `test_controls_with_a_dirty_tree_or_no_figure_do_not_open_the_gate`, `test_only_a_scratch_run_outside_the_data_root_skips_the_gate`, `test_the_generation_command_stops_at_the_gate` | Generation can't start before the controls are measured for its physics |
| `test_a_rendering_shows_the_frames_around_the_spike`, `test_a_rendering_at_the_start_of_a_scene_has_no_frames_before_it`, `test_test_c_figures_state_each_scene_s_side`, `test_control_sample_gives_test_c_controls_for_each_side`, `test_recheck_measures_one_saved_scene_on_the_current_measure` | The rule's rendering exists, `test_c` figures can be read per side, and a saved scene can be rechecked on the current measure |
| `tests/data/test_spotcheck.py` | The spot-check counts non-finite values, covers every split, draws the first late-obstacle scene, fails on a NaN or a wrong shape, and refuses a changed file or normaliser |
| `test_a_crash_before_the_scene_files_go_is_cleaned_up_by_a_rerun`, `test_the_estimate_sample_measures_its_scene_files` | A crash after the manifest leaves no scene files after a rerun, and no data changes. Peak disk is measured |
| `test_fine_state_pools_to_the_stored_frame`, `test_each_fine_frame_pools_to_its_own_frame_through_a_late_obstacle`, `test_late_obstacle_at_both_grids_is_the_one_sampled` | Solver starts and the true solid hold the right contents: a fine state pools to its stored frame to the byte, in every split |
| `test_replacing_a_split_and_its_manifest_together_is_detected`, `test_the_dataset_does_not_open_without_its_tracked_copy`, `test_the_lock_comes_before_the_tracked_copy`, `test_an_edited_normalizer_is_refused` | The planned dataset's manifest must match its tracked copy, and the normaliser its checksum |
| `test_manifest_counts_test_c_sides_from_buoyancy`, `test_normalizer_applies_a_late_obstacle_on_the_planned_frames`, `test_inflow_scale_is_the_peak_of_train` | Side counts and scales match figures worked out separately |
| `test_every_target_is_clear_of_the_mask_in_its_input`, `test_late_obstacle_changes_the_mask_at_its_frame`, `test_obstacle_seq_follows_a_late_obstacle` | Masks and targets agree through a late obstacle |
| `test_non_finite_state_stops_the_run_and_stores_nothing`, `test_float16_overflow_stops_the_run`, `test_unconverged_solves_are_recorded`, `test_manifest_records_solver_health`, `test_manifest_records_how_float16_compares_with_the_change_per_frame` | A bad state never reaches the store, and solver health and rounding are on record |
| `test_normalizer_uses_train_only`, `test_normalizer_refuses_to_fit_on_any_other_split`, `test_fit_matches_the_saved_normalizer`, `test_a_constant_channel_is_refused` | Stats match a separate calculation on train, and differ from another split's |
| `test_item_shapes`, `test_item_channels_are_in_the_planned_order`, `test_targets_are_the_following_frames`, `test_every_window_is_there_once`, `test_make_input_rebuilds_an_item_input` | Items at the planned 128×128 grid, in the planned channel order |

## Training (Task 5)

Model output shapes, solid cells forced to zero, zero divergence for uniform flow, overfitting one batch by at least 10×, atomic checkpoints under interruption, resume restoring optimiser and RNG state, and peak GPU memory below 5.5 GB for U-Net at batch 16.

## Ensemble and hybrid (Tasks 7 and 8)

Zero doubt from identical members and from a single member, disagreement raising doubt, doubt scores ignoring solid cells, low divergence after hand-off, schedule and budget policies behaving to specification, the threshold policy triggering only above `tau` or on non-finite doubt, a forced hand-off from the last finite frame under every policy, the oracle never reading the future, obstacle edits applying on the next frame, and timing including every ensemble member.

## Experiments (Task 9)

Known-answer tests for each metric. A blown-up rollout counts as capped maximum error and stays in the average. Non-finite doubt ranks highest in every ranking, and no frame is dropped. Selection steps, tie-breaks, baseline schedules and unmatched budgets follow `docs/PREREGISTRATION.md`. Results carry provenance, the renderer refuses results without it, and experiments open only `val` without `--final`.

## Public track (Task 10)

A small synthetic HDF5 file checks shapes and the test-split guard. After download, a further test confirms the real file's grid, steps and trajectories, and is skipped when the data is absent.

## Demo (Tasks 11 and 12)

| Area | Checks |
|---|---|
| Server | Frames stream, brush applies on the next frame, edits during hand-off apply after it, reconnect keeps running, a filled domain stays finite, a blown-up state recovers through the solver, a missing GPU or exhausted memory gives a clear message and non-zero exit |
| Front end | Canvas updates, drawing changes the obstacle layer, indicator switches in solver mode, no external requests, layout fits 1280×720 |
| Performance | Neural-mode frame rate measured for 60 s with 5 members, written to `results/demo_bench.json` |

## Scientific validity rules

- Tuning uses validation only. Test splits are read only by the single final evaluation.
- Rollouts that blow up are failures at maximum error, and are never dropped.
- Any bug fix after the final run is logged with the before and after numbers.
- Every number in a document comes from saved JSON and is never typed by hand.

## Release check (Task 13)

From a clean clone in a temporary directory: install, download Release weights, run tests, start the demo.
