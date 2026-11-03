# Sandeha: References

This file checks every reference in section 10 of `docs/DESIGN.md` against its source. Checks were made on 2026-11-03. Section 10 itself has not been edited, and no source has been dropped: all seven are here, followed by five the list leaves out.

This file says nothing about how this project's models perform. Where a paper's own numbers appear, they are about that paper's systems.

**How far each check went.** Entry 1 was read in full. For the others, the metadata was checked and the passages that bear on our use were read, not the whole paper. "Checked against" names what was opened.

## Summary

| # | Source | In section 10? | Exists as cited? | Says what we use it for? | Action |
|---|---|---|---|---|---|
| 1 | Lakshmanan and Chopra, Hybrid Neural World Models | Yes | Yes, authors were missing | Partly. "Closest" doesn't hold | Reword |
| 2 | Li et al., Fourier Neural Operator | Yes | Title shortened, year is the preprint's | Yes | Fix title and year |
| 3 | Lakshminarayanan et al., Deep Ensembles | Yes | Yes | Partly. Our doubt is the variant it argues against | State the difference |
| 4 | Brandstetter et al., Message Passing Neural PDE Solvers | Yes | Yes | Yes | Say our version differs |
| 5 | Kochkov et al., ML-accelerated CFD | Yes | Yes | Yes, as an idea, not a method | None needed |
| 6 | Stam, Stable Fluids | Yes | Yes | Yes for the splitting, not for all of the solver | Limit the claim |
| 7 | Ohana et al., The Well | Yes | Title shortened | Yes | Give full title |
| 8 | Fielding et al., Multiphase Gas | **No** | Yes | Yes | **Add. The dataset card asks for this citation** |
| 9 | Roy et al., ANCHOR | **No** | Yes | Yes | Add as closest mechanism |
| 10 | Srikishan et al., HyPER | **No** | Yes | Yes | Add |
| 11 | Ronneberger et al., U-Net | **No** | Yes | Yes | Add, since it is the backbone |
| 12 | Eckert et al., ScalarFlow | **No** | Yes | One detail is out of date | Add, and soften the sign-up claim |

## Wrong or out of date in section 10

1. **"Closest found prior work" is not supported.** Entry 1 hands whole trajectories to the solver and never resumes the network. Entry 9 does what our design does: it watches a signal during the rollout, runs the solver for a fixed number of steps when a threshold is passed, then hands back to the network. Suggested wording: "Related work on trust-gated solver fallback, at whole-trajectory level, using a single-network signal."
2. **Entry 1 has no authors.** They are Lakshmanan and Chopra.
3. **"So far only a summary of the first has been read" is out of date.** Entry 1 has now been read in full.
4. **Entry 2's title is shortened and its year is the preprint's.** The title ends "Partial Differential Equations". It was published at ICLR 2021.
5. **Entry 7's title is shortened.** The full title is in the entry below.
6. **The paper the dataset card asks us to cite is missing.** That paper is entry 8, not The Well.
7. **Entries 9, 10, 11 and 12 are missing.** We need 9 and 10 for a fair account of prior work, 11 because U-Net is the backbone, and 12 because the design names it as a rejected dataset.

Dataset claims in section 4 of the design that were checked and hold: CC-BY-4.0, `turbulent_radiative_layer_2D` at 7.9 GB (the hub reports 7,856,295,353 bytes, and the card's 6.9 GB is its estimate for the simulations alone), `shear_flow` at 471 GB, and `rayleigh_benard` at 370 GB. The card and hub sizes were read without an account. No data file was downloaded for this check.

## Citation for the public dataset

**Is a citation required by the dataset card? Yes, it asks for one, and not for the paper in our list.**

- The card for `polymathic-ai/turbulent_radiative_layer_2D` says: "Please cite the associated paper if you use this data in your research". The paper it gives is Fielding et al. 2020 (entry 8). The project's documentation page for the dataset says the same.
- The card does not mention The Well paper.
- The Well's GitHub README says: "If you find this project useful for your research, please consider citing". It gives Ohana et al. (entry 7).
- The CC-BY-4.0 licence doesn't name a paper. When material is shared, section 3(a) asks for credit to the creators, a link to the material, a licence notice, and a note of any changes. This is our plain reading, not legal advice.

**What we'll do:** cite both entries 7 and 8 wherever the public track appears, and credit the dataset, its licence and its link in the report and README.

## Section 10 references

### 1. Hybrid Neural World Models

- **Citation:** Pranav Lakshmanan and Paras Chopra. *Hybrid Neural World Models.* arXiv:2605.28317v1, 27 May 2026. Preprint, marked "Under review".
- **Link:** https://arxiv.org/abs/2605.28317
- **Used for:** related work on sending a surrogate's uncertain predictions to a solver.
- **Checked against:** the full arXiv PDF, including appendices. arXiv lists v1 as the only version.
- **Unverified:** it has not been peer reviewed, and its code "will be released on acceptance", so none of its numbers can be checked independently.
- **Wrong in design:** no authors given, and "closest" doesn't hold. See below.

The paper is real and does include surrogate-to-solver fallback, but it differs from our design in almost every part that matters:

| | The paper | Sandeha, as designed |
|---|---|---|
| Model | One network predicts any horizon `T` in one pass | One network step advances a fixed 4 solver steps, in a rollout |
| Signal | Step-doubling: one prediction at `T` against two chained at `T/2` | Per-cell spread across an ensemble |
| Decision | One per trajectory | One per frame |
| Fallback | The solver replaces the whole trajectory | The solver runs for a set number of frames, then the network resumes |
| Solver's error | Deferred trajectories count as zero error | The solver starts from the network's state, so it can inherit that state's error |
| Systems | Reaction-diffusion, compressible Euler, a bouncing ball | Incompressible smoke with obstacles |

Three points to handle when citing it:

- **It treats ensembles as a baseline to beat.** Its Table 3 gives a 3-member deep ensemble a mean AUROC of 0.77 against 0.82 for step-doubling, on its own systems. The report should mention this rather than cite the paper only as support. It says nothing about our system.
- **Its random-deferral floor doesn't carry over.** Its "exactly 25%" floor for deferring 25% at random relies on deferred trajectories having zero error. Our solver doesn't start from the true state.
- **Its numbers don't all agree, so quote them with care.** Section 5.5 gives a 43% to 52% cut at `h=64` on the test split, while the same row of Table 6 gives 40%, 57% and 43%. Section 5.2 gives 0.43 ± 0.02 for Ball 3D far-OOD at `h=16`, while Table 5 has 0.59 ± .05 there and 0.43 ± .02 at `h=32`. Section 3.1 lists horizons from 2, while Table 1 lists them from 1.

### 2. Fourier Neural Operator

- **Citation:** Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart and Anima Anandkumar. *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021. arXiv:2010.08895.
- **Link:** https://arxiv.org/abs/2010.08895
- **Used for:** the second architecture compared against U-Net (design section 5.2, `results/arch_selection.json`).
- **Checked against:** arXiv metadata and the v3 PDF, whose header reads "Published as a conference paper at ICLR 2021". Its passages on periodic and non-periodic boundaries were read.
- **Unverified:** the OpenReview page refused automated access. The ICLR 2021 listing was seen only in search results.
- **Wrong in design:** title shortened to "PDEs", and year given as 2020, which is the preprint's. Our coordinate channels for the non-periodic box are our own choice, not from this paper, which handles non-periodic boundaries with its linear bias term.

### 3. Deep Ensembles

- **Citation:** Balaji Lakshminarayanan, Alexander Pritzel and Charles Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* Advances in Neural Information Processing Systems 30 (NIPS 2017), pages 6402-6413. arXiv:1612.01474.
- **Link:** https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
- **Used for:** the idea that independently trained networks with different seeds and data order can express uncertainty, including higher uncertainty away from the training data.
- **Checked against:** the proceedings page and the arXiv v3 PDF. The page range comes from a bibliography listing, as the proceedings page doesn't show one.
- **Unverified:** nothing in the citation. The paper doesn't test fluid simulation or rollouts.
- **Wrong in design:** the entry is correct, but the citation needs a limit. The paper trains each network to predict a mean and a variance using negative log-likelihood. Our members are trained on MSE and we take the spread of their outputs. The paper calls that "a commonly used heuristic" and says "this is inferior to learning the variance by training using NLL", shown on a toy regression task. So cite it for seeds and shuffling and for the general idea, and say that our doubt is the MSE variant. This project has not measured whether the difference matters here. Sharing the ensemble mean between members at each rollout step is also ours, not the paper's.

### 4. Message Passing Neural PDE Solvers

- **Citation:** Johannes Brandstetter, Daniel Worrall and Max Welling. *Message Passing Neural PDE Solvers.* ICLR 2022 (Spotlight). arXiv:2202.03376.
- **Link:** https://arxiv.org/abs/2202.03376
- **Used for:** the pushforward trick in rollout training (design section 5.3).
- **Checked against:** arXiv metadata and the v3 PDF, whose header reads "Published as a conference paper at ICLR 2022". The pushforward section was read.
- **Unverified:** the ICLR proceedings page itself was not opened.
- **Wrong in design:** nothing. Note that the paper describes "unrolling the solver for 2 steps but only backpropagating errors on the last unroll step", while we push a batch 1 to 3 steps on half the batches after a warm-up (`docs/LLD.md`). So write "based on" rather than "as in". The abstract doesn't use the word pushforward, so cite the section, not the abstract.

### 5. Machine learning-accelerated CFD

- **Citation:** Dmitrii Kochkov, Jamie A. Smith, Ayya Alieva, Qing Wang, Michael P. Brenner and Stephan Hoyer. *Machine learning-accelerated computational fluid dynamics.* Proceedings of the National Academy of Sciences 118(21), e2101784118, 2021. DOI 10.1073/pnas.2101784118. arXiv:2102.01010.
- **Link:** https://doi.org/10.1073/pnas.2101784118
- **Used for:** the idea of training a coarse-grid model against a finer simulation, as the reason for the 256×256 reference.
- **Checked against:** Crossref metadata and the arXiv PDF. The passages on learned solvers and the training procedure were read.
- **Unverified:** the PNAS page refused automated access, so the arXiv text was read, not the published one.
- **Wrong in design:** nothing. Note that its models work inside a coarse-grid solver, as learned interpolation or a learned correction, whereas ours is a network that replaces solver steps. It supports the idea, not our method.

### 6. Stable Fluids

- **Citation:** Jos Stam. *Stable Fluids.* Proceedings of SIGGRAPH '99, pages 121-128, ACM Press, 1999. DOI 10.1145/311535.311548.
- **Link:** https://doi.org/10.1145/311535.311548
- **Used for:** the basis of the reference solver, which is semi-Lagrangian advection followed by a pressure projection.
- **Checked against:** Crossref metadata and the author's PDF from the University of Toronto.
- **Unverified:** nothing in the citation.
- **Wrong in design:** "Solver basis" is fine only for that splitting. The paper prefers a cell-centred grid, while our solver stores velocity on cell faces (`docs/LLD.md`). It mentions multigrid in general, but not conjugate gradients, and buoyancy appears only as a passing remark. Those parts of our solver shouldn't be credited to it, and they have no source in this list. Add one if the report describes them as coming from the literature.

### 7. The Well

- **Citation:** Ruben Ohana, Michael McCabe, Lucas Meyer, Rudy Morel and others. *The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning.* Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track, pages 44989-45037. DOI 10.52202/079017-1430. arXiv:2412.00568.
- **Link:** https://arxiv.org/abs/2412.00568
- **Used for:** the source and benchmark of the public-dataset track.
- **Checked against:** arXiv metadata, the v2 PDF, the NeurIPS proceedings page, Crossref, and the project's GitHub README and licence file. The paper's table gives this dataset as 128×384, 101 steps and 90 trajectories. Its data licence statement says CC-BY-4.0. The repository's code is BSD 3-Clause.
- **Unverified:** the author count differs. arXiv lists 27, while the proceedings and Crossref list 26, with Keaton Burns absent. "Et al." avoids the problem, and a full author list should say which version it follows.
- **Wrong in design:** title shortened to "The Well". It is also not the citation the dataset card asks for. See "Citation for the public dataset" above.

## Sources missing from section 10

### 8. Multiphase Gas and the Fractal Nature of Radiative Turbulent Mixing Layers

- **Citation:** Drummond B. Fielding, Eve C. Ostriker, Greg L. Bryan and Adam S. Jermyn. *Multiphase Gas and the Fractal Nature of Radiative Turbulent Mixing Layers.* The Astrophysical Journal Letters 894(2), L24, 2020. DOI 10.3847/2041-8213/ab8d2c. arXiv:2003.08390.
- **Link:** https://doi.org/10.3847/2041-8213/ab8d2c
- **Used for:** the citation the dataset card asks for when `turbulent_radiative_layer_2D` is used.
- **Checked against:** the dataset card's BibTeX, Crossref, the arXiv PDF, and The Well paper, which cites it for these simulations.
- **Unverified:** where the 2D runs themselves are described. This paper reports a suite of three-dimensional simulations, and "2D" appears in it only for slices of them. Cite it because the card asks, but don't point to it for the 2D setup. Use the dataset card and The Well for that.
- **Wrong in design:** missing.

### 9. ANCHOR

- **Citation:** Rajyasri Roy, Dibyajyoti Nayak and Somdatta Goswami. *ANCHOR: Error-Controlled Adaptive Numerical Correction for Neural Operator Time Marching.* arXiv:2512.19643v2, 14 June 2026. v1 was titled *The Best of Both Worlds: Hybridizing Neural Operators and Solvers for Stable Long-Horizon Inference.*
- **Link:** https://arxiv.org/abs/2512.19643
- **Used for:** the closest prior work we found on the hand-off mechanism. A signal is tracked during the rollout, and passing a threshold runs the solver for a fixed number of steps before handing back.
- **Checked against:** the v2 PDF's abstract, method and setup passages, and arXiv metadata for both titles.
- **Unverified:** any peer-reviewed venue. None was found. The full paper was not read.
- **Wrong in design:** missing. The main difference from us is the signal, which is a moving average of the normalised PDE residual, so it needs the governing equation. It isn't an ensemble spread.

### 10. HyPER

- **Citation:** Bharat Srikishan, Daniel O'Malley, Mohamed Mehana, Nicholas Lubbers and Nikhil Muralidhar. *Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout.* ICLR 2025. arXiv:2503.10048.
- **Link:** https://arxiv.org/abs/2503.10048
- **Used for:** related work where a learned policy chooses the surrogate or the simulator at each step of a rollout, and is compared with a random policy that calls the simulator the same number of times.
- **Checked against:** the arXiv v2 PDF, whose header reads "Published as a conference paper at ICLR 2025". The abstract and the method and baseline passages were read.
- **Unverified:** the ICLR page itself was not opened. The full paper was not read.
- **Wrong in design:** missing. Its signal is a policy trained by reinforcement learning, not ensemble spread.

### 11. U-Net

- **Citation:** Olaf Ronneberger, Philipp Fischer and Thomas Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015, Lecture Notes in Computer Science, pages 234-241, 2015. DOI 10.1007/978-3-319-24574-4_28.
- **Link:** https://doi.org/10.1007/978-3-319-24574-4_28
- **Used for:** the origin of the U-Net backbone (design section 5.2).
- **Checked against:** Crossref metadata only.
- **Unverified:** the paper's text was not read. It is about image segmentation, so cite it for the architecture only.
- **Wrong in design:** missing, though U-Net is the ensemble's backbone.

### 12. ScalarFlow

- **Citation:** Marie-Lena Eckert, Kiwon Um and Nils Thuerey. *ScalarFlow: a large-scale volumetric data set of real-world scalar transport flows for computer animation and machine learning.* ACM Transactions on Graphics 38(6), 2019. DOI 10.1145/3355089.3356545. arXiv:2011.10284.
- **Link:** https://ge.in.tum.de/publications/2019-scalarflow-eckert/
- **Used for:** the record of a rejected dataset (design section 4, `docs/DECISIONS.md` D11).
- **Checked against:** Crossref metadata, the arXiv record, the group's 2019 announcement, and its current publication page.
- **Unverified:** whether a sign-up is still needed. The 2019 announcement says "You can request a download link via the following sign-up form". The current page says "The full data set is ca. 450 GB in size, but can be downloaded in parts" and links to a university repository, which refused automated access.
- **Wrong in design:** missing from section 10, and "needs a sign-up form" may be out of date. Being 3D and large still holds, and either reason is enough to reject it for this project.

## What we looked for and did not find

A search for work where ensemble spread triggers a solver hand-off in the middle of a neural rollout, with the network then resuming, found no such paper. The search was limited to a few web queries and the related-work section of entry 1. It doesn't show that none exists, so the report shouldn't call this approach new.

One lead is still open. Entry 9 says solver intervention after a fixed number of surrogate steps is common practice, citing Oommen et al. 2022 (npj Computational Materials 8(1):190) and Ovadia et al. 2025 (Computer Methods in Applied Mechanics and Engineering 441:117982). Neither has been opened. They bear on our fixed-schedule baseline, so check them before the report cites that baseline as prior practice.
