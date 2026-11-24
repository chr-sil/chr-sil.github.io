+++
title = "Revisiting PaiNN + FiLM"
date = 2026-10-02
description = "A retrospective on the choices behind a supercritical-CO₂ solubility model: what holds up, and what I would do differently."
[extra]
lang = "en"
subtitle = "A retrospective on the choices behind a supercritical-CO₂ solubility model"
mermaid = true
note = "Written in October 2026 during peer review, before publication. The <a href=\"/research/scco2-solubility-painn-film/\">published version</a> may differ."
+++

## Why this post

A few months ago I helped scope and steer a project to predict analyte solubility in supercritical CO₂ (scCO₂) from molecular structure and thermodynamic conditions — an equivariant graph neural network (PaiNN), conditioned on temperature, pressure, density and a handful of other thermodynamic descriptors via FiLM, with a walk-based relevance-propagation explainer bolted on top so the predictions aren't just a black box. The paper is now submitted and sitting with two reviewers.

I was involved in the decisions as they were being made — the architecture choice, the conditioning mechanism, what the explainer needed to show — but I didn't write every line, and some of the engineering judgment calls happened without me in the room. Now that the codebase has landed back in my lap and reviewers have started poking at it, I wanted to go back through it properly: not "does it work," but *how* does it work, which of the choices I'd defend today, and which ones I'd do differently if I were starting over. This is that write-up.

Fair warning: this isn't a polished "we built X and it's great" post. It's closer to an audit. I found things I'm proud of, and I found at least one issue serious enough that it changes how much I trust the paper's headline number. I'd rather write that down honestly than pretend the retrospective found nothing.

## The problem, briefly

scCO₂ is an attractive green solvent for sample preparation and extraction — it's non-toxic, evaporates cleanly on decompression, and avoids the organic-solvent waste that traditional extraction methods generate. The catch is that scCO₂ only exists above 31°C and ~74 bar, and measuring how well a given molecule dissolves under those conditions is slow, equipment-heavy, and — per the literature we cited in the abstract — disagrees across labs by as much as two orders of magnitude for the same compound. If you could predict solubility in silico with any confidence, you could skip a lot of that experimental overhead before ever touching a reactor.

The existing options were a trade-off: semi-empirical equations (Chrastil-style) that are cheap but don't transfer to new molecules well, versus a GNN from Makarov et al. (2025) that generalizes better but doesn't explain *why* it predicts what it predicts. We wanted both: a model that generalizes to unseen chemistry **and** one where you can point at a prediction and ask "which atoms drove this, and does that match known chemistry."

## Data: curation, 3D structure, and features

<pre class="mermaid">
flowchart TB
    A["Makarov et al. 2025&lt;br&gt;raw compiled dataset&lt;br&gt;30,771 rows"] --&gt; B["Fragment handling&lt;br&gt;(drop salts / counter-ions,&lt;br&gt;metal-free organic fragment)"]
    B --&gt; C["Chemistry filters&lt;br&gt;(scCO2 conditions only,&lt;br&gt;no metals, MW 50-700/1000 Da,&lt;br&gt;neutral charge, &gt;=1 carbon)"]
    C --&gt; D["RDKit 2D parse"]
    D --&gt; E["ETKDG 3D embedding&lt;br&gt;+ MMFF94s / UFF fallback"]
    E --&gt; F["Node / edge / global&lt;br&gt;feature calculation"]
    F --&gt; G["molecule_data_3D.pkl&lt;br&gt;958 molecules, 27,461 rows&lt;br&gt;(93.3% retained)"]
</pre>

The dataset builds on Makarov et al.'s compiled scCO₂ solubility measurements. Before anything touches RDKit, the curation notebook has to deal with the usual mess of real-world chemical data: multi-component SMILES (salts, co-crystals, solvent adducts) get fragmented and the largest metal-free organic piece is kept; subcritical measurements (below 304 K / 74 bar) get dropped; anything containing a transition/lanthanide/actinide metal is excluded because coordinative binding doesn't translate cleanly into the 3D structural input the model needs; and a molecular-weight window (50–700/1000 Da depending on the notebook variant) keeps the dataset in the "small organic molecule" regime SFE actually targets.

3D structure generation is genuinely solid engineering: explicit hydrogens are added, ETKDG embedding is attempted with a deterministic seed (`seed + attempt * 1337`, up to three attempts), MMFF94s force-field optimization is tried first with a UFF fallback, and coordinates are checked for NaN/Inf both before and after optimization. Only 3 molecules out of 958 fell back to UFF and only 2 failed outright. That's the kind of defensive conformer-generation code I'd want in any 3D-structure pipeline — most quick scripts skip the NaN/Inf checks entirely.

What I'd flag, looking back:

- **The MW/charge/carbon filter was tuned by looking at a previous model's errors**, not decided a priori. The markdown commentary is honest about this — it cites actual MAE numbers ("~0.8 vs ~0.5 for extreme MW," "0.91 vs 0.51 for ionic compounds") as the justification for the thresholds. That's defensible domain reasoning, but it means the paper's "domain of applicability" was partly *fit to* where an earlier version of the model already struggled, rather than defined independently of model performance. It's not test-set leakage in the strict sense, but it's adjacent to it, and it's worth being candid about rather than presenting the filter as a purely a-priori chemistry decision.
- **No shared module across notebooks.** Every notebook that needs to build a graph re-implements feature engineering, vocabulary construction, and caching from scratch. I noticed near-identical ~200-line helper blocks copy-pasted between the two 3D-generation notebooks (main dataset and the Makarov-comparison variant). This is a known shape for notebook-first research code, but it's exactly the kind of thing that lets a fix applied in one place silently not propagate to its siblings.
- **Hardcoded, machine-specific paths** are baked into committed cell outputs (`d:\GIT_repository\...`). Nobody can re-run this notebook today without manually editing the input/output paths first — there's no config file, no CLI args, nothing parameterized.
- **No RDKit or Python version pinned anywhere in the notebook**, despite ETKDG and MMFF94s both being version-sensitive. This is one of the concrete things a reviewer flagged, and it's a fair catch.

## Architecture: PaiNN, and what we changed

PaiNN (Polarizable Atom Interaction Neural Network, Schütt et al. 2021) was the right starting point for a structure-aware solubility model because it keeps *two* parallel per-atom representations — an invariant scalar channel and an equivariant vector channel — instead of collapsing everything to rotation-invariant distances the way SchNet does. That distinction matters here for a concrete reason: distance-only models can't distinguish a molecule from its mirror image, because pairwise distances are identical for a chiral pair. A vector channel that rotates *with* the molecule can.

<pre class="mermaid">
flowchart TB
    subgraph Layer["One interaction layer"]
        direction TB
        S["scalar state s_i"] --&gt; MSG["message:&lt;br&gt;Φ_s(s_j) ⊙ W(‖r_ij‖)"]
        V["vector state v_i"] --&gt; MSGV["message:&lt;br&gt;direction + propagation terms"]
        MSG --&gt; FILM1["FiLM (message stage)&lt;br&gt;γ_m·x + β_m"]
        FILM1 --&gt; UPD["update:&lt;br&gt;scalar + vector mixing,&lt;br&gt;scalar product ⟨Uv,Vv⟩"]
        MSGV --&gt; UPD
        UPD --&gt; FILM2["FiLM (update stage)&lt;br&gt;γ_u·x + β_u"]
        FILM2 --&gt; Sout["s_i (next layer)"]
        UPD --&gt; Vout["v_i (next layer), unconditioned"]
    end
    Cond["Global thermo features&lt;br&gt;(T, P, ρ, ΔHvap, Tc, Pc, MP, log MW)"] -.-&gt;|"small MLP → γ,β&lt;br&gt;layers ≥ condition_after_layer only"| FILM1
    Cond -.-&gt; FILM2
</pre>

The two architectural additions on top of vanilla PaiNN are FiLM conditioning and the richer feature set, and the reasoning for both holds up:

- **FiLM instead of super-nodes.** Pressure and temperature are properties of the *experiment*, not the molecule — there's no atom to hang them on. A "super-node" representing the global condition is a common pattern in the literature, but it collides with wanting a physically faithful molecular graph. FiLM sidesteps that by rescaling the scalar message/update at each conditioned layer (`out = γ·x + β`, with γ and β produced by small MLPs from the global features) — and critically, it's applied **only to the scalar channel, never the vector channel**, which is the detail that keeps the whole model equivariant. Conditioning only kicks in from a tunable layer onward (`condition_after_layer`), so early layers learn pure structural representations before any thermodynamic signal is injected.
- **Richer, chemistry-aware node/edge features.** With only ~1,000 unique molecules, leaning on hand-crafted functional-group flags (amines, carboxylates, alcohols, etc.), Gasteiger/MMFF charges, and bonded-vs-non-bonded edge separation is a reasonable way to buy data efficiency that a from-scratch embedding wouldn't get for free.
- **Residual update instead of overwrite**, and a Hadamard-gated message function — both lifted from the original PaiNN paper, but combined with enough depth (4 layers) that the residual form matters for stable gradient flow.

### Which notebook is actually "the" model?

This is the part of the retrospective I didn't expect going in, and it's the single most important finding of this whole exercise.

The repo has two notebooks that each define a full `HybridConditionedPaiNN`. `PaiNN_Training_wo_Optuna.ipynb` is named, and described in the repo's own documentation, as *the* clean, reproducible run — the one with the best hyperparameters hardcoded in, meant to be the reference implementation. `PaiNN_Training_scaffoldsplit.ipynb` is nominally just "the same model, but split by scaffold instead of by molecule."

They are not the same model. Side by side:

| | `PaiNN_Training_wo_Optuna` | `PaiNN_Training_scaffoldsplit` (matches the paper) |
|---|---|---|
| Layers | 3 | **4** |
| Hidden / vector dim | 197 / 82 | **182 / 50** |
| Message function | edge features + distance only — **never reads the neighbor's scalar state** | factorized `φ_s(s_j) ⊙ filter(r_ij)`, matching PaiNN's actual message equation |
| Vector message | direction-injection term only | both terms (direction injection **and** existing-vector propagation) |
| Global features used | 3 of 8 (comment literally reads `# OHNE DRUCK!` — "without pressure!") | all 8, matching the paper |
| FiLM | single-stage | dual-stage (message *and* update), matching the supplementary description |
| Pooling | plain attention pooling | hybrid (attention + sum), matching the paper |
| Scheduler | ReduceLROnPlateau | Cosine annealing + warmup, matching the paper |
| Cross-validation | none — single 80/10/10 split | 5-fold, scaffold-grouped |

The hyperparameter values in the scaffold-split notebook are the ones that actually match the manuscript's supplementary text (182/50 dimensions, 0.03/0.21 Huber deltas, the full thermodynamic feature set with inline comments citing Makarov's own SHAP feature ranking). The "reproducibility" notebook implements something closer to a SchNet-style continuous-filter convolution than the PaiNN architecture the paper describes.

If I'd tried to reproduce the published number from the notebook that's *supposed* to be the clean reference implementation, I would have silently gotten a different model and a different score. That's not a reviewer-facing problem — it's a "someone picks this repo up in a year and wastes a day" problem, and it's on me to fix the naming and the config before this goes any further. I also found an `EPOCHS=5` override sitting a few cells after an `EPOCHS=500` declaration in that same notebook — I don't know yet whether that's leftover smoke-test code or whether it means the cached outputs in that notebook reflect a five-epoch run. Either way, it needs checking, not assuming.

One more loose end: a `PaiNN_Training_wo_Optuna.py` file — a plain-Python export of the notebook — existed as an untracked file at the start of this audit and is gone by the end of it. I don't know if that was an in-progress fix that got deleted, moved, or never actually saved. If there was work in there addressing any of this, it's worth tracking down before assuming it's lost.

On the equivariance front specifically, the simplified notebook also has a `v_init` parameter — a single learnable vector shared across every node, initialized to zero — with nothing constraining it to *stay* near zero during training. If gradients push it away from zero, it becomes a fixed directional bias that doesn't rotate with the molecule, which quietly breaks the equivariance guarantee the whole two-channel design exists to provide. I haven't confirmed whether this actually drifts in practice, but it's a concrete thing to check rather than assume away.

## Hyperparameter optimization — and a real validation problem

This is the section I went in least worried about and came out most worried about.

The manuscript describes a nested cross-validation scheme specifically for the "3D-structure" hyperparameters (number of radial basis functions, RBF cutoff, non-bonded radius, non-bonded max neighbors): three outer folds, three inner folds, 80 trials. **That code doesn't exist anywhere in the current repo.** Both `PaiNN_Optuna.ipynb` and `PaiNN_Training.ipynb` treat those four parameters as fixed constants, explicitly commented as "pre-optimized" (`FIXED_3D_PARAMS`). Either the nested-CV run happened in a notebook version that never got committed, or it happened interactively and the record of it is gone. Either way, I can't currently verify from this repo that the step the manuscript credits with avoiding leakage for those four parameters actually did.

The much bigger issue is in the hyperparameter search that *is* in the repo — the 200-trial Optuna search over architecture and training hyperparameters (hidden dims, layer count, learning rate, dropout, `condition_after_layer`, and so on).

<pre class="mermaid">
flowchart TB
    subgraph HPO["Architecture search (200 trials)"]
        D1["Full dataset"] --&gt; F1["3-fold CV split&lt;br&gt;(seed=42)"]
        F1 --&gt; O["Optuna objective =&lt;br&gt;mean RMSE across all 3 folds"]
        O --&gt;|"every molecule validates&lt;br&gt;in exactly one fold"| D1
    end
    subgraph Final["Final 5-fold evaluation"]
        D2["Full dataset"] --&gt; F2["5-fold CV split&lt;br&gt;(same seed=42, same&lt;br&gt;pre-shuffle order)"]
        F2 --&gt; TR["train on 80%"]
        TR --&gt; EP["each epoch:&lt;br&gt;evaluate on the 20% val fold"]
        EP --&gt;|"lowest val RMSE&lt;br&gt;epoch wins"| CKPT["best checkpoint"]
        CKPT --&gt; REPORT["report val-fold metrics&lt;br&gt;for this exact fold&lt;br&gt;as the OOF result"]
    end
    HPO -.-&gt;|"chosen architecture already&lt;br&gt;informed by signal from&lt;br&gt;every molecule"| Final
</pre>

Two separate problems compound here:

1. **The architecture search's objective is a 3-fold CV mean, which touches 100% of the dataset.** This matches the manuscript's own phrasing almost exactly — "the complete dataset is used" for the refined search region — because every molecule serves as validation data in one of the three folds, and Optuna is choosing hyperparameters to minimize the *average* of those three. That means the architecture itself was selected with knowledge of how well it does on every molecule, before any "held-out" 5-fold evaluation ever runs. And the final 5-fold split reuses the identical seed and shuffle order as the 3-fold HPO split, so fold membership between the two schemes is correlated, not independent.
2. **Early stopping and final reporting read the same validation fold.** I checked the actual training loop: the checkpoint with the lowest validation-fold loss is kept, and *that same validation fold* is what gets evaluated and recorded as the fold's out-of-fold result. There's no third split — no internal training-only validation slice used purely for early stopping, separate from the fold being reported. That's picking your best epoch based on performance on the exact data you then report performance on. It isn't a documentation ambiguity; one of the reviewers raised exactly this concern, and tracing the code confirms they're right.

I don't think this is evidence of carelessness across the board — elsewhere in the codebase, feature scaling is explicitly computed per-split with a code comment calling out the leakage risk by name ("no global scaler — prevents data leakage"). Whoever wrote that was thinking about this exact failure mode. It just didn't get applied consistently to the validation-fold/early-stopping design, which tells me this was a specific blind spot, not a systemic one.

What this means practically: the headline R² = 0.767 is very likely optimistic by some amount I can't quantify without re-running the experiment with a genuine three-way split (train / early-stopping validation / held-out test). That's not a cosmetic fix — it's a re-run, and I think it's the right thing to do regardless of how the review process goes.

## Making the model explain itself

The explainability stack is the part of this project I'm most proud of, and also the part with the most delicate implementation detail.

The core idea is layer-wise relevance propagation (LRP) adapted into a *walk-based* method: instead of attributing relevance to a node in one shot, you enumerate paths ("walks") through the message-passing layers and accumulate relevance along each one. For a 4-layer network that's potentially a huge number of walks per molecule, so there's a hard cap of 500,000 — and that cap is enforced by just stopping enumeration once you hit it, in whatever order the underlying loops happen to visit walks. There's no importance-based prioritization, so for large, high-degree molecules, the walks you keep aren't guaranteed representative. That interacts with something the paper already reports: fold 4, which has the largest molecules in its validation split, also shows the worst "conservation" (the fraction of relevance that traces back to actual input atoms rather than evaporating into the FiLM bias term). I'd want to know how much of that fold-4 effect is "truly harder molecules" versus "truncated-walk sampling artifact," and right now I can't tell those apart.

<pre class="mermaid">
flowchart LR
    P["Prediction"] --&gt; H["Head MLP"]
    H --&gt; FR["FiLM readout&lt;br&gt;γ_r · g (β_r excluded —&lt;br&gt;untraceable to any node)"]
    FR --&gt; Pool["Attention + sum pooling"]
    Pool --&gt; L4["Layer 4 (conditioned)"]
    L4 --&gt; L3["Layer 3 (conditioned)"]
    L3 --&gt; L2["Layer 2 (unconditioned)"]
    L2 --&gt; L1["Layer 1 (unconditioned)"]
    L1 --&gt; Atoms["Per-atom relevance&lt;br&gt;(walk-enumerated, capped at 500k)"]
</pre>

The FiLM β term (the additive half of `γ·x + β`) is handled the only way it honestly can be: it's explicitly subtracted out before computing relevance shares, because an additive constant from outside the graph has no atom to attribute it to. That's the source of the "conservation" loss the paper reports — roughly 10% per model head, compounding to a target of about 0.73 with two proxy heads plus the main head. What's worth knowing is that this target isn't computed from a formula at runtime; the per-fold conservation numbers used to validate it are hardcoded from a specific prior run. If the model gets retrained, that table goes stale silently — there's no check that recomputes and compares.

I also found a mismatch between what the manuscript says and what the code that produces the headline faithfulness numbers actually does. The text describes the ablation approach as "soft masking — setting features to zero" at readout. The mode actually driving the reported faithfulness scores masks atoms through the *attention pooling* layer (giving masked atoms a `-inf` attention logit), not by zeroing their features. The zero-feature approach exists in the code too, but only as a secondary diagnostic, not the one behind the published numbers. It's a small thing, but it's exactly the kind of detail a careful reader comparing paper to code would catch, and I'd rather fix the description than wait for someone to ask.

What I'd defend without reservation: the masking-at-readout-only design (rather than literally deleting a node from the graph) is a deliberate, well-reasoned choice — removing a node outright would still leak its influence into neighbors via message passing from earlier layers, so the authors accepted a documented *underestimate* of the faithfulness score rather than build an artificial, out-of-distribution ablated graph. And there's a self-test cell in the explainer notebook that raises an `AssertionError` if a correctness regression creeps back in — that's unusual rigor for notebook-first research code, and it shows real debugging history (several comments reference a specific earlier bug in how relevance propagated through the pooling layer that got fixed and then guarded against).

## Results, comparisons, and the numbers that don't match

Three things came out of tracing the metrics code that are worth separating clearly, because they're different in kind.

**The R² formula in the manuscript is a typo, not a bug.** Equation 7 as printed in the paper is the in-sample explained/total-variance ratio, which is only valid for a least-squares fit on the same data it was fit to — wrong for anything out-of-fold or external. But every actual computation in the codebase — the model comparison notebook, the external-validation script, the figure-generation notebook — uses the correct predictive form (`1 − Σ(y−ŷ)²/Σ(y−ȳ)²`, via `sklearn.metrics.r2_score` or an equivalent manual implementation). The authors clearly know the right formula; it just didn't survive the transcription into LaTeX. No numbers need to change — just the equation.

**The external-validation number mismatch (R² = 0.838 in the figure vs. 0.77 in the conclusion, for nominally the same 251-vs-252-point set) has at least a partial, concrete explanation.** The figure-generation code silently drops any row with a non-finite prediction before computing its metrics; if exactly one of 252 external predictions came back NaN, that alone produces the n=251 we see in the figure. The script that produces the conclusion's numbers, by contrast, has no such guard — which means those two numbers didn't come from the same pass over the same data under the same code path, even if they're supposed to represent the same experiment. I couldn't pin down the exact provenance of the conclusion's specific numbers without the original run artifacts, but the mechanism causing the n-mismatch is clear, and it points at a broader problem: there are at least three independently-written R²/Q²F3 implementations in this codebase, and nothing forces them to agree on edge-case handling. Before responding to reviewers, I think the right fix isn't just reconciling these two numbers — it's consolidating to one script that produces every reported number, so this class of bug can't recur.

**The "our GNN is more precise than Makarov's" claim doesn't hold up as currently supported, and the comparison behind it is less controlled than the writing implies.** The actual comparison isn't a re-run of Makarov's model under an identical protocol — it's a join between PaiNN's own out-of-fold predictions and Makarov's *own, separately-generated* cross-validation output files, aligned by row after the fact. Framed that way, the 0.769-vs-0.767 gap (where the baseline is nominally *better*, not worse) is an artifact of two independently-run pipelines being stitched together, not a controlled apples-to-apples test — and there's no significance test anywhere in the repo backing the "more precise" claim regardless. I'd rather either run a real paired comparison with a significance test, or soften the claim to match what was actually measured.

For what it's worth, not every comparison choice has this problem — the same notebook explicitly drops water and iodine from *both* models' evaluation sets "for a fair comparison," which is a deliberate, disclosed adjustment in the right direction. The inconsistency isn't that fairness was ignored; it's that it wasn't applied evenly.

On the "external" validation set's independence from training data: the overlap check is exactly what it looks like in the paper — exact canonical-SMILES matching, nothing similarity-based. Given that several of the external compounds are steroid drugs and cholesterol esters that plausibly have close relatives in the training set, adding a Tanimoto/Morgan-fingerprint similarity check is a genuinely small change to make (the script already parses every molecule with RDKit; it's maybe 25–30 lines to add a max-similarity column), and I think it's worth doing regardless of whether the reviewers had asked.

## What I'd do differently

Pulling this all together, here's the honest list:

1. **Re-run the final evaluation with a real three-way split** — train, an internal validation slice for early stopping, and a genuinely untouched test fold. This is the one item on this list I'm not willing to call a "nice to have."
2. **Delete or fix `PaiNN_Training_wo_Optuna.ipynb`.** A reference-reproducibility notebook that doesn't reproduce the reference model is worse than no notebook at all.
3. **One canonical metrics function**, used everywhere a reported number comes from, with explicit non-finite-value handling that's logged rather than silent.
4. **Locate (or accept the loss of) whatever was in the untracked `.py` file** that vanished during this audit, before assuming any in-progress fix needs to be redone from scratch.
5. **Add the Tanimoto-similarity check** to the overlap script — small effort, meaningfully strengthens the "external" claim either way it comes out.
6. **Either genuinely re-run the Makarov comparison under a controlled, identical protocol with a significance test, or rewrite the comparison claim to match what a join-of-two-independent-runs can actually support.**
7. **Pin RDKit/PyTorch/Optuna versions and seeds in the paper itself**, not just in a notebook cell nobody outside the repo will ever see.

None of this changes my view that PaiNN + FiLM was the right architectural call, or that the explainability work is genuinely more rigorous than most "we added SHAP" GNN papers bother to be. But I came into this audit assuming the main risk was going to be writing-quality issues a copyeditor could fix, and I'm leaving it convinced that at least the validation-leakage finding needs compute, not just prose, before I'd stand fully behind the headline number. I'd rather say that here than only find it out from a third reviewer.
