+++
title = "Prediction of supercritical CO₂ solubility with a thermodynamics-conditioned equivariant graph neural network model"
date = 2026-10-14
description = "Solubility as conditional regression: an equivariant GNN (PaiNN) over 3D molecular structure, conditioned on thermodynamic state via FiLM. R² = 0.77 on unseen molecules, 0.71 on unseen scaffolds, with relevance propagation to check that it learns chemistry rather than shortcuts."

[extra]
authors = ["Moritz Effner", "Christopher Siller", "Mijo Stanic", "Michael Lämmerhofer"]
venue = "Green Analytical Chemistry 19, 100395"
doi = "10.1016/j.greeac.2026.100395"
tags = ["Equivariant GNNs", "FiLM", "Explainability", "Molecular property prediction"]
+++

How well a molecule dissolves in supercritical CO₂ depends both on its structure and on the thermodynamic state of the solvent. We treat it as a conditional regression problem. A rotation-equivariant message-passing network (PaiNN) encodes the 3D conformer, and Feature-wise Linear Modulation (FiLM) injects global state descriptors into the message passing: temperature, pressure, density, evaporation enthalpy and critical point.

The model is trained on 27,461 measurements across 958 molecules. In five-fold cross-validation it reaches R² = 0.77 (±0.05) on held-out molecules. Under a scaffold split, where the test molecules share no core structure with the training set, it reaches 0.71. To check that the model reasons chemically, we adapted walk-based GNN layer-wise relevance propagation to the FiLM architecture and added structural ablations. Atom attributions are consistent across folds (ρ = 0.72) and follow known trends with molecular weight, polar surface area and acetylation.
