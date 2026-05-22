# PPT Results — Model Performance Summary

This file extracts aggregated per-model metrics (from backend/instruction_files) and formats them as slide-ready bullets and short speaker notes.

---

**Slide: Random Forest (RandomForestClassifier)**
- Total runs: 6,405
- Mean accuracy: 0.90
- Median accuracy: 0.93
- Std (accuracy): 0.06
- Best conditions: numerical-dominant datasets, low-dimensional (2–3 features), balanced classes
- Worst conditions: noisy symmetry/smoothness feature combos, imbalanced targets without class_weight
- Speaker note: RF is a strong default for mixed-type tabular data with a few high-signal features; watch for overfitting when trees are unbounded and for class imbalance.

Source: [backend/instruction_files/RandomForestClassifier_instructions.txt](backend/instruction_files/RandomForestClassifier_instructions.txt)

---

**Slide: CatBoost (CatBoostClassifier)**
- Total runs: 6,404
- Mean accuracy: 0.87
- Median accuracy: 0.90
- Std (accuracy): 0.10
- Mean precision: 0.862, mean F1: 0.860
- High-accuracy runs (>=0.95): 1,090 (mostly breast-cancer diagnosis)
- Best conditions: 3-feature subsets, numerical or high-signal categorical features, balanced datasets
- Worst conditions: single-feature subsets, imbalanced targets, weak categorical-only spaces
- Speaker note: CatBoost shows excellent gains with 2–3 informative features and handles categorical signals well; imbalance reduces performance ~7–8%.

Source: [backend/instruction_files/CatBoostClassifier_instruction.txt](backend/instruction_files/CatBoostClassifier_instruction.txt)

---

**Slide: SVM (SVC)**
- Total runs: 6,405
- Mean accuracy: 0.86
- Median accuracy: 0.91
- Std (accuracy): 0.10
- Best conditions: scaled, low-dimensional numerical datasets (2–3 features)
- Worst conditions: unscaled data, noisy or high-dimensional feature sets
- Speaker note: SVC achieves high median accuracy when preprocessing (scaling) and feature selection are applied; computational cost grows with dataset size.

Source: [backend/instruction_files/SVC_Instructions.txt](backend/instruction_files/SVC_Instructions.txt)

---

**Slide: Logistic Regression**
- Total runs: 6,405
- Mean accuracy: 0.81
- Median accuracy: 0.84
- Std (accuracy): 0.09
- Best conditions: scaled numerical low-dimensional datasets (biomedical)
- Worst conditions: noisy features, non-linear class boundaries
- Speaker note: LR is reliable on linearly separable low-D problems; use when interpretability and simplicity are priorities.

Source: [backend/instruction_files/LogisticRegression_instructions.txt](backend/instruction_files/LogisticRegression_instructions.txt)

---

**Slide: Decision Tree (DecisionTreeClassifier)**
- Total runs: 6,405
- Mean accuracy: 0.83
- Median accuracy: 0.87
- Std (accuracy): 0.11
- Mean precision: 0.830, mean F1: 0.827
- High-accuracy runs (>=0.95): 383 runs
- Best conditions: 3-feature subsets, balanced datasets, high-signal features
- Worst conditions: single-feature weak signals, noisy elemental feature combos
- Speaker note: Decision Trees are interpretable and perform well on structured data, but pruning/regularisation is recommended to avoid overfitting on noisy features.

Source: [backend/instruction_files/DecisionTreeClassifier_instruction.txt](backend/instruction_files/DecisionTreeClassifier_instruction.txt)

---

**Notes & Next Steps**
- More per-model instruction files exist (e.g., LightGBM, XGBoost, GaussianNB, KNN). If you want those included, I can extract them similarly.
- If you prefer dataset-specific experiment rows instead of aggregated instruction-file stats, I can fetch experiments from the running backend or query the DB and produce per-dataset slides.

Files used:
- [backend/instruction_files/RandomForestClassifier_instructions.txt](backend/instruction_files/RandomForestClassifier_instructions.txt)
- [backend/instruction_files/CatBoostClassifier_instruction.txt](backend/instruction_files/CatBoostClassifier_instruction.txt)
- [backend/instruction_files/SVC_Instructions.txt](backend/instruction_files/SVC_Instructions.txt)
- [backend/instruction_files/LogisticRegression_instructions.txt](backend/instruction_files/LogisticRegression_instructions.txt)
- [backend/instruction_files/DecisionTreeClassifier_instruction.txt](backend/instruction_files/DecisionTreeClassifier_instruction.txt)

Saved as `PPT_RESULTS.md` at repository root.
