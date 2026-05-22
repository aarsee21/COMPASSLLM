# Claude PPT Instruction File for COMPASSLLM

## Purpose
This file is the only instruction input to give to Claude for generating a professional, top-tier conference-style PPT outline and slide content for the COMPASSLLM project.

The generated output must include the following sections:
- Introduction
- Related Work
- Research Gap
- Problem Statement
- Objectives
- Methodology
- Results
- Conclusion
- Limitations
- Future Scope

The tone must be formal, detailed, technically accurate, and appropriate for venues such as CVPR, ICSA, ICSE, or ECSA.

---

## Project Context and Code References
Project name: `COMPASSLLM`.

This is an implemented system with a real backend and frontend. Use the code references below as the basis for the presentation.

### Code-level architecture
- `backend/main.py`: FastAPI app setup, router registration, and auto table creation.
- `backend/routes/dataset_routes.py`: `POST /datasets/upload` handles CSV upload, validation, preview, and dataset metadata insertion.
- `backend/routes/analysis_routes.py`: `POST /datasets/analyze` runs analysis, stores selected target and excluded columns, and writes dataset meta-features to `dataset_features`.
- `backend/routes/recommendation_routes.py`: `POST /models/recommend` reads dataset metadata and calls `services/gemini_service.py`.
- `backend/routes/training_routes.py`: `POST /models/train` performs selected model training, writes experiments, and creates knowledge base entries; it also supports `GET /models/download/{artifact_id}` and `GET /models/knowledge-base`.
- `backend/services/feature_extraction_service.py`: computes meta-features used for recommendation, including missing ratio, imbalance, numeric/categorical counts, and column profiles.
- `backend/services/gemini_service.py`: constructs the Gemini prompt, optionally uses RAG with `instruction_files/*.txt`, and returns exactly three model names plus concise reasoning.
- `backend/services/model_training_service.py`: selects, trains, evaluates, and serializes models; uses `select_and_train_models` and `train_models` loops over candidate models and final shortlist.
- `backend/services/knowledge_base_service.py`: saves recommendation outcomes, top recommendation success, and system guidance list.
- `backend/services/dataset_service.py`: normalizes excluded columns, validates the target column, persists CSV data into `backend/uploads/`, and reloads datasets.
- `backend/database/models.py`: defines tables `datasets`, `dataset_features`, `recommendations`, `model_artifacts`, and `knowledge_base_entries`.

### Data flow and persistence
- Dataset upload stores the CSV file under `backend/uploads/` and metadata in `datasets`.
- Analysis stores `target_column`, `excluded_columns`, and numeric summary data in `dataset_features`.
- Recommendation uses persisted dataset analysis plus sampled rows to build the Gemini prompt.
- Training uses `select_and_train_models` with candidate scoring, then stores experiments and artifact paths.
- Knowledge base entries record whether the top recommended model matched the final best model.

### Important implementation observations
- The recommendation prompt builder in `backend/services/gemini_service.py` includes dataset characteristics, sample rows, and optional user instruction.
- If the Gemini API key is not configured or the response fails, the system falls back to `Random Forest`, `XGBoost`, and `Support Vector Machine`.
- `backend/services/model_training_service.py` trains candidate models using a ColumnTransformer with median imputation, scaling, one-hot encoding, and a model step.
- The candidate pool includes models such as Logistic Regression, Random Forest, Decision Tree, SVM, KNN, GaussianNB, XGBoost, LightGBM, CatBoost, and Gradient Boosting.
- Training evaluation uses cross-validation when possible; otherwise it falls back to a stratified or non-stratified train/test split.
- Artifacts are serialized to `.joblib` and exposed via a download API.
- The knowledge base rulebook is a fixed guidance list in `backend/services/knowledge_base_service.py`.

---

## Detailed Slide Requirements
For every slide, provide:
1. a clear slide title,
2. 5–8 explicit bullet points,
3. one concise speaker note sentence when relevant.

### 1. Introduction
- Define the broader engineering problem: selecting classification models for tabular data in software systems.
- Explain why tabular model selection is still resource-intensive and often arbitrary in practice.
- Introduce COMPASSLLM as a full-stack system that integrates dataset profiling, LLM recommendation, and training.
- Name the stack: React + TypeScript frontend, FastAPI backend, PostgreSQL storage, pandas/numpy analysis, scikit-learn/XGBoost training, Gemini LLM.
- Stress the focus on sustainability: reducing wasted compute and repeatable model selection decisions.
- Speaker note: highlight the product-like nature of the system, not just a research prototype.

### 2. Related Work
- Summarize AutoML and algorithm selection systems such as auto-sklearn, H2O AutoML, and TPOT.
- Describe meta-feature and meta-learning approaches used historically for algorithm selection.
- Mention recent work that uses large language models for software engineering reasoning and decision support.
- Explain how knowledge-base-driven recommendation systems provide persistent evidence and feedback.
- Position COMPASSLLM as an integrative system that combines dataset meta-features, LLM reasoning, and recordable outcomes.
- Speaker note: emphasize that the contribution is system-level integration across multiple ideas.

### 3. Research Gap
- State that many AutoML systems still use brute-force search or static heuristics instead of data-aware recommendations.
- Note the lack of systems that persist the LLM's reasoning together with training outcomes.
- Highlight the missing end-to-end path from dataset analysis to downloadable model artifacts.
- Explain that existing systems rarely support explicit feature exclusion and user-guided recommendations on tabular datasets.
- Emphasize the need for a pipeline that can iteratively improve with recorded experience.
- Speaker note: frame the gap as a practical engineering challenge in addition to a research problem.

### 4. Problem Statement
- Define the problem: recommending a small set of classification models for a new tabular dataset with minimal waste.
- Specify the constraints: mixed numeric/categorical features, missing values, class imbalance, and variable dataset sizes.
- State the objective: use dataset meta-features plus LLM guidance to produce robust candidate model recommendations.
- Clarify the desired system properties: repeatability, traceability, and artifact exportability.
- Include the operational requirement: support target selection and column exclusion before recommendation.
- Speaker note: make it clear this is a tool for engineering teams, not abstract model research.

### 5. Objectives
- Build a full-stack recommendation pipeline for tabular classification datasets.
- Support dataset analysis and explicit preprocessing choices in `backend/routes/analysis_routes.py`.
- Create an LLM recommendation endpoint in `backend/routes/recommendation_routes.py` using `gemini_service.py`.
- Implement model training, metric recording, and artifact serialization in `model_training_service.py`.
- Persist recommendation outcomes in `knowledge_base_service.py` for future improvement.
- Ensure safe fallback behavior when the Gemini LLM response is unavailable.

### 6. Methodology
- Describe the architecture: frontend handles upload and UI, backend routes handle data operations, services contain business logic.
- Explain dataset profiling in `feature_extraction_service.py`: row/feature counts, numeric/categorical counts, missing-value ratio, class imbalance, and column profiles.
- Detail column exclusion and target selection in `dataset_service.py` and `analysis_routes.py`.
- Explain recommendation generation: sample rows are selected, excluded columns are filtered, and `gemini_service.py` builds a prompt with instruction documents.
- Describe model selection and training in `model_training_service.py`: candidate scoring, cross-validation, train/test split, and `.joblib` artifact creation.
- Mention the knowledge base: `create_knowledge_base_entry` records recommendation list, reasoning, best model, and whether the top recommendation worked.
- Speaker note: cite the concrete files and routes to underscore the implemented engineering flow.

### 7. Results
- Describe the implemented end-to-end flow in the codebase from upload to model download.
- List key implemented capabilities: CSV upload, dataset analysis, target/exclusion support, LLM recommendation, training pipeline, dashboard persistence, artifact download.
- Mention that the recommendation endpoint persists both model suggestions and reasoning text in `recommendations` table.
- Explain validation through smoke tests and the existence of persisted knowledge base entries after training.
- Clarify that current results are based on system functionality, not a completed benchmarking study.
- Speaker note: discuss what is delivered today and what remains for formal evaluation.

### 8. Conclusion
- Summarize the contribution: a deployed system that merges dataset profiling, LLM recommendation, and model training.
- Highlight the practical value of combining data-aware analysis with LLM reasoning and persistent tracking.
- Stress the engineering impact: reducing blind experimentation, improving repeatability, and delivering artifacts.
- Note that the system exposes `GET /models/download/{artifact_id}` and knowledge base support.
- State that this implementation is a strong foundation for future research and system refinement.
- Speaker note: present the work as a practical, conference-ready engineering contribution.

### 9. Limitations
- Acknowledge the current absence of curated benchmark datasets and explicit cross-domain evaluation.
- Note that the repo does not yet compute recommendation KPIs such as hit rate, best-model accuracy, or latency.
- Mention dependence on the external Gemini LLM service and the fallback recommendation path.
- Point out that automatic continuous tuning of prompts and knowledge-based retrieval is not yet implemented.
- Clarify that the current scope is system integration and readiness, not final production optimization.
- Speaker note: be transparent to strengthen credibility.

### 10. Future Scope
- Propose retrieval-augmented recommendations using historical knowledge base entries.
- Suggest building an offline benchmark runner to compare recommendation quality over time.
- Recommend adding KPIs: recommendation hit rate, average best-model accuracy, model selection latency, and feedback improvement.
- Advise adding reproducibility metadata, dataset ownership controls, and role-based access.
- Recommend expanding support to more candidate models and automated prompt tuning.
- Speaker note: position these as logical next steps for research and productization.

---

## Output Requirements
- Generate a slide deck outline with a title and explicit bullets for every required section.
- Each slide should be granular, specific, and grounded in the actual backend implementation.
- Avoid vague language and generic assertions.
- Do not invent experimental metrics or evaluation numbers.
- Keep phrasing professional, conference-appropriate, and precise.
- If a closing slide is included, label it `Closing / Q&A`.

---

## Notes for Claude
- Use the code references and backend flow described above as the authoritative model of the system.
- Focus on engineering contributions, end-to-end integration, and practical outcomes.
- Do not include code snippets or internal API details.
- Avoid claims not supported by the current repository implementation.
