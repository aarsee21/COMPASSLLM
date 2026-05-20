"""
Fix script for COMPASSLLM_metadata_engine.ipynb.
Applies ALL patches from the original .bak so this script is idempotent.

Fixes:
1. Import timezone from datetime + uuid as _uuid_module + math as _math_module
2. OneHotEncoder(sparse=False) -> OneHotEncoder(sparse_output=False)
3. Remove deprecated use_label_encoder=False from XGBClassifier
4. Replace datetime.utcnow() with datetime.now(timezone.utc)
5. Add _sanitize_for_json + _safe_json_dumps that handle:
   - uuid.UUID -> str
   - float Inf/NaN -> string / None  (PostgreSQL JSON rejects bare Infinity/NaN)
6. Replace all json.dumps(...) calls in storage functions with _safe_json_dumps(...)
7. [NEW] Fix PCA n_components: adapt to min(2, n_features) at fit-time to avoid
   ValueError when a 1-feature combination is passed.
8. [NEW] Cap sample size for slow O(n²) models (OPTICS, SpectralClustering,
   MeanShift) to 3000 rows so the pipeline finishes in minutes, not hours.
"""

import json
import shutil
from pathlib import Path

NOTEBOOK = Path("COMPASSLLM_metadata_engine.ipynb")
BACKUP   = Path("COMPASSLLM_metadata_engine.ipynb.bak")

# Always start from the original backup so the script is idempotent
if BACKUP.exists():
    shutil.copy(BACKUP, NOTEBOOK)
    print(f"Restored from backup: {BACKUP} -> {NOTEBOOK}")
else:
    shutil.copy(NOTEBOOK, BACKUP)
    print(f"Backup created: {BACKUP}")

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))


# ── Helper ────────────────────────────────────────────────────────────────────

def patch_source(source: list[str], patches: list[tuple[str, str]]) -> list[str]:
    """Apply (old, new) string substitutions to the joined cell source."""
    joined = "".join(source)
    for old, new in patches:
        joined = joined.replace(old, new)
    return joined.splitlines(keepends=True)


# ── Patch definitions ─────────────────────────────────────────────────────────

CELL_PATCHES: dict[str, list[tuple[str, str]]] = {}

# ── Cell: Configuration — add timezone + _uuid_module + _math_module imports ──
CELL_PATCHES["cell_config"] = [
    (
        "from datetime import datetime\n",
        "from datetime import datetime, timezone\nimport uuid as _uuid_module\nimport math as _math_module\n",
    ),
]

# ── Cell: Preprocessing — fix OneHotEncoder sparse -> sparse_output ───────────
CELL_PATCHES["cell_preprocessing"] = [
    (
        "OneHotEncoder(handle_unknown='ignore', sparse=False)",
        "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
    ),
]

# ── Cell: Supervised models — remove deprecated use_label_encoder + utcnow ────
CELL_PATCHES["cell_supervised"] = [
    (
        "xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')",
        "xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')",
    ),
    ("start_train = datetime.utcnow()", "start_train = datetime.now(timezone.utc)"),
    ("end_train = datetime.utcnow()",   "end_train = datetime.now(timezone.utc)"),
    ("start_infer = datetime.utcnow()", "start_infer = datetime.now(timezone.utc)"),
    ("end_infer = datetime.utcnow()",   "end_infer = datetime.now(timezone.utc)"),
]

# ── Cell: Unsupervised models ──────────────────────────────────────────────────
# Fix 1: datetime.utcnow() deprecation
# Fix 2: PCA adaptive n_components  (crash when n_features < 2)
# Fix 3: Sample cap for slow O(n²) models  (OPTICS/SpectralClustering/MeanShift)
CELL_PATCHES["cell_unsupervised"] = [
    # utcnow -> timezone-aware
    ("start_train = datetime.utcnow()", "start_train = datetime.now(timezone.utc)"),
    ("end_train = datetime.utcnow()",   "end_train = datetime.now(timezone.utc)"),
    ("start_infer = datetime.utcnow()", "start_infer = datetime.now(timezone.utc)"),
    ("end_infer = datetime.utcnow()",   "end_infer = datetime.now(timezone.utc)"),

    # PCA crash fix + slow-model sample cap:
    # Replace the bare `model.fit(X)` inside run_unsupervised_experiment with
    # guarded logic that adapts PCA n_components and caps rows for slow models.
    (
        "    try:\n"
        "        start_train = datetime.now(timezone.utc)\n"
        "        model.fit(X)\n"
        "        end_train = datetime.now(timezone.utc)\n"
        "        result['train_time'] = (end_train - start_train).total_seconds()\n"
        "        start_infer = datetime.now(timezone.utc)\n"
        "        result['metrics'] = evaluate_unsupervised_model(model_name, model, X)\n"
        "        end_infer = datetime.now(timezone.utc)\n",

        "    try:\n"
        "        # ── PCA: adapt n_components to the actual number of features ──────\n"
        "        if model_name == 'PCA':\n"
        "            n_comp = min(2, X.shape[1])\n"
        "            if n_comp < 1:\n"
        "                result['status'] = 'skipped'\n"
        "                result['error_details'] = 'Insufficient features for PCA'\n"
        "                return result\n"
        "            model = PCA(n_components=n_comp)\n"
        "        # ── Cap rows for O(n²) models to keep runtime reasonable ──────────\n"
        "        _SLOW_MODELS = {'OPTICS', 'SpectralClustering', 'MeanShift'}\n"
        "        _MAX_SLOW_ROWS = 3000\n"
        "        _X_fit = (\n"
        "            X.sample(_MAX_SLOW_ROWS, random_state=42)\n"
        "            if model_name in _SLOW_MODELS and len(X) > _MAX_SLOW_ROWS\n"
        "            else X\n"
        "        )\n"
        "        start_train = datetime.now(timezone.utc)\n"
        "        model.fit(_X_fit)\n"
        "        end_train = datetime.now(timezone.utc)\n"
        "        result['train_time'] = (end_train - start_train).total_seconds()\n"
        "        start_infer = datetime.now(timezone.utc)\n"
        "        result['metrics'] = evaluate_unsupervised_model(model_name, model, _X_fit)\n"
        "        end_infer = datetime.now(timezone.utc)\n",
    ),
]

# ── Cell: Storage — UUID+Infinity-safe JSON helpers ───────────────────────────
UUID_ENCODER_BLOCK = """\
import uuid

def _sanitize_for_json(obj):
    \"\"\"Recursively make an object safe for JSON serialisation.

    Handles:
    - uuid.UUID  -> str
    - float Inf  -> string 'Infinity' / '-Infinity'  (PostgreSQL rejects bare token)
    - float NaN  -> None
    \"\"\"
    if isinstance(obj, _uuid_module.UUID):
        return str(obj)
    if isinstance(obj, float):
        if _math_module.isinf(obj):
            return 'Infinity' if obj > 0 else '-Infinity'
        if _math_module.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj

def _safe_json_dumps(obj) -> str:
    \"\"\"json.dumps that handles UUID, Infinity and NaN values.\"\"\"
    return json.dumps(_sanitize_for_json(obj))

"""

CELL_PATCHES["cell_storage"] = [
    # Insert the encoder block right after `import uuid`
    ("import uuid\n", UUID_ENCODER_BLOCK),
    # Replace all json.dumps calls in storage functions
    ("json.dumps(metadata)", "_safe_json_dumps(metadata)"),
    ("json.dumps(run_metadata['parameters'])",        "_safe_json_dumps(run_metadata['parameters'])"),
    ("json.dumps(run_metadata['selected_features'])", "_safe_json_dumps(run_metadata['selected_features'])"),
    ("json.dumps(run_metadata['preprocessing'])",     "_safe_json_dumps(run_metadata['preprocessing'])"),
    ("json.dumps(run_metadata['metrics'])",           "_safe_json_dumps(run_metadata['metrics'])"),
    ("json.dumps(run_metadata.get('extra_metadata', {}))", "_safe_json_dumps(run_metadata.get('extra_metadata', {}))"),
]

# ── Apply ─────────────────────────────────────────────────────────────────────

CELL_KEYWORDS = {
    "cell_config":        "MAX_FEATURE_COMBINATION_SIZE",
    "cell_preprocessing": "build_preprocessing_pipeline",
    "cell_supervised":    "build_supervised_models",
    "cell_unsupervised":  "build_unsupervised_models",
    "cell_storage":       "upsert_dataset_record",
}

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

patched_count = 0
for patch_key, keyword in CELL_KEYWORDS.items():
    patches = CELL_PATCHES.get(patch_key, [])
    if not patches:
        continue
    for cell in code_cells:
        source_text = "".join(cell["source"])
        if keyword in source_text:
            original = source_text
            cell["source"] = patch_source(cell["source"], patches)
            new_text = "".join(cell["source"])
            if new_text != original:
                print(f"  ✓ Patched [{patch_key}]")
                patched_count += 1
            else:
                print(f"  ~ Skipped [{patch_key}] (already applied or pattern not found)")
            break
    else:
        print(f"  ✗ WARNING: cell not found for [{patch_key}] (keyword: {keyword!r})")

print(f"\n{patched_count} cell(s) patched.")
NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Fixed notebook written to: {NOTEBOOK}")
