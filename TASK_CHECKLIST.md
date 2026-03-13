# COMPASSLLM Task Checklist

Track implementation status here.

## Done

- [x] Frontend + backend project structure in place.
- [x] CSV dataset upload and preview flow.
- [x] Dataset analysis API and UI integration.
- [x] Target-column selection support.
- [x] Column exclusion support with re-run analysis.
- [x] LLM recommendation endpoint integration.
- [x] Optional user instruction passed to LLM.
- [x] Sample row context passed to LLM.
- [x] Model training pipeline for recommended models.
- [x] Training robustness improvements for hard target distributions.
- [x] Dashboard metrics connected to backend summary data.
- [x] Knowledge base persistence for recommendation outcomes.
- [x] Downloadable model artifact generation and API route.
- [x] Training results UI download links.
- [x] Updated project documentation (`README`, frontend/backend docs, download guide).
- [x] Run a full end-to-end smoke test on a fresh restart (upload -> analyze -> recommend -> train -> download).
- [x] Validate knowledge base entry creation after a fresh training run.

## Pending

- [ ] Increase LLM recommendation efficiency (prompt optimization, token reduction, caching, and latency tracking).
- [ ] Find and curate benchmarking datasets for evaluation across multiple domains.
- [ ] Fine-tune the recommendation model/prompt strategy using observed experiment outcomes.
- [ ] Send knowledge base feedback back to the LLM for automatic continuous fine-tuning.

## Next Suggested Milestones

- [ ] Retrieval-augmented recommendation: include similar successful historical runs from knowledge base in the LLM prompt.
- [ ] Build an offline benchmark runner to compare recommendation quality before vs after fine-tuning.
- [ ] Add a periodic feedback loop job that summarizes new experiments into LLM-ready tuning signals.
- [ ] Introduce measurable KPIs: recommendation hit rate, average top-model accuracy, and recommendation latency.
- [ ] Add model artifact metadata (schema/version/hash) for reproducibility.
- [ ] Add role-based access and dataset-level ownership controls.

## Execution Order (No Phases)

- [ ] Priority 1: Introduce measurable KPIs (hit rate, top-model accuracy, latency). Estimated effort: 1-2 days.
- [ ] Priority 2: Find and curate benchmarking datasets across domains. Estimated effort: 2-4 days.
- [ ] Priority 3: Build an offline benchmark runner for before/after comparisons. Estimated effort: 2-3 days.
- [ ] Priority 4: Increase LLM efficiency (prompt compression, response caching, token budget guards). Estimated effort: 2-4 days.
- [ ] Priority 5: Implement retrieval from knowledge base into recommendation prompts. Estimated effort: 2-3 days.
- [ ] Priority 6: Add periodic feedback job to summarize experiments for tuning signals. Estimated effort: 2-3 days.
- [ ] Priority 7: Fine-tune recommendation strategy using benchmark + production feedback. Estimated effort: 3-6 days.
- [ ] Priority 8: Automate knowledge-base-to-LLM continuous tuning loop with guardrails. Estimated effort: 3-5 days.
