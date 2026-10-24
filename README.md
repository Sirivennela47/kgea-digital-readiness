# KGEA: evidence acquisition under knowledge mismatch

Synthetic research prototype for DBA 802. No real enterprise records or validated readiness labels are used.

## Reproduce
Python 3.12. Install `pip install -r requirements.txt`, then run `python run.py --seeds 20`.
The command runs invariant checks and writes all numerical results, two tables and two experimental figures into `results/`. Twenty independent training samples, 2,000 training observations each, and 500 held-out observations per scenario/seed are used. Outputs are deterministic for the pinned environment. Runtime is approximately seconds to minutes on a CPU.

## Interpretation
All policies share a classifier trained on complete observations from the matched simulation. Reversed rules changes the test outcome mechanism, while Shift changes test capability distributions. These are deliberate stress tests, not new enterprise datasets. Intervals describe Monte Carlo variability across independent simulations, not uncertainty over businesses. Figure 2 intervals are unadjusted exploratory intervals. No state-of-the-art or causal claim is supported.

The fixed baseline evaluates sensitivity at the prior mean; it is not an optimized global policy. The no-knowledge ablation changes acquisition only, retaining the nonlinear predictor. Costs and indicator noise are assumed, not estimated. At ties, the lowest-index indicator/group wins. Re-running each budget starts a fresh acquisition trajectory. Random priorities are seeded and reproducible.

## Submission status
The manuscript is a research draft. Human author names, affiliation, correspondence, funding, competing interests and CRediT contributions must be supplied. Verify novelty and journal eligibility before submission. Strong learned AFA baselines and external validation remain outstanding. Do not submit as field-validated research.

## Compact layout
`run.py`, `requirements.txt`, `README.md`, `manuscript.md`; generated outputs live in one `results/` directory. Figures are programmatically plotted, not AI-generated. Source manuscript tables are generated from actual run summaries.
