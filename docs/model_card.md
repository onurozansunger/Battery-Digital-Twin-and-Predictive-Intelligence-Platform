# Model Card — Battery RUL Predictor (Milestone 1)

## Model details

| | |
|---|---|
| **Task** | Regression — remaining useful life of a lithium-ion cell, in discharge cycles |
| **Version** | 0.1.0 |
| **Type** | Model zoo of nine estimators; one champion selected per run |
| **Champion (default config)** | Transformer encoder — attention-pooled, pre-norm, 20-cycle window |
| **Headline metric** | See `reports/metrics.json → nested_evaluation`. The previously published MAE 8.06 / R² 0.850 figure is **withdrawn** — it predates the Milestone 1.1 hardening and is not comparable. |
| **Input** | 80 causal features derived from a cell's own charge/discharge/EIS history |
| **Output** | A single scalar: estimated remaining cycles until 70 % SoH |
| **Training data** | NASA Ames PCoE battery aging dataset — see `docs/dataset_card.md` |
| **Licence** | MIT (code). Data is NASA public-domain. |

## Intended use

**Intended.** Research and engineering exploration of data-driven battery
prognostics; a reference implementation of leakage-free time-series ML; the
foundation for a digital-twin system (milestone 2).

**Not intended.** Safety-critical decisions. Do not use this model to decide
whether a cell is safe to keep in service, to certify a battery pack, or as the
sole input to a warranty or maintenance action. It is trained on five laboratory
cells of one chemistry and has never been validated in the field.

**Out of scope.** Other chemistries (LFP, NMC, solid-state), other formats
(pouch, prismatic), pack-level or module-level prediction, thermal-runaway or
safety-event prediction, and calendar-ageing-dominated deployments.

## Target definition

```
RUL(k) = k_EOL − k
```

where `k_EOL` is the first **persistent** discharge cycle at which the
trailing-median-smoothed capacity falls to or below 70 % of the 2.0 Ah rated
capacity (1.40 Ah). "Persistent" means the crossing holds for three consecutive
cycles — lithium-ion cells recover capacity after rest, so a single dip below
threshold is routine and taking the first bare crossing systematically
under-estimates life.

Units are **discharge cycles**, not calendar time: the NASA rig ran cells with
long idle gaps, so wall-clock age is an artifact of lab scheduling.

The target is **not** capped. A piecewise-linear cap is the standard remedy for
the fact that early-life RUL is nearly unpredictable, and `target.cap_at` enables
it, but it is off by default so headline numbers remain comparable with the
raw-RUL literature.

## Architecture — the champion

A pre-norm Transformer encoder over a 20-cycle sliding window:

* Linear projection of 80 features to `d_model = 96`
* Fixed sinusoidal positional encoding (learned embeddings overfit at this sample size)
* 2 encoder layers, 4 heads, feed-forward 192, dropout 0.15
* Learned attention pooling over the window, then a 2-layer MLP head
* ≈162 k parameters

Trained with AdamW (lr 2e-3, weight decay 1e-4), Huber loss (δ = 5), gradient
clipping at 1.0, `ReduceLROnPlateau`, and early stopping on validation loss with
patience 20.

**Huber rather than MSE** because RUL residuals have heavy tails near end of life;
squared error there drags the whole fit toward the last few cycles of each cell.

## Performance

Regenerate with `python scripts/run_pipeline.py --config configs/default.yaml`;
live numbers are in `reports/metrics.json` and `reports/evaluation_report.md`.

### Headline — leave-one-battery-out cross-validation

Each of the 5 cells is held out in turn; the feature pipeline is re-fit inside
every fold; out-of-fold predictions are pooled.

| | MAE | RMSE | R² | Bias | within 10 cycles |
|---|---|---|---|---|---|
| **Transformer, pooled (400 rows)** | ~~8.06~~ | ~~9.93~~ | ~~0.850~~ | ~~−1.84~~ | ~~61.3 %~~ |

> **Withdrawn.** These figures predate Milestone 1.1. They were produced with
> pre-split feature pruning, an end-of-life rule that accepted an unconfirmed
> two-cycle crossing at the end of a record, and a champion selected on a
> one-cell validation partition. The current numbers — including the nested
> estimate that accounts for model selection — are in `reports/metrics.json` and
> `reports/nested_model_comparison*.csv`. Do not quote the two side by side; see
> `docs/MILESTONE_1_1_HARDENING.md`.

| Held-out cell | n | MAE | RMSE | R² | Bias |
|---|---|---|---|---|---|
| B0005 | 103 | 6.66 | 8.85 | 0.911 | −4.64 |
| B0006 | 87 | 6.52 | 8.20 | 0.894 | −2.40 |
| B0018 | 75 | 8.47 | 9.77 | 0.797 | +8.46 |
| B0033 | 82 | 11.99 | 13.71 | 0.664 | −8.37 |
| B0034 | 53 | 6.66 | 7.46 | 0.762 | +0.09 |

Errors are in **cycles**. Per-fold MAE spans 6.5 – 12.0 (σ = 2.34). **Quote that
spread**, not a bootstrap interval over rows — rows within a cell are correlated,
so the bootstrap understates the real uncertainty.

### Single battery-holdout (train B0018/B0033/B0034, val B0006, test B0005)

| Model | MAE | RMSE | R² | Bias | α-λ (20 %) |
|---|---|---|---|---|---|
| Ridge | 10.83 | 13.19 | 0.860 | +3.60 | 0.590 |
| **Transformer** (champion) | 11.62 | 13.82 | 0.784 | **−1.19** | 0.398 |
| GRU | 15.10 | 16.86 | 0.678 | +3.42 | 0.350 |
| Random Forest | 19.19 | 23.51 | 0.555 | −5.15 | 0.361 |
| LightGBM | 20.77 | 23.62 | 0.550 | −4.34 | 0.238 |
| LSTM | 19.46 | 23.65 | 0.367 | −6.32 | 0.243 |
| XGBoost | 21.22 | 24.24 | 0.526 | −4.55 | 0.246 |
| CatBoost | 21.85 | 27.19 | 0.404 | −6.21 | 0.344 |
| Linear Regression | 31.13 | 33.80 | 0.079 | −31.13 | 0.008 |

## Factors affecting performance

| Factor | Effect |
|---|---|
| Position in life | Errors are largest early (RUL > 100), where cells are near-indistinguishable |
| Ambient temperature | Cells cycled at 4 °C behave differently enough that most are excluded by the cohort gates |
| Capacity recovery after rest | Creates non-monotonic capacity; smoothing lags it by a few cycles |
| Cell-to-cell variation | The dominant source of error under a battery-holdout split |
| Window length | Sequence models cannot score a cell's first *w*−1 cycles at all |

## Evaluation methodology

* **Split:** whole cells held out (`battery_holdout`). Train B0018, B0033, B0034;
  validate B0006; test B0005.
* **Headline:** leave-one-battery-out cross-validation over all 5 cells, with the
  feature pipeline re-fit inside each fold.
* **Selection (Milestone 1.1):** the quotable headline now comes from a **nested**
  leave-one-battery-out design — family selection runs inside every outer fold, so
  the pooled metric estimates the whole procedure rather than an already-chosen
  model, and interpretable baselines (cohort median life, capacity-fade
  extrapolation, SOH nearest-analogue, elastic net, ridge) compete on identical
  folds. The single-split champion below is still fitted and persisted for the
  Milestone 1 artifact path.
* **Selection (Milestone 1, retained):** champion chosen by validation RMSE. The test partition is scored
  once, after selection.
* **Metrics:** MAE, RMSE, MAPE (denominator floored at 1 cycle, since RUL reaches
  zero), SMAPE, R², max error, signed bias, α-λ accuracy, prognostic horizon.
* **Uncertainty:** the spread of per-fold metrics across the cross-validation. A
  percentile bootstrap over rows is also reported but **understates** true
  uncertainty, since rows within a cell are correlated.

## Explainability

SHAP (TreeExplainer for tree models, KernelExplainer otherwise), permutation
importance (computed within-cell so marginals are preserved), and native
importances — all three are reported side by side.

**Read them at the level of signal families, not individual columns.** The
feature set is deliberately collinear (a 5-cycle and a 10-cycle rolling mean of
the same signal are near-redundant), and under collinearity both SHAP and
permutation importance distribute credit among correlated features rather than
identifying a unique cause. The report therefore also aggregates importance into
physical families: capacity, resistance, voltage, temperature, charge timing,
discharge timing, current, cycle position.

Sequence models are explained by permutation importance only; faithful SHAP for
windowed inputs needs DeepExplainer over 3-D tensors and is deferred.

## Ethical considerations

Low direct risk — no personal data, no human subjects. The meaningful risk is
**misplaced confidence**: an R² of 0.85 on five laboratory cells could be read
as fitness for deployment. It is not. A wrong RUL prediction in a real system
means either a cell retired early (wasted capacity, unnecessary e-waste) or a
cell kept past its safe window (a genuine hazard). This model's bias favours the former early in
life but reverses near end of life — exactly where the cost is highest.

## Caveats and recommendations

1. Retrain and re-validate on your own chemistry and duty cycle. Do not transfer
   these weights.
2. Treat predictions as a range, not a number, especially early in life.
3. The model has no notion of uncertainty. Quantile regression or a conformal
   wrapper is the obvious next step (see `docs/limitations.md`).
4. Monitor for drift: a model trained on chamber-cycled 2008 cells will decay on
   field data, and nothing here detects that. Drift monitoring is milestone 5.
5. Right-censored cells are excluded, not modelled. In a fleet where most cells
   are still healthy, that is most of your data — you will need survival methods.

## Reproducibility

```bash
python scripts/download_data.py
python scripts/run_pipeline.py --config configs/default.yaml
```

Seed 42 throughout. `reports/metrics.json` embeds the git revision, Python
version, platform and every package version used.
