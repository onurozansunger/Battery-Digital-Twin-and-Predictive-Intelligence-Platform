# Model Card — Battery RUL Predictor (Milestone 1)

## Model details

| | |
|---|---|
| **Task** | Regression — remaining useful life of a lithium-ion cell, in discharge cycles |
| **Version** | 0.1.0 |
| **Type** | Model zoo of nine estimators; one champion selected per run |
| **Champion (default config)** | Transformer encoder — attention-pooled, pre-norm, 20-cycle window |
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
sole input to a warranty or maintenance action. It is trained on eight laboratory
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

Held-out cells **B0005** and **B0034**, never seen during training, scaling,
feature selection, or model choice. Regenerate with
`python scripts/run_pipeline.py --config configs/default.yaml`; live numbers are
in `reports/metrics.json` and `reports/evaluation_report.md`.

| Model | MAE | RMSE | R² | α-λ (20 %) | within 10 cycles |
|---|---|---|---|---|---|
| **Transformer** | **11.0** | **13.0** | **0.790** | 0.378 | 51.9 % |
| GRU | 12.5 | 14.6 | 0.733 | 0.269 | 41.0 % |
| Ridge | 14.1 | 17.5 | 0.718 | 0.454 | 41.2 % |
| LSTM | 14.9 | 18.5 | 0.575 | 0.282 | 47.4 % |
| Linear regression | 19.7 | 20.2 | 0.625 | 0.139 | 5.7 % |
| Random forest | 17.4 | 20.7 | 0.606 | 0.351 | 35.1 % |
| CatBoost | 17.8 | 23.4 | 0.498 | 0.418 | 43.3 % |
| XGBoost | 22.9 | 25.2 | 0.417 | 0.186 | 13.9 % |
| LightGBM | 23.2 | 25.5 | 0.402 | 0.170 | 13.9 % |

Errors are in **cycles**. RMSE 95 % bootstrap CI for the champion: 11.77–14.20.

The report also contains a **like-for-like** table restricted to the 156 rows
every model can score (sequence models cannot score a cell's first 19 cycles, and
those early rows are the hardest). On that footing Ridge rises to 2nd with a bias
of −1.0 cycles against the Transformer's −6.8 — it is materially better
calibrated, and materially easier to explain. If the deployment need were an
unbiased estimate rather than the lowest RMSE, Ridge would be the right choice.

### Reading these numbers honestly

* **Gradient boosting underperforms the linear and neural models.** That is not a
  bug — it is what a battery-holdout split does to tree ensembles. Trees
  extrapolate by returning a constant outside the training range, and each
  held-out cell has a slightly different capacity scale. The linear and recurrent
  models extrapolate; the trees cannot. Under a chronological split
  (`configs/chronological.yaml`) the ranking reverses.
* **The neural models carry a systematic negative bias** (≈ −7 cycles): they
  predict *less* remaining life than the cell has. Ridge, by contrast, is nearly
  unbiased (−1.0). For maintenance planning under-prediction is the safe
  direction, but it is still a systematic error — and it means the champion's
  advantage over Ridge is in variance, not calibration.
* **The α-λ accuracy is low** (38 %) and the **prognostic horizon is not reached
  on either test cell** — predictions never settle inside the ±20 % relative cone
  and stay there. The cone tightens to two or three cycles near end of life, so
  this is a demanding bar at an MAE of 11, but it is the honest answer: this model
  tells you "roughly how long left", not "swap it on Tuesday".
* **Two test cells.** The per-cell table in the evaluation report carries more
  information than any aggregate here.

## Factors affecting performance

| Factor | Effect |
|---|---|
| Position in life | Errors are largest early (RUL > 100), where cells are near-indistinguishable |
| Ambient temperature | Cells cycled at 4 °C behave differently enough that most are excluded by the cohort gates |
| Capacity recovery after rest | Creates non-monotonic capacity; smoothing lags it by a few cycles |
| Cell-to-cell variation | The dominant source of error under a battery-holdout split |
| Window length | Sequence models cannot score a cell's first *w*−1 cycles at all |

## Evaluation methodology

* **Split:** whole cells held out (`battery_holdout`). Train B0018, B0033, B0043,
  B0044; validate B0006, B0042; test B0005, B0034.
* **Selection:** champion chosen by validation RMSE. The test partition is scored
  once, after selection.
* **Metrics:** MAE, RMSE, MAPE (denominator floored at 1 cycle, since RUL reaches
  zero), SMAPE, R², max error, signed bias, α-λ accuracy, prognostic horizon.
* **Uncertainty:** percentile bootstrap over rows. Rows within a cell are
  correlated, so the interval **understates** true uncertainty.

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
**misplaced confidence**: an R² of 0.79 on eight laboratory cells could be read
as fitness for deployment. It is not. A wrong RUL prediction in a real system
means either a cell retired early (wasted capacity, unnecessary e-waste) or a
cell kept past its safe window (a genuine hazard). This model's negative bias
favours the former, but that is a property of this dataset, not a guarantee.

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
