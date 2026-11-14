"""
Compare forecasting models on a workload CSV, without data leakage.

    python compare_models.py workload.csv --column request_rate --lookback 12 --horizon 4

Rules used:
  * chronological 70% / 15% / 15% split (train / validation / test), no shuffling
  * scaling is fitted on the training part only
  * windows are built inside each part, so none crosses a split boundary
  * neural models can be repeated over several random seeds (--seeds)
    and are reported as mean and standard deviation

Models: naive (last value), ma (moving average), linear, lightgbm, gru.
lightgbm and gru are skipped with a message if the library is missing.
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

from forecast import fit_linear, make_windows, predict_linear


def load_series(path, column, resample=None, agg="mean"):
    df = pd.read_csv(path)
    index = pd.to_datetime(df.iloc[:, 0], utc=True)
    series = pd.Series(df[column].astype(float).to_numpy(), index=index).dropna().sort_index()
    if resample:
        series = getattr(series.resample(resample), agg)().dropna()
    return series


def prepare(values, lookback, horizon):
    n = len(values)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    parts = {
        "train": values[:n_train],
        "val": values[n_train - lookback:n_train + n_val],
        "test": values[n_train + n_val - lookback:],
    }
    lo, hi = parts["train"].min(), parts["train"].max()
    if hi == lo:
        hi = lo + 1.0
    data = {k: make_windows((v - lo) / (hi - lo), lookback, horizon) for k, v in parts.items()}
    return data, lo, hi


# ------------------------------------------------------------------
# Models: each returns (scaled predictions for the test windows, train seconds, parameters)
# ------------------------------------------------------------------

def run_naive(d, seed=0):
    return d["test"][0][:, -1], 0.0, 0


def run_ma(d, seed=0):
    return d["test"][0].mean(axis=1), 0.0, 0


def run_linear(d, seed=0):
    t0 = time.perf_counter()
    coef = fit_linear(*d["train"])
    return predict_linear(coef, d["test"][0]), time.perf_counter() - t0, len(coef)


def run_lightgbm(d, seed=0):
    import lightgbm as lgb
    t0 = time.perf_counter()
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                              random_state=seed, verbosity=-1)
    model.fit(*d["train"])
    return model.predict(d["test"][0]), time.perf_counter() - t0, 200


def run_gru(d, seed=0):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(seed)
    np.random.seed(seed)
    lookback = d["train"][0].shape[1]
    Xtr, ytr = d["train"][0][..., None], d["train"][1]
    Xva, yva = d["val"][0][..., None], d["val"][1]
    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.GRU(64, return_sequences=True), layers.Dropout(0.2),
        layers.GRU(32), layers.Dropout(0.2),
        layers.Dense(16, activation="relu"), layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=60, batch_size=64,
              callbacks=[stop], verbose=0)
    elapsed = time.perf_counter() - t0
    return model.predict(d["test"][0][..., None], verbose=0).ravel(), elapsed, model.count_params()


MODELS = {"naive": run_naive, "ma": run_ma, "linear": run_linear,
          "lightgbm": run_lightgbm, "gru": run_gru}
LABELS = {"naive": "Naive (last value)", "ma": "Moving average", "linear": "Linear regression",
          "lightgbm": "LightGBM", "gru": "GRU"}
NEURAL = {"gru"}


def score(actual, predicted):
    err = actual - predicted
    nonzero = actual != 0
    ss_res, ss_tot = np.sum(err ** 2), np.sum((actual - actual.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan"),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def compare(values, lookback, horizon, models, seeds=1):
    data, lo, hi = prepare(np.asarray(values, float), lookback, horizon)
    actual = data["test"][1] * (hi - lo) + lo
    rows, curves = [], {}
    for name in models:
        n_runs = seeds if name in NEURAL else 1
        scores, times, params, last_pred = [], [], 0, None
        try:
            for seed in range(n_runs):
                pred, seconds, params = MODELS[name](data, seed)
                pred = np.clip(pred * (hi - lo) + lo, 0, None)
                scores.append(score(actual, pred))
                times.append(seconds)
                last_pred = pred
        except ImportError as error:
            print(f"Skipping {LABELS[name]}: {error}")
            continue
        row = {"model": LABELS[name], "train_seconds": float(np.mean(times)), "parameters": params,
               "runs": n_runs}
        for key in ("mae", "rmse", "mape", "r2"):
            row[key] = float(np.mean([s[key] for s in scores]))
        row["rmse_std"] = float(np.std([s["rmse"] for s in scores]))
        rows.append(row)
        curves[LABELS[name]] = last_pred
    table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return table, actual, curves, {k: len(v[1]) for k, v in data.items()}


def make_plot(table, actual, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    n_show = min(300, len(actual))
    axes[0].plot(actual[:n_show], label="actual", linewidth=1.2)
    for name in list(table["model"])[:3]:
        axes[0].plot(curves[name][:n_show], label=name, linewidth=1, alpha=0.85)
    axes[0].set_title(f"Predicted vs actual (first {n_show} test points, best 3 models)")
    axes[0].set_xlabel("time step"); axes[0].legend()
    axes[1].bar(table["model"], table["rmse"], yerr=table["rmse_std"], capsize=4)
    axes[1].set_title("Test RMSE by model (lower is better; bars show std over seeds)")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--column", default="request_rate")
    p.add_argument("--resample", default=None, help="e.g. 5min; groups points into buckets")
    p.add_argument("--agg", default="mean", choices=["mean", "sum"], help="how to combine points when resampling")
    p.add_argument("--lookback", type=int, default=12, help="input window length in steps")
    p.add_argument("--horizon", type=int, default=1, help="how many steps ahead to predict")
    p.add_argument("--models", default="naive,ma,linear,lightgbm,gru")
    p.add_argument("--seeds", type=int, default=1, help="repeat neural models this many times")
    p.add_argument("--out", default="compare", help="prefix for the result files")
    args = p.parse_args(argv)

    series = load_series(args.csv, args.column, args.resample, args.agg)
    if len(series) < 10 * args.lookback:
        sys.exit(f"Only {len(series)} points; need at least {10 * args.lookback}. Collect more data.")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in MODELS]
    if bad:
        sys.exit(f"Unknown model(s): {bad}. Choose from {list(MODELS)}")

    table, actual, curves, sizes = compare(series.to_numpy(), args.lookback, args.horizon, models, args.seeds)
    print(f"\n{len(series)} points | windows train/val/test = {sizes['train']}/{sizes['val']}/{sizes['test']}"
          f" | lookback {args.lookback}, horizon {args.horizon}")
    shown = table.copy()
    for col in ("mae", "rmse", "rmse_std", "mape", "r2", "train_seconds"):
        shown[col] = shown[col].round(3)
    print(shown.to_string(index=False))
    best = table.iloc[0]
    print(f"\nLowest RMSE: {best['model']}. Gaps of a few percent are within normal run-to-run "
          f"variation for neural models; prefer the simplest model whose accuracy is close to the best.")
    table.to_csv(f"{args.out}_results.csv", index=False)
    make_plot(table, actual, curves, f"{args.out}_plot.png")
    print(f"Saved {args.out}_results.csv and {args.out}_plot.png")


if __name__ == "__main__":
    main()
