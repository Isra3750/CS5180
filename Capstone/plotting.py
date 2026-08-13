# plotting.py -- learning-curve plots and a summary table over a results DataFrame.
import pandas as pd
import matplotlib.pyplot as plt

from config import paper_reference


# Plot each (algorithm, seed) eval curve on its own line, useful for spotting per-seed spread
def plot_eval_curve(df, title=None):
    eval_df = df.dropna(subset=["eval_return"]).copy() # keep only rows that carry an eval return
    plt.figure(figsize=(9, 5.5))
    for (algo, env_id, seed), g in eval_df.groupby(["algorithm", "env_id", "seed"]):
        plt.plot(g["timestep"], g["eval_return"], marker="o", ms=3, label=f"{algo}, seed={seed}")
    plt.xlabel("Environment timesteps"); plt.ylabel("Average evaluation return")
    plt.title(title or "Evaluation curve"); plt.legend(); plt.grid(True, alpha=0.3); plt.show()


# Plot mean +/- std across seeds, one shaded band per (algorithm, env)
def plot_mean_std_curve(df, title="Mean evaluation return"):
    eval_df = df.dropna(subset=["eval_return"]).copy()
    # Aggregate seeds at each timestep into a mean and std.
    summary = (eval_df.groupby(["algorithm", "env_id", "timestep"], as_index=False)
               .agg(mean_return=("eval_return", "mean"), std_return=("eval_return", "std")))
    summary["std_return"] = summary["std_return"].fillna(0.0)  # single-seed groups have NaN std
    plt.figure(figsize=(9, 5.5))
    for (algo, env_id), g in summary.groupby(["algorithm", "env_id"]):
        x = g["timestep"].to_numpy()
        mean = g["mean_return"].to_numpy()
        std = g["std_return"].to_numpy()
        plt.plot(x, mean, label=f"{algo}")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)   # +/- 1 std band
    plt.xlabel("Environment timesteps"); plt.ylabel("Average evaluation return")
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.3); plt.show()


# Overlay mean curves for every algorithm present + the paper TD3 reference line
def plot_compare(df, env_id, show_paper=True):
    eval_df = df.dropna(subset=["eval_return"]).copy()

    # Mean +/- std across seeds per algorithm (env is fixed by the caller).
    summary = (eval_df.groupby(["algorithm", "timestep"], as_index=False)
               .agg(mean_return=("eval_return", "mean"), std_return=("eval_return", "std")))
    summary["std_return"] = summary["std_return"].fillna(0.0)

    plt.figure(figsize=(9, 5.5))
    for algo, g in summary.groupby("algorithm"):
        x = g["timestep"].to_numpy()
        mean = g["mean_return"].to_numpy()
        std = g["std_return"].to_numpy()
        plt.plot(x, mean, label=algo)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)
        
    # Dashed horizontal line for the paper's TD3 number (skip if the env isn't in the table).
    ref = paper_reference(env_id)
    if show_paper and ref is not None:
        plt.axhline(ref, ls="--", color="gray", label=f"Paper TD3 (v1): {ref:g}")
    plt.xlabel("Environment timesteps"); plt.ylabel("Average evaluation return")
    plt.title(f"{env_id}: algorithm comparison"); plt.legend(); plt.grid(True, alpha=0.3); plt.show()


# Build a per-(algorithm, seed) table of max and last-10-eval return, with the paper reference
def summarize(df):
    eval_df = df.dropna(subset=["eval_return"]).copy()
    rows = []
    for (algo, env_id, seed), g in eval_df.groupby(["algorithm", "env_id", "seed"]):
        g = g.sort_values("timestep")
        rows.append({"algorithm": algo, "env_id": env_id, "seed": seed,
                     "max_return": g["eval_return"].max(),
                     "last10_avg": g["eval_return"].tail(10).mean(), # end-of-training performance
                     "paper_td3_v1": paper_reference(env_id)})
    # Sort best-scoring rows to the top within each env.
    out = pd.DataFrame(rows).sort_values(["env_id", "max_return"], ascending=[True, False])
    return out.reset_index(drop=True)
