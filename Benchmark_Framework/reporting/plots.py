"""Plot helpers for advanced planning evaluation reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DIFF_ORDER = ["easy", "medium", "hard", "unknown"]
RANK_METRICS = {
    "Success_Rate": "max",
    "FASR": "max",
    "IWSR": "max",
    "Exec": "max",
    "Halluc": "min",
    "PAS": "max",
    "CoT_Alignment": "max",
    "Retry_Gap": "min",
    "Temporal_Distance": "min",
}


def add_warning(warnings_out: list[dict[str, Any]], warning_type: str, message: str, **extra: Any) -> None:
    payload = {"type": warning_type, "message": message}
    payload.update(extra)
    warnings_out.append(payload)


def compute_iteration_profile(model_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute P(Valid | reached attempt k) for k = 1 … max_iterations.

    Metric correlation: iteration profile — discriminates Stochastic Searcher
    (flat P(Valid|k) curve) from Efficient Corrector (rising curve).
    Rationale: overall SR conflates first-attempt successes with retry-dependent
    ones. The iteration profile makes the retry structure explicit: at each attempt
    slot k, "reached_count" is the number of problems that were attempted at least
    k times; "exact_count" is the subset that required exactly k attempts; and
    P(Valid|k) = mean(Valid) among those exact-k rows. A rising curve means the
    model learns from earlier failures; a flat or declining curve means retries are
    independent random draws (Stechly et al. 2023).
    Code purpose: produces the ``iteration_profile`` list stored in each model's
    JSON payload under ``tables.iteration_profile``, and feeds ``profile_is_flat``
    for the Stochastic Searcher profile condition.
    Detail: iterates k from 1 to max_iter; ``reached`` = rows where Iterations ≥ k;
    ``exact`` = rows where Iterations == k. Returns empty list for empty DataFrames.
    """
    max_iter = int(model_df["Iterations"].dropna().max()) if not model_df["Iterations"].dropna().empty else 0
    rows: list[dict[str, Any]] = []
    for k in range(1, max_iter + 1):
        reached = model_df[model_df["Iterations"] >= k]
        exact = reached[reached["Iterations"] == k]
        probability = exact["Valid"].mean() if len(exact) else float("nan")
        rows.append(
            {
                "iteration": k,
                "reached_count": int(len(reached)),
                "exact_count": int(len(exact)),
                "p_valid_given_reached": probability,
            }
        )
    return rows



PLOT_DESCRIPTIONS = {
    "hallucination_heatmap": "Mean hallucination rate by model and domain. Lower is better.",
    "hallucination_by_model_domain": "Action hallucination rate by model and domain.",
    "object_hallucination_by_model_domain": "Object hallucination rate by model and domain.",
    "executability_by_model_domain": "Executability ratio distribution by model and domain.",
    "failure_type_breakdown": "Sequencing errors versus state fabrications per model.",
    "executability_vs_length": "Executability ratio against plan length, faceted by domain.",
    "temporal_distance_by_model": "Mean temporal distance for sequencing errors per model.",
    "cot_alignment_by_model_domain": "Mean CoT plan alignment score by model and domain.",
    "cot_success_rate": "Success rate split by CoT flag.",
    "cot_alignment_validity": "CoT plan alignment distribution for valid versus invalid plans.",
    "fasr_by_model_domain": "First-attempt success rate by model and domain.",
    "sr_vs_fasr": "Overall success rate compared with first-attempt success rate.",
    "fasr_by_difficulty": "First-attempt success rate by difficulty tier.",
    "iwsr_by_model_domain": "Iteration-weighted success rate by model and domain.",
    "sr_fasr_iwsr_by_model": "SR, FASR, and IWSR comparison per model.",
    "retry_gap_by_model": "SR minus FASR; higher values mean stronger retry dependence.",
    "fasr_iwsr_scatter": "FASR versus IWSR with success rate encoded by dot size and color.",
    "success_rate_heatmap": "Success rate heatmap by model and domain.",
    "failure_mode_taxonomy": "Failure taxonomy by hallucination rate and PAS.",
    "composite_scores": "Composite Planning Score by model.",
    "p_valid_given_k": "P(Valid | reached attempt k) iteration profile — discriminates Stochastic Searcher (flat line) from Efficient Corrector (rising curve).",
    "domain_ranking_heatmap": "Normalised within-domain rank heatmap (0=best, 1=worst) for all RANK_METRICS across all domains.",
    "rank_variance": "Mean rank standard deviation across domains per model — high variance signals domain specialisation.",
    "domain_correlation": "Spearman correlation matrix of model success rates across domains — detects redundancy or complementarity.",
    "ps_by_domain_stacked": "Composite Planning Score stacked by domain contribution — reveals whether high PS is broad or concentrated.",
    "metrics_summary_table": "Tabular summary of all key aggregate metrics rendered as a matplotlib figure for archiving.",
    "radar_chart": "Polar radar chart of FASR, IWSR, Exec, IHR, and PAS — larger filled area means stronger overall planning capability.",
}


def maybe_make_plots(
    df_metrics: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    show_plots: bool,
    save_plots: bool,
    plots_dir: Path,
    warnings_out: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate all benchmark visualisation plots and optionally save / display them.

    Metric correlation: all 27 plots cover every computed metric — hallucination
    (strict, fuzzy, object), executability, PAS, temporal distance, CoT plan alignment,
    FASR, IWSR, SR, Retry Gap, PS, within-domain rank, cross-model correlation,
    and the iteration profile.
    Rationale: generating plots inside the main script rather than a notebook
    ensures they are reproducible from the command line and stored alongside the
    JSON report. Each plot is registered with ``register(name, fig, models)`` so
    its path, title, and description are embedded in the JSON report under each
    model's ``plots`` list.
    Code purpose: single function that produces all figures using seaborn / matplotlib,
    handles conditional generation (only if show or save is requested), records
    paths, and closes figures to free memory. Returns a list of plot descriptor
    dicts for ``build_model_payloads``.
    Detail: ``model_palette`` assigns a stable colour per model from the tab20
    palette so the same model always has the same colour across all plots.
    ``register()`` appends to the local ``figures`` list; the save loop at the end
    iterates it once. Figures with insufficient data (all-NaN columns, empty tables)
    are skipped via guard conditions before the plt.subplots call.
    """
    if not show_plots and not save_plots:
        return []
    if df_metrics.empty:
        add_warning(warnings_out, "plot_skipped", "No data available for plotting.")
        return []

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        add_warning(warnings_out, "plot_import_error", str(exc))
        return []

    sns.set_theme(style="whitegrid", palette="tab10")
    saved: list[dict[str, Any]] = []
    figures: list[tuple[str, Any, list[str]]] = []
    all_models = sorted(df_metrics["Model"].dropna().unique())
    base_colors = sns.color_palette("tab20", n_colors=max(len(all_models), 1))
    model_palette = {model: base_colors[index] for index, model in enumerate(all_models)}

    def register(name: str, fig: Any, related_models: Optional[list[str]] = None) -> None:
        figures.append((name, fig, related_models or all_models))

    if not df_metrics["hallucination_rate"].isna().all():
        pivot = df_metrics.pivot_table(values="hallucination_rate", index="Model", columns="Domain", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4), max(4, len(pivot.index) * 0.5 + 1)))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, vmin=0, vmax=1, ax=ax)
        ax.set_title("Mean Hallucination Rate (Model x Domain)")
        register("hallucination_heatmap", fig)

        agg = df_metrics.groupby(["Model", "Domain"])["hallucination_rate"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=agg, x="Model", y="hallucination_rate", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("Hallucination Rate by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("hallucination_by_model_domain", fig)

        agg_obj = df_metrics.groupby(["Model", "Domain"])["object_hallucination_rate"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=agg_obj, x="Model", y="object_hallucination_rate", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("Object Hallucination Rate by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("object_hallucination_by_model_domain", fig)

    if not df_metrics["executability_ratio"].isna().all():
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=df_metrics.dropna(subset=["executability_ratio"]), x="Model", y="executability_ratio", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("Executability Ratio by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("executability_by_model_domain", fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        failure = tables["failure_breakdown"].set_index("Model")[["sequencing_error_proportion", "state_fabrication_proportion"]]
        failure.plot(kind="bar", stacked=True, ax=ax, color=["steelblue", "tomato"], edgecolor="white")
        ax.set_title("Failure Type Breakdown per Model")
        ax.tick_params(axis="x", rotation=30)
        register("failure_type_breakdown", fig)

        domains = sorted(df_metrics["Domain"].dropna().unique())
        fig, axes = plt.subplots(max(1, len(domains)), 1, figsize=(6, max(4, 4 * len(domains))), squeeze=False)
        for ax, domain in zip(axes[:, 0], domains):
            sub = df_metrics[df_metrics["Domain"] == domain].dropna(subset=["executability_ratio", "Length"])
            for model, group in sub.groupby("Model"):
                ax.scatter(group["Length"], group["executability_ratio"], label=model, alpha=0.65, s=30, color=model_palette.get(model, "gray"))
            ax.set_title(f"Domain: {domain}")
            ax.set_xlabel("Plan Length")
            ax.set_ylabel("Executability Ratio")
            ax.legend(fontsize=7)
        fig.suptitle("Executability Ratio vs Plan Length")
        fig.tight_layout()
        register("executability_vs_length", fig)

        temporal = df_metrics.dropna(subset=["mean_temporal_distance"])
        if not temporal.empty:
            models_td = sorted(temporal["Model"].unique())
            fig, axes = plt.subplots(max(1, len(models_td)), 1, figsize=(6, max(4, 3.5 * len(models_td))), squeeze=False)
            for ax, model in zip(axes[:, 0], models_td):
                values = temporal[temporal["Model"] == model]["mean_temporal_distance"]
                ax.hist(values.dropna(), bins=10, color=model_palette.get(model, "steelblue"), edgecolor="white", alpha=0.8)
                ax.set_title(model)
                ax.set_xlabel("Mean Temporal Distance")
                ax.set_ylabel("Count")
            fig.suptitle("Temporal Distance for Sequencing Errors")
            fig.tight_layout()
            register("temporal_distance_by_model", fig, models_td)

    if not df_metrics["cot_plan_alignment_score"].isna().all():
        cot_sub = df_metrics.dropna(subset=["cot_plan_alignment_score"])
        agg = cot_sub.groupby(["Model", "Domain"])["cot_plan_alignment_score"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=agg, x="Model", y="cot_plan_alignment_score", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("Mean CoT Plan Alignment Score by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("cot_alignment_by_model_domain", fig)

        cot_sr = df_metrics.copy()
        cot_sr["_cot_flag"] = cot_sr["Chain_of_Thought"].apply(lambda value: str(value).lower() in {"true", "1", "yes"})
        cot_sr = cot_sr.groupby(["Model", "_cot_flag"])["Valid"].mean().reset_index()
        cot_sr["CoT"] = cot_sr["_cot_flag"].map({True: "CoT=True", False: "CoT=False"})
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=cot_sr, x="Model", y="Valid", hue="CoT", ax=ax, palette="Set1")
        ax.set_title("Success Rate: CoT=True vs CoT=False")
        ax.tick_params(axis="x", rotation=30)
        register("cot_success_rate", fig)

        cot_valid = cot_sub.copy()
        cot_valid["Validity"] = cot_valid["Valid"].map({True: "Valid", False: "Invalid"})
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.violinplot(data=cot_valid, x="Model", y="cot_plan_alignment_score", hue="Validity", ax=ax, palette="Set1", inner="quart", dodge=True, cut=0, bw_adjust=0.8)
        ax.set_title("CoT Plan Alignment Distribution: Valid vs Invalid Plans")
        ax.tick_params(axis="x", rotation=30)
        register("cot_alignment_validity", fig)

    fasr_agg = tables["by_domain"][["Model", "Domain", "FASR", "Success_Rate", "IWSR"]]
    if not fasr_agg.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=fasr_agg, x="Model", y="FASR", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("First-Attempt Success Rate by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("fasr_by_model_domain", fig)

        model_rates = tables["overall"][["Model", "Success_Rate", "FASR"]]
        x_pos = np.arange(len(model_rates))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x_pos - 0.18, model_rates["Success_Rate"], 0.36, label="Success Rate", color="steelblue")
        ax.bar(x_pos + 0.18, model_rates["FASR"], 0.36, label="FASR", color="tomato")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_rates["Model"], rotation=30)
        ax.set_title("Success Rate vs FASR")
        ax.legend()
        register("sr_vs_fasr", fig)

        diff_table = tables["by_difficulty"].copy()
        diff_table["Difficulty"] = pd.Categorical(diff_table["Difficulty"], categories=DIFF_ORDER, ordered=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=diff_table, x="Difficulty", y="FASR", hue="Model", ax=ax, palette=model_palette, order=DIFF_ORDER)
        ax.set_title("FASR by Difficulty")
        register("fasr_by_difficulty", fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=fasr_agg, x="Model", y="IWSR", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("IWSR by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("iwsr_by_model_domain", fig)

        compare = tables["overall"][["Model", "Success_Rate", "FASR", "IWSR"]]
        x_pos = np.arange(len(compare))
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.25
        ax.bar(x_pos - width, compare["Success_Rate"], width, label="Success Rate", color="steelblue")
        ax.bar(x_pos, compare["FASR"], width, label="FASR", color="tomato")
        ax.bar(x_pos + width, compare["IWSR"], width, label="IWSR", color="seagreen")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(compare["Model"], rotation=30)
        ax.set_title("SR vs FASR vs IWSR per Model")
        ax.legend()
        register("sr_fasr_iwsr_by_model", fig)

        rg = tables["retry_gap"].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["tomato" if value > 0 else "steelblue" for value in rg["Retry_Gap"]]
        ax.barh(rg["Model"], rg["Retry_Gap"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("Retry Gap (SR - FASR) per Model")
        register("retry_gap_by_model", fig)

        scatter = tables["overall"][["Model", "FASR", "IWSR", "Success_Rate"]]
        fig, ax = plt.subplots(figsize=(8, 6))
        points = ax.scatter(scatter["FASR"], scatter["IWSR"], s=scatter["Success_Rate"] * 1500 + 50, c=scatter["Success_Rate"], cmap="RdYlGn", vmin=0, vmax=1, edgecolors="gray")
        for _, row in scatter.iterrows():
            ax.annotate(row["Model"], (row["FASR"], row["IWSR"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
        fig.colorbar(points, ax=ax, label="Overall Success Rate")
        ax.set_title("FASR vs IWSR (size = Success Rate)")
        register("fasr_iwsr_scatter", fig)

    sr_pivot = df_metrics.groupby(["Model", "Domain"])["Valid"].mean().unstack("Domain").fillna(0)
    if not sr_pivot.empty:
        fig, ax = plt.subplots(figsize=(max(8, len(sr_pivot.columns) * 1.4), max(4, len(sr_pivot.index) * 0.5 + 1)))
        sns.heatmap(sr_pivot, annot=True, fmt=".2f", cmap="YlGn", linewidths=0.4, vmin=0, vmax=1, ax=ax)
        ax.set_title("Success Rate Heatmap (Model x Domain)")
        register("success_rate_heatmap", fig)

    taxonomy = tables["overall"].copy()
    if not taxonomy.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        points = ax.scatter(taxonomy["Halluc"], taxonomy["PAS"], s=taxonomy["FASR"] * 1200 + 80, c=taxonomy["Success_Rate"], cmap="RdYlGn", vmin=0, vmax=1, edgecolors="dimgray")
        for _, row in taxonomy.iterrows():
            ax.annotate(row["Model"], (row["Halluc"], row["PAS"]), textcoords="offset points", xytext=(8, 4), fontsize=8)
        fig.colorbar(points, ax=ax, label="Overall Success Rate")
        ax.set_xlabel("Hallucination Rate")
        ax.set_ylabel("Precondition Awareness Score")
        ax.set_title("Failure Mode Taxonomy")
        register("failure_mode_taxonomy", fig)

    composite = tables["composite_score"]
    if not composite.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ordered = composite.sort_values("PS_overall", ascending=True)
        ax.barh(ordered["Model"], ordered["PS_overall"], xerr=[ordered["err_lo"].clip(lower=0), ordered["err_hi"].clip(lower=0)], color="steelblue", edgecolor="white")
        ax.set_title("Composite Planning Score by Model")
        ax.set_xlabel("PS")
        register("composite_scores", fig)

    # ── Plot 21: P(Valid | reached attempt k) ─────────────────────────────────────
    # Metric: Iteration profile — P(valid | reached attempt k) for k = 1…max_iter.
    # Rationale: A Stochastic Searcher produces a flat curve (each retry is an
    # independent sample, success probability stays constant). An Efficient Corrector
    # shows a rising curve (model leverages feedback, later attempts have higher
    # conditional success). This plot is the primary discriminator between these two
    # profiles (Stechly et al. 2023). The 0.5 dashed baseline marks the inflection
    # point where retries become "more likely to succeed than fail".
    # Data source: compute_iteration_profile() applied per-model to df_metrics.
    if not df_metrics[["Iterations", "Valid"]].dropna().empty:
        _iter_rows: list[dict] = []
        for _model in all_models:
            for _rec in compute_iteration_profile(df_metrics[df_metrics["Model"] == _model]):
                _p = _rec["p_valid_given_reached"]
                if math.isfinite(float(_p)):
                    _iter_rows.append({"Model": _model, "k": _rec["iteration"], "P(Valid|k)": float(_p)})
        _iter_df = pd.DataFrame(_iter_rows)
        if not _iter_df.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            for _model in all_models:
                _sub = _iter_df[_iter_df["Model"] == _model].sort_values("k")
                if not _sub.empty:
                    ax.plot(_sub["k"], _sub["P(Valid|k)"], marker="o", linewidth=2, label=_model, color=model_palette.get(_model))
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title("P(Valid | reached attempt k) — Iteration Profile")
            ax.set_xlabel("Attempt k")
            ax.set_ylabel("P(Valid | reached k)")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            register("p_valid_given_k", fig)

    # ── Plots 22–23: Within-domain ranking heatmap and rank variance ───────────────
    # Metric: RANK_METRICS — one rank per model per metric per domain.
    # Rationale: Absolute metric values are hard to compare across domains with
    # different difficulty distributions. Ranking within each domain removes this
    # bias and shows each model's relative standing among its peers on each metric.
    # The heatmap normalises ranks to [0,1] (0=best, RdYlGn_r → green=best) so
    # all subplots share the same colour scale regardless of number of models.
    # rank_variance computes the mean std of normalised rank across domains:
    # a low std means the model ranks similarly everywhere (generalist); a high std
    # means it dominates some domains and underperforms in others (specialist).
    # Data source: tables["rank_within_domain"]
    _rank_df = tables.get("rank_within_domain", pd.DataFrame())
    _rank_metrics_cols = [m for m in RANK_METRICS if not _rank_df.empty and m in _rank_df.columns]
    if not _rank_df.empty and _rank_metrics_cols:
        _domains_ranked = sorted(_rank_df["Domain"].dropna().unique())
        _n_dom = len(_domains_ranked)
        _n_cols_grid = min(_n_dom, 3)
        _n_rows_grid = math.ceil(_n_dom / _n_cols_grid)
        _fig_w = max(6, 4.5 * _n_cols_grid)
        _fig_h = max(4, (len(all_models) * 0.8 + 2.0) * _n_rows_grid)
        fig, axes = plt.subplots(_n_rows_grid, _n_cols_grid, figsize=(_fig_w, _fig_h), squeeze=False)
        for _idx, _dom in enumerate(_domains_ranked):
            _ri, _ci = divmod(_idx, _n_cols_grid)
            _ax = axes[_ri][_ci]
            _dom_df = _rank_df[_rank_df["Domain"] == _dom].set_index("Model")
            _pivot = _dom_df[[c for c in _rank_metrics_cols if c in _dom_df.columns]]
            _n_mod = len(_pivot)
            _pivot_norm = (_pivot - 1) / max(_n_mod - 1, 1) if _n_mod > 1 else _pivot.clip(0, 0)
            sns.heatmap(_pivot_norm, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1,
                        linewidths=0.3, ax=_ax, cbar=False)
            _ax.set_title(_dom, fontsize=9)
            _ax.tick_params(axis="x", rotation=45, labelsize=7)
            _ax.tick_params(axis="y", labelsize=7)
        for _idx in range(_n_dom, _n_rows_grid * _n_cols_grid):
            _ri, _ci = divmod(_idx, _n_cols_grid)
            axes[_ri][_ci].set_visible(False)
        fig.suptitle("Normalised Within-Domain Rank (0 = best, 1 = worst)")
        fig.tight_layout()
        register("domain_ranking_heatmap", fig)

        # Rank variance: std of rank across domains reveals domain specialisation
        _var_df = _rank_df.groupby("Model")[_rank_metrics_cols].std().reset_index()
        _var_df["mean_rank_std"] = _var_df[_rank_metrics_cols].mean(axis=1)
        _var_df = _var_df.sort_values("mean_rank_std", ascending=False)
        fig, ax = plt.subplots(figsize=(9, max(4, len(_var_df) * 0.65 + 1.2)))
        _var_colors = [model_palette.get(m, "steelblue") for m in _var_df["Model"]]
        ax.barh(_var_df["Model"], _var_df["mean_rank_std"], color=_var_colors, edgecolor="white")
        ax.set_title("Rank Variance Across Domains — Higher = More Domain-Specific")
        ax.set_xlabel("Mean Std of Normalised Within-Domain Rank")
        register("rank_variance", fig)

    # ── Plot 24: Spearman correlation of success rates across domains ──────────────
    # Metric: SR (success rate) per model per domain.
    # Rationale: Builds a Model×Model Spearman ρ matrix using each model's vector
    # of domain success rates as its "signature". High positive ρ means two models
    # succeed on exactly the same domains (they share the same capability frontier).
    # Near-zero or negative ρ suggests orthogonal capability profiles — one model
    # might be an expert where the other fails, pointing toward ensemble utility.
    # Also reveals whether domain difficulty is model-independent (all models
    # correlate strongly) or model-specific (low cross-model correlation).
    # Data source: df_metrics pivot Model×Domain → scipy/pandas Spearman corr.
    _sr_pivot = df_metrics.groupby(["Model", "Domain"])["Valid"].mean().unstack("Domain").fillna(0)
    if _sr_pivot.shape[0] > 1:
        _corr = _sr_pivot.T.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(max(6, len(_corr) * 1.1), max(5, len(_corr) * 1.1)))
        sns.heatmap(_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    linewidths=0.4, ax=ax)
        ax.set_title("Spearman Correlation of Model Success Rates Across Domains")
        register("domain_correlation", fig)

    # ── Plot 25: PS stacked by domain ─────────────────────────────────────────────
    # Metric: Composite Planning Score (PS) per model per domain.
    # Rationale: A high overall PS can hide an uneven domain distribution. If one
    # model scores 0.9 on one domain and 0.1 on four others, its overall PS is
    # similar to a model that scores 0.5 everywhere. The stacked bar makes this
    # visible: an even stack = generalist; a dominant single segment = specialist.
    # Complements the heatmap by giving a cumulative view rather than a comparative
    # one, and allows reading the "total PS budget" each model earns across domains.
    # Data source: tables["by_domain"]["PS"] pivoted Model×Domain.
    _by_domain_tbl = tables.get("by_domain", pd.DataFrame())
    if not _by_domain_tbl.empty and "PS" in _by_domain_tbl.columns:
        _pivot_ps = _by_domain_tbl.pivot_table(values="PS", index="Model", columns="Domain", fill_value=0)
        if not _pivot_ps.empty:
            fig, ax = plt.subplots(figsize=(max(8, len(_pivot_ps.columns) * 1.4), 6))
            _pivot_ps.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
            ax.set_title("Composite Planning Score — Stacked by Domain")
            ax.set_ylabel("Cumulative PS")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            fig.tight_layout()
            register("ps_by_domain_stacked", fig)

    # ── Plot 26: Metrics summary table ────────────────────────────────────────────
    # Metric: All major aggregate metrics from tables["overall"].
    # Rationale: A matplotlib table figure is captured alongside the graphical plots
    # in the plot archive, making it easy to include in a PDF report or share as a
    # PNG without exporting the DataFrame separately. Values are formatted to 3 d.p.;
    # NaN entries are shown as "—" to indicate metric unavailability (e.g. CoT
    # alignment is NaN for protocols without chain-of-thought).
    # Data source: tables["overall"], columns [Model, SR, FASR, IWSR, Exec, IHR,
    # PAS, CoT_Alignment, Retry_Gap, PS].
    _overall_tbl = tables.get("overall", pd.DataFrame())
    if not _overall_tbl.empty:
        _tbl_cols = ["Model", "Success_Rate", "FASR", "IWSR", "Exec", "IHR", "PAS", "CoT_Alignment", "Retry_Gap", "PS"]
        _tbl_avail = [c for c in _tbl_cols if c in _overall_tbl.columns]
        _tbl_sub = _overall_tbl[_tbl_avail].copy()
        for _col in _tbl_avail:
            if _col != "Model":
                _tbl_sub[_col] = _tbl_sub[_col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        _fig_w = max(10, len(_tbl_avail) * 1.4)
        _fig_h = max(2.5, len(_tbl_sub) * 0.55 + 1.5)
        fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
        ax.axis("off")
        _tbl_obj = ax.table(cellText=_tbl_sub.values, colLabels=_tbl_sub.columns,
                            loc="center", cellLoc="center")
        _tbl_obj.auto_set_font_size(False)
        _tbl_obj.set_fontsize(8)
        _tbl_obj.auto_set_column_width(col=list(range(len(_tbl_avail))))
        ax.set_title("Key Metrics Summary", pad=14, fontsize=10)
        register("metrics_summary_table", fig)

    # ── Plot 27: Capability radar chart ───────────────────────────────────────────
    # Metrics: FASR, IWSR, Exec (executability ratio), IHR (inverse hallucination),
    # PAS (precondition awareness) — all already in [0, 1].
    # Rationale: The pentagon shape encodes the five orthogonal capability axes
    # simultaneously. A Genuine Planner fills most of the pentagon; a No Grounding
    # model stays near the origin; a Vocabulary-Only model shows high IHR but low
    # PAS and Exec. The radar makes the profile signature visually immediate,
    # complementing the numerical PROFILE_DEFINITIONS threshold table.
    # Note: matplotlib polar axes require angles in radians; angles list is closed
    # by appending the first element so the polygon outline completes.
    # Data source: tables["overall"].
    if not _overall_tbl.empty:
        _radar_metrics = ["FASR", "IWSR", "Exec", "IHR", "PAS"]
        _avail_radar = [m for m in _radar_metrics if m in _overall_tbl.columns]
        if len(_avail_radar) >= 3:
            _N = len(_avail_radar)
            _angles = [n / float(_N) * 2 * math.pi for n in range(_N)]
            _angles += _angles[:1]
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
            ax.set_theta_offset(math.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(_angles[:-1])
            ax.set_xticklabels(_avail_radar, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
            for _model in all_models:
                _mrow = _overall_tbl[_overall_tbl["Model"] == _model]
                if _mrow.empty:
                    continue
                _vals = [float(_mrow.iloc[0].get(m, 0) or 0) for m in _avail_radar]
                _vals += _vals[:1]
                ax.plot(_angles, _vals, linewidth=1.8, label=_model, color=model_palette.get(_model))
                ax.fill(_angles, _vals, alpha=0.07, color=model_palette.get(_model))
            ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18), fontsize=8)
            ax.set_title("Capability Radar: FASR / IWSR / Exec / IHR / PAS", pad=22)
            register("radar_chart", fig)

    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    for name, fig, related_models in figures:
        description = PLOT_DESCRIPTIONS.get(name, "")
        print(f"\nGrafico: {name}")
        if description:
            print(f"  {description}")
        path = None
        if save_plots:
            path = plots_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(
            {
                "name": name,
                "title": fig.axes[0].get_title() if fig.axes else name,
                "description": description,
                "path": str(path) if path else None,
                "related_models": related_models,
            }
        )

    if show_plots:
        try:
            plt.show()
        except Exception as exc:
            add_warning(warnings_out, "plot_show_error", str(exc))
    for _, fig, _ in figures:
        plt.close(fig)

    return saved

