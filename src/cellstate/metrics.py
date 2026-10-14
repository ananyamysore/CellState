"""Documented reproducibility metrics for clustering and ranked signatures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import adjusted_rand_score


def cluster_stability(labels_a: Sequence[object], labels_b: Sequence[object]) -> float | None:
    """Adjusted Rand index (chance-adjusted partition agreement; permutation-invariant)."""
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("cluster label vectors must be matching 1D arrays")
    if np.asarray([x is None for x in a]).any() or np.asarray([x is None for x in b]).any():
        raise ValueError("cluster labels must not contain missing values")
    if len(a) < 2:
        return None
    return float(adjusted_rand_score(a, b))


def signature_agreement(
    genes_a: Sequence[str],
    scores_a: Sequence[float],
    genes_b: Sequence[str],
    scores_b: Sequence[float],
    *,
    top_n: int = 100,
) -> dict[str, float | int | None]:
    """Compare ranked gene signatures via top-N Jaccard and Spearman score ranks.

    Metrics require gene identifiers to match exactly across datasets. Jaccard ignores
    ordering; rank Spearman uses the union of genes with absent genes assigned the lowest rank.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if len(genes_a) != len(scores_a) or len(genes_b) != len(scores_b):
        raise ValueError("gene and score lengths must match")
    if len(set(genes_a)) != len(genes_a) or len(set(genes_b)) != len(genes_b):
        raise ValueError("gene identifiers must be unique within each signature")
    if (
        not np.isfinite(np.asarray(scores_a, dtype=float)).all()
        or not np.isfinite(np.asarray(scores_b, dtype=float)).all()
    ):
        raise ValueError("signature scores must be finite")
    order_a = np.argsort(np.abs(np.asarray(scores_a, dtype=float)))[::-1]
    order_b = np.argsort(np.abs(np.asarray(scores_b, dtype=float)))[::-1]
    top_a = {str(genes_a[i]) for i in order_a[:top_n]}
    top_b = {str(genes_b[i]) for i in order_b[:top_n]}
    union = top_a | top_b
    jac = len(top_a & top_b) / len(union) if union else None
    all_genes = sorted(set(map(str, genes_a)) | set(map(str, genes_b)))
    if len(all_genes) < 2:
        rho = None
    else:
        va = dict(zip(map(str, genes_a), np.asarray(scores_a, dtype=float)))
        vb = dict(zip(map(str, genes_b), np.asarray(scores_b, dtype=float)))
        floor_a = min(va.values(), default=0) - 1
        floor_b = min(vb.values(), default=0) - 1
        ra = rankdata([va.get(g, floor_a) for g in all_genes])
        rb = rankdata([vb.get(g, floor_b) for g in all_genes])
        rho = float(spearmanr(ra, rb).statistic)
    return {
        "top_n": top_n,
        "top_gene_jaccard": jac,
        "rank_spearman": rho,
        "shared_genes": len(set(map(str, genes_a)) & set(map(str, genes_b))),
        "union_genes": len(all_genes),
    }


def _bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[
        ::-1
    ]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def composition_table(obs, *, sample_key: str, condition_key: str, cell_type_key: str):
    """Cell-type proportions per independent sample; descriptive, not cell-level inference."""
    needed = {sample_key, condition_key, cell_type_key}
    if not needed.issubset(obs.columns):
        raise ValueError(f"missing composition metadata: {sorted(needed - set(obs.columns))}")
    if obs[list(needed)].isna().any().any():
        raise ValueError("composition metadata contains missing values")
    counts = (
        obs.groupby([sample_key, condition_key, cell_type_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    counts["fraction"] = counts["n_cells"] / counts.groupby(sample_key)["n_cells"].transform("sum")
    return counts


def composition_differences(
    obs,
    *,
    sample_key: str,
    condition_key: str,
    cell_type_key: str,
    donor_key: str | None = None,
    group_a: str | None = None,
    group_b: str | None = None,
    min_units: int = 2,
) -> list[dict[str, object]]:
    """Compare cell-type fractions using samples/donors as units, never cells.

    Paired donors use paired t-tests; otherwise sample-level Welch tests are used.
    Fractions are descriptive compositional outcomes and the t-test ignores their
    joint sum-to-one constraint; interpret p-values cautiously and prefer a
    study-specific compositional model for confirmatory inference.
    """
    import pandas as pd
    from scipy import stats

    needed = {sample_key, condition_key, cell_type_key}
    if not needed.issubset(obs.columns):
        raise ValueError(f"missing composition metadata: {sorted(needed - set(obs.columns))}")
    frame = (
        obs[list(needed | ({donor_key} if donor_key else set()))]
        .dropna(subset=[sample_key, condition_key, cell_type_key])
        .copy()
    )
    frame[condition_key] = frame[condition_key].astype(str)
    conditions = list(pd.unique(frame[condition_key]))
    if group_a is None or group_b is None:
        if len(conditions) != 2:
            return [
                {
                    "status": "inconclusive",
                    "reason": "composition contrast requires exactly two conditions or explicit groups",
                }
            ]
        group_a, group_b = conditions
    group_a, group_b = str(group_a), str(group_b)
    if group_a not in conditions or group_b not in conditions or group_a == group_b:
        raise ValueError("invalid composition contrast")
    sample_cols = [sample_key, condition_key, cell_type_key]
    if donor_key and donor_key in frame:
        sample_cols.append(donor_key)
    counts = frame.groupby(sample_cols, observed=True).size().rename("n").reset_index()
    counts["fraction"] = counts.n / counts.groupby(sample_key).n.transform("sum")
    unit = (
        donor_key
        if donor_key and donor_key in counts and counts[donor_key].notna().all()
        else sample_key
    )
    if unit != sample_key:
        counts = (
            counts.groupby([unit, condition_key, cell_type_key], observed=True)
            .fraction.mean()
            .reset_index()
        )
    output = []
    celltypes = sorted(frame[cell_type_key].astype(str).unique())
    for cell in celltypes:
        sub = counts[counts[cell_type_key].astype(str) == cell]
        a = sub[sub[condition_key] == group_a].set_index(unit).fraction
        b = sub[sub[condition_key] == group_b].set_index(unit).fraction
        common = a.index.intersection(b.index)
        if len(common) >= min_units:
            method = "paired_t_test"
            av = a.loc[common].to_numpy()
            bv = b.loc[common].to_numpy()
            raw_p = float(stats.ttest_rel(bv, av).pvalue)
            p = raw_p if np.isfinite(raw_p) else None
            n_a = n_b = len(common)
        elif len(a) >= min_units and len(b) >= min_units:
            method = "welch_t_test"
            av = a.to_numpy()
            bv = b.to_numpy()
            raw_p = float(stats.ttest_ind(bv, av, equal_var=False).pvalue)
            p = raw_p if np.isfinite(raw_p) else None
            n_a, n_b = len(a), len(b)
        else:
            output.append(
                {
                    "cell_type": cell,
                    "condition_a": group_a,
                    "condition_b": group_b,
                    "status": "inconclusive",
                    "n_a": len(a),
                    "n_b": len(b),
                    "reason": "fewer than minimum independent units",
                }
            )
            continue
        output.append(
            {
                "cell_type": cell,
                "condition_a": group_a,
                "condition_b": group_b,
                "status": "descriptive_inference",
                "mean_fraction_a": float(np.mean(av)),
                "mean_fraction_b": float(np.mean(bv)),
                "difference_b_minus_a": float(np.mean(bv) - np.mean(av)),
                "p_value": p,
                "n_a": n_a,
                "n_b": n_b,
                "unit_key": unit,
                "test": method,
            }
        )
    valid = [i for i, x in enumerate(output) if x.get("p_value") is not None]
    adj = _bh(np.array([output[i]["p_value"] for i in valid], dtype=float))
    for i, padj in zip(valid, adj):
        output[i]["padj_bh"] = float(padj)
    return output
