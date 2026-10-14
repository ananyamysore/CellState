"""Sample-level pseudobulk signatures for condition comparisons."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving the input order."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if not len(pv):
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    tmp = np.empty(m)
    tmp[order] = np.clip(adj, 0, 1)
    out[valid] = tmp
    return out


def pseudobulk_signatures(
    adata: ad.AnnData,
    *,
    condition_key: str = "condition",
    cell_type_key: str = "cell_type",
    sample_key: str = "sample_id",
    donor_key: str = "donor_id",
    group_a: str | None = None,
    group_b: str | None = None,
    min_replicates: int = 2,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Estimate condition-associated genes from donor/sample pseudobulk expression.

    Cells are summed within independent donor (preferred) or sample, condition,
    and annotated cell type, then converted to log2 counts-per-million. Welch's
    t-test is used for unpaired donors; paired donors use paired t-tests. This is
    a transparent screening method, not a substitute for a count-model DE method
    in a powered study. Fewer than ``min_replicates`` biological units per group
    yields an empty table marked inconclusive.
    """
    for key in (condition_key, cell_type_key):
        if key not in adata.obs:
            raise ValueError(f"required metadata field {key!r} is missing")
    if "counts" not in adata.layers:
        raise ValueError("raw counts layer is required for pseudobulk")
    unit_key = (
        donor_key if donor_key in adata.obs and adata.obs[donor_key].notna().all() else sample_key
    )
    if unit_key not in adata.obs or adata.obs[unit_key].isna().any():
        return pd.DataFrame(
            columns=["cell_type", "gene", "log2fc", "p_value", "padj", "n_a", "n_b", "method"]
        ), {
            "status": "inconclusive",
            "reason": "independent sample or donor identifiers are missing",
        }
    obs = adata.obs[[condition_key, cell_type_key, unit_key]].copy()
    if obs[[condition_key, cell_type_key, unit_key]].isna().any().any():
        raise ValueError("condition, cell type, and replicate IDs cannot be missing")
    conditions = list(pd.unique(obs[condition_key].astype(str)))
    if group_a is None or group_b is None:
        if len(conditions) != 2:
            return pd.DataFrame(), {
                "status": "inconclusive",
                "reason": "provide group_a/group_b when there are not exactly two conditions",
                "conditions": conditions,
            }
        group_a, group_b = conditions
    group_a, group_b = str(group_a), str(group_b)
    if group_a == group_b or group_a not in conditions or group_b not in conditions:
        raise ValueError("contrast groups must be distinct observed conditions")
    matrix = adata.layers["counts"]
    result = []
    rep_counts = {}
    for cell_type in pd.unique(obs[cell_type_key].astype(str)):
        sub = np.flatnonzero(obs[cell_type_key].astype(str).to_numpy() == cell_type)
        # Pseudobulk within biological unit and condition. Donor-level aggregation avoids pseudoreplication.
        keys = obs.iloc[sub].assign(
            _unit=obs.iloc[sub][unit_key].astype(str),
            _cond=obs.iloc[sub][condition_key].astype(str),
        )
        groups = keys.groupby(["_unit", "_cond"], observed=True).indices
        pb = []
        for (unit, cond), local in groups.items():
            idx = sub[np.asarray(local, dtype=int)]
            sums = np.asarray(matrix[idx].sum(axis=0)).ravel().astype(float)
            lib = sums.sum()
            logcpm = np.log2(sums / lib * 1e6 + 0.5) if lib > 0 else np.full(adata.n_vars, np.nan)
            pb.append({"unit": unit, "condition": cond, "values": logcpm})
        a = {r["unit"]: r["values"] for r in pb if r["condition"] == group_a}
        b = {r["unit"]: r["values"] for r in pb if r["condition"] == group_b}
        n_a, n_b = len(a), len(b)
        rep_counts[cell_type] = {group_a: n_a, group_b: n_b}
        if min(n_a, n_b) < min_replicates:
            continue
        paired = sorted(set(a) & set(b))
        if len(paired) >= min_replicates:
            av = np.stack([a[x] for x in paired])
            bv = np.stack([b[x] for x in paired])
            method = "paired_pseudobulk_t_test"
            n_a = n_b = len(paired)
            p = stats.ttest_rel(bv, av, axis=0, nan_policy="omit").pvalue
        else:
            av = np.stack(list(a.values()))
            bv = np.stack(list(b.values()))
            method = "welch_pseudobulk_t_test"
            p = stats.ttest_ind(bv, av, axis=0, equal_var=False, nan_policy="omit").pvalue
        fc = np.nanmean(bv, axis=0) - np.nanmean(av, axis=0)
        padj = _bh(np.asarray(p))
        result.extend(
            {
                "cell_type": cell_type,
                "gene": str(gene),
                "log2fc": float(fc[i]),
                "p_value": float(p[i]),
                "padj": float(padj[i]),
                "n_a": int(n_a),
                "n_b": int(n_b),
                "method": method,
                "condition_a": group_a,
                "condition_b": group_b,
            }
            for i, gene in enumerate(adata.var_names)
            if np.isfinite(p[i])
        )
    table = pd.DataFrame(result)
    status = "ok" if not table.empty else "inconclusive"
    reason = (
        None
        if status == "ok"
        else f"insufficient independent units; counts by annotated cell type: {rep_counts}"
    )
    return table, {
        "status": status,
        "condition_a": group_a,
        "condition_b": group_b,
        "unit_key": unit_key,
        "unit_level": ("donor" if unit_key == donor_key else "sample"),
        "minimum_replicates": min_replicates,
        "replicates_by_cell_type": rep_counts,
        "method": "sample/donor pseudobulk log2CPM with paired or Welch t-tests; BH correction",
        "reason": reason,
    }
