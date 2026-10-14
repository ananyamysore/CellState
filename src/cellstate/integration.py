"""Cross-dataset comparison and optional expression-space ComBat correction."""

from __future__ import annotations

import anndata as ad
import pandas as pd
import scanpy as sc

from .metrics import signature_agreement


def diagnose_condition_batch_confounding(
    obs: pd.DataFrame, *, condition_key: str = "condition", batch_key: str = "batch"
) -> dict[str, object]:
    """Flag perfect one-to-one condition/batch confounding as non-identifiable."""
    missing = [k for k in (condition_key, batch_key) if k not in obs]
    if missing:
        return {
            "status": "inconclusive",
            "reason": f"missing metadata: {missing}",
            "confounded": None,
        }
    valid = obs[[condition_key, batch_key]].dropna()
    if valid.empty:
        return {
            "status": "inconclusive",
            "reason": "no complete condition/batch rows",
            "confounded": None,
        }
    cross = pd.crosstab(valid[condition_key].astype(str), valid[batch_key].astype(str))
    condition_to_batch = (cross.gt(0).sum(axis=1) <= 1).all()
    batch_to_condition = (cross.gt(0).sum(axis=0) <= 1).all()
    confounded = bool(
        condition_to_batch and batch_to_condition and cross.shape[0] > 1 and cross.shape[1] > 1
    )
    return {
        "status": "warning" if confounded else "ok",
        "confounded": confounded,
        "reason": "condition and batch are perfectly one-to-one; their effects cannot be separated"
        if confounded
        else None,
        "condition_batch_table": cross.to_dict(),
        "n_conditions": int(cross.shape[0]),
        "n_batches": int(cross.shape[1]),
    }


def combat_representation(
    adata: ad.AnnData, *, batch_key: str = "batch", condition_key: str = "condition"
) -> tuple[ad.AnnData, dict[str, object]]:
    """Optional ComBat representation stored separately from normalized expression.

    Refuses perfect condition/batch confounding. Raw counts and normalized ``X``
    are retained; corrected values are stored in ``layers['combat_logexpr']`` and
    PCA coordinates in ``obsm['X_combat_pca']``. Corrected data are not evidence
    that biological signal has been preserved.
    """
    diagnosis = diagnose_condition_batch_confounding(
        adata.obs, condition_key=condition_key, batch_key=batch_key
    )
    if diagnosis["confounded"] is None:
        return adata.copy(), {
            "status": "skipped",
            "reason": diagnosis["reason"],
            "confounding": diagnosis,
        }
    if diagnosis["confounded"]:
        return adata.copy(), {
            "status": "inconclusive",
            "reason": diagnosis["reason"],
            "confounding": diagnosis,
        }
    if batch_key not in adata.obs or adata.obs[batch_key].isna().any():
        return adata.copy(), {
            "status": "skipped",
            "reason": "valid batch labels unavailable",
            "confounding": diagnosis,
        }
    out = adata.copy()
    original = out.X.copy()
    try:
        sc.pp.combat(out, key=batch_key)
        corrected = out.X.copy()
        out.X = original
        out.layers["combat_logexpr"] = corrected
        n_comps = min(30, max(2, min(out.n_obs - 1, out.n_vars - 1)))
        # ``layer`` support varies across Scanpy versions. Temporarily expose the
        # corrected matrix as X, then restore the uncorrected normalized values.
        out.X = corrected
        sc.pp.pca(out, n_comps=n_comps, random_state=0)
        out.obsm["X_combat_pca"] = out.obsm["X_pca"].copy()
        del out.obsm["X_pca"]
        out.X = original
        return out, {
            "status": "ok",
            "method": "Scanpy ComBat on normalized log-expression",
            "batch_key": batch_key,
            "corrected_values_layer": "combat_logexpr",
            "representation": "X_combat_pca",
            "warning": "Batch correction can remove biological signal; assess with held-out donors and marker/signature preservation.",
        }
    except Exception as exc:
        return adata.copy(), {
            "status": "failed",
            "reason": f"ComBat failed: {type(exc).__name__}: {exc}",
            "confounding": diagnosis,
        }


def compare_signature_tables(
    tables: list[pd.DataFrame],
    dataset_ids: list[str],
    *,
    top_n: int = 100,
    cell_type_key: str = "cell_type",
) -> list[dict[str, object]]:
    """Compare only explicitly annotated matching cell types and same contrasts."""
    if len(tables) != len(dataset_ids):
        raise ValueError("tables and dataset_ids length must match")
    pairs = []
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            a, b = tables[i], tables[j]
            common = (
                sorted(set(a[cell_type_key].astype(str)) & set(b[cell_type_key].astype(str)))
                if not a.empty and not b.empty
                else []
            )
            if not common:
                pairs.append(
                    {
                        "dataset_a": dataset_ids[i],
                        "dataset_b": dataset_ids[j],
                        "status": "inconclusive",
                        "reason": "no shared explicit cell-type annotations or signatures",
                    }
                )
                continue
            for cell in common:
                aa = a[a[cell_type_key].astype(str) == cell]
                bb = b[b[cell_type_key].astype(str) == cell]
                ca = set(
                    zip(
                        aa.get("condition_a", pd.Series(dtype=str)),
                        aa.get("condition_b", pd.Series(dtype=str)),
                    )
                )
                cb = set(
                    zip(
                        bb.get("condition_a", pd.Series(dtype=str)),
                        bb.get("condition_b", pd.Series(dtype=str)),
                    )
                )
                if ca and cb and ca != cb:
                    pairs.append(
                        {
                            "dataset_a": dataset_ids[i],
                            "dataset_b": dataset_ids[j],
                            "cell_type": cell,
                            "status": "inconclusive",
                            "reason": "condition contrasts do not match",
                        }
                    )
                    continue
                metrics = signature_agreement(
                    aa["gene"].astype(str).tolist(),
                    aa["log2fc"].to_numpy(),
                    bb["gene"].astype(str).tolist(),
                    bb["log2fc"].to_numpy(),
                    top_n=top_n,
                )
                pairs.append(
                    {
                        "dataset_a": dataset_ids[i],
                        "dataset_b": dataset_ids[j],
                        "cell_type": cell,
                        "status": "descriptive",
                        "metrics": metrics,
                        "interpretation": "rank agreement is descriptive and does not establish biological replication",
                    }
                )
    return pairs
