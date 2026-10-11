"""Plot Figure 2K from patient-aggregated Stage IVA MISTy intra relationships."""

from __future__ import annotations

import pandas as pd

import plot_figure2e_reference_style as base


RESULT_DIR = base.ROOT / "results" / "spatial_misty" / "figure2i_advanced"
base.INPUT = RESULT_DIR / "advanced_misty_group_importance.csv"
base.SAMPLE_INPUT = RESULT_DIR / "advanced_misty_interactions_by_patient.csv"
base.OUTPUT = base.OUTPUT_DIR / "Fig2K_MISTy_intra_network_advanced_OSCC"
base.EDGE_AUDIT = base.OUTPUT_DIR / "Fig2K_MISTy_intra_network_edges.csv"
base.NODE_AUDIT = base.OUTPUT_DIR / "Fig2K_MISTy_intra_network_nodes.csv"
STAGE_AUDIT = base.OUTPUT_DIR / "Fig2K_MISTy_intra_network_stage_comparison.csv"
base.PANEL_LETTER = "K"
base.STABLE_TIER_LABEL = "stable_5of6"
base.CANDIDATE_TIER_LABEL = "candidate_3or4of6"
base.EDGE_NOTE = (
    "Intra-view median importance z >= 1; solid edges have patient-level z >= 1 "
    "in >=5/6 patients, dashed edges in 3-4/6 patients; arrow points predictor to "
    "target; reciprocal arrows are curved apart; background modules use weighted "
    "Louvain community detection (seed 181919)"
)


def write_stage_comparison() -> None:
    early = pd.read_csv(base.OUTPUT_DIR / "Fig2E_MISTy_intra_network_edges.csv")
    advanced = pd.read_csv(base.EDGE_AUDIT)
    columns = ["predictor", "target", "median_importance_z", "stability_fraction", "evidence_tier"]
    comparison = early[columns].rename(
        columns={column: f"early_{column}" for column in columns[2:]}
    ).merge(
        advanced[columns].rename(
            columns={column: f"advanced_{column}" for column in columns[2:]}
        ),
        on=["predictor", "target"],
        how="outer",
    )
    early_present = comparison["early_evidence_tier"].notna()
    advanced_present = comparison["advanced_evidence_tier"].notna()
    early_stable = comparison["early_evidence_tier"].astype(str).str.startswith("stable")
    advanced_stable = comparison["advanced_evidence_tier"].astype(str).str.startswith("stable")
    comparison["stage_pattern"] = "displayed_both"
    comparison.loc[~early_present & advanced_present, "stage_pattern"] = "advanced_only"
    comparison.loc[early_present & ~advanced_present, "stage_pattern"] = "early_only"
    comparison.loc[early_stable & advanced_stable, "stage_pattern"] = "conserved_stable"
    comparison.loc[
        early_stable & advanced_present & ~advanced_stable, "stage_pattern"
    ] = "stable_to_candidate"
    comparison.to_csv(STAGE_AUDIT, index=False)
    print(f"Saved {STAGE_AUDIT}")


if __name__ == "__main__":
    base.main()
    write_stage_comparison()
