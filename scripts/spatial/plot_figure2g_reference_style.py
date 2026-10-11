"""Plot Figure 2G: representative advanced-stage OSCC cell-type map."""

from __future__ import annotations

import numpy as np
import pandas as pd

import plot_figure2a_reference_style as base


base.OUTPUT = base.OUTPUT_DIR / "Fig2G_cell2location_advanced_OSCC"
base.AUDIT = base.OUTPUT_DIR / "Fig2G_representative_sample_selection.csv"
base.PANEL_LETTER = "G"
base.PANEL_TITLE = "cell2location-advanced OSCC"


def choose_advanced_representative(
    obs: pd.DataFrame, proportions: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    advanced = obs["stage"].astype(str).eq("Stage IVA")
    sample_means = proportions.loc[advanced].groupby(
        obs.loc[advanced, "sample_id"].astype(str)
    ).mean()
    metadata = (
        obs.loc[advanced, ["sample_id", "patient_id", "stage"]]
        .assign(
            sample_id=lambda frame: frame["sample_id"].astype(str),
            patient_id=lambda frame: frame["patient_id"].astype(str),
        )
        .drop_duplicates("sample_id")
        .set_index("sample_id")
    )

    clr = np.log(sample_means.clip(lower=1e-6))
    clr = clr.sub(clr.mean(axis=1), axis=0)
    patient_clr = clr.groupby(metadata["patient_id"]).mean()
    centroid = patient_clr.mean(axis=0)
    distances = np.sqrt(clr.sub(centroid, axis=1).pow(2).sum(axis=1))

    audit = metadata.join(distances.rename("aitchison_distance_to_patient_balanced_centroid"))
    audit = audit.join(sample_means.add_prefix("mean_"))
    audit = audit.sort_values("aitchison_distance_to_patient_balanced_centroid")
    audit.insert(0, "selected", False)
    audit.iloc[0, audit.columns.get_loc("selected")] = True
    return str(audit.index[0]), audit.reset_index()


base.choose_representative = choose_advanced_representative


if __name__ == "__main__":
    base.main()
