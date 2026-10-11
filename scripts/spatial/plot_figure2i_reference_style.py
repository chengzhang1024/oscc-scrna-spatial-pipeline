"""Plot Figure 2I from patient-aggregated Stage IVA MISTy results."""

from __future__ import annotations

import plot_figure2c_reference_style as base


base.INPUT = (
    base.ROOT
    / "results"
    / "spatial_misty"
    / "figure2i_advanced"
    / "advanced_misty_group_importance.csv"
)
base.OUTPUT = base.ROOT / "figures" / "paper9" / "figure2" / "Fig2I_MISTy_advanced_OSCC"
base.PANEL_LETTER = "I"


if __name__ == "__main__":
    base.main()
