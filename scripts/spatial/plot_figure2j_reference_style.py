"""Plot Figure 2J: abundance maps for the Figure 2G advanced section."""

from __future__ import annotations

import plot_figure2d_reference_style as base


base.SELECTION = base.OUTPUT_DIR / "Fig2G_representative_sample_selection.csv"
base.OUTPUT = base.OUTPUT_DIR / "Fig2J_cell2location_abundance_advanced_OSCC"
base.SCALE_AUDIT = base.OUTPUT_DIR / "Fig2J_abundance_scale_limits.csv"
base.PANEL_LETTER = "J"
base.PANEL_TITLE = "Abundance-advanced OSCC"


if __name__ == "__main__":
    base.main()
