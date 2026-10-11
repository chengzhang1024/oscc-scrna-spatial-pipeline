"""Plot Figure 2H: marker maps for the Figure 2G advanced section."""

from __future__ import annotations

import plot_figure2b_reference_style as base


base.SELECTION = base.OUTPUT_DIR / "Fig2G_representative_sample_selection.csv"
base.OUTPUT = base.OUTPUT_DIR / "Fig2H_markers_advanced_OSCC"
base.PANEL_LETTER = "H"
base.PANEL_TITLE = "Markers-advanced OSCC"


if __name__ == "__main__":
    base.main()
