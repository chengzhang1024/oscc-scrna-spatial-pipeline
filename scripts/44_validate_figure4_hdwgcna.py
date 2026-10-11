"""Check input identity, module coverage, displayed statistics, and exported figures."""
import hashlib
import json
from pathlib import Path
import fitz
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/paper9/figure4_epithelial_hdwgcna"
FIG = ROOT / "figures/paper9/figure4_epithelial_hdwgcna"
cells = pd.read_csv(RES / "input/cells.csv")
members = pd.read_csv(RES / "metacell_membership.csv")
mods = pd.read_csv(RES / "gene_modules.csv")
meta = json.loads((RES / "analysis_metadata.json").read_text(encoding="utf-8"))
assert len(cells) == 5609 and cells.r_cell_id.is_unique
for key in ["sample_id", "patient_id", "disease_status", "epithelial_cluster"]:
    assert members.groupby("metacell_id")[key].nunique().eq(1).all(), key
assert members.groupby("metacell_id").size().eq(meta["k"]).all()
assert members.r_cell_id.isin(cells.r_cell_id).all()
max_shared = 0
for _, group in members.groupby(["sample_id", "epithelial_cluster"]):
    sets = group.groupby("metacell_id").r_cell_id.apply(set).tolist()
    for i, a in enumerate(sets):
        for b in sets[:i]:
            max_shared = max(max_shared, len(a & b))
assert max_shared <= meta["max_shared"]
assert len(mods) == meta["n_genes"] and mods.gene_name.is_unique
before = pd.read_csv(RES / "modules_before_connectivity.csv")
assert set(mods.loc[mods.module.eq("grey"), "gene_name"]) == set(before.loc[before.module.eq("grey"), "gene_name"])
me = pd.read_csv(RES / "cell_module_eigengenes.csv").set_index("r_cell_id")
assert set(me.index) == set(cells.r_cell_id) and np.isfinite(me.to_numpy()).all()
go = pd.read_csv(RES / "GO_BP_2023_display.csv")
assert go.fdr.lt(.05).all() and go.overlap.ge(3).all()
selection = pd.read_csv(RES / "representative_module_selection.csv")
assert not selection.peak_cluster.isin(["C5", "C8"]).any()
pdfs = sorted(FIG.glob("*.pdf"))
assert len(pdfs) == 7
report = []
for path in pdfs:
    doc = fitz.open(path)
    assert len(doc) == 1
    page = doc[0]
    assert all(f[2] != "Type3" for f in page.get_fonts())
    outside = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                box = fitz.Rect(span["bbox"])
                if not (page.rect + (-1,-1,1,1)).contains(box):
                    outside.append(span["text"])
    assert not outside, (path.name, outside)
    png = np.asarray(Image.open(path.with_suffix(".png")).convert("RGB"))
    nonwhite = float(np.mean(np.min(png,axis=2)<245))
    assert .01 < nonwhite < .8 and min(png.shape[:2]) > 1000
    report.append(dict(file=path.name,pages=1,type3_fonts=0,text_outside_page=0,
                       png_width=png.shape[1],png_height=png.shape[0],nonwhite_fraction=nonwhite,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    doc.close()
summary = dict(status="passed",n_cells=len(cells),n_metacells=members.metacell_id.nunique(),
               n_unique_cells_in_network=members.r_cell_id.nunique(),mixed_identity_metacells=0,
               max_shared_cells=max_shared,n_network_genes=len(mods),n_grey_genes=int(mods.module.eq("grey").sum()),
               n_GO_display_terms=len(go),all_display_GO_FDR_lt_005=True,figures=report)
(RES/"validation_report.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
