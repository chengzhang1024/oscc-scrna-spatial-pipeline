"""Render Figure 5 and source-linked methods from completed program replication."""
from pathlib import Path
import json
import re
import hashlib
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
import fitz

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/paper9/figure5_program_replication"
FIG = ROOT / "figures/paper9/figure5_program_replication"
BLUE = "#397D9B"
RED = "#BC5264"
GREEN = "#4B977F"
INK = "#243444"
NICHE = {"N1": "#65BBA0", "N2": "#F7B52B", "N3": "#CF535E"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white", "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": INK})


def read(name):
    return pd.read_csv(RES / name)


def natural(s):
    return int(re.search(r"\d+", s).group())


def rect(base, x, y, w, h):
    a, b, c, d = base
    return [a+c*x, b+d*y, c*w, d*h]


def heading(fig, base, letter, title):
    x, y, w, h = base
    fig.text(x, y+h, letter, fontsize=19, weight="bold", va="top")
    fig.text(x+.032, y+h-.002, title, fontsize=12, weight="bold", va="top")


def pformat(v):
    return f"{v:.2g}" if v < .001 else f"{v:.3f}"


def save(fig, name, dpi=240):
    fig.savefig(FIG / (name+".pdf"))
    fig.savefig(FIG / (name+".png"), dpi=dpi)
    plt.close(fig)


def load():
    return dict(sig=json.loads((RES/"signatures.json").read_text(encoding="utf-8")),
        sc=read("replication_cell_scores.csv.gz"), genes=read("replication_check_gene_correlations.csv"),
        rho=read("replication_patient_effects.csv"), summary=read("replication_summary.csv").set_index("comparison"),
        baseline=json.loads((RES/"specificity_baseline.json").read_text(encoding="utf-8")),
        sp=read("spatial_spot_scores.csv.gz"), spatial=read("spatial_summary.csv"),
        spatial_patients=read("spatial_patient_effects.csv"))


def panel_a(fig, base, data):
    heading(fig, base, "A", "Discovery-frozen programs, independent-cohort replication and spatial support")
    ax = fig.add_axes(rect(base, .035, .02, .945, .72)); ax.axis("off")
    texts = [
        ("GSE181919 | DISCOVERY", "5,609 epithelial-compartment cells\n5 eligible patient-paired comparisons\n60 markers: 27 score + 29 check + 4 controls"),
        ("GSE310797 | REPLICATION", "24,000 epithelial cells | 16 patients\nFixed signatures; disjoint checking genes\nPatient-level effects; no forced C1 cluster"),
        ("GSE208253 | SPATIAL SUPPORT", "23,029 spots | 12 sections | 10 patients\nExisting Fig. 2 niches; mixed-spot scores\nAbundance-adjusted, patient-level summaries")]
    for i, (title, text) in enumerate(texts):
        x = i*.345
        ax.add_patch(FancyBboxPatch((x, .08), .307, .87, boxstyle="round,pad=0.009,rounding_size=0.02",
                    transform=ax.transAxes, facecolor=["#EEF4F8", "#EDF5F0", "#F8F1EB"][i], edgecolor="#D6DFE3"))
        ax.text(x+.014, .78, title, fontsize=11, weight="bold", transform=ax.transAxes)
        ax.text(x+.014, .57, text, fontsize=10, va="top", linespacing=1.6, transform=ax.transAxes)
        if i < 2:
            ax.annotate("", xy=(x+.338, .53), xytext=(x+.314, .53), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=1.5, color=INK))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def panel_b(fig, base, data, all_genes=False):
    heading(fig, base, "B", "Disjoint checking genes support a coordinated program")
    order = sorted(data["sc"].patient.unique(), key=natural)
    genes = data["sig"]["C1_check"] if all_genes else data["sig"]["C1_check"][:12]
    matrix = data["genes"].pivot(index="gene", columns="patient", values="rho").reindex(index=genes, columns=order)
    ax = fig.add_axes(rect(base, .17, .27, .72, .58))
    im = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(0, vmin=-1, vmax=1), aspect="auto")
    ax.set_xticks(range(len(order)), order, rotation=55, ha="right")
    ax.set_yticks(range(len(genes)), genes, fontstyle="italic")
    ax.tick_params(length=0, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ca = fig.add_axes(rect(base, .92, .40, .018, .35))
    cb = fig.colorbar(im, cax=ca); cb.set_ticks([-1, 0, 1]); cb.outline.set_visible(False)
    cb.set_label("Within-patient partial rho", fontsize=9)
    q = data["summary"].loc["C1_check_QC"]
    b = data["summary"].loc["C1_check_QC_biology"]
    baseline = data["baseline"]
    lines = [f"Disjoint score/check: median rho {q['median']:.2f} | {int(q.n_positive)}/{int(q.n_patients)} positive | q={pformat(q.bh_q)}",
             f"+ basal/cycle adjustment: rho {b['median']:.2f} | matched-null 95th: {baseline['null_95th']:.2f}",
             "First 12 checking genes in discovery rank order; all genes in supplement." if not all_genes else "All checking genes; adjusted for counts, detected genes and mitochondrial fraction."]
    for i, text in enumerate(lines):
        fig.text(*rect(base, .045, .145-i*.052, 0, 0)[:2], text, fontsize=9)


def panel_c(fig, base, data):
    heading(fig, base, "C", "Continuous program expression across patients")
    frame = data["sc"]
    order = sorted(frame.patient.unique(), key=natural)
    ax = fig.add_axes(rect(base, .13, .28, .83, .58))
    groups = [frame[frame.patient.eq(p)].C1_score.to_numpy() for p in order]
    bp = ax.boxplot(groups, positions=np.arange(len(order)), widths=.58, showfliers=False,
                    patch_artist=True, medianprops=dict(color=INK, lw=1.2),
                    whiskerprops=dict(color=BLUE, lw=.8), capprops=dict(color=BLUE, lw=.8))
    for patch in bp["boxes"]:
        patch.set(facecolor="#D7E7EE", edgecolor=BLUE, linewidth=.8)
    ax.set_xticks(range(len(order)), order, rotation=55, ha="right")
    ax.set_ylabel("C1 score\nmean log1p(CP10k)")
    ax.set_xlim(-.7, len(order)-.3)
    ax.grid(axis="y", alpha=.18)
    counts = fig.add_axes(rect(base, .13, .14, .83, .045)); counts.axis("off")
    counts.set_xlim(ax.get_xlim()); counts.set_ylim(0, 1)
    counts.text(-1.1, .5, "n", ha="right", va="center", fontsize=9)
    for i, values in enumerate(groups):
        counts.text(i, .5, str(len(values)), rotation=55, ha="right", va="center", fontsize=8)
    fig.text(*rect(base, .10, .025, 0, 0)[:2], "All 16 patients retained; no per-patient high-score cutoff imposed.", fontsize=9)


def panel_d(fig, base, data):
    heading(fig, base, "D", "Do the disjoint M1 and M5 programs retain their relationship with C1?")
    order = sorted(data["sc"].patient.unique(), key=natural)
    for j, (key, label, color) in enumerate([("C1_M1", "M1 | basal / adhesion", BLUE), ("C1_M5", "M5 | proliferation", RED)]):
        ax = fig.add_axes(rect(base, .07+j*.49, .24, .39, .55))
        tab = data["rho"][data["rho"].comparison.eq(key)].set_index("patient").reindex(order)
        s = data["summary"].loc[key]
        ax.axhline(0, color="#999999", ls="--", lw=.8)
        ax.axhspan(s.ci_low, s.ci_high, color=color, alpha=.10)
        ax.axhline(s["median"], color=color, lw=1.0)
        ax.scatter(range(len(order)), tab.rho, color=color, s=32, zorder=3)
        ax.set_xticks(range(len(order)), order, rotation=55, ha="right")
        ax.set_ylim(-.65, 1); ax.set_yticks([-.5, 0, .5, 1])
        ax.set_ylabel("QC-adjusted partial rho")
        ax.set_title(f"{label} | {len(data['sig']['M1' if j==0 else 'M5'])} non-overlapping genes", loc="left", fontsize=10)
        fig.text(*rect(base, .07+j*.49, .035, 0, 0)[:2],
                 f"Median {s['median']:.2f} [{s.ci_low:.2f}, {s.ci_high:.2f}] | {int(s.n_positive)}/16 positive | q={pformat(s.bh_q)}", fontsize=10)


def spatial_limits(data):
    return {s: tuple(np.quantile(data["sp"][s], [.02, .98])) for s in ["C1_score", "M1", "M5"]}


def spatial_map(ax, f, column, limits, dot=3):
    if column == "niche_id":
        ax.scatter(f.x, f.y, c=f.niche_id.map(NICHE), s=dot, linewidths=0, rasterized=True)
        im = None
    else:
        vmin, vmax = limits[column]
        im = ax.scatter(f.x, f.y, c=f[column], s=dot, cmap="viridis", vmin=vmin, vmax=vmax,
                        linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    return im


def panel_e(fig, base, data):
    heading(fig, base, "E", "Spatial localization in fixed representative sections")
    limits = spatial_limits(data)
    columns = ["niche_id", "C1_score", "M1", "M5"]
    labels = ["Fig. 2 niches", "C1 program", "M1 disjoint", "M5 disjoint"]
    for row, sample in enumerate(["S6", "S2"]):
        f = data["sp"][data["sp"].sample_id.eq(sample)]
        bottom = .48 if row == 0 else .14
        for j, (column, label) in enumerate(zip(columns, labels)):
            ax = fig.add_axes(rect(base, .04+j*.242, bottom, .226, .30))
            im = spatial_map(ax, f, column, limits, dot=4)
            if row == 0:
                ax.set_title(label, fontsize=10, pad=5)
            if j == 0:
                ax.text(-.03, .5, f"{sample} | {f.patient_id.iloc[0]}\n{f.stage.iloc[0]}", rotation=90,
                        transform=ax.transAxes, ha="right", va="center", fontsize=9)
            if row == 1 and im is not None:
                ca = fig.add_axes(rect(base, .09+j*.242, .087, .12, .012))
                cb = fig.colorbar(im, cax=ca, orientation="horizontal")
                cb.set_ticks(np.round(limits[column], 1)); cb.ax.tick_params(labelsize=8, length=2)
                cb.outline.set_visible(False)
    for i, color in enumerate(NICHE.values()):
        fig.text(*rect(base, .03+i*.32, .015, 0, 0)[:2],
                 ["N1 fibrovascular", "N2 immune-mixed", "N3 epithelial-rich"][i], color=color, fontsize=9, weight="bold")


def panel_f(fig, base, data):
    heading(fig, base, "F", "Abundance-adjusted niche associations")
    primary = data["spatial"][data["spatial"]["mode"].eq("primary")].set_index(["signature", "contrast"])
    sensitivity = data["spatial"][data["spatial"]["mode"].eq("spatial_trend")].set_index(["signature", "contrast"])
    labels, values = [], []
    for sig in ["C1_score", "M1", "M5"]:
        for contrast in ["N2-N1", "N3-N1"]:
            labels.append(("C1" if sig == "C1_score" else sig)+"  "+contrast.replace("-", " vs "))
            values.append((sig, contrast))
    ax = fig.add_axes(rect(base, .23, .32, .49, .46))
    all_ci = []
    for i, key in enumerate(values):
        if key not in primary.index:
            continue
        s = primary.loc[key]
        color = [GREEN, BLUE, RED][i//2]
        ax.plot([s.ci_low, s.ci_high], [5-i, 5-i], color=color, lw=1.3)
        ax.scatter(s["median"], 5-i, color=color, s=32, zorder=3)
        all_ci += [s.ci_low, s.ci_high]
        if key in sensitivity.index:
            t = sensitivity.loc[key]
            ax.scatter(t["median"], 5-i-.19, marker="D", facecolors="none", edgecolors=color, s=24)
        shown_q = pformat(s.bh_q) if s.sufficient_patients else "descriptive"
        ax.text(1.03, (5-i+.5)/6.5, f"{int(s.n_patients)} | {shown_q}", transform=ax.transAxes,
                va="center", fontsize=9)
    ax.axvline(0, color="#888888", ls="--", lw=.8)
    lim = max(.3, np.max(np.abs(all_ci))*1.18) if all_ci else 1.
    ax.set_xlim(-lim, lim); ax.set_ylim(-.5, 6)
    ax.set_yticks(range(5, -1, -1), labels)
    ax.set_xlabel("Adjusted difference\n(section SD units)")
    ax.tick_params(axis="y", length=0)
    ax.text(1.03, 1.015, "patients | q", transform=ax.transAxes, fontsize=9)
    notes = ["Dots + bars: patient median and 95% interval",
             "Open diamonds: + quadratic spatial trend",
             "<6 patients: descriptive, not an inferential claim",
             "Patient resampling; no independent-spot P values"]
    for i, line in enumerate(notes):
        fig.text(*rect(base, .04, .135-i*.043, 0, 0)[:2], line, fontsize=9)


PANELS = {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d, "E": panel_e, "F": panel_f}


def supplementary(data):
    path = FIG / "Figure5_spatial_all_sections.pdf"
    lim = spatial_limits(data)
    with PdfPages(path) as pdf:
        for sample in sorted(data["sp"].sample_id.unique(), key=natural):
            frame = data["sp"][data["sp"].sample_id.eq(sample)]
            fig, axes = plt.subplots(1, 5, figsize=(15, 4.5))
            columns = ["niche_id", "C1_score", "M1", "M5", "epithelial_abundance"]
            limits = dict(lim, epithelial_abundance=tuple(np.quantile(data["sp"].epithelial_abundance, [.02, .98])))
            for ax, column in zip(axes, columns):
                im = spatial_map(ax, frame, column, limits, dot=5)
                ax.set_title(column.replace("_", " "))
                if im is not None:
                    fig.colorbar(im, ax=ax, shrink=.6, orientation="horizontal", pad=.05)
            fig.suptitle(f"{sample} | {frame.patient_id.iloc[0]} | {frame.stage.iloc[0]} | {len(frame):,} spots\nObserved mixed-spot programs; identical per-program scales across sections", fontsize=12)
            fig.subplots_adjust(top=.78, bottom=.05, left=.02, right=.98)
            pdf.savefig(fig); plt.close(fig)
    with PdfPages(FIG / "Figure5_replication_details.pdf") as pdf:
        fig = plt.figure(figsize=(13, 12)); panel_b(fig, [.015, .02, .97, .96], data, all_genes=True)
        pdf.savefig(fig); plt.close(fig)
        discovery = read("discovery_signature_scores.csv.gz")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, col in zip(axes, ["epithelial_cluster", "disease_status"]):
            ds = discovery.groupby(["patient_id", col]).C1_score.median().reset_index()
            names = sorted(ds[col].unique())
            rng = np.random.default_rng(20260916)
            for i, name in enumerate(names):
                y = ds.loc[ds[col].eq(name), "C1_score"]
                ax.scatter(i+rng.uniform(-.13, .13, len(y)), y, s=25, color=BLUE, alpha=.75)
                ax.plot([i-.25, i+.25], [y.median()]*2, color=INK, lw=2)
            ax.set_xticks(range(len(names)), names); ax.set_ylabel("Patient-group median C1 score")
            ax.set_title("Discovery: "+col.replace("_", " "))
        fig.suptitle("Descriptive discovery context; no independent disease-specificity claim", fontsize=12)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        null = read("replication_matched_null_correlations.csv").groupby("null_signature").rho.median()
        ax.hist(null, bins=18, color="#A6BDC7", edgecolor="white")
        ax.axvline(data["baseline"]["observed_patient_median"], color=RED, lw=2, label="Disjoint checking set")
        ax.axvline(data["baseline"]["null_95th"], color=INK, ls="--", label="Matched-null 95th percentile")
        ax.set_xlabel("Patient-median partial rho (QC + basal/cycle adjusted)")
        ax.set_ylabel("Matched checking sets")
        ax.set_title("Discovery-matched specificity baseline (100 sets)")
        ax.legend(frameon=False); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    patient_order = sorted(data["sp"].patient_id.unique(), key=natural)
    modes = [("primary", "o", BLUE, -.15), ("spatial_trend", "D", RED, 0), ("proportion", "x", GREEN, .15)]
    for i, signature in enumerate(["C1_score", "M1", "M5"]):
        for j, contrast in enumerate(["N2-N1", "N3-N1"]):
            ax = axes[i, j]
            for mode, marker, color, offset in modes:
                t = data["spatial_patients"]
                t = t[t['mode'].eq(mode) & t.signature.eq(signature) & t.contrast.eq(contrast)].set_index("patient_id")
                for k, patient in enumerate(patient_order):
                    if patient in t.index:
                        ax.scatter(k+offset, t.loc[patient, "coefficient"], color=color, marker=marker, s=30)
            ax.axhline(0, color="#999999", ls="--", lw=.7)
            ax.set_title(("C1" if signature == "C1_score" else signature)+" | "+contrast)
            ax.set_ylabel("Patient mean coefficient\n(section SD units)")
            ax.set_xticks(range(len(patient_order)), patient_order, rotation=45)
            ax.set_xlim(-.5, len(patient_order)-.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=m, color=c, lw=0, label=n.replace("_", " ")) for n, m, c, _ in modes]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .962))
    fig.suptitle("Spatial associations by patient and analysis choice", fontsize=14, y=.99)
    fig.text(.08, .016, "Missing dots indicate insufficient within-section common support or a missing niche; no zero effect is imputed.", fontsize=10)
    fig.subplots_adjust(left=.08, right=.97, bottom=.08, top=.90, hspace=.28, wspace=.28)
    fig.savefig(FIG / "Figure5_spatial_effect_details.pdf"); plt.close(fig)
    pd.DataFrame([dict(signature=k, lower=v[0], upper=v[1],
                       fraction_clipped=float(((data["sp"][k]<v[0])|(data["sp"][k]>v[1])).mean()))
                  for k, v in lim.items()]).to_csv(RES / "spatial_display_limits.csv", index=False)


def report(data):
    sig = data["sig"]
    s = data["summary"]
    manifest = json.loads((RES/"frozen_manifest.json").read_text(encoding="utf-8"))
    lines = ["# 新Figure 5：上皮程序的独立队列重复与空间支持", "",
        "这是探索性程序复现分析，不是预后分析；未修改历史图5、bulk输入或生存模型。", "",
        "## 本次签名与独立性", "",
        "重新在GSE181919按患者平衡筛选C1描述性marker，排除C5/C8。仅P6、P15、P21、P31、P51满足两组各>=10细胞；这只有5位可配对患者，是重要限制。",
        f"固定前60个一致方向候选，按发现排名奇偶分成评分与核验两半；移除与基底/增殖控制集重叠的4个基因后，评分使用{len(sig['C1_score'])}基因，核验使用{len(sig['C1_check'])}基因。完整名单在frozen_signature_genes.csv。",
        f"正式hdWGCNA M1/M5分别为295/217基因，去除全部60个C1候选后，主分析分别使用{len(sig['M1'])}/{len(sig['M5'])}基因。没有使用旧120基因并集、旧两基因模型或四基因模型作为答案。",
        "该签名是患者平衡的描述性候选，不冒称60个显著DEG。评分与核验基因互不重叠，但都在同一发现队列选出；真正的支持来自不同患者队列中的关系。",
        "逐患者剔除时，60个已选基因均继续满足方向/检出率条件，但这不等于特征列表已在全新发现队列中验证。发现队列中C2等亚群也有较高评分，因此本图定位为C1相关表达程序支持，而非C1特异亚群鉴定。患者bootstrap区间以固定签名为条件，不包含重新筛选签名的不确定性。",
        "GSE310797已用于整体图谱且此前已分析过C1/M1/M5关系，本次为新增探索性独立队列表达支持，不能称为未触碰的确认性验证。", "",
        "## 单细胞结果", "",
        "采用24,000个上皮细胞、16位患者。所有患者保留，包括细胞数较少者；另提供>=100细胞敏感性分析。每位患者内做秩残差相关，调整总counts、检出基因数和线粒体比例。统计单位是患者。", "",
        "|检验|患者内rho中位数（患者bootstrap 95%区间）|正方向患者|BH q|",
        "|---|---|---|---|"]
    labels = {"C1_check_QC":"评分集—核验集（QC调整）", "C1_check_QC_biology":"评分集—核验集（另调整基底/增殖程序）", "C1_M1":"C1—M1去重叠", "C1_M5":"C1—M5去重叠"}
    for key, label in labels.items():
        r = s.loc[key]
        lines.append(f"|{label}|{r['median']:.3f}（{r.ci_low:.3f}至{r.ci_high:.3f}）|{int(r.n_positive)}/{int(r.n_patients)}|{r.bh_q:.4g}|")
    b = data["baseline"]
    lines += ["", f"发现队列表达量/检出率匹配的100组随机核验集：患者中位rho的95百分位={b['null_95th']:.3f}；真实核验集={b['observed_patient_median']:.3f}；经验上尾比例={b['empirical_upper_tail']:.4f}。这只评估程序特异性的一个基线，不证明癌特异性。", "",
        "## 空间结果", "",
        "GSE208253沿用图2已有niche：23,029 spots、12切片、10患者。原始混合spot表达经过归一化后评分。代表切片预先沿用图2选择的S6/S2；全12切片均在补充PDF中，未按新结果挑图。",
        "比较N2/N3相对于N1的程序差异；限制到两组上皮丰度10%-90%范围的交集，要求每组>=30 spots；调整上皮丰度样条、测序复杂度和线粒体比例。系数单位为切片内程序分数标准差。重复切片先合并至患者，患者等权。置信区间按整患者bootstrap；P为患者方向符号检验，六项统一BH。没有逐spot独立推断。", "",
        "|程序|niche比较|患者数|调整后差异中位数（95%区间）|BH q|",
        "|---|---|---|---|---|"]
    for _, r in data["spatial"][data["spatial"]["mode"].eq("primary")].iterrows():
        shown_q = f"{r.bh_q:.4g}" if r.sufficient_patients else "不足6患者，仅描述"
        lines.append(f"|{r.signature}|{r.contrast}|{int(r.n_patients)}|{r['median']:.3f}（{r.ci_low:.3f}至{r.ci_high:.3f}）|{shown_q}|")
    section = read("spatial_section_effects.csv")
    primary = section[section['mode'].eq('primary')]
    lines += ["", f"主分析切片×程序×对比中，{int((~primary.eligible).sum())}/{len(primary)}项因共同支持/可识别性不足未估计，完整失败原因保留。",
        "空间敏感性：增加二维坐标二次趋势（主图空心菱形）以及改用上皮比例，见spatial_summary.csv。M1的N2-N1方向在空间趋势调整后仍一致，但改用上皮比例后BH不显著；因此不能写成不依赖建模方式的稳健定位结论。C1没有获得一致的调整后niche关联。N3-N1仅4位患者可比较，按预设要求仅作描述。",
        "晚期保留既有投影niche归属，包括超出早期参考分布的spot；out_of_distribution标签保存在源表。调整后仍可能受spot混合、组成共线性、空间趋势和反卷积误差影响；不解释为上皮内在表达或因果生态位效应。", "",
        "## 图注", "",
        "A，发现—重复—空间支持设计及固定特征。B，未参与评分的核验基因与C1评分的患者内QC调整相关；主图显示按发现排名的前12个核验基因，全基因见补充。C，16患者连续C1分数分布及细胞数，不强行定义高低比例。D，C1与去重叠M1/M5评分的患者内关联；横线和阴影为患者中位数及bootstrap区间。E，既有代表切片中的固定程序与niche图，程序色阶按全12切片统一；这是混合spot空间支持。F，上皮丰度共同支持区域内的患者层niche系数，实点/线为主分析，空心菱形加入二维空间趋势。", "",
        "## 结论边界", "",
        "支持程度按结果逐项判断，不要求M1/M5同时阳性。高分梯度不是离散C1亚群重现；分数关系不是完整hdWGCNA网络保留；基底上皮程序重复不是癌特异、干性或白斑癌变预测的证明。对未获得稳定支持的结果不解释为已证实的情境依赖性。",
        "现有数据不足以通过一张图证明整条生物学机制链。图5用于量化哪些表达关系跨患者重复、哪些空间关联仍不确定。", "",
        "## 文件与重现", "",
        "主图：figures/paper9/figure5_program_replication/Figure5_program_replication.pdf/png；另有A-F单图、全切片空间PDF和全核验基因/匹配基线PDF。", "",
        "```powershell", "& '.envs/figure5_py/Scripts/python.exe' -X utf8 scripts/56_figure5_program_replication.py all",
        "& '.envs/figure5_py/Scripts/python.exe' -X utf8 scripts/57_plot_figure5_program_replication.py", "```",
        "", "analysis_plan.md在新终点计算前写入；frozen_manifest.json在外部评分前保存定义与来源hash。validation_report.json记录数值和图件检查。"]
    (RES / "README_方法结果与局限.md").write_text("\n".join(lines), encoding="utf-8")
    (FIG / "README.md").write_text("# Figure 5: program replication and spatial support\n\nOpen Figure5_program_replication.pdf or .png first. Panels A-F are available separately. Figure5_replication_details.pdf contains all checking genes and discovery context; Figure5_spatial_all_sections.pdf includes all 12 sections. Methods, source tables and limitations are in results/paper9/figure5_program_replication/README_方法结果与局限.md. Historical prognostic figures are unchanged.\n", encoding="utf-8")


def audit_pdfs():
    rows = []
    for path in sorted(FIG.glob("*.pdf")):
        d = fitz.open(path)
        out, minsize = [], []
        for i, p in enumerate(d):
            for block in p.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for s in line.get("spans", []):
                        if s["text"].strip():
                            minsize.append(s["size"])
                            if not (p.rect+(-1, -1, 1, 1)).contains(fitz.Rect(s["bbox"])):
                                out.append(dict(page=i+1, text=s["text"]))
        rows.append(dict(file=path.name, pages=len(d), min_font_pt=min(minsize) if minsize else None, outside_page=out))
        d.close()
    (RES / "figure_layout_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    assert not any(r["outside_page"] for r in rows), "Text outside PDF page; inspect figure_layout_audit.json"
    report = json.loads((RES / "validation_report.json").read_text(encoding="utf-8"))
    report["figures"] = rows
    report["plot_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["print_limitation"] = "Main figure is a large review plate; individual panel PDFs supplied. Journal-width reduction may require splitting panels."
    (RES / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = load()
    fig = plt.figure(figsize=(16, 17.5))
    fig.text(.03, .989, "Epithelial program replication and spatial support", fontsize=18, weight="bold", va="top")
    panel_a(fig, [.025, .853, .95, .108], data)
    panel_b(fig, [.025, .566, .465, .26], data)
    panel_c(fig, [.515, .566, .46, .26], data)
    panel_d(fig, [.025, .375, .95, .178], data)
    panel_e(fig, [.025, .071, .56, .287], data)
    panel_f(fig, [.602, .071, .373, .287], data)
    fig.text(.035, .037, "Exploratory program-level support | Gene sets fixed in discovery; GSE310797 had been inspected previously.", fontsize=10)
    fig.text(.035, .022, "Only five eligible discovery patient pairs | Spatial signals are mixed-spot measurements, not epithelial-intrinsic or causal effects.", fontsize=10)
    save(fig, "Figure5_program_replication")
    for letter, func in PANELS.items():
        size = {"A": (15, 2.8), "B": (10, 7), "C": (10, 6), "D": (14, 5), "E": (12, 8), "F": (9, 7)}[letter]
        fig = plt.figure(figsize=size)
        func(fig, [.025, .03, .95, .93], data)
        save(fig, "Fig5"+letter+"_program_replication")
    supplementary(data)
    report(data)
    audit_pdfs()
    print("FIGURE 5 AND METHODS GENERATED", flush=True)


if __name__ == "__main__":
    main()
