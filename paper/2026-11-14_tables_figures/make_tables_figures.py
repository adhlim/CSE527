"""Build paper Table 1, Table 2 and Figure 2 from existing result CSVs. No new experiments.
Run from anywhere:  python make_tables_figures.py
"""
import csv, glob, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "..", "experiments")
SRC = {
    "decomp": os.path.join(EXP, "2026-11-07_loss_decomposition", "results", "summary.csv"),
    "matrix": os.path.join(EXP, "2026-11-08_accuracy_matrix", "results", "summary.csv"),
    "speed": glob.glob(os.path.join(EXP, "2026-11-08_fair_speed", "results_colab*", "results_colab*", "summary.csv"))[0],
    "scale": [os.path.join(EXP, "2026-11-08_scaling", f"results_laptop_run{i}", "summary.csv") for i in (1, 2)],
}
rd = lambda p: list(csv.DictReader(open(p, newline="")))
DS = [("HPatches", "HPatches"), ("COCO_synthetic", "COCO synthetic")]

def write(name, header, rows):
    with open(os.path.join(HERE, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with open(os.path.join(HERE, name + ".md"), "w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows: f.write("| " + " | ".join(str(x) for x in r) + " |\n")

# ---------- Table 1: loss decomposition (precision@3px, validation) ----------
dec = {(r["dataset"], int(r["bins"]), r["method"]): float(r["precision_3px"]) for r in rd(SRC["decomp"])}
rows = []
for ds, dsn in DS:
    for b in (8, 16):
        A, B, E, D = (dec[(ds, b, m)] for m in ("A_float_cosine", "B_recon_cosine", "E_bin_L2", "D_bin_L1"))
        tot, q, d = A - D, A - B, B - D
        rows.append([dsn, b, f"{A:.2f}", f"{B:.2f}", f"{E:.2f}", f"{D:.2f}", f"{tot:.2f}",
                     f"{q:.2f} ({100*q/tot:.0f}%)", f"{d:.2f} ({100*d/tot:.0f}%)"])
write("table1_loss_decomposition",
      ["Dataset", "Bins", "A float cos", "B recon cos", "E bin L2", "D bin L1 (thermometer)",
       "Total loss A-D (pp)", "Quantization A-B (pp, share)", "Distance B-D (pp, share)"], rows)

# ---------- Table 2: memory and speed ----------
mat = rd(SRC["matrix"])
def acc(ds, strat, lv, dist):
    for r in mat:
        if r["dataset"] == ds and r["strategy"] == strat and int(r["levels"]) == lv and r["distance"] == dist:
            return float(r["precision_3px"])
sp = {(r["dataset"], r["method"]): r for r in rd(SRC["speed"])}
sc = defaultdict(list)
for p in SRC["scale"]:
    for r in rd(p):
        if int(r["N"]) >= 65536: sc[r["method"]].append(float(r["speedup_vs_float32_blas"]))
N = 524288
# label, bytes, accuracy source, fair_speed method, scaling method
M = [("float32 (cosine)", 256, ("matrix", "float", 0, "cosine"), "float_blas_single_gemm", "float32_blas"),
     ("int8 per-tensor (dot)", 64, ("speed", "int8_dot"), "int8_dot", "int8_dot"),
     ("8-level quantile, byte, L2", 64, ("matrix", "quantile", 8, "idx_L2"), "byte8_L2", "byte8_L2"),
     ("8-level quantile, byte, L1", 64, ("matrix", "quantile", 8, "idx_L1"), "byte8_L1", "byte8_L1"),
     ("8-level quantile, thermometer, L1", 56, ("matrix", "quantile", 8, "idx_L1"), "thermo8_L1_original", "thermo8_L1"),
     ("16-level percentile, 4-bit packed, L2", 32, ("matrix", "percentile", 16, "idx_L2"), None, "nib16_L2_packed")]
rows = []
for lab, by, a, fs, ss in M:
    r = [lab, by, f"{256/by:.1f}x", f"{by*N/2**20:.0f}"]
    for ds, _ in DS:
        v = acc(ds, *a[1:]) if a[0] == "matrix" else float(sp[(ds, a[1])]["precision_3px"])
        r.append(f"{v:.2f}")
    if fs:
        t = float(sp[("HPatches", fs)]["total_ms_median"])
        base = float(sp[("HPatches", "float_blas_single_gemm")]["total_ms_median"])
        off = float(sp[("HPatches", "official_torch_cosine")]["total_ms_median"])
        r += [f"{t:.2f}", f"{base/t:.2f}x", f"{off/t:.2f}x"]
    else:
        r += ["not measured"] * 3
    r.append(f"{min(sc[ss]):.2f}-{max(sc[ss]):.2f}x" if ss != "float32_blas" else "1.00x")
    rows.append(r)
write("table2_memory_speed",
      ["Method", "Bytes/desc", "Compression", "DB MB at 524,288 desc", "HPatches P@3px", "COCO P@3px",
       "Two-image MNN ms (encode+match, HPatches median)", "Speedup vs float BLAS", "Speedup vs official XFeat match",
       "1-NN speedup vs float BLAS, N>=65,536 (range over 3 sizes x 2 runs)"], rows)

# ---------- Figure 2: calibration x bit-width x distance (HPatches, validation) ----------
STY = [("uniform", "uniform (min-max)", "#2a78d6", "o", "-"), ("percentile", "percentile clip", "#eb6834", "s", "-"),
       ("quantile", "quantile", "#1baf7a", "^", "-"), ("lloyd", "Lloyd-Max", "#eda100", "D", "-"),
       ("uniform_shared", "uniform, shared", "#e87ba4", "v", "--"), ("quantile_shared", "quantile, shared", "#008300", "P", "--")]
LV = [4, 8, 16, 32, 256]; BITS = [2, 3, 4, 5, 8]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42})
def figure(ds, dsn, fname, ylo):
    fl = acc(ds, "float", 0, "cosine")
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.45), sharey=True)
    for ax, (dist, title) in zip(axs, [("idx_L1", "Index L1 (thermometer Hamming)"), ("idx_L2", "Index L2"),
                                         ("recon_cos", "Reconstruction cosine")]):
        ax.axhline(fl, color=INK, lw=0.8, ls=":", zorder=1)
        for s, lab, c, mk, ls in STY:
            ax.plot(range(5), [acc(ds, s, l, dist) for l in LV], color=c, marker=mk, ms=4, lw=1.2, ls=ls,
                    mec="white", mew=0.5, label=lab, zorder=3)
        ax.set_title(title, fontsize=8, loc="left", pad=4)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"{b}" for b in BITS]); ax.set_xlabel("Bits per dimension")
        ax.set_ylim(ylo, fl + 1.2); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
    axs[0].set_ylabel(f"{dsn} precision @3px (%)")
    axs[0].annotate(f"float32 cosine {fl:.2f}", (0, fl), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7, handlelength=2.6, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.8)
    for ext in ("pdf", "png"): fig.savefig(os.path.join(HERE, f"{fname}.{ext}"), dpi=300)
    plt.close(fig)
figure("HPatches", "HPatches", "fig2_accuracy_matrix_hpatches", 46)
figure("COCO_synthetic", "COCO synthetic", "fig2_accuracy_matrix_coco", 53)
print("done")
