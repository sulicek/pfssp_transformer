# %% init
# ====================================================================================
# ====================================================================================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tools import get_project_root_dir, load_eval_test_case_results

sns.set_theme()

tc_tl, results_tl, _ = load_eval_test_case_results("evalg64_cbase600_taillard")
tc_vrf, results_vrf, _ = load_eval_test_case_results("evalg64_cbase600_VRF_large")
_, cplex_results_tl, _ = load_eval_test_case_results("cbase600_taillard")
_, cplex_results_vrf, _ = load_eval_test_case_results("cbase600_VRF_large")

data = []
names = {
    "random model": "random",
    # "lowerbound model",
    "better lowerbound model": "lowerbound",
    # pfsnet20
    "transformer linear model": "pfsnet20_bn",
    "transformer linear model no eval": "pfsnet20_at",
    "pfsnet20_ln": "pfsnet20_ln",
    # pfsnetgen
    "transformer embed model": "pfsnetgen_bn",
    "transformer embed model no eval": "pfsnetgen_at",
    "pfsnetgen_ln": "pfsnetgen_ln",
}


def get_desc(i, tc):
    return tc[i].shape[1], tc[i].shape[2]


# add taillard results
for i, result in enumerate(results_tl):
    for name, display_name in names.items():
        if name not in result.__dict__:
            continue
        row = (
            "tl",
            *get_desc(i, tc_tl),
            i,
            display_name,
            float(result.__dict__[name].l_mean),
            float(result.__dict__[name].l_ratio_mean * 100.0),
            float(result.__dict__[name].times),
        )
        data.append(row)

# add vrf results
for i, result in enumerate(results_vrf):
    for name, display_name in names.items():
        if name not in result.__dict__:
            continue
        row = (
            "vrf",
            *get_desc(i, tc_vrf),
            i,
            display_name,
            float(result.__dict__[name].l_mean),
            float(result.__dict__[name].l_ratio_mean * 100.0),
            float(result.__dict__[name].times),
        )
        data.append(row)

# add cplex baseline for taillard and vrf
for i, result in enumerate(results_tl):
    row = (
        "tl",
        *get_desc(i, tc_tl),
        i,
        "cplex baseline",
        float(cplex_results_tl[i].cplex.l_mean),
        float(cplex_results_tl[i].cplex.l_ratio_mean * 100.0),
        float(cplex_results_tl[i].cplex.times),
    )
    data.append(row)

for i, result in enumerate(results_vrf):
    row = (
        "vrf",
        *get_desc(i, tc_vrf),
        i,
        "cplex baseline",
        float(cplex_results_vrf[i].cplex.l_mean),
        float(cplex_results_vrf[i].cplex.l_ratio_mean * 100.0),
        float(cplex_results_vrf[i].cplex.times),
    )
    data.append(row)

# add comparison paper
proposed_model_by_size = {
    20: [6.03, 4.71, 7.49, 9.82, 6.15, 5.02, 7.26, 6.72, 13.09, 11.37],
    50: [0.88, 3.71, 2.1, 4.03, 1.68, 5.55, 3.78, 2.42, 3.76, 1.98],
    100: [4.64, 1.65, 1.6, 1.64, 3.31, 2.16, 0.67, 3.44, 2.68, 1.33],
}

for size, vals in proposed_model_by_size.items():
    for i, val in enumerate(vals):
        row = ("tl", size, 5, i, "Xu_et_al", -1, val, -1)
        data.append(row)

# pl = {"cplex baseline": sns.color_palette()[0]}

# for i, v in enumerate(names.values()):
#     pl[v] = sns.color_palette()[i + 1]
# pl["Xu_et_al"] = sns.color_palette()[1]


# Define base hues (0 to 360)
HUE_20 = 150 / 256  # Seaborn Blue/Indigo
HUE_GEN = 30 / 256  # Seaborn Orange/Amber

# Normalized S and L values (between 0.0 and 1.0)
pl = {
    # pfsnet20 Family (Blue tones)
    "pfsnet20_bn": sns.husl_palette(1, h=HUE_20, s=0.90, l=0.65)[0],  # Vibrant Blue
    "pfsnet20_at": sns.husl_palette(1, h=HUE_20, s=0.85, l=0.45)[0],  # Dark Navy
    "pfsnet20_ln": sns.husl_palette(1, h=HUE_20, s=0.95, l=0.80)[0],  # Sky Blue
    # pfsnetgen Family (Orange tones)
    "pfsnetgen_bn": sns.husl_palette(1, h=HUE_GEN, s=0.95, l=0.65)[0],  # Vibrant Orange
    "pfsnetgen_at": sns.husl_palette(1, h=HUE_GEN, s=0.85, l=0.45)[
        0
    ],  # Burnt Iron/Dark Orange
    "pfsnetgen_ln": sns.husl_palette(1, h=HUE_GEN, s=0.95, l=0.80)[0],  # Light Amber
}

# Rest of your baseline colors
pl["Xu_et_al"] = sns.color_palette("deep")[3]
pl["cplex baseline"] = sns.color_palette("deep")[7]  # Deep Grey
pl["lowerbound"] = "#000000"  # Solid Black


df_eval = pd.DataFrame.from_records(
    data,
    columns=[
        "dataset",
        "n_jobs",
        "n_machines",
        "instance",
        "name",
        "makespan",
        "gap",
        "time",
    ],
)

df_eval["comb"] = (
    # "j" +
    # df_eval["n_jobs"].astype(str) +
    "m" + df_eval["n_machines"].astype(str) + "_" + df_eval["dataset"].astype(str)
)

# warm start
_, results_ws_vrf, _ = load_eval_test_case_results("ws600_evalg64_cbase600_VRF_large")
_, results_ws_tl, i = load_eval_test_case_results("ws600_evalg64_cbase600_taillard")


data = []
names = {"cplex ws": "cplex ws", "cplex": "cplex"}


def get_desc(i, tc):
    return tc[i].shape[1], tc[i].shape[2]


for i, result in enumerate(results_ws_tl):
    for name, display_name in names.items():
        row = (
            "tl",
            *get_desc(i, tc_tl),
            i,
            display_name,
            float(result.__dict__[name].l_mean),
            float(result.__dict__[name].l_ratio_mean * 100.0),
            float(result.__dict__[name].times),
        )
        data.append(row)
        if display_name == "cplex ws":
            model_time = float(results_tl[i].__dict__["transformer embed model"].times)
            row = (
                "tl",
                *get_desc(i, tc_tl),
                i,
                display_name + " +model",
                float(result.__dict__[name].l_mean),
                float(result.__dict__[name].l_ratio_mean * 100.0),
                float(result.__dict__[name].times + model_time),
            )
            data.append(row)
for i, result in enumerate(results_ws_vrf):
    for name, display_name in names.items():
        row = (
            "vrf",
            *get_desc(i, tc_vrf),
            i,
            display_name,
            float(result.__dict__[name].l_mean),
            float(result.__dict__[name].l_ratio_mean * 100.0),
            float(result.__dict__[name].times),
        )
        data.append(row)
        if display_name == "cplex ws":
            model_time = float(results_vrf[i].__dict__["transformer embed model"].times)
            row = (
                "vrf",
                *get_desc(i, tc_vrf),
                i,
                display_name + " +model",
                float(result.__dict__[name].l_mean),
                float(result.__dict__[name].l_ratio_mean * 100.0),
                float(result.__dict__[name].times + model_time),
            )
            data.append(row)

# add cplex baseline
for i, result in enumerate(results_ws_tl):
    row = (
        "tl",
        *get_desc(i, tc_tl),
        i,
        "cplex baseline",
        float(cplex_results_tl[i].cplex.l_mean),
        float(cplex_results_tl[i].cplex.l_ratio_mean * 100.0),
        float(cplex_results_tl[i].cplex.times),
    )
    data.append(row)

for i, result in enumerate(results_ws_vrf):
    row = (
        "vrf",
        *get_desc(i, tc_vrf),
        i,
        "cplex baseline",
        float(cplex_results_vrf[i].cplex.l_mean),
        float(cplex_results_vrf[i].cplex.l_ratio_mean * 100.0),
        float(cplex_results_vrf[i].cplex.times),
    )
    data.append(row)

cl = sns.color_palette()
pl_ws = {
    "cplex baseline": cl[0],
    "cplex ws": cl[1],
    "cplex": cl[2],
    "cplex ws +model": cl[3],
}

df_ws = pd.DataFrame.from_records(
    data,
    columns=[
        "dataset",
        "n_jobs",
        "n_machines",
        "instance",
        "name",
        "makespan",
        "gap",
        "time",
    ],
)
df_ws["comb"] = (
    "j" + df_ws["n_jobs"].astype(str) + "m" + df_ws["n_machines"].astype(str)
    # + "_"
    # + df_ws["dataset"].astype(str)
)
# ====================================================================================
# ====================================================================================

# %% set latex rendering
# ====================================================================================
# ====================================================================================
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],  # Optional: Specify a serif font
        "text.latex.preamble": r"\usepackage{amsmath}",  # Optional: Include amsmath package
    }
)

plt.rcParams.update({"font.size": 50})
# ====================================================================================
# ====================================================================================


# %% graph pfsnet20 taillard + vrf gap with [22] compare
# ====================================================================================
# ====================================================================================
dfp = df_eval


dfp = dfp.query("n_machines<=20")
dfp = dfp.query(
    "name in ['random','pfsnet20_bn', 'pfsnet20_at','pfsnet20_ln','Xu_et_al','cplex baseline','lowerbound']"
)


g = sns.catplot(
    data=dfp,
    x="n_jobs",
    y="gap",
    kind="point",
    hue="name",
    col="comb",
    palette=pl,
    aspect=0.7,
    sharey=False,
    sharex=False,
    hue_order=[
        "pfsnet20_ln",
        "pfsnet20_bn",
        "pfsnet20_at",
        "Xu_et_al",
        "lowerbound",
    ],
)
# g._legend.set_title("Model")
g.fig.subplots_adjust(right=0.81, wspace=0.3)
g.set_axis_labels("Job count", "\\textsc{Gap} (\\%)", size=22)

g._legend.set_title("")
new_labels = [
    "${\\rm PFSNET}_{m=20, \\rm LN}$",
    "${\\rm PFSNET}_{m=20, \\rm BN}$",
    "${\\rm PFSNET}_{m=20, \\rm AT}$",
    "Results from [25]",
    "Lowerbound",
]
for text, label in zip(g._legend.texts, new_labels):
    text.set_text(label)
    text.set_size(22)

new_col_labels = [
    "Machines: 5, Tl",
    "Machines: 10, Tl",
    "Machines: 20, Tl",
    "Machines: 20, VRF",
]
for i, ax in enumerate(g.axes.flat):
    ax.set_title(new_col_labels[i], size=22)


for ax in g.axes.flat:  # loop over all axes in the FacetGrid]
    # Draw the horizontal blue line at y=0
    ax.axhline(0, color=pl["cplex baseline"], linestyle="--", linewidth=1.5, zorder=1)

    # Loop over x-labels to resize AND rotate them
    for label in ax.get_xticklabels():
        label.set_size(14)
        label.set_rotation(30)  # Rotates the text 30 degrees
        # label.set_ha('right')        # Aligns the right edge of text to the tick
    # ax.tick_params(axis='x', pad=20)

    for label in ax.get_xticklabels():
        label.set_size(14)
    for label in ax.get_yticklabels():
        label.set_size(14)


plt.savefig(
    get_project_root_dir() / ("../plots/e2e_pfsnet20_tl_vrf.pdf"),
    format="pdf",
    bbox_inches="tight",
)
# ====================================================================================
# ====================================================================================

# %% graph pfsnetgen vrf + taillard gap
# ====================================================================================
# ====================================================================================
dfp = df_eval

dfp = dfp.query(
    "name in ['random','pfsnetgen_bn', 'pfsnetgen_at','pfsnetgen_ln', 'cplex baseline','lowerbound','Xu_et_al']"
)
g = sns.catplot(
    data=dfp,
    x="n_jobs",
    y="gap",
    kind="point",
    hue="name",
    col="comb",
    palette=pl,
    aspect=0.7,
    col_wrap=3,
    sharey=False,
    sharex=False,
    hue_order=[
        "pfsnetgen_ln",
        "pfsnetgen_bn",
        "pfsnetgen_at",
        "Xu_et_al",
        "lowerbound",
    ],
)

# g._legend.set_title("Model")
g.fig.subplots_adjust(right=0.76, wspace=0.3, hspace=0.3)
g.set_axis_labels("Job count", "\\textsc{Gap} (\\%)", size=22)

g._legend.set_title("")
new_labels = [
    "${\\rm PFSNET}_{\\rm gen, \\rm LN}$",
    "${\\rm PFSNET}_{\\rm gen, \\rm BN}$",
    "${\\rm PFSNET}_{\\rm gen, \\rm AT}$",
    "Results from [25]",
    "Lowerbound",
]
for text, label in zip(g._legend.texts, new_labels):
    text.set_text(label)
    text.set_size(22)

new_col_labels = [
    "Machines: 5, Tl",
    "Machines: 10, Tl",
    "Machines: 20, Tl",
    "Machines: 20, VRF",
    "Machines: 40, VRF",
    "Machines: 60, VRF",
]
for i, ax in enumerate(g.axes.flat):
    ax.set_title(new_col_labels[i], size=22)


for ax in g.axes.flat:  # loop over all axes in the FacetGrid]
    # Draw the horizontal blue line at y=0
    ax.axhline(0, color=pl["cplex baseline"], linestyle="--", linewidth=1.5, zorder=1)

    # Loop over x-labels to resize AND rotate them
    for label in ax.get_xticklabels():
        label.set_size(14)
        label.set_rotation(30)  # Rotates the text 30 degrees
        # label.set_ha('right')        # Aligns the right edge of text to the tick
    # ax.tick_params(axis='x', pad=20)

    for label in ax.get_xticklabels():
        label.set_size(14)
    for label in ax.get_yticklabels():
        label.set_size(14)


plt.savefig(
    get_project_root_dir() / ("../plots/e2e_pfsnetgen_vrf_tl.pdf"),
    format="pdf",
    bbox_inches="tight",
)
# ====================================================================================
# ====================================================================================

# %% graph pfsnetgen vs pfsten20 comparison
# ====================================================================================
# ====================================================================================
dfp = df_eval

dfp = dfp.query("n_machines <= 20")

dfp = dfp.query(
    "name in ['pfsnet20_bn', 'pfsnet20_at','pfsnet20_ln','pfsnetgen_bn', 'pfsnetgen_at','pfsnetgen_ln','Xu_et_al']"
)
g = sns.catplot(
    data=dfp,
    x="n_jobs",
    y="gap",
    kind="point",
    hue="name",
    col="comb",
    palette=pl,
    aspect=1.0,
    col_wrap=2,
    sharey=False,
    sharex=False,
    hue_order=[
        "pfsnet20_ln",
        "pfsnet20_bn",
        "pfsnet20_at",
        "pfsnetgen_ln",
        "pfsnetgen_bn",
        "pfsnetgen_at",
        "Xu_et_al",
    ],
)
# g.set(yscale="symlog")
# g._legend.set_title("Model")
g.fig.subplots_adjust(right=0.76, wspace=0.3, hspace=0.3)
g.set_axis_labels("Job count", "\\textsc{Gap} (\\%)", size=22)

g._legend.set_title("")
new_labels = [
    "${\\rm PFSNET}_{m=20, \\rm LN}$",
    "${\\rm PFSNET}_{m=20, \\rm BN}$",
    "${\\rm PFSNET}_{m=20, \\rm AT}$",
    "${\\rm PFSNET}_{\\rm gen, \\rm LN}$",
    "${\\rm PFSNET}_{\\rm gen, \\rm BN}$",
    "${\\rm PFSNET}_{\\rm gen, \\rm AT}$",
    "Results from [25]",
]
for text, label in zip(g._legend.texts, new_labels):
    text.set_text(label)
    text.set_size(22)

new_col_labels = [
    "Machines: 5, Tl",
    "Machines: 10, Tl",
    "Machines: 20, Tl",
    "Machines: 20, VRF",
]
for i, ax in enumerate(g.axes.flat):
    ax.set_title(new_col_labels[i], size=22)


for ax in g.axes.flat:  # loop over all axes in the FacetGrid]
    # Draw the horizontal blue line at y=0
    ax.axhline(0, color=pl["cplex baseline"], linestyle="--", linewidth=1.5, zorder=1)

    # Loop over x-labels to resize AND rotate them
    for label in ax.get_xticklabels():
        label.set_size(14)
        label.set_rotation(30)  # Rotates the text 30 degrees
        # label.set_ha('right')        # Aligns the right edge of text to the tick
    # ax.tick_params(axis='x', pad=20)

    for label in ax.get_xticklabels():
        label.set_size(14)
    for label in ax.get_yticklabels():
        label.set_size(14)


plt.savefig(
    get_project_root_dir() / ("../plots/e2e_pfsnetgen_pfsnet20.pdf"),
    format="pdf",
    bbox_inches="tight",
)
# ====================================================================================
# ====================================================================================

# %% graph warmstart vrf + taillard
# ====================================================================================
# ====================================================================================

# filter interesting instances
dfp = df_ws.copy()
# dfp = df_ws.query("(dataset=='vrf')")

dfp = dfp.query("name not in ['cplex baseline', 'cplex ws']")

dfp["dataset_machines"] = (
    "m" + dfp["n_machines"].astype(str) + "_" + dfp["dataset"].astype(str)
)

# order = ["cplex ws", "cplex ws +model", "cplex", "cplex baseline"]
order = ["cplex ws +model", "cplex"]
# g = sns.catplot(kind="box", data=dfp, x="n_jobs", y="time", row="dataset", col="n_machines", hue="name", hue_order=order, palette=pl_ws)
g = sns.catplot(
    kind="box",
    data=dfp,
    col="dataset_machines",
    col_wrap=3,
    x="n_jobs",
    y="time",
    sharex=False,
    sharey=False,
    hue="name",
    hue_order=order,
    palette=pl_ws,
    aspect=0.7,
    width=0.5,
)
g._legend.set_title("")

g.fig.subplots_adjust(right=0.78, wspace=0.2)
g.set_axis_labels("", "Time (s)", size=22)

new_col_labels = [
    "Machines: 5, Tl",
    "Machines: 10, Tl",
    "Machines: 20, Tl",
    "Machines: 20, VRF",
    "Machines: 40, VRF",
    "Machines: 60, VRF",
]
for i, ax in enumerate(g.axes.flat):
    ax.set_title(new_col_labels[i], size=22)

new_labels = ["Warm--start", "No Warm--start"]
for text, label in zip(g._legend.texts, new_labels):
    text.set_text(label)
    text.set_size(22)
# rotate x-axis labels
for ax in g.axes.flat:  # loop over all axes in the FacetGrid
    for label in ax.get_xticklabels():
        label.set_rotation(30)  # or any angle you like
        label.set_size(14)
        # label.set_ha('right')   # horizontal alignment
    ax.tick_params(axis="y", labelsize=14, pad=0.0)  # 'pad' controls tick text distance
    for label in ax.get_yticklabels():
        label.set_size(14)
        # label.set_ha('right')   # horizontal alignment

plt.savefig(
    get_project_root_dir() / "../plots/wsplot_all.pdf",
    format="pdf",
    bbox_inches="tight",
)
# ====================================================================================
# ====================================================================================

# %% speedup
# ====================================================================================
# ====================================================================================
dfp = df_ws.query("(dataset=='vrf') or (dataset=='taillard')")
dfp = dfp[(dfp["name"] == "cplex ws +model") | (dfp["name"] == "cplex")].copy()
a = (
    dfp[(dfp["name"] == "cplex")]["time"].to_numpy()
    - dfp[(dfp["name"] == "cplex ws +model")]["time"].to_numpy()
)
b = np.maximum(
    dfp[(dfp["name"] == "cplex")]["time"].to_numpy(),
    dfp[(dfp["name"] == "cplex ws +model")]["time"].to_numpy(),
)
r = (a / b).mean() * 100
print(f"Average Relative speedup is {r:.2f}%")
# ====================================================================================
# ====================================================================================


# %% comparison
# ====================================================================================
# ====================================================================================

dfp = df_eval.copy()
dfp = dfp.query("dataset in ['tl'] and n_machines in [5]")
names = {
    "Xu_et_al": "Results form [22]",
    "pfsnetgen_bn": "PFSNET_gen_bn",
    "pfsnetgen_at": "PFSNET_gen_at",
    "pfsnetgen_ln": "PFSNET_gen_ln",
    "pfsnet20_bn": "PFSNET_20_bn",
    "pfsnet20_at": "PFSNET_20_at",
    "pfsnet20_ln": "PFSNET_20_ln",
}
gaps = dfp.groupby("name")["gap"].mean()
for k, v in names.items():
    print(f"{v}: {gaps[k]:.2f}%")
print(f"PFSNET_20 outperforms [22] by {gaps['Xu_et_al'] - gaps['pfsnet20_bn']:.2f}%")
# ====================================================================================
# ====================================================================================

# %% comparison
# ====================================================================================
# ====================================================================================

dfp = df_eval.copy()
dfp = dfp.query("dataset in ['tl'] and n_machines in [5]")

# Map keys to their exact LaTeX math-mode representation
latex_names = {
    "Xu_et_al": "Results from \\cite{fs_transformer}",
    "pfsnetgen_bn": "${\\rm PFSNET}_{\\rm gen, \\rm BN}$",
    "pfsnetgen_at": "${\\rm PFSNET}_{\\rm gen, \\rm AT}$",
    "pfsnetgen_ln": "${\\rm PFSNET}_{\\rm gen, \\rm LN}$",
    "pfsnet20_bn": "${\\rm PFSNET}_{m=20, \\rm BN}$",
    "pfsnet20_at": "${\\rm PFSNET}_{m=20, \\rm AT}$",
    "pfsnet20_ln": "${\\rm PFSNET}_{m=20, \\rm LN}$",
}

gaps = dfp.groupby("name")["gap"].mean()

# --- Print the LaTeX Table Code Block ---
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Comparison of Model Performance Metrics}")
print("\\label{tab:model_paper_comparison}")
print("\\begin{tabular}{lr}")
print("\\toprule")
print(f"\\textbf{{Model / Source}} & \\textbf{{Mean Gap (\\%)}} \\\\")
print("\\midrule")

# Print baseline row
print(f"{latex_names['Xu_et_al']:<30} & {gaps['Xu_et_al']:>6.2f} \\\\")
print("\\midrule")

# Print gen variants
for k in ["pfsnetgen_bn", "pfsnetgen_at", "pfsnetgen_ln"]:
    print(f"{latex_names[k]:<30} & {gaps[k]:>6.2f} \\\\")
print("\\midrule")

# Print 20 variants (bolding the best result)
for k in ["pfsnet20_bn", "pfsnet20_at", "pfsnet20_ln"]:
    if k == "pfsnet20_bn":
        print(f"{latex_names[k]:<30} & \\textbf{{{gaps[k]:.2f}}} \\\\")
    else:
        print(f"{latex_names[k]:<30} & {gaps[k]:>6.2f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}\n")

# Keep your original text summary check at the very bottom
improvement = gaps["Xu_et_al"] - gaps["pfsnet20_bn"]
print(f"% Summary: PFSNET_20 outperforms [22] by {improvement:.2f}%")
# ====================================================================================
# ====================================================================================

# %% table: gap towards the NEH heuristic, compared to arXiv:2210.17178 (Table 2)
# ====================================================================================
# ====================================================================================
# arXiv:2210.17178 (Li et al.) reports gap-towards-NEH on these Taillard/VRF sizes;

_, neh_results_tl, _ = load_eval_test_case_results("evalneh_taillard")
_, neh_results_vrf, _ = load_eval_test_case_results("evalneh_VRF_large")

names_neh = {
    "transformer embed model": "pfsnetgen_bn",
    "transformer embed model no eval": "pfsnetgen_at",
    "pfsnetgen_ln": "pfsnetgen_ln",
    "transformer linear model": "pfsnet20_bn",
    "transformer linear model no eval": "pfsnet20_at",
    "pfsnet20_ln": "pfsnet20_ln",
}

data = []
for dataset, results, neh_results, tc, cplex_results in [
    ("tl", results_tl, neh_results_tl, tc_tl, cplex_results_tl),
    ("vrf", results_vrf, neh_results_vrf, tc_vrf, cplex_results_vrf),
]:
    for i, result in enumerate(results):
        l_neh = float(neh_results[i].neh.l_mean)
        if l_neh < 0:  # not one of the sizes evalneh.py ran, see note above
            continue
        for name, display_name in names_neh.items():
            if name not in result.__dict__:
                continue
            row = (dataset, *get_desc(i, tc), display_name, float(result.__dict__[name].l_mean), l_neh)
            data.append(row)
        if cplex_results[i].cplex is not None:
            row = (
                dataset,
                *get_desc(i, tc),
                "cplex baseline",
                float(cplex_results[i].cplex.l_mean),
                l_neh,
            )
            data.append(row)

df_neh = pd.DataFrame.from_records(
    data, columns=["dataset", "n_jobs", "n_machines", "name", "makespan", "l_neh"]
)
df_neh["gap"] = (df_neh["makespan"] - df_neh["l_neh"]) / df_neh["l_neh"] * 100.0

# gap towards NEH (%) as reported in Table 2 of arXiv:2210.17178
paper_gap_neh = {
    ("tl", 50, 5): {"RL": 14.6, "IL": 12.8},
    ("tl", 100, 5): {"RL": 13.2, "IL": 12.3},
    ("tl", 100, 20): {"RL": 12.5, "IL": 12.5},
    ("tl", 200, 20): {"RL": 12.6, "IL": 10.7},
    ("tl", 500, 20): {"RL": 9.9, "IL": 9.4},
    ("vrf", 600, 20): {"RL": 9.7, "IL": 9.1},
    ("vrf", 700, 20): {"RL": 10.0, "IL": 9.5},
    ("vrf", 800, 20): {"RL": 9.1, "IL": 8.2},
}
columns = list(paper_gap_neh.keys())
col_labels = {c: f"({c[1]},{c[2]})" for c in columns}

latex_names_neh = {
    "pfsnetgen_bn": "${\\rm PFSNET}_{\\rm gen, \\rm BN}$",
    "pfsnetgen_at": "${\\rm PFSNET}_{\\rm gen, \\rm AT}$",
    "pfsnetgen_ln": "${\\rm PFSNET}_{\\rm gen, \\rm LN}$",
    "pfsnet20_bn": "${\\rm PFSNET}_{m=20, \\rm BN}$",
    "pfsnet20_at": "${\\rm PFSNET}_{m=20, \\rm AT}$",
    "pfsnet20_ln": "${\\rm PFSNET}_{m=20, \\rm LN}$",
    "cplex baseline": "CPOptimizer baseline",
}

# wide table: rows = model/source, columns = (dataset, n_jobs, n_machines)
piv = df_neh.groupby(["dataset", "n_jobs", "n_machines", "name"])["gap"].mean().unstack(["dataset", "n_jobs", "n_machines"])
piv.columns = list(piv.columns)
piv = piv.reindex(index=list(latex_names_neh.keys()), columns=columns)
piv = piv.rename(index=latex_names_neh)

paper_df = pd.DataFrame(
    {src: {c: paper_gap_neh[c][src] for c in columns} for src in ["IL"]}
).T
paper_df = paper_df.rename(
    index={"RL": "Results from \\cite{TODORL}", "IL": "Results from \\cite{TODOIL}"}
)

full = pd.concat([paper_df, piv])

cplex_label = latex_names_neh["cplex baseline"]

col_min = full.drop(index=cplex_label).min()


def fmt(val, col):
    if pd.isna(val):
        return "--"
    text = f"{val:.2f}"
    return f"\\textbf{{{text}}}" if val == col_min[col] else text


def fmt_italic(val):
    if pd.isna(val):
        return "--"
    return f"\\textit{{{val:.2f}}}"


def print_gap_subtable(dataset_label, dataset):
    subcols = [c for c in columns if c[0] == dataset]
    # tabular* + \extracolsep{\fill} stretches every block to the same \textwidth,
    # regardless of how many data columns it has, so the two blocks line up.
    print("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "r" * len(subcols) + "}")
    print("\\toprule")
    header = (
        f"\\textbf{{{dataset_label}}} & "
        + " & ".join(f"\\textbf{{{col_labels[c]}}}" for c in subcols)
        + " \\\\"
    )
    print(header)
    print("\\midrule")

    for idx, (label, row) in enumerate(full.iterrows()):
        if idx == len(paper_df) or label == cplex_label:
            print("\\midrule")
        if label == cplex_label:
            cells = [fmt_italic(row[c]) for c in subcols]
        else:
            cells = [fmt(row[c], c) for c in subcols]
        print(f"{label:<30} & " + " & ".join(cells) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular*}")


print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Mean gap (\\%) towards the NEH heuristic, compared to \\cite{TODO}}")
print("\\label{tab:neh_gap}")
print_gap_subtable("Taillard", "tl")
print("\\vspace{1em}")
print_gap_subtable("VRF", "vrf")
print("\\end{table}")

# summary: mean gap across all reported sizes, and the best (non-CPOptimizer) model per size
print()
print("% Mean gap towards NEH, averaged across all reported sizes:")
for label, val in full.mean(axis=1).sort_values().items():
    print(f"%   {label}: {val:.2f}%")

print()
print("% Best (non-CPOptimizer) model per column:")
for col, label in full.drop(index=cplex_label).idxmin().items():
    print(f"%   {col_labels[col]}: {label}")
# ====================================================================================
# ====================================================================================

# %% table: mean time (s) per model per instance size, split into <=8-column tables
# ====================================================================================
# ====================================================================================

dfp = df_eval.copy()
dfp = dfp.query("name != 'Xu_et_al' and name != 'lowerbound' and name != 'random'")

times = dfp.groupby(["dataset", "n_machines", "n_jobs", "name"])["time"].mean()

pd.set_option("display.width", 200)

for (dataset, n_machines), group in times.groupby(level=["dataset", "n_machines"]):
    table = group.unstack("n_jobs").droplevel(["dataset", "n_machines"])
    table = table[sorted(table.columns)]
    table = table.dropna(how="all")

    print(f"--- mean time (s), dataset={dataset}, machines={n_machines} ---")
    print(table.round(3).to_string())
    print()
# ====================================================================================
# ====================================================================================
