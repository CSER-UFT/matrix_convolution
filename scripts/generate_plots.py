#!/usr/bin/env python3
"""Gera os gráficos científicos da campanha de convolução e GEMM."""

import argparse
import csv
import math
import re
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None


COLORS = {"double": "#2166ac", "float": "#b2182b"}
TECH_LINE = {"full": "-", "skip_kernel": "--", "skip_k": "--"}
TECH_MARK = {"full": "o", "skip_kernel": "s", "skip_k": "s"}
OP_LABEL = {
    "conv_linear": "Convolução contígua",
    "conv_malloc": "Convolução por linhas",
    "gemm": "Multiplicação de matrizes",
}
DIST_LABEL = {0: "uniforme", 1: "normal", 2: "exponencial"}
PROGRAM_SHORT = {
    "conv_linear_double_full": "CL D completa",
    "conv_linear_float_full": "CL F completa",
    "conv_linear_double_skip_kernel": "CL D aproximada",
    "conv_linear_float_skip_kernel": "CL F aproximada",
    "conv_malloc_double_full": "CM D completa",
    "conv_malloc_float_full": "CM F completa",
    "conv_malloc_double_skip_kernel": "CM D aproximada",
    "conv_malloc_float_skip_kernel": "CM F aproximada",
    "gemm_double_full": "GEMM D completa",
    "gemm_float_full": "GEMM F completa",
    "gemm_double_skip_k": "GEMM D aproximada",
    "gemm_float_skip_k": "GEMM F aproximada",
}
METRIC_LABEL = {
    "cycles": "Ciclos",
    "instructions": "Instruções",
    "ipc": "Instruções por ciclo (IPC)",
    "time_sec": "Tempo observado (s)",
}


def boxplot_label_argument(labels):
    """Compatibilidade entre Matplotlib 3.7 (labels) e 3.9+ (tick_labels)."""
    version = tuple(int(part) for part in re.findall(r"\d+", matplotlib.__version__)[:2])
    return {"tick_labels" if version >= (3, 9) else "labels": labels}


class Plotter:
    def __init__(self, data_dir: Path, output: Path, formats, dpi: int):
        self.data_dir = data_dir
        self.output = output
        self.formats = formats
        self.dpi = dpi
        self.index = []
        self.raw = pd.read_csv(data_dir / "results_raw.csv", keep_default_na=False)
        self.summary = pd.read_csv(data_dir / "results_summary.csv", keep_default_na=False)
        self.comp_raw = pd.read_csv(data_dir / "comparisons_raw.csv", keep_default_na=False)
        self.comp = pd.read_csv(data_dir / "comparisons_summary.csv", keep_default_na=False)
        self.accuracy = pd.read_csv(data_dir / "accuracy.csv", keep_default_na=False)
        for frame in [self.raw, self.summary, self.comp_raw, self.comp, self.accuracy]:
            for col in ["N", "dist", "parameter", "repetition"]:
                if col in frame:
                    frame[col] = pd.to_numeric(frame[col], errors="raise").astype(int)

    def save(self, fig, category, name, title, question, source, filters="Todos os casos aplicáveis"):
        directory = self.output / category
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for extension in self.formats:
            path = directory / f"{name}.{extension}"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            paths.append(path.relative_to(self.output).as_posix())
        plt.close(fig)
        self.index.append({
            "categoria": category,
            "figura": name,
            "titulo": title,
            "pergunta": question,
            "filtros": filters,
            "fonte": source,
            "arquivos": "; ".join(paths),
        })

    @staticmethod
    def finish(ax, xlabel=None, ylabel=None, title=None, legend=True, logy=False):
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, loc="left", fontweight="bold", pad=10)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.28, linewidth=0.7)
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False, fontsize=8)

    @staticmethod
    def mean_ci(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return np.nan, np.nan
        mean = values.mean()
        if len(values) < 2:
            return mean, 0.0
        sem = values.std(ddof=1) / math.sqrt(len(values))
        critical = stats.t.ppf(0.975, len(values) - 1) if stats else 1.96
        return mean, critical * sem

    def performance(self):
        for operation in ["conv_linear", "conv_malloc", "gemm"]:
            params = sorted(self.raw.loc[self.raw.operation == operation, "parameter"].unique())
            for metric in ["cycles", "instructions", "ipc", "time_sec"]:
                fig, axes = plt.subplots(1, len(params), figsize=(4.3 * len(params), 4.3), sharey=True)
                axes = np.atleast_1d(axes)
                selected_dist = -1 if operation == "gemm" else 0
                subset = self.raw[(self.raw.operation == operation) & (self.raw.dist == selected_dist)]
                for ax, parameter in zip(axes, params):
                    panel = subset[subset.parameter == parameter]
                    for precision in ["double", "float"]:
                        for technique in sorted(panel.technique.unique(), key=lambda x: x != "full"):
                            points = []
                            for n, group in panel[(panel.precision == precision) & (panel.technique == technique)].groupby("N"):
                                mean, ci = self.mean_ci(group[metric])
                                points.append((n, mean, ci))
                            if not points:
                                continue
                            points.sort()
                            x, y, err = map(np.asarray, zip(*points))
                            tech_label = "completa" if technique == "full" else "aproximada"
                            ax.errorbar(x, y, yerr=err, color=COLORS[precision],
                                        linestyle=TECH_LINE[technique], marker=TECH_MARK[technique],
                                        capsize=2.5, linewidth=1.7, markersize=5,
                                        label=f"{precision} · {tech_label}")
                    symbol = "K" if operation != "gemm" else "B"
                    self.finish(ax, "Ordem da matriz (N)", METRIC_LABEL[metric], f"{symbol} = {parameter}")
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
                title = f"{OP_LABEL[operation]}: {METRIC_LABEL[metric].lower()} por tamanho"
                fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")
                note = "Média e IC 95% de 50 repetições; entrada uniforme."
                if metric == "time_sec":
                    note += " Tempo afetado pelo governador powersave."
                fig.text(0.02, 0.01, note, fontsize=8, color="#555555")
                fig.tight_layout(rect=(0, 0.04, 1, 0.94))
                self.save(fig, "01_desempenho", f"{operation}_{metric}", title,
                          f"Como {METRIC_LABEL[metric].lower()} escala com N, precisão e técnica?",
                          "results_raw.csv", f"dist={selected_dist}; 50 repetições; média e IC 95%")

    def precision_speedup(self):
        for operation in ["conv_linear", "conv_malloc", "gemm"]:
            selected_dist = -1 if operation == "gemm" else 0
            panel = self.comp_raw[(self.comp_raw.operation == operation) &
                                  (self.comp_raw.comparison == "precision") &
                                  (self.comp_raw.dist == selected_dist)]
            params = sorted(panel.parameter.unique())
            fig, axes = plt.subplots(1, len(params), figsize=(4.2 * len(params), 4), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, parameter in zip(axes, params):
                points = []
                for n, group in panel[panel.parameter == parameter].groupby("N"):
                    mean, ci = self.mean_ci(group.cycles_speedup)
                    points.append((n, mean, ci))
                points.sort()
                if points:
                    x, y, err = map(np.asarray, zip(*points))
                    ax.errorbar(x, y, yerr=err, color="#4d4d4d", marker="o", capsize=3)
                ax.axhline(1, color="#777777", linestyle=":")
                symbol = "K" if operation != "gemm" else "B"
                self.finish(ax, "Ordem da matriz (N)", "Speedup em ciclos (double/float)", f"{symbol} = {parameter}", False)
            title = f"Efeito da precisão em {OP_LABEL[operation].lower()}"
            fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")
            fig.text(0.02, 0.01, "Acima de 1: float consumiu menos ciclos. Média e IC 95% pareados.", fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 0.94))
            self.save(fig, "02_comparacoes", f"precisao_{operation}", title,
                      "A precisão float reduz os ciclos da versão completa?",
                      "comparisons_raw.csv", f"comparison=precision; dist={selected_dist}")

    def approximation_speedup(self):
        for operation in ["conv_linear", "conv_malloc", "gemm"]:
            selected_dist = -1 if operation == "gemm" else 0
            panel = self.comp_raw[(self.comp_raw.operation == operation) &
                                  (self.comp_raw.comparison.str.startswith("approximation")) &
                                  (self.comp_raw.dist == selected_dist)]
            fig, ax = plt.subplots(figsize=(8, 5))
            for precision in ["double", "float"]:
                comp_name = f"approximation_{precision}"
                for n, group_n in panel[panel.comparison == comp_name].groupby("N"):
                    points = []
                    for parameter, group in group_n.groupby("parameter"):
                        mean, ci = self.mean_ci(group.cycles_speedup)
                        points.append((parameter, mean, ci))
                    points.sort()
                    x, y, err = map(np.asarray, zip(*points))
                    ax.errorbar(x, y, yerr=err, color=COLORS[precision],
                                marker={512: "o", 1024: "s", 2048: "^"}[n],
                                linestyle="-", capsize=2.5, label=f"{precision} · N={n}")
            ax.axhline(1, color="#777777", linestyle=":")
            xlabel = "Tamanho do kernel (K)" if operation != "gemm" else "Tamanho do bloco (B)"
            title = f"Ganho da aproximação em {OP_LABEL[operation].lower()}"
            self.finish(ax, xlabel, "Speedup em ciclos (completa/aproximada)", title)
            fig.text(0.12, 0.01, "Acima de 1: a versão aproximada consumiu menos ciclos. Entrada uniforme.", fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            self.save(fig, "02_comparacoes", f"aproximacao_{operation}", title,
                      "Quanto a técnica aproximada reduz os ciclos em cada configuração?",
                      "comparisons_raw.csv", f"approximations; dist={selected_dist}; média e IC 95%")

    def layout_comparison(self):
        conv = self.raw[self.raw.operation.isin(["conv_linear", "conv_malloc"])].copy()
        keys = ["precision", "technique", "N", "dist", "parameter", "repetition"]
        wide = conv.pivot_table(index=keys, columns="operation", values=["cycles", "instructions"], aggfunc="first")
        wide = wide.dropna().reset_index()
        wide.columns = ["_".join(str(part) for part in col if part).rstrip("_")
                        if isinstance(col, tuple) else col for col in wide.columns]
        for metric in ["cycles", "instructions"]:
            speedup_col = f"{metric}_speedup"
            wide[speedup_col] = wide[f"{metric}_conv_linear"] / wide[f"{metric}_conv_malloc"]
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
            for ax, technique in zip(axes, ["full", "skip_kernel"]):
                panel = wide[(wide.technique == technique) & (wide.dist == 0)]
                for precision in ["double", "float"]:
                    for k, group_k in panel[panel.precision == precision].groupby("parameter"):
                        points = []
                        for n, group in group_k.groupby("N"):
                            mean, ci = self.mean_ci(group[speedup_col])
                            points.append((n, mean, ci))
                        points.sort()
                        x, y, err = map(np.asarray, zip(*points))
                        ax.errorbar(x, y, yerr=err, color=COLORS[precision], marker={3: "o", 5: "s", 7: "^"}[k],
                                    capsize=2.5, label=f"{precision} · K={k}")
                ax.axhline(1, color="#777777", linestyle=":")
                self.finish(ax, "Ordem da matriz (N)", f"Razão contígua/linhas em {METRIC_LABEL[metric].lower()}",
                            "Completa" if technique == "full" else "Aproximada")
            title = f"Efeito da organização da memória sobre {METRIC_LABEL[metric].lower()}"
            fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")
            fig.text(0.02, 0.01, "Acima de 1: a implementação por linhas usou menos recursos.", fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 0.94))
            self.save(fig, "02_comparacoes", f"layout_{metric}", title,
                      "Qual implementação de convolução usa menos recursos?", "results_raw.csv",
                      "conv_linear pareada com conv_malloc; dist=0")

    def gemm_blocks(self):
        panel = self.raw[(self.raw.operation == "gemm") & (self.raw.dist == -1)]
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), sharey=True)
        for ax, n in zip(axes, sorted(panel.N.unique())):
            for precision in ["double", "float"]:
                for technique in ["full", "skip_k"]:
                    points = []
                    for block, group in panel[(panel.N == n) & (panel.precision == precision) &
                                              (panel.technique == technique)].groupby("parameter"):
                        mean, ci = self.mean_ci(group.cycles)
                        points.append((block, mean, ci))
                    points.sort()
                    x, y, err = map(np.asarray, zip(*points))
                    ax.errorbar(x, y, yerr=err, color=COLORS[precision], linestyle=TECH_LINE[technique],
                                marker=TECH_MARK[technique], capsize=2.5,
                                label=f"{precision} · {'completa' if technique == 'full' else 'aproximada'}")
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32, 64, 128], labels=[8, 16, 32, 64, 128])
            self.finish(ax, "Tamanho do bloco (B)", "Ciclos", f"N = {n}")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        title = "Sensibilidade da GEMM ao tamanho do bloco"
        fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        self.save(fig, "03_gemm_blocos", "gemm_blocos_ciclos", title,
                  "Qual tamanho de bloco minimiza os ciclos da GEMM?", "results_raw.csv", "dist=-1 (GEMM não usa distribuição)")

    def accuracy_plots(self):
        precision = self.accuracy[self.accuracy.comparison == "precision"]
        for operation in ["conv_linear", "conv_malloc", "gemm"]:
            panel = precision[precision.operation == operation]
            params = sorted(panel.parameter.unique())
            fig, axes = plt.subplots(1, len(params), figsize=(4.1 * len(params), 4), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, parameter in zip(axes, params):
                for dist, group in panel[panel.parameter == parameter].groupby("dist"):
                    group = group.sort_values("N")
                    ax.plot(group.N, group.error_rel_mean, marker={-1: "o", 0: "o", 1: "s", 2: "^"}[dist],
                            label=DIST_LABEL.get(dist, "sem distribuição"))
                symbol = "K" if operation != "gemm" else "B"
                self.finish(ax, "Ordem da matriz (N)", "Erro relativo médio", f"{symbol} = {parameter}", logy=True)
            title = f"Erro de float em relação a double: {OP_LABEL[operation].lower()}"
            fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            self.save(fig, "04_qualidade_numerica", f"precisao_erro_{operation}", title,
                      "Qual erro é introduzido ao trocar double por float?", "accuracy.csv", "comparison=precision")

        conv = self.accuracy[(self.accuracy.operation == "conv_linear") &
                             (self.accuracy.comparison.str.startswith("approximation"))]
        for metric, label, logy in [("error_rel_mean", "Erro relativo médio", True), ("rmse", "RMSE", True)]:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
            for ax, dist in zip(axes, [0, 1, 2]):
                for comparison, group in conv[conv.dist == dist].groupby("comparison"):
                    precision_name = comparison.replace("approximation_", "")
                    for n, group_n in group.groupby("N"):
                        group_n = group_n.sort_values("parameter")
                        ax.plot(group_n.parameter, group_n[metric], color=COLORS[precision_name],
                                marker={512: "o", 1024: "s", 2048: "^"}[n],
                                label=f"{precision_name} · N={n}")
                self.finish(ax, "Tamanho do kernel (K)", label, DIST_LABEL[dist].capitalize(), logy=logy)
            title = f"Erro numérico do skip_kernel: {label}"
            fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")
            fig.text(0.02, 0.01, "As curvas de float e double coincidem; alguns símbolos ficam sobrepostos.", fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 0.94))
            self.save(fig, "04_qualidade_numerica", f"skip_kernel_{metric}", title,
                      "Como o erro do skip_kernel varia com K, N, precisão e distribuição?",
                      "accuracy.csv", "conv_linear; aproximações")

        gemm = self.accuracy[(self.accuracy.operation == "gemm") &
                             (self.accuracy.comparison.str.startswith("approximation"))]
        fig, ax = plt.subplots(figsize=(8, 5))
        for comparison, group in gemm.groupby("comparison"):
                precision_name = comparison.replace("approximation_", "")
                for block, group_b in group.groupby("parameter"):
                    group_b = group_b.sort_values("N")
                    ax.plot(group_b.N, group_b.error_rel_mean, color=COLORS[precision_name], alpha=0.75,
                            marker="o", label=f"{precision_name} · B={block}")
        self.finish(ax, "Ordem da matriz (N)", "Erro relativo médio", None, logy=True)
        title = "Erro numérico do skip_k na GEMM"
        fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")
        fig.text(0.12, 0.01, "As curvas de float e double coincidem; alguns símbolos ficam sobrepostos.", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        self.save(fig, "04_qualidade_numerica", "skip_k_error_rel", title,
                  "Como o erro do skip_k varia com N, bloco, precisão e distribuição?", "accuracy.csv")

    def tradeoff(self):
        merged = self.comp.merge(
            self.accuracy,
            on=["operation", "N", "dist", "parameter", "comparison", "reference_program", "candidate_program"],
            how="inner",
        )
        for operation in ["conv_linear", "conv_malloc", "gemm"]:
            panel = merged[(merged.operation == operation) &
                           (merged.comparison.str.startswith("approximation"))]
            distributions = [-1] if operation == "gemm" else [0, 1, 2]
            width = 7.2 if len(distributions) == 1 else 12.5
            fig, axes = plt.subplots(1, len(distributions), figsize=(width, 4), sharex=False, sharey=True)
            axes = np.atleast_1d(axes)
            for ax, dist in zip(axes, distributions):
                for comparison, group in panel[panel.dist == dist].groupby("comparison"):
                    precision_name = comparison.replace("approximation_", "")
                    for n, group_n in group.groupby("N"):
                        ax.scatter(group_n.error_rel_mean, group_n.cycles_speedup_mean,
                                   color=COLORS[precision_name], marker={512: "o", 1024: "s", 2048: "^"}[n],
                                   s=38, alpha=0.8, label=f"{precision_name} · N={n}")
                panel_title = "GEMM" if dist == -1 else DIST_LABEL[dist].capitalize()
                self.finish(ax, "Erro relativo médio", "Speedup em ciclos", panel_title)
                ax.set_xscale("log")
                ax.axhline(1, color="#777777", linestyle=":")
            title = f"Desempenho e erro: {OP_LABEL[operation].lower()}"
            fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            self.save(fig, "05_compromisso", f"tradeoff_{operation}", title,
                      "Quanto de erro acompanha o ganho de ciclos da aproximação?",
                      "comparisons_summary.csv + accuracy.csv")

    def microarchitecture(self):
        valid_dist = ((self.summary.operation == "gemm") & (self.summary.dist == -1)) | \
                     ((self.summary.operation != "gemm") & (self.summary.dist == 0))
        panel = self.summary[(self.summary.N == self.summary.N.max()) & valid_dist].copy()
        preferred = ((panel.operation.str.startswith("conv") & (panel.parameter == 5)) |
                     ((panel.operation == "gemm") & (panel.parameter == 16)))
        panel = panel[preferred]
        fig, ax = plt.subplots(figsize=(9, 6))
        for _, row in panel.iterrows():
            color = COLORS[row.precision]
            marker = "o" if row.technique == "full" else "s"
            ax.scatter(row.cache_miss_rate_mean, row.ipc_mean, color=color, marker=marker, s=85,
                       edgecolor="black", linewidth=0.4)
            ax.annotate(PROGRAM_SHORT[row.program], (row.cache_miss_rate_mean, row.ipc_mean),
                        xytext=(5, 4), textcoords="offset points", fontsize=7)
        title = "Assinatura microarquitetural das implementações"
        self.finish(ax, "Taxa de faltas de cache", "IPC", title, False)
        fig.text(0.12, 0.01, "N=2048, entrada uniforme, K=5 para convolução e B=16 para GEMM.", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        self.save(fig, "06_microarquitetura", "ipc_cache", title,
                  "Como IPC e faltas de cache distinguem as implementações?", "results_summary.csv")

        metrics = ["ipc_mean", "cache_miss_rate_mean", "branch_miss_rate_mean",
                   "l1_dcache_load_misses_mean", "llc_load_misses_mean"]
        matrix = panel.set_index("program")[metrics].astype(float)
        normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min()).replace(0, np.nan)
        normalized = normalized.fillna(0)
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        image = ax.imshow(normalized.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(normalized)), [PROGRAM_SHORT.get(x, x) for x in normalized.index], fontsize=8)
        ax.set_xticks(range(len(metrics)), ["IPC", "Taxa cache", "Taxa desvios", "Faltas L1D", "Faltas LLC"],
                      rotation=25, ha="right")
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                value = normalized.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.55 else "black")
        fig.colorbar(image, ax=ax, label="Normalização min–max por coluna")
        title = "Perfil normalizado dos contadores"
        ax.set_title(title, loc="left", fontweight="bold")
        fig.tight_layout()
        self.save(fig, "06_microarquitetura", "perfil_contadores", title,
                  "Quais programas têm os maiores valores relativos em cada contador?", "results_summary.csv",
                  "N=2048; dist=0; K=5/B=16")

    def reproducibility(self):
        valid_dist = ((self.summary.operation == "gemm") & (self.summary.dist == -1)) | \
                     ((self.summary.operation != "gemm") & (self.summary.dist == 0))
        panel = self.summary[(self.summary.N == self.summary.N.max()) & valid_dist].copy()
        preferred = ((panel.operation.str.startswith("conv") & (panel.parameter == 5)) |
                     ((panel.operation == "gemm") & (panel.parameter == 16)))
        panel = panel[preferred].sort_values("program")
        metrics = ["time_sec_cv", "cycles_cv", "instructions_cv"]
        matrix = panel.set_index("program")[metrics].astype(float) * 100
        fig, ax = plt.subplots(figsize=(7.8, 6.2))
        positive = matrix.values[matrix.values > 0]
        floor = positive.min() / 10 if len(positive) else 1e-10
        log_values = np.log10(np.maximum(matrix.values, floor))
        image = ax.imshow(log_values, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(matrix)), [PROGRAM_SHORT.get(x, x) for x in matrix.index], fontsize=8)
        ax.set_xticks(range(3), ["Tempo", "Ciclos", "Instruções"])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.3g}%", ha="center", va="center", fontsize=7,
                        color="white" if log_values[i, j] > np.nanmedian(log_values) else "black")
        fig.colorbar(image, ax=ax, label="log10 do CV (%)")
        title = "Reprodutibilidade das métricas"
        ax.set_title(title, loc="left", fontweight="bold")
        fig.tight_layout()
        self.save(fig, "07_reprodutibilidade", "cv_metricas", title,
                  "Quais métricas variaram mais entre as 50 repetições?", "results_summary.csv",
                  "N=2048; dist=0; K=5/B=16")

        valid_dist_raw = ((self.raw.operation == "gemm") & (self.raw.dist == -1)) | \
                         ((self.raw.operation != "gemm") & (self.raw.dist == 0))
        selected = self.raw[(self.raw.N == self.raw.N.max()) & valid_dist_raw].copy()
        selected = selected[((selected.operation.str.startswith("conv")) & (selected.parameter == 5)) |
                            ((selected.operation == "gemm") & (selected.parameter == 16))]
        for metric in ["cycles", "time_sec"]:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=False)
            for ax, operation in zip(axes, ["conv_linear", "conv_malloc", "gemm"]):
                operation_data = selected[selected.operation == operation]
                programs = sorted(operation_data.program.unique())
                values = [operation_data.loc[operation_data.program == p, metric].to_numpy() for p in programs]
                labels = []
                for program in programs:
                    precision = "float" if "_float_" in program else "double"
                    technique = "aprox." if ("skip_kernel" in program or "skip_k" in program) else "completa"
                    labels.append(f"{precision}\n{technique}")
                boxes = ax.boxplot(values, patch_artist=True, showfliers=True,
                                   medianprops={"color": "black"}, **boxplot_label_argument(labels))
                for patch, program in zip(boxes["boxes"], programs):
                    patch.set_facecolor(COLORS["float" if "_float_" in program else "double"])
                    patch.set_alpha(0.65)
                self.finish(ax, None, METRIC_LABEL[metric], OP_LABEL[operation], False)
                ax.tick_params(axis="x", labelsize=7)
                if metric == "cycles":
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            title = f"Distribuição das repetições: {METRIC_LABEL[metric].lower()}"
            fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")
            note = "Cada painel usa sua própria escala para preservar a distribuição das 50 repetições."
            if metric == "time_sec":
                note += " Tempo afetado pelo governador powersave."
            fig.text(0.02, 0.01, note, fontsize=8)
            fig.tight_layout(rect=(0, 0.05, 1, 0.93))
            self.save(fig, "07_reprodutibilidade", f"boxplot_{metric}", title,
                      f"Como as 50 medições de {METRIC_LABEL[metric].lower()} se distribuem?",
                      "results_raw.csv", "N=2048; dist=0; K=5/B=16")

    def distribution_effect(self):
        conv = self.summary[self.summary.operation.str.startswith("conv")].copy()
        keys = ["operation", "program", "precision", "technique", "N", "parameter"]
        rows = []
        for key, group in conv.groupby(keys):
            low, high = group.cycles_mean.min(), group.cycles_mean.max()
            rows.append((*key, 100 * (high - low) / group.cycles_mean.mean()))
        effect = pd.DataFrame(rows, columns=keys + ["amplitude_percentual"])
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        groups = []
        labels = []
        for operation in ["conv_linear", "conv_malloc"]:
            for precision in ["double", "float"]:
                groups.append(effect[(effect.operation == operation) & (effect.precision == precision)].amplitude_percentual)
                labels.append(("Contígua" if operation == "conv_linear" else "Por linhas") + f"\n{precision}")
        ax.boxplot(groups, patch_artist=True, showfliers=True, **boxplot_label_argument(labels))
        title = "Efeito da distribuição das entradas sobre os ciclos"
        self.finish(ax, None, "Amplitude entre distribuições (%)", title, False)
        fig.tight_layout()
        self.save(fig, "08_diagnosticos", "efeito_distribuicao", title,
                  "A distribuição dos valores altera o custo das convoluções?", "results_summary.csv")

    def correlation(self):
        columns = ["time_sec", "cycles", "instructions", "ipc", "cache_miss_rate",
                   "branch_miss_rate", "l1_dcache_load_misses", "llc_load_misses"]
        corr = self.raw[columns].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        labels = ["Tempo", "Ciclos", "Instruções", "IPC", "Taxa cache", "Taxa desvios", "Faltas L1D", "Faltas LLC"]
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(corr.iloc[i, j]) > 0.55 else "black")
        fig.colorbar(image, ax=ax, label="Correlação de Spearman")
        title = "Correlação entre métricas da campanha"
        ax.set_title(title, loc="left", fontweight="bold")
        fig.tight_layout()
        self.save(fig, "08_diagnosticos", "correlacao_metricas", title,
                  "Quais métricas variam juntas no conjunto completo?", "results_raw.csv")

    def write_index(self):
        self.output.mkdir(parents=True, exist_ok=True)
        csv_path = self.output / "INDICE_GRAFICOS.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.index[0].keys())
            writer.writeheader()
            writer.writerows(self.index)
        lines = [
            "# Índice dos gráficos", "",
            "Cada figura é gerada em PDF vetorial e PNG a 300 DPI. Os gráficos de tempo devem ser interpretados com a limitação do governador `powersave`; para esta campanha, ciclos são a métrica principal de desempenho.", "",
            f"Total: **{len(self.index)} figuras**, em **{len(self.index) * len(self.formats)} arquivos gráficos**.", "",
            "| Categoria | Figura | Pergunta | Filtros | Fonte |", "| --- | --- | --- | --- | --- |",
        ]
        for item in self.index:
            lines.append(f"| `{item['categoria']}` | `{item['figura']}` | {item['pergunta']} | {item['filtros']} | `{item['fonte']}` |")
        (self.output / "INDICE_GRAFICOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self):
        self.performance()
        self.precision_speedup()
        self.approximation_speedup()
        self.layout_comparison()
        self.gemm_blocks()
        self.accuracy_plots()
        self.tradeoff()
        self.microarchitecture()
        self.reproducibility()
        self.distribution_effect()
        self.correlation()
        self.write_index()


def configure_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 1.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def main():
    parser = argparse.ArgumentParser(description="Gera PDF e PNG dos resultados do TCC2.")
    parser.add_argument("--data", default="results/analysis_data", help="diretório preparado dos CSVs")
    parser.add_argument("--output", default="results/figures_full_20260922", help="diretório de saída")
    parser.add_argument("--formats", default="pdf,png", help="formatos separados por vírgula")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--clean", action="store_true", help="remove somente a saída anterior de gráficos")
    args = parser.parse_args()
    data = Path(args.data).resolve()
    output = Path(args.output).resolve()
    required = ["results_raw.csv", "results_summary.csv", "comparisons_raw.csv",
                "comparisons_summary.csv", "accuracy.csv"]
    missing = [name for name in required if not (data / name).is_file()]
    if missing:
        raise SystemExit("[ERRO] Execute prepare_analysis_data.py antes. Ausentes: " + ", ".join(missing))
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    invalid = [item for item in formats if item not in {"pdf", "png", "svg"}]
    if invalid:
        raise SystemExit("[ERRO] Formatos não aceitos: " + ", ".join(invalid))
    if args.clean and output.exists():
        shutil.rmtree(output)
    configure_style()
    plotter = Plotter(data, output, formats, args.dpi)
    plotter.run()
    print(f"Gráficos gerados: {len(plotter.index)} figuras em {output}")
    print(f"Índice: {output / 'INDICE_GRAFICOS.md'}")


if __name__ == "__main__":
    main()
