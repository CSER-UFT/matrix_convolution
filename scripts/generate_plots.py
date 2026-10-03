#!/usr/bin/env python3
"""Gera os gráficos científicos da campanha de convolução e GEMM (TCC2).

Adaptado ao esquema de dados revisado (eixo step, tipos de kernel, comparação
combined, erro pela norma e energia RAPL).

Convenções visuais:
  * Legendas sempre FORA da área de desenho, abaixo da figura, compartilhadas
    entre os painéis. Nenhuma legenda dentro dos eixos.
  * Cor identifica a precisão (double azul, float laranja) ou, quando a
    precisão já está em outro canal, o tamanho N ou o passo.
  * Estilo de linha e marcador identificam o passo (1 = versão completa), de
    modo que a identidade nunca depende só da cor.
  * Speedups e economias: média geométrica com IC 95% (t de Student no log).
  * Tempos e energias: média com IC 95% das repetições independentes.

Cada figura sai em PDF vetorial e PNG, com índice em INDICE_GRAFICOS.md/.csv.
"""

import argparse
import csv
import math
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter
from matplotlib.transforms import offset_copy
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None

# Paletas categóricas validadas para daltonismo (todas as combinações de pares):
#   PREC_COLOR: reservada para a precisão (double azul, float laranja), sempre.
#   DIM: outra dimensão categórica (N, distribuição, passo, parâmetro), no máximo 3.
# O magenta tem contraste baixo com o fundo; por isso toda série também tem
# marcador próprio e entrada na legenda.
PREC_COLOR = {"double": "#2a78d6", "float": "#eb6834"}
DIM = ["#008300", "#4a3aa7", "#e87ba4"]
STEP_STYLE = {1: ("-", "o"), 2: ("--", "s"), 3: (":", "^"), 4: ("-.", "D")}
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#d9d9d6"

OP_LABEL = {
    "conv_linear": "Convolução (alocação contígua)",
    "conv_malloc": "Convolução (alocação por linhas)",
    "gemm": "Multiplicação de matrizes (GEMM)",
}
OPERATIONS = ["conv_linear", "conv_malloc", "gemm"]
DIST_LABEL = {0: "uniforme", 1: "normal", 2: "exponencial"}
KERNEL_LABEL = {"rand": "aleatório", "gauss": "gaussiano", "sobel": "Sobel"}
COMPARISON_LABEL = {
    "approximation_double": "aproximada double / completa double",
    "approximation_float": "aproximada float / completa float",
    "combined": "aproximada float / completa double",
}
COMPARISON_MARK = {"approximation_double": "o", "approximation_float": "s", "combined": "^"}

# Recortes usados nas figuras que não mostram todas as combinações.
CONV_DIST = 0
CONV_KERNEL = "gauss"
GEMM_DIST = 0


def step_label(step):
    return "passo 1 (completa)" if step == 1 else f"passo {step}"


def param_symbol(operation):
    return "B" if operation == "gemm" else "K"


def boxplot_label_argument(labels):
    """Compatibilidade entre Matplotlib 3.7 (labels) e 3.9+ (tick_labels)."""
    version = tuple(int(part) for part in re.findall(r"\d+", matplotlib.__version__)[:2])
    return {"tick_labels" if version >= (3, 9) else "labels": labels}


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    mean = values.mean()
    if len(values) < 2:
        return mean, 0.0
    critical = stats.t.ppf(0.975, len(values) - 1) if stats else 1.96
    return mean, critical * values.std(ddof=1) / math.sqrt(len(values))


def gmean_ci(values):
    """Média geométrica e intervalo assimétrico (inferior, superior) em escala original."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return np.nan, (np.nan, np.nan)
    logs = np.log(values)
    center = logs.mean()
    half = 0.0
    if len(values) >= 2:
        critical = stats.t.ppf(0.975, len(values) - 1) if stats else 1.96
        half = critical * logs.std(ddof=1) / math.sqrt(len(values))
    g = math.exp(center)
    return g, (g - math.exp(center - half), math.exp(center + half) - g)


class Plotter:
    def __init__(self, data_dir: Path, output: Path, formats, dpi: int):
        self.output = output
        self.formats = formats
        self.dpi = dpi
        self.index = []
        self.raw = self.load(data_dir / "results_raw.csv")
        self.comp = self.load(data_dir / "comparisons_raw.csv")
        self.accuracy = self.load(data_dir / "accuracy.csv")
        self.summary = self.load(data_dir / "results_summary.csv")
        self.has_energy = "energy_j" in self.raw and self.raw["energy_j"].notna().any()

    @staticmethod
    def load(path):
        frame = pd.read_csv(path)
        for col in ["N", "dist", "parameter", "repetition", "step"]:
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="raise").astype(int)
        if "kernel_type" in frame:
            frame["kernel_type"] = frame["kernel_type"].fillna("none").astype(str)
        return frame

    # ------------------------------------------------------------------ util
    @staticmethod
    def representative(values):
        """Até 3 valores (menor, central, maior), respeitando o limite de cores."""
        values = sorted(values)
        if len(values) <= 3:
            return values
        return [values[0], values[len(values) // 2], values[-1]]

    @staticmethod
    def log_error_axis(ax, axis="x"):
        """Eixo log legível também quando os dados cobrem menos de uma década."""
        target = ax.xaxis if axis == "x" else ax.yaxis
        target.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=10))
        target.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
        target.set_minor_formatter(NullFormatter())

    def slice_op(self, frame, operation, conv_dist=CONV_DIST, kernel=CONV_KERNEL, gemm_dist=GEMM_DIST):
        sub = frame[frame.operation == operation]
        if operation == "gemm":
            return sub[sub.dist == gemm_dist]
        return sub[(sub.dist == conv_dist) & (sub.kernel_type == kernel)]

    def slice_note(self, operation):
        if operation == "gemm":
            return f"entrada {DIST_LABEL[GEMM_DIST]}"
        return f"entrada {DIST_LABEL[CONV_DIST]}, kernel {KERNEL_LABEL[CONV_KERNEL]}"

    def save(self, fig, category, name, title, question, source, filters):
        directory = self.output / category
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for extension in self.formats:
            path = directory / f"{name}.{extension}"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            paths.append(path.relative_to(self.output).as_posix())
        plt.close(fig)
        self.index.append({"categoria": category, "figura": name, "titulo": title,
                           "pergunta": question, "filtros": filters, "fonte": source,
                           "arquivos": "; ".join(paths)})

    @staticmethod
    def style_axes(ax, xlabel=None, ylabel=None, title=None, logx=False, logy=False):
        if xlabel:
            ax.set_xlabel(xlabel, color=INK)
        if ylabel:
            ax.set_ylabel(ylabel, color=INK)
        if title:
            ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
        if logx:
            ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
            Plotter.log_error_axis(ax, "y")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)

    @staticmethod
    def set_n_ticks(ax, values):
        values = sorted(set(int(v) for v in values))
        ax.set_xticks(values)
        ax.set_xticklabels([str(v) for v in values])
        ax.minorticks_off() if ax.get_xscale() == "log" else None

    @staticmethod
    def reference_line(ax, value=1.0):
        ax.axhline(value, color=MUTED, linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)

    @staticmethod
    def legend_below(fig, handles=None, labels=None, axes=None, ncol=None, title=None):
        """Legenda única da figura, abaixo dos painéis, fora da área de desenho."""
        if handles is None:
            seen = {}
            for ax in np.atleast_1d(axes).ravel():
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    if label not in seen and not label.startswith("_"):
                        seen[label] = handle
            labels, handles = list(seen.keys()), list(seen.values())
        if not handles:
            return
        ncol = ncol or min(len(handles), 4)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   ncol=ncol, frameon=False, fontsize=8, title=title, title_fontsize=8,
                   handlelength=2.6, columnspacing=1.6)

    @staticmethod
    def title(fig, text, subtitle=None):
        """Título e subtítulo acima dos painéis, com afastamento fixo em pontos.

        Deve ser chamado DEPOIS do tight_layout (que usa rect com topo 1.0)."""
        base = fig.transFigure
        y_title = 4 if not subtitle else 18
        if subtitle:
            fig.text(0.01, 1.0, subtitle, ha="left", va="bottom", fontsize=8.5, color=MUTED,
                     transform=offset_copy(base, fig=fig, y=4, units="points"))
        fig.text(0.01, 1.0, text, ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK,
                 transform=offset_copy(base, fig=fig, y=y_title, units="points"))

    def panels(self, count, width=4.0, height=3.4, sharey=True):
        fig, axes = plt.subplots(1, count, figsize=(max(width * count, 6.0), height),
                                 sharey=sharey, squeeze=False)
        return fig, axes[0]

    def series_line(self, ax, x, y, err, color, step, label, asymmetric=False):
        linestyle, marker = STEP_STYLE.get(step, ("-", "o"))
        yerr = np.array(err).T if asymmetric else err
        ax.errorbar(x, y, yerr=yerr, color=color, linestyle=linestyle, marker=marker,
                    markersize=5.5, linewidth=1.6, capsize=2.5, elinewidth=0.9, label=label,
                    markeredgecolor="white", markeredgewidth=0.6)

    # ----------------------------------------------------------- desempenho
    def performance(self):
        metrics = [
            ("time_sec", "Tempo por execução do kernel (s)", True, "tempo"),
            ("effective_gops", "GFLOP/s efetivos (trabalho da versão completa)", False, "gflops"),
            ("ipc", "Instruções por ciclo (IPC)", False, "ipc"),
        ]
        for operation in OPERATIONS:
            data = self.slice_op(self.raw, operation)
            if data.empty:
                continue
            params = sorted(data.parameter.unique())
            for metric, label, logy, short in metrics:
                if metric not in data or data[metric].isna().all():
                    continue
                fig, axes = self.panels(len(params))
                for ax, parameter in zip(axes, params):
                    panel = data[data.parameter == parameter]
                    for precision in ["double", "float"]:
                        for step in sorted(panel.step.unique()):
                            group = panel[(panel.precision == precision) & (panel.step == step)]
                            points = [(n, *mean_ci(g[metric])) for n, g in group.groupby("N")]
                            if not points:
                                continue
                            x, y, e = map(np.asarray, zip(*sorted(points)))
                            self.series_line(ax, x, y, e, PREC_COLOR[precision], step,
                                             f"{precision}, {step_label(step)}")
                    self.style_axes(ax, "Ordem da matriz (N)", label if ax is axes[0] else None,
                                    f"{param_symbol(operation)} = {parameter}", logx=True, logy=logy)
                    self.set_n_ticks(ax, panel.N.unique())
                title = f"{OP_LABEL[operation]}: {label.split(' (')[0].lower()}"
                fig.tight_layout()
                self.title(fig, title, f"Média e IC 95% das repetições; {self.slice_note(operation)}.")
                self.legend_below(fig, axes=axes)
                self.save(fig, "01_desempenho", f"{short}_{operation}", title,
                          f"Como {label.lower()} varia com N, precisão e passo?", "results_raw.csv",
                          self.slice_note(operation))

    # ---------------------------------------------------------- comparações
    def precision_speedup(self):
        fig, axes = self.panels(len(OPERATIONS), sharey=False)
        for ax, operation in zip(axes, OPERATIONS):
            data = self.slice_op(self.comp, operation)
            data = data[data.comparison == "precision"]
            for i, parameter in enumerate(self.representative(data.parameter.unique())):
                group = data[data.parameter == parameter]
                points = [(n, *gmean_ci(g.time_speedup)) for n, g in group.groupby("N")]
                if not points:
                    continue
                points.sort()
                x = np.array([p[0] for p in points]); y = np.array([p[1] for p in points])
                err = [p[2] for p in points]
                ax.errorbar(x, y, yerr=np.array(err).T, color=DIM[i], linestyle=["-", "--", ":"][i],
                            marker="os^"[i], markersize=5, capsize=2.5, linewidth=1.5,
                            label=f"{param_symbol(operation)} = {parameter}",
                            markeredgecolor="white", markeredgewidth=0.6)
            self.reference_line(ax)
            self.style_axes(ax, "Ordem da matriz (N)", "Speedup double / float" if ax is axes[0] else None,
                            OP_LABEL[operation], logx=True)
            self.set_n_ticks(ax, data.N.unique())
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # legenda própria de cada painel, abaixo dele (K e B diferem)
                ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.24),
                          ncol=min(len(handles), 3), frameon=False, fontsize=7.5)
        title = "Ganho de trocar double por float (versões completas)"
        fig.tight_layout()
        self.title(fig, title, "Acima de 1: float mais rápido. Média geométrica e IC 95% das repetições "
                               "pareadas; até 3 valores de K ou B (menor, central, maior).")
        self.save(fig, "02_comparacoes", "precisao_speedup", title,
                  "Quanto a precisão float acelera cada operação?", "comparisons_raw.csv",
                  "comparison=precision; recorte padrão de entrada e kernel")

    def approximation_speedup(self):
        for operation in OPERATIONS:
            data = self.slice_op(self.comp, operation)
            data = data[data.comparison.isin(["approximation_double", "approximation_float"])]
            if data.empty:
                continue
            params = sorted(data.parameter.unique())
            sizes = sorted(data.N.unique())
            fig, axes = self.panels(len(params))
            for ax, parameter in zip(axes, params):
                panel = data[data.parameter == parameter]
                for comparison, precision in [("approximation_double", "double"), ("approximation_float", "float")]:
                    for i, n in enumerate(sizes):
                        group = panel[(panel.comparison == comparison) & (panel.N == n)]
                        points = [(s, *gmean_ci(g.time_speedup)) for s, g in group.groupby("step")]
                        if not points:
                            continue
                        points.sort()
                        x = np.array([p[0] for p in points]); y = np.array([p[1] for p in points])
                        ax.errorbar(x, y, yerr=np.array([p[2] for p in points]).T,
                                    color=DIM[i % 3], linestyle="-" if precision == "double" else "--",
                                    marker="o" if precision == "double" else "s", markersize=5,
                                    capsize=2.5, linewidth=1.5, label=f"N = {n}, {precision}",
                                    markeredgecolor="white", markeredgewidth=0.6)
                self.reference_line(ax)
                self.style_axes(ax, "Passo da aproximação", "Speedup completa / aproximada" if ax is axes[0] else None,
                                f"{param_symbol(operation)} = {parameter}")
                ax.set_xticks(sorted(panel.step.unique()))
            title = f"Ganho de tempo da aproximação: {OP_LABEL[operation].lower()}"
            fig.tight_layout()
            self.title(fig, title, f"Mesma precisão na referência e na versão aproximada; {self.slice_note(operation)}.")
            self.legend_below(fig, axes=axes, ncol=len(sizes))
            self.save(fig, "02_comparacoes", f"aproximacao_{operation}", title,
                      "Quanto cada passo de aproximação reduz o tempo?", "comparisons_raw.csv",
                      self.slice_note(operation))

    def layout_comparison(self):
        conv = self.raw[self.raw.operation.isin(["conv_linear", "conv_malloc"])
                        & (self.raw.dist == CONV_DIST) & (self.raw.kernel_type == CONV_KERNEL)]
        if conv.empty:
            return
        keys = ["precision", "step", "N", "parameter", "repetition"]
        wide = conv.pivot_table(index=keys, columns="operation", values="time_sec", aggfunc="first").dropna()
        wide["ratio"] = wide["conv_linear"] / wide["conv_malloc"]
        wide = wide.reset_index()
        params = sorted(wide.parameter.unique())
        fig, axes = self.panels(len(params))
        for ax, parameter in zip(axes, params):
            panel = wide[wide.parameter == parameter]
            for precision in ["double", "float"]:
                for step in sorted(panel.step.unique()):
                    group = panel[(panel.precision == precision) & (panel.step == step)]
                    points = [(n, *gmean_ci(g.ratio)) for n, g in group.groupby("N")]
                    if not points:
                        continue
                    points.sort()
                    self.series_line(ax, [p[0] for p in points], [p[1] for p in points],
                                     [p[2] for p in points], PREC_COLOR[precision], step,
                                     f"{precision}, {step_label(step)}", asymmetric=True)
            self.reference_line(ax)
            self.style_axes(ax, "Ordem da matriz (N)", "Tempo contígua / tempo por linhas" if ax is axes[0] else None,
                            f"K = {parameter}", logx=True)
            self.set_n_ticks(ax, panel.N.unique())
        title = "Efeito da organização da memória na convolução"
        fig.tight_layout()
        self.title(fig, title, f"Acima de 1: alocação por linhas mais rápida; {self.slice_note('conv_linear')}.")
        self.legend_below(fig, axes=axes)
        self.save(fig, "02_comparacoes", "layout_memoria", title,
                  "A alocação por linhas muda o tempo da convolução?", "results_raw.csv",
                  self.slice_note("conv_linear"))

    def gemm_blocks(self):
        data = self.slice_op(self.raw, "gemm")
        if data.empty:
            return
        sizes = sorted(data.N.unique())
        fig, axes = self.panels(len(sizes))
        for ax, n in zip(axes, sizes):
            panel = data[data.N == n]
            for precision in ["double", "float"]:
                for step in sorted(panel.step.unique()):
                    group = panel[(panel.precision == precision) & (panel.step == step)]
                    points = [(b, *mean_ci(g.effective_gops)) for b, g in group.groupby("parameter")]
                    if not points:
                        continue
                    x, y, e = map(np.asarray, zip(*sorted(points)))
                    self.series_line(ax, x, y, e, PREC_COLOR[precision], step, f"{precision}, {step_label(step)}")
            self.style_axes(ax, "Tamanho do bloco (B)", "GFLOP/s efetivos" if ax is axes[0] else None,
                            f"N = {n}", logx=True)
            self.set_n_ticks(ax, panel.parameter.unique())
        title = "Sensibilidade da GEMM ao tamanho do bloco"
        fig.tight_layout()
        self.title(fig, title, f"GFLOP/s calculados com o trabalho da versão completa; {self.slice_note('gemm')}.")
        self.legend_below(fig, axes=axes)
        self.save(fig, "03_gemm_blocos", "gemm_blocos_gflops", title,
                  "Qual bloco maximiza a vazão útil da GEMM?", "results_raw.csv", self.slice_note("gemm"))

    # ------------------------------------------------------------- qualidade
    def accuracy_plots(self):
        acc = self.accuracy
        # Precisão: float contra double, versões completas.
        prec = acc[acc.comparison == "precision"]
        fig, axes = self.panels(len(OPERATIONS))
        for ax, operation in zip(axes, OPERATIONS):
            panel = prec[prec.operation == operation]
            dists = sorted(panel.dist.unique())
            for i, dist in enumerate(dists):
                group = panel[panel.dist == dist].groupby("N").error_rel_norm.max().sort_index()
                ax.plot(group.index, group.values, color=DIM[i % 3], marker="osD"[i % 3],
                        linestyle=["-", "--", ":"][i % 3], linewidth=1.5, markersize=5,
                        label=f"entrada {DIST_LABEL.get(dist, dist)}", markeredgecolor="white", markeredgewidth=0.6)
            self.style_axes(ax, "Ordem da matriz (N)", "Erro relativo pela norma" if ax is axes[0] else None,
                            OP_LABEL[operation], logx=True, logy=True)
            self.set_n_ticks(ax, panel.N.unique())
        title = "Erro de float em relação a double (versões completas)"
        fig.tight_layout()
        self.title(fig, title, "Pior caso entre parâmetros (K, B) e kernels; erro ‖C − R‖ / ‖R‖.")
        self.legend_below(fig, axes=axes)
        self.save(fig, "04_qualidade_numerica", "precisao_erro", title,
                  "Qual erro a troca de double por float introduz?", "accuracy.csv", "comparison=precision")

        # Aproximação na convolução: erro por passo, um painel por kernel.
        conv = acc[(acc.operation == "conv_linear") & (acc.comparison == "approximation_double")]
        kernels = [k for k in ["rand", "gauss", "sobel"] if k in set(conv.kernel_type)]
        if kernels:
            fig, axes = self.panels(len(kernels))
            n_max = conv.N.max()
            for ax, kernel in zip(axes, kernels):
                panel = conv[(conv.kernel_type == kernel) & (conv.N == n_max)]
                for i, dist in enumerate(sorted(panel.dist.unique())):
                    for j, parameter in enumerate(sorted(panel.parameter.unique())):
                        group = panel[(panel.dist == dist) & (panel.parameter == parameter)].sort_values("step")
                        ax.plot(group.step, group.error_rel_norm, color=DIM[i % 3],
                                linestyle=["-", "--", ":"][j % 3], marker="os^"[j % 3], markersize=5,
                                linewidth=1.5, label=f"entrada {DIST_LABEL[dist]}, K = {parameter}",
                                markeredgecolor="white", markeredgewidth=0.6)
                self.style_axes(ax, "Passo da aproximação", "Erro relativo pela norma" if ax is axes[0] else None,
                                f"Kernel {KERNEL_LABEL[kernel]}", logy=True)
                ax.set_xticks(sorted(panel.step.unique()))
            title = "Erro do skip_kernel por tipo de kernel"
            fig.tight_layout()
            self.title(fig, title, f"N = {n_max}; versão double (float coincide até 10⁻⁶); borda excluída.")
            self.legend_below(fig, axes=axes, ncol=len(conv.dist.unique()))
            self.save(fig, "04_qualidade_numerica", "skip_kernel_erro", title,
                      "Como o erro do skip_kernel depende do kernel, da entrada e do passo?",
                      "accuracy.csv", f"conv_linear; approximation_double; N={n_max}")

        # Aproximação na GEMM: erro por passo, um painel por distribuição.
        gemm = acc[(acc.operation == "gemm") & (acc.comparison == "approximation_double")]
        if not gemm.empty:
            dists = sorted(gemm.dist.unique())
            fig, axes = self.panels(len(dists))
            for ax, dist in zip(axes, dists):
                panel = gemm[gemm.dist == dist]
                for i, n in enumerate(sorted(panel.N.unique())):
                    group = panel[panel.N == n].groupby("step").error_rel_norm.mean().sort_index()
                    ax.plot(group.index, group.values, color=DIM[i % 3], marker="os^"[i % 3],
                            linestyle=["-", "--", ":"][i % 3], linewidth=1.5, markersize=5,
                            label=f"N = {n}", markeredgecolor="white", markeredgewidth=0.6)
                self.style_axes(ax, "Passo da aproximação", "Erro relativo pela norma" if ax is axes[0] else None,
                                f"Entrada {DIST_LABEL.get(dist, dist)}", logy=True)
                ax.set_xticks(sorted(panel.step.unique()))
            title = "Erro do skip_k na GEMM"
            fig.tight_layout()
            self.title(fig, title, "Versão double; o erro não depende do bloco (mesmos índices k mantidos).")
            self.legend_below(fig, axes=axes)
            self.save(fig, "04_qualidade_numerica", "skip_k_erro", title,
                      "Como o erro do skip_k depende da distribuição, de N e do passo?", "accuracy.csv",
                      "gemm; approximation_double")

    # ------------------------------------------------------------ compromisso
    def tradeoff_handles(self, steps):
        handles = [Line2D([], [], color=DIM[i % 3], marker="o", linestyle="none", markersize=7,
                          label=f"passo {s}") for i, s in enumerate(steps)]
        handles += [Line2D([], [], color=MUTED, marker=COMPARISON_MARK[c], linestyle="none", markersize=7,
                           markerfacecolor="white" if c == "combined" else MUTED,
                           label=COMPARISON_LABEL[c]) for c in COMPARISON_MARK]
        return handles

    def tradeoff(self):
        speed = (self.comp.groupby(["operation", "N", "dist", "parameter", "kernel_type", "comparison", "step"])
                 .time_speedup.apply(lambda v: gmean_ci(v)[0]).rename("speedup").reset_index())
        merged = speed.merge(self.accuracy, on=["operation", "N", "dist", "parameter", "kernel_type",
                                                "comparison", "step"], how="inner")
        merged = merged[merged.comparison.isin(COMPARISON_MARK)]
        for operation in OPERATIONS:
            data = merged[merged.operation == operation]
            if data.empty:
                continue
            if operation == "gemm":
                facets = [(f"Entrada {DIST_LABEL.get(d, d)}", data[data.dist == d]) for d in sorted(data.dist.unique())]
            else:
                facets = [(f"Kernel {KERNEL_LABEL.get(k, k)}", data[data.kernel_type == k])
                          for k in ["rand", "gauss", "sobel"] if k in set(data.kernel_type)]
            steps = sorted(data.step.unique())
            fig, axes = self.panels(len(facets), height=3.6)
            for ax, (label, panel) in zip(axes, facets):
                for i, step in enumerate(steps):
                    for comparison, marker in COMPARISON_MARK.items():
                        group = panel[(panel.step == step) & (panel.comparison == comparison)]
                        if group.empty:
                            continue
                        ax.scatter(group.error_rel_norm, group.speedup, s=30, marker=marker,
                                   facecolor=DIM[i % 3] if comparison != "combined" else "white",
                                   edgecolor=DIM[i % 3], linewidth=1.1, alpha=0.9, zorder=3)
                self.reference_line(ax)
                self.style_axes(ax, "Erro relativo pela norma (log)", "Speedup de tempo" if ax is axes[0] else None,
                                label)
                ax.set_xscale("log")
                self.log_error_axis(ax, "x")
            title = f"Compromisso entre erro e desempenho: {OP_LABEL[operation].lower()}"
            subtitle = "Cada ponto é uma combinação de N e parâmetro, todas as entradas."
            fig.tight_layout()
            self.title(fig, title, subtitle)
            handles = self.tradeoff_handles(steps)
            self.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles],
                              ncol=max(len(steps), 3))
            self.save(fig, "05_compromisso", f"tradeoff_{operation}", title,
                      "Quanto de erro acompanha cada ganho de tempo?", "comparisons_raw.csv + accuracy.csv",
                      "todas as entradas; comparações aproximadas e combinada")

    # ---------------------------------------------------------------- energia
    def energy(self):
        if not self.has_energy:
            return
        # Energia absoluta por execução do kernel.
        for metric, label, short, logy in [("energy_j", "Energia por execução (J)", "energia", True),
                                           ("power_w", "Potência média (W)", "potencia", False)]:
            fig, axes = self.panels(len(OPERATIONS), sharey=False)
            for ax, operation in zip(axes, OPERATIONS):
                data = self.slice_op(self.raw, operation)
                parameter = sorted(data.parameter.unique())[len(data.parameter.unique()) // 2] if not data.empty else None
                data = data[data.parameter == parameter]
                for precision in ["double", "float"]:
                    for step in sorted(data.step.unique()):
                        group = data[(data.precision == precision) & (data.step == step)]
                        points = [(n, *mean_ci(g[metric])) for n, g in group.groupby("N")]
                        if not points:
                            continue
                        x, y, e = map(np.asarray, zip(*sorted(points)))
                        self.series_line(ax, x, y, e, PREC_COLOR[precision], step, f"{precision}, {step_label(step)}")
                self.style_axes(ax, "Ordem da matriz (N)", label if ax is axes[0] else None,
                                f"{OP_LABEL[operation]}, {param_symbol(operation)} = {parameter}",
                                logx=True, logy=logy)
                self.set_n_ticks(ax, data.N.unique())
            title = label.split(" (")[0] + " do pacote e da DRAM"
            fig.tight_layout()
            self.title(fig, title, "RAPL em modo sistema durante o lote de repetições; média e IC 95%.")
            self.legend_below(fig, axes=axes)
            self.save(fig, "06_energia", short, title, f"Como {label.lower()} varia com N, precisão e passo?",
                      "results_raw.csv", "parâmetro central de cada operação; recorte padrão")

        # Economia de energia contra speedup de tempo.
        comp = self.comp[self.comp.comparison.isin(COMPARISON_MARK)].dropna(subset=["energy_saving"])
        if comp.empty:
            return
        agg = (comp.groupby(["operation", "N", "dist", "parameter", "kernel_type", "comparison", "step"])
               .agg(speedup=("time_speedup", lambda v: gmean_ci(v)[0]),
                    saving=("energy_saving", lambda v: gmean_ci(v)[0])).reset_index())
        steps = sorted(agg.step.unique())
        fig, axes = self.panels(len(OPERATIONS), sharey=False)
        for ax, operation in zip(axes, OPERATIONS):
            data = agg[agg.operation == operation]
            for i, step in enumerate(steps):
                for comparison, marker in COMPARISON_MARK.items():
                    group = data[(data.step == step) & (data.comparison == comparison)]
                    if group.empty:
                        continue
                    ax.scatter(group.speedup, group.saving, s=30, marker=marker,
                               facecolor=DIM[i % 3] if comparison != "combined" else "white",
                               edgecolor=DIM[i % 3], linewidth=1.1, alpha=0.9, zorder=3)
            if not data.empty:
                low = min(data.speedup.min(), data.saving.min(), 1.0) * 0.95
                high = max(data.speedup.max(), data.saving.max(), 1.0) * 1.05
                ax.plot([low, high], [low, high], color=MUTED, linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
            self.style_axes(ax, "Speedup de tempo", "Economia de energia (referência / candidata)"
                            if ax is axes[0] else None, OP_LABEL[operation])
        title = "Economia de energia contra ganho de tempo"
        fig.tight_layout()
        self.title(fig, title, "Diagonal: economia de energia igual ao speedup. Acima dela, a potência caiu.")
        handles = self.tradeoff_handles(steps)
        self.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles],
                          ncol=max(len(steps), 3))
        self.save(fig, "06_energia", "energia_vs_speedup", title,
                  "A economia de energia acompanha o ganho de tempo?", "comparisons_raw.csv",
                  "todas as entradas; comparações aproximadas e combinada")

    # ------------------------------------------------------ microarquitetura
    def microarchitecture(self):
        if "cache_miss_rate" not in self.raw:
            return
        n_max = self.raw.N.max()
        fig, axes = self.panels(len(OPERATIONS), sharey=False)
        for ax, operation in zip(axes, OPERATIONS):
            data = self.slice_op(self.raw, operation)
            data = data[data.N == n_max]
            for precision in ["double", "float"]:
                for step in sorted(data.step.unique()):
                    group = data[(data.precision == precision) & (data.step == step)]
                    if group.empty:
                        continue
                    agg = group.groupby("parameter")[["cache_miss_rate", "ipc"]].mean()
                    _, marker = STEP_STYLE.get(step, ("-", "o"))
                    ax.scatter(agg.cache_miss_rate, agg.ipc, s=38, marker=marker, color=PREC_COLOR[precision],
                               edgecolor="white", linewidth=0.6, zorder=3, label=f"{precision}, {step_label(step)}")
            self.style_axes(ax, "Taxa de faltas de cache", "IPC" if ax is axes[0] else None, OP_LABEL[operation])
        title = "Assinatura microarquitetural das implementações"
        fig.tight_layout()
        self.title(fig, title, f"N = {n_max}; cada ponto é um valor de K ou B; recorte padrão de entrada e kernel.")
        self.legend_below(fig, axes=axes)
        self.save(fig, "07_microarquitetura", "ipc_cache", title,
                  "Como IPC e faltas de cache distinguem as versões?", "results_raw.csv", f"N={n_max}")

    # --------------------------------------------------- reprodutibilidade
    def reproducibility(self):
        n_max = self.raw.N.max()
        columns = [("time_sec", "Tempo"), ("cycles", "Ciclos"), ("instructions", "Instruções"),
                   ("energy_j", "Energia")]
        columns = [(c, l) for c, l in columns if c in self.raw and self.raw[c].notna().any()]
        rows, labels = [], []
        for operation in OPERATIONS:
            data = self.slice_op(self.raw, operation)
            data = data[data.N == n_max]
            if data.empty:
                continue
            parameter = sorted(data.parameter.unique())[len(data.parameter.unique()) // 2]
            data = data[data.parameter == parameter]
            for (precision, step), group in data.groupby(["precision", "step"]):
                rows.append([100 * group[c].std(ddof=1) / group[c].mean() for c, _ in columns])
                labels.append(f"{OP_LABEL[operation].split(' (')[0]} {operation.split('_')[-1] if operation != 'gemm' else ''} "
                              f"{precision}, passo {step}".replace("  ", " "))
        if not rows:
            return
        matrix = np.array(rows)
        fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(rows) + 1.8))
        positive = matrix[np.isfinite(matrix) & (matrix > 0)]
        floor = positive.min() / 10 if len(positive) else 1e-6
        image = ax.imshow(np.log10(np.maximum(matrix, floor)), aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xticks(range(len(columns)), [l for _, l in columns], fontsize=8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                shade = (np.log10(max(value, floor)) - np.log10(floor)) / max(np.log10(positive.max()) - np.log10(floor), 1e-9) if len(positive) else 0
                ax.text(j, i, f"{value:.2g}%", ha="center", va="center", fontsize=6.5,
                        color="white" if shade > 0.6 else INK)
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.08, fraction=0.04)
        cbar.set_label("log10 do coeficiente de variação (%)", fontsize=8)
        title = "Reprodutibilidade das métricas"
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        self.save(fig, "08_reprodutibilidade", "cv_metricas", title,
                  "Quais métricas variaram mais entre as repetições?", "results_raw.csv",
                  f"N={n_max}; parâmetro central; recorte padrão")

        fig, axes = self.panels(len(OPERATIONS), sharey=False, height=3.8)
        for ax, operation in zip(axes, OPERATIONS):
            data = self.slice_op(self.raw, operation)
            data = data[data.N == n_max]
            if data.empty:
                continue
            parameter = sorted(data.parameter.unique())[len(data.parameter.unique()) // 2]
            data = data[data.parameter == parameter]
            groups = sorted(data.groupby(["precision", "step"]), key=lambda kv: (kv[0][0], kv[0][1]))
            values = [g.time_sec.to_numpy() * 1e3 for _, g in groups]
            labels = [f"{p[0]}\npasso {p[1]}" for p, _ in groups]
            boxes = ax.boxplot(values, patch_artist=True, showfliers=True, widths=0.55,
                               medianprops={"color": INK}, flierprops={"markersize": 3},
                               **boxplot_label_argument(labels))
            for patch, (key, _) in zip(boxes["boxes"], groups):
                patch.set_facecolor(PREC_COLOR[key[0]])
                patch.set_alpha(0.55)
                patch.set_edgecolor(PREC_COLOR[key[0]])
            self.style_axes(ax, None, "Tempo por execução (ms)" if ax is axes[0] else None,
                            f"{OP_LABEL[operation]}, {param_symbol(operation)} = {parameter}")
            ax.tick_params(axis="x", labelsize=7)
        title = "Distribuição das repetições: tempo por execução"
        fig.tight_layout()
        self.title(fig, title, f"N = {n_max}; cada caixa reúne as execuções independentes.")
        self.save(fig, "08_reprodutibilidade", "boxplot_tempo", title,
                  "Como as medições de tempo se distribuem?", "results_raw.csv", f"N={n_max}")

    # ------------------------------------------------------------ diagnósticos
    def diagnostics(self):
        if "effective_ghz" not in self.raw or self.raw.effective_ghz.isna().all():
            return
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        groups = [(op, self.raw[self.raw.operation == op].effective_ghz.dropna().to_numpy()) for op in OPERATIONS]
        groups = [(op, v) for op, v in groups if len(v)]
        boxes = ax.boxplot([v for _, v in groups], patch_artist=True, widths=0.5, medianprops={"color": INK},
                           flierprops={"markersize": 3}, **boxplot_label_argument([OP_LABEL[o] for o, _ in groups]))
        for patch in boxes["boxes"]:
            patch.set_facecolor(PREC_COLOR['double']); patch.set_alpha(0.45); patch.set_edgecolor(PREC_COLOR['double'])
        self.style_axes(ax, None, "Ciclos / tempo (GHz)")
        ax.tick_params(axis="x", labelsize=7.5)
        title = "Frequência efetiva durante o kernel"
        fig.tight_layout()
        self.title(fig, title, "Com governor performance e sem turbo, deve ficar perto da nominal (2,4 GHz no E5-2407 v2).")
        self.save(fig, "09_diagnosticos", "frequencia_efetiva", title,
                  "A frequência ficou estável durante a campanha?", "results_raw.csv", "todas as execuções")

    def write_index(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "INDICE_GRAFICOS.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.index[0].keys())
            writer.writeheader()
            writer.writerows(self.index)
        lines = ["# Índice dos gráficos", "",
                 f"Total: **{len(self.index)} figuras**, em {', '.join(self.formats)}.", "",
                 f"Recorte padrão: convolução com entrada {DIST_LABEL[CONV_DIST]} e kernel "
                 f"{KERNEL_LABEL[CONV_KERNEL]}; GEMM com entrada {DIST_LABEL[GEMM_DIST]}.", "",
                 "| Categoria | Figura | Pergunta | Filtros | Fonte |", "| --- | --- | --- | --- | --- |"]
        for item in self.index:
            lines.append(f"| `{item['categoria']}` | `{item['figura']}` | {item['pergunta']} | "
                         f"{item['filtros']} | `{item['fonte']}` |")
        (self.output / "INDICE_GRAFICOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self):
        self.performance()
        self.precision_speedup()
        self.approximation_speedup()
        self.layout_comparison()
        self.gemm_blocks()
        self.accuracy_plots()
        self.tradeoff()
        self.energy()
        self.microarchitecture()
        self.reproducibility()
        self.diagnostics()
        self.write_index()


def configure_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "lines.linewidth": 1.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
    })


def main():
    parser = argparse.ArgumentParser(description="Gera PDF e PNG dos resultados do TCC2.")
    parser.add_argument("--data", default="results/analysis_data", help="diretório com os CSVs da campanha")
    parser.add_argument("--output", default="results/figures", help="diretório de saída")
    parser.add_argument("--formats", default="pdf,png", help="formatos separados por vírgula")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--clean", action="store_true", help="remove a saída anterior de gráficos")
    args = parser.parse_args()
    data, output = Path(args.data).resolve(), Path(args.output).resolve()
    required = ["results_raw.csv", "results_summary.csv", "comparisons_raw.csv", "accuracy.csv"]
    missing = [name for name in required if not (data / name).is_file()]
    if missing:
        raise SystemExit("[ERRO] CSVs ausentes em " + str(data) + ": " + ", ".join(missing))
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
