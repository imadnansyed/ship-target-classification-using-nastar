import json
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Layout and theme
# ---------------------------------------------------------------------------
PAGE = (8.27, 11.69)  # A4 portrait, inches
MARGIN = 0.08

NAVY = "#12284C"
ACCENT = "#1F77B4"
TEXT = "#222222"
MUTED = "#6B7280"
RULE = "#D1D5DB"
ZEBRA = "#F3F6FA"
NOTICE_BG = "#FFF4D6"
NOTICE_EDGE = "#E0A800"
SERIES = ["#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD"]

# ---------------------------------------------------------------------------
# Static content
# ---------------------------------------------------------------------------
DATASET_REF = (
    "Hosseiny, B., Kamirul, K., Pappas, O., & Achim, A. (2026). "
    "NASTaR: A NovaSAR-Based Automated Ship Target Recognition Dataset. "
    "IEEE Geoscience and Remote Sensing Letters, 23, Article 1502605. "
    "https://doi.org/10.1109/LGRS.2026.3679941. Dataset DOI: "
    "https://doi.org/10.5523/bris.2tfa6x37oerz2lyiw6hp47058. "
    "Dataset licence: Non-Commercial Government Licence for public sector "
    "information; check the supplied LICENSE.txt before redistribution."
)

FIGURE_PAGES = [
    ("fig_cv_confusion_matrices.png",
     "Confusion matrices (cross-validated, out-of-fold)",
     "Each row is a true class; the diagonal is the per-class recall."),
    ("fig_per_class_f1.png",
     "Per-class F1 score",
     "Pooled out-of-fold predictions."),
    ("fig_cv_shoreline.png",
     "Inshore versus offshore",
     "Macro-F1 on inshore and offshore patches."),
    ("fig_cv_reliability.png",
     "Calibration",
     "Confidence versus observed accuracy; the dashed line is perfect calibration."),
    ("fig_cv_confident_errors.png",
     "Most confident mistakes of the best model",
     "Raw patches (log stretch) with true class, predicted class and confidence."),
    ("fig_gradcam_examples.png",
     "Grad-CAM examples",
     "Warm regions raised the score of the predicted class."),
]

LIMITATIONS = [
    "The dataset is small: about 1,200 images in four classes (cargo, tanker, fishing, "
    "passenger). Small differences between models are within noise.",
    "Only these four ship types are recognised. Any other vessel, or any other kind of "
    "image, will still be assigned one of the four classes.",
    "Patches come from one sensor (NovaSAR, S-band, mostly HH) at 2.5 m per pixel. "
    "The models are not validated on other sensors.",
    "Ship length and width from AIS alone give a competitive score, so part of what an "
    "image model learns may be ship size.",
    "Cross-validation groups images by vessel. Patches from the same scene can still "
    "share sea state; the scene-grouped check addresses this.",
    "Confidence scores are calibrated on a small validation set and should be read as a "
    "guide, not a guarantee.",
    "The system supports analysts. It must not be the only basis for an operational decision.",
]


class ReportGenerator:
    """Class-diagram ReportGenerator: metrics, visualizations, outputPath, reportFormat."""

    def __init__(self, results_dir, figures_dir, models_dir, output_dir,
                 title="Ship Target Classification from SAR Images (NASTaR)"):
        self.results_dir, self.figures_dir = Path(results_dir), Path(figures_dir)
        self.models_dir, self.output_path = Path(models_dir), Path(output_dir)
        self.title, self.report_format = title, "pdf"
        self.metrics, self.visualizations, self.pages, self.notices = {}, [], [], []
        self.n_pages = 0
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _csv(self, name, **kw):
        p = self.results_dir / name
        return pd.read_csv(p, **kw) if p.exists() else None

    def _load(self):
        self.metrics["comparison"] = self._csv("final_comparison.csv", index_col=0)
        self.metrics["folds"] = self._csv("cv_fold_metrics.csv")
        self.metrics["significance"] = self._csv("cv_significance.csv")
        self.metrics["runs"] = self._csv("runs.csv")
        self.metrics["holdout"] = self._csv("holdout_test_results.csv")

        p = self.results_dir / "best_model.json"
        self.metrics["best"] = json.loads(p.read_text()) if p.exists() else None

        if self.metrics["comparison"] is None:
            self.notices.append(
                "Comparison results are not available. "
                "This report describes the single model that was trained.")
        if self.metrics["holdout"] is None:
            self.notices.append(
                "The selected model's locked holdout result is not available. "
                "Run Section 11 Step 08 once after model selection.")

    def _kind(self, model):
        folds = self.metrics["folds"]
        if folds is not None and model in set(folds["model"]):
            return folds.loc[folds["model"] == model, "kind"].iloc[0]
        return "baseline" if model.startswith(("Always", "AIS")) else "model"

    # ------------------------------------------------------------------
    # Page building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _rect(fig, x, y, w, h, color, edge="none", lw=0.0, z=0):
        fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure,
                                 facecolor=color, edgecolor=edge, linewidth=lw, zorder=z))

    def _page(self, heading):
        """A content page with an accent band, a heading and a thin rule."""
        fig = Figure(figsize=PAGE)
        self._rect(fig, 0, 0.985, 1, 0.015, ACCENT)
        fig.text(MARGIN, 0.955, heading, fontsize=16, fontweight="bold", color=NAVY, va="top")
        self._rect(fig, MARGIN, 0.918, 1 - 2 * MARGIN, 0.002, RULE)
        return fig

    def _text(self, fig, lines, y=0.89, size=10, width=95, gap=0.012,
              color=TEXT, bullet=False):
        """Write wrapped paragraphs top-down and return the next free y position."""
        for line in lines:
            if bullet:
                wrapped = textwrap.fill(line, width - 4, initial_indent="\u2022  ",
                                        subsequent_indent="    ")
            else:
                wrapped = textwrap.fill(line, width)
            fig.text(MARGIN, y, wrapped, fontsize=size, va="top", color=color)
            y -= 0.022 * (wrapped.count("\n") + 1) + gap
        return y

    def _callout(self, fig, text, y, size=10, width=92):
        """Highlighted notice box. Returns the next free y position."""
        wrapped = textwrap.fill(text, width)
        height = 0.022 * (wrapped.count("\n") + 1) + 0.02
        self._rect(fig, MARGIN - 0.01, y - height, 1 - 2 * MARGIN + 0.02, height,
                   NOTICE_BG, NOTICE_EDGE, 0.8)
        self._rect(fig, MARGIN - 0.01, y - height, 0.008, height, NOTICE_EDGE)
        fig.text(MARGIN + 0.015, y - 0.01, wrapped, fontsize=size, va="top", color=TEXT)
        return y - height - 0.015

    @staticmethod
    def _style_table(table, font_size=8, row_height=2.0, widths=True, ncols=0):
        """Navy header row, zebra stripes and light grid lines."""
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, row_height)
        if widths and ncols:
            table.auto_set_column_width(list(range(ncols)))
        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor(RULE)
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor(NAVY)
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor(ZEBRA if row % 2 == 0 else "white")
                cell.set_text_props(color=TEXT)

    @staticmethod
    def _style_axes(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(alpha=0.3, color=RULE)
        ax.set_axisbelow(True)

    def _image_page(self, path, heading, caption):
        fig = self._page(heading)
        ax = fig.add_axes([0.06, 0.2, 0.88, 0.68])
        ax.imshow(imread(str(path)))
        ax.axis("off")
        self._text(fig, [caption], y=0.17, size=9, color=MUTED)
        return fig

    def _add_footers(self):
        """Add a footer with the report title and 'Page n of N' (cover excluded)."""
        total = len(self.pages)
        for i, fig in enumerate(self.pages, start=1):
            if i == 1 or getattr(fig, "_footer_done", False):
                continue
            self._rect(fig, MARGIN, 0.045, 1 - 2 * MARGIN, 0.001, RULE)
            fig.text(MARGIN, 0.03, self.title, fontsize=7, color=MUTED, va="center")
            fig.text(1 - MARGIN, 0.03, f"Page {i} of {total}", fontsize=7,
                     color=MUTED, va="center", ha="right")
            fig._footer_done = True

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def generate_plots(self):
        """Training curves of the model chosen for the app, and the comparison bars."""
        self.visualizations = []
        best = self.metrics["best"]
        run = best.get("app_run_name") if best else None
        hist_path = self.models_dir / run / "history.csv" if run else None

        if hist_path is not None and hist_path.exists():
            h = pd.read_csv(hist_path)
            fig = self._page("Training curves of the deployed model")
            panels = [("loss", "Loss"), ("acc", "Accuracy"), ("macro_f1", "Macro-F1")]
            ax = None
            for k, (key, name) in enumerate(panels):
                ax = fig.add_axes([0.1, 0.70 - k * 0.26, 0.82, 0.19])
                ax.plot(h["epoch"], h[f"train_{key}"], label="train",
                        color=SERIES[0], linewidth=1.8)
                ax.plot(h["epoch"], h[f"val_{key}"], label="validation",
                        color=SERIES[2], linewidth=1.8)
                ax.set_title(name, fontsize=10, color=NAVY, loc="left", fontweight="bold")
                self._style_axes(ax)
                ax.legend(fontsize=8, frameon=False)
            ax.set_xlabel("epoch", fontsize=9, color=MUTED)
            self.visualizations.append(("curves", fig))

        comp = self.metrics["comparison"]
        if comp is not None:
            fig = self._page("Model comparison (cross-validated)")
            ax = fig.add_axes([0.1, 0.3, 0.85, 0.55])
            keys = [("acc", "Accuracy"), ("macro_f1", "Macro-F1"),
                    ("bal_acc", "Balanced acc.")]
            x, w = np.arange(len(comp)), 0.8 / len(keys)
            for j, (k, name) in enumerate(keys):
                ax.bar(x + j * w, comp[f"{k}_mean"], w, yerr=comp[f"{k}_std"],
                       capsize=3, label=name, color=SERIES[j],
                       error_kw={"ecolor": MUTED, "elinewidth": 0.8})
            ax.set_xticks(x + w)
            ax.set_xticklabels([textwrap.fill(str(m), 16) for m in comp.index], fontsize=7)
            ax.set_ylim(0, 1)
            self._style_axes(ax)
            ax.grid(axis="x", visible=False)
            ax.legend(fontsize=8, frameon=False, ncol=len(keys), loc="upper center",
                      bbox_to_anchor=(0.5, 1.08))
            self._text(fig, ["Bars show the mean over the cross-validation folds; "
                             "error bars are one standard deviation."],
                       y=0.22, size=9, color=MUTED)
            self.visualizations.append(("comparison", fig))
        return self.visualizations

    # ------------------------------------------------------------------
    # Report pages
    # ------------------------------------------------------------------
    def _cover_page(self, best):
        fig = Figure(figsize=PAGE)
        self._rect(fig, 0, 0.55, 1, 0.45, NAVY)
        self._rect(fig, 0, 0.545, 1, 0.01, ACCENT)
        fig.text(0.5, 0.78, textwrap.fill(self.title, 34), fontsize=24, fontweight="bold",
                 color="white", ha="center", va="center", linespacing=1.3)
        fig.text(0.5, 0.64, "Performance report", fontsize=15, color="#C9D6EA",
                 ha="center", va="center")
        fig.text(0.5, 0.50, time.strftime("%d %B %Y"), fontsize=11, color=MUTED, ha="center")
        if best:
            fig.text(0.5, 0.42, f"Deployed model: {best['label']}", fontsize=13,
                     fontweight="bold", color=NAVY, ha="center")
            fig.text(0.5, 0.385,
                     f"Cross-validated macro-F1 {best['macro_f1_mean']:.3f} "
                     f"\u00b1 {best['macro_f1_std']:.3f}",
                     fontsize=11, color=TEXT, ha="center")
        self._rect(fig, MARGIN, 0.135, 1 - 2 * MARGIN, 0.001, RULE)
        fig.text(MARGIN, 0.125, "Dataset reference", fontsize=8, fontweight="bold", color=NAVY)
        self._text(fig, [DATASET_REF], y=0.105, size=7.5, width=110, color=MUTED)
        return fig

    def _summary_page(self, best, comp):
        fig = self._page("Summary of results")
        y = 0.89
        for note in self.notices:
            y = self._callout(fig, "NOTICE: " + note, y)

        if comp is not None:
            rows = []
            for model, r in comp.sort_values("macro_f1_mean", ascending=False).iterrows():
                rows.append([textwrap.fill(str(model), 26), self._kind(model)] +
                            [f"{r[f'{k}_mean']:.3f} \u00b1 {r[f'{k}_std']:.3f}"
                             for k in ("acc", "macro_f1", "bal_acc")])
            top_edge = y - 0.01
            ax = fig.add_axes([0.05, top_edge - 0.34, 0.9, 0.34])
            ax.axis("off")
            t = ax.table(cellText=rows,
                         colLabels=["Model", "Kind", "Accuracy", "Macro-F1", "Balanced acc."],
                         loc="upper center", cellLoc="left")
            self._style_table(t, font_size=8, row_height=2.0, ncols=5)

            lines = []
            nets = comp[[self._kind(m) == "model" for m in comp.index]]
            if len(nets):
                top = nets["macro_f1_mean"].idxmax()
                lines.append(f"Best network by mean macro-F1: {top} "
                             f"({nets.loc[top, 'macro_f1_mean']:.3f} "
                             f"\u00b1 {nets.loc[top, 'macro_f1_std']:.3f}).")
                size_rows = [m for m in comp.index if m.startswith("AIS")]
                if size_rows:
                    gap = nets.loc[top, "macro_f1_mean"] - comp.loc[size_rows[0], "macro_f1_mean"]
                    verdict = ("The network is clearly ahead of ship size alone."
                               if gap > nets.loc[top, "macro_f1_std"]
                               else "The network is not clearly ahead of ship length "
                                    "and width alone.")
                    lines.append(f"Difference to the ship-size baseline: {gap:+.3f} "
                                 f"macro-F1. {verdict}")
            if best:
                lines.append(f"Model chosen for the app: {best['label']} ({best['reason']}).")
            self._text(fig, lines, y=top_edge - 0.03 * (len(rows) + 1) - 0.04, size=10)
        elif self.metrics["runs"] is not None:
            r = self.metrics["runs"].tail(5)[["run_name", "val_acc", "val_macro_f1"]].round(3)
            ax = fig.add_axes([0.05, 0.55, 0.9, 0.3])
            ax.axis("off")
            t = ax.table(cellText=r.values.tolist(), colLabels=list(r.columns),
                         loc="upper center")
            self._style_table(t, font_size=7, row_height=1.6, widths=False)
        return fig

    def _holdout_page(self, holdout):
        row = holdout.iloc[0]
        fig = self._page("Selected model \u2014 locked holdout test")
        rows = [["Run", str(row["run_name"])],
                ["Number of test patches", str(int(row["n"]))]]
        for key, label in (("acc", "Accuracy"),
                           ("macro_precision", "Macro precision"),
                           ("macro_recall", "Macro recall"),
                           ("macro_f1", "Macro-F1"),
                           ("bal_acc", "Balanced accuracy"),
                           ("flagged share", "Low-confidence flagged share")):
            if key in row.index and pd.notna(row[key]):
                rows.append([label, f"{float(row[key]):.3f}"])

        ax = fig.add_axes([0.08, 0.50, 0.84, 0.38])
        ax.axis("off")
        t = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     loc="upper center", cellLoc="left", colWidths=[0.6, 0.4])
        self._style_table(t, font_size=10, row_height=2.0, widths=False)
        self._text(fig, ["This is a single evaluation of the model selected using "
                         "cross-validation."], y=0.88 - 0.03 * (len(rows) + 1) - 0.03,
                   size=10, color=MUTED)
        return fig

    def _significance_page(self, sig):
        fig = self._page("Are the differences between models real?")
        s = sig[["A", "B", "diff A-B", "95% CI", "McNemar p"]].copy()
        s["diff A-B"] = s["diff A-B"].round(3)
        s["McNemar p"] = s["McNemar p"].round(3)
        s["A"] = s["A"].map(lambda v: textwrap.fill(str(v), 22))
        s["B"] = s["B"].map(lambda v: textwrap.fill(str(v), 22))
        ax = fig.add_axes([0.03, 0.25, 0.94, 0.6])
        ax.axis("off")
        t = ax.table(cellText=s.values.tolist()[:14], colLabels=list(s.columns),
                     loc="upper center", cellLoc="left")
        self._style_table(t, font_size=6.5, row_height=1.8, ncols=5)
        self._text(fig, ["The interval is a vessel-level bootstrap of the macro-F1 "
                         "difference. If it contains 0 there is no clear difference."],
                   y=0.2, size=9, color=MUTED)
        return fig

    def _limitations_page(self):
        fig = self._page("Limitations and notes")
        self._text(fig, LIMITATIONS, size=10.5, bullet=True, gap=0.016)
        return fig

    # ------------------------------------------------------------------
    # Compilation and export
    # ------------------------------------------------------------------
    def compile_report(self):
        self.pages = []
        best, comp = self.metrics["best"], self.metrics["comparison"]

        self.pages.append(self._cover_page(best))
        self.pages.append(self._summary_page(best, comp))

        holdout = self.metrics.get("holdout")
        if holdout is not None and len(holdout):
            self.pages.append(self._holdout_page(holdout))

        plots = self.generate_plots()
        self.pages.extend(fig for name, fig in plots if name == "comparison")

        for fname, heading, caption in FIGURE_PAGES:
            p = self.figures_dir / fname
            if p.exists():
                self.pages.append(self._image_page(p, heading, caption))

        self.pages.extend(fig for name, fig in plots if name == "curves")

        sig = self.metrics["significance"]
        if sig is not None and len(sig):
            self.pages.append(self._significance_page(sig))

        self.pages.append(self._limitations_page())
        self._add_footers()
        return self.pages

    def export_pdf(self, path=None):
        if not self.pages:
            self.compile_report()
        self.output_path.mkdir(parents=True, exist_ok=True)
        path = Path(path) if path else self.output_path / "model_performance_report.pdf"
        with PdfPages(path) as pdf:
            for fig in self.pages:
                pdf.savefig(fig)
        self.n_pages = len(self.pages)
        return path

    def save_results(self):
        """Machine-readable summary next to the PDF."""
        comp = self.metrics["comparison"]
        holdout = self.metrics.get("holdout")
        out = {
            "title": self.title,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "best_model": self.metrics["best"],
            "notices": self.notices,
            "n_pages": self.n_pages,
            "comparison": None if comp is None
            else comp.round(4).reset_index().to_dict(orient="records"),
            "holdout": None if holdout is None
            else holdout.round(4).to_dict(orient="records"),
        }
        self.output_path.mkdir(parents=True, exist_ok=True)
        p = self.output_path / "report_summary.json"
        p.write_text(json.dumps(out, indent=2))
        return p