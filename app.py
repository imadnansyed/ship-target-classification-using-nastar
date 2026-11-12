import os

os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ship_inference import ClassificationPipeline

CONFIG = json.loads((HERE / "app_config.json").read_text())
CLASS_NAMES = CONFIG["class_names"]
MODELS = {m["label"]: HERE / m["run_dir"] for m in CONFIG["models"]}
DEFAULT_MODEL = CONFIG.get("default_model") or next(iter(MODELS))
MAIN_SCHEME = CONFIG.get("main_scheme", "cv_mmsi")
STATS = HERE / "results" / "input_stats.json"
EX_CSV = HERE / "examples" / "examples.csv"
EXAMPLES = pd.read_csv(EX_CSV) if EX_CSV.exists() else pd.DataFrame(columns=["file", "true_label", "shoreline"])
_PIPES = {}

DISCLAIMER = (
    "Classifies a **NovaSAR (S-band) ship patch** into one of four types: "
    + ", ".join(CLASS_NAMES) + ". Any other vessel or image will still be given one of these four labels. "
    "The result supports an analyst's judgement; it is not a verified identification."
)


def get_pipeline(label):
    if label not in _PIPES:
        _PIPES[label] = ClassificationPipeline(MODELS[label], STATS if STATS.exists() else None)
    return _PIPES[label]


def _path(f):
    return f if isinstance(f, str) else getattr(f, "name", None)


def _truth(path):
    row = EXAMPLES[EXAMPLES["file"] == Path(path).name]
    return None if row.empty else str(row["true_label"].iloc[0])


def _fig(name):
    p = HERE / "figures" / name
    return str(p) if p.exists() else None


# ---------------- tab 1: classify one image (UC-07, UC-08) ----------------
def classify_single(file, model_label, show_cam):
    path = _path(file)
    if not path:
        return None, "Please upload a SAR patch or pick an example.", None, None
    try:
        pipe = get_pipeline(model_label)
    except Exception as exc:                                    # model files missing or incompatible
        return None, f"### The model could not be loaded\n{exc}", None, None
    res = pipe.classify_image(path, explain=bool(show_cam))
    msg = pipe.display_result(res)
    if not res["ok"]:
        return None, msg, None, None
    truth = _truth(path)
    if truth:
        msg += f"\n\nKnown ship type of this example (from AIS): **{truth}**"
    return res["preview"], msg, res["probs"], res["overlay"]

# ---------------- tab 1 extra: compare all models on one image ----------------
def classify_all_models(file):
    path = _path(file)
    if not path:
        return pd.DataFrame({"info": ["Upload an image first."]})
    rows = []
    for label in MODELS:
        try:
            pipe = get_pipeline(label)
            r = pipe.classify_image(path)
        except Exception as exc:
            rows.append({"Model": label, "Predicted class": "ERROR", "Confidence": "-", "Note": str(exc)[:60]})
            continue
        if r["ok"]:
            rows.append({"Model": label, "Predicted class": r["class_name"],
                        "Confidence": f"{r['confidence'] * 100:.1f}%",
                        "Note": "Low confidence" if r["low_confidence"] else ""})
        else:
            rows.append({"Model": label, "Predicted class": "ERROR", "Confidence": "-", "Note": r["error"][:60]})
    return pd.DataFrame(rows)

# ---------------- tab 2: evaluation (UC-05) ----------------
def evaluation_table():
    p = HERE / "results" / "cv_fold_metrics.csv"
    if not p.exists():
        return pd.DataFrame({"info": ["results/cv_fold_metrics.csv was not found in this app folder."]})
    ft = pd.read_csv(p)
    ft = ft[ft["scheme"] == MAIN_SCHEME]
    cols = {"acc": "Accuracy", "macro_f1": "Macro-F1", "bal_acc": "Balanced acc.",
            "macro_precision": "Macro precision", "macro_recall": "Macro recall"}
    g = ft.groupby("model")[list(cols)]
    mean, std = g.mean(), g.std()
    out = pd.DataFrame({v: [f"{mean.loc[m, k]:.3f} ± {std.loc[m, k]:.3f}" for m in mean.index]
                        for k, v in cols.items()}, index=mean.index)
    return out.sort_values("Macro-F1", ascending=False).reset_index().rename(columns={"model": "Model"})


# ---------------- tab 3: comparison and report (UC-06, UC-09) ----------------
def comparison_table():
    p = HERE / "results" / "final_comparison.csv"
    if not p.exists():
        return pd.DataFrame({"info": ["No comparison results found. Only a single model can be reported."]})
    c = pd.read_csv(p, index_col=0)
    out = pd.DataFrame({"Accuracy": [f"{r.acc_mean:.3f} ± {r.acc_std:.3f}" for r in c.itertuples()],
                        "Macro-F1": [f"{r.macro_f1_mean:.3f} ± {r.macro_f1_std:.3f}" for r in c.itertuples()]},
                       index=c.index)
    return out.sort_values("Macro-F1", ascending=False).reset_index().rename(columns={"model": "Model"})


def make_report():
    try:
        from ship_report import ReportGenerator
    except ImportError:
        return None, "The report module (ship_report.py) is not in this app folder."
    try:
        rg = ReportGenerator(HERE / "results", HERE / "figures", HERE / "models",
                             Path(tempfile.mkdtemp()))
        pdf = rg.export_pdf()
    except Exception as exc:
        return None, f"The report could not be generated: {exc}"
    note = " ".join(rg.notices)
    return str(pdf), f"Report generated ({rg.n_pages} pages). {note}".strip()


# ---------------- layout ----------------
with gr.Blocks(title="Ship Target Classification (SAR)") as demo:
    gr.Markdown("# Ship target classification from SAR images\n" + DISCLAIMER)
    with gr.Tabs():
        with gr.Tab("Classify ship image"):
            with gr.Row():
                with gr.Column():
                    file_in = gr.File(label="SAR patch (TIFF, PNG or JPG)",
                                      file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"])
                    model_dd = gr.Dropdown(choices=list(MODELS), value=DEFAULT_MODEL, label="Model")
                    cam_cb = gr.Checkbox(value=True, label="Show Grad-CAM (where the model looked)")
                    run_btn = gr.Button("Classify", variant="primary")
                    ex_paths = [str(HERE / "examples" / f) for f in EXAMPLES["file"]
                                if (HERE / "examples" / f).exists()]
                    if ex_paths:
                        gr.Examples(examples=[[p] for p in ex_paths], inputs=[file_in],
                                    label="Example patches (hold-out test images)")
                with gr.Column():
                    result_md = gr.Markdown()
                    label_out = gr.Label(label="Confidence per ship type", num_top_classes=len(CLASS_NAMES))
                    with gr.Row():
                        preview = gr.Image(label="Model input view", interactive=False)
                        cam_img = gr.Image(label="Grad-CAM overlay", interactive=False)
            run_btn.click(classify_single, [file_in, model_dd, cam_cb], [preview, result_md, label_out, cam_img])
            compare_btn = gr.Button("Compare across all models")
            compare_table = gr.Dataframe(label="Prediction by model", interactive=False)
            compare_btn.click(classify_all_models, [file_in], [compare_table])

        with gr.Tab("Model evaluation"):
            gr.Markdown(f"Cross-validated test results ({MAIN_SCHEME}, mean ± standard deviation over folds). "
                        "Every image was predicted by a model that never saw its vessel in training.")
            gr.Dataframe(value=evaluation_table(), interactive=False)
            gr.Image(value=_fig("fig_cv_confusion_matrices.png"), label="Confusion matrices", interactive=False)
            with gr.Row():
                gr.Image(value=_fig("fig_cv_shoreline.png"), label="Inshore vs offshore", interactive=False)
                gr.Image(value=_fig("fig_cv_reliability.png"), label="Calibration", interactive=False)

        with gr.Tab("Comparison and report"):
            gr.Dataframe(value=comparison_table(), interactive=False)
            gr.Image(value=_fig("fig_model_comparison.png"), label="Model comparison", interactive=False)
            gr.Markdown("The reference rows show a model that always predicts the most common class, and one that "
                        "sees only the ship length and width from AIS (no image).")
            report_btn = gr.Button("Generate performance report (PDF)", variant="primary")
            report_status = gr.Markdown()
            report_file = gr.File(label="Report")
            report_btn.click(make_report, None, [report_file, report_status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)