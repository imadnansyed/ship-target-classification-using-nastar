# Ship Target Classification: Analyst App

A tool that classifies a NovaSAR (S-band) ship patch as **cargo**, **tanker**, **fishing**, or **passenger**. You can choose between four trained models: an EfficientNet-B0 (default), a ResNet-18, a custom CNN baseline, and a Swin Transformer.

> **Note:** The output is meant to support an analyst's judgement. It is not a verified identification.

## Getting Started

Create a virtual environment, install the dependencies, and launch the app:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:7860> in your browser. You can either pick one of the built-in example patches or upload your own 512 × 512 TIFF patch.

### First Run

On the first launch, the app downloads the trained model weights and the figures folder from Google Drive, so an internet connection is required. Files that are already present are skipped, so later launches start immediately.

To download everything without starting the app, run:

```bash
python download_models.py
```

If a download fails, check that you are using the latest `gdown` (`pip install -U gdown`) and that the Google Drive files are shared as "Anyone with the link". Very long folder paths on Windows can also cause failures, so keep the project in a short path.

## Resources

- **Notebook:** `notebook/`
- **Executed notebook:** [Google Colab](https://colab.research.google.com/drive/1iIO3BvX1QfnYo9K0m75lVR9F0EP0HmH5?usp=sharing)
- **Dataset:** [Google Colab](https://colab.research.google.com/drive/1iIO3BvX1QfnYo9K0m75lVR9F0EP0HmH5?usp=sharing)

## Project Structure

| Path                   | Description                                                                                |
|------------------------|--------------------------------------------------------------------------------------------|
| `app.py`               | Entry point that downloads missing files and launches the app                              |
| `app_config.json`      | App configuration: class names, model list, thresholds and Google Drive IDs                |
| `download_models.py`   | Downloads missing model weights and figure files from Google Drive                         |
| `models/`              | Trained models (`weights.pt` and `model_card.json`), downloaded automatically on first run |
| `examples/`            | Hold-out test patches                                                                      |
| `results/`, `figures/` | Evaluation outputs (`figures/` is downloaded automatically on first run)                   |
| `ship_preprocess.py`   | Preprocessing stage of the pipeline                                                        |
| `ship_inference.py`    | Inference stage of the pipeline                                                            |
| `ship_report.py`       | Report generation stage of the pipeline                                                    |