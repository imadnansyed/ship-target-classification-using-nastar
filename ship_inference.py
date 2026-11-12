import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ship_preprocess import (SARPreprocessor, InputError, read_image, validate_image, load_input_stats)


class CustomCNN(nn.Module):
    def __init__(self, num_classes, in_chans=1, widths=(16, 32, 64, 128, 256), dropout=0.4):
        super().__init__()
        blocks, c_in = [], in_chans
        for i, w in enumerate(widths):
            blocks.append(nn.Sequential(
                nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
                nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU()))
            if i < len(widths) - 1:
                blocks.append(nn.MaxPool2d(2))
            c_in = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(x))


def build_network(arch, num_classes, dropout=0.0):
    if arch == "custom":
        return CustomCNN(num_classes, in_chans=1, dropout=dropout)
    if arch in ("resnet18", "efficientnet_b0"):
        return timm.create_model(arch, pretrained=False, num_classes=num_classes, in_chans=1)
    if arch == "swin_tiny":
        return timm.create_model("swin_tiny_patch4_window7_224", pretrained=False,
                                 num_classes=num_classes, in_chans=1)
    raise ValueError(f"Unknown architecture: {arch}")


class GradCAM:
    """Class-diagram GradCAM: targetLayer, gradients, activations.

    Weights each feature map of the target layer by the average gradient of the class score,
    sums them, keeps the positive part and upsamples to the input size.
    """

    def __init__(self, net, target_layer, device, channels_last=False):
        self.net, self.target_layer, self.device = net, target_layer, device
        self.channels_last = channels_last
        self.gradients = self.activations = None
        self._handle = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "gradients", g))

    def close(self):
        self._handle.remove()

    def compute_gradients(self, x, class_idx=None):
        """Forward pass, then backward pass of one class score. Returns (logits, class index)."""
        self.net.eval()
        self.gradients = self.activations = None
        x = x.to(self.device)
        with torch.enable_grad():
            self.net.zero_grad(set_to_none=True)
            logits = self.net(x)
            idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
            logits[0, idx].backward()
        if self.gradients is None:
            raise RuntimeError("no gradients reached the target layer")
        return logits.detach(), idx

    def generate_heatmap(self, x, class_idx=None):
        """x: tensor (1, 1, H, W). Returns (heatmap (H, W) in [0, 1], class index, logits)."""
        logits, idx = self.compute_gradients(x, class_idx)
        A, G = self.activations.detach(), self.gradients.detach()
        if A.ndim != 4:
            raise RuntimeError(f"Grad-CAM needs a 4-D feature map, got shape {tuple(A.shape)}")
        if self.channels_last:
            A, G = A.permute(0, 3, 1, 2), G.permute(0, 3, 1, 2)
        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tuple(x.shape[-2:]), mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), idx, logits

    @staticmethod
    def generate_heatmap_overlay(view, cam, alpha=0.45):
        """view: grayscale image in [0, 1] (H, W); cam: heatmap in [0, 1] (H, W). Returns an RGB uint8 image."""
        import matplotlib.pyplot as plt
        rgb = plt.get_cmap("jet")(cam)[..., :3]
        base = np.repeat(view[..., None], 3, axis=2)
        return (255 * ((1 - alpha) * base + alpha * rgb)).clip(0, 255).astype(np.uint8)

    @staticmethod
    def display_heatmap(view, cam, title=""):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(view, cmap="gray")
        ax[0].set_title("model input")
        ax[1].imshow(GradCAM.generate_heatmap_overlay(view, cam))
        ax[1].set_title("Grad-CAM")
        for a in ax:
            a.axis("off")
        fig.suptitle(title)
        plt.show()


def get_target_layer(net, arch):
    """The last convolutional stage of each supported network."""
    if arch == "custom":
        return net.features[-1]
    if arch == "resnet18":
        return net.layer4[-1]
    if arch == "efficientnet_b0":
        return net.conv_head
    if arch == "swin_tiny":
        return net.layers[-1]            # channels-last (B, 7, 7, 768) in recent timm versions
    raise ValueError(f"No Grad-CAM target layer defined for architecture '{arch}'")


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LoadedModel:
    """A trained network plus everything needed to use it: class order, calibration, threshold, preprocessing."""

    def __init__(self, run_dir, device=None):
        run_dir = Path(run_dir)
        self.card = json.loads((run_dir / "model_card.json").read_text())
        rc = self.card["run_config"]
        self.arch, self.class_names = rc["arch"], list(self.card["class_names"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = build_network(self.arch, len(self.class_names), rc.get("dropout", 0.0))
        state = torch.load(run_dir / self.card["weights_file"], map_location="cpu")
        try:
            self.net.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError("The saved weights do not match this timm version "
                               f"(trained with timm {self.card.get('timm')}, installed {timm.__version__}). "
                               f"Install timm=={self.card.get('timm')}.") from exc
        self.net.to(self.device).eval()
        self.temperature, self.threshold = float(self.card["temperature"]), float(self.card["threshold"])
        self.preprocessor = SARPreprocessor.from_card(self.card)
        self.name = self.card.get("run_name", run_dir.name)


class ClassificationPipeline:
    """Class-diagram ClassificationPipeline: trainedModel, preprocessor, confidenceThreshold."""

    def __init__(self, run_dir, stats_path=None, device=None):
        self.trained_model = LoadedModel(run_dir, device)
        self.preprocessor = self.trained_model.preprocessor
        self.confidence_threshold = self.trained_model.threshold
        self.stats = load_input_stats(stats_path) if stats_path else None
        self.label = self.trained_model.name

    def validate_input(self, source):
        """source: a file path or a 2-D array. Returns (512 x 512 array, warnings). Raises InputError."""
        if isinstance(source, (str, Path)):
            arr, info, notes = read_image(source)
        else:
            raw = np.asarray(source)
            arr, info, notes = raw.astype(np.float32), {"dtype": str(raw.dtype), "shape": raw.shape}, []
        arr, warns = validate_image(arr, info, self.stats)
        return arr, notes + warns

    def flag_low_confidence(self, confidence):
        return confidence < self.confidence_threshold

    def classify_image(self, source, explain=False, target_class=None):
        t0 = time.time()
        try:
            arr, warnings = self.validate_input(source)
        except InputError as exc:
            return {"ok": False, "error": str(exc), "warnings": []}
        m = self.trained_model
        x = torch.from_numpy(self.preprocessor.to_model_input(arr))
        with torch.no_grad():
            logits = m.net(x.to(m.device)).float().cpu().numpy()
        probs = softmax_np(logits / m.temperature)[0]
        pred = int(probs.argmax())
        unreliable = any(w.startswith(("This is an 8-bit", "Brightness")) for w in warnings)
        res = {"ok": True, "class_name": m.class_names[pred], "class_index": pred, "confidence": float(probs[pred]),
               "probs": {c: float(p) for c, p in zip(m.class_names, probs)},
                "unreliable_input": unreliable,
               "low_confidence": bool(self.flag_low_confidence(float(probs[pred])) or unreliable),
               "threshold": self.confidence_threshold, "warnings": warnings, "model": self.label,
               "preview": self.preprocessor.model_view_uint8(arr), "overlay": None}
        if explain:
            try:
                res["overlay"] = self._explain(x, arr, pred if target_class is None else target_class)
            except Exception as exc:                        # Grad-CAM is optional (FR-06.1)
                res["warnings"].append(f"Grad-CAM is not available for this model ({exc}).")
        res["seconds"] = time.time() - t0
        return res

    def _explain(self, x, arr, class_idx):
        m = self.trained_model
        cam_gen = GradCAM(m.net, get_target_layer(m.net, m.arch), m.device, channels_last=(m.arch == "swin_tiny"))
        try:
            cam, _, _ = cam_gen.generate_heatmap(x, class_idx)
        finally:
            cam_gen.close()
        return GradCAM.generate_heatmap_overlay(self.preprocessor.model_view(arr), cam)

    def display_result(self, res):
        if not res["ok"]:
            return f"### Cannot classify this file\n{res['error']}"
        lines = [f"### Predicted ship type: **{res['class_name']}**",
                 f"Confidence: **{res['confidence'] * 100:.1f}%** (model: {res['model']})"]
        if res.get("unreliable_input"):
            lines.append("\n**Unreliable input**: this does not look like a NovaSAR intensity patch. Do not rely on this result.")
        elif res["low_confidence"]:
            lines.append(f"\n**Low confidence** (below {res['threshold']:.2f}). Consider manual review.")
        lines += [f"\n- {w}" for w in res["warnings"]]
        return "\n".join(lines)

    def batch_classify(self, sources, explain=False):
        results, rows = [], []
        for s in sources:
            name = Path(s).name if isinstance(s, (str, Path)) else "array"
            r = self.classify_image(s, explain=explain)
            results.append(r)
            if r["ok"]:
                rows.append({"file": name, "predicted": r["class_name"], "confidence": round(r["confidence"], 3),
                             "low confidence": r["low_confidence"], "notes": "; ".join(r["warnings"])})
            else:
                rows.append({"file": name, "predicted": "", "confidence": np.nan,
                             "low confidence": False, "notes": r["error"]})
        return results, pd.DataFrame(rows)
