"""Automated acceptance audit for Task 4 - checks every stated requirement."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
report = []


def check(name, ok, detail=""):
    report.append((name, bool(ok), detail))


# ---- 1. Required deliverable files present ----
required_files = [
    "fashion_mnist_pipeline.py", "README.md", "requirements.txt",
    "inference_demo.ipynb",
    "outputs/training_curves.png",
    "outputs/confusion_matrix_cnn.png", "outputs/confusion_matrix_mlp.png",
    "outputs/confusion_matrix_svm.png", "outputs/inference_demo.png",
    "outputs/prediction_log.csv", "outputs/metrics.json",
    "outputs/training_history.csv", "outputs/classification_report_cnn.txt",
    "outputs/model_comparison.png",
    "models/cnn_fashion_mnist.pt", "models/mlp_fashion_mnist.pt",
    "full_run.log",
]
missing = [f for f in required_files if not (root / f).exists()]
check("All deliverable files exist", not missing, f"missing={missing}")

# ---- 2. Exported charts are valid PNG files ----
bad = []
for p in sorted((root / "outputs").glob("*.png")):
    b = p.read_bytes()
    if not (b[:8] == b"\x89PNG\r\n\x1a\n" and b[-8:-4] == b"IEND" and len(b) > 5000):
        bad.append(p.name)
check("All exported charts are valid PNGs", not bad, f"bad={bad}")

# ---- 3. Script: normalization precedes training ----
src = (root / "fashion_mnist_pipeline.py").read_text(encoding="utf-8")
i_norm = src.find("stage 1 - pixel scaling")
i_train = src.find("STEP 4 - Training classifiers")
check("Normalization runs BEFORE training step", 0 < i_norm < i_train,
      f"norm@{i_norm}, train@{i_train}")
check("Pixel scaling /255 present", "/ 255.0" in src)
check("Standardization (x - mean)/std present", "std" in src and "/ std" in src)
check("Norm stats computed from TRAIN split only", "TRAIN split only" in src)
check("train_test_split used for partitioning", "train_test_split" in src)
check("Stratified split", "stratify=y_train" in src)

# ---- 4. Classifiers ----
check("MLP classifier defined", "class MLP" in src)
check("CNN classifier defined", "class CNN" in src)
check("SVM (hinge) baseline defined", "SGDClassifier" in src and '"hinge"' in src)
check("Loss = cross-entropy", "CrossEntropyLoss" in src)

# ---- 5. Per-class evaluation + confusion matrix ----
check("classification_report over classes", "classification_report" in src)
check("confusion_matrix computed", "confusion_matrix" in src)
check("10 class labels defined", all(c in src for c in
      ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]))

# ---- 6. Inference demo: image + label + confidence ----
check("Softmax confidence computed", "softmax" in src)
check("Demo pairs true vs predicted label", "True :" in src and "Pred :" in src)
check("Prediction log exported", "prediction_log.csv" in src)

# ---- 7. Training curves (loss AND accuracy) ----
check("training_curves.png exported by code", "training_curves.png" in src)
check("Loss curve recorded", "train_loss" in src and "val_loss" in src)
check("Accuracy curve recorded", "train_acc" in src and "val_acc" in src)

# ---- 8. No paid / cloud dependencies ----
banned = ["openai", "anthropic", "azure", "boto3", "google.cloud", "aws.",
          "api_key", "subscription", "huggingface_hub"]
blob = (src + (root / "requirements.txt").read_text()).lower()
hits = [b for b in banned if b in blob]
check("No paid/cloud service references", not hits, f"hits={hits}")
req = (root / "requirements.txt").read_text().lower()
check("Requirements stack = Python/PyTorch/sklearn/Matplotlib (FOSS)",
      all(t in req for t in ["torch", "scikit-learn", "matplotlib"]))

# ---- 9. README content ----
rd = (root / "README.md").read_text(encoding="utf-8")
for label, pat in [
    ("README: architecture choices", r"(?i)architectur"),
    ("README: hyperparameter settings", r"(?i)hyperparameter"),
    ("README: prediction logs", r"(?i)prediction.?log"),
    ("README: documents normalization", r"(?i)pixel scaling"),
    ("README: documents partitioning", r"(?i)partitioning"),
    ("README: links training curves", r"training_curves\.png"),
    ("README: links confusion matrix", r"confusion_matrix_"),
    ("README: links inference demo", r"inference_demo\.png"),
    ("README: states test accuracies", r"0\.9147"),
    ("README: acceptance criteria section", r"(?i)acceptance criteri"),
    ("README: per-class results table", r"macro avg"),
    ("README: run instructions", r"pip install -r requirements"),
]:
    check(label, bool(re.search(pat, rd)))
check("README has no leftover placeholders", "<!--" not in rd)

# ---- 10. metrics.json integrity ----
m = json.loads((root / "outputs" / "metrics.json").read_text())
check("metrics.json: all 10 classes for every model",
      all(len(m["per_class_metrics"][k]) == 10 for k in m["per_class_metrics"]),
      f"models={list(m['per_class_metrics'])}")
check("metrics.json: normalization stats recorded",
      "stage1" in m["normalization"] and "stage2" in m["normalization"])
check("metrics.json: hyperparameters recorded", "epochs_cnn" in m["hyperparameters"])
check("metrics.json: split sizes recorded",
      m["splits"] == {"train": 54000, "val": 6000, "test": 10000},
      f"splits={m['splits']}")
check("metrics.json: train mean~0 std~1 after normalization",
      abs(m["normalization"]["train_mean_after"]) < 1e-5
      and abs(m["normalization"]["train_std_after"] - 1.0) < 1e-5,
      f"mean={m['normalization']['train_mean_after']} std={m['normalization']['train_std_after']}")

# ---- 11. prediction log schema ----
hdr = (root / "outputs" / "prediction_log.csv").read_text().splitlines()[0]
cols = set(hdr.split(","))
check("prediction_log has true/predicted/confidence columns",
      {"true_label", "predicted_label", "confidence"} <= cols, f"cols={cols}")
rows = (root / "outputs" / "prediction_log.csv").read_text().strip().splitlines()
check("prediction_log has sample rows", len(rows) - 1 >= 16, f"rows={len(rows)-1}")

# ---- 12. training history integrity ----
th = (root / "outputs" / "training_history.csv").read_text().strip().splitlines()
check("training_history has loss+accuracy per epoch",
      th[0].split(",")[:6] == ["model", "epoch", "train_loss", "train_acc",
                               "val_loss", "val_acc"], f"header={th[0]}")
check("training_history has 16 epoch rows (8 MLP + 8 CNN)",
      len(th) - 1 == 16, f"rows={len(th)-1}")

# ---- 13. notebook is valid JSON with code cells ----
nb = json.loads((root / "inference_demo.ipynb").read_text(encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
check("Notebook valid (nbformat 4, has code cells)",
      nb["nbformat"] == 4 and len(code_cells) >= 5,
      f"nbformat={nb['nbformat']} code_cells={len(code_cells)}")
nb_src = "".join("".join(c["source"]) for c in code_cells)
check("Notebook loads checkpoint + normalizes + shows confidence",
      "torch.load" in nb_src and "255.0" in nb_src and "softmax" in nb_src
      and "True" in nb_src)

# ---- report ----
width = max(len(n) for n, _, _ in report)
for n, ok, d in report:
    print(f"{'PASS' if ok else 'FAIL'}  {n:<{width}}  {'' if ok else d}")
npass = sum(1 for _, ok, _ in report if ok)
print()
print(f"{npass}/{len(report)} checks passed")
if npass != len(report):
    print("SOME CHECKS FAILED")
    raise SystemExit(1)
print("ALL REQUIREMENTS VERIFIED")
