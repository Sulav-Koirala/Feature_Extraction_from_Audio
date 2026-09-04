import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from . import config
from .features import FEATURE_NAMES


def load_dataset(csv_path=None) -> pd.DataFrame:
    csv_path = Path(csv_path or config.FEATURES_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m src.build_dataset` first."
        )
    return pd.read_csv(csv_path)


def _plot_confusion(cm, classes, title, out_path):
    import matplotlib

    matplotlib.use("Agg") 
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _make_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=config.SEED,
        n_jobs=-1,
    )


def train_one(name, X, y, classes, out_prefix):
    print(f"\n=== Training {name} model ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )

    clf = _make_classifier()
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    min_class = int(pd.Series(y_train).value_counts().min())
    cv_folds = max(2, min(5, min_class))
    cv = cross_val_score(clf, X_train, y_train, cv=cv_folds)

    print(f"  train accuracy : {train_acc:.3f}")
    print(f"  test  accuracy : {test_acc:.3f}")
    print(f"  {cv_folds}-fold CV     : {cv.mean():.3f} +/- {cv.std():.3f}")
    print(f"  train-test gap : {train_acc - test_acc:.3f}  (small => no overfit)")
    print("\n" + classification_report(y_test, y_pred, zero_division=0))

    present = [c for c in classes if c in set(np.unique(y))]
    cm = confusion_matrix(y_test, y_pred, labels=present)
    config.CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = config.CONFUSION_DIR / f"{out_prefix}_confusion.png"
    _plot_confusion(cm, present, f"{name} - confusion matrix (test set)", cm_path)
    print(f"  saved confusion matrix -> {cm_path}")

    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: t[1], reverse=True
    )
    top = ", ".join(f"{n}({v:.2f})" for n, v in importances[:5])
    print(f"  top features   : {top}")

    bundle = {"model": clf, "feature_names": FEATURE_NAMES, "classes": present}
    metrics = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "cv_folds": cv_folds,
        "cv_mean": float(cv.mean()),
        "cv_std": float(cv.std()),
        "classes": present,
        "confusion_matrix": cm.tolist(),
        "top_features": [[n, float(v)] for n, v in importances[:10]],
    }
    return bundle, metrics


def main(csv_path=None):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)

    all_metrics = {}
    if config.METRICS_JSON.exists():
        try:
            all_metrics = json.loads(config.METRICS_JSON.read_text())
        except Exception:
            all_metrics = {}

    speed_bundle, speed_metrics = train_one(
        "Speed", X, df["speed_label"].to_numpy(), config.SPEED_CLASSES, "speed"
    )
    joblib.dump(speed_bundle, config.SPEED_MODEL_PATH)
    print(f"  saved model -> {config.SPEED_MODEL_PATH}")

    loud_bundle, loud_metrics = train_one(
        "Loudness", X, df["loudness_label"].to_numpy(), config.LOUDNESS_CLASSES, "loudness"
    )
    joblib.dump(loud_bundle, config.LOUDNESS_MODEL_PATH)
    print(f"  saved model -> {config.LOUDNESS_MODEL_PATH}")

    all_metrics["speed_model"] = speed_metrics
    all_metrics["loudness_model"] = loud_metrics
    config.METRICS_JSON.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nSaved metrics -> {config.METRICS_JSON}")
    print("Done. Try:  python -m src.predict <audio_file>")


if __name__ == "__main__":
    main()
