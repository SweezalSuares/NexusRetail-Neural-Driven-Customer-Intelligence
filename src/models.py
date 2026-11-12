"""Keras MLP (ANN) builder, trainer and evaluation helpers."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import keras
from keras import layers
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .features import inv_slog1p, slog1p


def set_seed(seed=SEED):
    keras.utils.set_random_seed(seed)


def build_mlp(input_dim, task, n_classes=1, hidden=(128, 64, 32), dropout=0.3,
              l2=1e-4, lr=1e-3):
    """Multilayer perceptron: [Dense -> BatchNorm -> ReLU -> Dropout] x N + head."""
    inp = keras.Input(shape=(input_dim,), name="features")
    x = inp
    for i, units in enumerate(hidden):
        x = layers.Dense(units, kernel_regularizer=keras.regularizers.l2(l2), name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.Dropout(dropout if i < len(hidden) - 1 else dropout / 2, name=f"drop_{i+1}")(x)
    if task == "binary":
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
        loss, metrics = "binary_crossentropy", [keras.metrics.AUC(name="auc")]
    elif task == "multiclass":
        out = layers.Dense(n_classes, activation="softmax", name="output")(x)
        loss, metrics = "sparse_categorical_crossentropy", ["accuracy"]
    else:
        out = layers.Dense(1, name="output")(x)
        loss, metrics = keras.losses.Huber(), [keras.metrics.MeanAbsoluteError(name="mae")]
    model = keras.Model(inp, out, name="mlp")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=metrics)
    return model


def _fit(model, Xtr, ytr, Xva, yva, class_weight=None, epochs=150, batch=256):
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)]
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch,
                  class_weight=class_weight, callbacks=cbs, verbose=0)
    return {k: [float(v) for v in vals] for k, vals in h.history.items()}


def _downsample(a, n=120):
    a = np.asarray(a)
    if len(a) <= n:
        return a.tolist()
    return a[np.linspace(0, len(a) - 1, n).astype(int)].tolist()


def train_and_evaluate(key, task, Xtr_df, ytr, Xte_df, yte, groups, to_matrix, cols,
                       class_names=None, hidden=(128, 64, 32), dropout=0.3):
    """Train one MLP, evaluate on the temporal hold-out and return (model, bundle, metrics, curves)."""
    set_seed()
    Xtr_raw, Xte_raw = to_matrix(Xtr_df, cols), to_matrix(Xte_df, cols)
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_s, Xte_s = scaler.transform(Xtr_raw).astype("float32"), scaler.transform(Xte_raw).astype("float32")

    ytr, yte = np.asarray(ytr), np.asarray(yte)
    ytr_m, ymean, ystd = ytr, 0.0, 1.0
    if task == "regression":
        z = slog1p(ytr)
        ymean, ystd = float(z.mean()), float(z.std() + 1e-9)
        ytr_m = ((z - ymean) / ystd).astype("float32")
    elif task == "multiclass":
        ytr_m = ytr.astype(int)
    else:
        ytr_m = ytr.astype("float32")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_idx, va_idx = next(gss.split(Xtr_s, groups=groups))
    cw = None  # no class re-weighting: keeps predicted probabilities calibrated

    n_classes = len(class_names) if class_names else 1
    model = build_mlp(Xtr_s.shape[1], task, n_classes, hidden, dropout)
    history = _fit(model, Xtr_s[tr_idx], ytr_m[tr_idx], Xtr_s[va_idx], ytr_m[va_idx], cw)

    bundle = dict(scaler=scaler, features=list(cols), task=task, class_names=class_names,
                  y_mean=ymean, y_std=ystd, threshold=0.5, hidden=list(hidden), key=key)
    metrics = dict(key=key, task=task, n_train=int(len(tr_idx)), n_val=int(len(va_idx)),
                   n_test=int(len(Xte_s)), n_features=int(Xtr_s.shape[1]),
                   epochs_run=len(history["loss"]), params=int(model.count_params()))
    curves = dict(history=history)

    if task == "binary":
        pv = model.predict(Xtr_s[va_idx], verbose=0).ravel()
        grid = np.linspace(0.05, 0.95, 91)
        f1s = [f1_score(ytr_m[va_idx], pv >= t, zero_division=0) for t in grid]
        thr = float(grid[int(np.argmax(f1s))])
        bundle["threshold"] = thr
        p = model.predict(Xte_s, verbose=0).ravel()
        pred = (p >= thr).astype(int)
        base = LogisticRegression(max_iter=500).fit(Xtr_s, ytr_m)
        pb = base.predict_proba(Xte_s)[:, 1]
        rs = np.random.RandomState(SEED)
        boots = []
        for _ in range(300):
            ix = rs.randint(0, len(yte), len(yte))
            if yte[ix].min() != yte[ix].max():
                boots.append(roc_auc_score(yte[ix], p[ix]))
        auc_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        fpr, tpr, _ = roc_curve(yte, p)
        fpb, tpb, _ = roc_curve(yte, pb)
        metrics.update(roc_auc=float(roc_auc_score(yte, p)), pr_auc=float(average_precision_score(yte, p)),
                       f1=float(f1_score(yte, pred)), precision=float(precision_score(yte, pred, zero_division=0)),
                       recall=float(recall_score(yte, pred)), accuracy=float(accuracy_score(yte, pred)),
                       threshold=thr, prevalence=float(np.mean(yte)), roc_auc_ci=auc_ci,
                       baseline_roc_auc=float(roc_auc_score(yte, pb)),
                       baseline_pr_auc=float(average_precision_score(yte, pb)),
                       baseline_name="Logistic Regression")
        curves.update(roc=dict(fpr=_downsample(fpr), tpr=_downsample(tpr)),
                      roc_baseline=dict(fpr=_downsample(fpb), tpr=_downsample(tpb)),
                      confusion=confusion_matrix(yte, pred).tolist())
    elif task == "multiclass":
        P = model.predict(Xte_s, verbose=0)
        pred = P.argmax(1)
        yte_i = yte.astype(int)
        top3 = np.mean([yte_i[i] in np.argsort(P[i])[::-1][:3] for i in range(len(yte_i))])
        base = LogisticRegression(max_iter=500).fit(Xtr_s, ytr_m)
        pb = base.predict(Xte_s)
        maj = np.bincount(ytr_m.astype(int)).argmax()
        metrics.update(accuracy=float(accuracy_score(yte_i, pred)), top3_accuracy=float(top3),
                       macro_f1=float(f1_score(yte_i, pred, average="macro")),
                       baseline_accuracy=float(accuracy_score(yte_i, pb)),
                       majority_accuracy=float(np.mean(yte_i == maj)),
                       random_accuracy=float(1 / len(class_names)), baseline_name="Logistic Regression")
        curves.update(confusion=confusion_matrix(yte_i, pred, labels=range(len(class_names))).tolist(),
                      class_names=class_names)
    else:
        zp = model.predict(Xte_s, verbose=0).ravel() * ystd + ymean
        pred = inv_slog1p(zp)
        zt = slog1p(yte)
        base = Ridge(alpha=1.0).fit(Xtr_s, ((slog1p(ytr) - ymean) / ystd))
        pb = inv_slog1p(base.predict(Xte_s) * ystd + ymean)
        mean_pred = np.full_like(yte, np.mean(ytr), dtype=float)
        rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
        metrics.update(mae=float(mean_absolute_error(yte, pred)), rmse=rmse(yte, pred),
                       r2=float(r2_score(yte, pred)), r2_log=float(r2_score(zt, zp)),
                       baseline_mae=float(mean_absolute_error(yte, pb)), baseline_r2_log=float(r2_score(zt, slog1p(pb))),
                       baseline_name="Ridge Regression",
                       mean_mae=float(mean_absolute_error(yte, mean_pred)),
                       target_mean=float(np.mean(yte)), target_median=float(np.median(yte)))
        rs = np.random.RandomState(SEED)
        idx = rs.choice(len(yte), size=min(500, len(yte)), replace=False)
        curves.update(scatter=dict(actual=yte[idx].tolist(), pred=pred[idx].tolist()))
    return model, bundle, metrics, curves


def predict(model, bundle, X_raw):
    """Run a trained MLP on raw feature matrix; returns probabilities / values."""
    Xs = bundle["scaler"].transform(X_raw).astype("float32")
    out = model.predict(Xs, verbose=0)
    if bundle["task"] == "regression":
        return inv_slog1p(out.ravel() * bundle["y_std"] + bundle["y_mean"])
    if bundle["task"] == "binary":
        return out.ravel()
    return out
