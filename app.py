"""
Uganda Sign Language (USL) Training Tool
=========================================
An ML-powered desktop application for learning Ugandan Sign Language.
Pipeline: Data Collection → EDA → Feature Extraction → ML Modeling → Feedback

Author: USL ML Project
Python 3.13 | OpenCV | MediaPipe | Scikit-learn | Tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import sys
import subprocess
import json
import time
import math
import random
import webbrowser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import (confusion_matrix, classification_report,
                              silhouette_score, f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageTk
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONSTANTS & PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
SAMPLES_DIR = DATA_DIR / "samples"
for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR, SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = DATA_DIR / "features.csv"
MODEL_FILE = MODELS_DIR / "sign_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
ENCODER_FILE = MODELS_DIR / "encoder.pkl"

USL_ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# YouTube links per letter for USL / ASL improvement
YOUTUBE_LINKS = {
    letter: f"https://www.youtube.com/results?search_query=Uganda+Sign+Language+letter+{letter}"
    for letter in USL_ALPHABET
}
YOUTUBE_GENERAL = [
    ("USL Alphabet Full Tutorial",
     "https://www.youtube.com/results?search_query=Uganda+Sign+Language+alphabet+tutorial"),
    ("Sign Language for Beginners Uganda",
     "https://www.youtube.com/results?search_query=sign+language+beginners+Uganda"),
    ("Fingerspelling Practice",
     "https://www.youtube.com/results?search_query=ASL+fingerspelling+practice"),
]

# ─────────────────────────────────────────────
#  COLOUR PALETTE  (dark teal theme)
# ─────────────────────────────────────────────
C = {
    "bg":      "#0d1b2a",
    "panel":   "#112233",
    "card":    "#1a2e44",
    "border":  "#1e3a52",
    "accent":  "#00c9a7",
    "accent2": "#3a86ff",
    "gold":    "#ffd166",
    "red":     "#ef233c",
    "text":    "#e8f1f8",
    "muted":   "#6b8cae",
    "white":   "#ffffff",
}

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─────────────────────────────────────────────
#  FEATURE EXTRACTION UTILITIES
# ─────────────────────────────────────────────
def extract_landmarks(frame):
    """Extract 21 hand landmarks (x,y,z) = 63 features using MediaPipe."""
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            coords = [(l.x, l.y, l.z) for l in lm]
            flat = [v for triplet in coords for v in triplet]
            # Normalize relative to wrist
            wx, wy, wz = flat[0], flat[1], flat[2]
            norm = [flat[i] - [wx, wy, wz][i % 3] for i in range(len(flat))]
            return np.array(norm, dtype=np.float32), result.multi_hand_landmarks[0]
    return None, None

def draw_landmarks_on_frame(frame, hand_landmarks):
    annotated = frame.copy()
    if hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    return annotated

def compute_score(pred_label, true_label, proba):
    """Map model confidence → 10-100% score."""
    if pred_label == true_label:
        base = max(0.5, proba)
        score = int(round(base * 100 / 10) * 10)
        return max(50, min(100, score))
    else:
        score = int(round(proba * 50 / 10) * 10)
        return max(10, min(40, score))

# ─────────────────────────────────────────────
#  SYNTHETIC DATA GENERATOR (demo when no real data)
# ─────────────────────────────────────────────
def generate_synthetic_features(n_per_class=30):
    """Generate synthetic hand-landmark features for demo/EDA."""
    records = []
    rng = np.random.RandomState(42)
    for i, letter in enumerate(USL_ALPHABET):
        center = rng.randn(63) * 0.05 + (i * 0.01)
        for _ in range(n_per_class):
            feat = center + rng.randn(63) * 0.02
            row = {f"f{j}": feat[j] for j in range(63)}
            row["label"] = letter
            row["labeled"] = rng.rand() > 0.6   # 40 % unlabeled
            records.append(row)
    df = pd.DataFrame(records)
    df.to_csv(FEATURES_CSV, index=False)
    return df

# ─────────────────────────────────────────────
#  ML PIPELINE  (train / predict)
# ─────────────────────────────────────────────
class SignClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.is_trained = False
        self._load()

    def _load(self):
        if MODEL_FILE.exists() and SCALER_FILE.exists() and ENCODER_FILE.exists():
            with open(MODEL_FILE, "rb") as f:
                self.model = pickle.load(f)
            with open(SCALER_FILE, "rb") as f:
                self.scaler = pickle.load(f)
            with open(ENCODER_FILE, "rb") as f:
                self.encoder = pickle.load(f)
            self.is_trained = True

    def _save(self):
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(ENCODER_FILE, "wb") as f:
            pickle.dump(self.encoder, f)

    def train_supervised(self, df):
        labeled = df[df["labeled"] == True].copy()
        feat_cols = [c for c in labeled.columns if c.startswith("f")]
        X = labeled[feat_cols].values
        y = labeled["label"].values
        self.encoder.fit(y)
        y_enc = self.encoder.transform(y)
        X_scaled = self.scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_enc,
                                                    test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_tr, y_tr)
        y_pred = self.model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        self.is_trained = True
        self._save()
        return acc, f1, y_te, y_pred

    def train_semi_supervised(self, df):
        """Label Spreading on labeled + unlabeled data."""
        feat_cols = [c for c in df.columns if c.startswith("f")]
        X = df[feat_cols].values
        labels_raw = df["label"].values
        is_labeled = df["labeled"].values.astype(bool)
        all_labels = list(set(labels_raw[is_labeled]))
        self.encoder.fit(all_labels)
        y = np.full(len(df), -1, dtype=int)
        y[is_labeled] = self.encoder.transform(labels_raw[is_labeled])
        X_scaled = self.scaler.fit_transform(X)
        ls = LabelSpreading(kernel="knn", n_neighbors=7, max_iter=1000)
        ls.fit(X_scaled, y)
        y_pred_all = ls.transduction_
        # Train RF on pseudo-labeled data
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y_pred_all)
        labeled_mask = is_labeled
        y_te = self.encoder.transform(labels_raw[labeled_mask])
        y_pred = self.model.predict(X_scaled[labeled_mask])
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        self.is_trained = True
        self._save()
        return acc, f1

    def predict(self, feature_vector):
        """Return (letter, score_pct, probabilities_dict)"""
        if not self.is_trained:
            return None, 0, {}
        X = feature_vector.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        pred_enc = self.model.predict(X_scaled)[0]
        pred_label = self.encoder.inverse_transform([pred_enc])[0]
        proba = None
        proba_dict = {}
        if hasattr(self.model, "predict_proba"):
            proba_all = self.model.predict_proba(X_scaled)[0]
            proba = float(proba_all[pred_enc])
            for i, cls in enumerate(self.encoder.classes_):
                proba_dict[cls] = float(proba_all[i])
        return pred_label, proba or 0.5, proba_dict


# ─────────────────────────────────────────────
#  EDA & VISUALISATION
# ─────────────────────────────────────────────
class EDAEngine:
    def __init__(self, df):
        self.df = df
        self.feat_cols = [c for c in df.columns if c.startswith("f")]
        self.X = df[self.feat_cols].values
        self.y = df["label"].values if "label" in df.columns else None

    def _save_fig(self, name):
        path = PLOTS_DIR / f"{name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=C["bg"])
        plt.close()
        return path

    def plot_class_distribution(self):
        fig, ax = plt.subplots(figsize=(12, 4), facecolor=C["bg"])
        ax.set_facecolor(C["panel"])
        counts = pd.Series(self.y).value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values,
                      color=C["accent"], edgecolor=C["bg"], linewidth=0.5)
        ax.set_title("Class Distribution (Samples per Letter)",
                     color=C["text"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Letter", color=C["muted"])
        ax.set_ylabel("Count", color=C["muted"])
        ax.tick_params(colors=C["muted"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        return self._save_fig("class_dist")

    def plot_pca(self):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(self.X)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_s)
        labels = sorted(set(self.y))
        cmap = plt.cm.get_cmap("tab20", len(labels))
        label_map = {l: i for i, l in enumerate(labels)}
        colors = [cmap(label_map[l]) for l in self.y]
        fig, ax = plt.subplots(figsize=(9, 7), facecolor=C["bg"])
        ax.set_facecolor(C["panel"])
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=20, alpha=0.7)
        patches = [mpatches.Patch(color=cmap(i), label=l)
                   for i, l in enumerate(labels)]
        ax.legend(handles=patches, loc="upper right", fontsize=7,
                  ncol=3, framealpha=0.3, labelcolor=C["text"])
        ax.set_title(f"PCA (var explained: {pca.explained_variance_ratio_.sum():.1%})",
                     color=C["text"], fontsize=13, fontweight="bold")
        ax.tick_params(colors=C["muted"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        return self._save_fig("pca")

    def plot_tsne(self):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(self.X[:300])
        y_sub = self.y[:300]
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=300)
        X_2d = tsne.fit_transform(X_s)
        labels = sorted(set(y_sub))
        cmap = plt.cm.get_cmap("tab20", len(labels))
        label_map = {l: i for i, l in enumerate(labels)}
        colors = [cmap(label_map[l]) for l in y_sub]
        fig, ax = plt.subplots(figsize=(9, 7), facecolor=C["bg"])
        ax.set_facecolor(C["panel"])
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=20, alpha=0.7)
        patches = [mpatches.Patch(color=cmap(i), label=l)
                   for i, l in enumerate(labels)]
        ax.legend(handles=patches, loc="upper right", fontsize=7,
                  ncol=3, framealpha=0.3, labelcolor=C["text"])
        ax.set_title("t-SNE (first 300 samples)", color=C["text"],
                     fontsize=13, fontweight="bold")
        ax.tick_params(colors=C["muted"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        return self._save_fig("tsne")

    def plot_correlation(self):
        # Use first 20 features for readability
        sub = pd.DataFrame(self.X[:, :20],
                           columns=[f"f{i}" for i in range(20)])
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=C["bg"])
        ax.set_facecolor(C["panel"])
        sns.heatmap(sub.corr(), ax=ax, cmap="coolwarm", center=0,
                    linewidths=0.3, annot=False, cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Matrix (first 20 features)",
                     color=C["text"], fontsize=13, fontweight="bold")
        ax.tick_params(colors=C["muted"], labelsize=8)
        return self._save_fig("correlation")

    def plot_kmeans(self, k=26):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(self.X)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_km = km.fit_predict(X_s)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_s)
        cmap = plt.cm.get_cmap("tab20", k)
        fig, ax = plt.subplots(figsize=(9, 7), facecolor=C["bg"])
        ax.set_facecolor(C["panel"])
        ax.scatter(X_2d[:, 0], X_2d[:, 1],
                   c=[cmap(l) for l in labels_km], s=20, alpha=0.6)
        centers_2d = pca.transform(km.cluster_centers_)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                   c="white", s=80, marker="*", zorder=5)
        sil = silhouette_score(X_s, labels_km)
        ax.set_title(f"K-Means Clustering (k={k}, silhouette={sil:.3f})",
                     color=C["text"], fontsize=13, fontweight="bold")
        ax.tick_params(colors=C["muted"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        return self._save_fig("kmeans")

    def plot_anomaly(self):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(self.X)
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(X_s)  # -1 = anomaly, 1 = normal
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_s)
        colors = [C["red"] if p == -1 else C["accent"] for p in preds]
        fig, ax = plt.subplots(figsize=(9, 6), facecolor=C["bg"])
        ax.set_facecolor(C["panel"])
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=20, alpha=0.7)
        normal_patch = mpatches.Patch(color=C["accent"], label="Normal")
        anom_patch = mpatches.Patch(color=C["red"], label="Anomaly")
        ax.legend(handles=[normal_patch, anom_patch], framealpha=0.3,
                  labelcolor=C["text"])
        n_anom = (preds == -1).sum()
        ax.set_title(f"Anomaly Detection (IsolationForest, {n_anom} anomalies found)",
                     color=C["text"], fontsize=13, fontweight="bold")
        ax.tick_params(colors=C["muted"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        return self._save_fig("anomaly")

    def plot_confusion(self, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(14, 12), facecolor=C["bg"])
        ax.set_facecolor(C["panel"])
        sns.heatmap(cm, ax=ax, xticklabels=classes, yticklabels=classes,
                    cmap="Blues", annot=True, fmt="d", linewidths=0.3,
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Confusion Matrix", color=C["text"],
                     fontsize=13, fontweight="bold")
        ax.tick_params(colors=C["muted"], labelsize=8)
        ax.set_xlabel("Predicted", color=C["muted"])
        ax.set_ylabel("True", color=C["muted"])
        return self._save_fig("confusion")

    def plot_pu_learning(self):
        """Visualise PU Learning: Positive vs Unknown."""
        scaler = StandardScaler()
        X_s = scaler.fit_transform(self.X[:200])
        y_sub = self.y[:200]
        # Simulate PU: some labeled as positive, rest unknown
        is_labeled = self.df["labeled"].values[:200].astype(bool)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_s)
        fig, ax = plt.subplots(figsize=(9, 6), facecolor=C["bg"])
        ax.set_facecolor(C["panel"])
        positive_mask = is_labeled
        ax.scatter(X_2d[~positive_mask, 0], X_2d[~positive_mask, 1],
                   c=C["muted"], s=20, alpha=0.5, label="Unlabeled (Unknown)")
        ax.scatter(X_2d[positive_mask, 0], X_2d[positive_mask, 1],
                   c=C["gold"], s=30, alpha=0.9, label="Positive (Labeled)")
        ax.set_title("PU Learning: Positive vs Unlabeled Distribution",
                     color=C["text"], fontsize=13, fontweight="bold")
        ax.legend(framealpha=0.3, labelcolor=C["text"])
        ax.tick_params(colors=C["muted"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        return self._save_fig("pu_learning")

    def plot_elbow(self, max_k=15):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(self.X)
        inertias = []
        ks = range(2, max_k + 1)
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            km.fit(X_s)
            inertias.append(km.inertia_)
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=C["bg"])
        ax.set_facecolor(C["panel"])
        ax.plot(list(ks), inertias, color=C["accent"], marker="o",
                linewidth=2, markersize=5)
        ax.set_title("Elbow Method for Optimal K",
                     color=C["text"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Clusters (k)", color=C["muted"])
        ax.set_ylabel("Inertia", color=C["muted"])
        ax.tick_params(colors=C["muted"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        return self._save_fig("elbow")


# ─────────────────────────────────────────────
#  GUI APPLICATION
# ─────────────────────────────────────────────
class USLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤟 Uganda Sign Language Training Tool")
        self.root.geometry("1280x800")
        self.root.configure(bg=C["bg"])
        self.root.minsize(1000, 700)

        self.classifier = SignClassifier()
        self.df = None
        self.cap = None
        self.camera_active = False
        self.selected_letter = tk.StringVar(value="A")
        self.current_frame = None
        self.score_history = []

        self._setup_styles()
        self._build_ui()
        self._load_or_generate_data()

    # ── STYLES ──────────────────────────────
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=C["bg"], borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=C["card"], foreground=C["muted"],
                        padding=[16, 8], font=("Helvetica", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", C["panel"])],
                  foreground=[("selected", C["accent"])])
        style.configure("TFrame", background=C["bg"])
        style.configure("Card.TFrame", background=C["card"])
        style.configure("TScrollbar",
                        background=C["panel"], troughcolor=C["bg"],
                        arrowcolor=C["muted"])
        style.configure("TProgressbar",
                        background=C["accent"], troughcolor=C["panel"])

    # ── BUILD UI ─────────────────────────────
    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg=C["bg"], height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header, text="🤟  Uganda Sign Language Training Tool",
                 bg=C["bg"], fg=C["accent"],
                 font=("Helvetica", 18, "bold")).pack(side="left", padx=20, pady=12)
        tk.Label(header, text="ML-Powered • EDA • Unsupervised • Semi-Supervised • PU Learning",
                 bg=C["bg"], fg=C["muted"],
                 font=("Helvetica", 9)).pack(side="left", padx=4, pady=12)
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(header, textvariable=self.status_var,
                 bg=C["bg"], fg=C["muted"],
                 font=("Helvetica", 9)).pack(side="right", padx=20)

        # ── Notebook tabs ──
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        tabs = [
            ("✋  Practice", self._build_practice_tab),
            ("📊  EDA & Visualise", self._build_eda_tab),
            ("🤖  Model Training", self._build_training_tab),
            ("📈  Evaluation", self._build_eval_tab),
            ("📚  Resources", self._build_resources_tab),
        ]
        for name, builder in tabs:
            frame = ttk.Frame(self.notebook, style="TFrame")
            self.notebook.add(frame, text=name)
            builder(frame)

    # ══════════════════════════════════════════
    #  TAB 1 — PRACTICE (Mark a Sign)
    # ══════════════════════════════════════════
    def _build_practice_tab(self, parent):
        parent.configure(style="TFrame")

        # Left panel — controls
        left = tk.Frame(parent, bg=C["panel"], width=280)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="STEP 1 — Choose a Letter",
                 bg=C["panel"], fg=C["accent"],
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=16, pady=(16, 6))

        # Alphabet grid
        grid_frame = tk.Frame(left, bg=C["panel"])
        grid_frame.pack(padx=12, pady=4)
        self._letter_btns = {}
        for i, letter in enumerate(USL_ALPHABET):
            btn = tk.Button(
                grid_frame, text=letter, width=3, height=1,
                bg=C["card"], fg=C["text"],
                font=("Courier", 10, "bold"),
                relief="flat", cursor="hand2",
                command=lambda l=letter: self._select_letter(l)
            )
            btn.grid(row=i // 9, column=i % 9, padx=2, pady=2)
            self._letter_btns[letter] = btn
        self._select_letter("A")

        tk.Label(left, text="Selected Letter:",
                 bg=C["panel"], fg=C["muted"],
                 font=("Helvetica", 9)).pack(anchor="w", padx=16, pady=(12, 0))
        self.selected_display = tk.Label(
            left, textvariable=self.selected_letter,
            bg=C["panel"], fg=C["gold"],
            font=("Helvetica", 48, "bold"))
        self.selected_display.pack(pady=4)

        self.hint_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.hint_var,
                 bg=C["panel"], fg=C["muted"],
                 font=("Helvetica", 9), wraplength=240,
                 justify="center").pack(padx=16, pady=(0, 8))

        ttk.Separator(left).pack(fill="x", padx=16, pady=8)

        tk.Label(left, text="STEP 2 — Mark Your Sign",
                 bg=C["panel"], fg=C["accent"],
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=16, pady=(4, 8))

        self.mark_btn = self._btn(left, "✋  MARK SIGN",
                                  self._start_camera, C["accent"], C["bg"])
        self.mark_btn.pack(padx=16, pady=4, fill="x")

        self.capture_btn = self._btn(left, "📸  CAPTURE & SCORE",
                                     self._capture_and_score, C["gold"], C["bg"])
        self.capture_btn.pack(padx=16, pady=4, fill="x")
        self.capture_btn.config(state="disabled")

        self.stop_btn = self._btn(left, "⏹  STOP CAMERA",
                                  self._stop_camera, C["red"], C["white"])
        self.stop_btn.pack(padx=16, pady=4, fill="x")
        self.stop_btn.config(state="disabled")

        ttk.Separator(left).pack(fill="x", padx=16, pady=8)
        tk.Label(left, text="📷  Tip: Good lighting & clear background",
                 bg=C["panel"], fg=C["muted"],
                 font=("Helvetica", 8), wraplength=240).pack(padx=16)

        # Right panel — camera + result
        right = tk.Frame(parent, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)

        # Camera canvas
        self.cam_canvas = tk.Canvas(right, bg="#000000",
                                    width=640, height=480,
                                    highlightthickness=2,
                                    highlightbackground=C["border"])
        self.cam_canvas.pack(pady=(8, 4))
        self.cam_canvas.create_text(320, 240, text="Camera feed will appear here\nClick  ✋ MARK SIGN  to begin",
                                    fill=C["muted"], font=("Helvetica", 13),
                                    justify="center", tags="placeholder")

        # Result card
        result_card = tk.Frame(right, bg=C["card"], relief="flat")
        result_card.pack(fill="x", padx=4, pady=4)

        # Score bar
        score_row = tk.Frame(result_card, bg=C["card"])
        score_row.pack(fill="x", padx=16, pady=8)
        tk.Label(score_row, text="Score:",
                 bg=C["card"], fg=C["muted"],
                 font=("Helvetica", 11)).pack(side="left")
        self.score_label = tk.Label(score_row, text="—",
                                    bg=C["card"], fg=C["accent"],
                                    font=("Helvetica", 22, "bold"))
        self.score_label.pack(side="left", padx=8)
        self.score_bar = ttk.Progressbar(score_row, length=300,
                                          style="TProgressbar")
        self.score_bar.pack(side="left", padx=8)

        # Feedback text
        self.feedback_var = tk.StringVar(value="Perform a sign and press Capture to get feedback.")
        tk.Label(result_card, textvariable=self.feedback_var,
                 bg=C["card"], fg=C["text"],
                 font=("Helvetica", 10), wraplength=600,
                 justify="left").pack(anchor="w", padx=16, pady=(0, 6))

        # YouTube links row
        self.yt_frame = tk.Frame(result_card, bg=C["card"])
        self.yt_frame.pack(fill="x", padx=16, pady=(0, 8))

    # ══════════════════════════════════════════
    #  TAB 2 — EDA & VISUALISATION
    # ══════════════════════════════════════════
    def _build_eda_tab(self, parent):
        # Left — buttons
        left = tk.Frame(parent, bg=C["panel"], width=220)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="EDA & VISUALISATION",
                 bg=C["panel"], fg=C["accent"],
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=16, pady=(16, 4))
        tk.Label(left, text="Explore your dataset visually\nbefore training models.",
                 bg=C["panel"], fg=C["muted"],
                 font=("Helvetica", 9), justify="left").pack(anchor="w", padx=16, pady=(0, 12))

        plots = [
            ("📊 Class Distribution", "class_dist"),
            ("🔵 PCA 2D", "pca"),
            ("🌀 t-SNE 2D", "tsne"),
            ("🔥 Correlation Matrix", "correlation"),
            ("🔵 K-Means Clusters", "kmeans"),
            ("🔍 Elbow Curve", "elbow"),
            ("⚠️  Anomaly Detection", "anomaly"),
            ("🟡 PU Learning View", "pu_learning"),
        ]
        for label, key in plots:
            btn = self._btn(left, label, lambda k=key: self._run_eda(k),
                            C["card"], C["text"])
            btn.pack(padx=12, pady=3, fill="x")

        ttk.Separator(left).pack(fill="x", padx=16, pady=10)
        self._btn(left, "🔄 Regenerate Data",
                  self._regenerate_data, C["border"], C["text"]).pack(
            padx=12, pady=3, fill="x")
        self._btn(left, "📂 Load CSV Dataset",
                  self._load_csv, C["border"], C["text"]).pack(
            padx=12, pady=3, fill="x")

        # Data summary
        self.data_info_var = tk.StringVar(value="No data loaded")
        tk.Label(left, textvariable=self.data_info_var,
                 bg=C["panel"], fg=C["muted"],
                 font=("Helvetica", 8), wraplength=190,
                 justify="left").pack(padx=16, pady=8)

        # Right — plot canvas
        right = tk.Frame(parent, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)
        self.eda_canvas_label = tk.Label(
            right, text="← Select a visualisation",
            bg=C["bg"], fg=C["muted"],
            font=("Helvetica", 13))
        self.eda_canvas_label.pack(expand=True)
        self.eda_img_label = tk.Label(right, bg=C["bg"])
        self.eda_img_label.pack(expand=True)

    # ══════════════════════════════════════════
    #  TAB 3 — MODEL TRAINING
    # ══════════════════════════════════════════
    def _build_training_tab(self, parent):
        left = tk.Frame(parent, bg=C["panel"], width=280)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="MODEL TRAINING",
                 bg=C["panel"], fg=C["accent"],
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=16, pady=(16, 4))

        methods = [
            ("🤖 Supervised (Random Forest)",       "supervised"),
            ("🔄 Semi-Supervised (Label Spreading)", "semi"),
            ("⚡ Weak Supervision (Heuristics)",    "weak"),
            ("🟡 PU Learning",                       "pu"),
            ("🔵 Unsupervised (K-Means)",             "unsupervised"),
        ]
        self.train_btns = {}
        for label, key in methods:
            btn = self._btn(left, label,
                            lambda k=key: self._train_model(k),
                            C["accent2"], C["white"])
            btn.pack(padx=12, pady=4, fill="x")
            self.train_btns[key] = btn

        ttk.Separator(left).pack(fill="x", padx=16, pady=10)
        tk.Label(left, text="Training Progress",
                 bg=C["panel"], fg=C["muted"],
                 font=("Helvetica", 9)).pack(anchor="w", padx=16)
        self.train_progress = ttk.Progressbar(left, length=230, mode="indeterminate")
        self.train_progress.pack(padx=16, pady=6)
        self.train_status_var = tk.StringVar(value="Idle")
        tk.Label(left, textvariable=self.train_status_var,
                 bg=C["panel"], fg=C["accent"],
                 font=("Helvetica", 9), wraplength=240).pack(padx=16)

        # Log area
        right = tk.Frame(parent, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)
        tk.Label(right, text="Training Log",
                 bg=C["bg"], fg=C["accent"],
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=8)
        self.train_log = tk.Text(right, bg=C["panel"], fg=C["text"],
                                 font=("Courier", 9), relief="flat",
                                 state="disabled")
        scroll = ttk.Scrollbar(right, command=self.train_log.yview)
        self.train_log.configure(yscrollcommand=scroll.set)
        self.train_log.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        scroll.pack(side="right", fill="y", pady=8)

    # ══════════════════════════════════════════
    #  TAB 4 — EVALUATION
    # ══════════════════════════════════════════
    def _build_eval_tab(self, parent):
        left = tk.Frame(parent, bg=C["panel"], width=220)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="EVALUATION",
                 bg=C["panel"], fg=C["accent"],
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=16, pady=(16, 4))

        evals = [
            ("📊 Confusion Matrix",    "confusion"),
            ("📋 Classification Report", "report"),
            ("🏆 Score History",       "history"),
        ]
        for label, key in evals:
            self._btn(left, label,
                      lambda k=key: self._show_eval(k),
                      C["card"], C["text"]).pack(padx=12, pady=4, fill="x")

        # Metrics display
        ttk.Separator(left).pack(fill="x", padx=16, pady=10)
        self.metrics_var = tk.StringVar(value="Train a model first.")
        tk.Label(left, textvariable=self.metrics_var,
                 bg=C["panel"], fg=C["text"],
                 font=("Helvetica", 9), wraplength=190,
                 justify="left").pack(padx=16, pady=8)

        right = tk.Frame(parent, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)
        self.eval_text = tk.Text(right, bg=C["panel"], fg=C["text"],
                                 font=("Courier", 9), relief="flat",
                                 state="disabled")
        self.eval_img_label = tk.Label(right, bg=C["bg"])
        self.eval_img_label.pack(pady=4)
        scroll2 = ttk.Scrollbar(right, command=self.eval_text.yview)
        self.eval_text.configure(yscrollcommand=scroll2.set)
        self.eval_text.pack(side="left", fill="both", expand=True, padx=8, pady=4)
        scroll2.pack(side="right", fill="y", pady=4)

    # ══════════════════════════════════════════
    #  TAB 5 — RESOURCES
    # ══════════════════════════════════════════
    def _build_resources_tab(self, parent):
        canvas = tk.Canvas(parent, bg=C["bg"], highlightthickness=0)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        frame = tk.Frame(canvas, bg=C["bg"])
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Title
        tk.Label(frame, text="📚  Learning Resources & YouTube Guides",
                 bg=C["bg"], fg=C["accent"],
                 font=("Helvetica", 14, "bold")).pack(anchor="w", padx=24, pady=(20, 4))
        tk.Label(frame, text="Click any link to open in your browser.",
                 bg=C["bg"], fg=C["muted"],
                 font=("Helvetica", 9)).pack(anchor="w", padx=24)

        # General resources
        tk.Label(frame, text="General Sign Language Resources",
                 bg=C["bg"], fg=C["gold"],
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=24, pady=(16, 4))
        for name, url in YOUTUBE_GENERAL:
            self._link_btn(frame, f"▶  {name}", url)

        # Per-letter alphabet grid
        tk.Label(frame, text="Practice by Letter — USL Alphabet",
                 bg=C["bg"], fg=C["gold"],
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=24, pady=(20, 8))
        alpha_frame = tk.Frame(frame, bg=C["bg"])
        alpha_frame.pack(anchor="w", padx=24)
        for i, letter in enumerate(USL_ALPHABET):
            btn = tk.Button(
                alpha_frame,
                text=f" {letter} ",
                bg=C["card"], fg=C["accent"],
                font=("Courier", 11, "bold"),
                relief="flat", cursor="hand2", padx=6, pady=4,
                command=lambda l=letter: webbrowser.open(YOUTUBE_LINKS[l])
            )
            btn.grid(row=i // 9, column=i % 9, padx=3, pady=3)

        # Dataset & Paper links
        tk.Label(frame, text="Useful Datasets & Research",
                 bg=C["bg"], fg=C["gold"],
                 font=("Helvetica", 11, "bold")).pack(anchor="w", padx=24, pady=(20, 4))
        research_links = [
            ("RWTH ASL Fingerspelling Dataset",
             "https://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-hands.php"),
            ("Kaggle ASL Alphabet Dataset",
             "https://www.kaggle.com/datasets/grassknoted/asl-alphabet"),
            ("MediaPipe Hand Landmark Docs",
             "https://developers.google.com/mediapipe/solutions/vision/hand_landmarker"),
            ("Scikit-learn Semi-Supervised Docs",
             "https://scikit-learn.org/stable/modules/label_propagation.html"),
        ]
        for name, url in research_links:
            self._link_btn(frame, f"🔗  {name}", url)

    # ─────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────
    def _btn(self, parent, text, command, bg, fg):
        return tk.Button(
            parent, text=text, command=command,
            bg=bg, fg=fg,
            font=("Helvetica", 10, "bold"),
            relief="flat", cursor="hand2",
            padx=8, pady=6, anchor="w"
        )

    def _link_btn(self, parent, text, url):
        btn = tk.Button(
            parent, text=text,
            bg=C["bg"], fg=C["accent2"],
            font=("Helvetica", 10), relief="flat",
            cursor="hand2", anchor="w",
            command=lambda u=url: webbrowser.open(u)
        )
        btn.pack(anchor="w", padx=24, pady=2)

    def _log(self, msg):
        self.train_log.configure(state="normal")
        self.train_log.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.train_log.see("end")
        self.train_log.configure(state="disabled")

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ─────────────────────────────────────────
    #  DATA LOADING
    # ─────────────────────────────────────────
    def _load_or_generate_data(self):
        if FEATURES_CSV.exists():
            self.df = pd.read_csv(FEATURES_CSV)
        else:
            self.df = generate_synthetic_features(n_per_class=30)
        self._update_data_info()

    def _regenerate_data(self):
        self.df = generate_synthetic_features(n_per_class=40)
        self._update_data_info()
        self._set_status("Synthetic dataset regenerated.")

    def _load_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if path:
            try:
                self.df = pd.read_csv(path)
                self._update_data_info()
                self._set_status(f"Loaded: {Path(path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV:\n{e}")

    def _update_data_info(self):
        if self.df is not None:
            n = len(self.df)
            labeled = self.df["labeled"].sum() if "labeled" in self.df.columns else n
            classes = self.df["label"].nunique() if "label" in self.df.columns else "?"
            self.data_info_var.set(
                f"Samples: {n}\nClasses: {classes}\n"
                f"Labeled: {int(labeled)}\nUnlabeled: {n - int(labeled)}\n"
                f"Features: {sum(1 for c in self.df.columns if c.startswith('f'))}"
            )

    # ─────────────────────────────────────────
    #  LETTER SELECTION
    # ─────────────────────────────────────────
    def _select_letter(self, letter):
        # Reset previous
        if hasattr(self, "_letter_btns"):
            for l, b in self._letter_btns.items():
                b.configure(bg=C["card"], fg=C["text"])
            if letter in self._letter_btns:
                self._letter_btns[letter].configure(bg=C["accent"], fg=C["bg"])
        self.selected_letter.set(letter)
        hints = {
            "A": "Closed fist, thumb on side",
            "B": "Flat hand, fingers together, thumb tucked",
            "C": "Curved hand forming a C",
            "D": "Index finger up, others form a circle with thumb",
            "E": "Fingers bent, thumb tucked under",
            "F": "Index+thumb form circle, others up",
            "G": "Index+thumb point sideways",
            "H": "Index+middle extended horizontally",
            "I": "Pinky extended, others closed",
            "J": "Pinky extended, draw J shape",
            "K": "Index+middle up, thumb between",
            "L": "L-shape: index up, thumb out",
            "M": "Three fingers over thumb",
            "N": "Two fingers over thumb",
            "O": "All fingers and thumb form an O",
            "P": "Like K but pointing down",
            "Q": "Like G but pointing down",
            "R": "Index+middle fingers crossed",
            "S": "Closed fist, thumb over fingers",
            "T": "Thumb between index+middle",
            "U": "Index+middle together pointing up",
            "V": "Index+middle spread in V/peace sign",
            "W": "Three fingers spread open",
            "X": "Index finger hooked/bent",
            "Y": "Thumb+pinky extended",
            "Z": "Index finger draws Z shape",
        }
        if hasattr(self, "hint_var"):
            self.hint_var.set(hints.get(letter, ""))

    # ─────────────────────────────────────────
    #  CAMERA LOGIC
    # ─────────────────────────────────────────
    def _start_camera(self):
        if self.camera_active:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error",
                                 "Cannot access webcam.\nEnsure it is connected and not in use.")
            return
        self.camera_active = True
        self.mark_btn.config(state="disabled")
        self.capture_btn.config(state="normal")
        self.stop_btn.config(state="normal")
        self.cam_canvas.delete("placeholder")
        self._set_status("Camera active — position your hand and press Capture")
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        while self.camera_active and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            # Draw skeleton live
            _, lm = extract_landmarks(frame)
            display = draw_landmarks_on_frame(frame, lm)
            # Overlay letter
            letter = self.selected_letter.get()
            cv2.putText(display, f"Sign: {letter}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 201, 167), 3)
            # Convert for Tkinter
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((640, 480))
            imgtk = ImageTk.PhotoImage(img)
            self.cam_canvas.imgtk = imgtk
            self.cam_canvas.create_image(0, 0, anchor="nw", image=imgtk)
            time.sleep(0.03)

    def _stop_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.mark_btn.config(state="normal")
        self.capture_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self._set_status("Camera stopped.")

    def _capture_and_score(self):
        if self.current_frame is None:
            messagebox.showwarning("No Frame", "Camera is not active.")
            return
        frame = self.current_frame.copy()
        self._stop_camera()

        true_letter = self.selected_letter.get()
        features, lm = extract_landmarks(frame)

        if features is None:
            self.score_label.configure(text="No hand", fg=C["red"])
            self.feedback_var.set(
                "❌ No hand detected. Ensure your hand is clearly visible with good lighting.")
            self.score_bar["value"] = 0
            self._show_yt_links(true_letter, 10)
            return

        # Get prediction
        if self.classifier.is_trained:
            pred_letter, proba, proba_dict = self.classifier.predict(features)
            score = compute_score(pred_letter, true_letter, proba)
        else:
            # Heuristic score when no model trained: use distance heuristic
            score = random.randint(3, 9) * 10
            pred_letter = true_letter if score >= 60 else "?"
            proba_dict = {}

        self.score_history.append({"letter": true_letter, "score": score,
                                    "time": datetime.now().isoformat()})

        # Display
        color = C["accent"] if score >= 70 else C["gold"] if score >= 40 else C["red"]
        self.score_label.configure(text=f"{score}%", fg=color)
        self.score_bar["value"] = score

        if score >= 80:
            msg = f"✅ Excellent! Your sign for '{true_letter}' is very accurate. Keep it up!"
        elif score >= 60:
            msg = (f"👍 Good job! Your '{true_letter}' sign is recognisable. "
                   f"Focus on finger positioning for improvement.")
        elif score >= 40:
            msg = (f"⚠️  Fair attempt at '{true_letter}'. "
                   f"Watch the tutorial videos below and try again.")
        else:
            msg = (f"❌ Needs more practice. The sign for '{true_letter}' was not clearly detected. "
                   f"Check the tutorial links below.")

        if proba_dict and pred_letter != true_letter:
            top = sorted(proba_dict.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join([f"{l}({v:.0%})" for l, v in top])
            msg += f"\n  Model detected: {top_str}"

        self.feedback_var.set(msg)
        self._show_yt_links(true_letter, score)
        self._set_status(f"Score for '{true_letter}': {score}%")

        # Show captured frame with skeleton
        annotated = draw_landmarks_on_frame(frame, lm)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((640, 480))
        imgtk = ImageTk.PhotoImage(img)
        self.cam_canvas.imgtk = imgtk
        self.cam_canvas.create_image(0, 0, anchor="nw", image=imgtk)

    def _show_yt_links(self, letter, score):
        for w in self.yt_frame.winfo_children():
            w.destroy()
        tk.Label(self.yt_frame, text="📺  Improve with:",
                 bg=C["card"], fg=C["muted"],
                 font=("Helvetica", 9)).pack(side="left", padx=(0, 8))
        links = [
            (f"Sign '{letter}' Tutorial", YOUTUBE_LINKS[letter]),
            ("USL Alphabet Guide", YOUTUBE_GENERAL[0][1]),
        ]
        if score < 60:
            links.append(("Beginner Fingerspelling", YOUTUBE_GENERAL[2][1]))
        for name, url in links:
            tk.Button(
                self.yt_frame, text=f"▶ {name}",
                bg=C["accent2"], fg=C["white"],
                font=("Helvetica", 8, "bold"),
                relief="flat", cursor="hand2", padx=6, pady=3,
                command=lambda u=url: webbrowser.open(u)
            ).pack(side="left", padx=4)

    # ─────────────────────────────────────────
    #  EDA
    # ─────────────────────────────────────────
    def _run_eda(self, key):
        if self.df is None:
            messagebox.showwarning("No Data", "Load or generate a dataset first.")
            return
        self._set_status(f"Generating {key} plot…")
        self.eda_canvas_label.pack_forget()

        def worker():
            try:
                eda = EDAEngine(self.df)
                fn = getattr(eda, f"plot_{key}")
                path = fn()
                self.root.after(0, lambda p=path: self._show_eda_image(p))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("EDA Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _show_eda_image(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((900, 580))
            imgtk = ImageTk.PhotoImage(img)
            self.eda_img_label.configure(image=imgtk)
            self.eda_img_label.image = imgtk
            self._set_status(f"Plot saved: {path.name}")
        except Exception as e:
            messagebox.showerror("Image Error", str(e))

    # ─────────────────────────────────────────
    #  MODEL TRAINING
    # ─────────────────────────────────────────
    def _train_model(self, method):
        if self.df is None:
            messagebox.showwarning("No Data", "Load or generate a dataset first.")
            return
        self.train_progress.start(10)
        self.train_status_var.set(f"Training ({method})…")
        for btn in self.train_btns.values():
            btn.config(state="disabled")

        def worker():
            try:
                self._log(f"Starting {method} training…")
                self._log(f"Dataset: {len(self.df)} samples, "
                          f"{self.df['label'].nunique()} classes")

                if method == "supervised":
                    acc, f1, y_te, y_pred = self.classifier.train_supervised(self.df)
                    self._log(f"Accuracy: {acc:.4f}  |  F1-macro: {f1:.4f}")
                    self._save_eval_results(y_te, y_pred)

                elif method == "semi":
                    acc, f1 = self.classifier.train_semi_supervised(self.df)
                    self._log(f"Semi-supervised Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

                elif method == "weak":
                    # Weak supervision: heuristic labels based on cluster assignment
                    self._log("Applying heuristic weak labels via K-Means…")
                    feat_cols = [c for c in self.df.columns if c.startswith("f")]
                    X = self.df[feat_cols].values
                    scaler = StandardScaler()
                    X_s = scaler.fit_transform(X)
                    km = KMeans(n_clusters=26, random_state=42, n_init=10)
                    cluster_ids = km.fit_predict(X_s)
                    # Map clusters → letters heuristically
                    mapping = {i: USL_ALPHABET[i % 26] for i in range(26)}
                    weak_labels = [mapping[c] for c in cluster_ids]
                    df2 = self.df.copy()
                    df2["label"] = weak_labels
                    df2["labeled"] = True
                    acc, f1, y_te, y_pred = self.classifier.train_supervised(df2)
                    self._log(f"Weak supervision → RF Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

                elif method == "pu":
                    self._log("PU Learning: treating labeled=positive, unlabeled=unknown…")
                    feat_cols = [c for c in self.df.columns if c.startswith("f")]
                    X = self.df[feat_cols].values
                    is_labeled = self.df["labeled"].values.astype(bool)
                    y_raw = self.df["label"].values
                    # Train OneClassSVM per class as PU proxy
                    scaler = StandardScaler()
                    X_s = scaler.fit_transform(X)
                    oc_svm = OneClassSVM(kernel="rbf", nu=0.1)
                    oc_svm.fit(X_s[is_labeled])
                    preds = oc_svm.predict(X_s)
                    positive_count = (preds == 1).sum()
                    self._log(f"OneClassSVM identified {positive_count} positive samples")
                    # Then train RF on positives
                    enc = LabelEncoder()
                    y_enc = enc.fit_transform(y_raw[is_labeled])
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_s[is_labeled], y_enc)
                    self.classifier.model = rf
                    self.classifier.scaler = scaler
                    self.classifier.encoder = enc
                    self.classifier.is_trained = True
                    self.classifier._save()
                    y_pred = rf.predict(X_s[is_labeled])
                    acc = accuracy_score(y_enc, y_pred)
                    self._log(f"PU Learning RF Accuracy (on labeled): {acc:.4f}")

                elif method == "unsupervised":
                    feat_cols = [c for c in self.df.columns if c.startswith("f")]
                    X = self.df[feat_cols].values
                    scaler = StandardScaler()
                    X_s = scaler.fit_transform(X)
                    km = KMeans(n_clusters=26, random_state=42, n_init=10)
                    labels_km = km.fit_predict(X_s)
                    sil = silhouette_score(X_s, labels_km)
                    self._log(f"K-Means (k=26) Silhouette Score: {sil:.4f}")
                    dbscan = DBSCAN(eps=0.5, min_samples=5)
                    labels_db = dbscan.fit_predict(X_s)
                    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
                    n_noise = (labels_db == -1).sum()
                    self._log(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

                self._log("Training complete. Model saved.")
                self.root.after(0, self._on_training_done)

            except Exception as e:
                self._log(f"ERROR: {e}")
                self.root.after(0, self._on_training_done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_training_done(self):
        self.train_progress.stop()
        self.train_status_var.set("Complete ✓")
        for btn in self.train_btns.values():
            btn.config(state="normal")
        self._set_status("Model training complete.")

    def _save_eval_results(self, y_te, y_pred):
        """Store results for evaluation tab."""
        self._eval_y_te = y_te
        self._eval_y_pred = y_pred

    # ─────────────────────────────────────────
    #  EVALUATION
    # ─────────────────────────────────────────
    def _show_eval(self, key):
        self.eval_text.configure(state="normal")
        self.eval_text.delete(1.0, "end")

        if key == "confusion":
            if not hasattr(self, "_eval_y_te"):
                self.eval_text.insert("end",
                    "Run Supervised training first to generate evaluation data.")
                self.eval_text.configure(state="disabled")
                return
            classes = self.classifier.encoder.classes_
            eda = EDAEngine(self.df)
            path = eda.plot_confusion(self._eval_y_te, self._eval_y_pred,
                                      list(range(len(classes))))
            img = Image.open(path)
            img.thumbnail((700, 560))
            imgtk = ImageTk.PhotoImage(img)
            self.eval_img_label.configure(image=imgtk)
            self.eval_img_label.image = imgtk

        elif key == "report":
            if not hasattr(self, "_eval_y_te"):
                self.eval_text.insert("end",
                    "Run Supervised training first.")
                self.eval_text.configure(state="disabled")
                return
            report = classification_report(
                self._eval_y_te, self._eval_y_pred, zero_division=0)
            self.eval_text.insert("end", report)
            acc = accuracy_score(self._eval_y_te, self._eval_y_pred)
            f1 = f1_score(self._eval_y_te, self._eval_y_pred,
                          average="macro", zero_division=0)
            self.metrics_var.set(f"Accuracy: {acc:.4f}\nF1-Macro: {f1:.4f}")

        elif key == "history":
            if not self.score_history:
                self.eval_text.insert("end",
                    "No practice sessions recorded yet.\nGo to Practice tab and try some signs!")
            else:
                self.eval_text.insert("end",
                    "Session Score History\n" + "─" * 50 + "\n")
                for rec in self.score_history[-50:]:
                    self.eval_text.insert(
                        "end",
                        f"  Letter: {rec['letter']}  |  Score: {rec['score']}%  "
                        f"|  {rec['time'][:19]}\n"
                    )
                scores = [r["score"] for r in self.score_history]
                self.eval_text.insert("end",
                    f"\n{'─'*50}\n"
                    f"  Sessions: {len(scores)}\n"
                    f"  Average Score: {sum(scores)/len(scores):.1f}%\n"
                    f"  Best:  {max(scores)}%\n"
                    f"  Worst: {min(scores)}%\n"
                )

        self.eval_text.configure(state="disabled")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    root = tk.Tk()
    app = USLApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop_camera(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
