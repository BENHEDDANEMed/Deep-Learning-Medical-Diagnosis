"""
Application Streamlit — Détection de Pneumonie par Radiographie Thoracique
==========================================================================
Lance avec :  streamlit run app_streamlit.py
"""

from __future__ import annotations

import datetime
import io
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ──────────────────────────────────────────────────────────────────────────────
# Constantes de style
# ──────────────────────────────────────────────────────────────────────────────
MEDICAL_BLUE = "#1F5C99"
MEDICAL_BLUE_LIGHT = "#2A7BC8"
MEDICAL_BLUE_DARK = "#163F6B"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F0F2F6"
NORMAL_GREEN = "#27AE60"
PNEUMONIA_RED = "#E74C3C"
WARNING_ORANGE = "#F39C12"

# ──────────────────────────────────────────────────────────────────────────────
# Configuration de la page
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI — Détection de Pneumonie",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS personnalisé
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
/* ---------- En-tête principal ---------- */
.main-header {{
    background: linear-gradient(135deg, {MEDICAL_BLUE_DARK}, {MEDICAL_BLUE}, {MEDICAL_BLUE_LIGHT});
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(31,92,153,0.25);
}}
.main-header h1 {{
    color: {WHITE};
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0 0 .6rem 0;
    letter-spacing: 1px;
}}
.main-header p {{
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
    margin: 0;
    line-height: 1.6;
}}

/* ---------- Carte diagnostic ---------- */
.diag-card {{
    background: {WHITE};
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    margin: 1rem 0;
    border-left: 6px solid;
}}
.diag-card.normal  {{ border-color: {NORMAL_GREEN}; }}
.diag-card.pneumonia {{ border-color: {PNEUMONIA_RED}; }}

/* ---------- Badge ---------- */
.badge {{
    display: inline-block;
    padding: .5rem 1.4rem;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: .5px;
    color: {WHITE};
}}
.badge-normal   {{ background: {NORMAL_GREEN}; }}
.badge-pneumonia {{ background: {PNEUMONIA_RED}; }}

/* ---------- Confiance ---------- */
.confidence-value {{
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    margin: .8rem 0;
}}
.conf-normal   {{ color: {NORMAL_GREEN}; }}
.conf-pneumonia {{ color: {PNEUMONIA_RED}; }}

/* ---------- Risque ---------- */
.risk-tag {{
    display: inline-block;
    padding: .35rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: .95rem;
    color: {WHITE};
}}
.risk-low    {{ background: {NORMAL_GREEN}; }}
.risk-mod    {{ background: {WARNING_ORANGE}; }}
.risk-high   {{ background: {PNEUMONIA_RED}; }}

/* ---------- Barre progression custom ---------- */
.prog-bar-bg {{
    background: #E8ECF0;
    border-radius: 10px;
    height: 22px;
    overflow: hidden;
    margin: .4rem 0;
}}
.prog-bar-fill {{
    height: 100%;
    border-radius: 10px;
    transition: width .8s ease;
}}

/* ---------- Avertissement médical ---------- */
.medical-warning {{
    background: linear-gradient(90deg, #FFF3CD, #FFF8E1);
    border-left: 5px solid {WARNING_ORANGE};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-top: 1.5rem;
    font-size: .92rem;
    color: #856404;
}}

/* ---------- Historique ---------- */
.hist-item {{
    background: {WHITE};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: .7rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    display: flex;
    align-items: center;
    gap: 1rem;
}}

/* Divers */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {MEDICAL_BLUE_DARK}, {MEDICAL_BLUE});
}}
[data-testid="stSidebar"] * {{
    color: {WHITE} !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Chargement du modèle (cache Streamlit)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du modèle ResNet18…")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(__file__).parent / "artifacts" / "resnet18_pneumonia_finetuned.pth"
    if not ckpt_path.exists():
        st.error(f"Modèle introuvable : {ckpt_path}")
        st.stop()
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt.get("classes", ["NORMAL", "PNEUMONIA"])
    img_size = int(ckpt.get("img_size", 224))

    net = models.resnet18(weights=None)
    net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.fc.in_features, len(classes)))
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return net, tfm, classes, device


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────
def get_gradcam(model, img_tensor, target_layer, device):
    model.eval()
    activations, gradients = {}, {}

    def fwd_hook(_, __, output):
        activations["v"] = output.detach()

    def bwd_hook(_, __, grad_out):
        gradients["v"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    inp = img_tensor.unsqueeze(0).to(device)
    out = model(inp)
    pred_idx = out.argmax(dim=1).item()
    score = out[0, pred_idx]
    model.zero_grad()
    score.backward()
    h1.remove(); h2.remove()

    act = activations["v"][0]
    grad = gradients["v"][0]
    weights = grad.mean(dim=(1, 2))
    heatmap = (weights[:, None, None] * act).sum(dim=0)
    heatmap = torch.relu(heatmap).cpu().numpy()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    probs = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
    return heatmap, pred_idx, probs


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * std + mean, 0, 1)


def overlay_heatmap(img_np, heatmap, alpha=0.4):
    h, w = img_np.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm8 = np.uint8(255 * hm)
    colored = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    img8 = np.uint8(255 * img_np)
    return cv2.addWeighted(img8, 1 - alpha, colored, alpha, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Prédiction
# ──────────────────────────────────────────────────────────────────────────────
def predict(image: Image.Image, model, tfm, classes, device):
    img_rgb = image.convert("RGB")
    img_tensor = tfm(img_rgb)
    target_layer = model.layer4[-1]
    heatmap, pred_idx, probs = get_gradcam(model, img_tensor, target_layer, device)
    img_np = denormalize(img_tensor)
    superimposed = overlay_heatmap(img_np, heatmap)
    return {
        "class": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
        "original_np": img_np,
        "heatmap": heatmap,
        "superimposed": superimposed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions d'affichage
# ──────────────────────────────────────────────────────────────────────────────
def render_progress_bar(label: str, value: float, color: str):
    pct = int(value * 100)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.6rem;margin:.3rem 0">
            <span style="min-width:110px;font-weight:600">{label}</span>
            <div class="prog-bar-bg" style="flex:1">
                <div class="prog-bar-fill" style="width:{pct}%;background:{color}"></div>
            </div>
            <span style="font-weight:700;min-width:50px;text-align:right">{pct}%</span>
        </div>""",
        unsafe_allow_html=True,
    )


def risk_level(conf: float, is_pneumonia: bool):
    if not is_pneumonia:
        return ("Faible", "risk-low")
    if conf < 0.70:
        return ("Modéré", "risk-mod")
    return ("Élevé", "risk-high")


def medical_warning():
    st.markdown(
        '<div class="medical-warning">⚠️ <strong>Avertissement médical</strong> '
        "— Cet outil est une <u>aide au diagnostic</u> et ne remplace en aucun cas "
        "l'avis d'un médecin qualifié. Les résultats doivent être interprétés par un "
        "professionnel de santé.</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Onglet : Métriques du modèle
# ──────────────────────────────────────────────────────────────────────────────
def metrics_tab():
    st.markdown("### 📊 Comparaison des Modèles")

    st.markdown("<p style='color: #666; margin-bottom: 20px;'>Métriques détaillées sur le test set (1 706 images)</p>", unsafe_allow_html=True)

    data = {
        "Métrique": ["Accuracy", "Balanced Accuracy", "Sensitivity (Pneumonia)", "Specificity (Normal)", "F1-Score (Pneumonia)", "ROC-AUC"],
        "Baseline CNN":       [0.9333, 0.9333, 1.0000, 0.8667, 0.9375, 0.9956],
        "ResNet18 TL":       [0.9200, 0.9200, 0.9600, 0.8800, 0.9231, 0.9800],
        "ResNet18 Fine-Tuned": [0.9789, 0.9789, 0.9707, 0.9871, 0.9787, 0.9979],
    }

    html_table = """
    <style>
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 2rem;
        font-family: sans-serif;
    }
    .metrics-table th, .metrics-table td {
        padding: 12px 15px;
        text-align: center;
        border-bottom: 1px solid #f0f2f6;
    }
    .metrics-table th:first-child, .metrics-table td:first-child {
        text-align: left;
        font-weight: 600;
    }
    .metrics-table th {
        color: white;
        font-weight: 600;
        font-size: 0.9em;
    }
    .th-baseline { background-color: #1A365D; }
    .th-finetuned { background-color: #2F857A; }
    .td-finetuned { background-color: #E6F2F0; font-weight: 600; color: #111; }
    .sub-text { font-size: 0.8em; color: #555; font-weight: 400; }
    .arrow-up { color: black; font-weight: 600; }
    </style>
    <table class="metrics-table">
        <thead>
            <tr>
                <th class="th-baseline">Métrique</th>
                <th class="th-baseline">Baseline CNN</th>
                <th class="th-baseline">ResNet18 TL</th>
                <th class="th-finetuned">ResNet18 Fine-Tuned</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Accuracy</td>
                <td>93.33%</td>
                <td>92.00%</td>
                <td class="td-finetuned">97.89%<br><span class="sub-text arrow-up">▲ +4.56 pp</span></td>
            </tr>
            <tr>
                <td>Balanced Accuracy</td>
                <td>93.33%</td>
                <td>92.00%</td>
                <td class="td-finetuned">97.89%</td>
            </tr>
            <tr>
                <td>Sensitivity (Pneumonia)</td>
                <td>100.00%<br><span class="sub-text">0 faux négatifs</span></td>
                <td>96.00%</td>
                <td class="td-finetuned">97.07%</td>
            </tr>
            <tr>
                <td>Specificity (Normal)</td>
                <td>86.67%</td>
                <td>88.00%</td>
                <td class="td-finetuned">98.71%<br><span class="sub-text arrow-up">▲ +12.04 pp</span></td>
            </tr>
            <tr>
                <td>F1-Score (Pneumonia)</td>
                <td>93.75%</td>
                <td>92.31%</td>
                <td class="td-finetuned">97.87%</td>
            </tr>
            <tr>
                <td>ROC-AUC</td>
                <td>99.56%</td>
                <td>98.00%</td>
                <td class="td-finetuned">99.79%</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(html_table, unsafe_allow_html=True)

    # Graphiques Plotly
    st.markdown("---")
    models_list = ["Baseline CNN", "ResNet18 TL", "ResNet18 Fine-Tuned"]
    colors = [MEDICAL_BLUE_LIGHT, WARNING_ORANGE, NORMAL_GREEN]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for i, m in enumerate(models_list):
            fig.add_trace(go.Bar(name=m, x=data["Métrique"], y=data[m],
                                 marker_color=colors[i], text=[f"{v:.1%}" for v in data[m]],
                                 textposition="outside"))
        fig.update_layout(barmode="group", title="Performance par métrique",
                          yaxis=dict(range=[0.8, 1.05], tickformat=".0%"),
                          template="plotly_white", height=450,
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig2 = go.Figure()
        for i, m in enumerate(models_list):
            fig2.add_trace(go.Scatterpolar(
                r=data[m] + [data[m][0]],
                theta=data["Métrique"] + [data["Métrique"][0]],
                fill="toself", name=m,
                line=dict(color=colors[i]),
                opacity=0.7,
            ))
        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.8, 1.0])),
            title="Radar — comparaison globale",
            template="plotly_white", height=450,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig2, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# Onglet : Historique
# ──────────────────────────────────────────────────────────────────────────────
def history_tab():
    st.markdown("### 📋 Historique des prédictions")
    if "history" not in st.session_state or not st.session_state.history:
        st.info("Aucune prédiction effectuée dans cette session.")
        return

    if st.button("🗑️ Vider l'historique", type="secondary"):
        st.session_state.history = []
        st.rerun()

    for item in reversed(st.session_state.history):
        is_pneu = item["class"] == "PNEUMONIA"
        badge_cls = "badge-pneumonia" if is_pneu else "badge-normal"
        badge_text = item["class"]
        conf = item["confidence"]
        ts = item["timestamp"]
        fname = item.get("filename", "image")
        st.markdown(
            f"""<div class="hist-item">
                <span class="badge {badge_cls}">{badge_text}</span>
                <span style="font-weight:600">{fname}</span>
                <span style="flex:1"></span>
                <span style="font-weight:700;font-size:1.1rem">{conf:.1%}</span>
                <span style="color:#888;font-size:.85rem">{ts}</span>
            </div>""",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Initialisation session
    if "history" not in st.session_state:
        st.session_state.history = []

    model, tfm, classes, device = load_model()

    # ── Sidebar ──
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/lungs.png", width=64)
        st.markdown("## PneumoScan AI")
        st.markdown("---")
        st.markdown("**Modèle actif**")
        st.markdown("ResNet18 Fine-Tuned")
        st.markdown(f"**Précision test** : 97.89 %")
        st.markdown(f"**Device** : `{device}`")
        st.markdown("---")
        st.markdown(
            "**Classes** : `NORMAL` · `PNEUMONIA`"
        )
        st.markdown("---")
        st.caption("© 2026 — Projet Deep Learning")

    # ── Header ──
    st.markdown(
        """<div class="main-header">
            <h1>🫁 PneumoScan AI</h1>
            <p>Système d'aide au diagnostic de pneumonie par analyse de radiographies
            thoraciques utilisant un réseau de neurones profond (ResNet18 Fine-Tuned).</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Onglets ──
    tab_diag, tab_metrics, tab_hist = st.tabs([
        "🔬 Diagnostic", "📊 Métriques du modèle", "📋 Historique"
    ])

    # ── TAB 1 : Diagnostic ──
    with tab_diag:
        st.markdown("### 📤 Upload d'image")
        uploaded = st.file_uploader(
            "Glissez-déposez ou sélectionnez une radiographie thoracique",
            type=["jpg", "jpeg", "png"],
            help="Formats acceptés : JPEG, PNG",
        )

        if uploaded is not None:
            image = Image.open(uploaded)

            with st.spinner("🔄 Analyse en cours…"):
                result = predict(image, model, tfm, classes, device)

            is_pneumonia = result["class"] == "PNEUMONIA"
            conf = result["confidence"]
            risk_text, risk_cls = risk_level(conf, is_pneumonia)

            # Carte diagnostic
            card_cls = "pneumonia" if is_pneumonia else "normal"
            badge_cls = "badge-pneumonia" if is_pneumonia else "badge-normal"
            conf_cls = "conf-pneumonia" if is_pneumonia else "conf-normal"

            st.markdown(
                f"""<div class="diag-card {card_cls}">
                    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap">
                        <div>
                            <span class="badge {badge_cls}">{result['class']}</span>
                            <span class="risk-tag {risk_cls}" style="margin-left:.8rem">Risque {risk_text}</span>
                        </div>
                        <div class="confidence-value {conf_cls}">{conf:.1%}</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Barres de progression
            for cls_name in classes:
                color = NORMAL_GREEN if cls_name == "NORMAL" else PNEUMONIA_RED
                render_progress_bar(cls_name, result["probs"][cls_name], color)

            st.markdown("---")

            # Grad-CAM
            st.markdown("### 🔥 Visualisation Grad-CAM")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Image originale**")
                st.image(image, width="stretch")
            with c2:
                st.markdown("**Heatmap Grad-CAM**")
                hm8 = np.uint8(255 * cv2.resize(result["heatmap"],
                               (result["original_np"].shape[1], result["original_np"].shape[0])))
                colored = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
                colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                st.image(colored, width="stretch")
            with c3:
                st.markdown("**Superposition**")
                st.image(result["superimposed"], width="stretch")

            # Légende colorée
            st.markdown(
                """<div style="display:flex;align-items:center;gap:1.5rem;margin:.8rem 0;justify-content:center">
                    <span style="display:flex;align-items:center;gap:.3rem">
                        <span style="width:16px;height:16px;background:#0000FF;border-radius:3px;display:inline-block"></span> Faible activation
                    </span>
                    <span style="display:flex;align-items:center;gap:.3rem">
                        <span style="width:16px;height:16px;background:#00FF00;border-radius:3px;display:inline-block"></span> Modérée
                    </span>
                    <span style="display:flex;align-items:center;gap:.3rem">
                        <span style="width:16px;height:16px;background:#FF0000;border-radius:3px;display:inline-block"></span> Forte activation
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )

            # Ajouter à l'historique
            st.session_state.history.append({
                "class": result["class"],
                "confidence": conf,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "filename": uploaded.name,
            })

        medical_warning()

    # ── TAB 2 : Métriques ──
    with tab_metrics:
        metrics_tab()
        medical_warning()

    # ── TAB 3 : Historique ──
    with tab_hist:
        history_tab()
        medical_warning()


if __name__ == "__main__":
    main()
