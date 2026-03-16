"""
╔══════════════════════════════════════════════════════════════════════╗
║   SYSTÈME D'ALERTE PRÉCOCE — RISQUE FOURNISSEURS MAROC              ║
║   Application Streamlit Cloud — Pipeline ML Non Supervisé           ║
║   Université Mohammed V Rabat — Master ML & Intelligence Logistique  ║
║   2024–2025                                                          ║
╠══════════════════════════════════════════════════════════════════════╣
║   LANCEMENT LOCAL :                                                  ║
║     pip install -r requirements.txt                                  ║
║     streamlit run app.py                                             ║
║                                                                      ║
║   DÉPLOIEMENT STREAMLIT CLOUD :                                      ║
║     1. Uploadez app.py + requirements.txt sur GitHub                 ║
║     2. Allez sur share.streamlit.io → New app                        ║
║     3. Sélectionnez votre dépôt → Deploy                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ── Imports standard ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import io
import time

import numpy  as np
import pandas as pd

import plotly.express        as px
import plotly.graph_objects  as go
from plotly.subplots import make_subplots

import streamlit as st

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition  import PCA
from sklearn.ensemble       import IsolationForest
from sklearn.impute         import SimpleImputer
from sklearn.metrics        import silhouette_score, davies_bouldin_score

import umap
import hdbscan
import shap

import torch
import torch.nn            as nn
import torch.optim         as optim
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════════════════════════════════════
#  0. CONFIGURATION DE LA PAGE  (doit être la PREMIÈRE instruction Streamlit)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Alerte Fournisseurs — Maroc",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  1. CSS  —  styles personnalisés injectés une seule fois
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── En-tête principal ── */
.entete {
    background: linear-gradient(135deg, #1F3864 0%, #2E5496 100%);
    padding: 20px 28px; border-radius: 10px;
    margin-bottom: 20px;
}
.entete h1 { color: white; font-size: 1.7rem; margin: 0; }
.entete p  { color: #BDD7EE; font-size: 0.9rem; margin: 5px 0 0; }

/* ── Cartes KPI ── */
.kpi-card {
    border: 1px solid #e0e0e0; border-radius: 8px;
    padding: 16px 10px; text-align: center;
    background: white; box-shadow: 0 2px 6px rgba(0,0,0,.07);
}
.kpi-val { font-size: 2.2rem; font-weight: 700; color: #1F3864; }
.kpi-lab { font-size: 0.8rem;  color: #595959; margin-top: 4px; }

/* ── Niveaux d'alerte ── */
.alerte-rouge  { background:#ffcccc; border-left:5px solid #C00000;
                  padding:12px 18px; border-radius:6px;
                  font-weight:700; margin:8px 0; }
.alerte-orange { background:#ffe0b2; border-left:5px solid #C55A11;
                  padding:12px 18px; border-radius:6px;
                  font-weight:700; margin:8px 0; }
.alerte-vert   { background:#ccffcc; border-left:5px solid #375623;
                  padding:12px 18px; border-radius:6px;
                  font-weight:700; margin:8px 0; }

/* ── Barre SHAP visuelle ── */
.shap-bar {
    display:inline-block; height:11px; border-radius:3px;
    vertical-align:middle; margin-left:8px;
}

/* ── Boîte info ── */
.info-box {
    background:#EEF4FB; border-left:4px solid #1F3864;
    padding:12px 16px; border-radius:6px; margin:8px 0;
}

/* ── Masquer menu hamburger Streamlit ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  2. CONSTANTES ET PARAMÈTRES PAR DÉFAUT
# ══════════════════════════════════════════════════════════════════════════════
# Colonnes à exclure du pipeline ML (identifiants et sorties)
EXCLURE = [
    "ID_Fournisseur", "Nom_Fournisseur", "Secteur", "Region_Maroc",
    "Cluster_Reel", "Note_Risque_Pays", "Certification",
    "Niveau_Alerte", "Priorite_Action", "Score_Risque",
    "Alerte_ML", "Priorite_ML", "Cluster_HDBSCAN",
]

COULEURS_ALERTES = {
    "🟢 Vert"  : "#4CAF50",
    "🟠 Orange": "#FF9800",
    "🔴 Rouge" : "#F44336",
}


# ══════════════════════════════════════════════════════════════════════════════
#  3. BARRE LATÉRALE — PARAMÈTRES DU PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Paramètres du pipeline")
    st.markdown("---")

    st.markdown("### 🔵 UMAP")
    p_neighbors = st.slider(
        "n_neighbors", 5, 50, 15,
        help="Nombre de voisins pour préserver la structure locale"
    )
    p_min_dist = st.slider(
        "min_dist", 0.0, 0.5, 0.1, 0.05,
        help="Distance minimale entre points projetés"
    )

    st.markdown("### 🟢 HDBSCAN")
    p_min_cluster = st.slider(
        "min_cluster_size", 3, 30, 10,
        help="Taille minimale pour qu'un groupe soit un cluster"
    )
    p_min_samples = st.slider(
        "min_samples", 1, 15, 5,
        help="Plus ce nombre est élevé, plus le clustering est conservateur"
    )

    st.markdown("### 🔴 Isolation Forest")
    p_contamination = st.slider(
        "Taux contamination (%)", 1, 20, 5,
        help="Pourcentage de fournisseurs supposés anormaux"
    ) / 100

    st.markdown("### 🟠 Poids du score composite")
    p_w_cluster = st.slider("Poids Cluster (%)",  10, 60, 35) / 100
    p_w_anomaly = st.slider("Poids Anomalie (%)", 10, 60, 30) / 100
    p_w_tempo   = st.slider("Poids Temporel (%)",  5, 40, 20) / 100
    p_w_dtw     = max(0.0, 1.0 - p_w_cluster - p_w_anomaly - p_w_tempo)
    st.info(f"Poids DTW (auto) : {p_w_dtw*100:.0f}%")

    st.markdown("### 🧠 SHAP")
    p_run_shap = st.checkbox(
        "Calculer SHAP", value=True,
        help="Décochez pour accélérer si le dataset est grand"
    )
    p_shap_top = st.slider("Top N variables SHAP", 5, 30, 15)

    st.markdown("### 🚦 Seuils d'alerte")
    p_seuil_vert   = st.slider("Seuil Vert / Orange",  10, 45, 29)
    p_seuil_orange = st.slider("Seuil Orange / Rouge", 40, 80, 59)

    st.markdown("---")
    st.caption(
        "Master ML & Intelligence Logistique\n"
        "Université Mohammed V — Rabat\n"
        "2024–2025"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  4. EN-TÊTE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="entete">
  <h1>🔍 Système d'Alerte Précoce — Risque Fournisseurs Maroc</h1>
  <p>Pipeline ML Non Supervisé · UMAP → HDBSCAN → Isolation Forest → VAE → SHAP</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  5. CHARGEMENT DU DATASET
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📁 Étape 1 — Chargez votre dataset")

fichier = st.file_uploader(
    "Importez votre fichier fournisseurs (Excel .xlsx ou CSV .csv)",
    type=["xlsx", "csv"],
    help="Le fichier doit contenir des colonnes numériques : OTD, Z-Score, ESG…"
)

# ── Si aucun fichier : afficher les instructions et s'arrêter ─────────────────
if fichier is None:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.info("**📁 Étape 1**\nChargez votre dataset via le bouton ci-dessus.")
    col2.info("**⚙️ Étape 2**\nAjustez les paramètres dans la barre latérale.")
    col3.info("**🚀 Étape 3**\nCliquez sur « Lancer l'analyse ».")

    st.markdown("### 📐 Format attendu")
    st.markdown("""
| Critère | Valeur attendue |
|---------|-----------------|
| Format  | Excel (.xlsx) ou CSV (.csv) |
| Lignes  | Fournisseurs (50 minimum recommandé) |
| Colonnes numériques | OTD_Pct, Altman_ZScore, Score_ESG, etc. |
| Valeurs manquantes | Acceptées — imputation automatique |
    """)
    st.stop()   # Arrête l'exécution ici si pas de fichier


# ══════════════════════════════════════════════════════════════════════════════
#  6. LECTURE DU FICHIER  (avec cache pour éviter la re-lecture à chaque clic)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def lire_fichier(contenu_bytes, nom_fichier):
    """Lit le fichier Excel ou CSV et retourne un DataFrame."""
    if nom_fichier.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(contenu_bytes))
    return pd.read_csv(io.BytesIO(contenu_bytes))


with st.spinner("Lecture du fichier…"):
    df_brut = lire_fichier(fichier.read(), fichier.name)

# Message de confirmation
st.success(
    f"✅  Fichier chargé : **{fichier.name}** — "
    f"**{df_brut.shape[0]} fournisseurs** × **{df_brut.shape[1]} colonnes**"
)

# Aperçu repliable
with st.expander("👁️  Aperçu des données brutes (5 premières lignes)"):
    st.dataframe(df_brut.head(), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Valeurs manquantes", int(df_brut.isnull().sum().sum()))
    c2.metric("Variables numériques",
              int(df_brut.select_dtypes(include=[np.number]).shape[1]))

# ── Sélection automatique des features (colonnes numériques non-sorties) ──────
features = [
    c for c in df_brut.columns
    if c not in EXCLURE
    and df_brut[c].dtype in ["float64", "int64", "float32", "int32"]
]

with st.expander(f"🔢  Variables ML sélectionnées ({len(features)} features)"):
    cols = st.columns(4)
    for i, f in enumerate(features):
        cols[i % 4].markdown(f"- `{f}`")

X_brut = df_brut[features].copy()

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  7. BOUTON DE LANCEMENT DU PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🚀 Étape 2 — Lancez le pipeline ML")

lancer = st.button(
    "▶  Lancer l'analyse complète",
    type="primary",
    use_container_width=True,
    help="Exécute les 7 étapes du pipeline ML"
)

if not lancer:
    st.info(
        "👆 Cliquez sur le bouton ci-dessus pour démarrer l'analyse.\n\n"
        "Durée estimée : **2–4 minutes** selon la taille du dataset et les paramètres SHAP."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  8. PIPELINE ML — 7 ÉTAPES
# ══════════════════════════════════════════════════════════════════════════════
barre     = st.progress(0,  text="Initialisation…")
zone_log  = st.empty()
t_debut   = time.time()

def maj_log(msg, pct, icone="⏳"):
    barre.progress(pct, text=f"{icone}  {msg}")
    zone_log.info(f"{icone}  {msg}")


# ── ÉTAPE 1 : Prétraitement ───────────────────────────────────────────────────
maj_log("Étape 1/7 — Prétraitement (imputation + normalisation)", 5)

imputer  = SimpleImputer(strategy="median")
X_imp    = imputer.fit_transform(X_brut)

scaler   = RobustScaler()
X_scale  = scaler.fit_transform(X_imp)

maj_log("Prétraitement ✅", 12, "✅")


# ── ÉTAPE 2 : PCA ─────────────────────────────────────────────────────────────
maj_log("Étape 2/7 — PCA (95 % de variance)", 15)

pca   = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scale)
n_pca = pca.n_components_
var_e = pca.explained_variance_ratio_.cumsum()[-1] * 100

maj_log(f"PCA ✅  {X_scale.shape[1]}D → {n_pca}D  ({var_e:.1f}% variance)", 22, "✅")


# ── ÉTAPE 3 : UMAP ────────────────────────────────────────────────────────────
maj_log("Étape 3/7 — UMAP 3D + 2D  (20–40 secondes…)", 25)

red_3d  = umap.UMAP(n_components=3, n_neighbors=p_neighbors,
                     min_dist=p_min_dist, random_state=42)
X_u3d   = red_3d.fit_transform(X_pca)

red_2d  = umap.UMAP(n_components=2, n_neighbors=p_neighbors,
                     min_dist=p_min_dist, random_state=42)
X_u2d   = red_2d.fit_transform(X_pca)

maj_log("UMAP ✅", 42, "✅")


# ── ÉTAPE 4 : HDBSCAN ────────────────────────────────────────────────────────
maj_log("Étape 4/7 — Clustering HDBSCAN", 46)

clusterer    = hdbscan.HDBSCAN(
    min_cluster_size = p_min_cluster,
    min_samples      = p_min_samples,
    metric           = "euclidean",
    prediction_data  = True,
)
labels       = clusterer.fit_predict(X_u3d)
n_clusters   = len(set(labels)) - (1 if -1 in labels else 0)
n_bruit      = int((labels == -1).sum())

masque       = labels != -1
if masque.sum() > 10 and len(set(labels[masque])) > 1:
    sil = silhouette_score(X_u3d[masque], labels[masque])
    dbi = davies_bouldin_score(X_u3d[masque], labels[masque])
else:
    sil, dbi = 0.0, 9.9

maj_log(
    f"HDBSCAN ✅  {n_clusters} clusters + {n_bruit} anomalies "
    f"(Silhouette={sil:.3f})",
    57, "✅"
)


# ── ÉTAPE 5 : Isolation Forest ────────────────────────────────────────────────
maj_log("Étape 5/7 — Isolation Forest", 60)

iso      = IsolationForest(
    n_estimators  = 200,
    contamination = p_contamination,
    random_state  = 42,
    n_jobs        = -1,
)
iso.fit(X_scale)
if_brut  = iso.decision_function(X_scale)
if_score = 1 - (if_brut - if_brut.min()) / (if_brut.max() - if_brut.min())
n_anom   = int((iso.predict(X_scale) == -1).sum())

maj_log(f"Isolation Forest ✅  {n_anom} anomalies détectées", 68, "✅")


# ── ÉTAPE 6 : VAE ─────────────────────────────────────────────────────────────
maj_log("Étape 6/7 — Variational Autoencoder (VAE)", 70)


class VAE(nn.Module):
    """Variational Autoencoder pour la détection d'anomalies."""

    def __init__(self, dim_entree, dim_latent=8):
        super().__init__()
        h = max(dim_entree // 2, dim_latent * 4)

        self.encodeur = nn.Sequential(
            nn.Linear(dim_entree, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1),
            nn.Dropout(0.2), nn.Linear(h, h // 2), nn.LeakyReLU(0.1),
        )
        self.fc_mu     = nn.Linear(h // 2, dim_latent)
        self.fc_logvar = nn.Linear(h // 2, dim_latent)
        self.decodeur  = nn.Sequential(
            nn.Linear(dim_latent, h // 2), nn.LeakyReLU(0.1),
            nn.Linear(h // 2, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1),
            nn.Linear(h, dim_entree),
        )

    def reparametriser(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        h      = self.encodeur(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z      = self.reparametriser(mu, logvar)
        return self.decodeur(z), mu, logvar


# Entraîner uniquement sur les fournisseurs normaux
X_normal  = torch.FloatTensor(X_scale[labels != -1])
chargeur  = DataLoader(TensorDataset(X_normal), batch_size=32, shuffle=True)
vae_model = VAE(X_scale.shape[1])
optimiseur= optim.Adam(vae_model.parameters(), lr=1e-3)

vae_model.train()
for epoch in range(60):
    for (lot,) in chargeur:
        optimiseur.zero_grad()
        rec, mu, lv = vae_model(lot)
        perte = (
            nn.functional.mse_loss(rec, lot, reduction="sum")
            - 5e-4 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        )
        perte.backward()
        optimiseur.step()

vae_model.eval()
with torch.no_grad():
    rec_tout, _, _ = vae_model(torch.FloatTensor(X_scale))
    erreur_vae = nn.functional.mse_loss(
        rec_tout, torch.FloatTensor(X_scale), reduction="none"
    ).mean(dim=1).numpy()

vae_score    = (erreur_vae - erreur_vae.min()) / (erreur_vae.max() - erreur_vae.min())
anom_compo   = (if_score + vae_score) / 2

maj_log("VAE ✅", 80, "✅")


# ── ÉTAPE 7 : Score composite ────────────────────────────────────────────────
maj_log("Étape 7/7 — Score composite et niveaux d'alerte", 84)

# Risque moyen IF par cluster
risque_cluster = {
    cl: float(if_score[labels == cl].mean()) if cl != -1 else 1.0
    for cl in set(labels)
}
composante_cl = np.array([risque_cluster[cl] for cl in labels])

# Dérive temporelle (PSI + PELT + Tendance OTD)
def get_col(nom, defaut=0.0):
    if nom in df_brut.columns:
        return df_brut[nom].fillna(0).astype(float).values
    return np.full(len(df_brut), defaut)

derive = (
    0.4 * np.clip(get_col("PSI_Score") / 0.5, 0, 1)
    + 0.4 * get_col("Changepoint_PELT")
    + 0.2 * np.clip(np.abs(get_col("Tendance_OTD_6M")) / 10, 0, 1)
)

# Score final 0–100
score_01  = (
    p_w_cluster * composante_cl
    + p_w_anomaly * anom_compo
    + p_w_tempo   * derive
    + p_w_dtw     * vae_score
)
score_100 = np.clip(score_01 * 100, 0, 100)


def niveau_alerte(s):
    if s <= p_seuil_vert:   return "🟢 Vert"
    if s <= p_seuil_orange: return "🟠 Orange"
    return "🔴 Rouge"


alertes  = np.array([niveau_alerte(s) for s in score_100])
n_vert   = int((alertes == "🟢 Vert").sum())
n_orange = int((alertes == "🟠 Orange").sum())
n_rouge  = int((alertes == "🔴 Rouge").sum())

# ── DataFrame résultat complet ────────────────────────────────────────────────
df_res = df_brut.copy()
df_res["Cluster_HDBSCAN"]      = labels
df_res["Score_IF"]             = np.round(if_score * 100, 1)
df_res["Score_VAE"]            = np.round(vae_score * 100, 1)
df_res["Score_Anomalie_Comp"]  = np.round(anom_compo * 100, 1)
df_res["Score_Risque_ML"]      = np.round(score_100, 1)
df_res["Alerte_ML"]            = alertes
df_res["Priorite_ML"]          = [
    "IMMÉDIAT"    if a == "🔴 Rouge"  else
    "SURVEILLANCE" if a == "🟠 Orange" else
    "STANDARD"
    for a in alertes
]

t_total = time.time() - t_debut
maj_log(f"Pipeline complet ✅  ({t_total:.0f} s)", 92, "✅")


# ── SHAP (optionnel) ──────────────────────────────────────────────────────────
shap_values = None
shap_df     = None

if p_run_shap:
    maj_log("Calcul SHAP…", 94)
    try:
        explic      = shap.TreeExplainer(iso)
        shap_values = explic.shap_values(X_scale)

        moy_abs   = np.abs(shap_values).mean(axis=0)
        total_s   = moy_abs.sum()
        shap_df   = pd.DataFrame({
            "Variable" : features,
            "SHAP_abs" : moy_abs,
            "SHAP_pct" : (moy_abs / total_s * 100).round(1),
        }).sort_values("SHAP_abs", ascending=False).reset_index(drop=True)

        # Ajouter la top variable SHAP à chaque fournisseur
        df_res["Top_Variable_SHAP"] = [
            features[int(np.argmax(np.abs(shap_values[i])))]
            for i in range(len(df_brut))
        ]
        df_res["Top_Contrib_SHAP_Pct"] = [
            round(
                float(np.abs(shap_values[i]).max())
                / (float(np.abs(shap_values[i]).sum()) + 1e-10) * 100, 1
            )
            for i in range(len(df_brut))
        ]

        maj_log("SHAP ✅", 98, "✅")

    except Exception as e:
        st.warning(f"SHAP non disponible : {e}")

barre.progress(100, text="✅  Analyse terminée !")
zone_log.success(
    f"✅  Pipeline complet en {t_total:.0f} secondes — "
    f"{len(df_brut)} fournisseurs analysés"
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  9. TABLEAU DE BORD — CARTES KPI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Étape 3 — Résultats")

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.markdown(f"""<div class="kpi-card">
  <div class="kpi-val">{len(df_brut)}</div>
  <div class="kpi-lab">Fournisseurs</div>
</div>""", unsafe_allow_html=True)

k2.markdown(f"""<div class="kpi-card">
  <div class="kpi-val">{n_clusters}</div>
  <div class="kpi-lab">Clusters ML</div>
</div>""", unsafe_allow_html=True)

k3.markdown(f"""<div class="kpi-card" style="border-color:#4CAF50">
  <div class="kpi-val" style="color:#375623">{n_vert}</div>
  <div class="kpi-lab">🟢 Alertes Vertes</div>
</div>""", unsafe_allow_html=True)

k4.markdown(f"""<div class="kpi-card" style="border-color:#FF9800">
  <div class="kpi-val" style="color:#C55A11">{n_orange}</div>
  <div class="kpi-lab">🟠 Alertes Orange</div>
</div>""", unsafe_allow_html=True)

k5.markdown(f"""<div class="kpi-card" style="border-color:#C00000">
  <div class="kpi-val" style="color:#C00000">{n_rouge}</div>
  <div class="kpi-lab">🔴 Alertes Rouges</div>
</div>""", unsafe_allow_html=True)

k6.markdown(f"""<div class="kpi-card">
  <div class="kpi-val">{score_100.mean():.1f}</div>
  <div class="kpi-lab">Score moyen /100</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Métriques de validation ML
with st.expander("📏  Métriques de validation ML"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette Score", f"{sil:.3f}",
              delta="✅ > 0.50" if sil > 0.50 else "⚠️ < 0.50")
    m2.metric("Davies-Bouldin",   f"{dbi:.3f}",
              delta="✅ < 1.50" if dbi < 1.50 else "⚠️ > 1.50")
    m3.metric("Anomalies IF",     n_anom)
    m4.metric("Anomalies HDBSCAN (bruit)", n_bruit)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  10. ONGLETS DE VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
ong1, ong2, ong3, ong4, ong5 = st.tabs([
    "📌 Clusters UMAP",
    "📊 Distribution alertes",
    "🔬 Scores anomalie",
    "🧠 SHAP",
    "📋 Tableau fournisseurs",
])

# ── Données pour les graphiques ───────────────────────────────────────────────
df_plot = df_brut.copy()
df_plot["UMAP1"]        = X_u2d[:, 0]
df_plot["UMAP2"]        = X_u2d[:, 1]
df_plot["Cluster"]      = [
    "Anomalie" if c == -1 else f"Cluster {c}" for c in labels
]
df_plot["Alerte"]       = alertes
df_plot["Score"]        = np.round(score_100, 1)
df_plot["Score_IF_100"] = np.round(if_score * 100, 1)

hover_cols = [c for c in ["ID_Fournisseur", "Secteur", "Region_Maroc"]
              if c in df_plot.columns]


# ── ONGLET 1 : Projection UMAP ────────────────────────────────────────────────
with ong1:
    st.subheader("Projection UMAP 2D — Profils de risque identifiés")

    choix_couleur = st.radio(
        "Colorier par :", ["Alerte", "Cluster", "Score_IF_100"],
        horizontal=True
    )

    if choix_couleur == "Alerte":
        fig = px.scatter(
            df_plot, x="UMAP1", y="UMAP2", color="Alerte",
            color_discrete_map=COULEURS_ALERTES,
            hover_data=hover_cols + ["Score"],
            title="Projection UMAP 2D — Niveaux d'alerte",
            template="plotly_white", height=520,
        )
    elif choix_couleur == "Cluster":
        fig = px.scatter(
            df_plot, x="UMAP1", y="UMAP2", color="Cluster",
            hover_data=hover_cols + ["Score"],
            title="Projection UMAP 2D — Clusters HDBSCAN",
            template="plotly_white", height=520,
        )
    else:
        fig = px.scatter(
            df_plot, x="UMAP1", y="UMAP2", color="Score_IF_100",
            color_continuous_scale="RdYlGn_r",
            hover_data=hover_cols + ["Score"],
            title="Projection UMAP 2D — Score Isolation Forest",
            template="plotly_white", height=520,
        )

    fig.update_traces(
        marker=dict(size=8, opacity=0.82, line=dict(width=0.5, color="white"))
    )
    fig.update_layout(font=dict(family="Arial", size=11),
                      title_font=dict(size=14, color="#1F3864"))
    st.plotly_chart(fig, use_container_width=True)


# ── ONGLET 2 : Distribution alertes ──────────────────────────────────────────
with ong2:
    st.subheader("Distribution des scores et répartition des alertes")

    col_a, col_b = st.columns(2)

    # Histogramme du score composite
    fig_hist = px.histogram(
        x=score_100, nbins=40,
        color_discrete_sequence=["#2E5496"],
        labels={"x": "Score de risque (0–100)", "count": "Fréquence"},
        title="Distribution du score de risque composite",
        template="plotly_white",
    )
    fig_hist.add_vline(x=p_seuil_vert,   line_dash="dash", line_color="#4CAF50",
                        annotation_text=f"Vert ({p_seuil_vert})")
    fig_hist.add_vline(x=p_seuil_orange, line_dash="dash", line_color="#FF9800",
                        annotation_text=f"Orange ({p_seuil_orange})")
    fig_hist.add_vline(x=score_100.mean(), line_dash="dot", line_color="#C00000",
                        annotation_text=f"Moy {score_100.mean():.1f}")
    col_a.plotly_chart(fig_hist, use_container_width=True)

    # Camembert
    fig_pie = px.pie(
        names=["🟢 Vert", "🟠 Orange", "🔴 Rouge"],
        values=[n_vert, n_orange, n_rouge],
        color=["🟢 Vert", "🟠 Orange", "🔴 Rouge"],
        color_discrete_map=COULEURS_ALERTES,
        title="Répartition des alertes",
        hole=0.4,
    )
    fig_pie.update_traces(textinfo="label+percent+value", textposition="outside")
    col_b.plotly_chart(fig_pie, use_container_width=True)

    # Boxplots par cluster
    df_box = pd.DataFrame({
        "Cluster": ["Anomalie" if c == -1 else f"C{c}" for c in labels],
        "Score"  : score_100,
    })
    fig_box = px.box(
        df_box, x="Cluster", y="Score", color="Cluster",
        title="Score de risque par cluster HDBSCAN",
        template="plotly_white", height=400, points="outliers",
    )
    fig_box.add_hline(y=p_seuil_vert,   line_dash="dash", line_color="#4CAF50")
    fig_box.add_hline(y=p_seuil_orange, line_dash="dash", line_color="#FF9800")
    fig_box.update_layout(showlegend=False,
                           font=dict(family="Arial", size=11))
    st.plotly_chart(fig_box, use_container_width=True)


# ── ONGLET 3 : Scores anomalie ────────────────────────────────────────────────
with ong3:
    st.subheader("Scores d'anomalie — Isolation Forest vs VAE")

    col_c, col_d = st.columns(2)

    fig_if = px.scatter(
        x=X_u2d[:, 0], y=X_u2d[:, 1],
        color=if_score * 100,
        color_continuous_scale="RdYlGn_r",
        labels={"x": "UMAP1", "y": "UMAP2", "color": "Score IF"},
        title="Score Isolation Forest (0–100)",
        template="plotly_white", height=420,
    )
    fig_if.update_traces(marker=dict(size=7, opacity=0.8))
    col_c.plotly_chart(fig_if, use_container_width=True)

    fig_vae = px.scatter(
        x=X_u2d[:, 0], y=X_u2d[:, 1],
        color=vae_score * 100,
        color_continuous_scale="RdYlGn_r",
        labels={"x": "UMAP1", "y": "UMAP2", "color": "Score VAE"},
        title="Score VAE — Erreur de Reconstruction (0–100)",
        template="plotly_white", height=420,
    )
    fig_vae.update_traces(marker=dict(size=7, opacity=0.8))
    col_d.plotly_chart(fig_vae, use_container_width=True)

    # Scatter IF vs VAE
    fig_corr = px.scatter(
        x=if_score * 100, y=vae_score * 100,
        color=alertes,
        color_discrete_map=COULEURS_ALERTES,
        labels={"x": "Score IF (×100)", "y": "Score VAE (×100)"},
        title="Corrélation IF vs VAE — Les deux détecteurs concordent-ils ?",
        template="plotly_white", height=420,
    )
    fig_corr.add_shape(
        type="line", x0=0, y0=0, x1=100, y1=100,
        line=dict(dash="dash", color="gray", width=1)
    )
    fig_corr.add_annotation(
        x=70, y=75,
        text="Diagonale : IF = VAE",
        showarrow=False,
        font=dict(color="gray", size=10)
    )
    fig_corr.update_traces(
        marker=dict(size=8, opacity=0.75, line=dict(width=0.4, color="white"))
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(
        "Points au-dessus de la diagonale : le VAE détecte plus d'anomalie que l'IF "
        "→ anomalies de structure complexe non visibles en haute dimension."
    )


# ── ONGLET 4 : SHAP ──────────────────────────────────────────────────────────
with ong4:
    st.subheader("Analyse SHAP — Importance globale des variables")

    if shap_df is None:
        st.info(
            "SHAP est désactivé. Cochez **Calculer SHAP** dans la barre latérale "
            "et relancez l'analyse."
        )
    else:
        top_n     = shap_df.head(p_shap_top)
        couleurs  = [
            "#C00000" if v > 10 else "#C55A11" if v > 5 else "#2E5496"
            for v in top_n["SHAP_pct"]
        ]

        fig_shap = go.Figure(go.Bar(
            x=top_n["SHAP_pct"],
            y=top_n["Variable"],
            orientation="h",
            marker_color=couleurs,
            text=[f"{v:.1f}%" for v in top_n["SHAP_pct"]],
            textposition="outside",
        ))
        fig_shap.update_layout(
            title=dict(
                text=f"Top {p_shap_top} variables — Importance SHAP globale",
                font=dict(size=14, color="#1F3864")
            ),
            xaxis_title="Contribution (%)",
            yaxis=dict(autorange="reversed"),
            height=max(420, p_shap_top * 27),
            template="plotly_white",
            font=dict(family="Arial", size=11),
            margin=dict(l=10, r=80, t=50, b=40),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Tableau SHAP
        st.markdown("**Détail des contributions :**")
        st.dataframe(
            shap_df.head(p_shap_top)[["Variable", "SHAP_pct"]]
            .rename(columns={"SHAP_pct": "Contribution (%)"}),
            use_container_width=True,
            height=350,
        )


# ── ONGLET 5 : Tableau fournisseurs ──────────────────────────────────────────
with ong5:
    st.subheader("Tableau complet des fournisseurs")

    # Filtres
    f1, f2, f3 = st.columns(3)
    sel_alerte = f1.multiselect(
        "Filtrer par alerte",
        ["🔴 Rouge", "🟠 Orange", "🟢 Vert"],
        default=["🔴 Rouge", "🟠 Orange"],
    )
    score_min = f2.slider("Score minimum", 0, 100, 0)
    score_max = f3.slider("Score maximum", 0, 100, 100)

    df_filtre = df_res[
        df_res["Alerte_ML"].isin(sel_alerte)
        & (df_res["Score_Risque_ML"] >= score_min)
        & (df_res["Score_Risque_ML"] <= score_max)
    ].sort_values("Score_Risque_ML", ascending=False)

    st.info(f"**{len(df_filtre)} fournisseurs** affichés sur {len(df_res)} total")

    cols_affich = [c for c in [
        "ID_Fournisseur", "Secteur", "Region_Maroc",
        "Alerte_ML", "Score_Risque_ML", "Score_IF", "Score_VAE",
        "Cluster_HDBSCAN", "Priorite_ML",
        "Top_Variable_SHAP", "Top_Contrib_SHAP_Pct",
    ] if c in df_filtre.columns]

    st.dataframe(
        df_filtre[cols_affich],
        use_container_width=True,
        height=420,
    )

    # Export CSV
    csv = df_res.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️  Télécharger tous les résultats (CSV)",
        data=csv,
        file_name="resultats_alertes_fournisseurs.csv",
        mime="text/csv",
        type="primary",
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  11. RAPPORT INDIVIDUEL FOURNISSEUR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📄 Étape 4 — Rapport individuel fournisseur")

# Sélecteur
if "ID_Fournisseur" in df_res.columns:
    options_id = df_res["ID_Fournisseur"].tolist()
else:
    options_id = list(df_res.index)

id_choisi = st.selectbox(
    "Sélectionnez un fournisseur",
    options=options_id,
    help="Choisissez le fournisseur pour lequel générer le rapport SHAP détaillé",
)

# Récupération de la ligne
if "ID_Fournisseur" in df_res.columns:
    idx   = df_res[df_res["ID_Fournisseur"] == id_choisi].index[0]
else:
    idx   = id_choisi

ligne     = df_res.loc[idx]
score_val = float(ligne["Score_Risque_ML"])
alerte_v  = str(ligne["Alerte_ML"])
f_id      = ligne.get("ID_Fournisseur", f"Index {idx}")
f_sect    = ligne.get("Secteur",       "N/A")
f_reg     = ligne.get("Region_Maroc",  "N/A")
f_cl      = int(ligne["Cluster_HDBSCAN"])
f_cl_lab  = "Anomalie" if f_cl == -1 else f"Cluster {f_cl}"

# ── Bandeau coloré ────────────────────────────────────────────────────────────
css_al = (
    "alerte-rouge"  if "Rouge"  in alerte_v else
    "alerte-orange" if "Orange" in alerte_v else
    "alerte-vert"
)
st.markdown(
    f'<div class="{css_al}">'
    f'🏭 <b>{f_id}</b> &nbsp;·&nbsp; {f_sect} &nbsp;·&nbsp; {f_reg}'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="{css_al}">'
    f'Niveau d\'alerte : <b>{alerte_v}</b> &nbsp;·&nbsp; '
    f'Score composite : <b>{score_val:.1f}/100</b> &nbsp;·&nbsp; '
    f'Profil ML : <b>{f_cl_lab}</b>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ── Métriques ─────────────────────────────────────────────────────────────────
r1, r2, r3, r4 = st.columns(4)
r1.metric("Score de risque",     f"{score_val:.1f}/100")
r2.metric("Score IF",            f"{ligne['Score_IF']:.1f}/100")
r3.metric("Score VAE",           f"{ligne['Score_VAE']:.1f}/100")
r4.metric("Score Anomalie Comp", f"{ligne['Score_Anomalie_Comp']:.1f}/100")

# ── Rapport SHAP individuel ───────────────────────────────────────────────────
if shap_values is not None:
    st.markdown("#### 🧠 Facteurs de risque (analyse SHAP locale)")

    pos_dans_tableau = df_res.index.get_loc(idx)
    sv_ind           = shap_values[pos_dans_tableau]
    top5             = np.argsort(np.abs(sv_ind))[::-1][:5]
    total_sv         = np.abs(sv_ind).sum() + 1e-10

    lignes_html = ""
    for rang, j in enumerate(top5, 1):
        contrib   = abs(sv_ind[j]) / total_sv * 100
        direction = "↑ Aggrave le risque" if sv_ind[j] > 0 else "↓ Réduit le risque"
        couleur_d = "#C00000" if sv_ind[j] > 0 else "#375623"
        var_nom   = features[j]
        try:
            valeur = round(float(df_brut.loc[idx, var_nom]), 3)
        except Exception:
            valeur = "N/A"
        bar_w = int(contrib * 3)
        bar_c = "#C00000" if sv_ind[j] > 0 else "#375623"
        bg_l  = "#FFF8F8" if sv_ind[j] > 0 else "#F8FFF8"

        lignes_html += (
            f"<tr style='background:{bg_l};border-bottom:1px solid #eee'>"
            f"<td style='padding:8px;text-align:center;font-weight:700'>{rang}</td>"
            f"<td style='padding:8px;font-weight:700;color:#1F3864'>{var_nom}</td>"
            f"<td style='padding:8px;text-align:center'><code>{valeur}</code></td>"
            f"<td style='padding:8px;text-align:center;font-weight:700'>{contrib:.1f}%</td>"
            f"<td style='padding:8px;color:{couleur_d};font-weight:600'>{direction}</td>"
            f"<td style='padding:8px'>"
            f"<span class='shap-bar' style='width:{bar_w}px;background:{bar_c}'></span>"
            f"</td></tr>"
        )

    tableau_shap = (
        "<table style='width:100%;border-collapse:collapse;"
        "font-family:Arial;font-size:13px'>"
        "<tr style='background:#1F3864;color:white'>"
        "<th style='padding:8px'>Rang</th>"
        "<th>Variable</th>"
        "<th>Valeur brute</th>"
        "<th>Contribution</th>"
        "<th>Direction</th>"
        "<th>Visualisation</th>"
        "</tr>"
        + lignes_html
        + "</table>"
    )
    st.markdown(tableau_shap, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

else:
    st.info(
        "Activez **Calculer SHAP** dans la barre latérale "
        "pour afficher l'analyse détaillée."
    )

# ── Recommandation ────────────────────────────────────────────────────────────
if "Rouge" in alerte_v:
    rec_css  = "alerte-rouge"
    rec_txt  = (
        "🔴 <b>ACTION IMMÉDIATE REQUISE</b> — Score critique ≥ 60. "
        "Activer le plan de contingence dans les 2 semaines. "
        "Qualifier un fournisseur alternatif en urgence."
    )
elif "Orange" in alerte_v:
    rec_css  = "alerte-orange"
    rec_txt  = (
        "🟠 <b>SURVEILLANCE RENFORCÉE</b> — Score modéré 30–59. "
        "Planifier un audit fournisseur dans le mois. "
        "Augmenter la fréquence de suivi à hebdomadaire."
    )
else:
    rec_css  = "alerte-vert"
    rec_txt  = (
        "🟢 <b>SURVEILLANCE STANDARD</b> — Score faible ≤ 29. "
        "Monitoring mensuel standard. "
        "Aucune action corrective immédiate requise."
    )

st.markdown(
    f'<div class="{rec_css}" style="margin-top:8px">'
    f'<b>Recommandation :</b> {rec_txt}'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ── Graphique radar ────────────────────────────────────────────────────────────
vars_radar = [v for v in [
    "OTD_Pct", "Altman_ZScore", "Score_ESG",
    "Stabilite_Politique", "Score_IF", "Current_Ratio", "Dependance_Mono",
] if v in df_res.columns]

if len(vars_radar) >= 4:
    with st.expander("📡  Profil radar du fournisseur (valeurs normalisées 0–100)"):
        valeurs_norm = []
        for v in vars_radar:
            serie = df_res[v].dropna()
            v_min, v_max = serie.min(), serie.max()
            val = float(df_res.loc[idx, v])
            valeurs_norm.append(
                float(np.clip((val - v_min) / (v_max - v_min + 1e-10), 0, 1) * 100)
            )

        fig_radar = go.Figure(go.Scatterpolar(
            r     = valeurs_norm + [valeurs_norm[0]],
            theta = vars_radar  + [vars_radar[0]],
            fill  = "toself",
            fillcolor = "rgba(46,84,150,0.15)",
            line  = dict(color="#2E5496", width=2),
            name  = str(f_id),
        ))
        fig_radar.update_layout(
            polar  = dict(radialaxis=dict(visible=True, range=[0, 100])),
            title  = dict(
                text=f"Profil normalisé — {f_id}",
                font=dict(color="#1F3864", size=13)
            ),
            height = 420,
            showlegend = False,
            font   = dict(family="Arial", size=10),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")
st.caption(
    "Système d'Alerte Précoce — Risque Fournisseurs Maroc  |  "
    "Université Mohammed V Rabat — Master ML & Intelligence Logistique — 2024–2025"
)
