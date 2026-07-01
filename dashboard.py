import sys
from pathlib import Path
import subprocess
import pandas as pd
import streamlit as st
import importlib

# Injection du répertoire parent dans le path system
sys.path.append(str(Path(__file__).resolve().parent))

# ==============================================================================
# CHARGEMENT & RECHARGEMENT STRICT DE LA CONFIGURATION
# ==============================================================================
# Ce bloc force Python à vider sa mémoire temporaire et à relire physiquement
# le fichier ldm_config.py sur le disque dur. C'est ce qui garantit que tu
# retrouveras tes dernières valeurs saisies lors de ta précédente session.
import configs.ldm_config as ldm_config
importlib.reload(ldm_config)
from configs import ldm_config  # Force la mise à jour des variables dans tout le script

from src.ldm_dataset import get_dataloader, summarize_dataset
from src.utils_common import set_seed

st.set_page_config(
    page_title="Dashboard LDM - Supervision Académique",
    layout="wide",
)

set_seed(42)


def dataset_source_disponible() -> bool:
    if not ldm_config.DATA_DIR.exists():
        return False

    extensions = ("*.jpeg", "*.jpg", "*.png")
    return any(
        next(ldm_config.DATA_DIR.rglob(extension), None) is not None
        for extension in extensions
    )


DATASET_SOURCE_DISPONIBLE = dataset_source_disponible()


# ==============================================================================
# BARRE LATÉRALE : GESTION DE LA CONFIGURATION (VERSION VULGARISÉE)
# ==============================================================================
st.sidebar.title("🎛️ Configuration du Framework")
st.sidebar.markdown(
    """
    Les ajustements modifient directement le fichier `ldm_config.py`. 
    Ils sont appliqués sur ton disque dur et seront conservés pour tes prochaines sessions.
    """
)

# --- SECTION 1 : L'AUTOENCODEUR (Le Compresseur) ---
st.sidebar.header("🎨 Phase 1 : L'Autoencodeur (AE)")
st.sidebar.caption("Objectif : Apprendre à compresser les images médicales lourdes en un format miniature sans perdre les détails cliniques importants.")

ae_batch = st.sidebar.number_input(
    "📦 Taille du Batch (AE_BATCH_SIZE)", 
    min_value=1, max_value=64, value=int(ldm_config.AE_BATCH_SIZE), step=1,
    help="Le nombre d'images que l'IA regarde à la fois avant de corriger ses erreurs. Plus ce chiffre est grand, plus le calcul est stable, mais cela demande plus de mémoire à ton ordinateur."
)
ae_epochs = st.sidebar.number_input(
    "🔄 Nombre d'Époques (AE_EPOCHS)", 
    min_value=1, max_value=500, value=int(ldm_config.AE_EPOCHS), step=1,
    help="Le nombre de fois que l'IA va réviser l'intégralité de ton jeu d'images médicales. Plus elle révise, meilleure elle devient (jusqu'à une certaine limite)."
)
ae_lr = st.sidebar.number_input(
    "⚡ Vitesse d'apprentissage (AE_LR)", 
    min_value=1e-6, max_value=1e-1, value=float(ldm_config.AE_LR), format="%.1e",
    help="La taille des pas que fait l'IA pour apprendre. Trop grand : elle rate sa cible et devient instable. Trop petit : elle mettra des semaines à comprendre."
)
ae_beta = st.sidebar.number_input(
    "⚖️ Facteur de Régularisation (AE_BETA_KL)", 
    min_value=1e-7, max_value=1e-2, value=float(ldm_config.AE_BETA_KL), format="%.1e",
    help="Contrôle l'organisation des données compressées. Il force l'IA à ranger les caractéristiques des images de façon harmonieuse et fluide, ce qui facilitera la création de nouvelles images par la suite."
)

st.sidebar.markdown("---")

# --- SECTION 2 : LE MODÈLE DE DIFFUSION (Le Générateur) ---
st.sidebar.header("🌌 Phase 2 : Modèle de Diffusion (LDM)")
st.sidebar.caption("Objectif : Apprendre à créer de nouvelles images médicales réalistes à partir du bruit (du faux pixel aléatoire) dans l'espace compressé.")

diff_batch = st.sidebar.number_input(
    "📦 Taille du Batch (DIFF_BATCH_SIZE)", 
    min_value=1, max_value=64, value=int(ldm_config.DIFF_BATCH_SIZE), step=1,
    help="Le nombre de représentations miniatures d'images traitées en même temps pendant l'entraînement du générateur."
)
diff_epochs = st.sidebar.number_input(
    "🔄 Nombre d'Époques (DIFF_EPOCHS)", 
    min_value=1, max_value=500, value=int(ldm_config.DIFF_EPOCHS), step=1,
    help="Le nombre de cycles complets d'entraînement pour que le modèle de diffusion maîtrise l'art de créer des textures médicales."
)
diff_lr = st.sidebar.number_input(
    "⚡ Vitesse d'apprentissage (DIFF_LR)", 
    min_value=1e-6, max_value=1e-1, value=float(ldm_config.DIFF_LR), format="%.1e",
    help="La vitesse à laquelle le générateur ajuste ses calculs pour corriger ses défauts visuels et rendre les tissus plus réalistes."
)
ddpm_steps = st.sidebar.number_input(
    "⏳ Étapes de Bruit Totales (DDPM_TIMESTEPS)", 
    min_value=10, max_value=2000, value=int(ldm_config.DDPM_TIMESTEPS), step=50,
    help="Le nombre total d'étapes utilisées pour détruire mathématiquement l'image avec du bruit (à l'aller). Cela définit l'échelle de temps que l'IA doit apprendre à inverser."
)
infer_steps = st.sidebar.number_input(
    "🎬 Étapes de Génération (DDPM_INFERENCE_STEPS)", 
    min_value=10, max_value=2000, value=int(ldm_config.DDPM_INFERENCE_STEPS), step=10,
    help="Le nombre de pas de nettoyage effectués pour fabriquer une nouvelle image à partir de rien. Plus ce nombre est élevé, plus l'image finale sera fine et détaillée, mais plus la génération prendra du temps."
)

st.sidebar.markdown("---")

# --- SECTION 3 : LE CLASSIFIEUR ET VOLUME DE SYNTHÈSE (Le Diagnostic & Objectifs) ---
st.sidebar.header("🎯 Phase 3 : Classification & Volume de Synthèse")
st.sidebar.caption("Objectif : Entraîner l'IA de diagnostic et configurer la quantité de fausses images médicales à générer pour booster les performances.")

class_batch = st.sidebar.number_input(
    "📦 Taille du Batch Classifieur (CLASSIFIER_BATCH_SIZE)", 
    min_value=1, max_value=64, value=int(ldm_config.CLASSIFIER_BATCH_SIZE), step=1,
    help="Le nombre de radiographies analysées simultanément par l'IA de diagnostic pendant son entraînement."
)
class_epochs = st.sidebar.number_input(
    "🔄 Époques Classifieur (CLASSIFIER_EPOCHS)", 
    min_value=1, max_value=200, value=int(ldm_config.CLASSIFIER_EPOCHS), step=1,
    help="Le nombre de révisions complètes des images par l'IA de diagnostic pour apprendre à repérer la pneumonie."
)
class_lr = st.sidebar.number_input(
    "⚡ Vitesse Apprentissage Classifieur (CLASSIFIER_LR)", 
    min_value=1e-6, max_value=1e-1, value=float(ldm_config.CLASSIFIER_LR), format="%.1e",
    help="Le taux d'attention accordé aux erreurs de diagnostic. Un réglage précis permet au classifieur de devenir très sensible aux opacités pulmonaires sans faire de faux diagnostics."
)

num_normal = st.sidebar.number_input(
    "🍏 Volume Normal à Générer (NUM_SYNTHETIC_NORMAL)", 
    min_value=0, max_value=5000, value=int(ldm_config.NUM_SYNTHETIC_NORMAL), step=10,
    help="Le nombre total de fausses radiographies de poumons SAINS (Normaux) que tu demandes au modèle de fabriquer de toutes pièces."
)
num_pneumo = st.sidebar.number_input(
    "🍎 Volume Pathologique à Générer (NUM_SYNTHETIC_PNEUMONIA)", 
    min_value=0, max_value=5000, value=int(ldm_config.NUM_SYNTHETIC_PNEUMONIA), step=10,
    help="Le nombre total de fausses radiographies de poumons ATTEINTS DE PNEUMONIE que tu demandes au modèle de fabriquer pour enrichir ton jeu de données."
)
max_qual = st.sidebar.number_input(
    "📊 Échantillons d'Évaluation (MAX_QUALITY_EVAL_IMAGES)", 
    min_value=5, max_value=500, value=int(ldm_config.MAX_QUALITY_EVAL_IMAGES), step=5,
    help="Le nombre d'images extraites pour les tests mathématiques de qualité (calculs MSE et SSIM). Limiter ce nombre permet de faire un contrôle qualité rapide sans bloquer l'ordinateur."
)

# Fonction de réécriture du fichier ldm_config.py
def sauvegarder_configuration():
    contenu_config = f"""from pathlib import Path

SEED = 42

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "raw" / "chest_xray"

OUTPUT_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
RECON_DIR = OUTPUT_DIR / "reconstructions"
SAMPLES_DIR = OUTPUT_DIR / "ldm_samples"
CLASSIF_DIR = OUTPUT_DIR / "classification"

MODEL_DIR = BASE_DIR / "models"
AE_DIR = MODEL_DIR / "autoencoder"
DIFF_DIR = MODEL_DIR / "diffusion"
CLASSIFIER_DIR = MODEL_DIR / "classifier"

IMAGE_SIZE = (128, 128)
IN_CHANNELS = 1
LATENT_CHANNELS = 3
LATENT_SIZE = (32, 32)

# --- Hyperparamètres ajustés dynamiquement via l'interface ---
AE_BATCH_SIZE = {ae_batch}
AE_EPOCHS = {ae_epochs}
AE_LR = {ae_lr}
AE_BETA_KL = {ae_beta}

DIFF_BATCH_SIZE = {diff_batch}
DIFF_EPOCHS = {diff_epochs}
DIFF_LR = {diff_lr}
DDPM_TIMESTEPS = {ddpm_steps}
DDPM_INFERENCE_STEPS = {infer_steps}

CLASSIFIER_BATCH_SIZE = {class_batch}
CLASSIFIER_EPOCHS = {class_epochs}
CLASSIFIER_LR = {class_lr}

NUM_SYNTHETIC_NORMAL = {num_normal}
NUM_SYNTHETIC_PNEUMONIA = {num_pneumo}
MAX_QUALITY_EVAL_IMAGES = {max_qual}

for directory in [
    OUTPUT_DIR, METRICS_DIR, FIGURES_DIR, RECON_DIR, SAMPLES_DIR,
    CLASSIF_DIR, MODEL_DIR, AE_DIR, DIFF_DIR, CLASSIFIER_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
"""
    chemin_config = Path("configs/ldm_config.py")
    with open(chemin_config, "w", encoding="utf-8") as f:
        f.write(contenu_config)

# Bouton d'enregistrement avec rechargement à chaud
st.sidebar.markdown("---")
if st.sidebar.button("💾 Enregistrer et appliquer la configuration", use_container_width=True):
    sauvegarder_configuration()
    importlib.reload(ldm_config)
    st.sidebar.success("✅ Fichier ldm_config.py mis à jour avec succès.")


# ==========================================
# CORPS DU TABLEAU DE BORD
# ==========================================
st.title("🔬 Génération d’Images Médicales Synthétiques via Modèles Diffusifs Latents")
st.subheader("Plateforme de pilotage, d'analyse mathématique et d'évaluation clinique du pipeline génératif")

if not DATASET_SOURCE_DISPONIBLE:
    st.warning(
        "Mode démonstration : les données sources complètes ne sont pas incluses dans le dépôt GitHub. "
        "Les résultats pré-calculés restent consultables dans les onglets d'analyse, mais les actions qui relisent "
        "le dataset brut sont désactivées."
    )

def executer_script(nom_script, description):
    st.info(f"⏳ **Framework Engine :** {description} en cours d'exécution sur le périphérique actif...")
    barre_progression = st.progress(0)
    
    try:
        process = subprocess.Popen(
            ["python", f"src/{nom_script}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        barre_progression.progress(30)
        stdout, _ = process.communicate()
        barre_progression.progress(100)
        
        if process.returncode == 0:
            st.success(f"✅ {description} exécuté avec succès.")
            with st.expander("Consulter les logs d'exécution de la console"):
                st.code(stdout)
        else:
            st.error(f"❌ Échec de l'exécution du script {nom_script}.")
            with st.expander("Analyser la trace d'erreur (Traceback)"):
                st.code(stdout)
                
    except Exception as e:
        st.error(f"Erreur d'instanciation du sous-processus : {str(e)}")

# Onglets structurés selon l'architecture du framework
tab_pilote, tab_dataset, tab_ae, tab_ldm, tab_quality, tab_classif = st.tabs(
    [
        "🚀 ORCHESTRATION PIPELINE",
        "📊 Analyse Dataset",
        "📐 AutoencoderKL (VAE)",
        "🌌 Latent Diffusion (LDM)",
        "📈 Évaluation Métrique",
        "🎯 Classification S1/S2",
    ]
)

# ==========================================
# ONGLET 1 : ORCHESTRATION PIPELINE
# ==========================================
with tab_pilote:
    st.header("🎮 Pipeline Execution Controller")
    st.markdown(
        """
        Ce module centralise l'orchestration séquentielle de ce framework de recherche. 
        Chaque phase ci-dessous correspond à un verrou technologique spécifique de l'architecture LDM.
        """
    )
    
    with st.expander("💡 Version Vulgarisée : Quel est l'objectif de cet onglet ?"):
        st.success(
            "**En clair : La chaîne de fabrication du pipeline.**\n\n"
            "Ce panneau de contrôle supervise la génération de radiographies synthétiques "
            "et l'évaluation de leur apport au sein d'un modèle de diagnostic. Chaque section "
            "permet d'exécuter les modules de manière séquentielle, respectant l'ordre logique de traitement."
        )
    if not DATASET_SOURCE_DISPONIBLE:
        st.info(
            "Pour la soutenance, cet onglet documente le protocole exécuté localement. "
            "Les preuves expérimentales sont disponibles dans les onglets suivants via les CSV, courbes, matrices "
            "et images synthétiques enregistrés dans `outputs/`."
        )
    st.markdown("---")
    
    # Phase 1
    st.subheader("Phase 1 : Entraînement de l'AutoencoderKL (VAE)")
    st.markdown(
        "**Objectif académique :** Compression de l'espace pixel $H \\times W$ vers un espace latent compact $h \\times w$ de basse dimension. "
        "La régularisation par divergence KL (Kullback-Leibler) contraint la distribution latente à suivre une loi normale standard $\\mathcal{N}(0, I)$."
    )
    with st.expander("💡 Explication simplifiée - Phase 1"):
        st.info("**Le compresseur d'images :** Ce module entraîne le modèle à encoder une image haute résolution sous la forme d'un vecteur condensé dans un espace latent, tout en veillant à préserver les caractéristiques géométriques essentielles lors du décodage.")
    if st.button("▶️ Initialiser l'optimisation du VAE (AE_EPOCHS)", disabled=not DATASET_SOURCE_DISPONIBLE):
        executer_script("train_autoencoder_kl.py", "Optimisation de l'AutoencoderKL")
        
    st.markdown("---")
    
    # Phase 2
    st.subheader("Phase 2 : Entraînement du Latent UNet (Processus de Diffusion Direct/Inverse)")
    st.markdown(
        "**Objectif académique :** Modélisation de la distribution de probabilité latente. "
        "Le réseau UNet apprend par rétropropagation à prédire le bruit gaussien ajouté à chaque pas de temps $t$ via l'optimisation de l'estimateur $\\epsilon_\\theta(z_t, t)$."
    )
    with st.expander("💡 Explication simplifiée - Phase 2"):
        st.info("**L'apprentissage du débruitage :** Le réseau apprend à inverser un processus de dégradation (ajout de bruit gaussien). Il développe la capacité de reconstruire pas à pas des structures pulmonaires cohérentes à partir de représentations latentes altérées.")
    if st.button("▶️ Lancer l'apprentissage de la Diffusion (DIFF_EPOCHS)", disabled=not DATASET_SOURCE_DISPONIBLE):
        executer_script("train_latent_diffusion.py", "Entraînement de la Diffusion Latente")
        
    st.markdown("---")
    
    # Phase 3
    st.subheader("Phase 3 : Échantillonnage de l'Espace Latent & Inférence DDPM")
    st.markdown(
        "**Objectif académique :** Synthèse d'images par inversion stochastique (Inférence). "
        "À partir d'un tenseur de bruit pur $\\mathcal{N}(0, I)$, le modèle applique le processus de débruitage itératif pas à pas pour reconstruire les vecteurs latents, ensuite décodés par le VAE en radiographies haute résolution."
    )
    with st.expander("💡 Explication simplifiée - Phase 3"):
        st.info("**La génération de nouvelles instances :** En partant d'un bruit aléatoire, le modèle applique le processus inverse appris pour matérialiser de nouvelles radiographies thoraciques totalement inédites, qui n'appartiennent à aucun patient réel.")
    if st.button("▶️ Exécuter l'échantillonnage de l'espace latent (Sampling)"):
        executer_script("sample_latent_diffusion.py", "Inférence DDPM & Échantillonnage")
        
    st.markdown("---")
    
    # Phase 4
    st.subheader("Phase 4 : Quantification de la Fidélité Structurelle")
    st.markdown(
        "**Objectif académique :** Analyse comparative pixellaire et perceptuelle. "
        "Évalue la distorsion mathématique entre les images d'origine et leurs reconstructions latentes."
    )
    with st.expander("💡 Explication simplifiée - Phase 4"):
        st.info("**Le contrôle de conformité visuelle :** Cette étape mesure la ressemblance mathématique entre les images originales et les versions reconstruites afin de s'assurer de l'absence de pertes d'informations cliniques.")
    if st.button("▶️ Calculer les métriques structurelles (MSE / SSIM)", disabled=not DATASET_SOURCE_DISPONIBLE):
        executer_script("evaluate_image_quality.py", "Calcul de la fidélité structurelle")
        
    st.markdown("---")
    
    # Phase 5
    st.subheader("Phase 5 : Entraînement du Classifieur S1 (Baseline Réelle)")
    st.markdown(
        "**Objectif académique :** Établir la performance de référence (Baseline). "
        "Le réseau de neurones convolutif (CNN) est entraîné exclusivement sur les données réelles du dataset d'origine pour apprendre à isoler la signature pathologique de la Pneumonie."
    )
    with st.expander("💡 Explication simplifiée - Phase 5"):
        st.info("**Le modèle de référence (Baseline) :** Entraînement d'un classifieur uniquement sur la base de données réelles initiales. Les scores obtenus serviront de point de comparaison pour mesurer l'apport des images synthétiques.")
    if st.button("▶️ Entraîner le modèle de classification Stade 1", disabled=not DATASET_SOURCE_DISPONIBLE):
        executer_script("train_classifier_s1.py", "Classification Stade 1")
        
    st.markdown("---")
    
    # Phase 6
    st.subheader("Phase 6 : Entraînement du Classifieur S2 (Augmentation de Données Synthétiques)")
    st.markdown(
        "**Objectif académique :** Mesurer le transfert d'apprentissage génératif. "
        "Entraînement du même CNN sur un dataset augmenté combinant la distribution réelle et la distribution synthétique générée par ce LDM."
    )
    with st.expander("💡 Explication simplifiée - Phase 6"):
        st.info("**L'évaluation du gain de performance :** Entraînement du même type de réseau, mais en introduisant les images synthétiques générées par le modèle au sein du jeu de données d'apprentissage, évaluant ainsi l'efficacité de l'augmentation.")
    if st.button("▶️ Entraîner le modèle de classification Stade 2", disabled=not DATASET_SOURCE_DISPONIBLE):
        executer_script("train_classifier_s2.py", "Classification Stade 2")
        
    st.markdown("---")
    
    # Phase 7
    st.subheader("Phase 7 : Évaluation Comparative de la Robustesse Clinique")
    st.markdown(
        "**Objectif académique :** Consolidation des résultats pour le rapport de thèse. "
        "Génération des rapports différentiels statistiques et des visualisations d'aide au diagnostic."
    )
    with st.expander("💡 Explication simplifiée - Phase 7"):
        st.info("**La confrontation finale :** Analyse comparative des performances des deux classifieurs afin de valider si l'intégration des données générées par le modèle améliore la détection des pathologies.")
    if st.button("▶️ Générer l'évaluation comparative finale"):
        executer_script("evaluate_classifier.py", "Analyse Comparative S1 vs S2")

# ==========================================
# ONGLET 2 : ANALYSE DATASET
# ==========================================
with tab_dataset:
    st.header("📊 Analyse Quantitative et Structurelle de la Distribution Source")
    st.info(
        "**Explication académique de l'onglet :** Cet espace permet de cartographier la structure des données sources. "
        "Avant d'optimiser une architecture générative, il est nécessaire de valider l'équilibre des classes cibles (Normal vs Pneumonie) "
        "et l'intégrité mathématique des tenseurs d'entrée (normalisation, plages de valeurs min/max)."
    )
    
    with st.expander("💡 Version Vulgarisée : L'Analyse du Dataset"):
        st.success(
            "**En clair : L'audit des données d'entrée.**\n\n"
            "Cet onglet présente un état des lieux des radiographies réelles utilisées. L'analyse se concentre sur deux aspects :\n"
            "1. La répartition numérique des classes pour prévenir tout biais algorithmique.\n"
            "2. Les propriétés tensorielles des images afin de garantir l'uniformité des données transmises aux réseaux."
        )
        
    if st.button(
        "Actualiser le manifest statistique de distribution (dataset_summary.csv)",
        disabled=not DATASET_SOURCE_DISPONIBLE,
    ):
        try:
            summary = summarize_dataset()
            st.success("Manifest statistique mis à jour.")
            st.dataframe(summary, width="stretch")
        except Exception as exc:
            st.error("Erreur d'analyse quantitative du dataset.")
            st.exception(exc)

    summary_path = ldm_config.METRICS_DIR / "dataset_summary.csv"
    if summary_path.exists():
        st.subheader("Manifest d'inventaire statistique")
        with st.expander("💡 Interprétation des données d'inventaire"):
            st.info("**Point d'attention :** Ce tableau expose les volumes globaux par classe. Une répartition équilibrée constitue une condition requise pour stabiliser la phase d'apprentissage.")
        st.dataframe(pd.read_csv(summary_path), width="stretch")

    st.header("Inspection Tensorielle d'un Mini-Batch")
    if DATASET_SOURCE_DISPONIBLE:
        batch_size_slider = st.slider("Dimension du mini-batch ($N$ images)", 1, 8, 4)
        if st.button("Charger et inspecter un mini-batch aléatoire"):
            try:
                loader = get_dataloader(split="train", batch_size=batch_size_slider, shuffle=True, num_workers=0)
                batch = next(iter(loader))
                images, labels, paths = batch["image"], batch["label"], batch["path"]
                
                st.write("**Tenseur Image Shapes (N, C, H, W) :**", tuple(images.shape))
                st.write("**Tenseur Label Shapes (N,) :**", tuple(labels.shape))
                st.write("**Intensité minimale du pixel :**", float(images.min()))
                st.write("**Intensité maximale du pixel :**", float(images.max()))
                
                cols = st.columns(batch_size_slider)
                for idx in range(batch_size_slider):
                    image = images[idx, 0].numpy()
                    label_name = "PNEUMONIA" if int(labels[idx].item()) == 1 else "NORMAL"
                    with cols[idx]:
                        st.image(image, caption=f"Classe cible : {label_name}", clamp=True, width="stretch")
                        st.caption(f"Nom du fichier : {Path(paths[idx]).name}")
            except Exception as exc:
                st.error("Échec de l'extraction tensorielle du batch.")
                st.exception(exc)
    else:
        st.info(
            "Inspection désactivée dans la version déployée : elle nécessite les radiographies sources complètes. "
            "Le résumé statistique déjà calculé reste affiché ci-dessus."
        )

# ==========================================
# ONGLET 3 : AUTOENCODERKL (VAE)
# ==========================================
with tab_ae:
    st.header("📐 Analyse de la Convergence de l'AutoencoderKL")
    st.info(
        "**Explication académique de l'onglet :** Cet onglet permet d'évaluer l'aptitude du modèle à projeter une radiographie "
        "au sein d'un espace de dimension réduite, sans altérer la topologie des structures pulmonaires. Le décodeur est chargé de l'opération inverse."
    )
    
    with st.expander("📝 Guide des métriques VAE (Interprétation scientifique)"):
        st.markdown(
            """
            *   **Loss Reconstruction (MSE) :** Évalue l'erreur quadratique moyenne entre l'image source $x$ et sa reconstruction $\\hat{x}$. Sa diminution progressive valide l'exactitude du décodage.
            *   **KL Divergence (Kullback-Leibler) :** Quantifie l'écart entre la distribution de l'espace latent et une loi gaussienne standard $\\mathcal{N}(0, I)$. Une régularisation KL maîtrisée garantit la continuité de l'espace latent, condition indispensable au bon fonctionnement du modèle de diffusion.
            """
        )

    with st.expander("💡 Version Vulgarisée : L'Autoencodeur"):
        st.success(
            "**En clair : L'évaluation du mécanisme de compression.**\n\n"
            "Afin d'optimiser les temps de calcul, l'architecture compresse les informations visuelles sous forme de représentations latentes réduites :\n"
            "1. L'encodeur extrait les caractéristiques fondamentales de l'image réelle.\n"
            "2. Ces propriétés sont organisées de manière rigoureuse au sein de l'espace latent.\n"
            "3. Le décodeur tente ensuite de reconstruire l'image originale haute résolution à partir de ces seules variables condensées.\n\n"
            "**Signification des indicateurs :**\n"
            "*   **Erreur de reconstruction (Loss) :** Plus ce score est bas, plus la réplication des détails anatomiques est fidèle.\n"
            "*   **Divergence KL :** Indique le niveau d'organisation de l'espace latent. Un espace bien structuré facilite les processus de génération ultérieurs."
        )

    train_metrics = ldm_config.METRICS_DIR / "autoencoder_train_metrics.csv"
    val_metrics = ldm_config.METRICS_DIR / "autoencoder_val_metrics.csv"
    
    if train_metrics.exists():
        st.subheader("Historique des logs d'entraînement (Loss Reconstruction + Terme KL)")
        with st.expander("💡 Analyse de l'historique quantitatif"):
            st.info("**Interprétation :** Une décroissance conjointe et régulière des valeurs de perte au fil des époques atteste d'une phase d'optimisation stable.")
        st.dataframe(pd.read_csv(train_metrics), width="stretch")
        
    if val_metrics.exists():
        st.subheader("Historique des logs de validation")
        st.dataframe(pd.read_csv(val_metrics), width="stretch")
        
    st.subheader("Courbes de Convergence de la Fonction de Coût (Loss Curves)")
    ae_curve = ldm_config.FIGURES_DIR / "autoencoder_train_curves.png"
    if ae_curve.exists():
        st.image(str(ae_curve), width="stretch")
        
    st.subheader("Évaluation de la Fidélité des Reconstructions Latentes")
    recon_images = sorted(ldm_config.RECON_DIR.glob("reconstruction_epoch_*.png"))
    if recon_images:
        with st.expander("💡 Analyse qualitative visuelle"):
            st.info("**Observation :** La comparaison directe entre l'image d'origine et la version reconstruite permet de valider visuellement la préservation des contrastes et des textures tissulaires.")
        st.image(str(recon_images[-1]), width="stretch")

# ==========================================
# ONGLET 4 : LATENT DIFFUSION (LDM)
# ==========================================
with tab_ldm:
    st.header("🌌 Analyse de Convergence du Latent Diffusion Model")
    st.info(
        "**Explication académique de l'onglet :** Suivi de l'apprentissage du réseau générateur (UNet). Durant le processus de diffusion inverse, "
        "ce modèle supprime de manière itérative le bruit afin de matérialiser des descripteurs anatomiques réalistes au sein de l'espace latent."
    )
    
    with st.expander("📝 Guide des métriques LDM (Interprétation scientifique)"):
        st.markdown(
            """
            *   **Noise MSE (Loss) :** Différence quadratique moyenne entre le bruit réel injecté lors de la phase directe et le bruit prédit par le réseau UNet. Une courbe stable et descendante démontre que l'algorithme assimile correctement les lois de distribution des structures saines et pathologiques.
            """
        )

    with st.expander("💡 Version Vulgarisée : Le Modèle de Diffusion"):
        st.success(
            "**En clair : Le processus de génération itératif.**\n\n"
            "Ce module supervise l'apprentissage du moteur de création de ce projet. Le principe repose sur l'inversion d'une dégradation stochastique :\n\n"
            "Le modèle s'exerce à supprimer méthodiquement le bruit appliqué aux représentations latentes. En parvenant à isoler et à retirer précisément ce bruit, il devient capable de construire une forme anatomique cohérente à partir d'une matrice initialement aléatoire.\n\n"
            "**Signification de la perte (Noise MSE) :** Cette valeur évalue la précision du modèle lors de l'estimation du bruit. La réduction de cette erreur témoigne de l'affinement des capacités génératives de l'IA."
        )

    diff_metrics = ldm_config.METRICS_DIR / "latent_diffusion_train_metrics.csv"
    if diff_metrics.exists():
        st.subheader("Évolution de l'Erreur Quadratique Moyenne du Bruit (Noise MSE)")
        with st.expander("💡 Interprétation du tableau de bruit"):
            st.info("**Observation :** La diminution progressive de la valeur de perte (`loss`) confirme l'optimisation des performances du réseau UNet au cours de l'entraînement.")
        st.dataframe(pd.read_csv(diff_metrics), width="stretch")
        
    diff_curve = ldm_config.FIGURES_DIR / "latent_diffusion_noise_mse_curve.png"
    if diff_curve.exists():
        st.image(str(diff_curve), width="stretch")
        
    st.header("Visualisation de la Distribution Synthétique Générée")
    grid_path = ldm_config.SAMPLES_DIR / "ldm_grid.png"
    if grid_path.exists():
        with st.expander("💡 Analyse de la grille d'échantillons synthétiques"):
            st.info("**Observation :** Les images affichées ci-dessous ont été générées artificiellement par le modèle de diffusion. Bien qu'elles ne correspondent à aucun dossier patient existant, elles doivent manifester un réalisme clinique rigoureux.")
        st.image(str(grid_path), width="stretch")
    else:
        st.info("Aucune grille matricielle de distribution générée n'est disponible.")
        
    manifest_path = ldm_config.METRICS_DIR / "generated_samples_manifest.csv"
    if manifest_path.exists():
        st.subheader("Manifest d'inventaire des images synthétiques")
        st.dataframe(pd.read_csv(manifest_path), width="stretch")

# ==========================================
# ONGLET 5 : ÉVALUATION MÉTRIQUE
# ==========================================
with tab_quality:
    st.header("📈 Métriques d'Évaluation de la Qualité et Distorsion Image")
    st.info(
        "**Explication académique de l'onglet :** Cet onglet quantifie mathématiquement la fidélité des images reconstruites "
        "par rapport aux données médicales sources, permettant de valider l'absence d'artefacts structurels significatifs."
    )
    
    with st.expander("📝 Comprendre le compromis MSE vs SSIM"):
        st.markdown(
            """
            *   **MSE (Mean Squared Error) :** Mesure de la distorsion absolue à l'échelle du pixel. Une valeur tendant vers 0 indique une similarité numérique parfaite.
            *   **SSIM (Structural Similarity Index Measure) :** Métrique normalisée entre -1 et 1 évaluant la préservation de la texture, du contraste et de la luminance, en forte corrélation avec la perception visuelle humaine. Un coefficient supérieur à $0.85$ atteste d'une restitution de qualité conforme aux exigences de l'imagerie médicale.
            """
        )

    with st.expander("💡 Version Vulgarisée : Le Contrôle Qualité"):
        st.success(
            "**En clair : La validation statistique de la qualité visuelle.**\n\n"
            "Au-delà de l'inspection visuelle, l'évaluation de la conformité clinique repose sur des indicateurs mathématiques précis :\n\n"
            "*   **L'indicateur MSE :** Évalue strictement les divergences numériques pixel par pixel. L'objectif est d'obtenir la valeur la plus basse possible.\n"
            "*   **L'indice SSIM :** Analyse la coherence structurelle globale (formes, contours, contrastes) de manière similaire à l'œil humain. Un score supérieur à 0.85 démontre que les structures anatomiques essentielles ont été fidèlement préservées."
        )

    quality_metrics = ldm_config.METRICS_DIR / "image_quality_metrics.csv"
    quality_summary = ldm_config.METRICS_DIR / "image_quality_summary.csv"
    
    if quality_metrics.exists():
        st.subheader("Analyse comparative par paire (MSE vs Structural Similarity Index - SSIM)")
        with st.expander("💡 Interprétation du tableau de qualité"):
            st.info("**Observation :** Les valeurs individuelles de l'indice SSIM permettent de vérifier l'homogénéité de la qualité des reconstructions sur l'ensemble de l'échantillon.")
        st.dataframe(pd.read_csv(quality_metrics), width="stretch")
        
    if quality_summary.exists():
        st.subheader("Synthèse statistique agrégée des indicateurs de fidélité")
        st.dataframe(pd.read_csv(quality_summary), width="stretch")

# ==========================================
# ONGLET 6 : CLASSIFICATION S1/S2
# ==========================================
with tab_classif:
    st.header("🎯 Validation Clinique & Évaluation de l'Impact de l'Augmentation Générative")
    st.info(
        "**Explication académique de l'onglet :** Cet espace constitue le noyau de la validation scientifique de ce travail de recherche. "
        "L'analyse compare les performances de deux classifieurs distincts afin de mesurer rigoureusement l'apport clinique des images synthétisées."
    )
    
    with st.expander("📝 Guide d'interprétation des métriques de performance"):
        st.markdown(
            """
            *   **Accuracy (Exactitude) :** Proportion globale de diagnostics corrects établis sur le jeu de test.
            *   **Recall (Sensibilité / Taux de Vrais Positifs) :** Capacité du modèle à identifier l'intégralité des cas pathologiques (Pneumonies). En imagerie médicale, il s'agit d'un indicateur critique visant à minimiser le taux de faux négatifs.
            *   **F1-Score :** Moyenne harmonique de la Précision et du Recall, fournissant une évaluation globale robuste de la performance algorithmique.
            *   **Objectif du projet :** La validation scientifique de cette étude est acquise si les performances du **Stade 2** (modèle entraîné avec l'apport des données synthétiques du LDM) surpassent celles du **Stade 1** (baseline réelle), démontrant ainsi l'efficacité de la stratégie d'augmentation.
            """
        )

    with st.expander("💡 Version Vulgarisée : L'Examen Comparatif des Classifieurs"):
        st.success(
            "**En clair : La démonstration de la valeur ajoutée de ce projet.**\n\n"
            "Cette phase finale vise à prouver que les données générées par le modèle de diffusion optimisent efficacement l'apprentissage d'un système de diagnostic automatisé. Deux configurations sont confrontées :\n\n"
            "*   **Le Classifieur Stade 1 (Baseline) :** Entraîné uniquement à partir des données réelles disponibles.\n"
            "*   **Le Classifieur Stade 2 (Augmenté) :** Entraîné sur un ensemble étendu intégrant les radiographies réelles et les images synthétiques générées par l'IA.\n\n"
            "**Critères d'évaluation du succès :**\n"
            "*   **L'Accuracy :** Mesure l'efficacité décisionnelle globale du système.\n"
            "*   **Le Recall :** Représente la sécurité diagnostique (aptitude à détecter tous les patients atteints d'une pathologie).\n"
            "*   **Validation des résultats :** Une progression positive de ces indices entre le Stade 1 et le Stade 2 confirme de manière empirique l'utilité clinique du pipeline génératif développé."
        )

    s1_path = ldm_config.METRICS_DIR / "classifier_s1_metrics.csv"
    s2_path = ldm_config.METRICS_DIR / "classifier_s2_metrics.csv"
    comparison_path = ldm_config.METRICS_DIR / "classification_comparison.csv"
    
    if s1_path.exists():
        st.subheader("Performances Stade 1 : Entraînement exclusif sur la distribution réelle")
        st.dataframe(pd.read_csv(s1_path), width="stretch")
    if s2_path.exists():
        st.subheader("Performances Stade 2 : Entraînement hybride (Distribution réelle + Distribution synthétique)")
        st.dataframe(pd.read_csv(s2_path), width="stretch")
        
    if comparison_path.exists():
        st.subheader("Rapport comparatif différentiel (Gains en Accuracy, Recall, F1-Score)")
        with st.expander("💡 Interprétation du rapport différentiel"):
            st.info("**Observation :** Ce tableau récapitule les variations de performance. Des deltas positifs au sein de la ligne d'évolution valident la pertinence et l'apport de la méthode d'augmentation de données développée.")
        st.dataframe(pd.read_csv(comparison_path), width="stretch")
        
    plot_path = ldm_config.FIGURES_DIR / "s1_vs_s2_barplot.png"
    if plot_path.exists():
        st.image(str(plot_path), width="stretch")
        
    st.subheader("Analyse Micro-Statistique via Matrices de Confusion")
    cm1 = ldm_config.CLASSIF_DIR / "confusion_matrix_s1.csv"
    cm2 = ldm_config.CLASSIF_DIR / "confusion_matrix_s2.csv"
    
    if cm1.exists() or cm2.exists():
        with st.expander("💡 Guide de lecture des matrices de confusion"):
            st.info("**Interprétation :** Les éléments diagonaux représentent les prédictions exactes du modèle (Vrais Positifs et Vrais Négatifs). Les cellules hors diagonale matérialisent les erreurs de classification. Le succès de la démarche s'illustre par une réduction de ces erreurs entre le Stade 1 et le Stade 2.")
            
    if cm1.exists():
        st.write("**Matrice de Confusion - Stade 1 (Baseline)**")
        st.dataframe(pd.read_csv(cm1), width="stretch")
    if cm2.exists():
        st.write("**Matrice de Confusion - Stade 2 (Augmenté)**")
        st.dataframe(pd.read_csv(cm2), width="stretch")
