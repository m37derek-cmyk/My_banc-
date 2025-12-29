import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Prédiction Risque Crédit",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# 2. CHARGEMENT ET ENTRAÎNEMENT (CACHE)
# ==========================================
@st.cache_resource
def load_and_train_model():
    # --- Gestion robuste du chemin du fichier ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "credit_data.csv")

    # --- Chargement des données ---
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()
    except FileNotFoundError:
        return None, None, None, None, None, f"Erreur : Le fichier est introuvable au chemin : {file_path}"
    except Exception as e:
        return None, None, None, None, None, f"Erreur inattendue : {e}"

    # --- Préparation des variables ---
    if not {'income', 'age', 'loan', 'LTI', 'default'}.issubset(df.columns):
        return None, None, None, None, None, "Erreur : Le fichier CSV ne contient pas les bonnes colonnes."

    X = df[['income', 'age', 'loan', 'LTI']]
    y = df['default']

    # --- Séparation Train / Test (Pour l'évaluation) ---
    # On garde 20% des données pour tester la matrice de confusion
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Standardisation ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Entraînement du modèle ---
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # --- Calcul des Métriques de performance ---
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # On retourne le modèle, le scaler, et les métriques
    return model, scaler, acc, cm, report, None

# Appel de la fonction
model, scaler, accuracy, cm, report, error_message = load_and_train_model()

# ==========================================
# 3. INTERFACE UTILISATEUR
# ==========================================
st.title("🏦 Système de Scoring Crédit (IA)")

# Gestion de l'affichage en cas d'erreur
if error_message:
    st.error(error_message)
    st.stop()

# --- Sidebar : Saisie ---
st.sidebar.header("Paramètres du Client")

def user_input_features():
    income = st.sidebar.number_input("Revenu Annuel (€)", min_value=1000.0, value=40000.0, step=500.0)
    age = st.sidebar.slider("Âge", min_value=18, max_value=100, value=30)
    loan = st.sidebar.number_input("Montant du Prêt demandé (€)", min_value=100.0, value=5000.0, step=100.0)
    
    lti = loan / income
    st.sidebar.info(f"Ratio Dette/Revenu (LTI) : {lti:.4f}")
    
    data = {
        'income': income,
        'age': age,
        'loan': loan,
        'LTI': lti
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Onglets Principaux ---
tab1, tab2 = st.tabs(["🔮 Prédiction", "📊 Performance du Modèle"])

with tab1:
    st.subheader("1. Profil du client")
    st.write(input_df)

    if st.button("Lancer l'analyse du risque"):
        # 1. Standardisation
        input_df_scaled = scaler.transform(input_df)

        # 2. Prédiction
        prediction_proba = model.predict_proba(input_df_scaled)
        risque_defaut = prediction_proba[0][1]

        st.subheader("2. Résultat de l'analyse")
        
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Probabilité de défaut :**")
            st.metric(label="Score de Risque", value=f"{risque_defaut:.2%}")
            
            if risque_defaut < 0.2:
                st.progress(risque_defaut)
            elif risque_defaut < 0.5:
                st.progress(risque_defaut)
            else:
                st.progress(risque_defaut)

        with col2:
            st.write("**Recommandation IA :**")
            if risque_defaut > 0.5:
                st.error("⛔ REFUS CONSEILLÉ")
                st.write("Le risque de non-remboursement est très élevé.")
            elif risque_defaut > 0.2:
                st.warning("⚠️ EXAMEN MANUEL REQUIS")
                st.write("Risque modéré, vérifiez les garanties.")
            else:
                st.success("✅ ACCORD CONSEILLÉ")
                st.write("Profil fiable et stable.")

with tab2:
    st.header("Évaluation de la performance")
    st.markdown("Ces métriques sont calculées sur un jeu de test (20% des données) que le modèle n'a jamais vu.")

    # Affichage des KPIs
    col_metric1, col_metric2 = st.columns(2)
    col_metric1.metric("Précision Globale (Accuracy)", f"{accuracy:.2%}")
    col_metric2.metric("Support (Nb de tests)", int(cm.sum()))

    st.markdown("---")
    
    col_graph1, col_graph2 = st.columns(2)

    with col_graph1:
        st.subheader("Matrice de Confusion")
        st.write("Visualisation des bonnes et mauvaises prédictions.")
        
        # Création du graphique avec Seaborn
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Predit')
        ax.set_ylabel('Réel')
        ax.set_xticklabels(['Payé (0)', 'Défaut (1)'])
        ax.set_yticklabels(['Payé (0)', 'Défaut (1)'])
        st.pyplot(fig)

    with col_graph2:
        st.subheader("Détails par classe")
        # Transformation du rapport en DataFrame pour un affichage propre
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
        
        st.info("""
        **Légende :**
        * **Precision** : Quand le modèle prédit "Défaut", a-t-il raison ?
        * **Recall** : Sur tous les vrais "Défauts", combien le modèle en a-t-il trouvé ?
        """)
