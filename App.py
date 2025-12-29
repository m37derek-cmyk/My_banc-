import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Prédiction Risque Crédit (CV)",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# 2. CHARGEMENT ET ENTRAÎNEMENT (CACHE)
# ==========================================
@st.cache_resource
def load_and_evaluate_model():
    # --- Gestion robuste du chemin du fichier ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "credit_data.csv")

    # --- Chargement des données ---
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()
    except FileNotFoundError:
        return None, None, None, None, None, None, f"Erreur : Le fichier est introuvable au chemin : {file_path}"
    except Exception as e:
        return None, None, None, None, None, None, f"Erreur inattendue : {e}"

    # --- Préparation des variables ---
    if not {'income', 'age', 'loan', 'LTI', 'default'}.issubset(df.columns):
        return None, None, None, None, None, None, "Erreur : Colonnes manquantes dans le CSV."

    X = df[['income', 'age', 'loan', 'LTI']]
    y = df['default']

    # --- Standardisation Globale ---
    # On standardise tout le dataset car on va faire de la validation croisée
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Configuration de la Validation Croisée ---
    # 5 plis (folds), mélangés aléatoirement
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression()

    # --- Étape 1 : Évaluation par Cross-Validation ---
    # Calcul des scores de précision pour chaque pli
    cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')
    
    # Génération des prédictions "hors échantillon" pour la matrice de confusion
    # (Chaque point est prédit lorsqu'il était dans le set de test)
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=kf)

    # Calcul des métriques globales basées sur la CV
    avg_accuracy = cv_scores.mean()
    std_accuracy = cv_scores.std()
    cm = confusion_matrix(y, y_pred_cv)
    report = classification_report(y, y_pred_cv, output_dict=True)

    # --- Étape 2 : Entraînement Final pour l'Application ---
    # Maintenant qu'on a évalué, on entraîne le modèle sur TOUTES les données
    # pour qu'il soit le plus performant possible pour l'utilisateur final.
    final_model = LogisticRegression()
    final_model.fit(X_scaled, y)

    return final_model, scaler, avg_accuracy, std_accuracy, cm, report, None

# Appel de la fonction
model, scaler, avg_acc, std_acc, cm, report, error_msg = load_and_evaluate_model()

# ==========================================
# 3. INTERFACE UTILISATEUR
# ==========================================
st.title("🏦 Système de Scoring Crédit (Avec Cross-Validation)")

if error_msg:
    st.error(error_msg)
    st.stop()

# --- Sidebar : Saisie ---
st.sidebar.header("Paramètres du Client")

def user_input_features():
    income = st.sidebar.number_input("Revenu Annuel (€)", min_value=1000.0, value=40000.0, step=500.0)
    age = st.sidebar.slider("Âge", min_value=18, max_value=100, value=30)
    loan = st.sidebar.number_input("Montant du Prêt demandé (€)", min_value=100.0, value=5000.0, step=100.0)
    
    lti = loan / income
    st.sidebar.info(f"Ratio Dette/Revenu (LTI) : {lti:.4f}")
    
    data = {'income': income, 'age': age, 'loan': loan, 'LTI': lti}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Onglets ---
tab1, tab2 = st.tabs(["🔮 Prédiction", "📊 Performance (Cross-Validation)"])

with tab1:
    st.subheader("1. Profil du client")
    st.write(input_df)

    if st.button("Lancer l'analyse du risque"):
        # Transformation avec le scaler entraîné sur tout le dataset
        input_df_scaled = scaler.transform(input_df)
        
        prediction_proba = model.predict_proba(input_df_scaled)
        risque_defaut = prediction_proba[0][1]

        st.subheader("2. Résultat de l'analyse")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Score de Risque", value=f"{risque_defaut:.2%}")
            if risque_defaut < 0.2:
                st.progress(risque_defaut)
            elif risque_defaut < 0.5:
                st.progress(risque_defaut)
            else:
                st.progress(risque_defaut)

        with col2:
            if risque_defaut > 0.5:
                st.error("⛔ REFUS CONSEILLÉ")
                st.write("Risque élevé.")
            elif risque_defaut > 0.2:
                st.warning("⚠️ EXAMEN MANUEL REQUIS")
                st.write("Risque modéré.")
            else:
                st.success("✅ ACCORD CONSEILLÉ")
                st.write("Profil fiable.")

with tab2:
    st.header("Évaluation Robuste (K-Fold Cross-Validation)")
    st.markdown("""
    Le modèle a été évalué en utilisant la **Validation Croisée à 5 plis**.
    Cela signifie que le modèle a été testé 5 fois sur des parties différentes des données.
    """)

    # Affichage des KPIs
    col_kpi1, col_kpi2 = st.columns(2)
    col_kpi1.metric("Précision Moyenne", f"{avg_acc:.2%}", delta=f"± {std_acc:.2%}")
    col_kpi2.metric("Support Total", int(cm.sum()))

    st.markdown("---")
    
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Matrice de Confusion Globale")
        st.write("Cumul des prédictions sur les 5 plis de test.")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Réalité')
        ax.set_xticklabels(['Payé', 'Défaut'])
        ax.set_yticklabels(['Payé', 'Défaut'])
        st.pyplot(fig)

    with col_g2:
        st.subheader("Rapport de Classification")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
