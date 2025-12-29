import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Prédiction Risque Crédit",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# 2. ENTRAÎNEMENT DU MODÈLE (CACHÉ)
# ==========================================
# On utilise @st.cache_resource pour ne pas ré-entraîner le modèle à chaque clic
@st.cache_resource
def load_and_train_model():
    # Chargement
    try:
        df = pd.read_csv("credit_data.csv")
        df = df.dropna()
    except FileNotFoundError:
        st.error("Le fichier 'credit_data.csv' est introuvable.")
        return None, None

    # Variables
    X = df[['income', 'age', 'loan', 'LTI']]
    y = df['default']

    # Standardisation (Très important pour la régression logistique)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entraînement
    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler

# Chargement du modèle et du scaler
model, scaler = load_and_train_model()

# ==========================================
# 3. INTERFACE UTILISATEUR (SIDEBAR)
# ==========================================
st.sidebar.header("Paramètres du Client")

def user_input_features():
    # Saisie des données
    income = st.sidebar.number_input("Revenu Annuel (€)", min_value=1000.0, value=40000.0, step=500.0)
    age = st.sidebar.slider("Âge", min_value=18, max_value=100, value=30)
    loan = st.sidebar.number_input("Montant du Prêt demandé (€)", min_value=100.0, value=5000.0, step=100.0)
    
    # Calcul automatique du LTI (Loan to Income)
    # LTI = Dette / Revenu
    lti = loan / income
    
    # Affichage du LTI calculé pour info
    st.sidebar.info(f"Ratio Dette/Revenu (LTI) calculé : {lti:.4f}")
    
    data = {
        'income': income,
        'age': age,
        'loan': loan,
        'LTI': lti
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Récupération des données saisies par l'utilisateur
input_df = user_input_features()

# ==========================================
# 4. PARTIE PRINCIPALE (MAIN)
# ==========================================
st.title("🏦 Système de Scoring Crédit (IA)")
st.markdown("""
Cette application utilise un modèle de **Régression Logistique** pour estimer 
la probabilité de défaut de paiement d'un client.
""")

# Affichage des données saisies
st.subheader("1. Profil du client analysé")
st.write(input_df)

# ==========================================
# 5. PRÉDICTION
# ==========================================
if st.button("Lancer l'analyse du risque"):
    if model is not None:
        # 1. Standardiser les nouvelles données comme lors de l'entraînement
        input_df_scaled = scaler.transform(input_df)

        # 2. Prédiction (Classe 0 ou 1)
        prediction = model.predict(input_df_scaled)
        
        # 3. Probabilité (Risque en %)
        prediction_proba = model.predict_proba(input_df_scaled)
        risque_defaut = prediction_proba[0][1] # Probabilité de la classe 1 (Défaut)

        st.subheader("2. Résultat de l'analyse")

        # Affichage dynamique selon le résultat
        col1, col2 = st.columns(2)
        with col1:
            st.write("Probabilité de défaut :")
            # Barre de progression colorée
            st.progress(risque_defaut)
            st.metric(label="Score de Risque", value=f"{risque_defaut:.2%}")

        with col2:
            st.write("Décision recommandée :")
            if risque_defaut > 0.5: # Seuil de 50% (modifiable par la banque)
                st.error("⛔ **REFUS CONSEILLÉ**")
                st.write("Le risque est trop élevé (Défaut Probable).")
            elif risque_defaut > 0.2:
                st.warning("⚠️ **EXAMEN MANUEL REQUIS**")
                st.write("Risque modéré.")
            else:
                st.success("✅ **ACCORD CONSEILLÉ**")
                st.write("Le client présente un profil fiable.")

        # Explication des facteurs (Coefficients)
        st.markdown("---")
        st.info("💡 **Note :** Le modèle privilégie l'âge (stabilité) et pénalise un ratio LTI élevé.")

    else:
        st.error("Erreur : Modèle non chargé.")