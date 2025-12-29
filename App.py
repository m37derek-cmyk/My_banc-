import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
        return None, None, f"Erreur : Le fichier est introuvable au chemin : {file_path}"
    except Exception as e:
        return None, None, f"Erreur inattendue : {e}"

    # --- Préparation des variables ---
    # X = Features (Revenu, Age, Prêt, Ratio Dette/Revenu)
    # y = Target (0 = Payé, 1 = Défaut)
    if not {'income', 'age', 'loan', 'LTI', 'default'}.issubset(df.columns):
        return None, None, "Erreur : Le fichier CSV ne contient pas les bonnes colonnes."

    X = df[['income', 'age', 'loan', 'LTI']]
    y = df['default']

    # --- Standardisation (Crucial pour la Régression Logistique) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Entraînement du modèle ---
    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler, None

# Appel de la fonction
model, scaler, error_message = load_and_train_model()

# ==========================================
# 3. INTERFACE UTILISATEUR
# ==========================================
st.title("🏦 Système de Scoring Crédit (IA)")

# Gestion de l'affichage en cas d'erreur de chargement
if error_message:
    st.error(error_message)
    st.stop() # Arrête l'application ici si le modèle n'est pas chargé

st.sidebar.header("Paramètres du Client")

def user_input_features():
    # Saisie des données
    income = st.sidebar.number_input("Revenu Annuel (€)", min_value=1000.0, value=40000.0, step=500.0)
    age = st.sidebar.slider("Âge", min_value=18, max_value=100, value=30)
    loan = st.sidebar.number_input("Montant du Prêt demandé (€)", min_value=100.0, value=5000.0, step=100.0)
    
    # Calcul automatique du LTI
    lti = loan / income
    st.sidebar.info(f"Ratio Dette/Revenu (LTI) : {lti:.4f}")
    
    data = {
        'income': income,
        'age': age,
        'loan': loan,
        'LTI': lti
    }
    return pd.DataFrame(data, index=[0])

# Récupération des saisies
input_df = user_input_features()

# Affichage du profil
st.subheader("1. Profil du client")
st.write(input_df)

# ==========================================
# 4. PRÉDICTION
# ==========================================
if st.button("Lancer l'analyse du risque"):
    # 1. Standardiser les données saisies (comme lors de l'entraînement)
    input_df_scaled = scaler.transform(input_df)

    # 2. Calculer la probabilité
    prediction_proba = model.predict_proba(input_df_scaled)
    risque_defaut = prediction_proba[0][1] # Probabilité de la classe 1

    st.subheader("2. Résultat de l'analyse")
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Probabilité de défaut :**")
        st.metric(label="Score de Risque", value=f"{risque_defaut:.2%}")
        
        # Couleur de la barre selon le risque
        if risque_defaut < 0.2:
            st.progress(risque_defaut) # Vert (implicite, barre courte)
        elif risque_defaut < 0.5:
            st.progress(risque_defaut) # Orange (barre moyenne)
        else:
            st.progress(risque_defaut) # Rouge (barre longue)

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
