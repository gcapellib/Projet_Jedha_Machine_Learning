import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Chargement du modèle ──────────────────────────────────────────────────────
# On charge le package complet : encodeur + modèle + seuil
# C'est fait une seule fois au démarrage du serveur, pas à chaque requête
with open('model.pkl', 'rb') as f:
    model_package = pickle.load(f)

encoder   = model_package['encoder']    # le ColumnTransformer
model     = model_package['model']      # le Random Forest
threshold = model_package['threshold']  # 0.4

# ── Initialisation de l'application FastAPI ──────────────────────────────────
app = FastAPI(title="CardioPredict API", version="1.0")

# CORS — permet au frontend HTML d'appeler l'API depuis un autre domaine
# Sans ça, le navigateur bloque les requêtes du frontend vers l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en production, remplacer par l'URL exacte du frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modèle de données attendu par l'API ──────────────────────────────────────
# Pydantic vérifie automatiquement que toutes les valeurs sont présentes
# et ont le bon type — si une valeur manque, l'API retourne une erreur claire
class Patient(BaseModel):
    Age: int
    Sex: str                # "M" ou "F"
    ChestPainType: str      # "ASY", "ATA", "NAP", "TA"
    RestingBP: float
    Cholesterol: float
    FastingBS: int          # 0 ou 1
    RestingECG: int         # 0, 1 ou 2
    MaxHR: float
    ExerciseAngina: str     # "Y" ou "N"
    Oldpeak: float
    ST_Slope: str           # "Up", "Flat", "Down"

# ── Endpoint de santé — pour vérifier que l'API tourne ───────────────────────
@app.get("/")
def root():
    return {"status": "CardioPredict API is running 🫀"}

# ── Endpoint de prédiction ────────────────────────────────────────────────────
@app.post("/predict")
def predict(patient: Patient):
    
    # 1. On convertit les données reçues en DataFrame — 
    #    exactement comme dans le notebook au Step 13
    df_patient = pd.DataFrame([patient.dict()])
    
    # 2. On applique le même encodeur que celui utilisé à l'entraînement
    #    IMPORTANT : transform() seulement, jamais fit_transform() ici
    patient_transformed = encoder.transform(df_patient)
    
    # 3. On récupère la probabilité d'être malade (colonne 1 = classe "malade")
    proba_malade = model.predict_proba(patient_transformed)[0][1]
    proba_sain   = model.predict_proba(patient_transformed)[0][0]
    
    # 4. On applique notre seuil de décision médical : 0.4
    diagnostic = "MALADE" if proba_malade >= threshold else "SAIN"
    
    # 5. On retourne le résultat complet au frontend
    return {
        "diagnostic"   : diagnostic,
        "proba_malade" : round(float(proba_malade) * 100, 1),
        "proba_sain"   : round(float(proba_sain) * 100, 1),
        "seuil_utilise": threshold
    }