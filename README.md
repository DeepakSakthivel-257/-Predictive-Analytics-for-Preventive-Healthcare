Setup

python -m venv .venv
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt


Run

# Preprocess data
python -m src.preprocess --input data/heart.csv --out data/processed.csv

# Train & save model
python -m src.train --data data/processed.csv --model models/heart_pipeline.joblib

# Evaluate & generate plots
python -m src.evaluate --data data/processed.csv --model models/heart_pipeline.joblib --plots plots

# CLI prediction example
python -m src.realtime_predict --model models/heart_pipeline.joblib \
  --age 57 --sex 1 --cp 0 --trestbps 140 --chol 241 --fbs 0 \
  --restecg 1 --thalach 123 --exang 1 --oldpeak 1.2 --slope 2 --ca 0 --thal 2

# Launch Streamlit app
streamlit run app/app.py


Features

Data preprocessing & cleaning
Model training & ROC-AUC evaluation
Confusion matrix, ROC, PR curve, feature importance plots
CLI prediction with treatment recommendations
Streamlit web app for interactive prediction
