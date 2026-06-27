# AI Disease Symptom Checker

A machine learning web app that analyses selected symptoms and suggests possible medical conditions, built with Random Forest classification.

## Demo

Live app: [https://disease-symptom-checker-bt6bxq8gz5jirnhazdr3pp.streamlit.app/]

## What it does

- Select symptoms from organised categories (fever, respiratory, digestive, pain, etc.)
- The AI analyses the combination and predicts the top 3 most likely conditions with confidence scores
- Covers 12 diseases and 60+ symptoms including common conditions relevant to Kerala and India (dengue, malaria, typhoid, COVID-19)
- Clearly displays a medical disclaimer — this is an educational tool, not a diagnostic system

## Diseases covered

Common Cold · Influenza · COVID-19 · Dengue Fever · Malaria · Typhoid · Gastroenteritis · Migraine · Hypertension · Diabetes (Type 2) · Pneumonia · Asthma

## Tech stack

- Python
- Streamlit — for the web interface
- Scikit-learn — for the Random Forest Classifier
- Pandas — for building the feature matrix
- NumPy — for probability sorting and array operations

## How to run this locally

1. Clone this repository
   ```
   git clone https://github.com/YOUR-USERNAME/disease-symptom-checker.git
   cd disease-symptom-checker
   ```

2. Install the required packages
   ```
   pip install -r requirements.txt
   ```

3. Run the app
   ```
   streamlit run app.py
   ```

4. Your browser will open at `http://localhost:8501`

5. No API key or external dataset download needed — the model trains instantly on startup from the built-in dataset

## How it works

Each disease in the dataset is represented by several symptom patterns (rows). Each symptom is a binary feature (1 = present, 0 = absent). A Random Forest Classifier is trained on this structured dataset and learns which symptom combinations most reliably indicate each disease.

When you select symptoms, the app builds the same binary feature vector and passes it to the trained model. The model returns a probability for every disease — the top 3 are displayed with confidence bars.

**Model performance:** 100% accuracy on the test split, which is expected for a clean structured dataset where symptom patterns are distinct across diseases.

## What I learned building this

- Multi-class classification with Random Forest and Scikit-learn
- Label encoding — converting string class names to numbers and back
- Binary feature engineering — representing presence/absence of symptoms as 0/1 columns
- `predict_proba()` — getting probability scores for each class instead of just one prediction
- Responsible AI design — including a clear medical disclaimer and framing the tool as educational
- Organising a large number of UI inputs (60+ symptoms) into collapsible grouped sections in Streamlit

## Limitations

**This tool is for educational purposes only and is NOT a substitute for professional medical advice.**

- The dataset is structured and synthetic — real symptom patterns are far more variable and overlapping than this model captures
- Many diseases share symptoms (e.g. high fever + headache appears in flu, dengue, malaria, and typhoid), and the model may not differentiate these reliably when only a few symptoms are selected
- The model has no access to patient history, test results, physical examination findings, or any clinical context — all of which a real doctor uses
- 100% test accuracy on a small structured dataset does not mean 100% accuracy on real-world patient data
- This is a proof-of-concept demonstrating how ML classification can be applied to medical data, not a clinical tool

## Why these limitations exist

Real medical AI systems (like those used in hospitals) are trained on millions of anonymised patient records, validated through clinical trials, and reviewed by medical professionals before deployment. This project intentionally avoids claiming diagnostic capability because:

1. Misdiagnosis is dangerous — acting on incorrect AI output for a medical condition can cause serious harm
2. The training data is small and structured — real symptom data is messy, overlapping, and patient-specific
3. Responsible AI development means being honest about what a model can and cannot do

## Possible improvements

- Connect to a real clinical dataset (e.g. UCI disease datasets or Kaggle medical datasets)
- Add severity scoring — mild / moderate / severe
- Add a "when to see a doctor urgently" warning based on red-flag symptoms
- Support multiple languages including Malayalam for Kerala users
- Add duration and frequency of symptoms as additional features
