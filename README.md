# AI-textdetektor

Ett neuralt nätverk som klassificerar om en text är skriven av en människa eller genererad av en AI. Projektet vänder på den vanliga frågeställningen — målet är inte enbart att avslöja AI-text, utan att förstå vad som skiljer mänskligt och maskinellt skrivande åt, och använda den kunskapen för att göra AI-kommunikation varmare och mer personlig. Verktyget är särskilt relevant för organisationer som psykologmottagningar och vårdinstanser som vill förbättra sin skriftliga patientkommunikation.

Modellen uppnår cirka **80% noggrannhet** på testdata.

---

## Projektstruktur

```
aiTextDetector/
├── train.py                    # Tränar och sparar modellen
├── app.py                      # Streamlit-webbapp för textanalys
├── config.py                   # Alla hyperparametrar och inställningar
├── eda.py                      # Utforskande dataanalys
├── evaluate_model.py           # Utvärderingsplots (ROC, konfusionsmatris m.m.)
├── feature_importance.py       # Permutationsimportans
├── text_features.py            # Källrensning och textstatistik
├── ai_textdetektor.ipynb       # Jupyter Notebook — hela pipeline på svenska
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Träna modellen

Alla inställningar styrs från `config.py` — inga ändringar behövs i övriga filer.

```bash
python train.py
```

Scriptet frågar om du vill köra grid search eller vanlig träning. Vid vanlig träning sparas följande filer:

- `ann_ai_detector_model.keras`
- `tfidf_vectorizer.joblib`
- `tfidf_char_vectorizer.joblib`
- `label_encoder.joblib`
- `scaler.joblib`

### Grid search

Välj grid search i prompten för att optimera hyperparametrar. Grid search körs på ett urval av datan för att hålla körtiden rimlig. Parameternätet definieras i `config.py` under `PARAM_GRID`.

---

## Starta webbappen

```bash
streamlit run app.py
```

Appen öppnas i webbläsaren på `http://localhost:8501`. Klistra in valfri text och få ett svar direkt — modellen visar konfidenspoäng, textstatistik och vilka ord som påverkade beslutet mest. En analysrapport kan laddas ned som CSV.

> Modellen måste vara tränad innan appen startas. Om modellfiler saknas aktiveras ett demo-läge automatiskt.

---

## Konfiguration (`config.py`)

| Parameter | Beskrivning |
|---|---|
| `SAMPLES_PER_CLASS` | Antal texter per klass att sampla för EDA |
| `TFIDF_MAX_FEATURES` | Max antal features i ord-TF-IDF |
| `TFIDF_CHAR_MAX_FEATURES` | Max antal features i tecken-TF-IDF |
| `HIDDEN_LAYERS` | Nätverksarkitektur — lista av `(neuroner, aktivering, dropout)` |
| `LEARNING_RATE` | Inlärningshastighet |
| `EPOCHS` | Max antal träningsepoker |
| `ES_PATIENCE` | Early stopping — antal epoker utan förbättring innan stopp |

---

## Metod

Varje text representeras av tre feature-kanaler som kombineras till en enda feature-matris:

1. **Ord-TF-IDF** — ord och bigramer (20 000 features)
2. **Tecken-TF-IDF** — tecken-n-gram 3–4 tecken, `char_wb`-analys (8 000 features)
3. **Textstatistik** — ordantal, genomsnittlig meningslängd, lexikal mångfald

All förbehandling (TF-IDF, skalning) anpassas enbart på träningsdatan för att undvika datainläckage.
