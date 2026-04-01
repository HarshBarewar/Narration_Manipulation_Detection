# Narrative Manipulation Detection System

Flask + NLP project that detects whether an article contains:
- Manipulative language
- Religious manipulation
- Political manipulation
- Anti-constitutional content

For each category, the output includes:
- Yes/No flag
- Confidence score
- Which line(s) of text triggered the detection

## Project Structure

- `app.py` - Flask app and API routes
- `detector.py` - Inference pipeline + explainable line-level detection
- `train_model.py` - Train Logistic Regression models and compute evaluation metrics
- `templates/` - Intro and analyzer UI pages
- `static/` - CSS and frontend JavaScript
- `models/` - Saved ML artifacts and metrics JSON
- `vercel.json` - Vercel deployment config

## How to Run Locally

1. Create and activate virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train model and generate metrics:

```powershell
python train_model.py
```

4. Start Flask app:

```powershell
python app.py
```

5. Open browser:
- `http://127.0.0.1:5000`

## API Endpoints

- `GET /` - Intro page with Get Started button
- `GET /analyze` - Analyzer UI
- `POST /api/analyze` - Analyze article text
- `GET /api/metrics` - Model evaluation metrics (accuracy/F1)

### Sample Request

```json
POST /api/analyze
{
  "article": "If you do not act now, everything will collapse..."
}
```

## ML Evaluation

`train_model.py` trains one Logistic Regression classifier per category using TF-IDF features.
It reports:
- Accuracy
- Precision
- Recall
- F1
- Macro averages

Saved to: `models/metrics.json`.

## Deploy on Vercel

1. Install Vercel CLI and login:

```powershell
npm i -g vercel
vercel login
```

2. Deploy:

```powershell
vercel
```

3. Production deploy:

```powershell
vercel --prod
```

## Notes

- Current training data is starter data embedded in `train_model.py` for quick setup.
- For better real-world accuracy, replace with a larger annotated dataset and retrain.
