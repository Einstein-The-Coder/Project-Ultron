# Project Ultron

Project Ultron is an Ultron-themed (Marvel-inspired) demo web app for
handwritten digit recognition. Draw a digit on the canvas, and a PyTorch
neural network classifies it in real time through a Django backend.

It's a learning project exploring a full ML pipeline end to end: dataset,
training, evaluation, model persistence, inference, and a themed web
front end to interact with it.

## How it works

1. You draw a digit (0–9) on an HTML5 canvas in the browser.
2. The client downsamples the drawing to an 8x8 grayscale image and scales
   pixel values to match scikit-learn's `digits` dataset format (0–16).
3. The pixel data is POSTed to a Django endpoint (`/predict/`).
4. A PyTorch feedforward neural network (trained on `sklearn.datasets.load_digits`)
   predicts the digit and returns a confidence score.
5. The result is displayed in the "ULTRON CORE" dashboard UI.

## Architecture

- **PyTorch** — neural network model, training, and inference
- **Django** — backend web server and `/predict/` API endpoint
- **HTML5 Canvas + vanilla JS** — drawing interface and front end
- **scikit-learn** — dataset and preprocessing (`StandardScaler`)

## Project structure

```
project-ultron/
├── ai/                  # Model definition, training, inference
│   ├── model.py
│   ├── training.py
│   ├── train.py
│   ├── inference.py
│   └── preprocessing.py
├── classifier/           # Django app (views, urls)
├── ultron/                # Django project (settings, urls, wsgi/asgi)
├── templates/ultron/      # index.html
├── static/ultron/         # app.js, style.css
├── models/                 # Saved model weights + scaler
├── data/                   # (reserved for future datasets)
└── manage.py
```

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/Einstein-The-Coder/Project-Ultron.git
cd Project-Ultron

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates models/ultron_digits.pth + ultron_scaler.pkl)
python -m ai.train

# 4. Run the Django server
python manage.py migrate
python manage.py runserver
```

Then open `http://127.0.0.1:8000/` in your browser and draw a digit.

## Status

Development build (v0.1) — a personal/learning project, not intended for
production use.

## License

MIT — see `LICENSE`.