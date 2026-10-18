from pathlib import Path

import joblib
import torch

from .model import UltronModel
from .preprocessing import prepare_input


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class UltronInference:
    def __init__(self, model_path=None, scaler_path=None):
        self.model_path = Path(model_path) if model_path else PROJECT_ROOT / "models" / "ultron_digits.pth"
        self.scaler_path = Path(scaler_path) if scaler_path else PROJECT_ROOT / "models" / "ultron_scaler.pkl"

        self.device = torch.device("cpu")
        self.model = UltronModel().to(self.device)
        self.model_loaded = False
        self.scaler = None

        if self.model_path.is_file():
            state_dict = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state_dict)
            self.model_loaded = True

        if self.scaler_path.is_file():
            # Load only scaler files produced by this project's trusted training script.
            self.scaler = joblib.load(self.scaler_path)

        self.model.eval()

    def predict(self, data):
        if not self.model_loaded:
            raise RuntimeError(
                "Ultron model has not been trained yet. Run python -m ai.train first."
            )

        prepared = prepare_input(data)
        if self.scaler is not None:
            prepared = self.scaler.transform(prepared)

        tensor = torch.as_tensor(prepared, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = probabilities.max(dim=1)

        return {
            "prediction": int(predicted_class.item()),
            "confidence": float(confidence.item()),
            "probabilities": [float(value) for value in probabilities[0].cpu().tolist()],
        }
