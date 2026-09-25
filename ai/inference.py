import os

import joblib
import torch

from .model import UltronModel
from .preprocessing import prepare_input


class UltronInference:

    def __init__(
        self,
        model_path="models/ultron_digits.pth",
        scaler_path="models/ultron_scaler.pkl"
    ):

        self.model = UltronModel()

        self.model_loaded = False


        # ----------------------------------------------------
        # Load trained model if it exists
        # ----------------------------------------------------

        if os.path.exists(model_path):

            self.model.load_state_dict(
                torch.load(
                    model_path,
                    map_location="cpu"
                )
            )

            self.model_loaded = True


        # ----------------------------------------------------
        # Load scaler
        # ----------------------------------------------------

        self.scaler = None

        if os.path.exists(scaler_path):

            self.scaler = joblib.load(
                scaler_path
            )


        self.model.eval()


    def predict(self, data):

        if not self.model_loaded:

            raise RuntimeError(
                "Ultron model has not been trained yet."
            )


        prepared = prepare_input(data)


        # Apply the same scaling used during training
        if self.scaler is not None:

            prepared = self.scaler.transform(
                prepared
            )


        tensor = torch.tensor(
            prepared,
            dtype=torch.float32
        )


        with torch.no_grad():

            output = self.model(
                tensor
            )


            probabilities = torch.softmax(
                output,
                dim=1
            )


            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()


            confidence = probabilities[
                0,
                prediction
            ].item()


        return {
            "prediction": prediction,
            "confidence": confidence
        }