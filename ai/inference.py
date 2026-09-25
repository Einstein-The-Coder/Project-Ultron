import torch

from .model import UltronModel
from .preprocessing import prepare_input


class UltronInference:

    def __init__(self, model_path=None):

        self.model = UltronModel()

        if model_path is not None:

            self.model.load_state_dict(
                torch.load(
                    model_path,
                    map_location="cpu"
                )
            )

        self.model.eval()


    def predict(self, data):

        data = prepare_input(data)

        tensor = torch.tensor(
            data,
            dtype=torch.float32
        )

        with torch.no_grad():

            output = self.model(tensor)

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