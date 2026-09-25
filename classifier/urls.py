import numpy as np
import joblib
import torch

from django.shortcuts import render
from django.http import JsonResponse

from .ml_model import model


scaler = joblib.load("ml/scaler.pkl")


def index(request):
    return render(request, "classifier/index.html")


def predict(request):

    if request.method != "POST":
        return JsonResponse(
            {"error": "POST request required"},
            status=405
        )

    try:
        data = request.POST.get("pixels")

        if not data:
            return JsonResponse(
                {"error": "No pixel data received"},
                status=400
            )

        pixels = np.array(
            [float(x) for x in data.split(",")],
            dtype=np.float32
        )

        if len(pixels) != 64:
            return JsonResponse(
                {"error": "Expected 64 pixels"},
                status=400
            )

        pixels = pixels.reshape(1, -1)

        pixels = scaler.transform(pixels)

        tensor = torch.tensor(
            pixels,
            dtype=torch.float32
        )

        with torch.no_grad():

            output = model(tensor)

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

        return JsonResponse({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=500
        )