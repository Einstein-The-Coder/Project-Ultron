<<<<<<< HEAD
import json
=======
import numpy as np
import joblib
import torch
>>>>>>> eff8e9926e1ef9d9e7e7b410c3bebda216e00d65

<<<<<<< HEAD
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .Ultron import ArtificialMind


ultron = ArtificialMind()


def index(request):
    return render(request, "classifier/index.html")


@csrf_exempt
def think(request):

    if request.method != "POST":
        return JsonResponse(
            {"error": "POST request required."},
            status=405
        )

    try:
        data = json.loads(request.body)

        user_input = data.get("input", "").strip()

        if not user_input:
            return JsonResponse(
                {"error": "No input provided."},
                status=400
            )

        perception = ultron.perceive(user_input)

        ultron.focus(perception)

        ultron.evaluate_confidence(1.0)

        thought = ultron.think()

        action = ultron.act(
            "Continue observing and learning."
        )

        ultron.introspect()

        return JsonResponse({
            "attention": ultron.state["attention"],
            "emotion": ultron.state["emotion"],
            "confidence": ultron.state["confidence"],
            "action": ultron.state["last_action"],
            "thought": thought,
            "self_name": ultron.self_model["name"],
            "self_age": ultron.self_model["age"],
            "memories": ultron.long_term_memory
        })

    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=500
        )
=======
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
>>>>>>> eff8e9926e1ef9d9e7e7b410c3bebda216e00d65