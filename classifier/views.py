from django.http import JsonResponse
from django.shortcuts import render

from ai.inference import UltronInference


ultron = UltronInference()


def index(request):
    return render(request, "ultron/index.html")


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

        pixels = [
            float(x)
            for x in data.split(",")
        ]

        if len(pixels) != 64:
            return JsonResponse(
                {"error": "Expected 64 pixels"},
                status=400
            )

        result = ultron.predict(pixels)

        return JsonResponse(result)

    except ValueError:
        return JsonResponse(
            {"error": "Invalid pixel data"},
            status=400
        )

    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=500
        )