import math

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from ai.inference import UltronInference


ultron = UltronInference()


@ensure_csrf_cookie
def index(request):
    return render(request, "ultron/index.html")


def predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=405)

    raw_pixels = request.POST.get("pixels", "").strip()
    if not raw_pixels:
        return JsonResponse({"error": "No pixel data received"}, status=400)

    try:
        pixels = [float(value.strip()) for value in raw_pixels.split(",")]
    except (TypeError, ValueError):
        return JsonResponse({"error": "Pixels must be comma-separated numbers"}, status=400)

    if len(pixels) != 64:
        return JsonResponse({"error": "Expected exactly 64 pixels"}, status=400)

    if not all(math.isfinite(value) for value in pixels):
        return JsonResponse({"error": "Pixel values must be finite numbers"}, status=400)

    if any(value < 0 or value > 16 for value in pixels):
        return JsonResponse({"error": "Pixel values must be between 0 and 16"}, status=400)

    try:
        result = ultron.predict(pixels)
    except RuntimeError:
        return JsonResponse(
            {"error": "The AI model is not ready. Train the model and try again."},
            status=503,
        )
    except (ValueError, TypeError):
        return JsonResponse(
            {"error": "The supplied pixel data could not be processed"},
            status=400,
        )
    except Exception:
        # Do not expose internal paths, model details, or tracebacks in API responses.
        return JsonResponse({"error": "An internal prediction error occurred"}, status=500)

    return JsonResponse(result)
