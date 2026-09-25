from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("think/", views.think, name="think"),
]