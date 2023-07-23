from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("get_title/", views.get_title, name="get_title"),
]
