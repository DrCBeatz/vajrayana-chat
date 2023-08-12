from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("get_title/", views.get_title, name="get_title"),
    path("experts/", views.ExpertListView.as_view(), name="expert-list"),
    path("experts/new/", views.ExpertCreateView.as_view(), name="expert-create"),
    path("experts/<int:pk>/", views.ExpertDetailView.as_view(), name="expert-detail"),
    path(
        "experts/<int:pk>/edit/", views.ExpertUpdateView.as_view(), name="expert-update"
    ),
    path(
        "experts/<int:pk>/delete/",
        views.ExpertDeleteView.as_view(),
        name="expert-delete",
    ),
    path("documents/", views.DocumentListView.as_view(), name="document-list"),
    path("documents/new/", views.DocumentCreateView.as_view(), name="document-create"),
    path(
        "documents/<int:pk>/",
        views.DocumentDetailView.as_view(),
        name="document-detail",
    ),
    path(
        "documents/<int:pk>/edit/",
        views.DocumentUpdateView.as_view(),
        name="document-update",
    ),
    path(
        "documents/<int:pk>/delete/",
        views.DocumentDeleteView.as_view(),
        name="document-delete",
    ),
]
