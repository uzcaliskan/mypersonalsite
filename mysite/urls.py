from django.urls import path
from . import views



urlpatterns = [
  path("", views.home, name="home"),
  path("projeler", views.projeler, name="projeler"),
  path("egitimler", views.egitimler, name="egitimler"),

  path("egitimler/<slug:slugname>", views.egitim_details, name="egitim_details"),
  path("projeler/<slug:slugname>", views.proje_details, name="proje_details")
]