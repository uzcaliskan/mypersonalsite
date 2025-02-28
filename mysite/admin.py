from django.contrib import admin
from .models import Category, Okul, Proje, Egitim


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ["name"]
    list_filter = ["name"]
    search_fields = ["name"]


@admin.register(Okul)
class OkulAdmin(admin.ModelAdmin):
    list_display = ["name", "level", "branch", "status", "start_date", "end_date", "city", "kategoriler"]
    list_filter = ["level", "status", "branch"]
    search_fields = ["level", "status", ]

    def kategoriler(self, obj):
        html = ""
        for cat in obj.categories.all():
            html += html + ", "


@admin.register(Proje)
class ProjeAdmin(admin.ModelAdmin):
    list_display = ["title"]


@admin.register(Egitim)
class EgitimAdmin(admin.ModelAdmin):
    list_display = ["title", "lecturer", "education_provider", "lecturer_provider", "kategoriler"]
    list_filter = ["title", "lecturer", "education_provider", "lecturer_provider"]
    search_fields = ["title", "lecturer", "lecturer_provider"]



    def kategoriler(self, obj):
        html = ""
        for cat in obj.categories.all():
            html += html + ", "