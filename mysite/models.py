from django.core.exceptions import ValidationError
from django.db import models
from django.utils.text import slugify
from ckeditor.fields import RichTextField
from unidecode import unidecode
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Category(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True, blank=True, db_index=True, editable=False)

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        self.slug = slugify(unidecode(self.name))
        super().save(*args, **kwargs)

    class Meta:
        verbose_name_plural = "Kategoriler"


class Egitim(models.Model):
    title = models.CharField(max_length=200)
    lecturer = models.CharField(max_length=150)
    education_provider = models.CharField(max_length=50)
    lecturer_provider = models.CharField(max_length=100)
    description = RichTextField()
    slug = models.SlugField(unique=True, blank=True, db_index=True, editable=False)
    categories = models.ManyToManyField(Category, blank=True, related_name="egitimler")

    def __str__(self):
        return self.title


    def save(self, *args, **kwargs):
        self.slug = slugify(unidecode(self.title))
        super().save(*args, **kwargs)

    class Meta:
        verbose_name_plural = "Eğitimler"


class Proje(models.Model):
    title = models.CharField(max_length=250)
    description = RichTextField()
    image = models.ImageField(upload_to="uploads")
    categories = models.ManyToManyField(Category, blank=True, related_name="projeler")
    slug = models.SlugField(unique=True, blank=True, db_index=True, editable=False)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        self.slug = slugify(unidecode(self.title))
        super().save(*args, **kwargs)

    class Meta:
        verbose_name_plural = "Projeler"


class Okul(models.Model):
    name = models.CharField(max_length=200)
    level = models.CharField(max_length=150)
    branch = models.CharField(max_length=150)
    status = models.CharField(max_length=50)
    start_date = models.DateField()
    end_date = models.DateField()
    city = models.CharField(max_length=50)
    slug = models.SlugField(unique=True, blank=True, db_index=True, editable=False)
    categories = models.ManyToManyField(Category, blank=True, related_name="okullar")

    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = "Okullar"