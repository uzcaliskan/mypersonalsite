from django import forms

class GirisForm(forms.Form):
    yil_tahmin = forms.IntegerField(
        label="Tahmin Yılı",  # Alanın etiketi
        min_value=2023,  # Minimum değer
        max_value=2050,  # Maksimum değer
        required=True,  # Zorunlu alan
        widget=forms.NumberInput(attrs={'step': 1, "class": "form-control text-center"})  # HTML'de step özelliğini ayarla
    )
