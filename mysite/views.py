from django.shortcuts import render
from .models import Proje, Egitim
import joblib
from .forms import GirisForm
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import datetime as dt
from django.shortcuts import get_object_or_404
from .ModelFiles.yl_codes import forecasting_with_lstm_final
import tensorflow as tf
import keras
import statsmodels


# Create your views here.

file_path = r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\carbon_emmision_info.csv"

def load_univariate_data(path,  country):
  uni_df = pd.read_csv(path, index_col=0)
  uni_country = uni_df[uni_df["Entity"] == country]
  uni_country.set_index("Year", inplace=True)
  uni_country.rename(columns={"Annual CO₂ emissions": "emissions"}, inplace=True)
  uni_country = uni_country["emissions"]

  return uni_country

uni_df_turkey = load_univariate_data(file_path, "Turkey")


def split_univariate_data(data_frame, train_frac=0.6, val_frac=0.2):
    dataframe = data_frame.copy()
    n = len(dataframe)
    train_size = int(n * train_frac)
    val_size = int(n * val_frac)
    test_size = n - train_size - val_size
    train_df = dataframe[:train_size]
    val_df = dataframe[train_size:train_size+val_size]
    test_df = dataframe[train_size+val_size:]
    return train_df, val_df, test_df
train_uni, val_uni, test_uni = split_univariate_data(uni_df_turkey)


def calculate_confidence_interval_univariate(data, confidence=0.95, n_iterations=1000):
  """This function is to calculate confidence intervals for denpended values like time series.
  Bootstrapping is represented in this funtion"""

  n_steps = len(data)

  bootstrap_samples = np.random.normal(loc=data, scale=data * (1-confidence), size=(n_iterations, n_steps))
  percentiles = np.percentile(bootstrap_samples, [(1-confidence) * 100 / 2, (confidence + (1-confidence) / 2)*100], axis=0)
  lower_bound = percentiles[0]
  upper_bound = percentiles[1]
  return lower_bound, upper_bound


def forecast_and_plot(model_fitted, future_years=10):
  forecast_steps = len(test_uni) + future_years
  forecast = model_fitted.forecast(steps=forecast_steps)
  forecast_test = forecast[:len(test_uni)]
  forecast_test.index = test_uni.index
  forecast_future = forecast[len(test_uni):]
  forecast_future.index = np.arange(2023, 2023 + future_years, step=1)
  rmse = np.sqrt(mean_squared_error(test_uni, forecast_test))

  ci_lower_test, ci_upper_test = calculate_confidence_interval_univariate(forecast_test.values)
  ci_lower_future, ci_upper_future = calculate_confidence_interval_univariate(forecast_future.values)

  fig = go.Figure()

  # Gerçek değerleri çiz
  fig.add_trace(go.Scatter(x=pd.concat([pd.Series(train_uni.index), pd.Series(val_uni.index)]),
                           y=pd.concat([pd.Series(train_uni), pd.Series(val_uni)]),
                           mode='lines',
                           name='True Values of Train and Validation'))

  fig.add_trace(go.Scatter(x=pd.concat([pd.Series(train_uni.index), pd.Series(val_uni.index)]),
                           y=model_fitted.fittedvalues,
                           mode='lines',
                           line=dict(dash='dash'),
                           name='Prediction Values of Train and Validation'))

  fig.add_trace(go.Scatter(x=test_uni.index, y=test_uni, mode='lines', name='True Values of Test'))
  fig.add_trace(
      go.Scatter(x=forecast_test.index, y=forecast_test, mode='lines', line=dict(dash='dash'), name='Test Predictions'))

  # Test güven aralıkları
  fig.add_trace(go.Scatter(x=forecast_test.index, y=ci_upper_test, fill=None, mode='lines', line_color='lightblue',
                           name='Test Confidence Upper'))
  fig.add_trace(go.Scatter(x=forecast_test.index, y=ci_lower_test, fill='tonexty', mode='lines', line_color='lightblue',
                           name='Test Confidence Lower', opacity=0.2))

  # Gelecek tahminleri
  fig.add_trace(go.Scatter(x=forecast_future.index, y=forecast_future, mode='lines', line=dict(dash='dash'),
                           name='Future Predictions'))

  # Gelecek güven aralıkları
  fig.add_trace(go.Scatter(x=forecast_future.index, y=ci_upper_future, fill=None, mode='lines', line_color='lightblue',
                           name='Forecast Confidence Upper'))
  fig.add_trace(
      go.Scatter(x=forecast_future.index, y=ci_lower_future, fill='tonexty', mode='lines', line_color='lightblue',
                 name='Forecast Confidence Lower', opacity=0.2))

  # Grafik başlığı ve grid ekleme
  fig.update_layout(title=f"TEST RMSE: {rmse / 1e7:5f}e+07          TEST MEAN: {test_uni.mean() / 1e7:5f}e+07 ",
                    title_x=0.5,
                    xaxis_title='Year',
                    yaxis_title='CO2 Emissions, Metric Tons',
                    legend_title='Legend',
                    template='plotly_white')

  # Grafiği HTML olarak oluştur
  plot_html = fig.to_html(full_html=False)

  return plot_html








def projeler():
    return Proje.objects.all()


def egitimler():
    return Egitim.objects.all()


projects = projeler()
education = egitimler()
context_main = {"project": projects, "education": education}


def home(request):
    return render(request, "base.html", context=context_main)





def proje_details(request, slugname):
    year = dt.datetime.today().year + 1
    projem = get_object_or_404(Proje, slug=slugname)
    model = joblib.load(r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\best_uni_arima.joblib")
    lstm_model = joblib.load(r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\best_lstm_final.joblib")
    scaler_y = joblib.load(r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\scaler_y.joblib")

    if request.method == "POST":
        print(request.POST)
        form = GirisForm(request.POST) #burda sadece integer field alıyor

        if form.is_valid():  # Formun geçerli olduğunu kontrol et
            print(form.cleaned_data.get("yil_tahmin"))
            year = int(request.POST.get("yil_tahmin"))
            modeller = request.POST.get("modeller", "ARIMA")
            if modeller == "ARIMA":
                graph = forecast_and_plot(model, future_years=year - 2022)  # 2022 is the last year
                selected = "ARIMA"
            elif modeller == "LSTM":
                graph = forecasting_with_lstm_final(lstm_model, scaler_y, steps=year - 2022)
                selected = "LSTM"
            context_main["selected"] = selected
            context_main["year"] = int(request.POST.get("yil_tahmin"))
            context_main["graph"] = graph
            context_main["form"] = form
            context_main["error"] = "Geçersiz yıl girişi!"
            print(context_main)
            return render(request, "proje_details.html", context=context_main)
    else:
        year = dt.datetime.today().year + 1
        form = GirisForm({"yil_tahmin": year})
        graph = forecast_and_plot(model, future_years=year-2022)
        context_main["selected"] = "ARIMA"
        context_main["projem"] = projem
        context_main["year"] = year
        context_main["graph"] = graph
        context_main["form"] = form

    return render(request, "proje_details.html", context=context_main)


def egitim_details(request, slugname):
    pass

