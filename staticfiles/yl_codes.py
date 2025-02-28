import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample


def split_data(dataframe, country, train_frac=0.6, val_frac=0.2):
    # dataframe = data_frame.copy()
    dataframe = dataframe.rename(columns={"Annual CO₂ emissions": "target"})

    dataframe = dataframe[dataframe["Entity"] == country]
    dataframe.set_index('Year', inplace=True)
    dataframe.drop(["Entity", "Code"], axis=1, inplace=True)
    n = len(dataframe)
    train_size = int(n * train_frac)
    val_size = int(n * val_frac)
    test_size = n - train_size - val_size
    train_df = dataframe[:train_size]
    val_df = dataframe[train_size:train_size + val_size]
    test_df = dataframe[train_size + val_size:]

    train_x = train_df.drop("target", axis=1)
    train_y = train_df["target"]

    val_x = val_df.drop("target", axis=1)
    val_y = val_df["target"]

    test_x = test_df.drop("target", axis=1)
    test_y = test_df["target"]
    return train_x, train_y, val_x, val_y, test_x, test_y, train_df, val_df, test_df


def scale_and_split_data(dataframe, country, scaler_x, scaler_y):
    main_train_x, main_train_y, main_val_x, main_val_y, main_test_x, main_test_y, train_df, val_df, test_df = split_data(
        dataframe, country)
    train_df.to_csv("train_df.csv")
    val_df.to_csv("val_df.csv")
    test_df.to_csv("test_df.csv")

    main_train_x_scaled = scaler_x.fit_transform(main_train_x)
    main_train_y_scaled = scaler_y.fit_transform(main_train_y.values.reshape(-1, 1)).flatten()

    main_val_x_scaled = scaler_x.transform(main_val_x)
    main_val_y = scaler_y.transform(main_val_y.values.reshape(-1, 1)).flatten()

    main_test_x_scaled = scaler_x.transform(main_test_x)
    main_test_y_scaled = scaler_y.transform(main_test_y.values.reshape(-1, 1)).flatten()
    return (scaler_x, scaler_y, main_train_x_scaled,
            main_train_y_scaled, main_val_x_scaled,
            main_val_y, main_test_x_scaled, main_test_y_scaled,
            train_df, val_df, test_df)


def create_timewindow(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys), time_steps


def prepare_data(train_x, train_y, val_x, val_y, test_x, test_y, time_steps=5):
    X_train, y_train, time_steps = create_timewindow(train_x, train_y, time_steps=time_steps)
    X_val, y_val, time_steps = create_timewindow(val_x, val_y, time_steps=time_steps)
    X_test, y_test, time_steps = create_timewindow(test_x, test_y, time_steps=time_steps)
    return X_train, y_train, X_val, y_val, X_test, y_test


# data = pd.read_csv(r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\carbon_emmision_info.csv",
#                    index_col=0)
# print(data.columns)
# scaler_x = MinMaxScaler()
# scaler_y = joblib.load(r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\scaler_y.joblib")
#
# scaler_x, scaler_y, train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, test_x_lstm, test_y_lstm, train_lstm, val_lstm, test_lstm = \
#     scale_and_split_data(data, "Turkey", scaler_x, scaler_y)
#
# X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm = prepare_data(train_x_lstm,
#                                                                                             train_y_lstm,
#                                                                                             val_x_lstm,
#                                                                                             val_y_lstm,
#                                                                                             test_x_lstm,
#                                                                                             test_y_lstm)


def forecast_lstm_final(lstm_model, steps=10):
    # Veriyi yükle
    data = pd.read_csv(
        r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\carbon_emmision_info.csv",
        index_col=0)
    print("Sütunlar:", data.columns)

    # Sütun adlarını düzenle ve gereksiz sütunları kaldır
    data = data.rename(columns={"Annual CO₂ emissions": "target"})
    if "co2" in data.columns:
        data.drop("co2", axis=1, inplace=True)

    # Eksik değerleri kaldır
    data = data.dropna()

    # Ölçeklendirme ve veri bölme
    scaler_x = MinMaxScaler()
    scaler_y = joblib.load(r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\scaler_y.joblib")
    scaler_x, scaler_y, train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, test_x_lstm, test_y_lstm, train_lstm, val_lstm, test_lstm = \
        scale_and_split_data(data, "Turkey", scaler_x, scaler_y)

    # Zaman penceresi oluştur
    X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm = prepare_data(
        train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, test_x_lstm, test_y_lstm
    )
    print(X_test_lstm.shape)
    # Tahminleri yap
    train_predictions = lstm_model.predict(X_train_lstm)
    val_predictions = lstm_model.predict(X_val_lstm)
    test_predictions = lstm_model.predict(X_test_lstm)

    # Gelecek tahminleri yap
    X_test = X_test_lstm[-1:]
    print(X_test_lstm.shape)
    np.save("X_test.npy", X_test)
    future_predictions = []
    for _ in range(steps):
        prediction = lstm_model.predict(X_test.reshape(1, X_test.shape[1], X_test.shape[2]))
        future_predictions.append(prediction[0, 0])
        prediction_reshaped = np.repeat(prediction, 13).reshape(1, 1, 13)
        X_test = np.concatenate((X_test[:, 1:], prediction_reshaped), axis=1)

    # Ölçeklendirmeyi geri al
    train_predictions_rescaled = scaler_y.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
    val_predictions_rescaled = scaler_y.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
    test_predictions_rescaled = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    future_predictions_rescaled = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Gerçek değerleri ölçeklendirmeyi geri al
    y_train_rescaled = scaler_y.inverse_transform(y_train_lstm.reshape(-1, 1)).flatten()
    y_val_rescaled = scaler_y.inverse_transform(y_val_lstm.reshape(-1, 1)).flatten()
    y_test_rescaled = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()

    # Sonuçları kaydet
    pd.DataFrame({"train_predictions": train_predictions_rescaled,
                  "y_train": y_train_rescaled}).to_csv("train_predictions.csv")

    pd.DataFrame({"test_predictions": test_predictions_rescaled,
                  "y_test": y_test_rescaled}).to_csv("test_predictions.csv")

    pd.DataFrame({"val_predictions": val_predictions_rescaled,
                  "y_val": y_val_rescaled, }).to_csv("val_predictions.csv")

    pd.DataFrame({"future_predictions": future_predictions_rescaled, }).to_csv("future_predictions.csv")

    return "tamamlandı"


# Modeli yükle ve tahminleri yap
model_path = r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\best_lstm_final.joblib"
lstm_model = joblib.load(model_path)


# forecast_lstm_final(lstm_model)
def calculate_confidence_interval_univariate(data, confidence=0.95, n_iterations=1000):
    """This function is to calculate confidence intervals for denpended values like time series.
    Bootstrapping is represented in this funtion"""

    n_steps = len(data)

    bootstrap_samples = np.random.normal(loc=data, scale=data * (1 - confidence), size=(n_iterations, n_steps))
    percentiles = np.percentile(bootstrap_samples,
                                [(1 - confidence) * 100 / 2, (confidence + (1 - confidence) / 2) * 100], axis=0)
    lower_bound = percentiles[0]
    upper_bound = percentiles[1]
    return lower_bound, upper_bound


scaler_y = joblib.load(r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\scaler_y.joblib")


def forecasting_with_lstm_final(lstm_model, scaler_y, steps=10):
    train_predictions = pd.read_csv(
        r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\train_predictions.csv")

    val_predictions = pd.read_csv(
        r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\val_predictions.csv")
    test_predictions = pd.read_csv(
        r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\test_predictions.csv")

    # Gelecek tahminleri yap
    X_test = np.load(r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\X_test.npy")
    future_predictions = []
    for _ in range(steps):
        prediction = lstm_model.predict(X_test.reshape(1, X_test.shape[1], X_test.shape[2]))
        future_predictions.append(prediction[0, 0])
        prediction_reshaped = np.repeat(prediction, 13).reshape(1, 1, 13)
        X_test = np.concatenate((X_test[:, 1:], prediction_reshaped), axis=1)

    # Ölçeklendirmeyi geri al
    future_predictions_rescaled = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    ci_lower_lstm, ci_upper_lstm = calculate_confidence_interval_univariate(future_predictions_rescaled)
    ci_lower_lstm_test, ci_upper_lstm_test = calculate_confidence_interval_univariate(test_predictions["test_predictions"])
    print(ci_lower_lstm)
    print(type(ci_lower_lstm))
    # print(X_test)
    train_df = pd.read_csv(r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\train_df.csv")
    train_years = train_df["Year"]
    # print(train_df)
    val_df = pd.read_csv(r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\val_df.csv")
    val_years = val_df["Year"]
    # print(val_df)
    test_df = pd.read_csv(r"C:\Users\Msi\DjangoProjects\MyWebsite\personalwebsite\mysite\ModelFiles\test_df.csv")
    test_years = test_df["Year"]
    # print(test_df)
    # Plotly ile grafik oluşturma
    fig = go.Figure()

    # Eğitim verisi ve tahminleri
    fig.add_trace(go.Scatter(
        x=train_years,
        y=train_predictions["y_train"].values,
        mode='lines',
        name='Eğitim Verisi (Gerçek)',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=train_years,
        y=train_predictions["train_predictions"],
        mode='lines',
        name='Eğitim Tahmini',
        line=dict(color='lightblue', width=2, dash='dash')
    ))
    # validasyon verisi ve tahminleri
    fig.add_trace(go.Scatter(
        x=val_years,
        y=val_predictions["y_val"],
        mode='lines',
        name='True Values of Validation',
        line=dict(color='#9467bd', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=val_years,
        y=val_predictions["val_predictions"],
        mode='lines',
        name='Validation Prediction',
        line=dict(color='#c5b0d5', width=2, dash='dash')
    ))
    # Test verisi ve tahminleri
    fig.add_trace(go.Scatter(
        x=test_years,
        y=test_predictions["y_test"],
        mode='lines',
        name='True Values of Test',
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=test_years,
        y=test_predictions["test_predictions"],
        mode='lines',
        name='Test Predictions',
        line=dict(color='lightgreen', width=2, dash='dash')
    ))

    # Gelecek tahminleri
    fig.add_trace(go.Scatter(
        x=np.arange(2023, 2022 + steps + 1),
        y=future_predictions_rescaled,
        mode='lines',
        name='Future Predictions',
        line=dict(color='red', width=2, dash="dash")
    ))
    # Güven Aralığını Gösterme (Gölgeli Bölge)
    fig.add_trace(go.Scatter(
        x=np.concatenate([np.arange(2023, 2022 + steps + 1), np.arange(2023, 2022 + steps + 1)[::-1]]),
        # X eksenini birleştir
        y=np.concatenate([ci_lower_lstm, ci_upper_lstm[::-1]]),  # Y eksenini birleştir
        fill='toself',  # Gölgeli bölge oluştur
        fillcolor='rgba(255, 0, 0, 0.2)',  # Kırmızı, yarı saydam
        line=dict(color='rgba(255, 255, 255, 0)'),  # Çizgi rengini şeffaf yap
        name='Forecast Confidence Interval',
        showlegend=True
    ))


    # Grafik düzenlemeleri
    test_rmse = np.sqrt(mean_squared_error(test_predictions["y_test"], test_predictions["test_predictions"]))
    test_mean = test_predictions["y_test"].mean()
    fig.update_layout(
        title=f"TEST RMSE: {test_rmse / 1e7:5f}e+07          TEST MEAN: {test_mean / 1e7:5f}e+07",
        title_x=0.5,  # Başlık ortalama için
        xaxis_title="Year",
        yaxis_title="CO2 Emissions, Metric Tons",
        template="plotly_white"
    )

    # Grafiği HTML olarak dönüştürme
    graph = fig.to_html(full_html=False)

    return graph


graph = forecasting_with_lstm_final(lstm_model, scaler_y, steps=10)
