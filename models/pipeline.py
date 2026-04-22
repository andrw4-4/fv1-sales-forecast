# -*- coding: utf-8 -*-
"""
Pipeline hibrido Prophet + XGBoost con:
- Optuna optimizando MAE (no SMAPE)
- Walk-forward en test (evaluacion realista)
- Reentrenamiento con todos los datos + prediccion proxima semana
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from models.features import (
    FEATURES_MODELO,
    crear_features,
    expandir_calendario,
    preparar_serie_semanal,
)


# ───────────────────────────────────────────────────────────────
# Prophet helpers
# ───────────────────────────────────────────────────────────────
def entrenar_prophet(serie, vacaciones, params, periods_ahead=0):
    from prophet import Prophet
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=params["seasonality_mode"],
        holidays=vacaciones,
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        holidays_prior_scale=params.get("holidays_prior_scale", 20.0),
        changepoint_range=params["changepoint_range"],
    )
    m.add_seasonality(name="semestral", period=24, fourier_order=5)
    m.add_country_holidays(country_name="CO")
    m.fit(serie)
    future = m.make_future_dataframe(periods=periods_ahead, freq="W-MON")
    forecast = m.predict(future)
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[c] = np.ceil(forecast[c].clip(0)).astype(int)
    return m, forecast


def optimizar_prophet(train_ser, val_ser, vacaciones, n_trials=20):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    val_activo = val_ser[val_ser["y"] > 0]

    def objetivo(trial):
        params = {
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 20.0, log=True),
            "holidays_prior_scale":    trial.suggest_float("holidays_prior_scale", 1.0, 30.0),
            "seasonality_mode":        trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
            "changepoint_range":       trial.suggest_float("changepoint_range", 0.70, 0.95),
        }
        try:
            _, fc = entrenar_prophet(train_ser, vacaciones, params, periods_ahead=len(val_ser))
            fc_val = fc[fc["ds"].isin(val_activo["ds"])][["ds", "yhat"]]
            merged = val_activo.merge(fc_val, on="ds")
            if len(merged) == 0:
                return 9999.0
            return float(np.mean(np.abs(merged["y"].values - merged["yhat"].values)))
        except Exception:
            return 9999.0

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ───────────────────────────────────────────────────────────────
# XGBoost helpers — objetivo MAE
# ───────────────────────────────────────────────────────────────
def optimizar_xgboost(df_train, features, n_trials=30):
    """Optuna sobre TimeSeriesSplit minimizando MAE en escala de ventas reales."""
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
    from xgboost import XGBRegressor
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X = df_train[features].values
    residuo = (df_train["y"] - df_train["yhat"]).values
    y_real = df_train["y"].values
    yhat = df_train["yhat"].values

    def objetivo(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 7),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 15),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 5.0),
            random_state=42, n_jobs=-1, verbosity=0,
        )
        tscv = TimeSeriesSplit(n_splits=3)
        maes = []
        for tr, va in tscv.split(X):
            model = XGBRegressor(**params)
            model.fit(X[tr], residuo[tr], verbose=False)
            pred = np.clip(yhat[va] + model.predict(X[va]), 0, None)
            maes.append(mean_absolute_error(y_real[va], pred))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def walk_forward_test(df_modelo, corte_test, features, xgb_params):
    """Re-entrena para cada semana de test y predice esa semana (evaluacion realista)."""
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error

    reales, preds, fechas = [], [], []
    n = len(df_modelo)
    for i in range(corte_test, n):
        train_wf = df_modelo.iloc[:i]
        target = df_modelo.iloc[[i]]
        model = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(train_wf[features], train_wf["y"] - train_wf["yhat"])
        residuo_pred = model.predict(target[features])
        pred_final = np.clip(target["yhat"].values + residuo_pred, 0, None)
        reales.append(target["y"].values[0])
        preds.append(pred_final[0])
        fechas.append(target["ds"].values[0])

    reales = np.array(reales)
    preds = np.array(preds)
    mae = mean_absolute_error(reales, preds)
    return {
        "fechas": fechas, "reales": reales, "predicciones": preds, "mae": mae,
    }


# ───────────────────────────────────────────────────────────────
# Pipeline completo por producto
# ───────────────────────────────────────────────────────────────
def pipeline_producto(ventas, producto, vacaciones,
                      n_trials_prophet=20, n_trials_xgb=30,
                      excluir_cierre=True):
    """
    1. Construye la serie
    2. Split 80/20 para tuning Prophet
    3. Prophet sobre serie completa con best params
    4. Features + split 80/20 para XGBoost
    5. Optuna XGBoost con TimeSeriesSplit (MAE)
    6. Evaluacion walk-forward en test
    7. Reentrena final sobre TODOS los datos
    8. Predice proxima semana
    Devuelve dict con resultados y prediccion.
    """
    # ── 1. Serie limpia
    serie = preparar_serie_semanal(ventas, producto)
    if excluir_cierre:
        serie = serie[~(
            (serie["ds"].dt.year == 2024) & (serie["ds"].dt.month.isin([10, 11, 12]))
        )].reset_index(drop=True)

    if len(serie) < 40:
        return {"producto": producto, "error": "serie muy corta", "n_semanas": len(serie)}

    # ── 2. Tuning Prophet (80/20)
    corte = int(len(serie) * 0.80)
    train_ser = serie.iloc[:corte]
    val_ser = serie.iloc[corte:]
    best_prophet = optimizar_prophet(train_ser, val_ser, vacaciones, n_trials=n_trials_prophet)

    # ── 3. Prophet final sobre serie completa
    _, forecast_full = entrenar_prophet(serie, vacaciones, best_prophet, periods_ahead=2)

    forecast_hist = forecast_full[forecast_full["ds"].isin(serie["ds"])][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()

    # ── 4. Features
    df_feat = crear_features(serie, vacaciones).merge(forecast_hist, on="ds", how="left")
    df_modelo = df_feat.dropna(subset=FEATURES_MODELO).copy().reset_index(drop=True)

    if len(df_modelo) < 25:
        return {"producto": producto, "error": "pocas filas modelables",
                "n_semanas": len(df_modelo)}

    corte_test = int(len(df_modelo) * 0.80)

    # ── 5. Tuning XGBoost (TimeSeriesSplit en train)
    train_h = df_modelo.iloc[:corte_test]
    best_xgb = optimizar_xgboost(train_h, FEATURES_MODELO, n_trials=n_trials_xgb)

    # ── 6. Walk-forward en test
    wf = walk_forward_test(df_modelo, corte_test, FEATURES_MODELO, best_xgb)

    # Baseline Prophet en test (para comparar)
    from sklearn.metrics import mean_absolute_error
    test_slice = df_modelo.iloc[corte_test:]
    mae_prophet = mean_absolute_error(test_slice["y"], test_slice["yhat"])

    # ── 7. Reentrenar con TODOS los datos
    from xgboost import XGBRegressor
    modelo_final = XGBRegressor(**best_xgb, random_state=42, n_jobs=-1, verbosity=0)
    modelo_final.fit(df_modelo[FEATURES_MODELO], df_modelo["y"] - df_modelo["yhat"])

    # ── 8. Predecir proxima semana
    proxima_semana = (serie["ds"].max() + pd.Timedelta(weeks=1)).normalize()
    # alinear al lunes
    proxima_semana = proxima_semana - pd.Timedelta(days=proxima_semana.weekday())

    serie_extendida = expandir_calendario(serie, semanas_adicionales=1)
    feat_ext = crear_features(serie_extendida, vacaciones)
    feat_ext = feat_ext.merge(
        forecast_full[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds", how="left"
    )
    fila_prox = feat_ext[feat_ext["ds"] == proxima_semana]
    if fila_prox.empty:
        # fallback: ultima fila con yhat
        fila_prox = feat_ext.dropna(subset=["yhat"]).tail(1)
        proxima_semana = fila_prox["ds"].iloc[0] if not fila_prox.empty else proxima_semana

    # Rellenar NaN en features con 0 (para la fila futura algunos lags no existen)
    for col in FEATURES_MODELO:
        if col not in fila_prox.columns:
            fila_prox = fila_prox.assign(**{col: 0})
    fila_prox = fila_prox[FEATURES_MODELO].fillna(0)

    if len(fila_prox) > 0:
        yhat_prox = float(fila_prox["yhat"].iloc[0])
        residuo_prox = float(modelo_final.predict(fila_prox.values)[0])
        prediccion = max(0.0, yhat_prox + residuo_prox)
    else:
        prediccion = 0.0
        yhat_prox = 0.0

    return {
        "producto": producto,
        "n_semanas": len(serie),
        "fecha_proxima_semana": proxima_semana,
        "prediccion_proxima_semana": round(prediccion, 1),
        "prophet_proxima_semana": round(yhat_prox, 1),
        "mae_test_walkforward": round(wf["mae"], 2),
        "mae_test_prophet_solo": round(mae_prophet, 2),
        "mejora_mae": round(mae_prophet - wf["mae"], 2),
        "best_prophet_params": best_prophet,
        "best_xgb_params": best_xgb,
        "historial_test": pd.DataFrame({
            "ds": wf["fechas"],
            "real": wf["reales"],
            "prediccion": wf["predicciones"],
        }),
    }
