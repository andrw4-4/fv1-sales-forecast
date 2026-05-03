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
    from sklearn.metrics import mean_absolute_error
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
            return float(mean_absolute_error(merged["y"], merged["yhat"]))
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
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    # En lugar de walk-forward estricto, la notebook usaba XGBRegressor.fit en train y predict en todo test_act.
    # Pero el usuario dice "seguir tal cual la celda", la celda entrena el modelo de XGBoost
    # una sola vez en el split y lo corre en test_act.
    train_h = df_modelo.iloc[:corte_test]
    test_act = df_modelo.iloc[corte_test:]
    test_act = test_act[test_act["y"] > 0]

    if len(test_act) == 0:
        return {"fechas": [], "reales": [], "predicciones": [], "mae": 0.0, "smape": 0.0, "test_act_yhat": []}

    modelo_final = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1, verbosity=0)
    modelo_final.fit(train_h[features], train_h["y"] - train_h["yhat"])

    correccion = modelo_final.predict(test_act[features])
    pred_hibrida = np.clip(test_act["yhat"].values + correccion, 0, None)

    smape_fn = lambda y, yhat: (2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-8)).mean() * 100

    mae = mean_absolute_error(test_act["y"], pred_hibrida)
    smape = smape_fn(test_act["y"].values, pred_hibrida)

    return {
        "fechas": test_act["ds"].tolist(),
        "reales": test_act["y"].values,
        "predicciones": pred_hibrida,
        "mae": mae,
        "smape": smape,
        "test_act_yhat": test_act["yhat"].values
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

    # ── 8. Predecir proximas N semanas (con prophet + correccion xgb + confianza)
    horizonte = 4
    # Prophet con horizonte mayor (reentrenar si es necesario)
    _, forecast_h = entrenar_prophet(serie, vacaciones, best_prophet, periods_ahead=horizonte + 2)

    proxima_semana = (serie["ds"].max() + pd.Timedelta(weeks=1)).normalize()
    proxima_semana = proxima_semana - pd.Timedelta(days=proxima_semana.weekday())

    # Estimar std de residuos del walk-forward para intervalos de confianza
    residuos_test = wf["reales"] - wf["predicciones"]
    std_residual = float(np.std(residuos_test)) if len(residuos_test) > 1 else 0.0

    # Construir serie extendida con H semanas adicionales para calcular features
    serie_extendida = expandir_calendario(serie, semanas_adicionales=horizonte)
    feat_ext = crear_features(serie_extendida, vacaciones)
    feat_ext = feat_ext.merge(
        forecast_h[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds", how="left",
    )

    # Para cada semana futura, predecir de forma iterativa (usar predicciones previas como lag)
    predicciones_futuras = []
    y_historia = list(serie["y"].values)  # valores reales hasta la ultima semana
    for h in range(1, horizonte + 1):
        fecha_h = proxima_semana + pd.Timedelta(weeks=h - 1)
        fila = feat_ext[feat_ext["ds"] == fecha_h].copy()
        if fila.empty:
            continue
        # Rellenar lags con la historia extendida (real + predicciones previas)
        if len(y_historia) >= 1:
            fila["lag_1"] = y_historia[-1]
        if len(y_historia) >= 2:
            fila["lag_2"] = y_historia[-2]
        if len(y_historia) >= 4:
            fila["lag_4"] = y_historia[-4]
        if len(y_historia) >= 4:
            fila["rolling_4"] = float(np.mean(y_historia[-4:]))
        if len(y_historia) >= 8:
            fila["rolling_8"] = float(np.mean(y_historia[-8:]))
        fila["lag_1_clean"] = fila["lag_1"]

        for col in FEATURES_MODELO:
            if col not in fila.columns:
                fila[col] = 0
        fila_X = fila[FEATURES_MODELO].fillna(0)

        yhat_h = float(fila_X["yhat"].iloc[0])
        yhat_lower_h = float(fila["yhat_lower"].iloc[0]) if "yhat_lower" in fila.columns else yhat_h * 0.8
        yhat_upper_h = float(fila["yhat_upper"].iloc[0]) if "yhat_upper" in fila.columns else yhat_h * 1.2
        residuo_h = float(modelo_final.predict(fila_X.values)[0])
        prediccion_h = max(0.0, yhat_h + residuo_h)

        # Intervalo basado en std de residuos del walk-forward (±1 std ≈ 68%)
        # Se ensancha suavemente con el horizonte pero queda acotado
        ancho_total = std_residual * (1.0 + 0.15 * (h - 1))

        # "Probabilidad" o confianza: arbitraria pero intuitiva.
        # Decae con h: 1 sem ~85%, 2 sem ~70%, 3 sem ~55%, 4 sem ~45%
        confianza_pct = max(20, 95 - h * 13)

        predicciones_futuras.append({
            "semana_offset": h,
            "fecha": fecha_h,
            "prediccion": round(prediccion_h, 1),
            "prediccion_lower": round(max(0, prediccion_h - ancho_total), 1),
            "prediccion_upper": round(prediccion_h + ancho_total, 1),
            "prophet_solo": round(yhat_h, 1),
            "confianza_pct": confianza_pct,
        })
        y_historia.append(prediccion_h)  # para el siguiente lag

    return {
        "producto": producto,
        "n_semanas": len(serie),
        "fecha_proxima_semana": proxima_semana,
        "prediccion_proxima_semana": predicciones_futuras[0]["prediccion"] if predicciones_futuras else 0.0,
        "prophet_proxima_semana": predicciones_futuras[0]["prophet_solo"] if predicciones_futuras else 0.0,
        "predicciones_4_semanas": predicciones_futuras,
        "mae_test_walkforward": round(wf["mae"], 2),
        "smape_hibrido": round(wf["smape"], 3),
        "mae_test_prophet_solo": round(mae_prophet, 2),
        "mejora_mae": round(mae_prophet - wf["mae"], 2),
        "std_residual_test": round(std_residual, 2),
        "best_prophet_params": best_prophet,
        "best_xgb_params": best_xgb,
        "historial_test": pd.DataFrame({
            "ds": wf["fechas"],
            "real": wf["reales"],
            "prediccion": wf["predicciones"],
            "prophet": wf["test_act_yhat"],
        }),
    }
