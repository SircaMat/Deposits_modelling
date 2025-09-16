
# deposit_dynamics_streamlit_tvpvar_keepbest_v9.py
# Streamlit app: Deposit dynamics (Thinking Mode) with SARIMAX, Elastic Net, VAR/VECM, TVP-VAR
# V9 = v8 + minor robustness on date parsing and empty predictor guards.

import os
import re
import json
import math
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed:int=42):
    np.random.seed(seed)

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true, y_pred = pd.Series(y_true).align(pd.Series(y_pred), join="inner")
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true.replace(0, np.nan)))) * 100.0
    return {"RMSE": float(rmse), "MAE": float(mae), "MAPE(%)": float(mape)}

def ljung_box_test(resid: pd.Series, lags: int = 12) -> float:
    resid = pd.Series(resid).dropna()
    if len(resid) < 3: return float('nan')
    lags = min(lags, max(1, len(resid)//3))
    lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
    return float(lb["lb_pvalue"].iloc[-1])

def arch_test(resid: pd.Series, lags: int = 12) -> float:
    resid = pd.Series(resid).dropna()
    if len(resid) < 10: return float('nan')
    stat, pval, *_ = het_arch(resid, nlags=min(lags, max(1, len(resid)//3)))
    return float(pval)

def jb_test(resid: pd.Series) -> float:
    resid = pd.Series(resid).dropna()
    if len(resid) < 10: return float('nan')
    stat, pval, *_ = jarque_bera(resid)
    return float(pval)

def adf_test(series: pd.Series) -> Dict[str, float]:
    series = series.dropna()
    if len(series) < 10:
        return {"stat": np.nan, "pval": np.nan}
    stat, pval, *_ = adfuller(series, autolag="AIC")
    return {"stat": float(stat), "pval": float(pval)}

def kpss_test(series: pd.Series) -> Dict[str, float]:
    series = series.dropna()
    if len(series) < 10:
        return {"stat": np.nan, "pval": np.nan}
    stat, pval, *_ = kpss(series, regression="c", nlags="auto")
    return {"stat": float(stat), "pval": float(pval)}

def uniquify_columns(cols: List[str]) -> List[str]:
    seen: Dict[str,int] = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

def make_lags(df: pd.DataFrame, cols: List[str], max_lag: int) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        colobj = out[c]
        if isinstance(colobj, pd.DataFrame):
            colseries = colobj.iloc[:, 0]
        else:
            colseries = colobj
        for L in range(1, max_lag + 1):
            out[f"{c}_lag{L}"] = colseries.shift(L)
    return out

def vif_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    X_ = sm.add_constant(X.dropna(), has_constant="add")
    cols = [c for c in X_.columns if c != "const"]
    vifs = []
    for i, _ in enumerate(cols):
        try:
            vifs.append(variance_inflation_factor(X_.values, i+1))
        except Exception:
            vifs.append(np.nan)
    return pd.DataFrame({"variable": cols, "VIF": vifs})

def train_test_time_split(df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(np.floor(train_frac * n))
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def timestamped_outdir(base="outputs"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"run_{ts}")
    os.makedirs(path, exist_ok=True)
    return path

def _apply_font_scale(scale: float):
    base = 10.0 * scale
    plt.rcParams.update({
        "font.size": base,
        "axes.titlesize": base * 1.1,
        "axes.labelsize": base,
        "xtick.labelsize": base * 0.9,
        "ytick.labelsize": base * 0.9,
        "legend.fontsize": base * 0.9
    })

def _scale_figsize(base_w: float, base_h: float, scale: float) -> Tuple[float,float]:
    return max(3, base_w * scale), max(2, base_h * scale)

def plot_actual_vs_pred(dates, actual, fitted, title: str, scale: float = 1.0):
    _apply_font_scale(scale)
    fw, fh = _scale_figsize(9, 4, scale)
    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.plot(dates, actual, label="Actual")
    ax.plot(dates, fitted, label="Fitted/Forecast")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def _safe_nlags(n: int, requested: int = 24) -> int:
    if n <= 4:
        return 1
    max_allowed = max(1, (n // 2) - 1)
    return max(1, min(requested, max_allowed))

def plot_residual_diagnostics(resid: pd.Series, title_prefix: str, scale: float = 1.0):
    _apply_font_scale(scale)
    resid = pd.Series(resid).dropna()
    n = len(resid)
    figs = []
    try:
        nl = _safe_nlags(n, 24)
        acf_vals = acf(resid, nlags=nl, fft=False)
        fw, fh = _scale_figsize(7, 3.5, scale)
        fig1, ax1 = plt.subplots(figsize=(fw, fh))
        ax1.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
        ax1.set_title(f"{title_prefix} Residual ACF")
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("ACF")
        ax1.grid(True, alpha=0.3)
        figs.append(fig1)
    except Exception as e:
        fig1, ax1 = plt.subplots(figsize=_scale_figsize(7, 3.5, scale))
        ax1.text(0.5,0.5,f"ACF unavailable: {e}",ha='center',va='center'); ax1.axis('off')
        figs.append(fig1)
    try:
        nl = _safe_nlags(n, 24)
        pacf_vals = pacf(resid, nlags=nl, method="yw")
        fw, fh = _scale_figsize(7, 3.5, scale)
        fig2, ax2 = plt.subplots(figsize=(fw, fh))
        ax2.stem(range(len(pacf_vals)), pacf_vals, basefmt=" ")
        ax2.set_title(f"{title_prefix} Residual PACF")
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("PACF")
        ax2.grid(True, alpha=0.3)
        figs.append(fig2)
    except Exception:
        try:
            nl = _safe_nlags(n, 10)
            pacf_vals = pacf(resid, nlags=nl, method="ols")
            fw, fh = _scale_figsize(7, 3.5, scale)
            fig2, ax2 = plt.subplots(figsize=(fw, fh))
            ax2.stem(range(len(pacf_vals)), pacf_vals, basefmt=" ")
            ax2.set_title(f"{title_prefix} Residual PACF (OLS fallback)")
            ax2.set_xlabel("Lag")
            ax2.set_ylabel("PACF")
            ax2.grid(True, alpha=0.3)
            figs.append(fig2)
        except Exception as e2:
            fig2, ax2 = plt.subplots(figsize=_scale_figsize(7, 3.5, scale))
            ax2.text(0.5,0.5,f"PACF unavailable: {e2}",ha='center',va='center'); ax2.axis('off')
            figs.append(fig2)
    try:
        fw, fh = _scale_figsize(7, 3.5, scale)
        fig3, ax3 = plt.subplots(figsize=(fw, fh))
        bins = max(10, min(30, n//2 if n>0 else 10))
        ax3.hist(resid, bins=bins)
        ax3.set_title(f"{title_prefix} Residual Histogram")
        ax3.set_xlabel("Residual")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)
        figs.append(fig3)
    except Exception as e:
        fig3, ax3 = plt.subplots(figsize=_scale_figsize(7, 3.5, scale))
        ax3.text(0.5,0.5,f"Histogram unavailable: {e}",ha='center',va='center'); ax3.axis('off')
        figs.append(fig3)
    return figs

def top_drivers_bar(names: List[str], effects: List[float], title: str, scale: float = 1.0):
    _apply_font_scale(scale)
    fw, fh = _scale_figsize(7, 3.5, scale)
    fig, ax = plt.subplots(figsize=(fw, fh))
    idx = np.arange(len(names))
    ax.bar(idx, effects)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Standardized effect / elasticity")
    ax.grid(True, axis="y", alpha=0.3)
    return fig

def elasticity(coeff: float, y: pd.Series, x: pd.Series) -> float:
    try:
        return coeff * (x.mean() / y.mean())
    except Exception:
        return np.nan

def tvp_var_fit(endog: pd.DataFrame, q_scale: float = 1e-4, r_scale: float = 1e-2, p0_scale: float = 1e4):
    Y = endog.values
    T, n = Y.shape
    Z_list, y_list, idx = [], [], []
    for t in range(1, T):
        Z_t = np.concatenate(([1.0], Y[t-1, :]))
        Z_list.append(Z_t)
        y_list.append(Y[t, :])
        idx.append(endog.index[t])
    Z = np.asarray(Z_list)
    Yt = np.asarray(y_list)
    k = Z.shape[1]

    betas = np.zeros((n, T-1, k))
    fitted = np.zeros((T-1, n))
    resid = np.zeros((T-1, n))

    Q = q_scale * np.eye(k)
    R = r_scale
    P0 = p0_scale * np.eye(k)

    for i in range(n):
        beta = np.zeros(k)
        P = P0.copy()
        for t in range(T-1):
            z = Z[t, :].reshape(-1, 1)
            beta_pred = beta
            P_pred = P + Q
            y_obs = Yt[t, i]
            S = float(z.T @ P_pred @ z + R)
            K = (P_pred @ z) / S
            innov = y_obs - float(beta_pred @ z.flatten())
            beta = beta_pred + (K.flatten() * innov)
            P = (np.eye(k) - K @ z.T) @ P_pred
            betas[i, t, :] = beta
            fitted[t, i] = float(beta @ z.flatten())
            resid[t, i] = y_obs - fitted[t, i]

    betas_dict = {}
    names = ["const"] + [f"L1.{c}" for c in endog.columns]
    for i, col in enumerate(endog.columns):
        dfb = pd.DataFrame(betas[i, :, :], index=idx, columns=names)
        betas_dict[col] = dfb

    fitted_df = pd.DataFrame(fitted, index=idx, columns=endog.columns)
    resid_df = pd.DataFrame(resid, index=idx, columns=endog.columns)

    return {"betas": betas_dict, "fitted": fitted_df, "resid": resid_df, "names": names}

def fit_sarimax(endog: pd.Series, exog: Optional[pd.DataFrame], order=(1,0,0), seasonal_order=(0,0,0,0)):
    model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def forecast_sarimax(res, exog_test: Optional[pd.DataFrame], n_test: int):
    fitted = res.fittedvalues
    forecast = res.get_forecast(steps=n_test, exog=exog_test).predicted_mean
    return fitted, forecast

def fit_elasticnet(X_train, y_train, thinking_mode: bool = True) -> Pipeline:
    l1_ratio = [0.1, 0.2, 0.5, 0.8] if thinking_mode else [0.2, 0.5, 0.8]
    n_alphas = 100 if thinking_mode else 50
    cv = 7 if thinking_mode else 5
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("enet", ElasticNetCV(l1_ratio=l1_ratio, n_alphas=n_alphas, cv=cv, random_state=42, max_iter=10000))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def elasticnet_importance(pipe: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    coefs = pipe.named_steps["enet"].coef_
    return pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)}).sort_values("abs_coef", ascending=False)

def fit_var(df_endog: pd.DataFrame, lags:int=1):
    model = VAR(df_endog.dropna())
    res = model.fit(lags)
    return {"model": model, "result": res}

def fit_vecm(df_endog: pd.DataFrame, det_terms="co", k_ar_diff: int = 1, coint_rank: int = 1):
    model = VECM(df_endog.dropna(), k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=det_terms)
    res = model.fit()
    return {"model": model, "result": res}

_lag_pat = re.compile(r'^(?P<base>.+?)_lag\d+$')
def to_base_predictors(pred_list: List[str], target: str, available_cols: List[str]) -> List[str]:
    bases = []
    for p in pred_list or []:
        m = _lag_pat.match(p)
        name = m.group('base') if m else p
        if name == target or name.startswith(target):
            continue
        if name in available_cols and name not in bases:
            bases.append(name)
    return bases

def build_selection_rationale(results: List[dict], best: dict, thinking_mode: bool) -> str:
    lines = []
    ranked = sorted(results, key=lambda r: r["rank_score"])
    best_rmse = best["metrics"]["RMSE"]
    lines.append(f"Chosen: **{best['family']}** with RMSE={best_rmse:.4f}.")
    others = [r for r in ranked if r is not best and not np.isinf(r["rank_score"])]
    if len(others) > 0:
        rmse2 = others[0]["metrics"]["RMSE"]
        if not np.isnan(best_rmse) and not np.isnan(rmse2):
            delta = rmse2 - best_rmse
            lines.append(f"It outperformed the next model (**{others[0]['family']}**) by ΔRMSE={delta:.4f}.")
    lines.append("Diagnostics on training residuals:")
    lines.append(f"- Ljung–Box p-value: {best['lb_p']:.3f} (≥0.05 suggests no autocorrelation).")
    lines.append(f"- ARCH p-value: {best['arch_p']:.3f} (≥0.05 suggests no conditional heteroskedasticity).")
    lines.append(f"- Jarque–Bera p-value: {best['jb_p']:.3f} (≥0.05 suggests residual normality).")
    if thinking_mode:
        penalties = []
        if (not np.isnan(best['lb_p']) and best['lb_p'] < 0.05) or (not np.isnan(best['arch_p']) and best['arch_p'] < 0.05):
            penalties.append("serial correlation / ARCH effects")
        if best.get("vif") is not None and hasattr(best["vif"], "empty") and not best["vif"].empty:
            try:
                if np.nanmax(best["vif"]["VIF"].values) > 5:
                    penalties.append("multicollinearity (VIF>5)")
            except Exception:
                pass
        if penalties:
            lines.append("Note: penalties applied for " + ", ".join(penalties) + ".")
    n_pred = len(best.get("selected_features", []) or [])
    lines.append(f"Parsimony: {n_pred} predictors retained (≤ configured cap).")
    return "\n".join(lines)

def _render_cached_results(cached: dict, plot_scale: float):
    target = cached["target"]
    st.markdown(f"### Results for target (cached): **{target}**")
    st.write(f"**Selected model**: {cached['family']} (cached)")
    st.json(cached["metrics"])
    st.markdown("#### Why this model? (Selection rationale & diagnostics)")
    st.markdown(cached["rationale"])
    preds = cached.get("predictors", [])
    if preds:
        st.write("Included predictors:", preds)
    if "params_df" in cached and cached["params_df"] is not None:
        try:
            dfp = pd.DataFrame(cached["params_df"])
            st.write("Coefficients with 95% CI (if applicable):"); st.dataframe(dfp)
        except Exception:
            st.write(cached["params_df"])
    if "drivers" in cached and cached["drivers"]:
        names, effects = zip(*cached["drivers"])
        st.pyplot(top_drivers_bar(list(names), list(effects), title=f"Top drivers — {target}", scale=plot_scale))
    if "tvp_betas" in cached and cached["tvp_betas"]:
        for col, dfb in cached["tvp_betas"].items():
            fw, fh = _scale_figsize(9, 3.5, plot_scale)
            fig, ax = plt.subplots(figsize=(fw, fh))
            _apply_font_scale(plot_scale)
            ax.plot(pd.to_datetime(dfb["index"]), dfb["value"])
            ax.set_title(f"{target}: time-varying coefficient for {col}")
            ax.set_xlabel("Date"); ax.set_ylabel("Coefficient"); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    y_index = pd.to_datetime(cached["y_index"])
    y_values = np.array(cached["y_values"], dtype=float)
    yhat_values = np.array(cached["yhat_values"], dtype=float)
    st.pyplot(plot_actual_vs_pred(y_index, y_values, yhat_values, f"{target}: Actual vs Fitted/Forecast", scale=plot_scale))
    resid = pd.Series(cached["resid_values"], index=pd.to_datetime(cached["resid_index"]))
    for fig in plot_residual_diagnostics(resid, f"{target} — {cached['family']}", scale=plot_scale):
        st.pyplot(fig)
    st.write(cached["diag_table"])

def main():
    st.set_page_config(page_title="Deposit Dynamics Modeling — Thinking Mode", layout="wide")
    st.title("Deposit Dynamics Modeling Framework")
    st.caption("GPT-5 Thinking Mode: slower & more thorough validation (diagnostics + theory).")
    set_seed(42)

    if "best_predictors" not in st.session_state:
        st.session_state["best_predictors"] = {}
    if "best_specs" not in st.session_state:
        st.session_state["best_specs"] = {}
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = {}

    st.sidebar.header("Thinking Mode")
    thinking_mode = st.sidebar.checkbox("Enable Thinking Mode (rigorous & slower)", value=True)
    st.sidebar.markdown("- Stricter diagnostics (LB/ARCH/JB, VIF) and deeper hyperparam search.")
    max_predictors_final = st.sidebar.number_input("Max predictors in final model", 1, 12, 5, 1)

    st.sidebar.header("Plot size")
    plot_scale = st.sidebar.slider("Scale (0.6 small … 1.5 large)", 0.6, 1.5, 0.7, 0.1)

    st.sidebar.header("TVP-VAR Hyperparameters")
    q_scale = st.sidebar.number_input("State noise scale (Q)", 1e-8, 1e-1, 1e-4, format="%.8f")
    r_scale = st.sidebar.number_input("Obs. noise scale (R)", 1e-8, 1e-1, 1e-2, format="%.8f")
    p0_scale = st.sidebar.number_input("Initial state var scale (P0)", 1.0, 1e6, 1e4, format="%.0f")

    with st.expander("1) Upload data and (optionally) contextual PDFs", expanded=True):
        excel_file = st.file_uploader("Upload Excel dataset", type=["xlsx", "xlsm"])
        sheet_name = st.text_input("Sheet name", value="Input_Dataset")
        pdf_files = st.file_uploader("(Optional) Upload contextual PDFs", type=["pdf"], accept_multiple_files=True)
        if pdf_files:
            st.write("Uploaded PDF sources:")
            for f in pdf_files:
                st.write("• " + f.name)

    if excel_file is None:
        st.info("Please upload the Excel dataset to proceed.")
        if st.session_state["last_results"]:
            st.markdown("---"); st.subheader("Cached results (adjusted to new plot size)")
            for tgt, cached in st.session_state["last_results"].items():
                _render_cached_results(cached, plot_scale)
        return

    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        return

    if df.columns.duplicated().any():
        old_cols = list(df.columns)
        df.columns = uniquify_columns([str(c) for c in df.columns])
        st.warning("Detected duplicated column names. They were auto-renamed with '__dup#'.")
        mapping_preview = pd.DataFrame({"original": old_cols, "renamed": df.columns})
        st.dataframe(mapping_preview.head(20))

    st.subheader("2) Identify date/time column")
    date_guess = None
    for c in df.columns[:5]:
        cl = str(c).lower()
        if "date" in cl or "month" in cl or "time" in cl or "period" in cl:
            date_guess = c; break
    date_col = st.selectbox("Select date column", options=list(df.columns), index=list(df.columns).index(date_guess) if date_guess in df.columns else 0)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False, infer_datetime_format=True)
    df = df.sort_values(by=date_col).reset_index(drop=True).set_index(date_col)
    st.write("Data preview:", df.head())

    st.subheader("3) Select time period (default: 2015-01 onward)")
    min_date = pd.to_datetime("2015-01-01")
    start_date = st.date_input("Start date", value=min_date.date())
    end_date = st.date_input("End date", value=df.index.max().date() if len(df)>0 else datetime.today().date())
    dff = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))].copy()
    if dff.empty:
        st.warning("Filtered dataset is empty.")
        return

    st.subheader("4) Select targets and predictors")
    col_names = list(dff.columns)
    default_targets = col_names[1:5] if len(col_names) >= 5 else col_names[:1]
    target_cols = st.multiselect("Select target variable(s)", options=col_names, default=default_targets, key="target_select")
    if not target_cols:
        st.info("Select at least one target.")
        if st.session_state["last_results"]:
            st.markdown("---"); st.subheader("Cached results (adjusted to new plot size)")
            for tgt, cached in st.session_state["last_results"].items():
                _render_cached_results(cached, plot_scale)
        return
    active_target = st.selectbox("Active target for configuration", options=target_cols, index=0, key="active_target")

    pred_default_start = 14
    default_pred_list = col_names[pred_default_start:] if len(col_names) > pred_default_start else col_names
    suggest_best = st.checkbox("Use best predictors from previous run (for the active target)", value=False, key="use_best_toggle")
    if suggest_best and active_target in st.session_state["best_specs"]:
        stored_feats = st.session_state["best_specs"][active_target].get("selected_features", []) or []
        base_from_exact = to_base_predictors(stored_feats, target=active_target, available_cols=col_names)
        if not base_from_exact and active_target in st.session_state["best_predictors"]:
            base_from_exact = st.session_state["best_predictors"][active_target]
        initial_pred_selection = base_from_exact if base_from_exact else default_pred_list
        if base_from_exact:
            st.caption(f"Loaded best (base) predictors for '{active_target}': {initial_pred_selection}")
        else:
            st.warning("No stored best predictors match current columns; using default list.")
    elif suggest_best and active_target in st.session_state["best_predictors"]:
        initial_pred_selection = st.session_state["best_predictors"][active_target]
        st.caption(f"Loaded best (base) predictors: {initial_pred_selection}")
    else:
        initial_pred_selection = default_pred_list

    predictors_all = st.multiselect("Select candidate predictors", options=col_names, default=initial_pred_selection, key="predictors_select")
    st.caption("Tip: Keep a manageable set; the app will later constrain the final model to ≤ the selected cap.")

    st.subheader("5) Preprocessing & feature engineering")
    c1, c2, c3 = st.columns(3)
    with c1:
        do_log_targets = st.checkbox("Log-transform targets (if positive)", value=False)
        do_log_predictors = st.checkbox("Log-transform predictors (if positive)", value=False)
        season_adj = st.checkbox("Seasonal differencing (lag 12)", value=False)
    with c2:
        diff_targets = st.selectbox("Difference targets", options=["None", "First diff (Δ)", "YoY diff (Δ12)"], index=0)
        diff_predictors = st.selectbox("Difference predictors", options=["None", "First diff (Δ)", "YoY diff (Δ12)"], index=0)
    with c3:
        max_lag = st.number_input("Max lag to consider for predictors/targets", min_value=0, max_value=24, value=12)
        train_frac = st.slider("Train fraction (time-ordered split)", 0.5, 0.95, 0.8, 0.05)

    preserve_exact_lags = st.checkbox("Respect EXACT lagged features from previous run (advanced)", value=True)

    def transform_series(s: pd.Series, name: str, is_target=False):
        s_trans = s.copy()
        notes = []
        if (is_target and do_log_targets) or ((not is_target) and do_log_predictors):
            if (s_trans <= 0).any():
                notes.append(f"{name}: not log-transformed (non-positive).")
            else:
                s_trans = np.log(s_trans); notes.append(f"{name}: log-transformed.")
        if season_adj:
            s_trans = s_trans - s_trans.shift(12); notes.append(f"{name}: seasonal difference (lag 12).")
        if is_target:
            if diff_targets == "First diff (Δ)":
                s_trans = s_trans.diff(); notes.append(f"{name}: first differenced.")
            elif diff_targets == "YoY diff (Δ12)":
                s_trans = s_trans.diff(12); notes.append(f"{name}: YoY differenced (lag 12).")
        else:
            if diff_predictors == "First diff (Δ)":
                s_trans = s_trans.diff(); notes.append(f"{name}: first differenced.")
            elif diff_predictors == "YoY diff (Δ12)":
                s_trans = s_trans.diff(12); notes.append(f"{name}: YoY differenced (lag 12).")
        return s_trans, notes

    st.subheader("6) Stationarity checks (ADF/KPSS)")
    target_preview = active_target
    y0_raw = dff[target_preview]
    y0, y0_notes = transform_series(y0_raw, target_preview, is_target=True)
    adf_res = adf_test(y0.dropna()); kpss_res = kpss_test(y0.dropna())
    st.write(f"Target preview: **{target_preview}**")
    st.write("Transformations:", "; ".join(y0_notes) if y0_notes else "None")
    st.write({"ADF p-value": adf_res["pval"], "KPSS p-value": kpss_res["pval"]})

    st.subheader("7) Modeling options")
    model_families = st.multiselect(
        "Candidate model families (the app will pick the best per target)",
        options=["SARIMAX (ARIMAX)", "Elastic Net (lagged regression)", "VAR/VECM (system)", "TVP-VAR (time-varying VAR(1))"],
        default=["SARIMAX (ARIMAX)", "Elastic Net (lagged regression)"]
    )
    arima_order_p = st.number_input("ARIMAX p", 0, 5, 1)
    arima_order_d = st.number_input("ARIMAX d", 0, 2, 0)
    arima_order_q = st.number_input("ARIMAX q", 0, 5, 0)

    keep_best_after_run = st.checkbox("After running, replace candidate predictors with the model's best (active target)", value=False, key="keep_best_chk")

    st.subheader("8) Run analysis per target")
    run_btn = st.button("Run selected models")

    if not run_btn and st.session_state["last_results"]:
        st.markdown("---"); st.subheader("Cached results (adjusted to new plot size)")
        for tgt, cached in st.session_state["last_results"].items():
            _render_cached_results(cached, plot_scale)
        return
    if not run_btn:
        return

    outdir = timestamped_outdir()
    st.write(f"Outputs will be saved to: `{outdir}`")

    for target in target_cols:
        st.markdown(f"### Results for target: **{target}**")
        y_raw = dff[target].copy()
        y, y_notes = transform_series(y_raw, target, is_target=True)

        X_raw = dff[predictors_all].copy() if predictors_all else pd.DataFrame(index=dff.index)
        X_trans = pd.DataFrame(index=dff.index)
        x_notes_all = []
        for c in X_raw.columns:
            xt, notes = transform_series(X_raw[c], c, is_target=False)
            X_trans[c] = xt
            x_notes_all.extend(notes)

        all_cols_to_lag = [target] + list(X_trans.columns)
        base_df = pd.concat([y.rename(target), X_trans], axis=1)
        lagged = make_lags(base_df, cols=all_cols_to_lag, max_lag=max_lag) if max_lag > 0 else base_df.copy()
        lagged = lagged.dropna().copy()

        train_df, test_df = train_test_time_split(lagged, train_frac=train_frac)
        y_train = train_df[target]; y_test = test_df[target]

        results = []

        # Elastic Net + OLS
        if "Elastic Net (lagged regression)" in model_families and len(X_trans.columns) > 0:
            feature_cols = [c for c in train_df.columns if c != target and not c.endswith("_lag0")]
            X_train = train_df[feature_cols]; X_test = test_df[feature_cols]
            with np.errstate(invalid="ignore"):
                vif_df = vif_dataframe(X_train.replace([np.inf, -np.inf], np.nan).dropna())

            stored_feats = st.session_state["best_specs"].get(active_target, {}).get("selected_features", []) if preserve_exact_lags else []
            stored_feats = [f for f in stored_feats if f in feature_cols]

            if stored_feats and target == active_target:
                selected_features = stored_feats[:int(max_predictors_final)]
            else:
                pipe = fit_elasticnet(X_train, y_train, thinking_mode=thinking_mode)
                imp = elasticnet_importance(pipe, feature_cols)
                k = int(max_predictors_final)
                imp_nonzero = imp[imp["abs_coef"] > 0].head(k)
                selected_features = imp_nonzero["feature"].tolist()
                if len(selected_features) == 0:
                    corr = X_train.apply(lambda s: s.corr(y_train)).abs().sort_values(ascending=False)
                    selected_features = list(corr.index[:k])

            X_train_sel = sm.add_constant(X_train[selected_features])
            X_test_sel = sm.add_constant(X_test[selected_features], has_constant="add")
            ols = sm.OLS(y_train, X_train_sel).fit()
            y_fitted = ols.fittedvalues
            y_forecast = ols.predict(X_test_sel)

            resid_train = y_train - y_fitted
            metrics = compute_metrics(y_test, y_forecast)
            lb_p = ljung_box_test(resid_train); arch_p = arch_test(resid_train); jb_p = jb_test(resid_train)
            aic = float(ols.aic) if hasattr(ols, "aic") else np.nan
            bic = float(ols.bic) if hasattr(ols, "bic") else np.nan

            results.append({
                "family": "ElasticNet+OLS",
                "selected_features": selected_features,
                "params": ols.params, "conf_int": ols.conf_int(),
                "train_fitted": y_fitted, "test_forecast": y_forecast,
                "resid_train": resid_train, "metrics": metrics,
                "lb_p": lb_p, "arch_p": arch_p, "jb_p": jb_p,
                "aic": aic, "bic": bic, "vif": vif_df
            })

        # SARIMAX (ARIMAX)
        if "SARIMAX (ARIMAX)" in model_families:
            exog_cols = [c for c in train_df.columns if c != target and not c.startswith(target + "_lag0")]
            exog_selected = []
            if preserve_exact_lags and target == active_target:
                stored_feats = st.session_state["best_specs"].get(active_target, {}).get("selected_features", [])
                exog_selected = [f for f in stored_feats if f in exog_cols][:int(max_predictors_final)]
            if not exog_selected and len(exog_cols) > 0:
                exog_corr = train_df[exog_cols].apply(lambda s: s.corr(y_train)).abs().sort_values(ascending=False)
                exog_selected = list(exog_corr.index[:int(max_predictors_final)])

            exog_train = train_df[exog_selected] if exog_selected else None
            exog_test = test_df[exog_selected] if exog_selected else None

            try:
                sarimax_res = fit_sarimax(y_train, exog_train, order=(arima_order_p, arima_order_d, arima_order_q))
                fitted, forecast = forecast_sarimax(sarimax_res, exog_test, n_test=len(test_df))
                resid_train = y_train - fitted.reindex(y_train.index)
                metrics = compute_metrics(y_test, forecast)
                lb_p = ljung_box_test(pd.Series(resid_train, index=y_train.index))
                arch_p = arch_test(pd.Series(resid_train, index=y_train.index))
                jb_p = jb_test(pd.Series(resid_train, index=y_train.index))
                aic = float(sarimax_res.aic); bic = float(sarimax_res.bic)

                results.append({
                    "family": "SARIMAX",
                    "selected_features": exog_selected,
                    "params": sarimax_res.params, "conf_int": sarimax_res.conf_int(),
                    "train_fitted": pd.Series(fitted, index=y_train.index),
                    "test_forecast": pd.Series(forecast, index=y_test.index),
                    "resid_train": pd.Series(resid_train, index=y_train.index),
                    "metrics": metrics, "lb_p": lb_p, "arch_p": arch_p, "jb_p": jb_p,
                    "aic": aic, "bic": bic, "vif": None
                })
            except Exception as e:
                st.warning(f"SARIMAX failed for {target}: {e}")

        # VAR/VECM (system)
        if "VAR/VECM (system)" in model_families and len(X_trans.columns) > 0:
            X_base_train = X_trans.loc[y_train.index].dropna(axis=1, how="all")
            top_names = []
            if preserve_exact_lags and active_target in st.session_state["best_specs"]:
                stored_feats = st.session_state["best_specs"][active_target].get("selected_features", [])
                base_from_stored = to_base_predictors(stored_feats, target=active_target, available_cols=list(X_base_train.columns))
                if base_from_stored:
                    top_names = list(dict.fromkeys(base_from_stored))[:2]
            if not top_names and not X_base_train.empty:
                corr_series = X_base_train.apply(lambda s: s.corr(y_train)).abs().sort_values(ascending=False)
                top_names = list(corr_series.index[:min(2, len(corr_series))])

            if len(top_names) > 0:
                endog = pd.concat([y.loc[y_train.index].rename(target), X_base_train[top_names]], axis=1).dropna()
                try:
                    cj = coint_johansen(endog.dropna(), det_order=0, k_ar_diff=1)
                    rank = int((cj.lr1 > cj.cvt[:, 1]).sum())
                except Exception:
                    rank = 0
                try:
                    if rank >= 1:
                        vecm = fit_vecm(endog, det_terms="co", k_ar_diff=1, coint_rank=1)
                        fitted = vecm["result"].fittedvalues[target]
                        forecast = pd.Series([np.nan]*len(y_test), index=y_test.index)
                        resid_train = endog[target].iloc[1:] - fitted
                        metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE(%)": np.nan}
                        lb_p = ljung_box_test(resid_train); arch_p = arch_test(resid_train); jb_p = jb_test(resid_train)
                        results.append({
                            "family": "VECM", "selected_features": top_names,
                            "params": None, "conf_int": None,
                            "train_fitted": fitted, "test_forecast": forecast,
                            "resid_train": resid_train, "metrics": metrics,
                            "lb_p": lb_p, "arch_p": arch_p, "jb_p": jb_p,
                            "aic": np.nan, "bic": np.nan, "vif": None
                        })
                    else:
                        var = fit_var(endog[[target] + top_names], lags=1)
                        fitted = var["result"].fittedvalues[target]
                        forecast = pd.Series([np.nan]*len(y_test), index=y_test.index)
                        resid_train = endog[target].iloc[1:] - fitted
                        metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE(%)": np.nan}
                        lb_p = ljung_box_test(resid_train); arch_p = arch_test(resid_train); jb_p = jb_test(resid_train)
                        results.append({
                            "family": "VAR(1)", "selected_features": top_names,
                            "params": None, "conf_int": None,
                            "train_fitted": fitted, "test_forecast": forecast,
                            "resid_train": resid_train, "metrics": metrics,
                            "lb_p": lb_p, "arch_p": arch_p, "jb_p": jb_p,
                            "aic": float(var["result"].aic), "bic": float(var["result"].bic), "vif": None
                        })
                except Exception as e:
                    st.warning(f"VAR/VECM failed for {target}: {e}")

        # TVP-VAR
        if "TVP-VAR (time-varying VAR(1))" in model_families and len(X_trans.columns) > 0:
            X_base_all = X_trans.dropna(axis=1, how="all")
            guided = []
            if preserve_exact_lags and active_target in st.session_state["best_specs"]:
                stored_feats = st.session_state["best_specs"][active_target].get("selected_features", [])
                guided = to_base_predictors(stored_feats, target=active_target, available_cols=list(X_base_all.columns))[:2]
            if guided:
                top2 = guided
            else:
                corr_all = X_base_all.apply(lambda s: s.corr(y), axis=0).abs().sort_values(ascending=False)
                top2 = list(corr_all.index[:2])
            endog_all = pd.concat([y.rename(target), X_base_all[top2]], axis=1).dropna()
            try:
                tvp = tvp_var_fit(endog_all, q_scale=float(q_scale), r_scale=float(r_scale), p0_scale=float(p0_scale))
                fit_full = tvp["fitted"][target]
                fit_aligned = fit_full.loc[fit_full.index.intersection(y_train.index)]
                resid_train = (y_train.loc[fit_aligned.index] - fit_aligned)
                forecast = pd.Series([np.nan]*len(y_test), index=y_test.index)
                metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE(%)": np.nan}
                lb_p = ljung_box_test(resid_train); arch_p = arch_test(resid_train); jb_p = jb_test(resid_train)

                beta_df = tvp["betas"][target].copy()
                driver_cols = [c for c in beta_df.columns if c.startswith("L1.")]
                imp_vals = beta_df[driver_cols].abs().mean(axis=0)
                imp_names = [c.replace("L1.","") for c in driver_cols]
                importance = pd.Series(imp_vals.values, index=imp_names)

                results.append({
                    "family": "TVP-VAR(1)", "selected_features": imp_names,
                    "params": None, "conf_int": None,
                    "train_fitted": fit_aligned, "test_forecast": forecast,
                    "resid_train": resid_train, "metrics": metrics,
                    "lb_p": lb_p, "arch_p": arch_p, "jb_p": jb_p,
                    "aic": np.nan, "bic": np.nan, "vif": None,
                    "tvp": tvp, "importance": importance
                })
            except Exception as e:
                st.warning(f"TVP-VAR failed for {target}: {e}")

        if len(results) == 0:
            st.error("No model succeeded. Try adjusting options or predictors.")
            continue

        for r in results:
            score = r["metrics"]["RMSE"] if not np.isnan(r["metrics"]["RMSE"]) else np.inf
            if thinking_mode:
                if (not np.isnan(r["lb_p"]) and r["lb_p"] < 0.05) or (not np.isnan(r["arch_p"]) and r["arch_p"] < 0.05):
                    score += 1e6
                if r.get("vif") is not None and hasattr(r["vif"], "empty") and not r["vif"].empty:
                    try:
                        if np.nanmax(r["vif"]["VIF"].values) > 5:
                            score += 5e5
                    except Exception:
                        pass
                score += len(r.get("selected_features", [])) * 1e-6
            r["rank_score"] = score

        best = sorted(results, key=lambda r: r["rank_score"])[0]

        st.write(f"**Selected model**: {best['family']} (≤ {int(max_predictors_final)} predictors cap applied where relevant)")
        st.json(best["metrics"])

        preds = best.get("selected_features", []) or []
        if preds:
            st.write("Included predictors:", preds)

        st.markdown("#### Why this model? (Selection rationale & diagnostics)")
        rationale = build_selection_rationale(results, best, thinking_mode)
        st.markdown(rationale)

        params_df_display = None
        drivers_pairs = []
        if best.get("params") is not None and best.get("conf_int") is not None:
            params_df = best["conf_int"]; params_df.columns = ["ci_low", "ci_high"]; params_df["coef"] = best["params"]
            params_df_display = params_df
            st.write("Coefficients with 95% CI (if applicable):"); st.dataframe(params_df)
            names, effects = [], []
            for n in preds:
                if n in params_df.index:
                    try:
                        coef = params_df.loc[n, "coef"]; eff = elasticity(coef, y_train, train_df[n])
                        names.append(n); effects.append(eff); drivers_pairs.append((n, float(eff)))
                    except Exception:
                        continue
            if names:
                st.pyplot(top_drivers_bar(names, effects, title=f"Top drivers (elasticity approx) — {target}", scale=plot_scale))

        tvp_betas_cache = {}
        if best["family"].startswith("TVP-VAR"):
            importance = best.get("importance")
            if importance is not None and not importance.empty:
                st.pyplot(top_drivers_bar(list(importance.index), list(importance.values), title=f"Top drivers (mean |beta_t|) — {target} (TVP)", scale=plot_scale))
            tvp = best.get("tvp")
            if tvp is not None:
                beta_df = tvp["betas"][target]
                for col in [c for c in beta_df.columns if c != "const"]:
                    fw, fh = _scale_figsize(9, 3.5, plot_scale)
                    fig, ax = plt.subplots(figsize=(fw, fh))
                    _apply_font_scale(plot_scale)
                    ax.plot(beta_df.index, beta_df[col])
                    ax.set_title(f"{target}: time-varying coefficient for {col}")
                    ax.set_xlabel("Date"); ax.set_ylabel("Coefficient"); ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    tvp_betas_cache[col] = {"index": [str(ix) for ix in beta_df.index], "value": beta_df[col].tolist()}

        fitted = pd.Series(best["train_fitted"], index=y_train.index)
        forecast = pd.Series(best["test_forecast"], index=y_test.index)
        y_all = pd.concat([y_train, y_test], axis=0)
        y_hat_all = fitted.reindex(y_all.index).combine_first(forecast.reindex(y_all.index))
        st.pyplot(plot_actual_vs_pred(y_all.index, y_all.values, y_hat_all.values, f"{target}: Actual vs Fitted/Forecast", scale=plot_scale))

        resid_train = pd.Series(best["resid_train"]).dropna()
        for fig in plot_residual_diagnostics(resid_train, f"{target} — {best['family']}", scale=plot_scale):
            st.pyplot(fig)
        diag_table = {"Ljung-Box p": best["lb_p"], "ARCH p": best["arch_p"], "Jarque-Bera p": best["jb_p"]}
        st.write(diag_table)

        if best.get("vif") is not None and hasattr(best["vif"], "empty") and not best["vif"].empty:
            st.write("VIF (train features):"); st.dataframe(best["vif"])

        if keep_best_after_run and target == active_target and preds:
            base_preds = to_base_predictors(preds, target=active_target, available_cols=col_names)
            if not base_preds:
                base_preds = [p for p in preds if p in col_names]
            st.session_state["best_predictors"][active_target] = base_preds
            st.session_state["best_specs"][active_target] = {"family": best["family"], "selected_features": preds}
            st.success(f"Stored best predictors for '{active_target}'. Base={base_preds}, exact={preds}")

        run_dir = timestamped_outdir()
        summary = {
            "target": target, "model_family": best["family"], "predictors": preds,
            "metrics": best["metrics"],
            "diagnostics": diag_table,
            "aic": best.get("aic"), "bic": best.get("bic"),
            "transform_notes": {"target": y_notes, "predictors": list(set(x_notes_all))},
            "train_period": [str(y_train.index.min().date()), str(y_train.index.max().date())],
            "test_period": [str(y_test.index.min().date()), str(y_test.index.max().date())],
            "thinking_mode": thinking_mode, "max_predictors_final": int(max_predictors_final),
            "plot_scale": float(plot_scale),
            "selection_rationale": rationale
        }
        with open(os.path.join(run_dir, f"{target}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        st.session_state["last_results"][target] = {
            "target": target,
            "family": best["family"],
            "metrics": best["metrics"],
            "rationale": rationale,
            "predictors": preds,
            "params_df": params_df_display.to_dict() if params_df_display is not None else None,
            "drivers": drivers_pairs,
            "tvp_betas": tvp_betas_cache,
            "y_index": [str(ix) for ix in y_all.index],
            "y_values": y_all.values.tolist(),
            "yhat_index": [str(ix) for ix in y_all.index],
            "yhat_values": y_hat_all.values.tolist(),
            "resid_index": [str(ix) for ix in resid_train.index],
            "resid_values": resid_train.values.tolist(),
            "diag_table": diag_table
        }

    st.success("Analysis completed. Summaries saved in timestamped folders within ./outputs/.")

if __name__ == "__main__":
    main()
