# app.py —— Japan S&D: OLS Regression with an Insights tab & cached backtests (English UI)
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Japan S&D Regression (OLS + Insights)", layout="wide")
st.title("Japan S&D: Multivariate Regression (OLS)")

# -------------------- Utilities --------------------
def load_and_prepare(df: pd.DataFrame,
                     date_col: str,
                     target_col: str,
                     predictor_cols=None):
    """Clean columns, set datetime index, coerce numerics, and add const if needed."""
    orig_cols = df.columns.tolist()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = dict(zip(df.columns, orig_cols))  # cleaned -> original

    # Parse datetime to index
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).set_index(date_col)
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()

    # Select predictors
    if predictor_cols is None:
        predictor_cols = [c for c in df.columns if c not in [target_col]]

    # Coerce numeric & drop NA rows
    for c in predictor_cols + [target_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=predictor_cols + [target_col]).copy()

    X = df[predictor_cols].copy()
    y = df[target_col].copy()

    # Respect user-provided Constant column if present
    has_user_constant = any(c.lower() == "constant" for c in X.columns)
    if not has_user_constant:
        X = sm.add_constant(X)
        intercept_name = "const"
    else:
        intercept_name = [c for c in X.columns if c.lower() == "constant"][0]

    return df, X, y, intercept_name, colmap

def pretty_name(v, colmap):
    if v == "const":
        return "Intercept (const)"
    return colmap.get(v, v)

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    denom = np.where(y_true != 0, np.abs(y_true), np.nan)
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    return float(mape)

def metrics_extended(y_true, y_pred):
    """Return RMSE/MAE/MAPE and Directional Hit Ratio (based on period-over-period sign)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = safe_mape(y_true, y_pred)

    dy_true = pd.Series(y_true).diff().to_numpy()
    dy_pred = pd.Series(y_pred).diff().to_numpy()
    mask = ~np.isnan(dy_true) & ~np.isnan(dy_pred)
    hit = (np.sign(dy_true[mask]) == np.sign(dy_pred[mask])).mean() * 100.0 if mask.sum() > 0 else np.nan

    return rmse, mae, mape, float(hit)

def get_var_names_from_model(model, X):
    if hasattr(model.model, "exog_names"):
        return list(model.model.exog_names)
    return list(X.columns)

def standardize_coefficients(params, X, y):
    """beta_std = beta * std(X) / std(y), excluding intercept."""
    series_params = pd.Series(np.asarray(params), index=getattr(params, "index", None))
    if series_params.index is None:
        series_params = pd.Series(np.asarray(params), index=X.columns)
    cols = [c for c in series_params.index if c.lower() not in ("const", "constant")]
    if len(cols) == 0:
        return pd.DataFrame(columns=["Variable", "Std_Coefficient"])
    std_x = X[cols].std()
    std_y = y.std()
    denom = (std_y if std_y != 0 else 1.0)
    std_coef = series_params[cols] * (std_x / denom)
    out = std_coef.to_frame("Std_Coefficient").reset_index().rename(columns={"index": "Variable"})
    return out.sort_values("Std_Coefficient", key=lambda s: s.abs(), ascending=False)

def avg_abs_contribution(params, X):
    """Mean(|beta_i * x_i|) per variable over the sample, excluding intercept."""
    p = pd.Series(np.asarray(params), index=getattr(params, "index", None))
    if p.index is None:
        p = pd.Series(np.asarray(params), index=X.columns)
    cols = [c for c in X.columns if c.lower() not in ("const", "constant")]
    if len(cols) == 0:
        return pd.DataFrame(columns=["Variable", "AvgAbsContribution"])
    contrib = (X[cols] * p[cols]).abs().mean().sort_values(ascending=False)
    return contrib.rename("AvgAbsContribution").reset_index().rename(columns={"index": "Variable"})

def contribution_row(params, rowX):
    """Return single-period contribution vector including intercept (if any)."""
    p = pd.Series(np.asarray(params), index=getattr(params, "index", None))
    if p.index is None:
        p = pd.Series(np.asarray(params), index=rowX.index)
    contrib = rowX * p[rowX.index]
    if "const" in p.index:
        contrib.loc["const"] = p["const"]
    elif "Constant" in p.index:
        contrib.loc["Constant"] = p["Constant"]
    return contrib

def scenario_slider_for_var(st_key_prefix, series, label, default_value):
    """Slider range by 5–95th percentiles for robustness."""
    q05, q95 = float(series.quantile(0.05)), float(series.quantile(0.95))
    vmin = min(q05, default_value)
    vmax = max(q95, default_value)
    return st.slider(label, vmin, vmax, float(default_value), key=f"{st_key_prefix}_{label}")

def walk_forward_predict(X, y, min_train=36):
    """Expanding-window, 1-step-ahead backtest."""
    n = len(X)
    if n <= min_train:
        return pd.Series(dtype=float, index=X.index)
    preds = []
    idxs = []
    for t in range(min_train, n):
        X_tr = X.iloc[:t]
        y_tr = y.iloc[:t]
        try:
            fit = sm.OLS(y_tr, X_tr).fit()
            yhat_t = float(np.asarray(fit.predict(X.iloc[t:t+1]))[0])
        except Exception:
            yhat_t = np.nan
        preds.append(yhat_t)
        idxs.append(X.index[t])
    return pd.Series(preds, index=idxs)

def dataset_key(df, X, y, date_col, target_col):
    """A lightweight signature to invalidate cache when data/columns change."""
    return (
        len(df),
        str(df.index.min()) if len(df) else "NA",
        str(df.index.max()) if len(df) else "NA",
        tuple(X.columns.tolist()),
        date_col,
        target_col,
    )

# Initialize session cache for backtests
if "oos_cache" not in st.session_state:
    st.session_state.oos_cache = {}
if "last_ds_key" not in st.session_state:
    st.session_state.last_ds_key = None

# -------------------- Tabs --------------------
tabs = st.tabs(["Power", "Insights", "Non Power (coming soon)"])

# ============================ Power Tab ============================
with tabs[0]:
    st.subheader("Power")

    file = st.file_uploader("Upload CSV (must include Time Period, Actual, and feature columns)", type=["csv"])
    if file is None:
        st.info("Please upload the dataset (e.g., Japan S&D Data.csv).")
        st.stop()

    raw = pd.read_csv(file)
    raw.columns = [c.strip() for c in raw.columns]

    date_col = st.selectbox(
        "Time index column",
        options=raw.columns.tolist(),
        index=raw.columns.tolist().index("Time Period") if "Time Period" in raw.columns else 0
    )
    target_col = st.selectbox(
        "Target (y)",
        options=raw.columns.tolist(),
        index=raw.columns.tolist().index("Actual") if "Actual" in raw.columns else 0
    )
    default_X = [c for c in raw.columns if c not in [date_col, target_col]]
    predictor_cols = st.multiselect(
        "Predictors (X)",
        options=[c for c in raw.columns if c != target_col],
        default=default_X
    )
    if len(predictor_cols) == 0:
        st.warning("Please select at least one predictor.")
        st.stop()

    df, X, y, intercept_name, colmap = load_and_prepare(
        raw.copy(), date_col, target_col, predictor_cols=predictor_cols
    )
    if df.empty:
        st.error("No rows left after cleaning. Check missing values / column choices.")
        st.stop()

    # Invalidate cached backtests if dataset signature changed
    ds_key = dataset_key(df, X, y, date_col, target_col)
    if st.session_state.last_ds_key != ds_key:
        st.session_state.oos_cache = {}
        st.session_state.last_ds_key = ds_key

    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)
    r2 = float(model.rsquared)
    r2_adj = float(model.rsquared_adj)
    corr = np.corrcoef(y, y_pred)[0, 1] if (np.std(y) > 0 and np.std(y_pred) > 0) else 0.0
    R = float(np.sign(corr) * np.sqrt(max(r2, 0.0)))

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("R", f"{R:.6f}")
    k2.metric("R²", f"{r2:.6f}")
    k3.metric("Adj. R²", f"{r2_adj:.6f}")

    # Coeff table
    var_names = get_var_names_from_model(model, X)
    coef_df = pd.DataFrame({
        "Variable": var_names,
        "Parameter": np.asarray(model.params).ravel(),
        "Std Error": np.asarray(model.bse).ravel(),
        "t-Stat":   np.asarray(model.tvalues).ravel(),
        "P>|t|":    np.asarray(model.pvalues).ravel()
    })
    coef_df["Variable"] = coef_df["Variable"].apply(lambda v: pretty_name(v, colmap))

    st.markdown("### Coefficients")
    st.dataframe(coef_df, use_container_width=True)

    buf = StringIO()
    coef_df.to_csv(buf, index=False)
    st.download_button("Download coefficients (CSV)", buf.getvalue(), "coefficients.csv", "text/csv")

    # Figures
    # 1) Time series: Actual vs Predicted
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df.index, y=y, mode="lines", name="Actual"))
    fig_ts.add_trace(go.Scatter(x=df.index, y=y_pred, mode="lines", name="Predicted"))
    fig_ts.update_layout(title="Actual vs Predicted (Time Series)",
                         xaxis_title="Time", yaxis_title=target_col)
    st.plotly_chart(fig_ts, use_container_width=True)

    # 2) Scatter: Predicted vs Actual + 45°
    minv = float(min(y.min(), y_pred.min()))
    maxv = float(max(y.max(), y_pred.max()))
    fig_scatter = px.scatter(x=y_pred, y=y,
                             labels={"x": "Predicted", "y": "Actual"},
                             title=f"Predicted vs Actual  |  R²={r2:.4f}  Adj.R²={r2_adj:.4f}")
    fig_scatter.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv],
                                     mode="lines", name="45° Line"))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 3) Residuals vs Fitted
    residuals = y - y_pred
    fig_res = px.scatter(x=y_pred, y=residuals,
                         labels={"x": "Fitted (Predicted)", "y": "Residual"},
                         title="Residuals vs Fitted")
    fig_res.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_res, use_container_width=True)

    # 4) Coeff bar (exclude intercept)
    plot_coefs = coef_df[coef_df["Variable"] != "Intercept (const)"].copy()
    if not plot_coefs.empty:
        fig_coef = go.Figure()
        fig_coef.add_trace(go.Bar(
            x=plot_coefs["Variable"],
            y=plot_coefs["Parameter"],
            error_y=dict(type="data", array=plot_coefs["Std Error"]),
            name="Coefficient"
        ))
        fig_coef.update_layout(title="Coefficient Estimates (±1 Std Error)",
                               xaxis_title="Variable", yaxis_title="Estimate")
        st.plotly_chart(fig_coef, use_container_width=True)

    with st.expander("Statsmodels Summary"):
        st.text(model.summary())

# ============================ Insights Tab ============================
with tabs[1]:
    st.subheader("Insights (Drivers • Scenarios • Backtest)")

    if "df" not in locals():
        st.info("Please upload and configure data in the Power tab first.")
        st.stop()

    # Performance metrics
    rmse, mae, mape, hit = metrics_extended(y, y_pred)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE", f"{rmse:,.4f}")
    m2.metric("MAE", f"{mae:,.4f}")
    m3.metric("MAPE", f"{mape:.2f}%")
    m4.metric("Directional Hit Rate", f"{hit:.2f}%")

    st.divider()

    # Top Drivers
    ctda, ctdb = st.columns(2)
    with ctda:
        st.markdown("**Standardized Coefficients (comparable magnitudes)**")
        std_coef = standardize_coefficients(model.params, X, y)
        st.dataframe(std_coef, height=320, use_container_width=True)
    with ctdb:
        st.markdown("**Average Absolute Contribution (mean |β·x|)**")
        aac = avg_abs_contribution(model.params, X)
        st.dataframe(aac, height=320, use_container_width=True)

    st.divider()

    # Single-period contribution waterfall
    st.markdown("### Single-period Contribution (Waterfall)")
    pick_dt = st.selectbox("Pick a date", options=list(df.index.astype(str))[-50:], index=len(df.index[-50:])-1)
    dt = pd.to_datetime(pick_dt)
    rowX = X.loc[dt]
    contrib = contribution_row(model.params, rowX)
    contrib_disp = contrib.copy()
    contrib_disp.index = [pretty_name(v, colmap) for v in contrib.index]
    contrib_disp = contrib_disp.sort_values(ascending=False)

    measures = ["relative"] * len(contrib_disp) + ["total"]
    x_labels = list(contrib_disp.index) + ["Predicted"]
    y_vals = list(contrib_disp.values) + [0]
    wf = go.Figure(go.Waterfall(measure=measures, x=x_labels, y=y_vals,
                                connector={"line": {"dash": "dot"}}))
    wf.update_layout(title=f"Contribution breakdown: {pick_dt}", yaxis_title="Contribution")
    st.plotly_chart(wf, use_container_width=True)
    st.dataframe(contrib_disp.rename("Contribution").to_frame(), use_container_width=True)

    st.divider()

    # Scenario sandbox
    st.markdown("### Scenario Sandbox")
    vars_for_scenario = [c for c in X.columns if c.lower() not in ("const", "constant")]
    pick_vars = st.multiselect("Select variables to adjust (others stay at baseline)",
                               vars_for_scenario,
                               default=[v for v in vars_for_scenario[: min(5, len(vars_for_scenario))]])

    baseline_latest = st.checkbox("Use latest period as baseline", value=True)
    if baseline_latest:
        base_row = X.iloc[-1].copy()
        base_label = str(X.index[-1])
    else:
        base_label = st.selectbox("Baseline date", options=list(X.index.astype(str)))
        base_row = X.loc[pd.to_datetime(base_label)].copy()

    new_row = base_row.copy()
    for v in pick_vars:
        slider_val = scenario_slider_for_var("sc", X[v], pretty_name(v, colmap), base_row[v])
        new_row[v] = slider_val

    y_base = float(np.asarray(model.predict(base_row.to_frame().T))[0])
    y_new = float(np.asarray(model.predict(new_row.to_frame().T))[0])

    d1, d2, d3 = st.columns(3)
    d1.metric("Baseline y", f"{y_base:,.4f}", help=f"Baseline at: {base_label}")
    d2.metric("Scenario y", f"{y_new:,.4f}")
    d3.metric("Δy", f"{(y_new - y_base):,.4f}")

    contrib_base = contribution_row(model.params, base_row)
    contrib_new = contribution_row(model.params, new_row)
    contrib_delta = (contrib_new - contrib_base).dropna()
    contrib_delta_disp = contrib_delta.copy()
    contrib_delta_disp.index = [pretty_name(v, colmap) for v in contrib_delta.index]
    contrib_delta_disp = contrib_delta_disp.sort_values(key=lambda s: s.abs(), ascending=False)

    fig_delta = go.Figure(go.Bar(x=contrib_delta_disp.index, y=contrib_delta_disp.values))
    fig_delta.update_layout(title="ΔContribution under scenario", xaxis_title="Variable", yaxis_title="Δ")
    st.plotly_chart(fig_delta, use_container_width=True)
    st.dataframe(contrib_delta_disp.rename("ΔContribution").to_frame(), use_container_width=True)

    st.divider()

    # Seasonality by month
    st.markdown("### Seasonality by Month")
    month_idx = df.index.month
    by_month = pd.DataFrame({
        "Actual": y.groupby(month_idx).mean(),
        "Predicted": pd.Series(np.asarray(y_pred), index=df.index).groupby(month_idx).mean()
    })
    by_month["Residual"] = by_month["Actual"] - by_month["Predicted"]
    by_month.index = [f"{m:02d}" for m in by_month.index]
    st.dataframe(by_month, use_container_width=True)
    fig_mon = go.Figure()
    fig_mon.add_trace(go.Bar(x=by_month.index, y=by_month["Actual"], name="Actual"))
    fig_mon.add_trace(go.Bar(x=by_month.index, y=by_month["Predicted"], name="Predicted"))
    fig_mon.update_layout(barmode="group", title="Monthly averages: Actual vs Predicted",
                          xaxis_title="Month", yaxis_title=target_col)
    st.plotly_chart(fig_mon, use_container_width=True)

    st.divider()

    # Walk-forward backtest with caching
    st.markdown("### Walk-forward Backtest (expanding window, 1-step ahead)")
    cbt1, cbt2, cbt3, cbt4 = st.columns([1,1,1,2])
    with cbt1:
        min_train = st.slider("Min train length", min_value=12, max_value=max(24, len(X)//2), value=min(36, max(24, len(X)//2)))
    with cbt2:
        run_bt = st.button("Run backtest", type="primary")
    with cbt3:
        clear_cache = st.button("Clear backtest cache")
    with cbt4:
        st.caption("Backtest results are cached by min train length.")

    if clear_cache:
        st.session_state.oos_cache = {}

    # Compute & cache this run
    if run_bt:
        oos = walk_forward_predict(X, y, min_train=min_train)
        if not oos.empty:
            y_oos = y.loc[oos.index]
            rmse_oos, mae_oos, mape_oos, hit_oos = metrics_extended(y_oos, oos)
            st.session_state.oos_cache[min_train] = {
                "oos": oos,
                "y_oos": y_oos,
                "metrics": {
                    "RMSE": rmse_oos, "MAE": mae_oos, "MAPE": mape_oos, "Hit": hit_oos
                }
            }
        else:
            st.warning("Sample too short for the selected min train length.")

    # Viewer for cached runs
    if len(st.session_state.oos_cache) > 0:
        st.markdown("#### View cached runs")
        keys_sorted = sorted(st.session_state.oos_cache.keys())
        sel_key = st.selectbox("Select min train length", options=keys_sorted, index=len(keys_sorted)-1)
        cache_run = st.session_state.oos_cache[sel_key]
        oos = cache_run["oos"]
        y_oos = cache_run["y_oos"]
        met = cache_run["metrics"]

        kx1, kx2, kx3, kx4 = st.columns(4)
        kx1.metric("OOS RMSE", f"{met['RMSE']:,.4f}")
        kx2.metric("OOS MAE", f"{met['MAE']:,.4f}")
        kx3.metric("OOS MAPE", f"{met['MAPE']:.2f}%")
        kx4.metric("OOS Directional Hit", f"{met['Hit']:.2f}%")

        fig_oos = go.Figure()
        fig_oos.add_trace(go.Scatter(x=y_oos.index, y=y_oos, mode="lines", name="Actual"))
        fig_oos.add_trace(go.Scatter(x=oos.index, y=oos, mode="lines", name="OOS Pred (1-step)"))
        fig_oos.update_layout(title=f"Walk-forward 1-step: Actual vs OOS Pred (min_train={sel_key})",
                              xaxis_title="Time", yaxis_title=target_col)
        st.plotly_chart(fig_oos, use_container_width=True)

        minv = float(min(y_oos.min(), oos.min()))
        maxv = float(max(y_oos.max(), oos.max()))
        fig_sc_oos = px.scatter(x=oos, y=y_oos, labels={"x": "OOS Pred", "y": "Actual"},
                                title="OOS: Predicted vs Actual")
        fig_sc_oos.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode="lines", name="45° Line"))
        st.plotly_chart(fig_sc_oos, use_container_width=True)
    else:
        st.info("No cached backtests yet. Run one with the button above.")

# ============================ Non Power Tab ============================
with tabs[2]:
    st.subheader("Non Power (placeholder)")
    st.info("Once you provide non power fields, we can mirror the same modeling & visuals here.")
