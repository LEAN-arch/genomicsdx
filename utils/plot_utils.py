# --- SME OVERHAUL: Definitive, Compliance-Focused, and Unabridged Version ---
"""
Plotting utilities for creating standardized, publication-quality visualizations.

This module contains functions that generate various Plotly figures
used throughout the GenomicsDx dashboard. It is augmented with a comprehensive
suite of specialized functions for creating plots essential for analytical
validation (AV), clinical validation (CV), quality control (QC), and process
monitoring for a genomic diagnostic, ensuring a consistent, professional,
and compliant visual style.
"""

# --- Standard Library Imports ---
import logging
from typing import Dict, List, Optional, Tuple, Any
import io

# --- Third-party Imports ---
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from scipy import stats

# --- Setup Logging ---
logger = logging.getLogger(__name__)


# ==============================================================================
# --- MODULE-LEVEL CONFIGURATION CONSTANTS ---
# ==============================================================================
# Centralized configuration for consistent plot styling and logic.

_PLOT_LAYOUT_CONFIG: Dict[str, Any] = {
    "margin": dict(l=50, r=30, t=80, b=50),
    "title_x": 0.5,
    "font": {"family": "Arial, sans-serif", "size": 12},
    "template": "plotly_white"
}

_ACTION_ITEM_COLOR_MAP: Dict[str, str] = {
    "Open": "#ff7f0e", "In Progress": "#1f77b4", "Overdue": "#d62728", "Completed": "#2ca02c"
}

_RISK_CONFIG: Dict[str, Any] = {
    'levels': {
        (1, 1): 'Low', (1, 2): 'Low', (1, 3): 'Medium', (1, 4): 'Medium', (1, 5): 'High',
        (2, 1): 'Low', (2, 2): 'Low', (2, 3): 'Medium', (2, 4): 'High', (2, 5): 'High',
        (3, 1): 'Medium', (3, 2): 'Medium', (3, 3): 'High', (3, 4): 'High', (3, 5): 'Unacceptable',
        (4, 1): 'Medium', (4, 2): 'High', (4, 3): 'High', (4, 4): 'Unacceptable', (4, 5): 'Unacceptable'},
    'colors': {
        'Unacceptable': 'rgba(139, 0, 0, 0.9)', 'High': 'rgba(214, 39, 40, 0.9)',
        'Medium': 'rgba(255, 127, 14, 0.9)', 'Low': 'rgba(44, 160, 44, 0.9)'},
    'order': ['Low', 'Medium', 'High', 'Unacceptable']
}


def _create_placeholder_figure(text: str, title: str, icon: str = "ℹ️") -> go.Figure:
    """Creates a standardized, empty figure with an icon and text annotation."""
    fig = go.Figure()
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        xaxis={'visible': False}, yaxis={'visible': False},
        annotations=[{'text': f"{icon}<br>{text}", 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16, 'color': '#7f7f7f'}}],
        height=300, **_PLOT_LAYOUT_CONFIG
    )
    return fig

# ==============================================================================
# --- GENERAL PURPOSE PLOTTING FUNCTIONS ---
# ==============================================================================

def create_risk_profile_chart(hazards_df: pd.DataFrame) -> go.Figure:
    """Creates a bar chart comparing initial vs. residual risk levels."""
    title = "<b>Risk Profile (Initial vs. Residual)</b>"
    try:
        if hazards_df.empty:
            return _create_placeholder_figure("No Risk Data Available", title)
        df = hazards_df.copy()
        
        get_level = lambda s, o: _RISK_CONFIG['levels'].get((s, o), "N/A")
        df['initial_level'] = df.apply(lambda row: get_level(row.get('initial_S'), row.get('initial_O')), axis=1)
        df['final_level'] = df.apply(lambda row: get_level(row.get('final_S'), row.get('final_O')), axis=1)
        
        risk_levels_order = _RISK_CONFIG['order']
        initial_counts = df['initial_level'].value_counts().reindex(risk_levels_order, fill_value=0)
        final_counts = df['final_level'].value_counts().reindex(risk_levels_order, fill_value=0)
        
        bar_colors = [_RISK_CONFIG['colors'][level] for level in risk_levels_order]
        fig = go.Figure(data=[
            go.Bar(name='Initial Risk', x=risk_levels_order, y=initial_counts.values, text=initial_counts.values, marker=dict(color=bar_colors, line=dict(color='rgba(0,0,0,0.5)', width=1)), opacity=0.6),
            go.Bar(name='Residual Risk', x=risk_levels_order, y=final_counts.values, text=final_counts.values, marker=dict(color=bar_colors, line=dict(color='rgba(0,0,0,1)', width=1.5)))
        ])
        fig.update_layout(barmode='group', title_text=title, legend_title_text='Risk State', xaxis_title="Calculated Risk Level", yaxis_title="Number of Hazards", **_PLOT_LAYOUT_CONFIG)
        fig.update_traces(textposition='outside')
        return fig
    except Exception as e:
        logger.error(f"Error creating risk profile chart: {e}", exc_info=True)
        return _create_placeholder_figure("Risk Chart Error", title, icon="⚠️")

def create_action_item_chart(actions_df: pd.DataFrame) -> go.Figure:
    """Creates a stacked bar chart of open action items."""
    title = "<b>Open Action Items by Owner</b>"
    try:
        if actions_df.empty or 'status' not in actions_df.columns or 'owner' not in actions_df.columns:
            return _create_placeholder_figure("No Action Items Found", title)
        open_items_df = actions_df[actions_df['status'] != 'Completed'].copy()
        if open_items_df.empty:
            return _create_placeholder_figure("All action items are completed.", title, icon="🎉")
        workload = pd.crosstab(index=open_items_df['owner'], columns=open_items_df['status'])
        status_order = ["Overdue", "In Progress", "Open"]
        for status in status_order:
            if status not in workload.columns: workload[status] = 0
        workload = workload[status_order]
        fig = px.bar(workload, title=title, labels={'value': 'Number of Items', 'owner': 'Assigned Owner', 'status': 'Item Status'}, color_discrete_map=_ACTION_ITEM_COLOR_MAP)
        fig.update_layout(barmode='stack', legend_title_text='Status', xaxis={'categoryorder':'total descending'}, **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating action item chart: {e}", exc_info=True)
        return _create_placeholder_figure("Action Item Chart Error", title, icon="⚠️")

# ==============================================================================
# --- SPECIALIZED GENOMICS, QC & ML PLOTS ---
# ==============================================================================

def create_roc_curve(df: pd.DataFrame, score_col: str, truth_col: str, title: str = "Receiver Operating Characteristic (ROC) Curve") -> go.Figure:
    """Generates a ROC curve for a diagnostic test. A cornerstone of any PMA submission."""
    try:
        fpr, tpr, _ = roc_curve(df[truth_col], df[score_col])
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.4f}', line=dict(color='darkorange', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No-Discrimination Line', line=dict(color='navy', width=2, dash='dash')))
        
        fig.update_layout(
            title=f"<b>{title}</b>",
            xaxis_title='1 - Specificity (False Positive Rate)',
            yaxis_title='Sensitivity (True Positive Rate)',
            xaxis=dict(range=[-0.01, 1.01]), yaxis=dict(range=[-0.01, 1.01]),
            legend=dict(x=0.5, y=0.1, xanchor='center', yanchor='bottom', bgcolor='rgba(255,255,255,0.6)'),
            **_PLOT_LAYOUT_CONFIG
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating ROC curve: {e}", exc_info=True)
        return _create_placeholder_figure("ROC Curve Error", title, icon="⚠️")

def create_lod_probit_plot(df: pd.DataFrame, conc_col: str, hit_rate_col: str, title: str = "Limit of Detection (LoD) by Probit Analysis") -> go.Figure:
    """Generates a Probit regression plot to determine the Limit of Detection (LoD)."""
    try:
        df_filtered = df.dropna(subset=[conc_col, hit_rate_col]).copy()
        df_filtered = df_filtered[df_filtered[conc_col] > 0]
        if df_filtered.empty or len(df_filtered) < 2:
            return _create_placeholder_figure("Insufficient data for Probit plot.", title)

        df_filtered['probit_hit_rate'] = stats.norm.ppf(df_filtered[hit_rate_col].clip(0.001, 0.999))
        log_conc = np.log10(df_filtered[conc_col])
        slope, intercept, r_val, p_val, std_err = stats.linregress(log_conc, df_filtered['probit_hit_rate'])
        
        lod_95_probit = stats.norm.ppf(0.95)
        lod_95 = 10**((lod_95_probit - intercept) / slope)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered[conc_col], y=df_filtered[hit_rate_col], mode='markers', name='Observed Data', marker=dict(size=10, color='blue')))
        
        x_fit = np.logspace(np.log10(df_filtered[conc_col].min() * 0.5), np.log10(df_filtered[conc_col].max() * 2), 100)
        y_fit_probit = intercept + slope * np.log10(x_fit)
        y_fit = stats.norm.cdf(y_fit_probit)
        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f'Probit Fit (R²={r_val**2:.3f})', line=dict(color='red', width=3)))
        
        fig.add_hline(y=0.95, line_dash="dash", line_color="black", annotation_text="95% Hit Rate", annotation_position="bottom right")
        fig.add_vline(x=lod_95, line_dash="dash", line_color="black", annotation_text=f"LoD = {lod_95:.4f}", annotation_position="top left")
        
        fig.update_layout(
            title=f"<b>{title}</b>",
            xaxis_title=f"Analyte Concentration ({conc_col})",
            yaxis_title="Detection Rate (Hit Rate)",
            xaxis_type="log", yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            legend=dict(x=0.05, y=0.95, xanchor='left', yanchor='top'),
            **_PLOT_LAYOUT_CONFIG
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating LoD Probit plot: {e}", exc_info=True)
        return _create_placeholder_figure("Probit Plot Error", title, icon="⚠️")

def create_levey_jennings_plot(spc_data: Dict[str, Any]) -> go.Figure:
    """Creates a Levey-Jennings chart for laboratory quality control monitoring."""
    title = "<b>Levey-Jennings Chart: Assay Control Monitoring</b>"
    try:
        if not spc_data or 'measurements' not in spc_data:
            return _create_placeholder_figure("SPC data is incomplete or missing.", title, "📊")

        meas = np.array(spc_data['measurements'])
        mu = spc_data.get('target', meas.mean())
        sigma = spc_data.get('stdev', meas.std())

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=meas, name='Control Value', mode='lines+markers', line=dict(color='#1f77b4'), marker=dict(size=8)))
        
        fig.add_hline(y=mu, line_dash="solid", line_color="green", annotation_text=f"Mean: {mu:.2f}")
        for i, color in zip([1, 2, 3], ["orange", "orange", "red"]):
            fig.add_hline(y=mu + i*sigma, line_dash="dash", line_color=color, annotation_text=f"+{i}SD")
            fig.add_hline(y=mu - i*sigma, line_dash="dash", line_color=color, annotation_text=f"-{i}SD")

        fig.update_layout(title=title, yaxis_title="Measured Value", xaxis_title="Run Number", **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating Levey-Jennings chart: {e}", exc_info=True)
        return _create_placeholder_figure("Levey-Jennings Chart Error", title, icon="⚠️")

def create_bland_altman_plot(df: pd.DataFrame, method1_col: str, method2_col: str, title: str = "Bland-Altman Agreement Plot") -> go.Figure:
    """Generates a Bland-Altman plot to assess agreement between two measurement methods."""
    try:
        df_val = df[[method1_col, method2_col]].dropna().copy()
        if len(df_val) < 2:
            return _create_placeholder_figure("Insufficient data for Bland-Altman plot.", title)

        df_val['average'] = (df_val[method1_col] + df_val[method2_col]) / 2
        df_val['difference'] = df_val[method1_col] - df_val[method2_col]
        
        mean_diff = df_val['difference'].mean()
        std_diff = df_val['difference'].std()
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_val['average'], y=df_val['difference'], mode='markers', name='Differences', marker=dict(color='rgba(31, 119, 180, 0.7)')))
        
        fig.add_hline(y=mean_diff, line=dict(color='blue', width=3, dash='dash'), name=f'Mean Diff: {mean_diff:.3f}')
        fig.add_hline(y=upper_loa, line=dict(color='red', width=2, dash='dash'), name=f'Upper LoA: {upper_loa:.3f}')
        fig.add_hline(y=lower_loa, line=dict(color='red', width=2, dash='dash'), name=f'Lower LoA: {lower_loa:.3f}')

        fig.update_layout(
            title=f"<b>{title}</b><br>({method1_col} vs. {method2_col})",
            xaxis_title="Average of Measurements",
            yaxis_title="Difference between Measurements",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **_PLOT_LAYOUT_CONFIG
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating Bland-Altman plot: {e}", exc_info=True)
        return _create_placeholder_figure("Bland-Altman Plot Error", title, icon="⚠️")

def create_pareto_chart(df: pd.DataFrame, category_col: str, title: str) -> go.Figure:
    """Creates a Pareto chart for identifying the most frequent categories."""
    try:
        if df.empty or category_col not in df.columns:
            return _create_placeholder_figure("No data for Pareto analysis.", title)
        
        counts = df[category_col].value_counts()
        pareto_df = pd.DataFrame({'Category': counts.index, 'Count': counts.values})
        pareto_df = pareto_df.sort_values(by='Count', ascending=False)
        pareto_df['Cumulative Percentage'] = (pareto_df['Count'].cumsum() / pareto_df['Count'].sum()) * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=pareto_df['Category'], y=pareto_df['Count'], name='Count', marker_color='cornflowerblue'), secondary_y=False)
        fig.add_trace(go.Scatter(x=pareto_df['Category'], y=pareto_df['Cumulative Percentage'], name='Cumulative %', mode='lines+markers', line=dict(color='darkorange')), secondary_y=True)

        fig.update_layout(title_text=f"<b>{title}</b>", **_PLOT_LAYOUT_CONFIG)
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Percentage (%)", range=[0, 101], secondary_y=True)
        return fig
    except Exception as e:
        logger.error(f"Error creating Pareto chart: {e}", exc_info=True)
        return _create_placeholder_figure("Pareto Chart Error", title, icon="⚠️")
        
def create_gauge_rr_plot(df: pd.DataFrame, part_col: str, operator_col: str, value_col: str) -> Tuple[go.Figure, pd.DataFrame]:
    """Performs Gauge R&R analysis and returns a summary plot and results table."""
    title = "<b>Measurement System Analysis (Gauge R&R)</b>"
    results_df = pd.DataFrame(columns=['Source', 'Variance Component', '% Contribution']).set_index('Source')
    try:
        # SME FIX: Use robust Q() syntax for patsy formula to handle all column names
        formula = f"Q('{value_col}') ~ C(Q('{operator_col}')) + C(Q('{part_col}')) + C(Q('{operator_col}')):C(Q('{part_col}'))"
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)
        
        ms_operator = anova_table.loc[f"C(Q('{operator_col}'))", 'sum_sq'] / anova_table.loc[f"C(Q('{operator_col}'))", 'df']
        ms_part = anova_table.loc[f"C(Q('{part_col}'))", 'sum_sq'] / anova_table.loc[f"C(Q('{part_col}'))", 'df']
        ms_interact = anova_table.loc[f"C(Q('{operator_col}')):C(Q('{part_col}'))", 'sum_sq'] / anova_table.loc[f"C(Q('{operator_col}')):C(Q('{part_col}'))", 'df']
        ms_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']

        n_parts = df[part_col].nunique()
        n_ops = df[operator_col].nunique()
        reps = len(df) / (n_parts * n_ops)
        
        var_repeat = ms_error
        var_repro = (ms_operator - ms_interact) / (n_parts * reps)
        var_interact = (ms_interact - ms_error) / reps
        var_repro += var_interact
        var_repro = max(0, var_repro)
        var_part = (ms_part - ms_interact) / (n_ops * reps)
        var_part = max(0, var_part)
        
        var_grr = var_repeat + var_repro
        total_var = var_grr + var_part
        
        results_data = {
            'Source': ['Total Gauge R&R', '  Repeatability', '  Reproducibility', 'Part-to-Part', 'Total Variation'],
            'Variance Component': [var_grr, var_repeat, var_repro, var_part, total_var],
            '% Contribution': [
                (var_grr / total_var) * 100 if total_var > 0 else 0, 
                (var_repeat / total_var) * 100 if total_var > 0 else 0, 
                (var_repro / total_var) * 100 if total_var > 0 else 0,
                (var_part / total_var) * 100 if total_var > 0 else 0, 
                100 if total_var > 0 else 0
            ]
        }
        results_df = pd.DataFrame(results_data).set_index('Source')
        
        fig = go.Figure(go.Bar(
            x=results_df['% Contribution'], y=results_df.index, orientation='h',
            marker_color=['crimson', 'lightcoral', 'lightsalmon', 'lightseagreen', 'grey']
        ))
        fig.update_layout(title_text=title, xaxis_title="% Contribution to Total Variation", **_PLOT_LAYOUT_CONFIG)
        return fig, results_df.round(4)
    except Exception as e:
        logger.error(f"Error creating Gauge R&R plot: {e}", exc_info=True)
        return _create_placeholder_figure("Gauge R&R Error", title, "⚠️"), results_df

def create_tost_plot(a: np.ndarray, b: np.ndarray, low: float, high: float) -> Tuple[go.Figure, float]:
    """Performs TOST and returns a plot and the max p-value."""
    title = "<b>Equivalence Test (TOST) Results</b>"
    try:
        diff = a.mean() - b.mean()
        n1, n2 = len(a), len(b)
        s_pool = np.sqrt(((n1 - 1) * a.var() + (n2 - 1) * b.var()) / (n1 + n2 - 2))
        se_diff = s_pool * np.sqrt(1/n1 + 1/n2)
        
        t_stat1 = (diff - low) / se_diff
        t_stat2 = (diff - high) / se_diff
        
        p1 = stats.t.sf(t_stat1, df=n1 + n2 - 2)
        p2 = stats.t.cdf(t_stat2, df=n1 + n2 - 2)
        p_value = max(p1, p2)

        ci_90 = stats.t.interval(0.90, df=n1+n2-2, loc=diff, scale=se_diff)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[ci_90[0], ci_90[1]], y=[1, 1], mode='lines', line=dict(color='blue', width=5), name='90% CI of Difference'))
        fig.add_trace(go.Scatter(x=[diff], y=[1], mode='markers', marker=dict(color='blue', size=12, symbol='x'), name='Mean Difference'))
        fig.add_shape(type='line', x0=low, y0=0, x1=low, y1=2, line=dict(color='red', dash='dash'), name='Lower Equivalence Bound')
        fig.add_shape(type='line', x0=high, y0=0, x1=high, y1=2, line=dict(color='red', dash='dash'), name='Upper Equivalence Bound')
        
        fig.update_layout(title=title, yaxis_visible=False, xaxis_title="Difference in Means", **_PLOT_LAYOUT_CONFIG)
        fig.update_xaxes(range=[min(low*1.2, ci_90[0]*1.2), max(high*1.2, ci_90[1]*1.2)])
        return fig, p_value
    except Exception as e:
        logger.error(f"Error creating TOST plot: {e}", exc_info=True)
        return _create_placeholder_figure("TOST Plot Error", title, "⚠️"), 1.0

def create_confusion_matrix_heatmap(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    """Creates a professional heatmap for a confusion matrix."""
    try:
        z = cm
        x = class_names
        y = class_names
        z_text = [[str(y) for y in x] for x in z]
        
        fig = go.Figure(data=go.Heatmap(
            z=z, x=x, y=y,
            text=z_text, texttemplate="%{text}",
            colorscale='Blues', showscale=False
        ))
        
        fig.update_layout(
            title_text="<b>Confusion Matrix</b>",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            **_PLOT_LAYOUT_CONFIG
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating confusion matrix heatmap: {e}", exc_info=True)
        return _create_placeholder_figure("Confusion Matrix Error", "Confusion Matrix", "⚠️")

def create_shap_summary_plot(shap_values: np.ndarray, features: pd.DataFrame) -> Optional[io.BytesIO]:
    """
    Creates a SHAP summary plot and returns it as an in-memory PNG image buffer.
    Returns None on failure.
    """
    try:
        import shap
        import matplotlib.pyplot as plt

        if shap_values.shape[1] != features.shape[1]:
            logger.error(f"SHAP plot error: Mismatch in shapes. SHAP values have {shap_values.shape[1]} features, data has {features.shape[1]}.")
            return None

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, features, show=False)
        plt.title("SHAP Feature Importance Summary", fontsize=16)
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        plt.close(fig) # Close the figure to free up memory
        return buf
    except Exception as e:
        logger.error(f"Error creating SHAP summary plot: {e}", exc_info=True)
        return None

def create_forecast_plot(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """Creates a time series forecast plot."""
    title = "<b>Time Series Forecast</b>"
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_df.index, y=history_df.iloc[:, 0], name='Historical Data', line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], name='Forecast', line=dict(color='darkorange', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], fill='tonexty', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'), name='95% CI Upper'))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], fill='tonexty', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'), name='95% CI Lower'))
        
        fig.update_layout(title_text=title, xaxis_title="Date", yaxis_title="Value", **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating forecast plot: {e}", exc_info=True)
        return _create_placeholder_figure("Forecast Plot Error", title, "⚠️")

def create_doe_effects_plot(df: pd.DataFrame, factor1: str, factor2: str, response: str) -> Tuple[go.Figure, go.Figure]:
    """
    Calculates and plots the main and interaction effects for a 2x2 factorial DOE.
    This is a numerically stable alternative to fitting a regression model.
    """
    try:
        # Calculate mean response for each of the 4 combinations
        means = df.groupby([factor1, factor2])[response].mean()
        
        f1_low, f1_high = df[factor1].unique()
        f2_low, f2_high = df[factor2].unique()

        y1 = means.loc[f1_low, f2_low]    # Factor1 Low, Factor2 Low
        y2 = means.loc[f1_high, f2_low]   # Factor1 High, Factor2 Low
        y3 = means.loc[f1_low, f2_high]   # Factor1 Low, Factor2 High
        y4 = means.loc[f1_high, f2_high]  # Factor1 High, Factor2 High

        # --- Main Effects Calculation ---
        main_effect_f1 = ((y2 - y1) + (y4 - y3)) / 2
        main_effect_f2 = ((y3 - y1) + (y4 - y2)) / 2
        effects_data = pd.DataFrame([
            {'Effect': f'Main Effect: {factor1}', 'Value': main_effect_f1},
            {'Effect': f'Main Effect: {factor2}', 'Value': main_effect_f2}
        ])
        
        # --- Interaction Effect Calculation ---
        interaction_effect = ((y4 - y3) - (y2 - y1)) / 2
        effects_data.loc[len(effects_data)] = {'Effect': f'Interaction: {factor1}:{factor2}', 'Value': interaction_effect}

        # --- Create Main Effects Plot ---
        effects_fig = px.bar(effects_data, x='Effect', y='Value', title="<b>Calculated Factor Effects</b>",
                             color='Effect', text_auto='.3f')
        effects_fig.update_layout(showlegend=False, yaxis_title="Effect on Response", **_PLOT_LAYOUT_CONFIG)

        # --- Create Interaction Plot ---
        interaction_df = means.reset_index()
        interaction_fig = px.line(interaction_df, x=factor1, y=response, color=factor2, markers=True,
                                  title=f"<b>Interaction Plot: {factor1} vs {factor2}</b>")
        interaction_fig.update_traces(marker=dict(size=12), line=dict(width=3))
        interaction_fig.update_layout(
            xaxis=dict(type='category'),
            xaxis_title=f"Factor: {factor1}",
            yaxis_title=f"Mean {response}",
            legend_title=f"Factor: {factor2}",
            **_PLOT_LAYOUT_CONFIG
        )
        
        return effects_fig, interaction_fig

    except Exception as e:
        logger.error(f"Error creating DOE effects plot: {e}", exc_info=True)
        title = "<b>DOE Analysis Error</b>"
        return _create_placeholder_figure("Could not calculate effects.", title, "⚠️"), \
               _create_placeholder_figure("Could not create interaction plot.", title, "⚠️")
