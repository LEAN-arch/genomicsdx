# --- SME OVERHAUL: Definitive, Compliance-Focused Version (Corrected) ---
"""
Plotting utilities for creating standardized, publication-quality visualizations.

This module contains functions that generate various Plotly figures
used throughout the GenomicsDx dashboard. It is augmented with specialized
functions for creating plots essential for analytical validation (AV), clinical
validation (CV), and laboratory quality control (QC) for a genomic diagnostic,
ensuring a consistent, professional, and compliant visual style.
"""

# --- Standard Library Imports ---
import logging
from typing import Dict, List, Optional, Tuple, Any

# --- Third-party Imports ---
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import stats

# --- SYNTAX CORRECTION ---
# 'probit' has been moved to scipy.special in modern versions.
from scipy.special import probit
# --- END CORRECTION ---

# --- Setup Logging ---
logger = logging.getLogger(__name__)


# ==============================================================================
# --- MODULE-LEVEL CONFIGURATION CONSTANTS ---
# ==============================================================================
# Centralized configuration for consistent plot styling and logic.

_PLOT_LAYOUT_CONFIG: Dict[str, any] = {
    "margin": dict(l=40, r=20, t=60, b=40),
    "title_x": 0.5,
    "font": {"family": "sans-serif"}
}

_ACTION_ITEM_COLOR_MAP: Dict[str, str] = {
    "Open": "#ff7f0e", "In Progress": "#1f77b4", "Overdue": "#d62728", "Completed": "#2ca02c"
}

_RISK_CONFIG: Dict[str, any] = {
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
# --- GENERAL PURPOSE PLOTTING FUNCTIONS (RETAINED) ---
# ==============================================================================

def create_risk_profile_chart(hazards_df: pd.DataFrame) -> go.Figure:
    """Creates a bar chart comparing initial vs. residual risk levels."""
    title = "<b>Risk Profile (Initial vs. Residual)</b>"
    try:
        if hazards_df.empty:
            return _create_placeholder_figure("No Risk Data Available", title)
        # ... (full implementation) ...
        df = hazards_df.copy()
        df['initial_level'] = df.apply(lambda row: _RISK_CONFIG['levels'].get((row.get('initial_S'), row.get('initial_O')), "N/A"), axis=1)
        df['final_level'] = df.apply(lambda row: _RISK_CONFIG['levels'].get((row.get('final_S'), row.get('final_O')), "N/A"), axis=1)
        risk_levels_order = _RISK_CONFIG['order']
        initial_counts = df['initial_level'].value_counts().reindex(risk_levels_order, fill_value=0)
        final_counts = df['final_level'].value_counts().reindex(risk_levels_order, fill_value=0)
        bar_colors = [_RISK_CONFIG['colors'][level] for level in risk_levels_order]
        fig = go.Figure(data=[
            go.Bar(name='Initial Risk', x=risk_levels_order, y=initial_counts.values, text=initial_counts.values, marker=dict(color=bar_colors, line=dict(color='rgba(0,0,0,0.5)', width=1)), opacity=0.6),
            go.Bar(name='Residual Risk', x=risk_levels_order, y=final_counts.values, text=final_counts.values, marker=dict(color=bar_colors, line=dict(color='rgba(0,0,0,1)', width=1.5)))
        ])
        fig.update_layout(barmode='group', title_text=title, legend_title_text='Risk State', xaxis_title="Calculated Risk Level", yaxis_title="Number of Hazards", height=300, **_PLOT_LAYOUT_CONFIG)
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
        fig.update_layout(barmode='stack', legend_title_text='Status', xaxis={'categoryorder':'total descending'}, height=300, **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating action item chart: {e}", exc_info=True)
        return _create_placeholder_figure("Action Item Chart Error", title, icon="⚠️")


# ==============================================================================
# --- SME-AUGMENTED: SPECIALIZED GENOMICS & QC PLOTS ---
# ==============================================================================

def create_roc_curve(df: pd.DataFrame, score_col: str, truth_col: str, title: str = "Receiver Operating Characteristic (ROC) Curve") -> go.Figure:
    """
    Generates a ROC curve for a diagnostic test. A cornerstone of any PMA submission.
    """
    try:
        fpr, tpr, _ = roc_curve(df[truth_col], df[score_col])
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.3f}', line=dict(color='darkorange', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='navy', width=2, dash='dash')))
        
        fig.update_layout(
            title=f"<b>{title}</b>",
            xaxis_title='1 - Specificity (False Positive Rate)',
            yaxis_title='Sensitivity (True Positive Rate)',
            legend=dict(x=0.5, y=0.1, xanchor='center', yanchor='bottom'),
            **_PLOT_LAYOUT_CONFIG
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating ROC curve: {e}", exc_info=True)
        return _create_placeholder_figure("ROC Curve Error", title, icon="⚠️")

def create_lod_probit_plot(df: pd.DataFrame, conc_col: str, hit_rate_col: str, title: str = "Limit of Detection (LoD) by Probit Analysis") -> go.Figure:
    """
    Generates a Probit regression plot to determine the Limit of Detection (LoD).
    Essential for Analytical Validation reports.
    """
    try:
        df_filtered = df.dropna(subset=[conc_col, hit_rate_col]).copy()
        df_filtered = df_filtered[df_filtered[conc_col] > 0] # Log scale requires positive concentrations
        if df_filtered.empty or len(df_filtered) < 2:
            return _create_placeholder_figure("Insufficient data for Probit plot.", title)

        # Using the corrected probit import from scipy.special
        df_filtered['probit_hit_rate'] = probit(df_filtered[hit_rate_col])
        
        log_conc = np.log10(df_filtered[conc_col])
        slope, intercept, _, _, _ = stats.linregress(log_conc, df_filtered['probit_hit_rate'])
        
        lod_95 = 10**((probit(0.95) - intercept) / slope)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered[conc_col], y=df_filtered[hit_rate_col], mode='markers', name='Observed Data', marker=dict(size=10)))
        
        x_fit = np.logspace(np.log10(df_filtered[conc_col].min()), np.log10(df_filtered[conc_col].max()), 100)
        y_fit_probit = intercept + slope * np.log10(x_fit)
        y_fit = stats.norm.cdf(y_fit_probit)
        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Probit Fit', line=dict(color='red')))
        
        fig.add_hline(y=0.95, line_dash="dash", annotation_text="95% Hit Rate", annotation_position="bottom right")
        fig.add_vline(x=lod_95, line_dash="dash", annotation_text=f"LoD = {lod_95:.4f}", annotation_position="top left")
        
        fig.update_layout(
            title=f"<b>{title}</b>",
            xaxis_title=f"Concentration ({conc_col})",
            yaxis_title="Detection Rate",
            xaxis_type="log",
            yaxis=dict(range=[0, 1.05]),
            **_PLOT_LAYOUT_CONFIG
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating LoD Probit plot: {e}", exc_info=True)
        return _create_placeholder_figure("Probit Plot Error", title, icon="⚠️")

def create_levey_jennings_plot(spc_data: Dict[str, Any]) -> go.Figure:
    """
    Creates a Levey-Jennings chart for laboratory quality control monitoring.
    """
    title = "Levey-Jennings Chart: Assay Control Monitoring"
    try:
        if not spc_data or 'measurements' not in spc_data:
            return _create_placeholder_figure("SPC data is incomplete or missing.", title, "📊")

        meas = np.array(spc_data['measurements'])
        mu = spc_data.get('target', meas.mean())
        sigma = spc_data.get('stdev', meas.std())

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=meas, name='Control Value', mode='lines+markers', line=dict(color='#1f77b4')))
        
        fig.add_hline(y=mu, line_dash="solid", line_color="black", annotation_text="Mean")
        for i in [1, 2, 3]:
            fig.add_hline(y=mu + i*sigma, line_dash="dash", line_color="orange" if i < 3 else "red", annotation_text=f"+{i}SD")
            fig.add_hline(y=mu - i*sigma, line_dash="dash", line_color="orange" if i < 3 else "red", annotation_text=f"-{i}SD")

        fig.update_layout(title=f"<b>{title}</b>", yaxis_title="Measured Value", xaxis_title="Run Number", **_PLOT_LAYOUT_CONFIG)
        return fig
    except Exception as e:
        logger.error(f"Error creating Levey-Jennings chart: {e}", exc_info=True)
        return _create_placeholder_figure("Levey-Jennings Chart Error", title, icon="⚠️")

def create_bland_altman_plot(df: pd.DataFrame, method1_col: str, method2_col: str, title: str = "Bland-Altman Agreement Plot") -> go.Figure:
    """
    Generates a Bland-Altman plot to assess agreement between two measurement methods.
    """
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
        fig.add_trace(go.Scatter(x=df_val['average'], y=df_val['difference'], mode='markers', name='Differences'))
        
        fig.add_hline(y=mean_diff, line=dict(color='blue', dash='dash'), name=f'Mean Diff: {mean_diff:.3f}')
        fig.add_hline(y=upper_loa, line=dict(color='red', dash='dash'), name=f'Upper LoA: {upper_loa:.3f}')
        fig.add_hline(y=lower_loa, line=dict(color='red', dash='dash'), name=f'Lower LoA: {lower_loa:.3f}')

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
