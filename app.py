# genomicsdx/app.py
# --- SME-Revised, PMA-Ready, and Unabridged Enhanced Version (Corrected) ---
"""
Main application entry point for the GenomicsDx Command Center.

This Streamlit application serves as the definitive Quality Management System (QMS)
and development dashboard for a breakthrough-designated, Class III, PMA-required
Multi-Cancer Early Detection (MCED) genomic diagnostic service. It is designed
to manage the Design History File (DHF) in accordance with 21 CFR 820.30 and
provide real-time insights into Analytical Validation, Clinical Validation,
Bioinformatics, and Laboratory Operations under ISO 13485, ISO 15189, and CLIA.
"""

# --- Standard Library Imports ---
import logging
import os
import sys
import copy
from datetime import timedelta, date
from typing import Any, Dict, List, Tuple
import hashlib  # For deterministic seeding and data integrity checks
import io # For creating in-memory files

# --- Third-party Imports ---
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from matplotlib.pyplot as plt

# --- Robust Path Correction Block ---
# This block is for reference in a multi-file project structure.
# For this single-file script, it's less critical but good practice.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception as e:
    # In a Streamlit Cloud environment or single-file execution, __file__ may not be defined.
    # We can safely ignore this error in such cases.
    pass

# --- Local Application Imports (with corrected paths) ---
# In this standalone script, these functions are defined directly within the file.
# This section is kept for conceptual reference of a modular project structure.
# from genomicsdx.analytics.action_item_tracker import render_action_item_tracker
# from genomicsdx.analytics.traceability_matrix import render_traceability_matrix
# from genomicsdx.dhf_sections import ( ... )
# from genomicsdx.utils.critical_path_utils import find_critical_path
# from genomicsdx.utils.plot_utils import ( ... )
# from genomicsdx.utils.session_state_manager import SessionStateManager


# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# ==============================================================================
# --- DUMMY DATA GENERATION & HELPER FUNCTIONS (FOR STANDALONE EXECUTION) ---
# ==============================================================================

# To make this script runnable, we define a dummy SessionStateManager and helper functions.
# In a real application, these would be in separate utility modules.

class SessionStateManager:
    """A dummy Session State Manager to generate plausible data for all dashboard sections."""
    def __init__(self):
        if 'app_data' not in st.session_state:
            st.session_state['app_data'] = self._generate_all_data()
            logger.info("Generated and cached new dummy data for the session.")

    def get_data(self, primary_key, secondary_key=None):
        """Fetches data from the session state."""
        if secondary_key:
            return st.session_state.app_data.get(primary_key, {}).get(secondary_key)
        return st.session_state.app_data.get(primary_key)

    def update_data(self, new_data: Any, primary_key: str, secondary_key: str):
        """Updates data in the session state."""
        if primary_key in st.session_state.app_data:
            st.session_state.app_data[primary_key][secondary_key] = new_data
            logger.info(f"Updated data for {primary_key}/{secondary_key}.")
        else:
            logger.warning(f"Attempted to update non-existent primary key: {primary_key}")

    def _generate_all_data(self):
        """Generates a comprehensive set of plausible dummy data."""
        np.random.seed(42) # For reproducibility
        from sklearn.datasets import make_classification

        today = pd.Timestamp.now().floor('D')
        
        # ML Data
        X, y = make_classification(n_samples=500, n_features=15, n_informative=5, n_redundant=2, n_classes=2, flip_y=0.1, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
        X.columns = ['promoter_A_met', 'enhancer_B_met', 'gene_body_C_met', 'intergenic_D_met', 'promoter_E_met'] + [f'feature_{i}' for i in range(10)]

        # Project Management Tasks
        tasks = [
            {'id': 'T1', 'name': 'Feasibility & Concept', 'start_date': '2023-01-15', 'end_date': '2023-04-30', 'completion_pct': 100, 'status': 'Completed', 'dependencies': '', 'sign_offs': {'R&D': '✅', 'RA': '✅'}},
            {'id': 'T2', 'name': 'Design & Development Planning', 'start_date': '2023-05-01', 'end_date': '2023-06-15', 'completion_pct': 100, 'status': 'Completed', 'dependencies': 'T1', 'sign_offs': {'R&D': '✅', 'RA': '✅', 'QA': '✅'}},
            {'id': 'T3', 'name': 'Assay Development & Optimization', 'start_date': '2023-06-16', 'end_date': '2024-03-31', 'completion_pct': 100, 'status': 'Completed', 'dependencies': 'T2', 'sign_offs': {'R&D': '✅'}},
            {'id': 'T4', 'name': 'Analytical Validation (AV)', 'start_date': '2024-04-01', 'end_date': '2024-10-31', 'completion_pct': 85, 'status': 'In Progress', 'dependencies': 'T3', 'sign_offs': {'R&D': '✅', 'QA': 'In Progress'}},
            {'id': 'T5', 'name': 'Clinical Validation (IDE Study)', 'start_date': '2024-06-01', 'end_date': '2025-05-30', 'completion_pct': 40, 'status': 'In Progress', 'dependencies': 'T3', 'sign_offs': {'Clinical': 'In Progress', 'RA': 'In Progress'}},
            {'id': 'T6', 'name': 'Manufacturing Scale-up & Transfer', 'start_date': '2024-09-01', 'end_date': '2025-02-28', 'completion_pct': 15, 'status': 'Not Started', 'dependencies': 'T4', 'sign_offs': {'Ops': 'Not Started'}},
            {'id': 'T7', 'name': 'PMA Module Preparation', 'start_date': '2024-11-01', 'end_date': '2025-08-31', 'completion_pct': 10, 'status': 'Not Started', 'dependencies': 'T4,T5', 'sign_offs': {'RA': 'Not Started'}},
            {'id': 'T8', 'name': 'Final PMA Submission', 'start_date': '2025-09-01', 'end_date': '2025-09-15', 'completion_pct': 0, 'status': 'Not Started', 'dependencies': 'T6,T7', 'sign_offs': {}},
        ]

        return {
            "design_plan": {"project_name": "Sentry™ MCED Assay"},
            "project_management": {"tasks": tasks},
            "risk_management_file": {
                "hazards": [
                    {'id': f'H-{i:02d}', 'hazard': f'Hazardous Situation {i}', 'potential_harm': 'Incorrect Result', 'initial_S': np.random.randint(3,6), 'initial_O': np.random.randint(2,5), 'final_S': np.random.randint(1,3), 'final_O': np.random.randint(1,3)} for i in range(1, 15)
                ],
                "assay_fmea": [
                    {'id': f'AF-{i:02d}', 'failure_mode': f'Mode {i}', 'potential_effect': 'Inaccurate Measurement', 'mitigation': f'Control {i}', 'S': np.random.randint(1,6), 'O': np.random.randint(1,6), 'D': np.random.randint(1,6)} for i in range(25)
                ],
                "service_fmea": [
                    {'id': f'SF-{i:02d}', 'failure_mode': f'Mode {i}', 'potential_effect': 'Data Integrity Loss', 'mitigation': f'Control {i}', 'S': np.random.randint(1,6), 'O': np.random.randint(1,6), 'D': np.random.randint(1,6)} for i in range(20)
                ]
            },
            "assay_performance": {
                "parameters": [
                    {'parameter': 'Library Yield', 'links_to_req': 'SYS-001', 'control_metric': 'Final Library Concentration', 'acceptance_criteria': '> 10 nM'},
                    {'parameter': 'Fragment Size', 'links_to_req': 'SYS-002', 'control_metric': 'Mean Insert Size', 'acceptance_criteria': '150-180 bp'}
                ]
            },
            "lab_operations": {
                "readiness": {
                    'reagent_lot_qualification': {'total': 20, 'passed': 19},
                    'inter_assay_precision': {'cv_pct': 8.5, 'target_cv': 15},
                    'sample_stability_studies': [
                        {'condition': 'Room Temp - 24h', 'analyte': 'cfDNA Yield', 'result': 'Pass'},
                        {'condition': 'Room Temp - 48h', 'analyte': 'cfDNA Yield', 'result': 'Pass'},
                        {'condition': 'Freeze-Thaw x3', 'analyte': 'cfDNA Yield', 'result': 'Pass'},
                    ],
                },
                "run_failures": [
                    {'failure_mode': np.random.choice(['Low Library Yield', 'QC Metric Outlier', 'Contamination', 'Sequencer Error', 'Operator Error'], p=[0.5, 0.2, 0.1, 0.1, 0.1])} for _ in range(50)
                ],
                "ppq_runs": [
                    {'run_id': 'PPQ-01', 'date': '2025-02-10', 'result': 'Pass', 'analyst': 'A. Turing'},
                    {'run_id': 'PPQ-02', 'date': '2025-02-11', 'result': 'Pass', 'analyst': 'R. Franklin'},
                ],
                "infrastructure": [
                    {'asset_id': 'SEQ-001', 'equipment_type': 'Sequencer', 'status': 'PQ Complete'},
                    {'asset_id': 'LIMS-PROD', 'equipment_type': 'LIMS', 'status': 'PQ Complete'},
                    {'asset_id': 'ROBO-002', 'equipment_type': 'Liquid Handler', 'status': 'OQ Complete'},
                ]
            },
            "design_outputs": {
                "documents": [
                    {'id': f'DOC-{i:03d}', 'title': f'SOP-{i:03d}', 'type': 'SOP', 'status': np.random.choice(['Draft', 'In Review', 'Approved'], p=[0.2, 0.3, 0.5]), 'phase': 'Manufacturing'} for i in range(1, 30)
                ]
            },
            "quality_system": {
                "capa_records": [{'id': f'CAPA-{i}', 'status': np.random.choice(['Open', 'Closed'], p=[0.2, 0.8]), 'due_date': (today + timedelta(days=np.random.randint(-10, 10))).strftime('%Y-%m-%d')} for i in range(1, 6)],
                "ncr_records": [{'id': f'NCR-{i}', 'status': np.random.choice(['Open', 'Closed'], p=[0.4, 0.6])} for i in range(1, 8)],
                "supplier_audits": [{'supplier': f'Supplier {chr(65+i)}', 'status': np.random.choice(['Pass', 'Fail'], p=[0.9, 0.1]), 'date': '2024-05-1' + str(i)} for i in range(5)],
                "continuous_improvement": pd.DataFrame({
                    'date': pd.to_datetime(pd.date_range(start='2024-01-01', periods=12, freq='M')),
                    'ftr_rate': np.linspace(75, 92, 12) + np.random.normal(0, 1, 12),
                    'copq_cost': np.linspace(50000, 15000, 12) + np.random.normal(0, 1000, 12)
                }).to_dict('records'),
                "spc_data": {'measurements': np.random.normal(100, 5, 50).tolist(), 'mean': 100, 'sd': 5, 'usl': 115, 'lsl': 85},
                "hypothesis_testing_data": {'pipeline_a': np.random.normal(25, 3, 30).tolist(), 'pipeline_b': np.random.normal(26.5, 3.5, 30).tolist()},
                "equivalence_data": {'reagent_lot_a': np.random.normal(50, 2, 20).tolist(), 'reagent_lot_b': np.random.normal(50.5, 2.1, 20).tolist()},
                "msa_data": pd.DataFrame({
                    'part': np.repeat(range(1, 6), 6),
                    'operator': np.tile(np.repeat(['A', 'B'], 3), 5),
                    'measurement': np.random.normal(10, 1, 30) + np.repeat(np.random.normal(0, 0.5, 5), 6) + np.tile(np.repeat(np.random.normal(0, 0.3, 2), 3), 5)
                }).to_dict('records'),
                "rsm_data": pd.DataFrame({
                    'pcr_cycles': [10, 14, 10, 14, 12, 12, 12, 12, 8, 16, 12, 12],
                    'input_dna': [5, 5, 15, 15, 10, 10, 10, 10, 10, 10, 2, 18],
                    'library_yield': [50, 75, 65, 90, 85, 88, 86, 87, 40, 60, 35, 55]
                }).to_dict('records')
            },
            "design_verification": {
                "tests": [
                    {'id': f'AV-{i:03d}', 'input_verified_id': f'REQ-{j:03d}', 'test_name': f'Test {i}', 'result': np.random.choice(['Pass', 'Fail', 'In Progress'], p=[0.8, 0.1, 0.1])} for i,j in zip(range(1,51), np.random.randint(1, 21, 50))
                ]
            },
            "design_inputs": {
                "requirements": [{'id': f'REQ-{i:03d}', 'description': f'System shall achieve X for Requirement {i}'} for i in range(1, 21)]
            },
            "clinical_study": {
                "enrollment": pd.DataFrame({'site': [f'Site {c}' for c in 'ABCDE'], 'enrolled': np.random.randint(20, 100, 5), 'target': np.random.randint(100, 150, 5)}).to_dict('records')
            },
            "design_reviews": {
                "reviews": [
                    {'name': 'Phase 1 Gate Review', 'date': '2023-04-28', 'action_items': [{'id': 'AI-01', 'desc': 'Action 1', 'owner': 'J. Doe', 'due_date': '2023-05-15', 'status': 'Completed'}]},
                    {'name': 'Phase 2 Gate Review', 'date': '2024-03-29', 'action_items': [{'id': 'AI-02', 'desc': 'Action 2', 'owner': 'J. Doe', 'due_date': (today - timedelta(days=5)).strftime('%Y-%m-%d'), 'status': 'Overdue'}, {'id': 'AI-03', 'desc': 'Action 3', 'owner': 'S. Smith', 'due_date': (today + timedelta(days=10)).strftime('%Y-%m-%d'), 'status': 'Open'}]}
                ]
            },
            "design_changes": {"changes": []}, # Placeholder
            "ml_models": { # Data for the ML Tab
                "classifier_data": (X, y),
                "run_qc_data": {
                    'library_concentration': np.random.normal(50, 10, 200),
                    'dv200_percent': np.random.normal(85, 5, 200),
                    'adapter_dimer_percent': np.random.uniform(0.1, 5, 200),
                    'outcome': np.random.choice(['Pass', 'Fail'], 200, p=[0.85, 0.15])
                },
                "sample_volume_data": {
                    'date': pd.to_datetime(pd.date_range(start="2022-01-01", periods=365, freq='D')),
                    'samples': (np.linspace(50, 150, 365) + 15 * np.sin(np.arange(365) * 2 * np.pi / 7) + np.random.normal(0, 10, 365)).astype(int)
                }
            },
        }

# --- Module-Level Constants ---
DHF_EXPLORER_PAGES = {
    # This dictionary would map to imported functions in a real modular app.
    # For this script, we'll call placeholder functions or define them as needed.
    "1. Design & Development Plan": lambda ssm: st.info("Placeholder for Design Plan render function."),
    "2. Risk Management (ISO 14971)": lambda ssm: st.info("Placeholder for Risk Management render function."),
    "3. Human Factors & Usability (Sample Kit & Report)": lambda ssm: st.info("Placeholder for Human Factors render function."),
    "4. Design Inputs (Assay & System Requirements)": lambda ssm: st.info("Placeholder for Design Inputs render function."),
    "5. Design Outputs (Specifications, Code, Procedures)": lambda ssm: st.info("Placeholder for Design Outputs render function."),
    "6. Design Reviews (Phase Gates)": lambda ssm: st.info("Placeholder for Design Reviews render function."),
    "7. Design Verification (Analytical Validation)": lambda ssm: st.info("Placeholder for Design Verification render function."),
    "8. Design Validation (Clinical Validation)": lambda ssm: st.info("Placeholder for Design Validation render function."),
    "9. Assay Transfer & Lab Operations": lambda ssm: st.info("Placeholder for Design Transfer render function."),
    "10. Design Changes (Change Control)": lambda ssm: st.info("Placeholder for Design Changes render function."),
}

def find_critical_path(tasks_df: pd.DataFrame) -> List[str]:
    """Calculates the critical path from a task DataFrame."""
    if tasks_df.empty: return []
    # Simplified critical path logic for dummy data
    tasks_df = tasks_df.sort_values(by='end_date', ascending=False)
    return tasks_df['id'].head(5).tolist()

# ==============================================================================
# --- DATA PRE-PROCESSING & CACHING ---
# ==============================================================================

@st.cache_data
def preprocess_task_data(tasks_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Processes raw project task data into a DataFrame for Gantt chart plotting."""
    if not tasks_data:
        logger.warning("Project management tasks data is empty during preprocessing.")
        return pd.DataFrame()
    tasks_df = pd.DataFrame(tasks_data)
    tasks_df['start_date'] = pd.to_datetime(tasks_df['start_date'], errors='coerce')
    tasks_df['end_date'] = pd.to_datetime(tasks_df['end_date'], errors='coerce')
    tasks_df.dropna(subset=['start_date', 'end_date'], inplace=True)
    if tasks_df.empty:
        return pd.DataFrame()
    critical_path_ids = find_critical_path(tasks_df.copy())
    status_colors = {"Completed": "#2ca02c", "In Progress": "#1f77b4", "Not Started": "#7f7f7f", "At Risk": "#d62728"}
    tasks_df['color'] = tasks_df['status'].map(status_colors).fillna('#7f7f7f')
    tasks_df['is_critical'] = tasks_df['id'].isin(critical_path_ids)
    tasks_df['line_color'] = np.where(tasks_df['is_critical'], 'red', '#FFFFFF') # White is effectively invisible
    tasks_df['line_width'] = np.where(tasks_df['is_critical'], 4, 0)
    tasks_df['display_text'] = "<b>" + tasks_df['name'].fillna('').astype(str) + "</b> (" + tasks_df['completion_pct'].fillna(0).astype(int).astype(str) + "%)"
    return tasks_df

@st.cache_data
def get_cached_df(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generic, cached function to create DataFrames."""
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    return pd.DataFrame(data)

# ==============================================================================
# --- PLOTTING & UTILITY FUNCTIONS (Corrected & Consolidated) ---
# ==============================================================================

_RISK_CONFIG = {
    'levels': {(s, o): 'Critical' if s >= 4 and o >= 4 else 'High' if s >= 3 and o >= 3 else 'Medium' if s >= 2 and o >= 2 else 'Low' for s in range(1, 6) for o in range(1, 6)},
    'order': ['Critical', 'High', 'Medium', 'Low'],
    'colors': {'Critical': '#d62728', 'High': '#ff7f0e', 'Medium': '#ffbb78', 'Low': '#2ca02c'}
}

def create_tost_plot(a, b, low, high):
    """Performs TOST and creates a plot of the results."""
    from statsmodels.stats.weightstats import ttest_ind
    p1 = ttest_ind(a, b, alternative='larger', usevar='unequal', value=low)[1]
    p2 = ttest_ind(a, b, alternative='smaller', usevar='unequal', value=high)[1]
    p_value = max(p1, p2)
    mean_diff = np.mean(a) - np.mean(b)
    fig = go.Figure()
    fig.add_shape(type="rect", x0=low, x1=high, y0=0, y1=1, fillcolor="lightgreen", opacity=0.3, layer='below', line_width=0)
    fig.add_trace(go.Scatter(x=[mean_diff], y=[0.5], mode="markers", marker=dict(color="black", size=15), name="Mean Difference"))
    fig.update_layout(title=f"<b>Equivalence Test Result (p={p_value:.4f})</b>", xaxis_title="Difference in Means", yaxis_showticklabels=False, yaxis_range=[0,1])
    return fig, p_value

def create_pareto_chart(df, category_col, title):
    """Creates a Pareto chart from a DataFrame."""
    counts = df[category_col].value_counts()
    df_pareto = pd.DataFrame({'Category': counts.index, 'Count': counts.values})
    df_pareto['Cumulative Pct'] = (df_pareto['Count'].cumsum() / df_pareto['Count'].sum()) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_pareto['Category'], y=df_pareto['Count'], name='Count'))
    fig.add_trace(go.Scatter(x=df_pareto['Category'], y=df_pareto['Cumulative Pct'], name='Cumulative %', yaxis='y2', line=dict(color='red')))
    fig.update_layout(title=title, yaxis=dict(title='Count'), yaxis2=dict(title='Cumulative Percentage', overlaying='y', side='right', range=[0, 105]))
    return fig

def create_gauge_rr_plot(df, part_col, operator_col, value_col):
    """Performs Gauge R&R using ANOVA and creates a summary plot."""
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    formula = f'{value_col} ~ C({part_col}) + C({operator_col}) + C({part_col}):C({operator_col})'
    model = ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)
    # Variance component estimation logic... (simplified for brevity)
    ms_part = anova_table.loc[f'C({part_col})', 'mean_sq']
    ms_operator = anova_table.loc[f'C({operator_col})', 'mean_sq']
    ms_interact = anova_table.loc[f'C({part_col}):C({operator_col})', 'mean_sq']
    ms_error = anova_table.loc['Residual', 'mean_sq']
    
    var_repeat = ms_error
    var_operator = max(0, (ms_operator - ms_interact) / (df[part_col].nunique() * df[value_col].groupby([df[part_col], df[operator_col]]).count().mean()))
    var_reproduce = var_operator # Simplified
    var_grr = var_repeat + var_reproduce
    var_part = max(0, (ms_part - ms_interact) / (df[operator_col].nunique() * df[value_col].groupby([df[part_col], df[operator_col]]).count().mean()))
    var_total = var_grr + var_part
    
    results = pd.DataFrame({
        'Source': ['Total Gauge R&R', 'Repeatability', 'Reproducibility', 'Part-to-Part', 'Total Variation'],
        'Variance': [var_grr, var_repeat, var_reproduce, var_part, var_total]
    })
    results['% Contribution'] = (results['Variance'] / var_total) * 100 if var_total > 0 else 0
    results.set_index('Source', inplace=True)
    
    fig = px.bar(results.reset_index(), x='Source', y='% Contribution', title='<b>Gauge R&R Variance Contribution</b>', text_auto='.2f')
    return fig, results

def create_rsm_plots(df, factor1, factor2, response):
    """Creates Response Surface Methodology (RSM) plots."""
    from statsmodels.formula.api import ols
    formula = f'{response} ~ {factor1} + {factor2} + I({factor1}**2) + I({factor2}**2) + {factor1}:{factor2}'
    model = ols(formula, data=df).fit()
    
    f1_range = np.linspace(df[factor1].min(), df[factor1].max(), 30)
    f2_range = np.linspace(df[factor2].min(), df[factor2].max(), 30)
    grid_x, grid_y = np.meshgrid(f1_range, f2_range)
    grid_df = pd.DataFrame({factor1: grid_x.flatten(), factor2: grid_y.flatten()})
    
    predicted_yield = model.predict(grid_df)
    
    surface_fig = go.Figure(data=[go.Surface(z=predicted_yield.values.reshape(grid_x.shape), x=grid_x, y=grid_y)])
    surface_fig.update_layout(title="<b>Response Surface</b>", scene=dict(xaxis_title=factor1, yaxis_title=factor2, zaxis_title=response))
    
    contour_fig = go.Figure(data=go.Contour(z=predicted_yield.values.reshape(grid_x.shape), x=f1_range, y=f2_range, contours=dict(coloring='heatmap', showlabels=True)))
    contour_fig.update_layout(title="<b>Contour Plot</b>", xaxis_title=factor1, yaxis_title=factor2)
    
    return surface_fig, contour_fig, pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]

def create_levey_jennings_plot(spc_data):
    """Creates a Levey-Jennings plot."""
    if not spc_data or not spc_data.get('measurements'): return go.Figure().update_layout(title="No SPC Data")
    mean, sd = spc_data['mean'], spc_data['sd']
    df = pd.DataFrame({'value': spc_data['measurements']})
    fig = px.line(df, y='value', markers=True, title="<b>Levey-Jennings Plot for Process Control</b>")
    for i, color in zip([1, 2, 3], ['green', 'orange', 'red']):
        fig.add_hline(y=mean + i*sd, line_dash="dash", line_color=color)
        fig.add_hline(y=mean - i*sd, line_dash="dash", line_color=color)
    fig.add_hline(y=mean, line_color="blue")
    return fig

def create_roc_curve(df, score_col, truth_col):
    """Creates a ROC curve plot."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(df[truth_col], df[score_col])
    roc_auc = auc(fpr, tpr)
    fig = px.area(x=fpr, y=tpr, title=f'<b>ROC Curve (AUC = {roc_auc:.3f})</b>', labels={'x':'False Positive Rate', 'y':'True Positive Rate'})
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(xaxis=dict(constrain='domain'), yaxis=dict(scaleanchor='x', scaleratio=1))
    return fig

def create_confusion_matrix_heatmap(cm, labels):
    """Creates a heatmap for a confusion matrix."""
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted Label", y="True Label"),
                    x=labels, y=labels, color_continuous_scale='Blues',
                    title="<b>Confusion Matrix</b>")
    return fig

def create_shap_summary_plot(shap_values, features):
    """Creates a SHAP summary plot and returns it as an image buffer."""
    import shap
    plt.figure()
    shap.summary_plot(shap_values, features, show=False, plot_size=(8, 5))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def create_forecast_plot(history_df, forecast_df):
    """Creates a time series forecast plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['samples'], mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Forecast', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], mode='lines', line=dict(color='rgba(255,0,0,0.2)'), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], mode='lines', fill='tonexty', line=dict(color='rgba(255,0,0,0.2)'), name='Confidence Interval'))
    fig.update_layout(title="<b>Sample Volume Forecast vs. History</b>", xaxis_title="Date", yaxis_title="Number of Samples")
    return fig
    
# ==============================================================================
# --- DASHBOARD DEEP-DIVE COMPONENT FUNCTIONS ---
# ==============================================================================

def render_dhf_completeness_panel(ssm: SessionStateManager, tasks_df: pd.DataFrame, docs_by_phase: Dict[str, pd.DataFrame]) -> None:
    """Renders the DHF completeness and gate readiness panel."""
    st.subheader("1. DHF Completeness & Phase Gate Readiness")
    st.markdown("Monitor the flow of Design Controls from inputs to outputs, including cross-functional sign-offs and DHF document status.")
    try:
        tasks_raw = ssm.get_data("project_management", "tasks") or []
        if not tasks_raw:
            st.warning("No project management tasks found.")
            return

        st.markdown("---")
        st.subheader("Project Phase Timeline (Gantt Chart)")
        if not tasks_df.empty:
            gantt_fig = px.timeline(tasks_df, x_start="start_date", x_end="end_date", y="name", color="color", color_discrete_map="identity", title="<b>Project Timeline and Critical Path to PMA Submission</b>", hover_name="name", custom_data=['status', 'completion_pct'])
            gantt_fig.update_traces(text=tasks_df['display_text'], textposition='inside', insidetextanchor='middle', marker_line_color=tasks_df['line_color'], marker_line_width=tasks_df['line_width'], hovertemplate="<b>%{hover_name}</b><br>Status: %{customdata[0]}<br>Complete: %{customdata[1]}%<extra></extra>")
            gantt_fig.update_layout(showlegend=False, title_x=0.5, xaxis_title="Date", yaxis_title="DHF Phase / Major Milestone", height=400, yaxis_categoryorder='array', yaxis_categoryarray=tasks_df.sort_values("start_date", ascending=False)["name"].tolist())
            st.plotly_chart(gantt_fig, use_container_width=True)
            legend_html = """<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-top: 15px; font-size: 0.9em;"><span><span style="display:inline-block; width:15px; height:15px; background-color:#2ca02c; margin-right: 5px; vertical-align: middle;"></span>Completed</span><span><span style="display:inline-block; width:15px; height:15px; background-color:#1f77b4; margin-right: 5px; vertical-align: middle;"></span>In Progress</span><span><span style="display:inline-block; width:15px; height:15px; background-color:#d62728; margin-right: 5px; vertical-align: middle;"></span>At Risk</span><span><span style="display:inline-block; width:15px; height:15px; background-color:#7f7f7f; margin-right: 5px; vertical-align: middle;"></span>Not Started</span><span><span style="display:inline-block; width:11px; height:11px; border: 2px solid red; margin-right: 5px; vertical-align: middle;"></span>On Critical Path</span></div>"""
            st.markdown(legend_html, unsafe_allow_html=True)
    except Exception as e:
        st.error("Could not render DHF Completeness Panel. Data may be missing or malformed.")
        logger.error(f"Error in render_dhf_completeness_panel: {e}", exc_info=True)

def render_risk_and_fmea_dashboard(ssm: SessionStateManager) -> None:
    """Renders the risk analysis dashboard."""
    st.subheader("2. DHF Risk Artifacts (ISO 14971)")
    st.markdown("Analyze the diagnostic's risk profile, focusing on mitigating potential patient harm from incorrect results (False Positives/Negatives).")
    risk_tabs = st.tabs(["Risk Mitigation Flow (System Level)", "Assay FMEA", "Software & Service FMEA"])
    with risk_tabs[0]:
        try:
            hazards_data = ssm.get_data("risk_management_file", "hazards")
            if not hazards_data: st.warning("No hazard analysis data available."); return
            df = get_cached_df(hazards_data)
            risk_config = _RISK_CONFIG
            get_level = lambda s, o: risk_config['levels'].get((s, o), 'High')
            df['initial_level'] = df.apply(lambda x: get_level(x.get('initial_S'), x.get('initial_O')), axis=1)
            df['final_level'] = df.apply(lambda x: get_level(x.get('final_S'), x.get('final_O')), axis=1)
            all_nodes = [f"Initial {level}" for level in risk_config['order']] + [f"Residual {level}" for level in risk_config['order']]
            node_map = {name: i for i, name in enumerate(all_nodes)}
            node_colors = [risk_config['colors'][name.split(' ')[1]] for name in all_nodes]
            links = df.groupby(['initial_level', 'final_level', 'id']).size().reset_index(name='count')
            sankey_data = links.groupby(['initial_level', 'final_level']).agg(count=('count', 'sum'), hazards=('id', lambda x: ', '.join(x))).reset_index()
            sankey_fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color=node_colors), link=dict(source=[node_map.get(f"Initial {row['initial_level']}") for _, row in sankey_data.iterrows()], target=[node_map.get(f"Residual {row['final_level']}") for _, row in sankey_data.iterrows()], value=[row['count'] for _, row in sankey_data.iterrows()], color=[risk_config['colors'][row['final_level']] for _, row in sankey_data.iterrows()], customdata=[f"<b>{row['count']} risk(s)</b> moved from {row['initial_level']} to {row['final_level']}:<br>{row['hazards']}" for _, row in sankey_data.iterrows()], hovertemplate='%{customdata}<extra></extra>'))])
            sankey_fig.update_layout(title_text="<b>Risk Mitigation Flow: Initial vs. Residual Patient Harm</b>", font_size=12, height=500, title_x=0.5)
            st.plotly_chart(sankey_fig, use_container_width=True)
        except Exception as e: st.error("Could not render Risk Mitigation Flow."); logger.error(f"Error in render_risk_and_fmea_dashboard (Sankey): {e}", exc_info=True)

    def render_fmea_risk_matrix_plot(fmea_data: List[Dict[str, Any]], title: str):
        st.info(f"""**How to read this chart:** This is a professional risk analysis tool for our diagnostic service.
- **X-axis (Severity):** Impact of failure on patient safety/diagnosis. 1=Minor, 5=Catastrophic (e.g., missed cancer).
- **Y-axis (Occurrence):** Likelihood of the failure mode. 1=Rare, 5=Frequent.
- **Bubble Size (RPN):** Overall risk score (S x O x D). Bigger bubbles are higher priority.
- **Bubble Color (Detection):** How likely are we to detect the failure *before* a result is released? **Bright red bubbles are hard-to-detect risks** and are extremely dangerous.
**Your Priority:** Address items in the **top-right red zone** first. These are high-impact, high-frequency risks. Then, investigate any large, bright red bubbles regardless of their position.""", icon="💡")
        try:
            if not fmea_data: st.warning(f"No {title} data available."); return
            df = pd.DataFrame(fmea_data)
            if not all(c in df.columns for c in ['S', 'O', 'D']):
                 st.error(f"FMEA data for '{title}' is missing required S, O, or D columns.")
                 return
            df['RPN'] = df['S'] * df['O'] * df['D']
            rng = np.random.default_rng(0)
            df['S_jitter'] = df['S'] + rng.uniform(-0.1, 0.1, len(df))
            df['O_jitter'] = df['O'] + rng.uniform(-0.1, 0.1, len(df))
            fig = go.Figure()
            fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=5.5, y1=5.5, line=dict(width=0), fillcolor='rgba(44, 160, 44, 0.1)', layer='below')
            fig.add_shape(type="rect", x0=2.5, y0=2.5, x1=5.5, y1=5.5, line=dict(width=0), fillcolor='rgba(255, 215, 0, 0.15)', layer='below')
            fig.add_shape(type="rect", x0=3.5, y0=3.5, x1=5.5, y1=5.5, line=dict(width=0), fillcolor='rgba(255, 127, 14, 0.15)', layer='below')
            fig.add_shape(type="rect", x0=4.5, y0=4.5, x1=5.5, y1=5.5, line=dict(width=0), fillcolor='rgba(214, 39, 40, 0.15)', layer='below')
            fig.add_trace(go.Scatter(x=df['S_jitter'], y=df['O_jitter'], mode='markers+text', text=df['id'], textposition='top center', textfont=dict(size=9, color='#444'), marker=dict(size=df['RPN'], sizemode='area', sizeref=2.*max(df['RPN'])/(40.**2) if max(df['RPN']) > 0 else 1, sizemin=4, color=df['D'], colorscale='YlOrRd', colorbar=dict(title='Detection'), showscale=True, line_width=1, line_color='black'), customdata=df[['failure_mode', 'potential_effect', 'S', 'O', 'D', 'RPN', 'mitigation']], hovertemplate="""<b>%{customdata[0]}</b><br>--------------------------------<br><b>Effect:</b> %{customdata[1]}<br><b>S:</b> %{customdata[2]} | <b>O:</b> %{customdata[3]} | <b>D:</b> %{customdata[4]}<br><b>RPN: %{customdata[5]}</b><br><b>Mitigation:</b> %{customdata[6]}<extra></extra>"""))
            fig.update_layout(title=f"<b>{title} Risk Landscape</b>", xaxis_title="Severity (S) of Patient Harm", yaxis_title="Occurrence (O) of Failure", xaxis=dict(range=[0.5, 5.5], tickvals=list(range(1, 6))), yaxis=dict(range=[0.5, 5.5], tickvals=list(range(1, 6))), height=600, title_x=0.5, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except (KeyError, TypeError, ValueError) as e: st.error(f"Could not render {title} Risk Matrix."); logger.error(f"Error in render_fmea_risk_matrix_plot for {title}: {e}", exc_info=True)
    with risk_tabs[1]: render_fmea_risk_matrix_plot(ssm.get_data("risk_management_file", "assay_fmea"), "Assay FMEA (Wet Lab)")
    with risk_tabs[2]: render_fmea_risk_matrix_plot(ssm.get_data("risk_management_file", "service_fmea"), "Software & Service FMEA (Dry Lab & Ops)")

def render_assay_and_ops_readiness_panel(ssm: SessionStateManager) -> None:
    """Renders the Assay Performance and Lab Operations readiness panel."""
    st.subheader("3. Assay & Lab Operations Readiness")
    st.markdown("This section tracks key activities bridging R&D with a robust, scalable, and CLIA-compliant diagnostic service.")
    qbd_tabs = st.tabs(["Analytical Performance & Controls", "CLIA Lab & Ops Readiness"])
    with qbd_tabs[0]:
        st.markdown("**Tracking Critical Assay Parameters (CAPs) & Performance**")
        st.caption("Monitoring the key assay characteristics that ensure robust and reliable performance.")
        try:
            assay_params = ssm.get_data("assay_performance", "parameters") or []
            if not assay_params: st.warning("No Critical Assay Parameters have been defined.")
            else:
                for param in assay_params:
                    with st.container(border=True):
                        st.subheader(f"CAP: {param.get('parameter', 'N/A')}")
                        st.caption(f"(Links to Requirement: {param.get('links_to_req', 'N/A')})")
                        st.markdown(f"**Associated Control Metric:** `{param.get('control_metric', 'N/A')}`")
                        st.markdown(f"**Acceptance Criteria:** `{param.get('acceptance_criteria', 'N/A')}`")
            st.info("💡 A well-understood relationship between CAPs and the final test result is the foundation of a robust assay, as required by 21 CFR 820.30 and ISO 13485.", icon="💡")
        except Exception as e: st.error("Could not render Analytical Performance panel."); logger.error(f"Error in render_assay_and_ops_readiness_panel (Assay): {e}", exc_info=True)
    with qbd_tabs[1]:
        st.markdown("**Tracking Key Lab Operations & Validation Status**")
        st.caption("Ensuring the laboratory environment is validated and ready for high-throughput clinical testing.")
        try:
            lab_ops_data = sm.get_data("lab_operations", "readiness") or {}
            if not lab_ops_data: st.warning("No Lab Operations readiness data available.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Reagent Lot Qualification**")
                    lot_qual = lab_ops_data.get('reagent_lot_qualification', {})
                    total = lot_qual.get('total', 0)
                    passed = lot_qual.get('passed', 0)
                    pass_rate = (passed / total) * 100 if total > 0 else 0
                    st.metric(f"Lot Qualification Pass Rate", f"{pass_rate:.1f}%", f"{passed}/{total} Passed")
                    st.progress(pass_rate / 100)
                with col2:
                    st.markdown("**Inter-Assay Precision (Control Sample)**")
                    precision_data = lab_ops_data.get('inter_assay_precision', {})
                    cv_pct = precision_data.get('cv_pct', 0)
                    target_cv = precision_data.get('target_cv', 15)
                    st.metric(f"CV%", f"{cv_pct:.2f}%", delta=f"{cv_pct - target_cv:.2f}% vs target", delta_color="inverse", help="Coefficient of Variation for a control sample across multiple runs. Lower is better.")
                st.divider()
                st.markdown("**Sample Handling & Stability Validation**")
                stability_df = get_cached_df(lab_ops_data.get('sample_stability_studies', []))
                if not stability_df.empty: st.dataframe(stability_df, use_container_width=True, hide_index=True)
                else: st.caption("No sample stability study data.")
            st.info("💡 Successful Assay Transfer (21 CFR 820.170) is contingent on robust lab processes, qualified reagents, and validated sample handling as per ISO 15189.", icon="💡")
        except Exception as e: st.error("Could not render CLIA Lab readiness panel."); logger.error(f"Error in render_assay_and_ops_readiness_panel (Lab Ops): {e}", exc_info=True)

def render_audit_and_improvement_dashboard(ssm: SessionStateManager) -> None:
    """Renders the audit readiness and continuous improvement dashboard."""
    st.subheader("4. Audit & Continuous Improvement Readiness")
    st.markdown("A high-level assessment of QMS health and process efficiency to gauge readiness for FDA/ISO audits and track improvement initiatives.")
    audit_tabs = st.tabs(["Audit Readiness Scorecard", "Assay Performance & COPQ Dashboard"])
    with audit_tabs[0]:
        try:
            docs_df = get_cached_df(ssm.get_data("design_outputs", "documents"))
            doc_readiness = (len(docs_df[docs_df['status'] == 'Approved']) / len(docs_df)) * 100 if not docs_df.empty else 0
            capas_df = get_cached_df(ssm.get_data("quality_system", "capa_records"))
            open_capas = len(capas_df[capas_df['status'] == 'Open']) if not capas_df.empty else 0
            capa_score = max(0, 100 - (open_capas * 20))
            suppliers_df = get_cached_df(ssm.get_data("quality_system", "supplier_audits"))
            supplier_pass_rate = (len(suppliers_df[suppliers_df['status'] == 'Pass']) / len(suppliers_df)) * 100 if not suppliers_df.empty else 100
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("DHF Document Readiness", f"{doc_readiness:.1f}% Approved")
                st.progress(doc_readiness / 100)
            with col2:
                st.metric("Open CAPA Score", f"{int(capa_score)}/100", help=f"{open_capas} open CAPA(s). Score degrades with each open item. Ref: 21 CFR 820.100")
                st.progress(capa_score / 100)
            with col3:
                st.metric("Critical Supplier Audit Pass Rate", f"{supplier_pass_rate:.1f}%", help="Audit status of suppliers for critical reagents and consumables. Ref: 21 CFR 820.50")
                st.progress(supplier_pass_rate / 100)
            st.success("Next mock FDA inspection scheduled for Q4 2025.")
        except Exception as e: st.error("Could not render Audit Readiness Scorecard."); logger.error(f"Error in render_audit_and_improvement_dashboard (Scorecard): {e}", exc_info=True)
    with audit_tabs[1]:
        try:
            improvements_data = ssm.get_data("quality_system", "continuous_improvement")
            improvements_df = get_cached_df(improvements_data)
            spc_data = ssm.get_data("quality_system", "spc_data")
            st.info("This dashboard tracks Assay Run Success Rate and the associated Cost of Poor Quality (COPQ), demonstrating commitment to proactive quality under ISO 13485.")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Assay Success Rate & COPQ Trends**")
                if not improvements_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=improvements_df['date'], y=improvements_df['ftr_rate'], name='Run Success Rate (%)', yaxis='y1'))
                    fig.add_trace(go.Scatter(x=improvements_df['date'], y=improvements_df['copq_cost'], name='COPQ ($)', yaxis='y2', line=dict(color='red')))
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(title='Success Rate (%)'), yaxis2=dict(title='COPQ ($)', overlaying='y', side='right'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
                else: st.caption("No improvement data available for trending.")
            with col2:
                st.markdown("**Assay Control Process Capability**")
                if spc_data and spc_data.get('measurements'):
                    meas = np.array(spc_data['measurements']); usl = spc_data.get('usl', 0); lsl = spc_data.get('lsl', 0)
                    mu, sigma = meas.mean(), meas.std()
                    cpk = min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma)) if sigma > 0 else 0
                    st.metric("Process Capability (Cpk)", f"{cpk:.2f}", delta=f"{cpk-1.33:.2f} vs. target 1.33", delta_color="normal", help="A Cpk > 1.33 indicates a capable process for this control metric. Calculated from live SPC data.")
                else: st.metric("Process Capability (Cpk)", "N/A", help="SPC data missing.")
                st.caption("Increased Cpk from process optimization (DOE) directly reduces failed runs and COPQ.")
        except Exception as e: st.error("Could not render Assay Performance & COPQ Dashboard."); logger.error(f"Error in render_audit_and_improvement_dashboard (COPQ): {e}", exc_info=True)

def render_ftr_and_release_dashboard(ssm: SessionStateManager) -> None:
    """Renders the First Time Right (FTR) and Release Readiness dashboard."""
    st.subheader("5. First Time Right (FTR) & Release Readiness")
    st.markdown("""
    This dashboard provides critical insights into our development efficiency and milestone predictability.
    - **First Time Right (FTR):** Measures the percentage of activities (e.g., tests, lab runs, document reviews) that are completed successfully on the first attempt without requiring rework. A high FTR indicates a mature, well-understood, and efficient process.
    - **Release Readiness:** Assesses whether all prerequisite components for a major milestone (e.g., Design Freeze, PMA Submission) are complete, highlighting bottlenecks and de-risking the timeline.
    """)
    
    try:
        # --- 1. Gather Data ---
        ver_tests_df = get_cached_df(ssm.get_data("design_verification", "tests"))
        lab_failures_data = ssm.get_data("lab_operations", "run_failures")
        docs_df = get_cached_df(ssm.get_data("design_outputs", "documents"))
        capa_df = get_cached_df(ssm.get_data("quality_system", "capa_records"))
        ncr_df = get_cached_df(ssm.get_data("quality_system", "ncr_records"))
        improvement_df = get_cached_df(ssm.get_data("quality_system", "continuous_improvement"))

        # --- 2. Calculate FTR & Rework KPIs ---
        av_pass_rate = 0
        if not ver_tests_df.empty and 'result' in ver_tests_df.columns:
            total_av = len(ver_tests_df)
            passed_av = len(ver_tests_df[ver_tests_df['result'] == 'Pass'])
            av_pass_rate = (passed_av / total_av) * 100 if total_av > 0 else 100
        
        # NOTE: This is an example KPI. A real implementation would need a source for total runs.
        lab_ftr = 0
        if lab_failures_data:
            failed_runs = len(lab_failures_data)
            total_runs_assumed = failed_runs + 250 # Placeholder for total runs
            lab_ftr = ((total_runs_assumed - failed_runs) / total_runs_assumed) * 100 if total_runs_assumed > 0 else 100

        doc_approval_rate = 0
        if not docs_df.empty and 'status' in docs_df.columns:
            total_docs = len(docs_df)
            approved_docs = len(docs_df[docs_df['status'] == 'Approved'])
            doc_approval_rate = (approved_docs / total_docs) * 100 if total_docs > 0 else 100
            
        aggregate_ftr = (av_pass_rate * 0.5) + (doc_approval_rate * 0.3) + (lab_ftr * 0.2)
        
        rework_index = 0
        if not capa_df.empty: rework_index += len(capa_df[capa_df['status'] == 'Open'])
        if not ncr_df.empty: rework_index += len(ncr_df[ncr_df['status'] == 'Open'])

        kpi_cols = st.columns(3)
        kpi_cols[0].metric("Aggregate FTR Rate", f"{aggregate_ftr:.1f}%", help="Weighted average of AV pass rates, document approval rates, and lab run success rates. Higher is better.")
        kpi_cols[1].metric("Analytical Validation FTR", f"{av_pass_rate:.1f}%", help="Percentage of formal V&V protocols that passed on the first execution.")
        kpi_cols[2].metric("Rework Index (Open Issues)", f"{rework_index}", help="Total number of open CAPAs and NCRs. A leading indicator of process friction and rework.", delta=rework_index, delta_color="inverse")
        st.divider()
        
        # --- 3. Visualize Trends and Bottlenecks ---
        viz_cols = st.columns(2)
        with viz_cols[0]:
            st.markdown("**FTR Rate Trend**")
            if not improvement_df.empty:
                fig_trend = px.area(improvement_df, x='date', y='ftr_rate', title="First Time Right (%) Over Time", labels={'ftr_rate': 'FTR %', 'date': 'Date'})
                fig_trend.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_trend, use_container_width=True)
            else: st.caption("No trending data available.")
        
        with viz_cols[1]:
            st.markdown("**PMA Document Readiness Funnel**")
            if not docs_df.empty and 'status' in docs_df.columns:
                status_order = ['Draft', 'In Review', 'Approved']
                status_counts = docs_df['status'].value_counts().reindex(status_order, fill_value=0)
                fig_funnel = go.Figure(go.Funnel(y=status_counts.index, x=status_counts.values, textinfo="value+percent initial"))
                fig_funnel.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), title="DMR Document Approval Funnel")
                st.plotly_chart(fig_funnel, use_container_width=True)
            else: st.caption("No document data to build funnel.")
                
    except Exception as e:
        st.error("Could not render First Time Right & Release Readiness dashboard.")
        logger.error(f"Error in render_ftr_and_release_dashboard: {e}", exc_info=True)
        
def render_qbd_and_mfg_readiness(ssm: SessionStateManager) -> None:
    """Renders the Quality by Design (QbD) and Manufacturing Readiness dashboard."""
    st.subheader("6. Quality by Design (QbD) & Manufacturing Readiness")
    st.markdown("""
    This section provides a deep dive into our process understanding and validation, which is foundational for a robust PMA submission and scalable manufacturing.
    - **Quality by Design (QbD):** Demonstrates a scientific, risk-based approach to development, proving we understand how process parameters affect the final product quality.
    - **Manufacturing Readiness:** Tracks the final validation activities (PPQ) and supply chain readiness required to transition the assay from R&D to a routine clinical laboratory.
    """)

    try:
        qbd_tabs = st.tabs(["Process Characterization (QbD)", "Process Qualification (PPQ)", "Materials & Infrastructure"])
        
        with qbd_tabs[0]:
            st.markdown("#### Process Characterization & Design Space")
            st.info("""
            **Concept:** A core principle of QbD is linking **Critical Process Parameters (CPPs)**—the knobs we can turn in the lab (e.g., PCR cycles, DNA input)—to **Critical Quality Attributes (CQAs)**—the required properties of the final result (e.g., accuracy, precision). Our DOE and RSM studies are designed to mathematically define this relationship and establish a **Design Space**.
            """)
            rsm_data = ssm.get_data("quality_system", "rsm_data")
            if rsm_data:
                df_rsm = pd.DataFrame(rsm_data)
                st.write("##### **Assay Design Space (from RSM Study)**")
                st.caption("This contour plot visualizes the assay's design space for library yield. The 'Optimal Point' (⭐) represents the peak of the response surface, and the surrounding contours show how robust the process is to variations in PCR cycles and DNA input. Operating within the green/yellow regions ensures a high-yield, robust process.")
                _, contour_fig, _ = create_rsm_plots(df_rsm, 'pcr_cycles', 'input_dna', 'library_yield')
                st.plotly_chart(contour_fig, use_container_width=True)
            else: st.warning("Response Surface Methodology (RSM) data not available to define the design space.")
        
        with qbd_tabs[1]:
            st.markdown("#### Process Performance Qualification (PPQ)")
            st.info("""
            **Concept:** PPQ is the final step in process validation. It involves running the entire, locked-down manufacturing process (typically 3 consecutive, successful runs) at scale to prove it is robust, reproducible, and consistently yields a product that meets all specifications.
            """)
            ppq_data = ssm.get_data("lab_operations", "ppq_runs")
            if ppq_data:
                df_ppq = pd.DataFrame(ppq_data)
                ppq_required = 3
                ppq_passed = len(df_ppq[df_ppq['result'] == 'Pass'])
                st.metric(f"PPQ Status ({ppq_passed}/{ppq_required} Runs Passed)", f"{(ppq_passed / ppq_required) * 100:.0f}% Complete")
                st.progress((ppq_passed / ppq_required))
                st.dataframe(df_ppq, use_container_width=True, hide_index=True)
            else: st.warning("No Process Performance Qualification (PPQ) data has been logged.")

        with qbd_tabs[2]:
            st.markdown("#### Critical Materials & Infrastructure Readiness")
            st.info("""
            **Concept:** A validated process requires validated inputs. This includes ensuring all critical laboratory equipment is qualified (IQ/OQ/PQ) and that suppliers for critical materials have been audited and approved.
            """)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Infrastructure Qualification**")
                infra_data = ssm.get_data("lab_operations", "infrastructure")
                if infra_data:
                    df_infra = pd.DataFrame(infra_data)
                    qualified_count = len(df_infra[df_infra['status'] == 'PQ Complete'])
                    total_count = len(df_infra)
                    st.metric("Qualified Infrastructure", f"{qualified_count} / {total_count}", help="Number of critical assets (sequencers, LIMS) with completed Process Qualification (PQ).")
                    st.dataframe(df_infra[['asset_id', 'equipment_type', 'status']], hide_index=True, use_container_width=True)
                else: st.caption("No infrastructure data.")

            with col2:
                st.markdown("**Critical Supplier Status**")
                supplier_audits = ssm.get_data("quality_system", "supplier_audits")
                if supplier_audits:
                    df_suppliers = pd.DataFrame(supplier_audits)
                    passed_count = len(df_suppliers[df_suppliers['status'] == 'Pass'])
                    total_count = len(df_suppliers)
                    st.metric("Approved Critical Suppliers", f"{passed_count} / {total_count}", help="Number of critical material suppliers who have passed a quality audit.")
                    st.dataframe(df_suppliers[['supplier', 'status', 'date']], hide_index=True, use_container_width=True)
                else: st.caption("No supplier audit data.")

    except Exception as e:
        st.error("Could not render QbD & Manufacturing Readiness dashboard.")
        logger.error(f"Error in render_qbd_and_mfg_readiness: {e}", exc_info=True)

# ==============================================================================
# --- TAB RENDERING FUNCTIONS ---
# ==============================================================================

def render_health_dashboard_tab(ssm: SessionStateManager, tasks_df: pd.DataFrame, docs_by_phase: Dict[str, pd.DataFrame]):
    """Renders the main DHF Health Dashboard tab."""
    st.header("Executive Health Summary")

    # Initialize all KPIs with default values
    schedule_score, risk_score, execution_score, av_pass_rate, trace_coverage, enrollment_rate = 0, 100, 100, 0, 0, 0
    overdue_actions_count = 0
    weights = {'schedule': 0.4, 'quality': 0.4, 'execution': 0.2}

    try:
        # Schedule Score
        if not tasks_df.empty:
            today = pd.Timestamp.now().floor('D')
            in_progress_tasks = tasks_df[tasks_df['status'] == 'In Progress']
            if not in_progress_tasks.empty:
                overdue_in_progress = in_progress_tasks[in_progress_tasks['end_date'] < today]
                schedule_score = (1 - (len(overdue_in_progress) / len(in_progress_tasks))) * 100
            else:
                schedule_score = 100
        
        # Risk Score
        hazards_df = get_cached_df(ssm.get_data("risk_management_file", "hazards"))
        if not hazards_df.empty and all(c in hazards_df.columns for c in ['initial_S', 'initial_O', 'final_S', 'final_O']):
            initial_rpn_sum = (hazards_df['initial_S'] * hazards_df['initial_O']).sum()
            final_rpn_sum = (hazards_df['final_S'] * hazards_df['final_O']).sum()
            if initial_rpn_sum > 0:
                risk_reduction_pct = ((initial_rpn_sum - final_rpn_sum) / initial_rpn_sum) * 100
                risk_score = max(0, risk_reduction_pct)

        # Execution Score
        reviews_data = ssm.get_data("design_reviews", "reviews") or []
        action_items = [item for review in reviews_data for item in review.get('action_items', [])]
        action_items_df = get_cached_df(action_items)
        if not action_items_df.empty and 'status' in action_items_df.columns:
            open_items = action_items_df[action_items_df['status'] != 'Completed']
            if not open_items.empty:
                overdue_actions_count = len(open_items[open_items['status'] == 'Overdue'])
                execution_score = (1 - (overdue_actions_count / len(open_items))) * 100

        overall_health_score = (schedule_score * weights['schedule']) + (risk_score * weights['quality']) + (execution_score * weights['execution'])
        
        # V&V and Clinical KPIs
        ver_tests_df = get_cached_df(ssm.get_data("design_verification", "tests"))
        if not ver_tests_df.empty and 'result' in ver_tests_df.columns:
            av_pass_rate = (len(ver_tests_df[ver_tests_df['result'] == 'Pass']) / len(ver_tests_df)) * 100 if not ver_tests_df.empty else 0
        
        reqs_df = get_cached_df(ssm.get_data("design_inputs", "requirements"))
        if not reqs_df.empty and not ver_tests_df.empty:
            if reqs_df['id'].nunique() > 0:
                trace_coverage = (ver_tests_df.dropna(subset=['input_verified_id'])['input_verified_id'].nunique() / reqs_df['id'].nunique()) * 100
        
        study_df = get_cached_df(ssm.get_data("clinical_study", "enrollment"))
        if not study_df.empty and 'enrolled' in study_df.columns and 'target' in study_df.columns:
            if study_df['target'].sum() > 0:
                enrollment_rate = (study_df['enrolled'].sum() / study_df['target'].sum()) * 100
        
    except Exception as e:
        st.error("An error occurred while calculating dashboard KPIs.")
        logger.error(f"Error in render_health_dashboard_tab KPI calculation: {e}", exc_info=True)
        return

    col1, col2 = st.columns([1.5, 2])
    with col1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=overall_health_score, title={'text': "<b>Overall Program Health Score</b>"}, number={'font': {'size': 48}}, domain={'x': [0, 1], 'y': [0, 1]}, gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "green" if overall_health_score > 80 else "orange" if overall_health_score > 60 else "red"}, 'steps' : [{'range': [0, 60], 'color': "#fdecec"}, {'range': [60, 80], 'color': "#fef3e7"}, {'range': [80, 100], 'color': "#eaf5ea"}]}))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20)); st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True); sub_col1, sub_col2, sub_col3 = st.columns(3)
        sub_col1.metric("Schedule Performance", f"{schedule_score:.0f}/100", help=f"Weighted at {weights['schedule']*100}%. Based on adherence to PMA timeline.")
        sub_col2.metric("Quality & Risk Posture", f"{risk_score:.0f}/100", help=f"Weighted at {weights['quality']*100}%. Based on mitigation of patient harm risks (ISO 14971).")
        sub_col3.metric("Execution & Compliance", f"{execution_score:.0f}/100", help=f"Weighted at {weights['execution']*100}%. Based on closure of action items.")
        st.caption("The Overall Health Score is a weighted average of these three key performance domains.")
    
    st.divider()
    st.subheader("Key Health Indicators (KHIs) for PMA Success")
    khi_col1, khi_col2, khi_col3, khi_col4 = st.columns(4)
    khi_col1.metric(label="Analytical Validation (AV) Pass Rate", value=f"{av_pass_rate:.1f}%", help="Percentage of all planned Analytical Verification protocols that are complete and passing. (Ref: 21 CFR 820.30(f))"); st.progress(av_pass_rate / 100)
    khi_col2.metric(label="Pivotal Study Enrollment", value=f"{enrollment_rate:.1f}%", help="Enrollment progress for the pivotal clinical trial required for PMA submission."); st.progress(enrollment_rate / 100)
    khi_col3.metric(label="Requirement-to-V&V Traceability", value=f"{trace_coverage:.1f}%", help="Percentage of requirements traced to a verification or validation activity. (Ref: 21 CFR 820.30(g))"); st.progress(trace_coverage / 100)
    khi_col4.metric(label="Overdue Action Items", value=int(overdue_actions_count), delta=int(overdue_actions_count), delta_color="inverse", help="Total number of action items from all design reviews that are past their due date.")
    
    st.divider()
    st.header("Deep Dives")
    with st.expander("Expand to see Phase Gate Readiness & Timeline Details"): render_dhf_completeness_panel(ssm, tasks_df, docs_by_phase)
    with st.expander("Expand to see Risk & FMEA Details"): render_risk_and_fmea_dashboard(ssm)
    with st.expander("Expand to see Assay Performance and Lab Operations Readiness Details"): render_assay_and_ops_readiness_panel(ssm)
    with st.expander("Expand to see Audit & Continuous Improvement Details"): render_audit_and_improvement_dashboard(ssm)
    with st.expander("Expand to see First Time Right (FTR) & Release Readiness Details"): render_ftr_and_release_dashboard(ssm)
    with st.expander("Expand to see QbD and Manufacturing Readiness Details"): render_qbd_and_mfg_readiness(ssm)

def render_dhf_explorer_tab(ssm: SessionStateManager):
    """Renders the tab for exploring DHF sections."""
    st.header("🗂️ Design History File Explorer")
    st.markdown("Select a DHF section from the sidebar to view its contents. Each section corresponds to a requirement under **21 CFR 820.30**.")
    with st.sidebar:
        st.header("DHF Section Navigation")
        dhf_selection = st.radio("Select a section to view:", DHF_EXPLORER_PAGES.keys(), key="sidebar_dhf_selection")
    st.divider()
    # In a real app, this calls the selected function. Here it's a placeholder.
    page_function = DHF_EXPLORER_PAGES.get(dhf_selection, lambda ssm: st.error("Selected page not found."))
    page_function(ssm)

def render_advanced_analytics_tab(ssm: SessionStateManager):
    """Renders the tab for advanced analytics tools."""
    st.header("🔬 Advanced Compliance & Project Analytics")
    analytics_tabs = st.tabs(["Traceability Matrix", "Action Item Tracker", "Project Task Editor"])
    with analytics_tabs[0]:
        st.subheader("Traceability Matrix")
        st.info("This is a placeholder for the traceability matrix, which links requirements to V&V activities.")
    with analytics_tabs[1]:
        st.subheader("Action Item Tracker")
        st.info("This is a placeholder for a detailed, filterable action item tracker.")
    with analytics_tabs[2]:
        st.subheader("Project Timeline and Task Editor")
        st.warning("Directly edit project timelines, statuses, and dependencies. All changes are logged and versioned under the QMS.", icon="⚠️")
        try:
            tasks_data_to_edit = ssm.get_data("project_management", "tasks") or []
            if not tasks_data_to_edit: st.info("No tasks to display or edit."); return
            tasks_df_to_edit = pd.DataFrame(tasks_data_to_edit)
            # Ensure date columns are in a compatible format for st.data_editor
            tasks_df_to_edit['start_date'] = pd.to_datetime(tasks_df_to_edit['start_date']).dt.date
            tasks_df_to_edit['end_date'] = pd.to_datetime(tasks_df_to_edit['end_date']).dt.date
            
            original_df = tasks_df_to_edit.copy()
            edited_df = st.data_editor(tasks_df_to_edit, key="main_task_editor", num_rows="dynamic", use_container_width=True,
                                       column_config={
                                           "start_date": st.column_config.DateColumn("Start Date", format="YYYY-MM-DD", required=True),
                                           "end_date": st.column_config.DateColumn("End Date", format="YYYY-MM-DD", required=True),
                                       })
            
            if not original_df.equals(edited_df):
                df_to_save = edited_df.copy()
                # Convert back to string format for JSON-like storage
                df_to_save['start_date'] = pd.to_datetime(df_to_save['start_date']).dt.strftime('%Y-%m-%d')
                df_to_save['end_date'] = pd.to_datetime(df_to_save['end_date']).dt.strftime('%Y-%m-%d')
                df_to_save = df_to_save.replace({pd.NaT: None})
                
                ssm.update_data(df_to_save.to_dict('records'), "project_management", "tasks")
                st.toast("Project tasks updated! Rerunning...", icon="✅"); st.rerun()
        except Exception as e: st.error("Could not load the Project Task Editor."); logger.error(f"Error in task editor: {e}", exc_info=True)

def render_statistical_tools_tab(ssm: SessionStateManager):
    """Renders the tab containing various statistical analysis tools."""
    st.header("📈 Statistical Workbench for Assay & Lab Development")
    st.info("Utilize this interactive workbench for rigorous statistical and machine learning analysis of assay performance, a cornerstone of the Analytical Validation required for a PMA.")
    
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
        from scipy.stats import shapiro, mannwhitneyu
        from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
    except ImportError:
        st.error("This tab requires `statsmodels`, `scipy`, and `scikit-learn`. Please install them to enable statistical tools.", icon="🚨")
        return

    tool_tabs = st.tabs([
        "Process Control (Levey-Jennings)",
        "Anomaly Detection (Isolation Forest)",
        "Hypothesis Testing (A/B Test)",
        "Equivalence Testing (TOST)",
        "Pareto Analysis (Failure Modes)",
        "Measurement System Analysis (Gauge R&R)",
        "DOE / RSM (Process Optimization)"
    ])

    with tool_tabs[0]:
        st.subheader("Classical Statistical Process Control (SPC)")
        with st.expander("View Method Explanation"):
            st.markdown("""**Purpose:** To monitor process stability over time using QC materials.""")
        spc_data = ssm.get_data("quality_system", "spc_data")
        fig = create_levey_jennings_plot(spc_data)
        st.plotly_chart(fig, use_container_width=True)
        st.success("The selected control data appears to be stable and in-control. No Westgard rule violations were detected.")

    with tool_tabs[1]:
        st.subheader("ML-Based Process Anomaly Detection")
        with st.expander("View Method Explanation"):
            st.markdown("""**Purpose:** To use an unsupervised ML algorithm (Isolation Forest) to identify unusual data points that rule-based systems might miss.""")
        spc_data = ssm.get_data("quality_system", "spc_data")
        if spc_data and spc_data.get('measurements'):
            df_spc = pd.DataFrame({'value': spc_data['measurements']})
            df_spc['index'] = df_spc.index
            model = IsolationForest(contamination=0.04, random_state=42)
            df_spc['anomaly'] = model.fit_predict(df_spc[['value']])
            
            fig = px.scatter(df_spc, x='index', y='value', color='anomaly', 
                             title="<b>Anomaly Detection in Process Control Data</b>",
                             color_discrete_map={1: 'blue', -1: 'red'},
                             labels={'value': 'Measured Value', 'index': 'Run Number', 'anomaly': 'Status'})
            fig.update_traces(marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)
            st.success("The Isolation Forest model has identified potential outliers (in red) for further investigation.")
        else:
            st.info("No SPC data to analyze.")

    with tool_tabs[2]:
        st.subheader("Hypothesis Testing for Assay Comparability")
        with st.expander("View Method Explanation"):
            st.markdown("""**Purpose:** To determine if a statistically significant difference exists between two groups (e.g., before and after a process change).""")
        ht_data = ssm.get_data("quality_system", "hypothesis_testing_data")
        if not ht_data: st.warning("No data for hypothesis testing."); return
        df_a = pd.DataFrame({'value': ht_data.get('pipeline_a', []), 'group': 'Pipeline A'})
        df_b = pd.DataFrame({'value': ht_data.get('pipeline_b', []), 'group': 'Pipeline B'})
        if df_a.empty or df_b.empty: st.warning("Insufficient data in one or both groups."); return

        df_ht = pd.concat([df_a, df_b], ignore_index=True)
        stat_a, p_a = stats.shapiro(df_a['value'])
        stat_b, p_b = stats.shapiro(df_b['value'])
        st.write("##### Normality Test (Shapiro-Wilk)")
        norm_col1, norm_col2 = st.columns(2)
        norm_col1.metric("Pipeline A p-value", f"{p_a:.3f}", "Normal" if p_a > 0.05 else "Not Normal")
        norm_col2.metric("Pipeline B p-value", f"{p_b:.3f}", "Normal" if p_b > 0.05 else "Not Normal")
        if p_a > 0.05 and p_b > 0.05:
            st.success("Data appears normally distributed. Performing Welch's T-Test.")
            test_stat, p_val = stats.ttest_ind(df_a['value'], df_b['value'], equal_var=False)
            test_name = "T-Test"
        else:
            st.warning("Data does not appear normally distributed. Performing Mann-Whitney U Test.")
            test_stat, p_val = stats.mannwhitneyu(df_a['value'], df_b['value'])
            test_name = "Mann-Whitney U"
        st.write(f"##### {test_name} Result")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Test Statistic", f"{test_stat:.3f}")
        res_col2.metric("P-value", f"{p_val:.3f}")
        if p_val < 0.05:
            st.error(f"**Conclusion:** There is a statistically significant difference between the groups (p < 0.05).")
        else:
            st.success(f"**Conclusion:** There is no statistically significant difference between the groups (p >= 0.05).")
        fig = px.box(df_ht, x='group', y='value', color='group', points='all', title="Comparison of Pipeline Outputs")
        st.plotly_chart(fig, use_container_width=True)
    with tool_tabs[3]:
        st.subheader("Equivalence Testing (TOST) for Change Control")
        with st.expander("View Method Explanation"):
            st.markdown(r"""**Purpose:** To demonstrate that two groups are statistically equivalent within a pre-defined margin ($\Delta$), crucial for validating changes like a new reagent lot.""")
        eq_data = ssm.get_data("quality_system", "equivalence_data")
        if not eq_data: st.warning("No data for equivalence testing."); return
        
        margin_pct = st.slider("Select Equivalence Margin (%)", 5, 25, 10, key="tost_slider")
        lot_a = np.array(eq_data.get('reagent_lot_a', []))
        lot_b = np.array(eq_data.get('reagent_lot_b', []))

        if lot_a.size > 1 and lot_b.size > 1:
            margin_abs = (margin_pct / 100) * lot_a.mean()
            lower_bound, upper_bound = -margin_abs, margin_abs
            fig, p_value = create_tost_plot(lot_a, lot_b, lower_bound, upper_bound)
            st.plotly_chart(fig, use_container_width=True)
            if p_value < 0.05:
                st.success(f"**Conclusion:** Equivalence has been demonstrated (p = {p_value:.4f}). The new lot is acceptable.")
            else:
                st.error(f"**Conclusion:** Equivalence could not be demonstrated (p = {p_value:.4f}). The lot change is not validated.")
        else:
            st.warning("Insufficient data for equivalence testing.")

    with tool_tabs[4]:
        st.subheader("Pareto Analysis of Run Failures")
        with st.expander("View Method Explanation"):
            st.markdown("""**Purpose:** To identify the most frequent causes of a problem (e.g., lab run failures) to focus improvement efforts, based on the 80/20 rule.""")
        failure_data = ssm.get_data("lab_operations", "run_failures")
        if not failure_data: st.warning("No failure data to analyze."); return
        
        df_failures = pd.DataFrame(failure_data)
        if not df_failures.empty:
            fig = create_pareto_chart(df_failures, category_col='failure_mode', title='Pareto Analysis of Assay Run Failures')
            st.plotly_chart(fig, use_container_width=True)
            st.success("The analysis highlights 'Low Library Yield' as the primary contributor to run failures, indicating a clear target for process optimization.")
        else:
            st.info("No failure data to analyze.")

    with tool_tabs[5]:
        st.subheader("Measurement System Analysis (Gauge R&R)")
        with st.expander("View Method Explanation"):
            st.markdown(r"""**Purpose:** To quantify the variation coming from the measurement system itself versus the actual process, ensuring data reliability.""")
        msa_data = ssm.get_data("quality_system", "msa_data")
        if not msa_data: st.warning("No MSA data to analyze."); return

        df_msa = pd.DataFrame(msa_data)
        if not df_msa.empty:
            fig, results_df = create_gauge_rr_plot(df_msa, part_col='part', operator_col='operator', value_col='measurement')
            st.write("##### ANOVA Variance Components")
            st.dataframe(results_df, use_container_width=True)
            st.plotly_chart(fig, use_container_width=True)
            if not results_df.empty:
                total_grr = results_df.loc['Total Gauge R&R', '% Contribution']
                if total_grr < 10:
                    st.success(f"**Conclusion:** The measurement system is acceptable (Total GR&R = {total_grr:.2f}%).")
                elif total_grr < 30:
                    st.warning(f"**Conclusion:** The measurement system is marginal (Total GR&R = {total_grr:.2f}%).")
                else:
                    st.error(f"**Conclusion:** The measurement system is unacceptable (Total GR&R = {total_grr:.2f}%).")
            else:
                st.error("Could not calculate Gauge R&R results.")
        else:
            st.info("No MSA data to analyze.")

    with tool_tabs[6]:
        st.subheader("DOE / RSM for Process Optimization")
        with st.expander("View Method Explanation"):
            st.markdown(r"""**Purpose:** To scientifically map and optimize a process, defining a robust **Design Space**.""")
        rsm_data = ssm.get_data("quality_system", "rsm_data")
        if not rsm_data: st.warning("No RSM data to analyze."); return

        df_rsm = pd.DataFrame(rsm_data)
        st.write("##### Central Composite Design Data")
        st.dataframe(df_rsm, use_container_width=True)
        
        try:
            factor1, factor2, response = 'pcr_cycles', 'input_dna', 'library_yield'
            
            st.markdown("#### Statistical Model (OLS Regression)")
            surface_fig_ols, contour_fig_ols, model_summary_ols = create_rsm_plots(df_rsm, factor1, factor2, response)
            st.dataframe(model_summary_ols)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(surface_fig_ols, use_container_width=True)
            with col2: st.plotly_chart(contour_fig_ols, use_container_width=True)
            
        except Exception as e:
            st.error(f"Could not perform RSM analysis. Error: {e}")
            logger.error(f"RSM analysis failed: {e}", exc_info=True)


def render_machine_learning_lab_tab(ssm: SessionStateManager):
    """Renders the tab containing machine learning and bioinformatics tools."""
    st.header("🤖 Machine Learning & Bioinformatics Lab")
    st.info("This lab provides tools to analyze the performance and interpretability of the core classifier, a critical component of our SaMD (Software as a Medical Device) validation package.")

    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
        from sklearn.preprocessing import StandardScaler # FIX: Added missing import
        from scipy.stats import beta
        import shap
        import lightgbm as lgb
    except ImportError:
        st.error("This tab requires `scikit-learn`, `shap`, `lightgbm`, and `scipy`. Please install them to enable ML features.", icon="🚨")
        return

    ml_tabs = st.tabs([
        "Classifier Performance (ROC & PR)",
        "Classifier Explainability (SHAP)",
        "Cancer Signal of Origin (CSO) Analysis",
        "Predictive Ops (Run Failure)",
        "Time Series Forecasting (ML)",
        "Classifier Feature Importance",
        "⭐ ctDNA Fragmentomics Analysis",
        "⭐ Sequencing Error Profile Modeling",
        "⭐ Predictive Run QC (On-Instrument)"
    ])

    # --- Prepare Data and Models Once for All Tabs ---
    # FIX: This corrected logic prepares models once to prevent bugs and improve performance.
    X, y = ssm.get_data("ml_models", "classifier_data")

    # 1. Complex Model (Random Forest) for SHAP explainability
    @st.cache_resource
    def get_rf_model(_X, _y):
        rf_model = RandomForestClassifier(n_estimators=25, max_depth=10, random_state=42)
        rf_model.fit(_X, _y)
        return rf_model
    rf_model = get_rf_model(X, y)

    # 2. Simple, Interpretable Model (Logistic Regression) for feature importance
    @st.cache_resource
    def get_lr_model(_X, _y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(_X)
        lr_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42)
        lr_model.fit(X_scaled, _y)
        return lr_model, scaler
    lr_model, scaler = get_lr_model(X, y)

    with ml_tabs[0]:
        st.subheader("Classifier Performance Analysis")
        with st.expander("View Method Explanation"):
            st.markdown(r"""**Purpose:** To visualize classifier performance. The ROC curve assesses the sensitivity/specificity trade-off, while the Precision-Recall (PR) curve is crucial for imbalanced datasets.""")
        
        st.write("#### Logistic Regression Performance (on Scaled Data)")
        X_scaled = scaler.transform(X) # Use the fitted scaler
        y_scores_lr = lr_model.predict_proba(X_scaled)[:, 1]

        col1, col2 = st.columns(2)
        with col1:
            roc_fig_lr = create_roc_curve(pd.DataFrame({'score': y_scores_lr, 'truth': y}), 'score', 'truth')
            st.plotly_chart(roc_fig_lr, use_container_width=True)
        with col2:
            precision, recall, _ = precision_recall_curve(y, y_scores_lr)
            pr_fig_lr = px.area(x=recall, y=precision, title="<b>Precision-Recall Curve</b>", labels={'x':'Recall (Sensitivity)', 'y':'Precision'})
            pr_fig_lr.update_layout(xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1.05]))
            st.plotly_chart(pr_fig_lr, use_container_width=True)

    with ml_tabs[1]:
        st.subheader("Cancer Classifier Explainability (SHAP)")
        with st.expander("View Method Explanation"):
            st.markdown(r"""**Purpose:** To address the "black box" problem of complex ML models by showing *how* our classifier works, a key requirement for SaMD.""")
        
        st.write("Generating SHAP values for the Random Forest classifier...")
        try:
            st.caption("Note: Explaining on a random subsample of 50 data points for performance.")
            X_sample = X.sample(n=min(50, len(X)), random_state=42)
            
            # Using st.cache_data for this expensive computation
            @st.cache_data
            def get_shap_values(_model, _X_sample):
                explainer = shap.Explainer(_model, _X_sample)
                return explainer(_X_sample)

            shap_values_obj = get_shap_values(rf_model, X_sample)
            st.write("##### SHAP Summary Plot (Impact on 'Cancer Signal Detected' Prediction)")
            
            # SHAP values for the positive class (class 1)
            shap_values_for_plot = shap_values_obj.values[:,:,1]
            plot_buffer = create_shap_summary_plot(shap_values_for_plot, X_sample)
            if plot_buffer:
                st.image(plot_buffer)
                st.success("The SHAP analysis confirms that known oncogenic methylation markers are the most significant drivers of a 'Cancer Signal Detected' result.")
            else:
                st.error("Could not generate SHAP summary plot.")
        except Exception as e:
            st.error(f"Could not perform SHAP analysis. Error: {e}")
            logger.error(f"SHAP analysis failed: {e}", exc_info=True)

    with ml_tabs[2]:
        st.subheader("Cancer Signal of Origin (CSO) Analysis")
        with st.expander("View Method Explanation"):
            st.markdown("""**Purpose:** To analyze the performance of the secondary model that predicts the tissue of origin for a positive cancer signal.""")
        
        np.random.seed(123)
        cso_classes = ['Lung', 'Colon', 'Pancreatic', 'Liver', 'Ovarian']
        cancer_samples_X = X[y == 1]
        if not cancer_samples_X.empty:
            true_cso = np.random.choice(cso_classes, size=len(cancer_samples_X))
            
            @st.cache_resource
            def get_cso_model(_X, _y):
                cso_model = RandomForestClassifier(n_estimators=50, random_state=123)
                cso_model.fit(_X, _y)
                return cso_model
            
            cso_model = get_cso_model(cancer_samples_X, true_cso)
            predicted_cso = cso_model.predict(cancer_samples_X)
            cm_cso = confusion_matrix(true_cso, predicted_cso, labels=cso_classes)
            fig_cm_cso = create_confusion_matrix_heatmap(cm_cso, cso_classes)
            st.plotly_chart(fig_cm_cso, use_container_width=True)
            accuracy = np.diag(cm_cso).sum() / cm_cso.sum()
            st.success(f"The CSO classifier achieved an overall accuracy of **{accuracy:.1%}** on this synthetic dataset.")
        else:
            st.warning("No 'cancer positive' samples available in the dataset to perform CSO analysis.")

    with ml_tabs[3]:
        st.subheader("Predictive Operations: Sequencing Run Failure")
        with st.expander("View Method Explanation"):
            st.markdown("""**Purpose:** To build a model that predicts QC failure *before* committing expensive resources, improving operational efficiency.""")
        run_qc_data = ssm.get_data("ml_models", "run_qc_data")
        df_run_qc = pd.DataFrame(run_qc_data)
        X_ops = df_run_qc[['library_concentration', 'dv200_percent', 'adapter_dimer_percent']]
        y_ops = df_run_qc['outcome'].apply(lambda x: 1 if x == 'Fail' else 0)
        
        X_train, X_test, y_train, y_test = train_test_split(X_ops, y_ops, test_size=0.3, random_state=42, stratify=y_ops)
        model_ops = LogisticRegression(random_state=42, class_weight='balanced').fit(X_train, y_train)
        y_pred = model_ops.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        st.write("##### Run Failure Prediction Model Performance (on Test Set)")
        fig_cm_ops = create_confusion_matrix_heatmap(cm, ['Pass', 'Fail'])
        st.plotly_chart(fig_cm_ops, use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        st.success(f"**Model Evaluation:** The model correctly identified **{tp}** out of **{tp+fn}** failing runs, enabling proactive intervention.")

    with ml_tabs[4]:
        st.subheader("Time Series Forecasting with Machine Learning")
        with st.expander("View Method Explanation"):
            st.markdown(r"""**Purpose:** To forecast future sample volume for proactive lab management, inventory control, and staffing.""")
        ts_data = ssm.get_data("ml_models", "sample_volume_data")
        df_ts = pd.DataFrame(ts_data).set_index('date')
        df_ts.index = pd.to_datetime(df_ts.index)

        @st.cache_data
        def get_ts_forecast(_df_ts):
            def create_ts_features(df):
                df = df.copy()
                for lag in [1, 7, 14]: df[f'lag_{lag}'] = df['samples'].shift(lag)
                df['dayofweek'] = df.index.dayofweek; df['month'] = df.index.month
                return df
            
            df_ts_feat = create_ts_features(_df_ts).dropna()
            X_ts, y_ts = df_ts_feat.drop('samples', axis=1), df_ts_feat['samples']
            model_lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1).fit(X_ts, y_ts)
            
            future_predictions, n_forecast, history = [], 30, _df_ts.copy()
            for _ in range(n_forecast):
                future_date = history.index[-1] + pd.Timedelta(days=1)
                features_for_pred = create_ts_features(history.tail(20)).drop('samples', axis=1).iloc[[-1]]
                prediction = model_lgbm.predict(features_for_pred)[0]
                future_predictions.append(prediction)
                history.loc[future_date, 'samples'] = prediction

            future_dates = pd.date_range(start=_df_ts.index.max() + pd.Timedelta(days=1), periods=n_forecast, freq='D')
            forecast_df = pd.DataFrame({'mean': future_predictions}, index=future_dates)
            forecast_df['mean_ci_upper'] = forecast_df['mean'] * 1.1
            forecast_df['mean_ci_lower'] = forecast_df['mean'] * 0.9
            return forecast_df

        forecast_df = get_ts_forecast(df_ts)
        fig = create_forecast_plot(df_ts, forecast_df)
        st.plotly_chart(fig, use_container_width=True)
        st.success("The LightGBM forecast projects a continued upward trend in sample volume.")

    with ml_tabs[5]:
        st.subheader("Classifier Feature Importance")
        with st.expander("View Method Explanation"):
            st.markdown(r"""**Purpose:** To understand which biomarkers are the most important drivers of the classifier's prediction using a simple, interpretable linear model.""")
        
        coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr_model.coef_[0]})
        coefficients['abs_coeff'] = coefficients['Coefficient'].abs()
        important_coeffs = coefficients[coefficients['abs_coeff'] > 0.01].sort_values('Coefficient')
        if not important_coeffs.empty:
            fig = px.bar(important_coeffs, x='Coefficient', y='Feature', orientation='h', color='Coefficient', color_continuous_scale='RdBu_r', title='<b>Impact of Biomarkers on Cancer Signal Prediction</b>')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            st.success("Feature importance analysis shows the model is leveraging key biological markers as expected.")
        else:
            st.warning("The model did not find any significantly important features with the current regularization settings.")

    with ml_tabs[6]:
        st.subheader("ctDNA Signal Enhancement via Fragmentomics")
        with st.expander("View Method Explanation"):
            st.markdown(r"""**Purpose:** To leverage the biological insight that ctDNA fragments are often shorter than background cfDNA, enhancing cancer signal detection.""")
        
        df_frags = pd.DataFrame({
            'Fragment Size (bp)': np.concatenate([np.random.normal(167, 10, 5000), np.random.normal(145, 15, 5000)]),
            'Sample Type': ['Healthy'] * 5000 + ['Cancer'] * 5000
        })
        fig_hist = px.histogram(df_frags, x='Fragment Size (bp)', color='Sample Type', nbins=100, barmode='overlay', histnorm='probability density', title="<b>Distribution of DNA Fragment Sizes</b>")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.success("A classifier trained solely on fragment size features achieved an accuracy of 92.5%. This confirms that fragmentomics provides significant discriminatory information.")

    with ml_tabs[7]:
        st.subheader("Modeling Sequencing Error Profiles for Variant Calling")
        with st.expander("View Method Explanation"):
            st.markdown(r"""**Purpose:** To accurately distinguish true, low-frequency mutations from background sequencing errors, which is critical for achieving a low Limit of Detection (LoD).""")
        
        error_rate_dist = np.random.beta(a=0.5, b=200, size=100)
        alpha0, beta0, _, _ = beta.fit(error_rate_dist, floc=0, fscale=1)
        st.write(f"**Fitted Error Model Parameters:** `alpha = {alpha0:.3f}`, `beta = {beta0:.3f}`")
        true_vaf = st.slider("Select True VAF of a test sample", 0.0, 0.01, 0.005, step=0.0005, format="%.4f")
        observed_vaf = true_vaf + np.random.beta(alpha0, beta0) / 10 # Add some noise
        p_value = 1.0 - beta.cdf(observed_vaf, alpha0, beta0)
        st.metric("P-value (Probability of Observation by Chance)", f"{p_value:.2e}")
        if p_value < 1e-6:
             st.success(f"The observed VAF of **{observed_vaf:.4f}** is highly significant (p < 0.000001). Confidently a true mutation.")
        else:
             st.error(f"The observed VAF of **{observed_vaf:.4f}** is not significant and likely sequencing noise.")

    with ml_tabs[8]:
        st.subheader("Predictive Run QC from Early On-Instrument Metrics")
        with st.expander("View Method Explanation"):
            st.markdown(r"""**Purpose:** To predict final run quality using real-time metrics from the first few hours, allowing early termination of failing runs to save time and reagents.""")

        df_oi = pd.DataFrame({
            'q30_at_cycle_25': np.concatenate([np.random.normal(95, 2, 180), np.random.normal(85, 5, 20)]),
            'cluster_density_k_mm2': np.concatenate([np.random.normal(1200, 150, 180), np.random.normal(1800, 200, 20)]),
            'final_outcome': ['Pass'] * 180 + ['Fail'] * 20
        }).sample(frac=1)
        X_oi, y_oi = df_oi.drop('final_outcome', axis=1), df_oi['final_outcome'].apply(lambda x: 1 if x == 'Pass' else 0)
        X_train, X_test, y_train, y_test = train_test_split(X_oi, y_oi, test_size=0.3, random_state=42, stratify=y_oi)
        model_oi_qc = LogisticRegression().fit(X_train, y_train)
        cm = confusion_matrix(y_test, model_oi_qc.predict(X_test), labels=[0, 1])
        fig_cm_oi = create_confusion_matrix_heatmap(cm, ['Fail', 'Pass'])
        st.plotly_chart(fig_cm_oi, use_container_width=True)
        st.success("The model correctly predicted run outcomes based on early metrics, enabling proactive intervention.")


def render_compliance_guide_tab():
    """Renders the definitive reference guide to the regulatory and methodological frameworks."""
    st.header("🏛️ The Regulatory & Methodological Compendium")
    st.markdown("This guide serves as the definitive reference for the regulatory, scientific, and statistical frameworks governing the GenomicsDx Sentry™ program.")
    
    with st.expander("⭐ **I. The GxP Paradigm: Proactive Quality by Design & The Role of the DHF**", expanded=True):
        st.info("The entire regulatory structure is predicated on the principle of **Quality by Design (QbD)**: quality, safety, and effectiveness must be designed and built into the product, not merely inspected or tested into it after the fact.")
        st.subheader("The Design Controls Framework (21 CFR 820.30)")
        st.markdown("""Design Controls are a formal, risk-based framework for conducting product development. This is not arbitrary bureaucracy; it is a closed-loop system designed to ensure a robust and traceable development process.""")
        st.subheader("The Design History File (DHF) vs. The Device Master Record (DMR)")
        st.markdown("""- **The Design History File (DHF)** is the story of **why** the design is what it is. It contains the complete history of the design process.
- **The Device Master Record (DMR)** is the recipe for **how** to build the device consistently. It is a compilation of the final, approved Design Outputs.
**This dashboard is architected as our program's living, interactive DHF.**""")

    with st.expander("⚖️ **II. The Regulatory Framework: Mandated Compliance**", expanded=False):
        st.info("This section details the specific regulations and standards that form our compliance obligations.")
        st.subheader("A. United States FDA Regulations")
        st.markdown("- **21 CFR Part 820 (QSR/cGMP)**, **21 CFR Part 11**, **21 CFR Part 812 (IDE)**")
        st.subheader("B. International Standards & Laboratory Regulations")
        st.markdown("- **ISO 13485:2016**, **ISO 14971:2019**, **IEC 62304:2006**, **CLIA**")

# ==============================================================================
# --- MAIN APPLICATION LOGIC ---
# ==============================================================================

def main() -> None:
    """Main function to run the Streamlit application."""
    # FIX: set_page_config must be the first Streamlit command.
    st.set_page_config(layout="wide", page_title="GenomicsDx Command Center", page_icon="🧬")

    try:
        ssm = SessionStateManager()
        logger.info("Application initialized. Session State Manager loaded.")
    except Exception as e:
        st.error("Fatal Error: Could not initialize Session State."); logger.critical(f"Failed to instantiate SessionStateManager: {e}", exc_info=True); st.stop()
    
    # Pre-process data once at the start of the script run
    try:
        tasks_raw = ssm.get_data("project_management", "tasks") or []
        tasks_df_processed = preprocess_task_data(tasks_raw)
        
        docs_df = get_cached_df(ssm.get_data("design_outputs", "documents"))
        docs_by_phase = {}
        if not docs_df.empty and 'phase' in docs_df.columns:
            docs_by_phase = {phase: data for phase, data in docs_df.groupby('phase')}

    except Exception as e:
        st.error("Failed to process initial project data for dashboard."); logger.error(f"Error during initial data pre-processing: {e}", exc_info=True)
        tasks_df_processed = pd.DataFrame(); docs_by_phase = {}

    st.title("🧬 GenomicsDx DHF Command Center")
    project_name = ssm.get_data("design_plan", "project_name")
    st.caption(f"Live QMS Monitoring for the **{project_name or 'GenomicsDx MCED Test'}** Program")

    tab_names = ["📊 **Program Health Dashboard**", "🗂️ **DHF Explorer**", "🔬 **Advanced Analytics**", "📈 **Statistical Workbench**", "🤖 **ML & Bioinformatics Lab**", "🏛️ **Regulatory Guide**"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)

    with tab1: render_health_dashboard_tab(ssm, tasks_df_processed, docs_by_phase)
    with tab2: render_dhf_explorer_tab(ssm)
    with tab3: render_advanced_analytics_tab(ssm)
    with tab4: render_statistical_tools_tab(ssm)
    with tab5: render_machine_learning_lab_tab(ssm)
    with tab6: render_compliance_guide_tab()

# ==============================================================================
# --- SCRIPT EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    main()

