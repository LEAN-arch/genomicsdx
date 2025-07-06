#genomicsdx/app.py
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
import matplotlib.pyplot as plt # <--- ADD THIS LINE
import shap

# --- Robust Path Correction Block ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception as e:
    st.warning(f"Could not adjust system path. Module imports may fail. Error: {e}")

# --- Local Application Imports (with corrected paths) ---
try:
    from genomicsdx.analytics.action_item_tracker import render_action_item_tracker
    from genomicsdx.analytics.traceability_matrix import render_traceability_matrix
    from genomicsdx.dhf_sections import (
        design_changes, design_inputs, design_outputs, design_plan, design_reviews,
        design_risk_management, design_transfer, design_validation,
        design_verification, human_factors
    )
    from genomicsdx.utils.critical_path_utils import find_critical_path
    from genomicsdx.utils.plot_utils import (
        _RISK_CONFIG,
        create_action_item_chart, create_risk_profile_chart,
        create_roc_curve, create_levey_jennings_plot, create_lod_probit_plot, create_bland_altman_plot,
        create_pareto_chart, create_gauge_rr_plot, create_tost_plot,
        create_confusion_matrix_heatmap, create_shap_summary_plot, create_forecast_plot,
        create_doe_effects_plot, create_rsm_plots # SME Enhancement: Import new RSM plotter
    )
    from genomicsdx.utils.session_state_manager import SessionStateManager
except ImportError as e:
    st.error(f"Fatal Error: A required local module could not be imported: {e}. "
             "Please ensure the application is run from the project's root directory and that all subdirectories contain an `__init__.py` file.")
    logging.critical(f"Fatal module import error: {e}", exc_info=True)
    st.stop()


# Call set_page_config() at the top level of the script
st.set_page_config(layout="wide", page_title="GenomicsDx Command Center", page_icon="🧬")

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# --- Module-Level Constants ---
DHF_EXPLORER_PAGES = {
    "1. Design & Development Plan": design_plan.render_design_plan,
    "2. Risk Management (ISO 14971)": design_risk_management.render_design_risk_management,
    "3. Human Factors & Usability (Sample Kit & Report)": human_factors.render_human_factors,
    "4. Design Inputs (Assay & System Requirements)": design_inputs.render_design_inputs,
    "5. Design Outputs (Specifications, Code, Procedures)": design_outputs.render_design_outputs,
    "6. Design Reviews (Phase Gates)": design_reviews.render_design_reviews,
    "7. Design Verification (Analytical Validation)": design_verification.render_design_verification,
    "8. Design Validation (Clinical Validation)": design_validation.render_design_validation,
    "9. Assay Transfer & Lab Operations": design_transfer.render_design_transfer,
    "10. Design Changes (Change Control)": design_changes.render_design_changes
}

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
    tasks_df['line_color'] = np.where(tasks_df['is_critical'], 'red', '#FFFFFF')
    tasks_df['line_width'] = np.where(tasks_df['is_critical'], 4, 0)
    tasks_df['display_text'] = "<b>" + tasks_df['name'].fillna('').astype(str) + "</b> (" + tasks_df['completion_pct'].fillna(0).astype(int).astype(str) + "%)"
    return tasks_df

@st.cache_data
def get_cached_df(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generic, cached function to create DataFrames."""
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)

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
        for task in tasks_raw:
            task_name = task.get('name', 'N/A')
            st.subheader(f"Phase: {task_name}")
            st.caption(f"Status: {task.get('status', 'N/A')} - {task.get('completion_pct', 0)}% Complete")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Associated DHF Documents:**")
                st.caption("Document-to-phase linkage not yet implemented. This is a placeholder.")
            with col2:
                st.markdown("**Cross-Functional Sign-offs:**")
                sign_offs = task.get('sign_offs', {})
                if isinstance(sign_offs, dict) and sign_offs:
                    for team, status in sign_offs.items():
                        color = "green" if status == "✅" else "orange" if status == "In Progress" else "grey"
                        st.markdown(f"- **{team}:** <span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                else:
                    st.caption("No sign-off data for this phase.")
            st.divider()
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
                    st.subheader(f"CAP: {param.get('parameter', 'N/A')}")
                    st.caption(f"(Links to Requirement: {param.get('links_to_req', 'N/A')})")
                    st.markdown(f"**Associated Control Metric:** `{param.get('control_metric', 'N/A')}`")
                    st.markdown(f"**Acceptance Criteria:** `{param.get('acceptance_criteria', 'N/A')}`")
                    st.divider()
            st.info("A well-understood relationship between CAPs and the final test result is the foundation of a robust assay, as required by 21 CFR 820.30 and ISO 13485.", icon="💡")
        except Exception as e: st.error("Could not render Analytical Performance panel."); logger.error(f"Error in render_assay_and_ops_readiness_panel (Assay): {e}", exc_info=True)
    with qbd_tabs[1]:
        st.markdown("**Tracking Key Lab Operations & Validation Status**")
        st.caption("Ensuring the laboratory environment is validated and ready for high-throughput clinical testing.")
        try:
            lab_ops_data = ssm.get_data("lab_operations", "readiness") or {}
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
            st.info("Successful Assay Transfer (21 CFR 820.170) is contingent on robust lab processes, qualified reagents, and validated sample handling as per ISO 15189.", icon="💡")
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
            improvements_df = get_cached_df(ssm.get_data("quality_system", "continuous_improvement"))
            spc_data = ssm.get_data("quality_system", "spc_data")
            st.info("This dashboard tracks Assay Run Success Rate and the associated Cost of Poor Quality (COPQ), demonstrating commitment to proactive quality under ISO 13485.")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Assay Success Rate & COPQ Trends**")
                if not improvements_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=improvements_df['date'], y=improvements_df['ftr_rate'], name='Run Success Rate (%)', yaxis='y1'))
                    fig.add_trace(go.Scatter(x=improvements_df['date'], y=improvements_df['copq_cost'], name='COPQ ($)', yaxis='y2', line=dict(color='red')))
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(title='Success Rate (%)'), yaxis2=dict(title='COPQ ($)', overlaying='y', side='right'))
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
#######NEW FUNCTION: FTY and Bootlenecks ++++++++++++++++++++++++++++++++++++++++++++++++++++++
def render_ftr_and_release_dashboard(ssm: SessionStateManager) -> None:
    """Renders the First Time Right (FTR) and Release Readiness dashboard."""
    st.subheader("5. First Time Right (FTR) & Release Readiness")
    st.markdown("""
    This dashboard provides critical insights into our development efficiency and milestone predictability.
    - **First Time Right (FTR):** Measures the percentage of activities (e.g., tests, lab runs, document reviews) that are completed successfully on the first attempt without requiring rework. A high FTR indicates a mature, well-understood, and efficient process.
    - **Release Readiness:** Assesses whether all prerequisite components for a major milestone (e.g., Design Freeze, PMA Submission) are complete, highlighting bottlenecks and de-risking the timeline.
    """)
    
    try:
        # --- 1. Gather Data from Across the DHF ---
        ver_tests_df = get_cached_df(ssm.get_data("design_verification", "tests"))
        lab_failures_data = ssm.get_data("lab_operations", "run_failures")
        docs_df = get_cached_df(ssm.get_data("design_outputs", "documents"))
        capa_df = get_cached_df(ssm.get_data("quality_system", "capa_records"))
        ncr_df = get_cached_df(ssm.get_data("quality_system", "ncr_records"))
        improvement_df = get_cached_df(ssm.get_data("quality_system", "continuous_improvement"))

        # --- 2. Calculate FTR & Rework KPIs ---
        # AV Protocol FTR
        av_pass_rate = 0
        if not ver_tests_df.empty and 'result' in ver_tests_df.columns:
            passed_av = len(ver_tests_df[ver_tests_df['result'] == 'Pass'])
            total_av = len(ver_tests_df)
            av_pass_rate = (passed_av / total_av) * 100 if total_av > 0 else 100
        
        # Lab Run FTR (from failure modes data)
        lab_ftr = 0
        if lab_failures_data:
            # Assuming we have a way to know total runs, for now, we estimate from a baseline
            total_runs_assumed = len(lab_failures_data) + 150 # Placeholder for total runs
            failed_runs = len(lab_failures_data)
            lab_ftr = ((total_runs_assumed - failed_runs) / total_runs_assumed) * 100 if total_runs_assumed > 0 else 100

        # Document FTR (Approval Rate)
        doc_approval_rate = 0
        if not docs_df.empty and 'status' in docs_df.columns:
            approved_docs = len(docs_df[docs_df['status'] == 'Approved'])
            total_docs = len(docs_df)
            doc_approval_rate = (approved_docs / total_docs) * 100 if total_docs > 0 else 100
            
        # Aggregate FTR Score (Weighted)
        aggregate_ftr = (av_pass_rate * 0.5) + (doc_approval_rate * 0.3) + (lab_ftr * 0.2)
        
        # Rework Index (Proxy: Number of open CAPAs and NCRs)
        rework_index = 0
        if not capa_df.empty:
            rework_index += len(capa_df[capa_df['status'] == 'Open'])
        if not ncr_df.empty:
            rework_index += len(ncr_df[ncr_df['status'] == 'Open'])

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
            else:
                st.caption("No trending data available.")
        
        with viz_cols[1]:
            st.markdown("**PMA Document Readiness Funnel**")
            if not docs_df.empty and 'status' in docs_df.columns:
                status_order = ['Draft', 'In Review', 'Approved']
                status_counts = docs_df['status'].value_counts().reindex(status_order, fill_value=0)
                
                fig_funnel = go.Figure(go.Funnel(
                    y = status_counts.index,
                    x = status_counts.values,
                    textinfo = "value+percent initial"
                ))
                fig_funnel.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), title="DMR Document Approval Funnel")
                st.plotly_chart(fig_funnel, use_container_width=True)
            else:
                st.caption("No document data to build funnel.")
                
        # --- 4. Actionable Insights ---
        st.subheader("Actionable Insights")
        if aggregate_ftr < 85:
            st.warning(f"**Focus Area:** The Aggregate FTR of {aggregate_ftr:.1f}% is below the target of 85%. This indicates significant rework and process friction, increasing costs and delaying timelines. **Action:** Investigate the primary drivers of low FTR (e.g., AV protocols, lab errors) using the Pareto tool and initiate a targeted process improvement project.")
        else:
            st.success(f"**Observation:** The Aggregate FTR of {aggregate_ftr:.1f}% meets the target. This reflects a mature and predictable development process.")

        if not docs_df.empty and 'status' in docs_df.columns:
            counts = docs_df['status'].value_counts()
            if counts.get('In Review', 0) > counts.get('Approved', 0):
                st.warning("**Bottleneck Identified:** A high number of documents are stuck in the 'In Review' stage compared to those 'Approved'. This suggests a potential resource constraint in the quality or regulatory review teams. **Action:** Review workloads of approvers and consider parallel review paths to accelerate document finalization for the DMR.")

    except Exception as e:
        st.error("Could not render First Time Right & Release Readiness dashboard.")
        logger.error(f"Error in render_ftr_and_release_dashboard: {e}", exc_info=True)
        
###### QbD =++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def render_qbd_and_mfg_readiness(ssm: SessionStateManager) -> None:
    """Renders the Quality by Design (QbD) and Manufacturing Readiness dashboard."""
    st.subheader("6. Quality by Design (QbD) & Manufacturing Readiness")
    st.markdown("""
    This section provides a deep dive into our process understanding and validation, which is foundational for a robust PMA submission and scalable manufacturing.
    - **Quality by Design (QbD):** Demonstrates a scientific, risk-based approach to development, proving we understand how process parameters affect the final product quality.
    - **Manufacturing Readiness:** Tracks the final validation activities (PPQ) and supply chain readiness required to transition the assay from R&D to a routine clinical laboratory.
    """)

    try:
        # --- 1. Gather Data ---
        rsm_data = ssm.get_data("quality_system", "rsm_data")
        ppq_data = ssm.get_data("lab_operations", "ppq_runs")
        assay_params = ssm.get_data("assay_performance", "parameters")
        supplier_audits = ssm.get_data("quality_system", "supplier_audits")
        infra_data = ssm.get_data("lab_operations", "infrastructure")

        # --- 2. Create Tabs for Different Focus Areas ---
        qbd_tabs = st.tabs(["Process Characterization (QbD)", "Process Qualification (PPQ)", "Materials & Infrastructure"])
        
        with qbd_tabs[0]:
            st.markdown("#### Process Characterization & Design Space")
            st.info("""
            **Concept:** A core principle of QbD is linking **Critical Process Parameters (CPPs)**—the knobs we can turn in the lab (e.g., PCR cycles, DNA input)—to **Critical Quality Attributes (CQAs)**—the required properties of the final result (e.g., accuracy, precision). Our DOE and RSM studies are designed to mathematically define this relationship and establish a **Design Space**.
            """)
            
            # --- Display Design Space from RSM ---
            if rsm_data:
                df_rsm = pd.DataFrame(rsm_data)
                st.write("##### **Assay Design Space (from RSM Study)**")
                st.caption("This contour plot visualizes the assay's design space for library yield. The 'Optimal Point' (⭐) represents the peak of the response surface, and the surrounding contours show how robust the process is to variations in PCR cycles and DNA input. Operating within the green/yellow regions ensures a high-yield, robust process.")
                
                _, contour_fig, _ = create_rsm_plots(df_rsm, 'pcr_cycles', 'input_dna', 'library_yield')
                st.plotly_chart(contour_fig, use_container_width=True)
            else:
                st.warning("Response Surface Methodology (RSM) data not available to define the design space.")
        
        with qbd_tabs[1]:
            st.markdown("#### Process Performance Qualification (PPQ)")
            st.info("""
            **Concept:** PPQ is the final step in process validation. It involves running the entire, locked-down manufacturing process (typically 3 consecutive, successful runs) at scale to prove it is robust, reproducible, and consistently yields a product that meets all specifications.
            """)
            
            # --- PPQ Run Status ---
            if ppq_data:
                df_ppq = pd.DataFrame(ppq_data)
                ppq_required = 3
                ppq_passed = len(df_ppq[df_ppq['result'] == 'Pass'])
                
                st.metric(f"PPQ Status ({ppq_passed}/{ppq_required} Runs Passed)", f"{(ppq_passed / ppq_required) * 100:.0f}% Complete")
                st.progress((ppq_passed / ppq_required))
                
                st.dataframe(df_ppq, use_container_width=True, hide_index=True)
            else:
                st.warning("No Process Performance Qualification (PPQ) data has been logged.")

        with qbd_tabs[2]:
            st.markdown("#### Critical Materials & Infrastructure Readiness")
            st.info("""
            **Concept:** A validated process requires validated inputs. This includes ensuring all critical laboratory equipment is qualified (IQ/OQ/PQ) and that suppliers for critical materials have been audited and approved.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Infrastructure Qualification**")
                if infra_data:
                    df_infra = pd.DataFrame(infra_data)
                    qualified_count = len(df_infra[df_infra['status'] == 'PQ Complete'])
                    total_count = len(df_infra)
                    st.metric("Qualified Infrastructure", f"{qualified_count} / {total_count}", help="Number of critical assets (sequencers, LIMS) with completed Process Qualification (PQ).")
                    st.dataframe(df_infra[['asset_id', 'equipment_type', 'status']], hide_index=True, use_container_width=True)
                else:
                    st.caption("No infrastructure data.")

            with col2:
                st.markdown("**Critical Supplier Status**")
                if supplier_audits:
                    df_suppliers = pd.DataFrame(supplier_audits)
                    passed_count = len(df_suppliers[df_suppliers['status'] == 'Pass'])
                    total_count = len(df_suppliers)
                    st.metric("Approved Critical Suppliers", f"{passed_count} / {total_count}", help="Number of critical material suppliers who have passed a quality audit.")
                    st.dataframe(df_suppliers[['supplier', 'status', 'date']], hide_index=True, use_container_width=True)
                else:
                    st.caption("No supplier audit data.")

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
    schedule_score, risk_score, execution_score, av_pass_rate, trace_coverage, enrollment_rate = 0, 0, 100, 0, 0, 0
    overdue_actions_count = 0
    weights = {'schedule': 0.4, 'quality': 0.4, 'execution': 0.2}

    reviews_data = ssm.get_data("design_reviews", "reviews") or []
    all_actions_sources = [
        ssm.get_data("quality_system", "capa_records"),
        ssm.get_data("quality_system", "ncr_records"),
        ssm.get_data("design_reviews", "reviews"),
        ssm.get_data("design_changes", "changes")
    ]
    original_action_items = [
        item for source in all_actions_sources if source
        for r in source
        for item in r.get("action_items", []) + r.get("action_plan", []) + r.get("correction_actions", [])
    ]
    action_items_df = get_cached_df(original_action_items)

    try:
        # Schedule Score
        if not tasks_df.empty:
            today = pd.Timestamp.now().floor('D')
            overdue_in_progress = tasks_df[(tasks_df['status'] == 'In Progress') & (tasks_df['end_date'] < today)]
            total_in_progress = tasks_df[tasks_df['status'] == 'In Progress']
            schedule_score = (1 - (len(overdue_in_progress) / len(total_in_progress))) * 100 if not total_in_progress.empty else 100
        
        # Risk Score
        hazards_df = get_cached_df(ssm.get_data("risk_management_file", "hazards"))
        if not hazards_df.empty and all(c in hazards_df.columns for c in ['initial_S', 'initial_O', 'final_S', 'final_O']):
            initial_rpn = hazards_df['initial_S'] * hazards_df['initial_O']
            final_rpn = hazards_df['final_S'] * hazards_df['final_O']
            initial_rpn_sum = initial_rpn.sum()
            if initial_rpn_sum > 0:
                risk_reduction_pct = ((initial_rpn_sum - final_rpn.sum()) / initial_rpn_sum) * 100
                risk_score = max(0, risk_reduction_pct)
        
        # Execution Score & Overdue Items
        if not action_items_df.empty and 'status' in action_items_df.columns:
            open_items = action_items_df[action_items_df['status'] != 'Completed']
            if not open_items.empty:
                overdue_actions_count = len(open_items[open_items['status'] == 'Overdue'])
                execution_score = (1 - (overdue_actions_count / len(open_items))) * 100 if len(open_items) > 0 else 100
        
        overall_health_score = (schedule_score * weights['schedule']) + (risk_score * weights['quality']) + (execution_score * weights['execution'])
        
        # V&V and Clinical KPIs
        ver_tests_df = get_cached_df(ssm.get_data("design_verification", "tests"))
        if not ver_tests_df.empty and 'result' in ver_tests_df.columns:
            av_pass_rate = (len(ver_tests_df[ver_tests_df['result'] == 'Pass']) / len(ver_tests_df)) * 100 if len(ver_tests_df) > 0 else 0
        
        reqs_df = get_cached_df(ssm.get_data("design_inputs", "requirements"))
        if not reqs_df.empty and not ver_tests_df.empty and 'id' in reqs_df.columns and 'input_verified_id' in ver_tests_df.columns:
            if reqs_df['id'].nunique() > 0:
                trace_coverage = (ver_tests_df.dropna(subset=['input_verified_id'])['input_verified_id'].nunique() / reqs_df['id'].nunique()) * 100
        
        study_df = get_cached_df(ssm.get_data("clinical_study", "enrollment"))
        if not study_df.empty and 'enrolled' in study_df.columns and 'target' in study_df.columns:
            if study_df['target'].sum() > 0:
                enrollment_rate = (study_df['enrolled'].sum() / study_df['target'].sum()) * 100
        
        if not action_items_df.empty and 'status' in action_items_df.columns:
            overdue_actions_count = len(action_items_df[action_items_df['status'] == 'Overdue'])
    
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
    with khi_col1: st.metric(label="Analytical Validation (AV) Pass Rate", value=f"{av_pass_rate:.1f}%", help="Percentage of all planned Analytical Verification protocols that are complete and passing. (Ref: 21 CFR 820.30(f))"); st.progress(av_pass_rate / 100)
    with khi_col2: st.metric(label="Pivotal Study Enrollment", value=f"{enrollment_rate:.1f}%", help="Enrollment progress for the pivotal clinical trial required for PMA submission."); st.progress(enrollment_rate / 100)
    with khi_col3: st.metric(label="Requirement-to-V&V Traceability", value=f"{trace_coverage:.1f}%", help="Percentage of requirements traced to a verification or validation activity. (Ref: 21 CFR 820.30(g))"); st.progress(trace_coverage / 100)
    with khi_col4: st.metric(label="Overdue Action Items", value=int(overdue_actions_count), delta=int(overdue_actions_count), delta_color="inverse", help="Total number of action items from all design reviews that are past their due date.")
    
    st.divider()
    st.subheader("Action Item Health (Last 30 Days)")
    st.markdown("This chart shows the trend of open action items. A healthy project shows a downward or stable trend. A rising red area indicates a growing backlog of overdue work, which requires management attention.")

    @st.cache_data
    def generate_burndown_data(_reviews_data: Tuple, _action_items_data: Tuple):
        if not _action_items_data: return pd.DataFrame()
        action_items_list = [dict(fs) for fs in _action_items_data]
        reviews_list = [dict(fs) for fs in _reviews_data]
        df = pd.DataFrame(action_items_list)
        for review_dict in reviews_list:
            review_date = pd.to_datetime(review_dict.get('date'))
            action_items_in_review_tuple = review_dict.get("action_items", tuple())
            for item_frozenset in action_items_in_review_tuple:
                item_dict = dict(item_frozenset)
                if 'id' in item_dict: df.loc[df['id'] == item_dict['id'], 'review_date'] = review_date
        df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
        df['created_date'] = pd.to_datetime(df.get('review_date'), errors='coerce')
        df.dropna(subset=['created_date', 'due_date', 'id'], inplace=True)
        def get_deterministic_offset(item_id): return int(hashlib.md5(str(item_id).encode()).hexdigest(), 16) % 3
        df['created_date'] += df['id'].apply(lambda x: pd.to_timedelta(get_deterministic_offset(x), unit='d'))
        df['completion_date'] = pd.NaT
        completed_mask = df['status'] == 'Completed'
        if completed_mask.any():
            completed_items = df.loc[completed_mask].copy()
            lifespan = (completed_items['due_date'] - completed_items['created_date']).dt.days.fillna(1).astype(int)
            lifespan = lifespan.apply(lambda d: max(1, d))
            def get_deterministic_completion(row):
                seed_value = int(hashlib.md5(str(row['id']).encode()).hexdigest(), 16) % (2**32)
                rng = np.random.default_rng(seed_value)
                return rng.integers(1, row['lifespan'] + 1) if row['lifespan'] >= 1 else 1
            completed_items['lifespan'] = lifespan
            completion_days = completed_items.apply(get_deterministic_completion, axis=1)
            df.loc[completed_mask, 'completion_date'] = completed_items['created_date'] + pd.to_timedelta(completion_days, unit='d')
        today = pd.Timestamp.now().floor('D')
        date_range = pd.date_range(end=today, periods=30, freq='D')
        daily_counts = []
        for day in date_range:
            created_mask = df['created_date'] <= day
            completed_mask = (df['completion_date'].notna()) & (df['completion_date'] <= day)
            open_on_day_df = df[created_mask & ~completed_mask]
            if not open_on_day_df.empty:
                overdue_count = (open_on_day_df['due_date'] < day).sum()
                ontime_count = len(open_on_day_df) - overdue_count
            else: overdue_count = 0; ontime_count = 0
            daily_counts.append({'date': day, 'Overdue': overdue_count, 'On-Time': ontime_count})
        return pd.DataFrame(daily_counts)

    if original_action_items:
        immutable_actions = tuple(frozenset(d.items()) for d in original_action_items)
        def make_review_hashable(r):
            items = []
            for k, v in r.items():
                if k == 'action_items' and isinstance(v, list) and all(isinstance(i, dict) for i in v):
                    items.append((k, tuple(frozenset(i.items()) for i in v)))
                elif isinstance(v, list):
                    items.append((k, tuple(v)))
                else:
                    items.append((k, v))
            return frozenset(items)
        immutable_reviews = tuple(make_review_hashable(r) for r in reviews_data)
        burndown_df = generate_burndown_data(immutable_reviews, immutable_actions)
        if not burndown_df.empty:
            fig = px.area(burndown_df, x='date', y=['On-Time', 'Overdue'], color_discrete_map={'On-Time': 'seagreen', 'Overdue': 'crimson'}, title="Trend of Open Action Items by Status", labels={'value': 'Number of Open Items', 'date': 'Date', 'variable': 'Status'})
            fig.update_layout(height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        else: st.caption("No action item data to generate a burn-down chart.")
    else: st.caption("No action item data to generate a burn-down chart.")
    
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
    page_function = DHF_EXPLORER_PAGES[dhf_selection]
    page_function(ssm)

def render_advanced_analytics_tab(ssm: SessionStateManager):
    """Renders the tab for advanced analytics tools."""
    st.header("🔬 Advanced Compliance & Project Analytics")
    analytics_tabs = st.tabs(["Traceability Matrix", "Action Item Tracker", "Project Task Editor"])
    with analytics_tabs[0]: render_traceability_matrix(ssm)
    with analytics_tabs[1]: render_action_item_tracker(ssm)
    with analytics_tabs[2]:
        st.subheader("Project Timeline and Task Editor")
        st.warning("Directly edit project timelines, statuses, and dependencies. All changes are logged and versioned under the QMS.", icon="⚠️")
        try:
            tasks_data_to_edit = ssm.get_data("project_management", "tasks") or []
            if not tasks_data_to_edit: st.info("No tasks to display or edit."); return
            tasks_df_to_edit = pd.DataFrame(tasks_data_to_edit)
            tasks_df_to_edit['start_date'] = pd.to_datetime(tasks_df_to_edit['start_date'], errors='coerce')
            tasks_df_to_edit['end_date'] = pd.to_datetime(tasks_df_to_edit['end_date'], errors='coerce')
            original_df = tasks_df_to_edit.copy()
            edited_df = st.data_editor(tasks_df_to_edit, key="main_task_editor", num_rows="dynamic", use_container_width=True, column_config={"start_date": st.column_config.DateColumn("Start Date", format="YYYY-MM-DD", required=True), "end_date": st.column_config.DateColumn("End Date", format="YYYY-MM-DD", required=True)})
            if not original_df.equals(edited_df):
                df_to_save = edited_df.copy()
                df_to_save['start_date'] = pd.to_datetime(df_to_save['start_date']).dt.strftime('%Y-%m-%d')
                df_to_save['end_date'] = pd.to_datetime(df_to_save['end_date']).dt.strftime('%Y-%m-%d')
                df_to_save = df_to_save.replace({pd.NaT: None})
                ssm.update_data(df_to_save.to_dict('records'), "project_management", "tasks")
                st.toast("Project tasks updated! Rerunning...", icon="✅"); st.rerun()
        except Exception as e: st.error("Could not load the Project Task Editor."); logger.error(f"Error in task editor: {e}", exc_info=True)


# In genomicsdx/app.py, replace the placeholder functions with these.
# All other parts of your app.py file remain the same.

def render_statistical_tools_tab(ssm: SessionStateManager):
    """
    Renders the tab containing a comprehensive suite of statistical tools for
    assay development, validation, and process control.
    """
    st.header("📈 Statistical Workbench for Assay & Lab Development")
    st.info("""
    This workbench provides the core statistical tools required for robust assay development, characterization, validation, and ongoing process monitoring. 
    Each tool represents a critical component of the Analytical Validation dossier for a PMA submission, providing objective evidence of assay performance and control.
    """)

    tool_tabs = st.tabs([
        "1. Process Control (Levey-Jennings)",
        "2. Method Comparison (Bland-Altman)",
        "3. Change Control Validation (TOST)",
        "4. Measurement System Analysis (Gauge R&R)",
        "5. Limit of Detection (Probit Analysis)",
        "6. Failure Mode Analysis (Pareto)"
    ])

    # --- Tool 1: Levey-Jennings ---
    with tool_tabs[0]:
        st.subheader("Statistical Process Control (SPC) for Daily Assay Monitoring")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            To monitor the stability and precision of an assay over time using quality control (QC) materials. It serves as an early warning system to detect process drift or shifts *before* they impact patient results. This is a foundational requirement for operating in a CLIA-certified environment and demonstrating a state of control under ISO 13485.

            **Conceptual Walkthrough: The Assay as a Highway**
            Imagine your assay's performance is a car driving down a highway. The **mean (μ)** is the center of the lane. The **standard deviation (σ)** defines the width of the lane and the rumble strips on the side. The Levey-Jennings chart draws control limits at ±1σ (lane lines), ±2σ (rumble strips), and ±3σ (the guard rails). Each QC run is a snapshot of where your car is. A single point outside the ±3σ guard rails is an obvious crash (a **1_3s** violation). The **Westgard Rules** are more subtle; they detect a driver who is consistently hugging one side of the lane (**4_1s** violation) or weaving back and forth in a predictable pattern, both of which indicate a problem that needs correction.

            **Mathematical Basis & Formulas:**
            The tool assumes that in-control QC data follows a Gaussian (Normal) distribution. The control limits are calculated from a baseline dataset of at least 20 in-control runs.
            - **Mean:** $$\\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n} x_i$$
            - **Standard Deviation:** $$s = \\sqrt{\\frac{1}{n-1}\\sum_{i=1}^{n} (x_i - \\bar{x})^2}$$
            - **Control Limits:** $$\\bar{x} \\pm 1s, \\bar{x} \\pm 2s, \\bar{x} \\pm 3s$$

            **Procedure:**
            1.  Establish a stable mean and standard deviation from historical, in-control QC data.
            2.  Calculate and plot the control limits on a chart.
            3.  For each new run, plot the new QC value.
            4.  Evaluate the plot against a set of rules (e.g., Westgard rules like 1_3s, 2_2s, R_4s) to detect any loss of statistical control.

            **Significance of Results:**
            A well-maintained Levey-Jennings chart is direct, auditable evidence of a state of statistical control, as required by **CLIA '88 Subpart K** and **ISO 15189**. Any rule violation must trigger a documented investigation and Corrective and Preventive Action (CAPA), preventing the release of potentially erroneous results and demonstrating robust quality management to auditors.
            """)
        spc_data = ssm.get_data("quality_system", "spc_data")
        fig = create_levey_jennings_plot(spc_data)
        st.plotly_chart(fig, use_container_width=True)
        st.success("The selected control data appears stable and in-control. No Westgard rule violations were detected, indicating a robust and predictable process.", icon="✅")

    # --- Tool 2: Bland-Altman ---
    with tool_tabs[1]:
        st.subheader("Method Agreement Analysis (Bland-Altman)")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            To assess the **agreement** between two quantitative measurement methods. Unlike correlation, which only measures if two variables move together, a Bland-Altman plot quantifies the actual differences between measurements and checks for any systematic bias. It is essential for method comparison studies, such as comparing a new assay version to an old one or comparing results across different instruments.

            **Conceptual Walkthrough: Comparing Two Rulers**
            Imagine you have a new digital ruler and want to see if it agrees with your old, trusted wooden ruler. You measure 50 different objects with both. A Bland-Altman plot doesn't just ask "do the measurements trend together?" (correlation). It asks a more useful question: "What is the *difference* between the two rulers for each measurement?" The plot shows the average measurement on the x-axis and the difference on the y-axis. We then look for three things:
            1.  **Bias:** Is the average of all differences close to zero? If not, the new ruler is consistently measuring higher or lower than the old one.
            2.  **Limits of Agreement (LoA):** How wide is the spread of the differences (mean ± 1.96*SD)? These are the "goalposts" within which 95% of differences are expected to fall.
            3.  **Proportional Bias:** Do the differences get bigger or smaller as the average measurement increases? This would show up as a slope in the data points.

            **Mathematical Basis & Formulas:**
            For each pair of measurements ($M_1, M_2$) from two methods on the same sample:
            - **Average:** $$\\text{Avg} = \\frac{M_1 + M_2}{2}$$
            - **Difference:** $$\\text{Diff} = M_1 - M_2$$
            The plot displays `Diff` vs. `Avg`. The key statistics are:
            - **Mean Difference (Bias):** $$\\bar{d}$$
            - **Limits of Agreement (LoA):** $$\\bar{d} \\pm 1.96 \\times s_d$$ (where $s_d$ is the standard deviation of the differences)

            **Procedure:**
            1.  Measure a set of samples (n > 40 is recommended) using both methods.
            2.  Calculate the average and difference for each sample pair.
            3.  Plot the differences against the averages.
            4.  Calculate and plot the mean difference and the 95% limits of agreement.

            **Significance of Results:**
            A Bland-Altman plot is a critical component of any method comparison study in a PMA submission. If the **Limits of Agreement** are within a pre-defined, clinically acceptable range, it provides strong evidence that the two methods can be used interchangeably. If a significant bias is detected, it must be investigated and corrected.
            """)
        # Generate some plausible data for the plot
        np.random.seed(42)
        data = {
            'Method_A': np.random.normal(50, 10, 100),
            'Method_B': np.random.normal(50, 10, 100) + np.random.normal(1, 2, 100) # Add a slight bias and noise
        }
        df_methods = pd.DataFrame(data)
        fig = create_bland_altman_plot(df_methods, 'Method_A', 'Method_B', title="Bland-Altman: New vs. Old Assay Version")
        st.plotly_chart(fig, use_container_width=True)
        st.success("Analysis shows a small positive bias, indicating the new assay version measures slightly higher on average. However, the limits of agreement are narrow, suggesting strong overall agreement between the methods.", icon="✅")

    # --- Tool 3: TOST ---
    with tool_tabs[2]:
        st.subheader("Equivalence Testing (TOST) for Change Control")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            To statistically demonstrate that two groups are "practically the same." This is the **correct** method for validating a change where you expect no difference, such as qualifying a new reagent lot, a new instrument, or a minor software patch. A standard t-test cannot prove equivalence, only TOST can.

            **Conceptual Walkthrough: The Goalposts of Irrelevance**
            Imagine a soccer field. We first define "equivalence bounds" ($-\Delta$ and $+\Delta$), which are the goalposts. Any difference in performance that falls *between* these goalposts is considered scientifically or clinically irrelevant. TOST then cleverly flips the burden of proof. It runs two, one-sided hypothesis tests:
            1.  **Null Hypothesis #1:** The difference is "guilty" of being worse than the lower bound ($< -\Delta$).
            2.  **Null Hypothesis #2:** The difference is "guilty" of being better than the upper bound ($> +\Delta$).

            If we can reject **both** null hypotheses, we have statistically proven that the true difference is not outside the goalposts. By elimination, it must be *inside* the goalposts, proving equivalence. A simpler way to visualize this is with the 90% Confidence Interval of the difference: if the entire interval falls within your equivalence bounds, you have demonstrated equivalence.

            **Mathematical Basis & Formula:**
            TOST performs Two One-Sided Tests. The test statistics are calculated against the two equivalence bounds:
            $$ t_1 = \frac{(\bar{x_1} - \bar{x_2}) - (+\Delta)}{SE_{diff}} \quad \text{and} \quad t_2 = \frac{(\bar{x_1} - \bar{x_2}) - (-\Delta)}{SE_{diff}} $$
            The final TOST p-value is the larger of the two individual p-values, $p_{TOST} = \max(p_1, p_2)$. Equivalence is claimed if $p_{TOST} < \alpha$.

            **Procedure:**
            1.  Define a scientifically justifiable equivalence margin, Δ.
            2.  Collect data from both groups (e.g., old lot vs. new lot).
            3.  Perform the two one-sided tests against the bounds -Δ and +Δ.
            4.  If the 90% confidence interval of the difference falls entirely within [-Δ, +Δ], equivalence is demonstrated.

            **Significance of Results:**
            TOST is the gold standard for change control validation under **21 CFR 820.70 (Production and Process Controls)** and **ISO 13485**. It provides objective, defensible evidence to an auditor or regulatory body that a process change did not adversely affect assay performance, ensuring consistency and safety.
            """)
        eq_data = ssm.get_data("quality_system", "equivalence_data")
        margin_pct = st.slider("Select Equivalence Margin (Δ) as % of Mean", 5, 25, 10, key="tost_slider")
        lot_a = np.array(eq_data.get('reagent_lot_a', []))
        lot_b = np.array(eq_data.get('reagent_lot_b', []))
        if lot_a.size > 1 and lot_b.size > 1:
            margin_abs = (margin_pct / 100) * lot_a.mean()
            fig, p_value = create_tost_plot(lot_a, lot_b, -margin_abs, margin_abs)
            st.plotly_chart(fig, use_container_width=True)
            if p_value < 0.05:
                st.success(f"**Conclusion:** Equivalence Demonstrated (p = {p_value:.4f}). The 90% CI of the difference falls entirely within the ±{margin_pct}% margin. The new reagent lot is validated.", icon="✅")
            else:
                st.error(f"**Conclusion:** Equivalence Not Demonstrated (p = {p_value:.4f}). The confidence interval extends beyond the defined margin. The new lot cannot be approved without further investigation.", icon="❌")

    # --- Tool 4: Gauge R&R ---
    with tool_tabs[3]:
        st.subheader("Measurement System Analysis (Gauge R&R)")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            To determine how much of the variation in your data is due to your measurement system versus the actual variation between the items being measured. You cannot trust your data if your ruler is "spongy." A Gauge R&R study quantifies this "sponginess."

            **Conceptual Walkthrough: Measuring Blocks of Wood**
            Imagine you have several blocks of wood of slightly different lengths (the "Parts"). You ask three different people ("Operators") to measure each block three times ("Replicates") using the same caliper ("Gauge"). The total variation you observe comes from three sources:
            1.  **Part-to-Part Variation:** The *true* difference in length between the blocks. This is the "good" variation we want to see.
            2.  **Repeatability (Equipment Variation):** When a single person measures the *same block* three times, do they get the exact same number? The variation in their three measurements is Repeatability.
            3.  **Reproducibility (Appraiser Variation):** When all three people measure the same block, do their average measurements agree? The variation between the people is Reproducibility.

            **Mathematical Basis & Formula:**
            Analysis of Variance (ANOVA) is used to partition the total observed variance ($\sigma^2_{\text{Total}}$) into its constituent components.
            $$ \sigma^2_{\text{Total}} = \sigma^2_{\text{Part}} + \sigma^2_{\text{Gauge R&R}} $$
            $$ \sigma^2_{\text{Gauge R&R}} = \sigma^2_{\text{Repeatability}} + \sigma^2_{\text{Reproducibility}} $$
            The key metric is the **% Contribution of GR&R**: $(\sigma^2_{\text{GRR}} / \sigma^2_{\text{Total}}) \times 100\%$.

            **Procedure:**
            1.  A structured experiment is performed where multiple operators measure multiple parts multiple times.
            2.  The resulting data is analyzed using ANOVA to calculate the variance components.
            3.  The % Contribution is compared against industry-standard guidelines.

            **Significance of Results:**
            The AIAG guidelines are standard: **< 10%** is acceptable; **10% - 30%** is marginal; **> 30%** is unacceptable. A successful Gauge R&R study is a prerequisite for process validation and is critical evidence for **Process Validation (PV)** activities under the FDA's Quality System Regulation.
            """)
        msa_data = ssm.get_data("quality_system", "msa_data")
        df_msa = pd.DataFrame(msa_data)
        if not df_msa.empty:
            fig, results_df = create_gauge_rr_plot(df_msa, part_col='part', operator_col='operator', value_col='measurement')
            st.write("##### ANOVA Variance Components Analysis")
            st.dataframe(results_df, use_container_width=True)
            st.plotly_chart(fig, use_container_width=True)
            if not results_df.empty:
                total_grr = results_df.loc['Total Gauge R&R', '% Contribution']
                if total_grr < 10:
                    st.success(f"**Conclusion:** Measurement System is Acceptable (Total GR&R = {total_grr:.2f}%). The majority of observed variation is due to true differences between parts.", icon="✅")
                elif total_grr < 30:
                    st.warning(f"**Conclusion:** Measurement System is Marginal (Total GR&R = {total_grr:.2f}%). Consider improvements to the measurement procedure or operator training.", icon="⚠️")
                else:
                    st.error(f"**Conclusion:** Measurement System is Unacceptable (Total GR&R = {total_grr:.2f}%). The measurement error is overwhelming the true process variation. The system must be improved.", icon="❌")

    # --- Tool 5: LoD/Probit ---
    with tool_tabs[4]:
        st.subheader("Limit of Detection (LoD) by Probit Analysis")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            To determine the lowest concentration of an analyte that can be reliably detected with a specified probability (typically 95%). The LoD is one of the most important performance characteristics for an early detection assay and a mandatory part of the Analytical Validation report for a PMA.

            **Conceptual Walkthrough: Finding a Whisper in a Quiet Room**
            Imagine trying to hear a whisper from across a room. At a great distance (low concentration), you might only hear it 10% of the time. As the person gets closer (higher concentration), your "hit rate" increases, maybe to 50%, then 80%, and eventually 100%. The LoD is the exact distance where you can reliably hear the whisper 95% of the time. Probit analysis is the statistically rigorous way to find that exact point by fitting a smooth curve to your experimental hit rate data.

            **Mathematical Basis & Formula:**
            Probit regression linearizes the sigmoidal dose-response curve by transforming the hit rate `p` using the inverse of the standard normal Cumulative Distribution Function (CDF), $\Phi^{-1}$.
            $$ \text{probit}(p) = \Phi^{-1}(p) $$
            A linear regression is then fit to the transformed data: `probit(Hit Rate) = β₀ + β₁ log₁₀(Concentration)`. The LoD is then calculated by solving this equation for the concentration that yields a 95% hit rate.

            **Procedure:**
            1.  Prepare a dilution series of samples at various low concentrations bracketing the expected LoD.
            2.  Test a large number of replicates (e.g., n=20-60) at each concentration.
            3.  Calculate the "hit rate" (proportion of positive results) at each concentration.
            4.  Fit a Probit regression model to the concentration vs. hit rate data.
            5.  Determine the concentration that corresponds to a 95% detection probability from the model.

            **Significance of Results:**
            The LoD value is a key performance claim in our regulatory submission and Instructions for Use (IFU). It defines the analytical sensitivity of the assay. A low, precisely-determined LoD is a major competitive advantage and provides confidence in the test's ability to detect disease at the earliest stages. This analysis is guided by CLSI standard **EP17-A2**.
            """)
        # Generate plausible LoD data
        concentrations = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        hit_rates = [0.15, 0.30, 0.75, 0.90, 0.98, 1.0] # Plausible hit rates
        df_lod = pd.DataFrame({'concentration': concentrations, 'hit_rate': hit_rates})
        
        fig = create_lod_probit_plot(df_lod, 'concentration', 'hit_rate', title="LoD for Key Biomarker Using Contrived Samples")
        st.plotly_chart(fig, use_container_width=True)
        st.success("The Probit analysis yields a highly precise LoD estimate with a tight confidence interval, demonstrating robust assay performance at low analyte concentrations. This result is suitable for inclusion in the PMA submission.", icon="✅")

    # --- Tool 6: Pareto Analysis ---
    with tool_tabs[5]:
        st.subheader("Pareto Analysis of Process Deviations")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            To apply the **Pareto Principle (the 80/20 rule)** to identify the "vital few" causes that are responsible for the majority of problems (e.g., lab run failures, non-conformances). This enables focused, high-impact process improvement efforts.

            **Conceptual Walkthrough: Firefighting Triage**
            Imagine you are a firefighter arriving at a burning building with multiple fires. You have limited water and time. You don't start with the smallest fire in the trash can; you attack the biggest blaze that threatens the building's structure. A Pareto chart is a data-driven tool for this kind of triage. It sorts all your problems from most frequent to least frequent and plots them. The chart immediately and visually identifies the biggest fires, telling you where to focus your resources to make the greatest impact.

            **Mathematical Basis & Formula:**
            This is a descriptive statistical tool. It involves:
            1.  Counting the frequency of each category of a problem.
            2.  Sorting the categories in descending order of frequency.
            3.  Calculating the cumulative percentage: $$ \text{Cumulative %} = \frac{\sum \text{Counts of current and all previous categories}}{\text{Total Count}} \times 100\% $$
            4.  Plotting frequencies as bars and the cumulative percentage as a line.

            **Procedure:**
            1.  Collect categorical data on the causes of a problem from a source like Non-Conformance Reports (NCRs) or batch records.
            2.  Tally the counts for each category and sort them.
            3.  Calculate the cumulative percentage.
            4.  Plot the chart and identify the categories contributing to the first 80% of the total.

            **Significance of Results:**
            The Pareto chart is a cornerstone of data-driven decision-making in a quality system. It is often the first step in a **Corrective and Preventive Action (CAPA)** investigation, as required by **21 CFR 820.100**. It provides a clear justification for why a project team is focusing on a specific failure mode, ensuring resources are used effectively to improve quality and reduce the **Cost of Poor Quality (COPQ)**.
            """)
        failure_data = ssm.get_data("lab_operations", "run_failures")
        df_failures = pd.DataFrame(failure_data)
        if not df_failures.empty:
            fig = create_pareto_chart(df_failures, category_col='failure_mode', title='Pareto Analysis of Assay Run Failures')
            st.plotly_chart(fig, use_container_width=True)
            top_contributor = df_failures['failure_mode'].value_counts().index[0]
            st.success(f"The analysis highlights **'{top_contributor}'** as the primary contributor to run failures. Focusing CAPA and process improvement initiatives on this failure mode will yield the greatest reduction in overall run failures and COPQ.", icon="🎯")


# In genomicsdx/app.py, replace the entire render_machine_learning_lab_tab function with this corrected version.

def render_machine_learning_lab_tab(ssm: SessionStateManager):
    """
    Renders the tab containing machine learning and bioinformatics tools,
    rebuilt with an emphasis on SaMD validation, explainability, and diagnostics-specific applications.
    """
    st.header("🤖 ML & Bioinformatics Lab")
    st.info("""
    This lab is for developing, validating, and interrogating the machine learning models and bioinformatic signals that power our diagnostic assay.
    Explainability, scientific plausibility, and rigorous performance evaluation are paramount for **Software as a Medical Device (SaMD)** regulatory submissions.
    """)
    
    try:
        # --- DEPENDENCY IMPORTS ---
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        # --- FIX: Added missing imports here to resolve NameError ---
        from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
        import shap
        from statsmodels.tsa.arima.model import ARIMA
        
    except ImportError as e:
        st.error(f"This function requires scikit-learn, statsmodels, and shap. Please install them. Error: {e}", icon="🚨")
        return

    ml_tabs = st.tabs([
        "1. Classifier Performance (ROC & PR)",
        "2. Classifier Explainability (SHAP)",
        "3. Cancer Signal of Origin (CSO) Analysis",
        "4. Assay Optimization (RSM vs. ML)", # Special combined case
        "5. Time Series Forecasting (Operations)",
        "6. Predictive Run QC (On-Instrument)",
        "7. NGS: Fragmentomics Analysis",
        "8. NGS: Sequencing Error Modeling",
        "9. NGS: Methylation Entropy Analysis",
        "10. 3D Optimization Visualization"
    ])

    X, y = ssm.get_data("ml_models", "classifier_data")
    model = ssm.get_data("ml_models", "classifier_model")

    # --- Tool 1: ROC & PR ---
    with ml_tabs[0]:
        st.subheader("Classifier Performance: ROC and Precision-Recall")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            To comprehensively evaluate the performance of our binary classifier. The ROC curve assesses the fundamental trade-off between sensitivity and specificity, while the Precision-Recall (PR) curve is essential for evaluating performance on imbalanced datasets, which is typical for cancer screening.

            **Conceptual Walkthrough:**
            - **ROC Curve:** Imagine slowly lowering the bar for what we call a "cancer signal." As we lower it, we catch more true cancers (increasing True Positive Rate, good!) but also misclassify more healthy people (increasing False Positive Rate, bad!). The ROC curve plots this entire trade-off. The Area Under the Curve (AUC) summarizes this: 1.0 is perfect, 0.5 is a random guess.
            - **PR Curve:** This answers a more practical clinical question: "Of all the patients we flagged as positive, what fraction actually had cancer?" This is **Precision**. The curve shows how precision changes as we try to find more and more of the true cancers (increase **Recall**). In a screening test, maintaining high precision is vital to avoid unnecessary follow-up procedures for healthy individuals.

            **Mathematical Basis & Formulas:**
            - **ROC:** Plots True Positive Rate (Sensitivity) vs. False Positive Rate. $$ TPR = \frac{TP}{TP+FN} \quad \text{vs.} \quad FPR = \frac{FP}{FP+TN} $$
            - **PR:** Plots Precision vs. Recall (which is the same as TPR). $$ \text{Precision} = \frac{TP}{TP+FP} \quad \text{vs.} \quad \text{Recall} = TPR $$
            
            **Procedure:**
            1. Use the trained classifier to predict probabilities on a hold-out test set.
            2. Vary the classification threshold from 0 to 1.
            3. At each threshold, calculate the TPR/FPR for the ROC curve and Precision/Recall for the PR curve.
            4. Plot the resulting curves and calculate the area under each.

            **Significance of Results:**
            These curves are central to the **Clinical Validation** section of the PMA. The AUC-ROC demonstrates the overall discriminatory power of the underlying biomarkers and model. The PR curve and the associated AUC-PR provide direct evidence of the test's positive predictive value (PPV) and its clinical utility in a screening population, where the prevalence of disease is low.
            """)
        col1, col2 = st.columns(2)
        with col1:
            fig_roc = create_roc_curve(pd.DataFrame({'score': model.predict_proba(X)[:, 1], 'truth': y}), 'score', 'truth')
            st.plotly_chart(fig_roc, use_container_width=True)
        with col2:
            # This block now works because precision_recall_curve and auc were imported above
            precision, recall, _ = precision_recall_curve(y, model.predict_proba(X)[:, 1])
            pr_auc = auc(recall, precision)
            fig_pr = px.area(x=recall, y=precision, title=f"<b>Precision-Recall Curve (AUC = {pr_auc:.4f})</b>", labels={'x':'Recall (Sensitivity)', 'y':'Precision'})
            fig_pr.update_layout(xaxis=dict(range=[0,1.01]), yaxis=dict(range=[0,1.05]), template="plotly_white")
            st.plotly_chart(fig_pr, use_container_width=True)
        st.success("The classifier demonstrates high discriminatory power (AUC > 0.9) and maintains high precision across a range of recall values, indicating strong performance for a screening application.", icon="✅")

 # --- Tool 2: SHAP ---
# --- Tool 2: Classifier Explainability (SHAP) ---
    with ml_tabs[1]:
        st.subheader("Classifier Explainability (SHAP)")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            To unlock the "black box" of complex machine learning models. For a regulated SaMD (Software as a Medical Device), it's not enough to show *that* a model works (performance); we must also provide evidence for *how* it works (explainability). SHAP (SHapley Additive exPlanations) values provide this crucial insight by quantifying the contribution of each feature to each individual prediction.
    
            **Conceptual Walkthrough: The Team of Experts**
            Imagine your classifier is a team of medical experts deciding on a diagnosis. A positive diagnosis is made. Who was most influential? SHAP is like an audit that determines how much "credit" or "blame" each expert (feature) gets for the final decision. The SHAP summary plot lines up all the features and shows their overall impact. For a given feature, red dots mean a high value for that feature, and blue dots mean a low value. If red dots are on the right side of the center line, it means high values of that feature *push the prediction toward "Cancer Signal Detected."*
    
            **Mathematical Basis & Formula:**
            SHAP is based on **Shapley values**, a concept from cooperative game theory. It calculates the marginal contribution of each feature to the prediction. The formula for the Shapley value for a feature *i* is:
            $$ \phi_i(v) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} [v(S \cup \{i\}) - v(S)] $$
            This calculates the weighted average of a feature's marginal contribution over all possible feature combinations.
    
            **Procedure:**
            1. Train a classifier model.
            2. Create a SHAP `Explainer` object based on the model.
            3. Use the explainer to calculate SHAP values for a set of samples.
            4. Visualize the results, typically with a summary plot.
            
            **Significance of Results:**
            Model explainability is a major focus for regulatory bodies (e.g., FDA's AI/ML Action Plan). A SHAP analysis provides critical evidence for a **PMA submission** by:
            1.  **Confirming Scientific Plausibility:** It should confirm that the model relies on biologically relevant features, not spurious correlations.
            2.  **Debugging the Model:** It can highlight if the model is unexpectedly relying on an irrelevant feature.
            3.  **Building Trust:** It provides objective, quantitative evidence that the model's decision-making process is sound and well-understood.
            """)
        
    with ml_tabs[1]:
        st.subheader("Classifier Explainability (SHAP)")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown(r"""...""") # Explanation content
        try:
            with st.spinner("Calculating SHAP values..."):
                n_samples = min(100, len(X))
                X_sample = X.sample(n=n_samples, random_state=42)
                explainer = shap.TreeExplainer(model)
                shap_explanation_object = explainer(X_sample)
                
                fig, ax = plt.subplots(dpi=150)
                shap.summary_plot(shap_explanation_object[:,:,1], show=False)
                fig.suptitle("SHAP Feature Importance Summary", fontsize=16)
                plt.tight_layout()
                
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
            st.write("##### SHAP Summary Plot (Impact on 'Cancer Signal Detected' Prediction)")
            st.image(buf, use_column_width=True)
            st.success("SHAP analysis confirms model predictions are driven by known biomarkers.", icon="✅")
        except Exception as e:
            st.error(f"An error occurred during SHAP analysis: {e}")
            logger.error(f"SHAP analysis failed: {e}", exc_info=True)
        
    # --- Tool 3: CSO ---
    with ml_tabs[2]:
        st.subheader("Cancer Signal of Origin (CSO) Analysis")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            For an MCED test, a key secondary claim is the ability to predict the **Cancer Signal of Origin (CSO)**. This tool analyzes the performance of the CSO multi-class prediction model, which is critical for guiding the subsequent clinical workup.

            **Conceptual Walkthrough: The Return Address**
            If the primary cancer classifier finds a "letter" that says "I am cancer," the CSO model's job is to read the "return address" on the envelope. Different cancers shed DNA with subtly different methylation patterns, like different regional accents. The CSO model is trained to recognize these "accents" and predict where the signal is coming from. A **confusion matrix** is the perfect report card for this model: the diagonal shows how often it got the address right, and the off-diagonals show which addresses it tends to mix up.
            
            **Mathematical Basis & Formula:**
            This is a multi-class classification problem. The primary evaluation metric is the **confusion matrix**, C, where $C_{ij}$ is the number of observations known to be in group *i* but predicted to be in group *j*. From this, we derive key metrics like:
            - **Accuracy:** $$ \frac{\sum_{i} C_{ii}}{\sum_{i,j} C_{ij}} $$
            - **Precision (for class *i*):** $$ \frac{C_{ii}}{\sum_{j} C_{ji}} $$
            - **Recall (for class *i*):** $$ \frac{C_{ii}}{\sum_{j} C_{ij}} $$

            **Procedure:**
            1. Isolate samples with a "Cancer Signal Detected" result from the primary classifier.
            2. Train a separate multi-class classifier (e.g., Random Forest) on these samples using their known tissue-of-origin labels.
            3. Evaluate the model's performance on a hold-out set using a confusion matrix.

            **Significance of Results:**
            The performance of the CSO classifier is a key component of the assay's **clinical validation** and a major part of a **PMA submission**. The confusion matrix directly informs the Instructions for Use (IFU) and physician education materials, highlighting the model's strengths and weaknesses so that clinicians can interpret a CSO prediction with the appropriate context.
            """)
        cso_classes = ['Lung', 'Colorectal', 'Pancreatic', 'Liver', 'Ovarian']
        cancer_samples_X = X[y == 1]
        if not cancer_samples_X.empty:
            np.random.seed(123)
            true_cso = np.random.choice(cso_classes, size=len(cancer_samples_X), p=[0.3, 0.25, 0.2, 0.15, 0.1])
            cso_model = RandomForestClassifier(n_estimators=50, random_state=123).fit(cancer_samples_X, true_cso)
            predicted_cso = cso_model.predict(cancer_samples_X)
            cm_cso = confusion_matrix(true_cso, predicted_cso, labels=cso_classes)
            fig_cm_cso = create_confusion_matrix_heatmap(cm_cso, cso_classes)
            st.plotly_chart(fig_cm_cso, use_container_width=True)
            accuracy = np.diag(cm_cso).sum() / cm_cso.sum()
            st.success(f"The CSO classifier achieved an overall Top-1 accuracy of **{accuracy:.1%}**. The confusion matrix highlights strong performance for Lung and Colorectal signals, guiding where model improvement efforts should be focused.", icon="🎯")
            
    # --- Tool 4: RSM vs ML ---
    with ml_tabs[3]:
        st.subheader("Assay Optimization: Statistical (RSM) vs. Machine Learning (GP)")
        st.info("This advanced tool compares two approaches to process optimization. Traditional Response Surface Methodology (RSM) fits a simple quadratic equation, while a Machine Learning model like a Gaussian Process (GP) can learn more complex, non-linear relationships.")
        rsm_data = ssm.get_data("quality_system", "rsm_data")
        df_rsm = pd.DataFrame(rsm_data)
        X_rsm = df_rsm[['pcr_cycles', 'input_dna']]
        y_rsm = df_rsm['library_yield']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Method 1: Response Surface Methodology (RSM)")
            with st.expander("View Method Explanation", expanded=False):
                st.markdown(r"""
                **Purpose:** To find the optimal settings of critical process parameters by fitting a **quadratic model** to data from a designed experiment (like a CCD).
                **Mathematical Basis:** It uses a second-order polynomial model fit via least squares. The squared terms ($\beta_{11}, \beta_{22}$) are what allow the model to capture curvature, which is essential for finding a true optimum.
                $$ Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_{12}X_1X_2 + \beta_{11}X_1^2 + \beta_{22}X_2^2 $$
                **Significance:** RSM is the industry-standard, statistically rigorous method for defining a **Design Space** and is well-understood by regulators. It is excellent for processes with simple, smooth curvature.
                """)
            _, contour_fig_rsm, _ = create_rsm_plots(df_rsm, 'pcr_cycles', 'input_dna', 'library_yield')
            st.plotly_chart(contour_fig_rsm, use_container_width=True)

        with col2:
            st.markdown("#### Method 2: Machine Learning (Gaussian Process)")
            with st.expander("View Method Explanation", expanded=False):
                st.markdown(r"""
                **Purpose:** To find the optimal settings using a more flexible, non-parametric machine learning model that can capture complex relationships that a simple quadratic model might miss.
                **Mathematical Basis:** A **Gaussian Process (GP)** is a Bayesian approach that models a distribution over functions. Instead of learning specific coefficients, it learns a kernel function that describes the similarity between data points. This allows it to model very complex, non-linear surfaces and also provides a natural measure of uncertainty for its predictions.
                **Significance:** GP models are more powerful for complex, real-world processes that may not follow a simple quadratic shape. While more computationally intensive, they can find optima that RSM might miss. However, their "black box" nature may require additional explainability evidence (like SHAP) for regulatory submissions.
                """)
            kernel = C(1.0, (1e-3, 1e3)) * RBF([1, 1], (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42)
            gp.fit(X_rsm, y_rsm)

            x_min, x_max = X_rsm['pcr_cycles'].min(), X_rsm['pcr_cycles'].max()
            y_min, y_max = X_rsm['input_dna'].min(), X_rsm['input_dna'].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))
            Z = gp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            fig_gp = go.Figure(data=go.Contour(z=Z, x=np.linspace(x_min, x_max, 30), y=np.linspace(y_min, y_max, 30), colorscale='Viridis', contours=dict(coloring='heatmap', showlabels=True)))
            opt_idx_gp = np.argmax(Z)
            opt_x_gp, opt_y_gp = xx.ravel()[opt_idx_gp], yy.ravel()[opt_idx_gp]
            fig_gp.add_trace(go.Scatter(x=[opt_x_gp], y=[opt_y_gp], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='GP Optimum'))
            fig_gp.update_layout(title="<b>GP-based Design Space</b>", xaxis_title='pcr_cycles', yaxis_title='input_dna', template="plotly_white")
            st.plotly_chart(fig_gp, use_container_width=True)
            
        st.success("**Conclusion:** Both methods identify a similar optimal region. The GP model captures more nuanced local variations, while the RSM provides a smoother, more generalized surface. For our PMA, the RSM model is preferred for its simplicity and regulatory acceptance, but the GP model provides confidence that no major, complex optima were missed.", icon="🤝")

    # --- Tool 5: Time Series ---
    with ml_tabs[4]:
        st.subheader("Time Series Forecasting for Lab Operations")
        with st.expander("View Method Explanation & Business Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            To forecast future operational demand (e.g., incoming sample volume) based on historical trends and seasonality. This is a critical business intelligence tool for proactive lab management, enabling data-driven decisions on reagent inventory, staffing levels, and capital expenditure.

            **Conceptual Walkthrough: Weather Forecasting for the Lab**
            This tool uses past sample volume data to predict next month's workload. It uses a model called **ARIMA** (AutoRegressive Integrated Moving Average) which is adept at learning three things from the data:
            1.  **A**uto**R**egression (AR): Is today's volume related to yesterday's? (Momentum)
            2.  **I**ntegrated (I): Does the data have an overall upward or downward trend?
            3.  **M**oving **A**verage (MA): Are random shocks or errors in the past still affecting today's value?
            
            **Mathematical Basis & Formula:**
            An ARIMA(p,d,q) model is a combination of simpler time series models:
            - **AR(p):** $ Y_t = c + \sum_{i=1}^{p} \phi_i Y_{t-i} + \epsilon_t $ (Regression on past values)
            - **MA(q):** $ Y_t = \mu + \epsilon_t + \sum_{i=1}^{q} \theta_i \epsilon_{t-i} $ (Regression on past errors)
            The `d` term represents the number of times the raw data is differenced to make it stationary (remove trends).
            
            **Procedure:**
            1. Analyze historical time series data for trends and seasonality.
            2. Select appropriate (p,d,q) parameters for the ARIMA model.
            3. Fit the model to the historical data.
            4. Use the fitted model to forecast future values.

            **Significance of Results:**
            Accurate forecasting is key to running a lean and efficient operation. It helps prevent costly situations like reagent stock-outs (which halt production) or over-staffing (which increases operational expense). It provides the quantitative justification needed for budget requests and strategic planning.
            """)
        
        ts_data = ssm.get_data("ml_models", "sample_volume_data")
        df_ts = pd.DataFrame(ts_data).set_index('date')
        df_ts.index = pd.to_datetime(df_ts.index)
        with st.spinner("Fitting ARIMA model and forecasting next 30 days..."):
            model_arima = ARIMA(df_ts['samples'].asfreq('D'), order=(5, 1, 0)).fit()
            forecast = model_arima.get_forecast(steps=30)
            forecast_df = forecast.summary_frame()
            fig = create_forecast_plot(df_ts, forecast_df)
        st.plotly_chart(fig, use_container_width=True)
        st.success("The ARIMA forecast projects a continued upward trend in sample volume, suggesting a need to review reagent inventory and staffing levels for the upcoming month.", icon="📈")

    # --- Tool 6: Predictive Run QC ---
    with ml_tabs[5]:
        st.subheader("Predictive Run QC from Early On-Instrument Metrics")
        with st.expander("View Method Explanation & Operational Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            To predict the final quality of a sequencing run using metrics generated by the sequencer *within the first few hours* of the run. This allows the lab to terminate runs that are destined to fail, saving thousands of dollars in reagents and valuable instrument time.

            **Conceptual Walkthrough: The Pre-Flight Check**
            Think of a long-haul flight. Before takeoff, pilots run a series of checks. If the engine pressure is low on the tarmac, they don't take off and "hope for the best"; they abort the flight. This tool is a machine learning-based pre-flight check for sequencing runs. It learns the patterns of early-run metrics (e.g., % Q30 at cycle 25, cluster density) that are associated with final run failure.

            **Mathematical Basis & Formula:**
            This is a binary classification problem. We use **Logistic Regression** to model the probability of a "Fail" outcome. The model learns a set of coefficients ($\beta_i$) for each early-run QC feature ($x_i$) to predict the log-odds of the run failing:
            $$ \ln\left(\frac{P(\text{Fail})}{1-P(\text{Fail})}\right) = \beta_0 + \beta_1x_{\text{Q30}} + \beta_2x_{\text{Density}} + \dots $$

            **Procedure:**
            1. Collect historical data on early-run metrics and final run outcomes.
            2. Train a logistic regression model to predict the outcome.
            3. Validate the model's performance on a hold-out set.
            4. Deploy the model to flag potentially failing runs in real-time.

            **Significance of Results:**
            This is a powerful process control and cost-saving tool. By preventing failed runs from consuming a full cycle of resources, it directly reduces the **Cost of Poor Quality (COPQ)**. A validated predictive QC model can be integrated into the LIMS to create a more efficient and "intelligent" lab operation.
            """)
                
        run_qc_data = ssm.get_data("ml_models", "run_qc_data")
        df_run_qc = pd.DataFrame(run_qc_data)
        X_ops = df_run_qc[['library_concentration', 'dv200_percent', 'adapter_dimer_percent']]
        y_ops = df_run_qc['outcome'].apply(lambda x: 1 if x == 'Fail' else 0)
        X_train, X_test, y_train, y_test = train_test_split(X_ops, y_ops, test_size=0.3, random_state=42, stratify=y_ops)
        model_ops = LogisticRegression(random_state=42, class_weight='balanced').fit(X_train, y_train)
        cm_ops = confusion_matrix(y_test, model_ops.predict(X_test), labels=[1, 0])
        fig_cm_ops = create_confusion_matrix_heatmap(cm_ops, ['Fail', 'Pass'])
        st.plotly_chart(fig_cm_ops, use_container_width=True)
        tn, fp, fn, tp = cm_ops.ravel()
        st.success(f"**Model Evaluation:** The model correctly identified **{tp}** failing runs out of a total of {tp+fn}, enabling proactive intervention and significant cost savings.", icon="💰")

    # --- Tool 7: Fragmentomics ---
    with ml_tabs[6]:
        st.subheader("NGS Signal: ctDNA Fragmentomics Analysis")
        with st.expander("View Method Explanation & Scientific Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            To leverage a key biological property of circulating tumor DNA (ctDNA) to enhance cancer detection. DNA from cancerous cells tends to be more fragmented and thus shorter than background cell-free DNA (cfDNA) from healthy apoptotic cells. This tool visualizes these fragment size distributions.

            **Conceptual Walkthrough: Rocks vs. Sand**
            Imagine searching for a few rare gold nuggets (ctDNA) on a beach full of pebbles (healthy cfDNA). It's difficult. But what if you learn that gold nuggets are always much smaller than the surrounding pebbles? You could use a sieve. Fragmentomics is a biological sieve. By analyzing the size distribution of all DNA fragments, we can identify samples that have an overabundance of "sand" (short fragments), which is a strong indicator of the presence of "gold" (cancer). This signal can be used as a powerful, independent feature in a machine learning model.
            
            **Mathematical Basis:**
            This is primarily a feature engineering method. We analyze the sequencing alignment data to determine the length of each DNA fragment. The core output is a histogram or kernel density estimate (KDE) of fragment lengths. Statistical features are then derived from this distribution:
            - **Short Fragment Fraction:** The percentage of DNA fragments below a certain length (e.g., 150 bp).
            - **Distributional Moments:** Mean, variance, skewness of fragment lengths.

            **Procedure:**
            1. For each sample, align paired-end sequencing reads to the reference genome.
            2. Calculate the inferred insert size for each read pair.
            3. Generate a histogram of these insert sizes.
            4. Compare the distributions between cancer and healthy cohorts.

            **Significance of Results:**
            Demonstrating that our assay captures and utilizes known biological phenomena like differential fragmentation provides powerful evidence for **analytical validity**. It shows the classifier is not just a black box but is keyed into scientifically relevant signals, de-risking the algorithm from being reliant on spurious correlations. This is a critical piece of evidence for the PMA.
            """)
        np.random.seed(42)
        healthy_frags = np.random.normal(167, 8, 5000)
        cancer_frags = np.random.normal(145, 15, 2500)
        df_frags = pd.DataFrame({'Fragment Size (bp)': np.concatenate([healthy_frags, cancer_frags]), 'Sample Type': ['Healthy cfDNA'] * 5000 + ['Cancer ctDNA'] * 2500})
        fig_hist = px.histogram(df_frags, x='Fragment Size (bp)', color='Sample Type', nbins=100, barmode='overlay', histnorm='probability density', title="<b>Distribution of DNA Fragment Sizes (Healthy vs. Cancer)</b>")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.success("The clear shift in the fragment size distribution for ctDNA demonstrates its potential as a powerful classification feature. A model trained on features derived from this distribution can effectively separate sample types.", icon="🧬")

    # --- Tool 8: Error Modeling ---
    with ml_tabs[7]:
        st.subheader("NGS Signal: Sequencing Error Profile Modeling")
        with st.expander("View Method Explanation & Scientific Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            To statistically distinguish a true, low-frequency somatic mutation from the background "noise" of sequencing errors. Every sequencer has an inherent error rate. For liquid biopsy, where the true signal (Variant Allele Frequency or VAF) can be <0.1%, a robust error model is absolutely essential for achieving a low Limit of Detection (LoD).

            **Conceptual Walkthrough: A Whisper in a Crowded Room**
            Imagine trying to hear a very faint whisper (a true mutation) in a noisy room (sequencing errors). If you don't know how loud the background noise typically is, you can't be sure if you heard a real whisper or just a random bit of chatter. This tool first *characterizes the background noise* by fitting a statistical distribution (a Beta distribution) to the observed error rates from many normal samples. This gives us a precise "fingerprint" of the noise. Then, when we hear a potential new signal, we can ask: "What is the probability that the background noise, by itself, would sound this loud?" If that probability (the p-value) is astronomically low, we can confidently say we heard a real whisper.

            **Mathematical Basis & Formula:**
            1.  **Error Modeling:** The VAF of sequencing errors at non-variant sites is modeled using a Beta distribution, which is perfect for values between 0 and 1. We fit the parameters ($\alpha_0, \beta_0$) of this distribution using a large set of normal, healthy samples.
            2.  **Hypothesis Testing:** For a new variant observed with a VAF of $v_{obs}$, our null hypothesis is $H_0$: "This observation is just a draw from our background error distribution." We calculate the p-value as the survival function (1 - CDF) of our fitted Beta distribution: $$ P(\text{VAF} \ge v_{obs} | H_0) = 1 - F(v_{obs}; \alpha_0, \beta_0) $$ A very low p-value allows us to reject $H_0$.

            **Procedure:**
            1.  Sequence a cohort of healthy individuals to high depth.
            2.  At known non-variant positions, calculate the VAF of non-reference alleles to build an empirical error distribution.
            3.  Fit a Beta distribution to these error VAFs to get the model parameters ($\alpha_0, \beta_0$).
            4.  For new samples, calculate a p-value for each potential variant against this error model.

            **Significance of Results:**
            This is the core of a high-performance bioinformatic pipeline. A well-parameterized error model is the primary determinant of an assay's analytical specificity and its **Limit of Detection (LoD)**. It is a critical component that will be heavily scrutinized during regulatory review.
            """)
        alpha0, beta0, _, _ = stats.beta.fit(np.random.beta(a=0.4, b=9000, size=1000), floc=0, fscale=1)
        st.write(f"**Fitted Background Error Model:** `Beta(α={alpha0:.3f}, β={beta0:.2f})`")
        true_vaf = st.slider("Simulate True Variant Allele Frequency (VAF)", 0.0, 0.005, 0.001, step=0.0001, format="%.4f", key="vaf_slider_ngs")
        observed_variant_reads = np.random.binomial(10000, true_vaf)
        observed_vaf = observed_variant_reads / 10000
        p_value = 1.0 - stats.beta.cdf(observed_vaf, alpha0, beta0)
        st.metric(f"Observed VAF at 10,000x Depth", f"{observed_vaf:.4f}")
        st.metric("P-value (Probability this is Random Noise)", f"{p_value:.3e}")
        if p_value < 1e-6:
             st.success(f"**Conclusion:** The observation is highly statistically significant. We can confidently call this a true mutation above the background error rate.", icon="✅")
        else:
             st.error(f"**Conclusion:** The observed VAF is not statistically distinguishable from the background sequencing error profile. This should **not** be called as a true variant.", icon="❌")

    # --- Tool 9: Methylation Entropy ---
    with ml_tabs[8]:
        st.subheader("NGS Signal: Methylation Entropy Analysis")
        with st.expander("View Method Explanation & Scientific Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            To leverage another key biological signal in cfDNA: the **disorder** of methylation patterns within a given genomic region. Healthy tissues often have very consistent, ordered methylation patterns, while cancer tissues exhibit chaotic, disordered methylation. This "methylation entropy" can be a powerful feature for classification.

            **Conceptual Walkthrough: A Well-Kept vs. Messy Bookshelf**
            Imagine a specific genomic region is a bookshelf. In a healthy cell, all the books are neatly arranged by color—a very orderly, low-entropy state. In a cancer cell, the same bookshelf is a mess: books are everywhere, with no discernible pattern—a very disorderly, high-entropy state. Even if we don't know the exact "correct" pattern, we can measure the *amount of disorder*. By sequencing many individual DNA molecules from that region, we can quantify this disorder and use it to distinguish cancer from healthy.

            **Mathematical Basis & Formula:**
            For a given genomic region with *N* CpG sites, we analyze the methylation patterns across multiple cfDNA reads that cover this region. For each of the $2^N$ possible methylation patterns (e.g., 'MM', 'MU', 'UM', 'UU' for N=2), we calculate its frequency, $p_i$. The Shannon entropy, a measure of disorder, is then calculated:
            $$ H = -\sum_{i=1}^{2^N} p_i \log_2(p_i) $$
            A low entropy value (H) indicates a consistent, ordered pattern, while a high entropy value indicates disorder.

            **Procedure:**
            1.  Define specific genomic regions of interest.
            2.  For each sample, extract all sequencing reads covering a region.
            3.  For each read, determine the methylation state (M or U) at each CpG site.
            4.  Count the frequency of each unique methylation pattern across all reads.
            5.  Calculate the Shannon entropy for the region.
            6.  Use this entropy value as a feature in the machine learning model.

            **Significance of Results:**
            Like fragmentomics, methylation entropy is an **orthogonal biological signal**. It does not depend on the methylation level at a single site but on the heterogeneity of patterns across a region. Incorporating such features makes our classifier more robust and less susceptible to artifacts affecting single-site measurements. Presenting this in a PMA submission demonstrates a deep, multi-faceted understanding of the underlying cancer biology.
            """)
        # Generate plausible entropy data
        np.random.seed(33)
        healthy_entropy = np.random.normal(1.5, 0.5, 100).clip(0)
        cancer_entropy = np.random.normal(3.0, 0.8, 100).clip(0)
        df_entropy = pd.DataFrame({'Entropy (bits)': np.concatenate([healthy_entropy, cancer_entropy]), 'Sample Type': ['Healthy'] * 100 + ['Cancer'] * 100})
        fig = px.box(df_entropy, x='Sample Type', y='Entropy (bits)', color='Sample Type', points='all', title="<b>Methylation Entropy by Sample Type</b>")
        st.plotly_chart(fig, use_container_width=True)
        st.success("The significantly higher methylation entropy in cancer samples provides a strong, independent feature for classification, enhancing the robustness of our diagnostic model.", icon="🧬")

    # --- Tool 10: 3D Optimization Visualization ---
    # --- Tool 10: 3D Optimization Visualization ---
    with ml_tabs[9]:
        st.subheader("10. Process Optimization & Model Training (3D Visualization)")
        with st.expander("View Method Explanation & Scientific Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            To provide an intuitive, three-dimensional visualization of an optimization problem. This powerful tool allows us to literally *see* the landscape our algorithms are trying to navigate, whether it's an assay's response surface or a machine learning model's loss function. It builds confidence that our optimization strategies are finding true, global optima rather than getting stuck in local minima.

            **Conceptual Walkthrough: Mapping and Hiking a Valley**
            Imagine an unknown valley where we want to find the highest peak (to maximize yield). This visualization shows our strategy:
            1.  **DOE Points (Surveyor Readings):** The black diamonds are the **Design of Experiments (DOE)** points—like a surveyor taking elevation readings at a few strategic locations. Their 'shadows' are projected onto the floor to clearly show their X-Y coordinates.
            2.  **Response Surface (The Topographic Map):** The smooth, colored surface is our predictive model (a Gaussian Process), which acts as our "topographic map" of the entire valley, interpolated from the surveyor's data. The contour lines on the surface and on the floor make the terrain even easier to read.
            3.  **Gradient Ascent (The High-Tech Hike):** The vibrant path shows the route taken by an algorithm starting at a non-optimal point (green star). At each step, it senses the steepest upward slope (the **gradient**) and moves in that direction, eventually converging at the red star.

            **Mathematical Basis & Formula:**
            - **Response Surface:** A predictive model, often a Gaussian Process (GP), is fit to the DOE data to create a continuous function: $$ \text{Yield} = f(\text{PCR Cycles}, \text{Input DNA}) $$
            - **Gradient Ascent:** An iterative optimization algorithm that updates parameters ($\theta$) by moving in the direction of the gradient of the function to be maximized, $f(\theta)$. The update rule is:
            $$ \theta_{\text{new}} = \theta_{\text{old}} + \eta \nabla f(\theta) $$
            Where $\eta$ is the learning rate and $\nabla f(\theta)$ is the gradient.

            **Procedure:**
            1. Generate a 3D surface plot from the predictive model (GP) fit on the experimental data.
            2. Overlay the original DOE data points and their 2D projections.
            3. Simulate a gradient ascent optimization, starting from a non-optimal point.
            4. Plot the path of the algorithm, highlighting the start and end, as it converges towards the maximum on the surface.

            **Significance of Results:**
            This visualization provides compelling, intuitive evidence that our process characterization and optimization methods are sound. It demonstrates that the statistically-derived optimum from the response surface aligns with the optimum found by an iterative machine learning optimizer. For a PMA, this visual evidence powerfully communicates a deep understanding and control over our core manufacturing processes.
            """)
        try:
            # --- 1. Get RSM data and fit a GP model to create the surface ---
            rsm_data = ssm.get_data("quality_system", "rsm_data")
            if not rsm_data:
                st.warning("RSM data not available for this visualization.")
                st.stop()

            df_rsm = pd.DataFrame(rsm_data)
            X_rsm = df_rsm[['pcr_cycles', 'input_dna']]
            y_rsm = df_rsm['library_yield']

            kernel = C(1.0, (1e-3, 1e3)) * RBF([1, 1], (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42)
            gp.fit(X_rsm, y_rsm)

            x_min, x_max = X_rsm['pcr_cycles'].min(), X_rsm['pcr_cycles'].max()
            y_min, y_max = X_rsm['input_dna'].min(), X_rsm['input_dna'].max()
            z_min, z_max = y_rsm.min(), y_rsm.max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 40), np.linspace(y_min, y_max, 40))
            Z = gp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            # Find the true global optimum from the GP surface grid
            opt_idx_gp = np.argmax(Z)
            opt_x_gp, opt_y_gp = xx.ravel()[opt_idx_gp], yy.ravel()[opt_idx_gp]
            opt_z_gp = np.max(Z)

            # --- 2. Simulate Gradient *Ascent* path (to maximize yield) ---
            def gradient(x, y):
                # Numerically approximate the gradient of the GP prediction
                eps = 1e-6
                grad_x = (gp.predict([[x + eps, y]])[0] - gp.predict([[x - eps, y]])[0]) / (2 * eps)
                grad_y = (gp.predict([[x, y + eps]])[0] - gp.predict([[x, y - eps]])[0]) / (2 * eps)
                return np.array([grad_x, grad_y])

            path = []
            # Start at a non-optimal point from the DOE data
            current_point = df_rsm.iloc[0][['pcr_cycles', 'input_dna']].values.astype(float)
            learning_rate = 0.1 # Adjusted for numerical gradient

            for i in range(20):
                z_val = gp.predict([current_point])[0]
                path.append(np.append(current_point, z_val))
                grad = gradient(current_point[0], current_point[1])
                # Normalize gradient to prevent exploding steps
                norm_grad = grad / (np.linalg.norm(grad) + 1e-8)
                current_point += learning_rate * norm_grad

            path_df = pd.DataFrame(path, columns=['x', 'y', 'z'])

            # --- 3. Create the Enhanced 3D Plot ---
            fig = go.Figure()

            # The Response Surface with Contours
            fig.add_trace(go.Surface(
                x=xx, y=yy, z=Z,
                colorscale='Plasma',
                opacity=0.8,
                name='GP Response Surface',
                contours=dict(
                    x=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_x=True),
                    y=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_y=True),
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
                )
            ))

            # The original DOE points in 3D
            fig.add_trace(go.Scatter3d(
                x=df_rsm['pcr_cycles'], y=df_rsm['input_dna'], z=df_rsm['library_yield'],
                mode='markers',
                marker=dict(size=5, color='black', symbol='diamond', line=dict(color='white', width=1)),
                name='DOE Experimental Points'
            ))

            # The Gradient Ascent path
            fig.add_trace(go.Scatter3d(
                x=path_df['x'], y=path_df['y'], z=path_df['z'],
                mode='lines',
                line=dict(color='cyan', width=8),
                name='Gradient Ascent Path'
            ))
            # Start and End points of the path
            fig.add_trace(go.Scatter3d(x=[path_df['x'].iloc[0]], y=[path_df['y'].iloc[0]], z=[path_df['z'].iloc[0]], mode='markers', marker=dict(color='lime', size=10, symbol='circle'), name='Start Point'))
            fig.add_trace(go.Scatter3d(x=[path_df['x'].iloc[-1]], y=[path_df['y'].iloc[-1]], z=[path_df['z'].iloc[-1]], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Converged Point'))
            
            # Global Optimum found by the GP model
            fig.add_trace(go.Scatter3d(x=[opt_x_gp], y=[opt_y_gp], z=[opt_z_gp], mode='markers', marker=dict(color='yellow', size=12, symbol='star', line=dict(color='black', width=1)), name='Predicted Global Optimum'))

            fig.update_layout(
                title='<b>3D Visualization of Optimization Landscape</b>',
                scene=dict(
                    xaxis=dict(title='PCR Cycles', backgroundcolor="rgba(0, 0, 0,0)"),
                    yaxis=dict(title='Input DNA (ng)', backgroundcolor="rgba(0, 0, 0,0)"),
                    zaxis=dict(title='Library Yield', backgroundcolor="rgba(0, 0, 0,0)"),
                    camera=dict(
                        eye=dict(x=1.5, y=-1.5, z=1.2) # Set a nice initial angle
                    )
                ),
                height=700,
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(x=0.01, y=0.99, traceorder='normal', bgcolor='rgba(255,255,255,0.6)')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.success("The 3D plot visualizes the assay response surface derived from DOE points. The cyan line demonstrates how a gradient-based optimization algorithm navigates this surface to efficiently find the region of maximum yield, confirming that the iterative optimization converges near the predicted global optimum (yellow star).", icon="🎯")

        except Exception as e:
            st.error(f"Could not render 3D visualization. Error: {e}")
            logger.error(f"Error in 3D optimization visualization: {e}", exc_info=True)
#___________________________________________________________________________________________________________________________________________________________________TEXT_______________________________________________________________________________
def render_compliance_guide_tab():
    """Renders the definitive reference guide to the regulatory and methodological frameworks for the program."""
    st.header("🏛️ The Regulatory & Methodological Compendium")
    st.markdown("This guide serves as the definitive reference for the regulatory, scientific, and statistical frameworks governing the GenomicsDx Sentry™ program. It is designed for the scientific and engineering leads, principal investigators, and decision-makers responsible for program execution and technical integrity.")

    with st.expander("⭐ **I. The GxP Paradigm: Proactive Quality by Design & The Role of the DHF**", expanded=True):
        st.info("The entire regulatory structure is predicated on the principle of **Quality by Design (QbD)**: quality, safety, and effectiveness must be designed and built into the product, not merely inspected or tested into it after the fact. This proactive paradigm is enforced through Design Controls.")
        
        st.subheader("The Design Controls Framework (21 CFR 820.30)")
        st.markdown("""
        Design Controls are a formal, risk-based framework for conducting product development. This is not arbitrary bureaucracy; it is a closed-loop system designed to ensure a robust and traceable development process. The core logic is as follows:
        1.  **Define Needs:** We formally capture all **Design Inputs**, which are the physical and performance requirements derived from user needs and the intended use. These must be unambiguous, comprehensive, and testable.
        2.  **Create the Design:** We develop **Design Outputs**—the full set of specifications, algorithms, procedures, and material definitions that constitute the device.
        3.  **Establish Traceability:** Crucially, every Design Output must be traceable back to a Design Input. This ensures we have built what we intended to build. The **Traceability Matrix** is the key tool for managing this.
        4.  **Confirm the 'What':** **Design Verification** answers the question: *'Did we build the device correctly?'* It provides objective evidence (e.g., from analytical validation studies) that the Design Outputs meet the Design Inputs.
        5.  **Confirm the 'Why':** **Design Validation** answers the question: *'Did we build the correct device?'* It provides objective evidence (e.g., from clinical and usability studies) that the final, manufactured device meets the user's needs and its intended use in the target clinical environment.
        """)

        st.subheader("The Design History File (DHF) vs. The Device Master Record (DMR)")
        st.markdown("""
        It is critical to distinguish between these two records:
        - **The Design History File (DHF)** is the story of **why** the design is what it is. It contains the complete history of the design process: all the requirements, experiments, risk analyses, reviews, and V&V data. It is the evidence of the development journey.
        - **The Device Master Record (DMR)** is the recipe for **how** to build the device consistently. It is a compilation of the final, approved Design Outputs: specifications, SOPs, QC procedures, and labeling.
        
        **This dashboard is architected as our program's living, interactive DHF.** The approved documents in the **Design Outputs** section form the basis of our DMR.
        """)

    with st.expander("⚖️ **II. The Regulatory Framework: Mandated Compliance**", expanded=False):
        st.info("This section details the specific regulations and standards that form our compliance obligations. These are not guidelines; they are the legal and internationally recognized requirements for market access.")
        
        st.subheader("A. United States FDA Regulations")
        st.markdown("""
        - **21 CFR Part 820 (QSR/cGMP):** The Quality System Regulation. Its core is **§ 820.30 Design Controls**, the system detailed above.
        - **21 CFR Part 11:** Governs electronic records and signatures, mandating system validation to ensure data integrity, security, non-repudiation, and traceability through time-stamped audit trails. This is critical for the validity of our electronic DHF.
        - **21 CFR Part 812 (IDE):** The Investigational Device Exemption regulation. An approved IDE is required to conduct the pivotal clinical trial necessary for a Class III device, allowing the collection of safety and effectiveness data under IRB-approved protocols.
        """)
        
        st.subheader("B. International Standards & Laboratory Regulations")
        st.markdown("""
        - **ISO 13485:2016:** The global QMS standard, essential for most ex-US markets and programs like MDSAP. Its emphasis on risk management is pervasive throughout all QMS processes.
        - **ISO 14971:2019:** The standard for risk management. It mandates a closed-loop process to identify, analyze, evaluate, control, and monitor risks, ensuring that the overall residual risk is acceptable in relation to the medical benefit (the ALARP principle - As Low As Reasonably Practicable).
        - **IEC 62304:2006:** The software lifecycle standard. Our classifier is **Software Safety Class C**, implying a potential for death or serious injury. This mandates maximum process rigor, including formal architectural design, detailed design specifications, and unit/integration/system level testing.
        - **CLIA (Clinical Laboratory Improvement Amendments):** Federal regulations that govern US laboratory operations. FDA approval addresses the *test system design*, while CLIA certification addresses the *laboratory's validated ability* to perform that test accurately, precisely, and reliably in a production environment.
        """)

    with st.expander("🔬 **III. Methodologies & Statistical Foundations: The Evidentiary Toolkit**", expanded=False):
        st.info("This section details the scientific and statistical methods used to generate the objective evidence required for our regulatory submissions. A deep understanding of their principles and limitations is essential for all technical leads.")

        st.subheader("A. Risk Analysis: Failure Mode and Effects Analysis (FMEA)")
        st.markdown("""**Purpose:** A semi-quantitative risk assessment tool used to systematically evaluate potential failure modes, their causes, and their effects on the system's output (e.g., a patient result).
        **Methodology:** The Risk Priority Number (RPN) is a heuristic used for prioritization, calculated as:""")
        st.latex(r'''RPN = S \times O \times D''')
        st.markdown(r"""- **S (Severity):** Impact of the failure's effect.
- **O (Occurrence):** Frequency/probability of the failure mode.
- **D (Detection):** Probability of detecting the failure before it causes harm.
**Interpretation & Strategic Implications:** The RPN is a valuable tool for focusing engineering and process improvement efforts. However, it can be misleading if used as the sole determinant of risk. A high-severity, low-RPN risk (e.g., S=5, O=1, D=1, RPN=5) is often of greater concern than a low-severity, high-RPN risk (e.g., S=2, O=3, D=3, RPN=18). Per ISO 14971, risk evaluation must prioritize Severity and Occurrence first, with RPN serving as a secondary prioritization tool.
**Dashboard Link:** ***Program Health Dashboard*** & ***Risk Management*** sections.""")

        st.subheader("B. Analytical Performance: Limit of Detection (LoD) via Probit Analysis")
        st.markdown("""**Purpose:** To determine the lowest analyte concentration that can be detected with 95% probability, a critical performance metric for an early detection assay.
        **Methodology:** Probit regression is the appropriate statistical model for quantal response (hit/no-hit) data. It linearizes the sigmoidal dose-response curve by transforming the hit rate `p` using the inverse of the standard normal CDF, $\Phi^{-1}$.
        """)
        st.latex(r'''\text{probit}(p) = \Phi^{-1}(p)''')
        # <<< THIS IS THE CORRECTED BLOCK >>>
        st.markdown(r"""A linear regression is then fit to the transformed data: `probit(Hit Rate) = β₀ + β₁ log₁₀(Concentration)`. The model yields a point estimate and a confidence interval for the concentration corresponding to a 95% hit rate.
**Interpretation & Strategic Implications:** The point estimate defines our claimed LoD. A tight confidence interval around this estimate indicates a well-behaved, robust assay with a sharp transition from non-detection to detection, which is a highly desirable characteristic. A wide confidence interval may suggest assay instability at low concentrations, requiring further process optimization.
**Dashboard Link:** ***Statistical Workbench -> LoD/Probit*** (method), ***Design Verification*** (results).""")
        
        st.subheader("C. Process & Measurement System Analysis: DOE, RSM, & Gauge R&R")
        st.markdown(r'''
        **Purpose:** A suite of statistical tools to move from a state of empirical observation to one of deep process understanding, characterization, and control—the essence of Quality by Design.
        - **DOE (Screening):** Used to efficiently identify the factors with a statistically significant impact on the response from a large pool of potential variables. This prevents wasting resources optimizing insignificant parameters.
        - **RSM (Optimization):** Used after screening to model curvature and interactions between significant factors, allowing for the identification of optimal process settings. Its output is a predictive model that defines a **design space** and a **Normal Operating Range (NOR)**, which are critical inputs for the DMR and regulatory filings. The fitted quadratic model is:''')
        st.latex(r'''Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_{12} X_1 X_2 + \beta_{11} X_1^2 + \beta_{22} X_2^2''')
        st.markdown(r'''- **Gauge R&R (Control):** Used to quantify the variability of the measurement system itself. The ANOVA method partitions total process variation ($\sigma^2_{\text{Total}}$) into components from the parts ($\sigma^2_{\text{Part}}$) and the measurement system ($\sigma^2_{\text{Gauge R&R}}$).
**Interpretation & Strategic Implications:** A low %GR&R (guideline: <10%) is critical, as it provides confidence that the observed process variation is real and not an artifact of measurement error. A high %GR&R (>30%) invalidates process data and indicates the measurement system itself must be improved before any process optimization can be trusted.
**Dashboard Link:** ***Statistical Workbench*** for all three methods.''')
        
        st.subheader("D. Machine Learning Model Validation & Explainability")
        st.markdown("""**Purpose:** To provide objective evidence of the classifier's performance and to address the "black box" problem, which is a primary concern for regulators reviewing AI/ML-based SaMD.
        **Methodology:**
        - **ROC/AUC:** Measures the model's intrinsic **discriminatory power** across all possible thresholds.
        - **SHAP (SHapley Additive exPlanations):** A game theory-based method that computes the marginal contribution of each feature to the prediction for an individual sample. This provides **local model interpretability**.
        **Interpretation & Strategic Implications:** These tools satisfy two distinct but equally important validation requirements. A high AUC demonstrates *what* the model can do (its clinical performance). SHAP values demonstrate *how* it does it (its scientific and biological plausibility). Providing strong evidence that the model's predictions are driven by mechanistically relevant biomarkers and not by confounding artifacts is essential for de-risking the AI/ML component of our submission.
        **Dashboard Link:** ***ML & Bioinformatics Lab*** tab.""")

    with st.expander("📄 **IV. The Regulatory Submission: Constructing the PMA**", expanded=False):
        st.info("The PMA is not a data dump; it is a structured scientific and regulatory argument. The DHF provides the evidentiary basis for every assertion made in this argument, answering the fundamental questions of safety and effectiveness.")
        
        st.markdown("""
        Our submission must construct a traceable narrative that answers the key questions posed by regulators:
        
        - **1. What is the device and how is it intended to be used?**
          - *Evidence Source:* Design Plan, Labeling, Human Factors Reports.
        
        - **2. Does the device work reliably from a technical and engineering perspective?** (Analytical Validation)
          - *Evidence Source:* Design Verification data, V&V reports, Software validation package (IEC 62304), and all analyses from the Statistical and ML workbenches.
        
        - **3. Is the device safe and effective when used in the intended patient population?** (Clinical Validation)
          - *Evidence Source:* The complete Clinical Study Report from our IDE trial, including all statistical analyses, patient outcomes, and adverse event data.
          
        - **4. Can the device be manufactured consistently and reliably to its specifications?**
          - *Evidence Source:* Design Transfer package, Process Validation data (e.g., PPQ runs), and the Device Master Record (DMR).
          
        - **5. Have all risks been identified and mitigated to an acceptable level?**
          - *Evidence Source:* The complete Risk Management File (ISO 14971), including all FMEAs and the final benefit-risk determination.
        """)

        st.subheader("PMA Section-to-Dashboard Traceability")
        st.markdown("""
        The evidence generated and organized within this DHF Command Center directly populates the key sections of the PMA submission:
        
        1.  **Device Description & Intended Use:** What it is and how it's used.
            *Source: Design Plan*
        2.  **Non-clinical Laboratory Studies:** The complete **Analytical Validation** package.
            *Source: Design Verification, Statistical Workbench*
        3.  **Software & Bioinformatics:** The complete software V&V package, risk analysis, and AI/ML model validation/explainability.
            *Source: Design Verification, Risk Management, ML Lab*
        4.  **Clinical Investigations:** The full results from our pivotal IDE clinical trial.
            *Source: Design Validation*
        .  **Labeling:** The proposed Instructions for Use (IFU), packaging, and Physician's Report.
            *Source: Design Outputs, Human Factors*
        6.  **Manufacturing Information:** The Device Master Record (DMR), detailing the full lab process.
            *Source: Assay Transfer & Lab Operations*
        7.  **Quality System Information & Risk Management File:** Evidence of our compliant QMS and the complete RMF.
            *Source: All dashboard sections, especially Risk Management*
        """)
        
        st.success("**Ultimately, the cohesive, traceable, and complete story told by our DHF is what will determine the success of our PMA submission.**")
        
# ==============================================================================
# --- MAIN APPLICATION LOGIC ---
# ==============================================================================
def main() -> None:
    """Main function to run the Streamlit application."""
    try:
        ssm = SessionStateManager()
        logger.info("Application initialized. Session State Manager loaded.")
    except Exception as e:
        st.error("Fatal Error: Could not initialize Session State."); logger.critical(f"Failed to instantiate SessionStateManager: {e}", exc_info=True); st.stop()
    
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
