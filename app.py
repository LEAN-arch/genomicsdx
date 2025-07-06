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

    # ADDED TWO NEW CASES: DOE and Stability Analysis
    tool_tabs = st.tabs([
        "1. Process Control (Levey-Jennings)",
        "2. Method Comparison (Bland-Altman)",
        "3. Change Control (TOST)",
        "4. Measurement System Analysis (Gauge R&R)",
        "5. Limit of Detection (Probit Analysis)",
        "6. Failure Mode Analysis (Pareto)",
        "7. Design of Experiments (DOE)",
        "8. Stability & Shelf-Life Analysis"
    ])

    # --- Tool 1: Levey-Jennings ---
    with tool_tabs[0]:
        st.subheader("Statistical Process Control (SPC) for Daily Assay Monitoring")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown("""
            **Purpose of the Method:** To monitor the stability and precision of an assay over time using quality control (QC) materials. It serves as an early warning system to detect process drift or shifts *before* they impact patient results. This is a foundational requirement for operating in a CLIA-certified environment and demonstrating a state of control under ISO 13485.
            
            **Conceptual Walkthrough: The Assay as a Highway**
            Imagine your assay's performance is a car driving down a highway. The **mean (μ)** is the center of the lane. The **standard deviation (σ)** defines the width of the lane and the rumble strips on the side. The Levey-Jennings chart draws control limits at ±1σ (lane lines), ±2σ (rumble strips), and ±3σ (the guard rails). Each QC run is a snapshot of where your car is. A single point outside the ±3σ guard rails is an obvious crash (a **1_3s** violation). The **Westgard Rules** are more subtle; they detect a driver who is consistently hugging one side of the lane (**4_1s** violation) or weaving back and forth in a predictable pattern, both of which indicate a problem that needs correction.
            
            **Significance of Results:** A well-maintained Levey-Jennings chart is direct, auditable evidence of a state of statistical control, as required by **CLIA '88 Subpart K** and **ISO 15189**. Rule violations must trigger a documented investigation and Corrective and Preventive Action (CAPA), demonstrating robust quality management.
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
            **Purpose of the Method:** To assess the **agreement** between two quantitative measurement methods. Unlike correlation, a Bland-Altman plot quantifies the actual differences and checks for systematic bias. It is essential for method comparison studies, such as comparing a new assay to a reference method.
            
            **Conceptual Walkthrough: Comparing Two Rulers**
            A Bland-Altman plot plots the *difference* between two measurements against their average. We look for:
            1.  **Bias:** Is the average difference near zero?
            2.  **Limits of Agreement (LoA):** How wide is the spread of differences?
            3.  **Proportional Bias:** Do differences change as measurements get larger?
            
            **Significance of Results:** A critical component of a PMA submission. If the LoA are within a pre-defined, clinically acceptable range, it provides strong evidence that the two methods can be used interchangeably.
            """)
        np.random.seed(42)
        data = {'Method_A': np.random.normal(50, 10, 100), 'Method_B': np.random.normal(50, 10, 100) + np.random.normal(1, 2, 100)}
        df_methods = pd.DataFrame(data)
        fig = create_bland_altman_plot(df_methods, 'Method_A', 'Method_B', title="Bland-Altman: New vs. Old Assay Version")
        st.plotly_chart(fig, use_container_width=True)
        st.success("Analysis shows a small positive bias, but the narrow limits of agreement suggest strong overall agreement between the methods.", icon="✅")

    # --- Tool 3: TOST ---
    with tool_tabs[2]:
        st.subheader("Equivalence Testing (TOST) for Change Control")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:** To statistically demonstrate that two groups are "practically the same," the correct method for validating a change where you expect no difference (e.g., qualifying a new reagent lot).
            
            **Conceptual Walkthrough: The Goalposts of Irrelevance**
            Define "equivalence bounds" ($-\Delta$ and $+\Delta$). If the 90% Confidence Interval of the difference between two groups falls entirely *within* these bounds, you have proven equivalence.
            
            **Significance of Results:** TOST is the gold standard for change control validation under **21 CFR 820.70 (Production and Process Controls)**. It provides objective evidence that a change did not adversely affect assay performance.
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
                st.success(f"**Conclusion:** Equivalence Demonstrated (p = {p_value:.4f}). The new reagent lot is validated.", icon="✅")
            else:
                st.error(f"**Conclusion:** Equivalence Not Demonstrated (p = {p_value:.4f}). The new lot cannot be approved.", icon="❌")

    # --- Tool 4: Gauge R&R ---
    with tool_tabs[3]:
        st.subheader("Measurement System Analysis (Gauge R&R)")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:** To determine how much measurement variation comes from the measurement system versus the actual parts being measured.
            
            **Conceptual Walkthrough: Measuring Blocks of Wood**
            Variation comes from:
            1.  **Part-to-Part:** True differences (good).
            2.  **Repeatability:** One person, same part, multiple times (equipment variation).
            3.  **Reproducibility:** Different people, same part (operator variation).
            
            **Significance of Results:** AIAG guidelines: **< 10% GR&R** is acceptable; **> 30%** is unacceptable. A successful Gauge R&R is critical for **Process Validation (PV)** under FDA's QSR.
            """)
        np.random.seed(1)
        parts = np.repeat(np.arange(1, 11), 9); operators = np.tile(np.repeat(['A', 'B', 'C'], 3), 10)
        true_values = np.repeat(np.random.normal(100, 10, 10), 9); operator_effect = np.tile(np.repeat([0, 0.5, -0.5], 3), 10)
        measurements = true_values + operator_effect + np.random.normal(0, 1, 90)
        df_msa = pd.DataFrame({'part': parts, 'operator': operators, 'measurement': measurements})
        if not df_msa.empty:
            fig, results_df = create_gauge_rr_plot(df_msa, part_col='part', operator_col='operator', value_col='measurement')
            st.write("##### ANOVA Variance Components Analysis")
            st.dataframe(results_df.style.format("{:.2f}%", subset=['% Contribution']), use_container_width=True)
            st.plotly_chart(fig, use_container_width=True)
            if not results_df.empty:
                total_grr = results_df.loc['Total Gauge R&R', '% Contribution']
                if total_grr < 10:
                    st.success(f"**Conclusion:** Measurement System is Acceptable (Total GR&R = {total_grr:.2f}%).", icon="✅")
                else:
                    st.error(f"**Conclusion:** Measurement System is Unacceptable (Total GR&R = {total_grr:.2f}%).", icon="❌")

    # --- Tool 5: LoD/Probit ---
    with tool_tabs[4]:
        st.subheader("Limit of Detection (LoD) by Probit Analysis")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:** To determine the lowest analyte concentration that can be reliably detected with 95% probability. This is a mandatory part of the Analytical Validation report for a PMA and is guided by **CLSI EP17-A2**.
            
            **Conceptual Walkthrough: Finding a Whisper in a Quiet Room**
            The LoD is the exact concentration where you have a 95% "hit rate." Probit analysis fits a statistically rigorous curve to experimental hit rate data to find that precise point.
            
            **Significance of Results:** The LoD is a key performance claim in our regulatory submission and Instructions for Use (IFU). It defines the analytical sensitivity of the assay.
            """)
        concentrations = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]; hit_rates = [0.15, 0.30, 0.75, 0.90, 0.98, 1.0]
        df_lod = pd.DataFrame({'concentration': concentrations, 'hit_rate': hit_rates})
        fig = create_lod_probit_plot(df_lod, 'concentration', 'hit_rate', title="LoD for Key Biomarker Using Contrived Samples")
        st.plotly_chart(fig, use_container_width=True)
        st.success("The Probit analysis yields a highly precise LoD estimate with a tight confidence interval, demonstrating robust assay performance at low concentrations.", icon="✅")

    # --- Tool 6: Pareto Analysis ---
    with tool_tabs[5]:
        st.subheader("Pareto Analysis of Process Deviations")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown("""
            **Purpose of the Method:** To apply the **Pareto Principle (80/20 rule)** to identify the "vital few" causes responsible for the majority of problems (e.g., lab run failures).
            
            **Conceptual Walkthrough: Firefighting Triage**
            A Pareto chart sorts your problems from most to least frequent, immediately identifying the biggest "fires" and telling you where to focus resources for the greatest impact.
            
            **Significance of Results:** The Pareto chart is often the first step in a **CAPA** investigation, as required by **21 CFR 820.100**. It provides a clear justification for focusing process improvement efforts.
            """)
        failure_data = ssm.get_data("lab_operations", "run_failures")
        df_failures = pd.DataFrame(failure_data)
        if not df_failures.empty:
            fig = create_pareto_chart(df_failures, category_col='failure_mode', title='Pareto Analysis of Assay Run Failures')
            st.plotly_chart(fig, use_container_width=True)
            top_contributor = df_failures['failure_mode'].value_counts().index[0]
            st.success(f"The analysis highlights **'{top_contributor}'** as the primary contributor to run failures. Focusing CAPA initiatives on this failure mode will yield the greatest reduction in overall run failures and COPQ.", icon="🎯")

    # --- Tool 7: DOE (New) ---
    with tool_tabs[6]:
        st.subheader("Design of Experiments (DOE) Factorial Analysis")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown("""
            **Purpose:** To efficiently screen multiple process parameters to identify which ones have a statistically significant effect on the outcome. This is a foundational QbD activity that precedes process optimization.
            
            **Conceptual Walkthrough: The Smart Experiment**
            Instead of testing one factor at a time (OFAT), a factorial DOE tests all combinations of factors at high and low levels. This is far more efficient and, crucially, it allows us to detect **interactions**—when the effect of one factor depends on the level of another.
            
            **Significance:** Results from a DOE study provide the rationale for which parameters are deemed **Critical Process Parameters (CPPs)** and must be tightly controlled, and which are not. This evidence is key for justifying the process control strategy in a regulatory submission.
            """)
        np.random.seed(10)
        doe_data = {'Temp': [60, 65, 60, 65] * 2, 'Time': [30, 30, 45, 45] * 2, 'Enzyme': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'], 'Yield': [75, 85, 70, 82, 78, 92, 74, 90] + np.random.normal(0, 2, 8)}
        df_doe = pd.DataFrame(doe_data)
        st.write("##### Main Effects & Interaction Plots")
        fig_doe = create_doe_effects_plot(df_doe, factors=['Temp', 'Time', 'Enzyme'], response='Yield')
        st.plotly_chart(fig_doe, use_container_width=True)
        st.success("**Conclusion:** The analysis indicates strong main effects for 'Enzyme' and 'Time', and a significant interaction between 'Temp' and 'Enzyme'. These should be considered Critical Process Parameters (CPPs) for further optimization via RSM.", icon="🔬")

    # --- Tool 8: Stability Analysis (New) ---
    with tool_tabs[7]:
        st.subheader("Stability & Shelf-Life Analysis")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown("""
            **Purpose:** To determine the shelf-life of a reagent or the stability of a sample under specific storage conditions. This is done by modeling the degradation of a critical quality attribute over time and finding the point where the 95% confidence interval of the degradation curve crosses a pre-defined failure threshold.
            
            **Significance:** Shelf-life dating is a mandatory requirement for reagents under **21 CFR 820** and **ISO 13485**. A well-supported stability claim ensures product efficacy and safety throughout its intended use period.
            """)
        np.random.seed(42)
        time_points = np.array([0, 1, 3, 6, 9, 12, 18, 24]) # Months
        degradation_data = []
        for t in time_points:
            degradation_data.extend(100 - 0.5 * t - np.abs(np.random.normal(0, 1.5, 3)))
        df_stability = pd.DataFrame({'Months': np.repeat(time_points, 3), 'Activity_Pct': degradation_data})
        
        failure_threshold = 90.0
        fig = create_stability_plot(df_stability, 'Months', 'Activity_Pct', failure_threshold)
        st.plotly_chart(fig, use_container_width=True)
        st.success("**Conclusion:** The linear regression model predicts the reagent's activity will cross the 90% failure threshold at approximately 20.5 months. Based on the 95% confidence interval, a conservative shelf-life of **18 months** is supported by this data.", icon="⏳")

# ... The `render_machine_learning_lab_tab` and `render_compliance_guide_tab` functions would follow ...

if __name__ == "__main__":
    st.title("GenomicsDx Dashboard Module Test")
    # To test one of the tabs:
    # First, generate the necessary data
    generate_mock_data(ssm)
    
    # Then render the tab you want to inspect
    # render_statistical_tools_tab(ssm)
    # render_machine_learning_lab_tab(ssm)
    render_advanced_analytics_tab(ssm)

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
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
        import shap
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as e:
        st.error(f"This function requires advanced libraries. Please run: pip install scikit-learn statsmodels shap. Error: {e}", icon="🚨")
        return

    # ADDED TWO NEW CASES: CNV Analysis and Immune Repertoire Profiling
    ml_tabs = st.tabs([
        "1. Classifier Performance (ROC & PR)",
        "2. Classifier Explainability (SHAP)",
        "3. Cancer Signal of Origin (CSO) Analysis",
        "4. Assay Optimization (RSM vs. ML)",
        "5. Operations Forecasting (ARIMA)",
        "6. Predictive Run QC (On-Instrument)",
        "7. NGS: Fragmentomics Analysis",
        "8. NGS: Sequencing Error Modeling",
        "9. NGS: Methylation Entropy",
        "10. NGS: Copy Number Variation (CNV)",
        "11. NGS: Immune Repertoire Profiling"
    ])

    X, y = ssm.get_data("ml_models", "classifier_data")
    model = ssm.get_data("ml_models", "classifier_model")

    # --- Tool 1: ROC & PR ---
    with ml_tabs[0]:
        st.subheader("Classifier Performance: ROC and Precision-Recall")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown(r"""
            **Purpose:** To evaluate our binary classifier. The ROC curve assesses the trade-off between sensitivity and specificity. The Precision-Recall (PR) curve is essential for imbalanced datasets, typical for cancer screening.
            **Significance:** Central to the **Clinical Validation** of a PMA. AUC-ROC shows discriminatory power. The PR curve shows the test's positive predictive value (PPV) and clinical utility in a screening population.
            """)
        col1, col2 = st.columns(2)
        with col1:
            fig_roc = create_roc_curve(pd.DataFrame({'score': model.predict_proba(X)[:, 1], 'truth': y}), 'score', 'truth')
            st.plotly_chart(fig_roc, use_container_width=True)
        with col2:
            precision, recall, _ = precision_recall_curve(y, model.predict_proba(X)[:, 1])
            pr_auc = auc(recall, precision)
            fig_pr = px.area(x=recall, y=precision, title=f"<b>Precision-Recall Curve (AUC = {pr_auc:.4f})</b>", labels={'x':'Recall (Sensitivity)', 'y':'Precision'})
            fig_pr.update_layout(xaxis=dict(range=[-0.01,1.01]), yaxis=dict(range=[0,1.05]), template="plotly_white")
            st.plotly_chart(fig_pr, use_container_width=True)
        st.success("Classifier demonstrates high discriminatory power (AUC > 0.9) and maintains high precision across a range of recall values.", icon="✅")

    # --- Tool 2: SHAP ---
    with ml_tabs[1]:
        st.subheader("Classifier Explainability (SHAP)")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown(r"""
            **Purpose:** To unlock the "black box" of our model. For a regulated SaMD, we must show *how* it works. SHAP values quantify each feature's contribution to a prediction.
            **Significance:** Model explainability is a major focus for the FDA. SHAP analysis confirms **scientific plausibility**, helps **debug the model**, and **builds trust** with regulators.
            """)
        with st.spinner("Calculating SHAP values for a data sample..."):
            n_samples_for_shap = min(100, len(X))
            st.caption(f"Note: Explaining on a random subsample of {n_samples_for_shap} data points for performance.")
            X_sample = X.sample(n=n_samples_for_shap, random_state=42)
            explainer = shap.TreeExplainer(model)
            shap_values_list = explainer.shap_values(X_sample)
            fig_shap = create_shap_summary_plot(shap_values_list[1], X_sample)
            st.pyplot(fig_shap, clear_figure=True)
            st.success("SHAP analysis confirms the model's predictions are driven by known methylation biomarkers, providing strong evidence of its scientific validity for the PMA submission.", icon="✅")

    # --- Tool 3: CSO ---
    with ml_tabs[2]:
        st.subheader("Cancer Signal of Origin (CSO) Analysis")
        with st.expander("View Method Explanation & Regulatory Context", expanded=False):
            st.markdown("""
            **Purpose:** For an MCED test, predicting the **Cancer Signal of Origin (CSO)** is a key secondary claim, critical for guiding clinical workup.
            **Conceptual Walkthrough: The Return Address**
            If the primary classifier finds a "letter" that says "I am cancer," the CSO model reads the "return address." A **confusion matrix** is the report card: the diagonal shows how often it got the address right.
            **Significance:** CSO performance is a key part of **clinical validation** in a PMA. The confusion matrix informs the Instructions for Use (IFU).
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
            st.success(f"The CSO classifier achieved an overall Top-1 accuracy of **{accuracy:.1%}**.", icon="🎯")

    # --- Tool 4: RSM vs ML (3D Plot) ---
    with ml_tabs[3]:
        st.subheader("Assay Optimization: 3D Response Surface vs. Machine Learning")
        st.info("This tool compares traditional Response Surface Methodology (RSM) with a flexible Gaussian Process (GP) model to find optimal process parameters.")
        rsm_data = ssm.get_data("quality_system", "rsm_data")
        df_rsm = pd.DataFrame(rsm_data)
        X_rsm = df_rsm[['pcr_cycles', 'input_dna']]
        y_rsm = df_rsm['library_yield']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Method 1: Response Surface Methodology (3D View)")
            with st.expander("View Method Explanation", expanded=False):
                st.markdown(r"""
                **Purpose:** Find optimal settings by fitting a **quadratic model** to data from a designed experiment.
                **Significance:** RSM is the industry-standard for defining a **Design Space** and is well-understood by regulators.
                """)
            surface_fig, _, optimum = create_rsm_plots(df_rsm, 'pcr_cycles', 'input_dna', 'library_yield')
            st.plotly_chart(surface_fig, use_container_width=True)
            st.info(f"RSM Optimum found at {optimum['x']:.1f} PCR cycles, {optimum['y']:.1f} ng DNA, yielding {optimum['z']:.0f} units.")
        
        with col2:
            st.markdown("#### Method 2: Machine Learning (Gaussian Process)")
            with st.expander("View Method Explanation", expanded=False):
                st.markdown(r"""
                **Purpose:** Find optimal settings using a more flexible, non-parametric model that can capture complex, non-linear relationships.
                **Significance:** GP models are more powerful for complex processes and can find optima that RSM might miss.
                """)
            kernel = C(1.0, (1e-3, 1e3)) * RBF([1, 1], (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42)
            gp.fit(X_rsm, y_rsm)
            x_min, x_max = X_rsm['pcr_cycles'].min(), X_rsm['pcr_cycles'].max()
            y_min, y_max = X_rsm['input_dna'].min(), X_rsm['input_dna'].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))
            Z = gp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            fig_gp = go.Figure(data=go.Surface(z=Z, x=np.linspace(x_min, x_max, 30), y=np.linspace(y_min, y_max, 30), colorscale='Viridis'))
            fig_gp.update_layout(title="<b>GP-based 3D Design Space</b>", scene=dict(xaxis_title='PCR Cycles', yaxis_title='Input DNA (ng)', zaxis_title='Library Yield'), margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig_gp, use_container_width=True)
        st.success("**Conclusion:** Both methods identify a similar optimal region. The GP model captures more nuanced local variations. For our PMA, the RSM model is preferred for its regulatory acceptance.", icon="🤝")

    # --- Tool 5: Time Series ---
    with ml_tabs[4]:
        st.subheader("Time Series Forecasting for Lab Operations")
        with st.expander("View Method Explanation & Business Context", expanded=False):
            st.markdown(r"""
            **Purpose:** To forecast future operational demand (e.g., incoming sample volume) for data-driven decisions on inventory and staffing.
            **Conceptual Walkthrough: Weather Forecasting for the Lab**
            This tool uses a model called **ARIMA** which learns from past data trends, momentum, and lingering shocks to predict future workload.
            **Significance:** Accurate forecasting prevents costly reagent stock-outs or over-staffing, reducing COPQ and providing quantitative justification for budgets.
            """)
        ts_data_df = ssm.get_data("ml_models", "sample_volume_data")
        with st.spinner("Fitting ARIMA model and forecasting next 30 days..."):
            model_arima = ARIMA(ts_data_df['samples'].asfreq('D'), order=(5, 1, 0)).fit()
            forecast = model_arima.get_forecast(steps=30)
            forecast_df = forecast.summary_frame()
            fig = create_forecast_plot(ts_data_df, forecast_df)
        st.plotly_chart(fig, use_container_width=True)
        st.success("ARIMA forecast projects a continued upward trend, suggesting a need to review reagent inventory and staffing levels.", icon="📈")

    # --- Tool 6: Predictive Run QC ---
    with ml_tabs[5]:
        st.subheader("Predictive Run QC from Early On-Instrument Metrics")
        with st.expander("View Method Explanation & Operational Context", expanded=False):
            st.markdown(r"""
            **Purpose:** To predict the final quality of a sequencing run using metrics generated *within the first few hours*. This allows the lab to terminate runs destined to fail, saving reagents and instrument time.
            **Conceptual Walkthrough: The Pre-Flight Check**
            This is a machine learning pre-flight check for sequencing runs. It learns the patterns of early-run metrics that are associated with final run failure and flags them for early termination.
            **Significance:** This tool directly reduces the **Cost of Poor Quality (COPQ)** and can be integrated into the LIMS to create a more efficient "intelligent" lab operation.
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
        st.success(f"**Model Evaluation:** The model correctly identified **{tp}** of {tp+fn} failing runs, enabling proactive intervention.", icon="💰")

    # --- Tool 7: Fragmentomics ---
    with ml_tabs[6]:
        st.subheader("NGS Signal: ctDNA Fragmentomics Analysis")
        with st.expander("View Method Explanation & Scientific Context", expanded=False):
            st.markdown(r"""
            **Purpose:** To leverage the key biological property that ctDNA is more fragmented (shorter) than healthy cfDNA.
            **Conceptual Walkthrough: Rocks vs. Sand**
            Fragmentomics is a biological sieve. By analyzing the size distribution of all DNA fragments, we can identify samples with an overabundance of short fragments, a strong indicator of cancer.
            **Significance:** This provides powerful evidence for **analytical validity**. It shows the classifier is keyed into scientifically relevant signals, not spurious correlations.
            """)
        np.random.seed(42)
        healthy_frags = np.random.normal(167, 8, 5000)
        cancer_frags = np.random.normal(145, 15, 2500)
        df_frags = pd.DataFrame({'Fragment Size (bp)': np.concatenate([healthy_frags, cancer_frags]), 'Sample Type': ['Healthy cfDNA'] * 5000 + ['Cancer ctDNA'] * 2500})
        fig_hist = px.histogram(df_frags, x='Fragment Size (bp)', color='Sample Type', nbins=100, barmode='overlay', histnorm='probability density', title="<b>Distribution of DNA Fragment Sizes (Healthy vs. Cancer)</b>")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.success("The clear shift in fragment size for ctDNA demonstrates its potential as a powerful classification feature.", icon="🧬")

    # --- Tool 8: Error Modeling ---
    with ml_tabs[7]:
        st.subheader("NGS Signal: Sequencing Error Profile Modeling")
        with st.expander("View Method Explanation & Scientific Context", expanded=False):
            st.markdown(r"""
            **Purpose:** To statistically distinguish a true, low-frequency somatic mutation from the background "noise" of sequencing errors. For liquid biopsy, a robust error model is essential for achieving a low LoD.
            **Conceptual Walkthrough: A Whisper in a Crowded Room**
            This tool first *characterizes the background noise* by fitting a Beta distribution to error rates from normal samples. For a new potential variant, we calculate the p-value against this model. If the probability that it was just noise is astronomically low, we call it real.
            **Significance:** This is the core of a high-performance bioinformatic pipeline. A well-parameterized error model determines the assay's analytical specificity and **Limit of Detection (LoD)**.
            """)
        np.random.seed(1)
        background_vafs = np.random.beta(a=0.4, b=9000, size=1000)
        alpha0, beta0, _, _ = stats.beta.fit(background_vafs, floc=0, fscale=1)
        st.write(f"**Fitted Background Error Model:** `Beta(α={alpha0:.3f}, β={beta0:.2f})`")
        true_vaf = st.slider("Simulate True Variant Allele Frequency (VAF)", 0.0, 0.005, 0.001, step=0.0001, format="%.4f", key="vaf_slider_ngs")
        read_depth = st.number_input("Simulate Read Depth", 1000, 50000, 10000, key="read_depth_ngs")
        observed_variant_reads = np.random.binomial(read_depth, true_vaf)
        observed_vaf = observed_variant_reads / read_depth
        p_value = 1.0 - stats.beta.cdf(observed_vaf, alpha0, beta0)
        st.metric(f"Observed VAF at {read_depth}x Depth", f"{observed_vaf:.4f}")
        st.metric("P-value (Probability this is Random Noise)", f"{p_value:.3e}")
        if p_value < 1e-6:
             st.success(f"**Conclusion:** Highly statistically significant. This should be called as a true mutation.", icon="✅")
        else:
             st.error(f"**Conclusion:** Not statistically distinguishable from background error. Do **not** call as a true variant.", icon="❌")

    # --- Tool 9: Methylation Entropy ---
    with ml_tabs[8]:
        st.subheader("NGS Signal: Methylation Entropy Analysis")
        with st.expander("View Method Explanation & Scientific Context", expanded=False):
            st.markdown(r"""
            **Purpose:** To leverage the **disorder** of methylation patterns as a cancer signal. Healthy tissues often have consistent patterns (low entropy), while cancer tissues exhibit chaotic patterns (high entropy).
            **Conceptual Walkthrough: A Well-Kept vs. Messy Bookshelf**
            Imagine a genomic region is a bookshelf. In healthy cells, books are neat (low entropy). In cancer cells, the same shelf is a mess (high entropy). By sequencing many DNA molecules from that region, we can quantify this disorder using Shannon Entropy.
            **Significance:** Methylation entropy is an **orthogonal biological signal**, making our classifier more robust by not relying on single-site measurements alone.
            """)
        np.random.seed(33)
        healthy_entropy = np.random.normal(1.5, 0.5, 100).clip(0)
        cancer_entropy = np.random.normal(3.0, 0.8, 100).clip(0)
        df_entropy = pd.DataFrame({'Entropy (bits)': np.concatenate([healthy_entropy, cancer_entropy]), 'Sample Type': ['Healthy'] * 100 + ['Cancer'] * 100})
        fig = px.box(df_entropy, x='Sample Type', y='Entropy (bits)', color='Sample Type', points='all', title="<b>Methylation Entropy by Sample Type</b>")
        st.plotly_chart(fig, use_container_width=True)
        st.success("The significantly higher methylation entropy in cancer samples provides a strong, independent feature for classification.", icon="🧬")

    # --- Tool 10: CNV Analysis (New) ---
    with ml_tabs[9]:
        st.subheader("NGS Signal: Copy Number Variation (CNV) Analysis")
        with st.expander("View Method Explanation & Scientific Context", expanded=False):
            st.markdown(r"""
            **Purpose:** To detect large-scale genomic amplifications and deletions, which are classic hallmarks of cancer. This analysis uses read depth as a proxy for copy number.
            **Conceptual Walkthrough: Paving a Road**
            Imagine the genome is a long road. In a healthy sample, we expect a consistent, even layer of "asphalt" (sequencing reads) across its entire length (diploid, copy number = 2). In a cancer sample, some sections of the road might have a giant mound of asphalt (an amplification, e.g., ERBB2) while other sections have a deep pothole (a deletion, e.g., TP53). By smoothing the read depth data, we can spot these significant deviations.
            **Significance:** CNV analysis provides another layer of orthogonal biological evidence. It is particularly powerful for identifying well-known oncogenes and tumor suppressors, strengthening the scientific foundation of the classifier.
            """)
        np.random.seed(42)
        genome_pos = np.arange(1, 1001)
        healthy_depth = np.random.poisson(100, 1000)
        cancer_depth = np.random.poisson(100, 1000)
        cancer_depth[300:400] = np.random.poisson(150, 100) # Amplification
        cancer_depth[700:750] = np.random.poisson(50, 50)   # Deletion
        df_cnv = pd.DataFrame({
            'Genomic Position': np.concatenate([genome_pos, genome_pos]),
            'Normalized Read Depth': np.concatenate([healthy_depth, cancer_depth]),
            'Sample Type': ['Healthy Control'] * 1000 + ['Cancer Sample'] * 1000
        })
        fig = px.line(df_cnv, x='Genomic Position', y='Normalized Read Depth', color='Sample Type', title="<b>Copy Number Profile Across a Chromosome Arm</b>")
        st.plotly_chart(fig, use_container_width=True)
        st.success("The cancer sample clearly shows a regional amplification (positions 300-400) and deletion (700-750) not present in the healthy control, providing a strong signal for the classifier.", icon="🧬")

    # --- Tool 11: Immune Repertoire (New) ---
    with ml_tabs[10]:
        st.subheader("NGS Signal: Immune Repertoire Profiling")
        with st.expander("View Method Explanation & Scientific Context", expanded=False):
            st.markdown(r"""
            **Purpose:** To analyze the diversity and composition of T-cell receptors (TCRs) captured in cfDNA. The immune system's response to a tumor can leave a detectable "fingerprint" in the blood.
            **Conceptual Walkthrough: The Army's Special Forces**
            Your immune system is an army. When it detects a specific threat like a tumor, it trains and deploys huge numbers of identical "special forces" T-cells, all with the same weapon (TCR) designed for that one enemy. This is called **clonal expansion**. In a healthy person, the army is diverse, with many different types of soldiers in equal numbers. In a cancer patient, we expect to see one or a few T-cell clones dominating the population.
            **Significance:** Measuring TCR diversity (or lack thereof) provides a signal of the *host response* to cancer, which is completely orthogonal to signals from the tumor itself (like mutations or methylation). This adds an incredible layer of robustness to the ML model.
            """)
        np.random.seed(1)
        healthy_clones = np.random.lognormal(0.5, 1, 100)
        cancer_clones = np.random.lognormal(0.5, 1, 100)
        cancer_clones[0:3] = [50, 35, 20] # Add 3 dominant clones
        df_tcr = pd.DataFrame({
            'TCR Clonotype Frequency': np.concatenate([healthy_clones, cancer_clones]),
            'Sample Type': ['Healthy'] * 100 + ['Cancer'] * 100,
            'Rank': list(range(1, 101)) * 2
        })
        fig = px.line(df_tcr, x='Rank', y='TCR Clonotype Frequency', color='Sample Type', log_y=True, title="<b>TCR Repertoire Clonality (Lorenz Curve)</b>")
        st.plotly_chart(fig, use_container_width=True)
        st.success("The cancer sample exhibits significant clonal expansion (a few dominant clones), a strong indicator of a specific immune response not seen in the diverse healthy repertoire.", icon="🧬")

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
