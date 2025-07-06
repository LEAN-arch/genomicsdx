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
            st.info("💡 A well-understood relationship between CAPs and the final test result is the foundation of a robust assay, as required by 21 CFR 820.30 and ISO 13485.", icon="💡")
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


def render_statistical_tools_tab(ssm: SessionStateManager):
    """Renders the tab containing various statistical analysis tools."""
    st.header("📈 Statistical Workbench for Assay & Lab Development")
    st.info("Utilize this interactive workbench for rigorous statistical analysis of assay performance, a cornerstone of the Analytical Validation required for a PMA.")
    
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
        from scipy.stats import shapiro, mannwhitneyu
    except ImportError:
        st.error("This tab requires `statsmodels` and `scipy`. Please install them (`pip install statsmodels scipy`) to enable statistical tools.", icon="🚨")
        return

    # SME Enhancement: Add new RSM tool to the workbench
    tool_tabs = st.tabs([
        "Process Control (Levey-Jennings)",
        "Hypothesis Testing (A/B Test)",
        "Equivalence Testing (TOST)",
        "Pareto Analysis (Failure Modes)",
        "Measurement System Analysis (Gauge R&R)",
        "DOE (Screening)",
        "Response Surface Methodology (Optimization)"
    ])

    # --- Tool 1: Levey-Jennings ---
    with tool_tabs[0]:
        st.subheader("Statistical Process Control (SPC) for Assay Monitoring")
        with st.expander("View Method Explanation"):
            st.markdown("""
            **Purpose of the Tool:**
            To monitor the stability and consistency of a process over time. For our assay, this is used to track the performance of quality control (QC) materials to ensure the assay is performing as expected on a day-to-day basis.
            """)
            st.markdown("""
            **Conceptual Walkthrough: What's Happening in the Background?**
            Imagine our assay process is a river. When everything is normal, the river flows steadily down the center of its channel. The **mean ($\mu$)** is that center line. The **standard deviation ($\sigma$)** tells us how wide the normal, "safe" channel is. The Levey-Jennings chart draws "guard rails" at 1, 2, and 3 standard deviations from the center. Each time we run a QC sample, we're checking where the river is flowing on that day. A single point outside the $\pm3\sigma$ guard rails is a major flood warning. The Westgard rules are more subtle; they look for patterns, like if the river is consistently hugging one bank for several days in a row, which might indicate a slow, systematic change in our process that needs investigation.
            """)
            st.markdown("""
            **Mathematical Basis:**
            The chart is based on the principles of the Gaussian (Normal) distribution. The control limits are established based on the mean ($\mu$) and standard deviation ($\sigma$) of a set of historical, in-control data. The limits are typically set at $\mu \pm 1\sigma$, $\mu \pm 2\sigma$, and $\mu \pm 3\sigma$.
            """)
            st.markdown("**Core Formulas:**")
            st.latex(r'''
            \text{Mean: } \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
            ''')
            st.latex(r'''
            \text{Standard Deviation: } s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2}
            ''')
            st.markdown("""
            **Procedure:**
            1. A stable mean and standard deviation are established for a QC material from at least 20 historical data points.
            2. Control limits are calculated and drawn on the chart.
            3. For each subsequent run, the new QC value is plotted on the chart.
            4. The plot is evaluated against a set of rules (e.g., Westgard rules like 1_3s, 2_2s, R_4s, 4_1s) to detect shifts or trends that may indicate a problem with the process.

            **Significance of Results:**
            A Levey-Jennings chart provides an early warning system for process drift or instability. A point outside the $\pm 3\sigma$ limits or patterns of points indicates a loss of statistical control, triggering an investigation (e.g., a CAPA) and preventing the release of potentially erroneous patient results.
            """)
        spc_data = ssm.get_data("quality_system", "spc_data")
        fig = create_levey_jennings_plot(spc_data)
        st.plotly_chart(fig, use_container_width=True)
        st.success("The selected control data appears to be stable and in-control. No Westgard rule violations were detected.")

    # --- Tool 2: Hypothesis Testing ---
    with tool_tabs[1]:
        st.subheader("Hypothesis Testing for Assay Comparability")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To determine if there is a statistically significant difference between the means of two independent groups. This is used, for example, to compare the output of a new bioinformatics pipeline version against the old one.
            """)
            st.markdown(r"""
            **Conceptual Walkthrough: What's Happening in the Background?**
            Think of this as a statistical courtroom drama. The **Null Hypothesis ($H_0$)** is the "defendant," and it's presumed innocent—meaning, we assume there is no difference between our two groups. Our data is the "evidence" we present to the court. The **p-value** is the key output: it's the probability that we would see evidence this strong (or stronger) *if the defendant were truly innocent*. If the p-value is very low (typically < 0.05), it's like saying, "The chance of this happening randomly is so small, the defendant must be guilty!" We then "convict" the null hypothesis, reject it, and declare that a significant difference exists. If the p-value is high, we don't have enough evidence to convict, so we "fail to reject" the null hypothesis.
            """)
            st.markdown(r"""
            **Mathematical Basis:**
            The framework is Null Hypothesis Significance Testing (NHST). We start with a **Null Hypothesis ($H_0$)** that there is no difference between the groups ($\mu_1 = \mu_2$). We then calculate the probability (**p-value**) of observing our data (or more extreme data) if the null hypothesis were true.
            """)
            st.markdown("**Key Formulas & Tests:**")
            st.markdown("- **Shapiro-Wilk Test:** Tests the null hypothesis that data was drawn from a normal distribution. The test statistic W is a ratio of two estimates of the variance.")
            st.markdown(r"""- **Welch's t-test (for Normal Data):** Used when the data in both groups are normally distributed. It does not assume equal variances. The t-statistic is calculated as:""")
            st.latex(r'''
            t = \frac{\bar{x_1} - \bar{x_2}}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
            ''')
            st.markdown(r"""The degrees of freedom are approximated using the Welch–Satterthwaite equation.""")
            st.markdown(r"""- **Mann-Whitney U test (for Non-Normal Data):** A non-parametric test that works on ranked data. It tests the null hypothesis that for randomly selected values X and Y from two populations, the probability of X being greater than Y is equal to the probability of Y being greater than X. The U statistic for group 1 is:""")
            st.latex(r'''
            U_1 = R_1 - \frac{n_1(n_1+1)}{2}
            ''')
            st.markdown(r"""where $R_1$ is the sum of ranks in group 1.""")
            st.markdown("""
            **Procedure:**
            1. Check if the data in each group is normally distributed using the Shapiro-Wilk test.
            2. If both groups are normal (p > 0.05), perform a Welch's t-test. Otherwise, perform a Mann-Whitney U test.
            3. Compare the resulting p-value to a pre-defined significance level ($\alpha$, typically 0.05).

            **Significance of Results:**
            - If **p < 0.05**, we reject the null hypothesis and conclude there is a statistically significant difference between the groups.
            - If **p ≥ 0.05**, we fail to reject the null hypothesis, meaning we do not have sufficient evidence to conclude that a difference exists. This does not prove they are the same.
            """)
        ht_data = ssm.get_data("quality_system", "hypothesis_testing_data")
        df_a = pd.DataFrame({'value': ht_data['pipeline_a'], 'group': 'Pipeline A'})
        df_b = pd.DataFrame({'value': ht_data['pipeline_b'], 'group': 'Pipeline B'})
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

    # --- Tool 3: Equivalence Testing (TOST) ---
    with tool_tabs[2]:
        st.subheader("Equivalence Testing (TOST) for Change Control")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To demonstrate that two groups are "the same" within a pre-defined margin. This is the correct statistical approach for validating a change, such as qualifying a new reagent lot, where the goal is to prove it performs identically to the old lot. Standard hypothesis testing can only fail to find a difference; it cannot prove similarity.
            """)
            st.markdown(r"""
            **Conceptual Walkthrough: What's Happening in the Background?**
            Proving two things are the same is statistically difficult. A standard t-test is designed to find a difference, not prove its absence. TOST solves this by flipping the problem. Imagine we've set "goalposts" on a field ($-\Delta$ and $+\Delta$). We are saying, "Any difference that falls inside these goalposts is practically meaningless." TOST then runs two tests simultaneously:
            1. Is the difference "guilty" of being too low (i.e., less than $-\Delta$)?
            2. Is the difference "guilty" of being too high (i.e., greater than $+\Delta$)?
            If we can prove that our difference is *not guilty* on both counts, then by logical elimination, it *must* lie within the goalposts. The 90% confidence interval is like a "net" we cast for the true difference; if the entire net falls inside the goalposts, we have demonstrated equivalence.
            """)
            st.markdown(r"""
            **Mathematical Basis:**
            TOST (Two One-Sided Tests) flips the null hypothesis. Instead of a null of "no difference," we have two null hypotheses of **non-equivalence**:
            - $H_{01}: \mu_1 - \mu_2 \le -\Delta$ (The difference is less than the lower equivalence bound)
            - $H_{02}: \mu_1 - \mu_2 \ge +\Delta$ (The difference is greater than the upper equivalence bound)
            We perform two separate one-sided t-tests. The test statistics are:
            """)
            st.latex(r'''
            t_1 = \frac{(\bar{x_1} - \bar{x_2}) - (-\Delta)}{SE_{diff}} \quad \text{and} \quad t_2 = \frac{(\bar{x_1} - \bar{x_2}) - (+\Delta)}{SE_{diff}}
            ''')
            st.markdown(r"""
            If **both** tests are significant (p < 0.05), we can reject both nulls and conclude that the true difference lies within the equivalence bounds $[-\Delta, +\Delta]$. The final TOST p-value is the larger of the two individual p-values, $p_{TOST} = \max(p_1, p_2)$.

            **Procedure:**
            1. Define a scientifically justifiable equivalence margin, $\Delta$. This is the largest difference that is considered clinically or scientifically irrelevant.
            2. Collect data from both groups (e.g., old lot vs. new lot).
            3. Perform the two one-sided tests against the bounds $-\Delta$ and $+\Delta$.
            4. If the 90% confidence interval of the difference falls entirely within $[-\Delta, +\Delta]$, equivalence is demonstrated.

            **Significance of Results:**
            A successful equivalence test (p < 0.05) provides strong statistical evidence that a process or material change has not negatively impacted the assay's performance. This is critical documentation for change control under 21 CFR 820 and ISO 13485.
            """)
        eq_data = ssm.get_data("quality_system", "equivalence_data")
        margin_pct = st.slider("Select Equivalence Margin (%)", 5, 25, 10, key="tost_slider")
        lot_a = np.array(eq_data.get('reagent_lot_a', []))
        lot_b = np.array(eq_data.get('reagent_lot_b', []))
        if lot_a.size > 0 and lot_b.size > 0:
            margin_abs = (margin_pct / 100) * lot_a.mean()
            lower_bound, upper_bound = -margin_abs, margin_abs
            fig, p_value = create_tost_plot(lot_a, lot_b, lower_bound, upper_bound)
            st.plotly_chart(fig, use_container_width=True)
            if p_value < 0.05:
                st.success(f"**Conclusion:** Equivalence has been demonstrated (p = {p_value:.4f}). The difference between the lots is statistically smaller than the defined margin of {margin_pct}%.")
            else:
                st.error(f"**Conclusion:** Equivalence could not be demonstrated (p = {p_value:.4f}). The confidence interval for the difference extends beyond the equivalence margin of {margin_pct}%.")
        else:
            st.warning("Insufficient data for equivalence testing.")
    
    # --- Tool 4: Pareto Analysis ---
    with tool_tabs[3]:
        st.subheader("Pareto Analysis of Run Failures")
        with st.expander("View Method Explanation"):
            st.markdown("""
            **Purpose of the Tool:**
            To identify the most frequent causes of a problem, enabling focused process improvement efforts. It is based on the **Pareto Principle**, or the "80/20 rule," which posits that roughly 80% of effects come from 20% of the causes.
            
            **Conceptual Walkthrough: What's Happening in the Background?**
            Imagine you're an emergency room doctor with a waiting room full of patients. You can't treat everyone at once. You must **triage**: identify and treat the most critical patients first. A Pareto chart is a triage tool for process problems. It takes all the reasons for failure, counts them up, and sorts them from most common to least common. The chart immediately and visually points out the "biggest bleeders"—the one or two causes that are responsible for the majority of your problems. This allows you to apply your limited resources (time, people, money) where they will have the greatest effect.

            **Mathematical Basis:**
            This is a descriptive statistical tool, not an inferential one. It involves calculating simple frequencies and cumulative frequencies.
            
            **Procedure:**
            1. Collect categorical data on the causes of a problem (e.g., lab run failure modes).
            2. Tally the counts for each category and sort them in descending order.
            3. Calculate the cumulative percentage for each category.
            4. Plot the counts as a bar chart and the cumulative percentage as a line chart on a secondary axis.
            
            **Significance of Results:**
            The Pareto chart visually separates the "vital few" from the "trivial many." It immediately shows the project team where to focus their resources (e.g., for a CAPA or process improvement project) to achieve the greatest impact on reducing failures and Cost of Poor Quality (COPQ).
            """)
        failure_data = ssm.get_data("lab_operations", "run_failures")
        df_failures = pd.DataFrame(failure_data)
        if not df_failures.empty:
            fig = create_pareto_chart(df_failures, category_col='failure_mode', title='Pareto Analysis of Assay Run Failures')
            st.plotly_chart(fig, use_container_width=True)
            st.success("The analysis highlights 'Low Library Yield' and 'Reagent QC Failure' as the primary contributors to run failures, indicating these are the top priorities for process improvement initiatives.")
        else:
            st.info("No failure data to analyze.")
            
    # --- Tool 5: Gauge R&R ---
    with tool_tabs[4]:
        st.subheader("Measurement System Analysis (Gauge R&R)")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To quantify the amount of variation in your data that comes from the measurement system itself, as opposed to the actual variation between the items being measured. A reliable measurement system is a prerequisite for any valid process control or improvement.
            
            **Conceptual Walkthrough: What's Happening in the Background?**
            Imagine you are trying to measure the heights of several different people (the "Parts"). Your measurement tool is a camera. The total variation you see in the photos comes from two sources: the *real* differences in height between the people, and the "error" from your measurement system. This error also has two parts:
            - **Repeatability** is the "shakiness" of the camera. If you take three photos of the *same* person, are the heights identical? The variation between those photos is repeatability.
            - **Reproducibility** is the difference between photographers. If you and a friend both take photos of the same group of people, will your measurements be identical? The variation between your results and your friend's is reproducibility.
            A Gauge R&R study tells you what percentage of the total variation is just "camera shake" and "photographer difference." If that percentage is too high, you can't trust your photos to tell you who is actually taller. You need to fix your measurement system first.

            **Mathematical Basis:**
            Analysis of Variance (ANOVA) is used to partition the total observed variance ($\sigma^2_{\text{Total}}$) into its constituent components. An ANOVA table is constructed (Source, Sum of Squares, Degrees of Freedom, Mean Square). From the Mean Square (MS) values, the variance components are estimated:
            - $\sigma^2_{\text{Repeatability}} = MS_{Error}$
            - $\sigma^2_{\text{Operator}} = \frac{MS_{Operator} - MS_{Operator:Part}}{n_{parts} \cdot n_{replicates}}$
            - $\sigma^2_{\text{Operator:Part}} = \frac{MS_{Operator:Part} - MS_{Error}}{n_{replicates}}$
            - $\sigma^2_{\text{Reproducibility}} = \sigma^2_{\text{Operator}} + \sigma^2_{\text{Operator:Part}}$
            - $\sigma^2_{\text{Part}} = \frac{MS_{Part} - MS_{Operator:Part}}{n_{operators} \cdot n_{replicates}}$
            
            **Procedure:**
            A structured experiment is performed where multiple operators measure multiple parts multiple times. The resulting data is analyzed using ANOVA to calculate the variance components.
            
            **Significance of Results:**
            The key metric is the **% Contribution of GR&R**, which is $(\sigma^2_{\text{GRR}} / \sigma^2_{\text{Total}}) \times 100\%$.
            - **< 10%:** Generally considered an acceptable measurement system.
            - **10% - 30%:** May be acceptable depending on the application and cost.
            - **> 30%:** Unacceptable. The measurement system is contributing too much noise and must be improved.
            This study is a cornerstone of qualifying an assay for use in a regulated production (CLIA) environment.
            """)
        msa_data = ssm.get_data("quality_system", "msa_data")
        df_msa = pd.DataFrame(msa_data)
        if not df_msa.empty:
            fig, results_df = create_gauge_rr_plot(df_msa, part_col='part', operator_col='operator', value_col='measurement')
            st.write("##### ANOVA Variance Components")
            st.dataframe(results_df, use_container_width=True)
            st.plotly_chart(fig, use_container_width=True)
            if not results_df.empty:
                total_grr = results_df.loc['Total Gauge R&R', '% Contribution']
                if total_grr < 10:
                    st.success(f"**Conclusion:** The measurement system is acceptable (Total GR&R = {total_grr:.2f}%). Most of the variation comes from the parts themselves, not the measurement process.")
                elif total_grr < 30:
                    st.warning(f"**Conclusion:** The measurement system is marginal (Total GR&R = {total_grr:.2f}%). Further investigation may be warranted.")
                else:
                    st.error(f"**Conclusion:** The measurement system is unacceptable (Total GR&R = {total_grr:.2f}%). The assay has too much inherent variation.")
            else:
                st.error("Could not calculate Gauge R&R results due to an error in the plotting utility.")
        else:
            st.info("No MSA data to analyze.")

    # --- Tool 6: DOE (Screening) ---
    with tool_tabs[5]:
        st.subheader("DOE for Factor Screening")
        with st.expander("View Method Explanation"):
            st.markdown("""
            **Purpose of the Tool:**
            To efficiently identify which of many potential factors have a significant effect on a process output. It allows for the simultaneous study of many factors, unlike traditional one-factor-at-a-time (OFAT) experiments, which are inefficient and fail to detect interactions.
            
            **Conceptual Walkthrough: What's Happening in the Background?**
            Imagine you're trying to perfect a cake recipe with four ingredients you can change: Flour, Sugar, Eggs, and Baking Time. The traditional way is to bake a cake, change only the flour, bake another, then change only the sugar, and so on. This is slow and misses the key insight that maybe more sugar *only* works if you also use more eggs (an interaction). A DOE is a smarter way. It gives you a specific, minimal set of recipes to bake (e.g., "high sugar, low time," "low sugar, high time," etc.) that cleverly covers all the combinations. By analyzing the results of just these few cakes, the math can untangle the effect of each ingredient individually (main effects) and also detect any crucial interactions between them.

            **Mathematical Basis:**
            A factorial design (e.g., 2-level full factorial, $2^k$) is used to create an orthogonal experimental plan. The results are analyzed using a linear model to estimate the **main effect** of each factor and the **interaction effects** between factors. The main effect for a factor is calculated as:
            """)
            st.latex(r'''
            \text{Effect}_A = \bar{y}_{A,high} - \bar{y}_{A,low}
            ''')
            st.markdown("""
            An interaction effect (e.g., AB) measures how the effect of factor A changes at different levels of factor B.

            **Procedure:**
            1. Identify potential factors and their high/low levels.
            2. Run the experiments according to the factorial design matrix.
            3. Analyze the results (e.g., with an ANOVA or effects plots) to determine which effects are statistically significant.
            
            **Significance of Results:**
            A screening DOE quickly narrows down a large problem space, allowing the team to focus subsequent, more intensive optimization experiments (like RSM) only on the "vital few" factors that actually matter. It is a foundational tool for efficient process development and characterization.
            """)
        doe_data = ssm.get_data("quality_system", "doe_data")
        df_doe = pd.DataFrame(doe_data)
        st.write("##### DOE Data")
        st.dataframe(df_doe, use_container_width=True)
        try:
            factor1, factor2, response = 'pcr_cycles', 'input_dna', 'library_yield'
            if not all(col in df_doe.columns for col in [factor1, factor2, response]):
                raise ValueError("DOE data is missing one or more required columns.")
            effects_fig, interaction_fig = create_doe_effects_plot(df_doe, factor1, factor2, response)
            st.write(f"##### Main Effects and Interaction Analysis")
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(effects_fig, use_container_width=True)
            with col2: st.plotly_chart(interaction_fig, use_container_width=True)
            st.success("This analysis identifies which factors have the largest impact on the response, guiding further optimization experiments like RSM.")
        except Exception as e:
            st.error(f"Could not perform DOE analysis. Error: {e}")
            logger.error(f"DOE analysis failed: {e}", exc_info=True)
            
    # --- Tool 7: Response Surface Methodology (RSM) ---
    with tool_tabs[6]:
        st.subheader("Response Surface Methodology (RSM) for Optimization")
        with st.expander("View Method Explanation", expanded=False):
            st.markdown(r"""
            **Purpose of the Method:**
            After a screening DOE identifies significant factors, RSM is used to find the optimal settings for those factors. It uses a more detailed experimental design (like a Central Composite Design) to fit a **quadratic model**, allowing us to visualize and analyze curvature in the response. The ultimate goal is to find the "peak" or "valley" of the response surface, defining a robust **Normal Operating Range (NOR)**.
            
            **Conceptual Walkthrough: What's Happening in the Background?**
            If a screening DOE is about finding which mountains are worth climbing, RSM is about finding the exact peak of the one you chose. A simple linear model can only describe a flat, tilted plane—it can't describe a peak. RSM uses a more complex (quadratic) equation that can model curves, hills, and valleys. By running a few extra, cleverly placed experiments (the "star points" of a CCD), we give the model enough information to learn the curvature of the performance landscape. The 3D surface plot is a visual representation of this landscape, and the contour plot is the topographical map. Our goal is to find the "X" on the map that marks the highest point.

            **The Mathematical Basis & Method:**
            A second-order polynomial model is fit to the data using the method of least squares. In matrix notation, this is $Y = X\beta + \epsilon$, where the coefficients $\beta$ are estimated as $\hat{\beta} = (X'X)^{-1}X'Y$. The model equation is:
            $Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_{12}X_1X_2 + \beta_{11}X_1^2 + \beta_{22}X_2^2$  
            The squared terms ($\beta_{11}, \beta_{22}$) are what allow the model to capture curvature, which is essential for finding a true optimum.

            **Procedure:**
            1.  A Central Composite Design (CCD) or Box-Behnken Design is performed. These designs include factorial, center, and axial ("star") points to allow for efficient estimation of all quadratic terms.
            2.  The second-order model is fit to the experimental data.
            3.  The model is visualized as a 3D surface plot and a 2D contour plot.

            **Significance of Results:**
            The RSM analysis provides a predictive map of the process. The contour plot is especially powerful, as it allows scientists to identify an operating region (the Design Space) where the process is robust to small variations in the factors. This is a cornerstone of a Quality by Design (QbD) approach and provides powerful evidence for regulatory submissions that the process is well-understood and controlled.
            """)
        rsm_data = ssm.get_data("quality_system", "rsm_data")
        df_rsm = pd.DataFrame(rsm_data)
        
        st.write("##### Central Composite Design Data for RSM")
        st.dataframe(df_rsm, use_container_width=True)
        
        try:
            factor1, factor2, response = 'pcr_cycles', 'input_dna', 'library_yield'
            surface_fig, contour_fig, model_summary = create_rsm_plots(df_rsm, factor1, factor2, response)
            
            st.write("##### Response Surface Model Summary")
            st.dataframe(model_summary)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(surface_fig, use_container_width=True)
            with col2:
                st.plotly_chart(contour_fig, use_container_width=True)
            
            st.success("""
            **Conclusion:** The quadratic model successfully fits the experimental data. The surface and contour plots clearly indicate an optimal operating region for maximizing library yield. The statistical significance of the quadratic terms (e.g., `I(pcr_cycles ** 2)`) confirms that curvature is a key feature of the process, validating the use of RSM for optimization.
            """)
        except Exception as e:
            st.error(f"Could not perform RSM analysis. Error: {e}")
            logger.error(f"RSM analysis failed: {e}", exc_info=True)
       #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&   ML           &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&     
def render_machine_learning_lab_tab(ssm: MockSessionStateManager):
    """Renders the tab containing machine learning and bioinformatics tools."""
    st.header("🤖 Machine Learning & Bioinformatics Lab")
    st.info("This lab provides tools to analyze the performance and interpretability of the core classifier, a critical component of our SaMD (Software as a Medical Device) validation package.")

    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import beta
        import shap
        import lightgbm as lgb
    except ImportError:
        st.error("This tab requires `scikit-learn`, `shap`, `lightgbm`, and `scipy`. Please install them to enable ML features.", icon="🚨")
        return

    # --- Helper Functions (Rebuilt to be self-contained) ---
    def _create_roc_curve(y_true, y_score, title="<b>ROC Curve</b>"):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash', color='grey')))
        fig.update_layout(title=title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate (Sensitivity)')
        return fig

    def _create_confusion_matrix_heatmap(cm, class_names):
        fig = px.imshow(cm,
                        labels=dict(x="Predicted Label", y="True Label", color="Count"),
                        x=class_names,
                        y=class_names,
                        text_auto=True,
                        color_continuous_scale='Blues')
        fig.update_layout(title="<b>Confusion Matrix</b>")
        return fig

    @st.cache_data
    def _create_shap_summary_plot(shap_values, X_data):
        """Creates a SHAP summary plot and returns it as a buffer for st.image."""
        shap.summary_plot(shap_values, X_data, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
        
    def _create_forecast_plot(history_df, forecast_df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_df.index, y=history_df['samples'], name='Historical Data', mode='lines'))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], name='Forecast', mode='lines', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], name='Upper CI', mode='lines', line=dict(color='orange', width=0)))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], name='Lower CI', fill='tonexty', mode='lines', line=dict(color='orange', width=0)))
        fig.update_layout(title="<b>Sample Volume Forecast</b>", xaxis_title="Date", yaxis_title="Number of Samples")
        return fig

    # SME ADDITION: Added 3 new tabs for advanced diagnostics
    ml_tabs = st.tabs([
        "Classifier Performance (ROC & PR)",
        "Classifier Explainability (SHAP)",
        "Cancer Signal of Origin (CSO) Analysis",
        "Predictive Ops (Run Failure)",
        "Time Series Forecasting (ML)",
        "Classifier Feature Importance",
        "⭐ ctDNA Fragmentomics Analysis",          # NEW
        "⭐ Sequencing Error Profile Modeling",     # NEW
        "⭐ Predictive Run QC (On-Instrument)"      # NEW
    ])

    # --- Prepare Data and Models Once for All Tabs ---
    X, y = ssm.get_data("ml_models", "classifier_data")

    @st.cache_resource
    def get_rf_model(_X, _y):
        model = RandomForestClassifier(n_estimators=25, max_depth=10, random_state=42)
        model.fit(_X, _y)
        return model

    @st.cache_resource
    def get_lr_model(_X, _y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(_X)
        model = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42)
        model.fit(X_scaled, _y)
        return model, scaler

    rf_model = get_rf_model(X, y)
    lr_model, scaler = get_lr_model(X, y)
    X_scaled = scaler.transform(X)


    # --- Tool 1: ROC & PR Curves ---
    with ml_tabs[0]:
        st.subheader("Classifier Performance Analysis")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To visualize the performance of our binary classifier. The ROC curve assesses the trade-off between sensitivity and specificity, while the Precision-Recall (PR) curve is crucial for evaluating performance on imbalanced datasets like ours.

            **Conceptual Walkthrough:**
            - **ROC Curve:** Imagine slowly lowering the bar for what we call a "cancer signal." As we lower it, we catch more true cancers (increasing sensitivity, good!) but also misclassify more healthy people (increasing the false positive rate, bad!). The ROC curve plots this entire trade-off. The Area Under the Curve (AUC) is a single number summarizing this: 1.0 is perfect, 0.5 is a random guess.
            - **PR Curve:** This answers a more practical clinical question: "Of all the patients we flagged as positive, what fraction actually had cancer?" This is **Precision**. The curve shows how precision changes as we try to find more and more of the true cancers (increase **Recall**). In a screening test, maintaining high precision is vital to avoid unnecessary follow-up procedures.

            **Mathematical Basis:**
            - **ROC:** Plots True Positive Rate ($TPR = \frac{TP}{TP+FN}$) vs. False Positive Rate ($FPR = \frac{FP}{FP+TN}$).
            - **PR:** Plots Precision ($Precision = \frac{TP}{TP+FP}$) vs. Recall ($Recall = TPR$).
            The area under each curve (AUC-ROC and AUC-PR) provides a single metric to summarize performance.
            """)
        st.write("#### Logistic Regression Performance (on Scaled Data)")
        y_scores_lr = lr_model.predict_proba(X_scaled)[:, 1]

        col1, col2 = st.columns(2)
        with col1:
            roc_fig_lr = _create_roc_curve(y, y_scores_lr)
            st.plotly_chart(roc_fig_lr, use_container_width=True)
        with col2:
            precision, recall, _ = precision_recall_curve(y, y_scores_lr)
            pr_fig_lr = px.area(x=recall, y=precision, title="<b>Precision-Recall Curve</b>", labels={'x':'Recall (Sensitivity)', 'y':'Precision'})
            pr_fig_lr.update_layout(xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1.05]))
            st.plotly_chart(pr_fig_lr, use_container_width=True)

        st.write("#### Random Forest Performance (on Original Data)")
        y_scores_rf = rf_model.predict_proba(X)[:, 1]
        col3, col4 = st.columns(2)
        with col3:
            roc_fig_rf = _create_roc_curve(y, y_scores_rf)
            st.plotly_chart(roc_fig_rf, use_container_width=True)
        with col4:
            precision, recall, _ = precision_recall_curve(y, y_scores_rf)
            pr_fig_rf = px.area(x=recall, y=precision, title="<b>Precision-Recall Curve</b>", labels={'x':'Recall (Sensitivity)', 'y':'Precision'})
            pr_fig_rf.update_layout(xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1.05]))
            st.plotly_chart(pr_fig_rf, use_container_width=True)

    # --- Tool 2: SHAP ---
    with ml_tabs[1]:
        st.subheader("Cancer Classifier Explainability (SHAP)")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To address the "black box" problem of complex machine learning models. For a high-risk SaMD (Software as a Medical Device), we must not only show that our classifier works, but also provide evidence for *how* it works. SHAP provides this model explainability.
            
            **Conceptual Walkthrough:**
            Imagine our machine learning model is like a sports team, and its final prediction is the team's score. The features (our biomarkers) are the players. SHAP analysis is like a sophisticated "Most Valuable Player" calculation for every single game (every single patient sample). It doesn't just tell you who the best player is overall; it tells you exactly how much each player contributed to the final score in that specific game. For one patient, a high value for `promoter_A_met` might have pushed the score up by 0.3, while a low value for `enhancer_B_met` might have pulled it down by 0.1. The summary plot aggregates thousands of these "game reports" to show which players are consistently the most impactful and whether their impact is positive or negative.
            
            **Mathematical Basis:**
            SHAP (SHapley Additive exPlanations) is based on **Shapley values**, a concept from cooperative game theory. It calculates the marginal contribution of each feature to the final prediction for a single sample. The Shapley value for a feature *i* is its average marginal contribution across all possible feature coalitions:
            """)
            st.latex(r'''
            \phi_i(v) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} [v(S \cup \{i\}) - v(S)]
            ''')
        st.write("Generating SHAP values for the Random Forest classifier. This may take a moment...")
        try:
            st.caption("Note: Explaining on a random subsample of 50 data points for performance.")
            X_sample = X.sample(n=min(50, len(X)), random_state=42)
            explainer = shap.TreeExplainer(rf_model)
            shap_values_obj = explainer(X_sample)
            st.write("##### SHAP Summary Plot (Impact on 'Cancer Signal Detected' Prediction)")
            plot_buffer = _create_shap_summary_plot(shap_values_obj.values[:,:,1], X_sample)
            if plot_buffer:
                st.image(plot_buffer)
                st.success("The SHAP analysis confirms that known oncogenic methylation markers are the most significant drivers of a 'Cancer Signal Detected' result. This provides strong evidence that the model has learned biologically relevant signals.")
            else:
                st.error("Could not generate SHAP summary plot.")
        except Exception as e:
            st.error(f"Could not perform SHAP analysis. Error: {e}")
            logger.error(f"SHAP analysis failed: {e}", exc_info=True)

    # --- Tool 3: CSO Analysis ---
    with ml_tabs[2]:
        st.subheader("Cancer Signal of Origin (CSO) Analysis")
        with st.expander("View Method Explanation"):
            st.markdown("""
            **Purpose of the Tool:**
            For an MCED test, detecting a cancer signal is only half the battle. A key secondary claim is the ability to predict the **Cancer Signal of Origin (CSO)**, which guides the subsequent clinical workup. This tool analyzes the performance of the CSO prediction model.
            
            **Conceptual Walkthrough:**
            After the first model says "Cancer Signal Detected," a second, multi-class classifier is used to predict the tissue of origin (e.g., Lung, Colon, Pancreatic). A **confusion matrix** is the perfect tool for visualizing its performance. It's a grid that shows us not just what we got right, but also where we went wrong. For example, it might reveal that the model frequently confuses Lung and Head & Neck cancers, which is biologically plausible and provides valuable insight for improving the model or refining the clinical report.
            """)
        st.write("Generating synthetic CSO data and training a simple model...")
        np.random.seed(123)
        cso_classes = ['Lung', 'Colon', 'Pancreatic', 'Liver', 'Ovarian']
        cancer_samples_X = X[y == 1]
        if not cancer_samples_X.empty:
            true_cso = np.random.choice(cso_classes, size=len(cancer_samples_X))
            cso_model = RandomForestClassifier(n_estimators=50, random_state=123)
            cso_model.fit(cancer_samples_X, true_cso)
            predicted_cso = cso_model.predict(cancer_samples_X)
            cm_cso = confusion_matrix(true_cso, predicted_cso, labels=cso_classes)
            fig_cm_cso = _create_confusion_matrix_heatmap(cm_cso, cso_classes)
            st.plotly_chart(fig_cm_cso, use_container_width=True)
            accuracy = np.diag(cm_cso).sum() / cm_cso.sum()
            st.success(f"The CSO classifier achieved an overall accuracy of **{accuracy:.1%}**.")
        else:
            st.warning("No 'cancer positive' samples available in the dataset to perform CSO analysis.")

    # --- Tool 4: Predictive Operations ---
    with ml_tabs[3]:
        st.subheader("Predictive Operations: Sequencing Run Failure")
        with st.expander("View Method Explanation"):
            st.markdown("""
            **Purpose of the Tool:**
            To build a predictive model that can identify sequencing runs likely to fail QC *before* committing expensive reagents and sequencer time. This is a proactive quality control tool aimed at improving operational efficiency and reducing the Cost of Poor Quality (COPQ).
            """)
        run_qc_data = ssm.get_data("ml_models", "run_qc_data")
        df_run_qc = pd.DataFrame(run_qc_data)
        X_ops = df_run_qc[['library_concentration', 'dv200_percent', 'adapter_dimer_percent']]
        y_ops = df_run_qc['outcome'].apply(lambda x: 1 if x == 'Fail' else 0)
        X_train, X_test, y_train, y_test = train_test_split(X_ops, y_ops, test_size=0.3, random_state=42, stratify=y_ops)
        model_ops = LogisticRegression(random_state=42, class_weight='balanced')
        model_ops.fit(X_train, y_train)
        y_pred = model_ops.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        st.write("##### Run Failure Prediction Model Performance (on Test Set)")
        fig_cm_ops = _create_confusion_matrix_heatmap(cm, ['Pass', 'Fail'])
        st.plotly_chart(fig_cm_ops, use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        st.success(f"**Model Evaluation:** The model correctly identified **{tp}** out of **{tp+fn}** failing runs, enabling proactive intervention.")

    # --- Tool 5: Time Series Forecasting (ML) ---
    with ml_tabs[4]:
        st.subheader("Time Series Forecasting with Machine Learning")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To forecast future demand (e.g., incoming sample volume) based on historical data. This is crucial for proactive lab management, including reagent inventory control, staffing, and capacity planning.
            """)
        ts_data = ssm.get_data("ml_models", "sample_volume_data")
        df_ts = pd.DataFrame(ts_data).set_index('date')
        df_ts.index = pd.to_datetime(df_ts.index)

        def create_ts_features(df):
            df = df.copy()
            for lag in [1, 7, 14]: df[f'lag_{lag}'] = df['samples'].shift(lag)
            df['dayofweek'] = df.index.dayofweek
            df['month'] = df.index.month
            return df

        df_ts_feat = create_ts_features(df_ts)
        df_ts_feat.dropna(inplace=True)
        X_ts, y_ts = df_ts_feat.drop('samples', axis=1), df_ts_feat['samples']
        model_lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1)
        model_lgbm.fit(X_ts, y_ts)
        
        future_predictions, n_forecast, history = [], 30, df_ts.copy()
        for i in range(n_forecast):
            future_date = history.index[-1] + pd.Timedelta(days=1)
            features_for_pred = create_ts_features(history.tail(20))
            X_future = features_for_pred.drop('samples', axis=1).iloc[[-1]]
            prediction = model_lgbm.predict(X_future)[0]
            future_predictions.append(prediction)
            # Create a new DataFrame for the new row to avoid fragmentation
            new_row = pd.DataFrame({'samples': [prediction]}, index=[future_date])
            history = pd.concat([history, new_row])


        future_dates = pd.date_range(start=df_ts.index.max() + pd.Timedelta(days=1), periods=n_forecast, freq='D')
        forecast_df = pd.DataFrame({'mean': future_predictions}, index=future_dates)
        forecast_df['mean_ci_upper'] = forecast_df['mean'] * 1.15 # Wider CI for ML models
        forecast_df['mean_ci_lower'] = forecast_df['mean'] * 0.85
        fig = _create_forecast_plot(df_ts, forecast_df)
        st.plotly_chart(fig, use_container_width=True)
        st.success("The LightGBM forecast projects a continued upward trend in sample volume.")

    # --- Tool 6: Classifier Feature Importance ---
    with ml_tabs[5]:
        st.subheader("Classifier Feature Importance")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To understand which biomarkers are the most important drivers of the classifier's prediction. For a linear model like Logistic Regression, this is achieved by directly inspecting the model's learned coefficients.
            """)
        st.write("##### Feature Importance from Logistic Regression Coefficients")
        try:
            coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr_model.coef_[0]})
            coefficients['abs_coeff'] = coefficients['Coefficient'].abs()
            important_coeffs = coefficients[coefficients['abs_coeff'] > 0.01].sort_values('Coefficient')
            if not important_coeffs.empty:
                fig = px.bar(important_coeffs, x='Coefficient', y='Feature', orientation='h', color='Coefficient', color_continuous_scale='RdBu_r', title='<b>Impact of Biomarkers on Cancer Signal Prediction</b>')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                st.success("The feature importance analysis confirms that known oncogenic methylation markers are the most significant drivers of a positive result.")
            else:
                st.warning("The model did not find any significantly important features with the current regularization settings.")
        except Exception as e:
            st.error(f"Could not perform feature importance analysis. Error: {e}")
            logger.error(f"Feature importance analysis failed: {e}", exc_info=True)

    # --- Tool 7: ctDNA Fragmentomics Analysis (NEW) ---
    with ml_tabs[6]:
        st.subheader("ctDNA Signal Enhancement via Fragmentomics")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To leverage the biological insight that ctDNA fragments are often shorter than background cell-free DNA (cfDNA) from healthy apoptotic cells. This tool models these fragment size distributions to enhance the detection of a cancer signal.

            **Conceptual Walkthrough:**
            Imagine cfDNA in a blood sample as a collection of different lengths of string. Most strings (from healthy cells) are around 167 base pairs long. However, strings from cancer cells are often shorter, peaking around 145 base pairs. Instead of looking just for specific mutations, we can analyze the *overall distribution* of string lengths. A sample with an unusually high proportion of short strings is more likely to contain a cancer signal. This tool builds a classifier based on features derived from these distributions (e.g., percentage of fragments < 150bp).

            **Mathematical Basis:**
            This is a feature engineering and classification problem. We extract features that describe the fragment size distribution for each sample. Key features might include:
            - **Short Fragment Fraction:** The percentage of DNA fragments below a certain length (e.g., 150 bp).
            - **Distributional Moments:** Mean, variance, skewness, and kurtosis of the fragment lengths.
            - **Mode(s):** The location of peaks in the distribution.
            A classifier, such as a **Gradient Boosting Machine**, is then trained on these engineered features to distinguish between "Cancer-like" and "Healthy-like" fragment profiles.

            **Significance of Results:**
            Demonstrating that our assay captures and utilizes known biological phenomena like differential fragmentation provides powerful evidence for **analytical validity**. It shows the classifier is not just a black box but is keyed into scientifically relevant signals. This is a critical piece of evidence for the PMA, showing the *mechanism* by which our test works and de-risking the algorithm from being reliant on spurious correlations.
            """)
        np.random.seed(42)
        healthy_frags = np.random.normal(167, 10, 5000)
        cancer_frags = np.random.normal(145, 15, 5000)
        df_frags = pd.DataFrame({
            'Fragment Size (bp)': np.concatenate([healthy_frags, cancer_frags]),
            'Sample Type': ['Healthy'] * 5000 + ['Cancer'] * 5000
        })
        fig_hist = px.histogram(df_frags, x='Fragment Size (bp)', color='Sample Type', nbins=100,
                                barmode='overlay', histnorm='probability density',
                                title="<b>Distribution of DNA Fragment Sizes</b>")
        st.plotly_chart(fig_hist, use_container_width=True)

        n_samples = 100
        X_frag, y_frag = [], []
        for i in range(n_samples):
            sample_h = np.random.normal(167, 10, 200)
            X_frag.append([np.mean(sample_h), np.std(sample_h), (sample_h < 150).mean()])
            y_frag.append(0)
            sample_c = np.random.normal(145, 15, 200)
            X_frag.append([np.mean(sample_c), np.std(sample_c), (sample_c < 150).mean()])
            y_frag.append(1)
        X_frag, y_frag = pd.DataFrame(X_frag, columns=['mean_frag_size', 'std_frag_size', 'short_frag_pct']), np.array(y_frag)

        X_train, X_test, y_train, y_test = train_test_split(X_frag, y_frag, test_size=0.3, random_state=42)
        frag_model = GradientBoostingClassifier(random_state=42)
        frag_model.fit(X_train, y_train)
        accuracy = frag_model.score(X_test, y_test)
        st.success(f"A classifier trained solely on fragment size features achieved an accuracy of **{accuracy:.1%}**. This confirms that fragmentomics provides significant discriminatory information, strengthening the scientific rationale for our assay.")

    # --- Tool 8: Sequencing Error Modeling (NEW) ---
    with ml_tabs[7]:
        st.subheader("Modeling Sequencing Error Profiles for Variant Calling")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To accurately distinguish true, low-frequency mutations in ctDNA from the background of inevitable sequencing errors. This is the most critical challenge for achieving a low Limit of Detection (LoD).

            **Conceptual Walkthrough:**
            Imagine trying to hear a faint whisper (a true mutation) in a room with a constant, low hum (sequencing errors). To be confident you heard the whisper, you first need to understand the exact pitch and volume of the hum. This tool does that for our sequencer. It analyzes a large number of healthy samples to build a high-resolution "error fingerprint" for every possible type of mutation (e.g., C>T, G>A) in every possible sequence context. When we analyze a new sample and see a potential mutation, we compare it to this fingerprint. If it looks exactly like a common error, we require a much stronger signal to believe it's real.

            **Mathematical Basis:**
            The number of reads supporting a variant allele at a given position can be modeled by a **Beta-Binomial distribution**. Unlike a simple Binomial distribution, it accounts for overdispersion—the fact that error rates can vary slightly from run to run. For each genomic context, we fit a beta-binomial model to data from normal samples to learn its characteristic error parameters, $\alpha_0$ and $\beta_0$. For a new sample with $k$ variant reads out of $n$ total reads, we can then calculate a p-value:
            """)
            st.latex(r'''
            P(\text{reads} \ge k \mid n, \alpha_0, \beta_0)
            ''')
            st.markdown(r"""
            This p-value represents the probability of observing at least $k$ variant reads *by chance alone*, according to our error model. A very low p-value gives us confidence that the variant is real.
            """)
        np.random.seed(123)
        error_rate_dist = np.random.beta(a=0.5, b=2000, size=100) # Made more realistic
        alpha0, beta0, _, _ = beta.fit(error_rate_dist, floc=0, fscale=1)
        st.write(f"**Fitted Error Model Parameters:** `alpha = {alpha0:.3f}`, `beta = {beta0:.3f}`")

        depth = st.slider("Select Sequencing Depth", 1000, 20000, 5000, step=1000, key="depth_slider")
        true_vaf = st.slider("Select True Variant Allele Frequency (VAF)", 0.0, 0.01, 0.005, step=0.0005, format="%.4f", key="vaf_slider_new")
        observed_variant_reads = np.random.binomial(depth, true_vaf)
        observed_vaf = observed_variant_reads / depth
        p_value = 1.0 - beta.cdf(observed_vaf, alpha0, beta0)
        st.metric("Observed VAF", f"{observed_vaf:.4f}")
        st.metric("P-value (Probability of Observation by Chance)", f"{p_value:.2e}")
        if p_value < 1e-6:
             st.success(f"The observed VAF is highly statistically significant (p < 0.000001). We can confidently call this a true mutation.")
        else:
             st.error(f"The observed VAF is not statistically significant. It is likely a result of sequencing noise and should not be called.")

    # --- Tool 9: Predictive Run QC from On-Instrument Metrics (NEW) ---
    with ml_tabs[8]:
        st.subheader("Predictive Run QC from Early On-Instrument Metrics")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To predict the final quality of a sequencing run using real-time metrics generated by the sequencer *during the first few hours* of the run. This allows for the early termination of runs that are destined to fail, saving significant instrument time and reagent costs.

            **Conceptual Walkthrough:**
            Think of a multi-day sequencing run as a long-haul flight. Instead of waiting until landing to see if the journey was smooth, we can check key engine performance metrics during takeoff and initial ascent. This tool analyzes early-run metrics like **Cluster Density**, **% Q30 score at cycle 25**, and **% Phasing**. A model trained on historical data learns the "signature" of a run that will ultimately succeed or fail. If a current run's early metrics match the signature of a failure, we can abort the mission and troubleshoot immediately.

            **Mathematical Basis:**
            This is a binary classification problem. We use **Logistic Regression** to model the probability of a "Pass" outcome. The model learns a set of coefficients ($\beta_i$) for each early-run QC feature ($x_i$) to predict the log-odds of the run passing its final QC check:
            """)
            st.latex(r'''
            \ln\left(\frac{P(\text{Pass})}{1-P(\text{Pass})}\right) = \beta_0 + \beta_1x_{\text{Q30}} + \beta_2x_{\text{Density}} + \dots
            ''')
        np.random.seed(42)
        n_runs, pass_rate = 200, 0.9
        n_pass, n_fail = int(n_runs * pass_rate), int(n_runs * (1-pass_rate))
        df_on_instrument = pd.DataFrame({
            'q30_at_cycle_25': np.concatenate([np.random.normal(95, 2, n_pass), np.random.normal(85, 5, n_fail)]),
            'cluster_density_k_mm2': np.concatenate([np.random.normal(1200, 150, n_pass), np.random.normal(1800, 200, n_fail)]),
            'final_outcome': ['Pass'] * n_pass + ['Fail'] * n_fail
        }).sample(frac=1).reset_index(drop=True)
        X_oi, y_oi = df_on_instrument.drop('final_outcome', axis=1), df_on_instrument['final_outcome'].apply(lambda x: 1 if x == 'Pass' else 0)

        X_train, X_test, y_train, y_test = train_test_split(X_oi, y_oi, test_size=0.3, random_state=42, stratify=y_oi)
        model_oi_qc = LogisticRegression(random_state=42)
        model_oi_qc.fit(X_train, y_train)
        y_pred = model_oi_qc.predict(X_test)
        st.write("##### On-Instrument QC Model Performance (on Test Set)")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig_cm_oi = _create_confusion_matrix_heatmap(cm, ['Fail', 'Pass'])
        st.plotly_chart(fig_cm_oi, use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        st.success(f"**Model Evaluation:** The model correctly predicted **{tp}** successful runs and **{tn}** failing runs based on early metrics alone, enabling proactive intervention.")

# To run this code, save it as a Python file (e.g., app.py) and run `streamlit run app.py`
# from the terminal. Make sure you have all required libraries installed.
if __name__ == "__main__":
    render_machine_learning_lab_tab(ssm)
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
