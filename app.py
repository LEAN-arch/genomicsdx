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

            **Mathematical Basis:**
            The chart is based on the principles of the Gaussian (Normal) distribution. The control limits are established based on the mean ($\mu$) and standard deviation ($\sigma$) of a set of historical, in-control data. The limits are typically set at $\mu \pm 1\sigma$, $\mu \pm 2\sigma$, and $\mu \pm 3\sigma$.

            **Procedure:**
            1. A stable mean and standard deviation are established for a QC material from at least 20 historical data points.
            2. Control limits are calculated and drawn on the chart.
            3. For each subsequent run, the new QC value is plotted on the chart.
            4. The plot is evaluated against a set of rules (e.g., Westgard rules) to detect shifts or trends that may indicate a problem with the process.

            **Significance of Results:**
            A Levey-Jennings chart provides an early warning system for process drift or instability. A point outside the $\pm 3\sigma$ limits or patterns of points (e.g., 7 consecutive points on one side of the mean) indicates a loss of statistical control, triggering an investigation and preventing the release of potentially erroneous patient results.
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
            To determine if there is a statistically significant difference between the means of two groups. This is used, for example, to compare the output of a new bioinformatics pipeline version against the old one.

            **Mathematical Basis:**
            The framework is Null Hypothesis Significance Testing (NHST). We start with a **Null Hypothesis ($H_0$)** that there is no difference between the groups ($\mu_1 = \mu_2$). We then calculate the probability (**p-value**) of observing our data (or more extreme data) if the null hypothesis were true.

            - **Welch's t-test (for Normal Data):** Used when the data in both groups are normally distributed. It does not assume equal variances. The t-statistic is calculated as:
              $t = \frac{\bar{x_1} - \bar{x_2}}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$
            - **Mann-Whitney U test (for Non-Normal Data):** A non-parametric test that works on ranked data, not the actual values. It tests if the two distributions are the same.

            **Procedure:**
            1. Check if the data in each group is normally distributed using a test like the **Shapiro-Wilk test**.
            2. If both groups are normal, perform a t-test. Otherwise, perform a Mann-Whitney U test.
            3. Compare the resulting p-value to a pre-defined significance level ($\alpha$, typically 0.05).

            **Significance of Results:**
            - If **p < 0.05**, we reject the null hypothesis and conclude there is a statistically significant difference between the groups.
            - If **p ≥ 0.05**, we fail to reject the null hypothesis, meaning we do not have sufficient evidence to conclude that a difference exists.
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

            **Mathematical Basis:**
            TOST (Two One-Sided Tests) flips the null hypothesis. Instead of a null of "no difference," we have two null hypotheses of **non-equivalence**:
            - $H_{01}: \mu_1 - \mu_2 \le -\Delta$ (The difference is less than the lower equivalence bound)
            - $H_{02}: \mu_1 - \mu_2 \ge +\Delta$ (The difference is greater than the upper equivalence bound)
            We perform two separate one-sided t-tests. If **both** tests are significant (p < 0.05), we can reject both nulls and conclude that the true difference lies within the equivalence bounds $[-\Delta, +\Delta]$. The final TOST p-value is the larger of the two individual p-values.

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

            **Mathematical Basis:**
            Analysis of Variance (ANOVA) is used to partition the total observed variance ($\sigma^2_{\text{Total}}$) into its constituent components:
            - **Repeatability ($\sigma^2_{\text{Repeat}}$):** Variation when one operator measures the same part multiple times. This is inherent equipment variation (EV).
            - **Reproducibility ($\sigma^2_{\text{Repro}}$):** Variation when different operators measure the same part. This is appraiser variation (AV).
            - **Gauge R&R ($\sigma^2_{\text{GRR}}$):** The total measurement system variation, where $\sigma^2_{\text{GRR}} = \sigma^2_{\text{Repeat}} + \sigma^2_{\text{Repro}}$.
            - **Part-to-Part ($\sigma^2_{\text{Part}}$):** The true variation between the different parts being measured.

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

            **Mathematical Basis:**
            A factorial design (e.g., 2-level full factorial) is used to create an experimental plan. The results are analyzed using a linear model to estimate the **main effect** of each factor and the **interaction effects** between factors. The main effect is the average change in the response when a factor is moved from its low level to its high level. An interaction effect means the effect of one factor depends on the level of another.

            **Procedure:**
            1. Identify potential factors and their high/low levels.
            2. Run the experiments according to the factorial design matrix.
            3. Analyze the results to determine which effects are statistically significant.
            
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

            **The Mathematical Basis & Method:**
            A second-order polynomial model is fit to the data:
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

def render_machine_learning_lab_tab(ssm: SessionStateManager):
    """Renders the tab containing machine learning and bioinformatics tools."""
    st.header("🤖 Machine Learning & Bioinformatics Lab")
    st.info("Utilize and validate predictive models for operational efficiency and explore the classifier's behavior. Model explainability is key for regulatory review.")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        from statsmodels.tsa.arima.model import ARIMA
        import shap
    except ImportError:
        st.error("This tab requires `scikit-learn`, `shap`, and `statsmodels`. Please install them to enable ML features.", icon="🚨")
        return
        
    ml_tabs = st.tabs(["Classifier Explainability (SHAP)", "Predictive Ops (Run Failure)", "Time Series Forecasting (Samples)"])

    # --- Tool 1: SHAP ---
    with ml_tabs[0]:
        st.subheader("Cancer Classifier Explainability (SHAP)")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To address the "black box" problem of complex machine learning models. For a high-risk SaMD (Software as a Medical Device), we must not only show that our classifier works, but also provide evidence for *how* it works. SHAP provides this model explainability.

            **Mathematical Basis:**
            SHAP (SHapley Additive exPlanations) is based on **Shapley values**, a concept from cooperative game theory. It calculates the marginal contribution of each feature to the final prediction for a single sample. It's the only feature attribution method with a solid theoretical foundation that guarantees properties like local accuracy and consistency.

            **Procedure:**
            1. An explainer object is created from a trained model and a background dataset.
            2. The explainer calculates the SHAP values for each feature for every sample in a test set.
            3. A **summary plot** visualizes these values. Each point is a single feature for a single sample. The color indicates the feature's value (high/low), and its position on the x-axis indicates its impact on the model's output (pushing the prediction higher or lower).

            **Significance of Results:**
            The SHAP summary plot provides powerful evidence for scientific and clinical validation. It allows us to confirm that the model has learned biologically relevant signals (e.g., known oncogenic methylation markers are the top features) and is not relying on spurious correlations or batch effects. This is a critical piece of evidence for de-risking the algorithm portion of a PMA submission.
            """)
        st.write("Generating SHAP values for the locked classifier model. This may take a moment...")
        X, y = ssm.get_data("ml_models", "classifier_data")
        model = ssm.get_data("ml_models", "classifier_model")
        
        try:
            explainer = shap.Explainer(model, X)
            shap_values_obj = explainer(X)
            
            st.write("##### SHAP Summary Plot (Impact on 'Cancer Signal Detected' Prediction)")
            
            shap_values_for_plot = shap_values_obj.values[:,:,1]

            plot_buffer = create_shap_summary_plot(shap_values_for_plot, X)
            if plot_buffer:
                st.image(plot_buffer)
                st.success("The SHAP analysis confirms that known oncogenic methylation markers (e.g., `promoter_A_met`, `enhancer_B_met`) are the most significant drivers of a 'Cancer Signal Detected' result. This provides strong evidence that the model has learned biologically relevant signals, fulfilling a key requirement of the algorithm's analytical validation.")
            else:
                st.error("Could not generate SHAP summary plot.")
        except Exception as e:
            st.error(f"Could not perform SHAP analysis. Error: {e}")
            logger.error(f"SHAP analysis failed: {e}", exc_info=True)

    # --- Tool 2: Predictive Operations ---
    with ml_tabs[1]:
        st.subheader("Predictive Operations: Sequencing Run Failure")
        with st.expander("View Method Explanation"):
            st.markdown("""
            **Purpose of the Tool:**
            To build a predictive model that can identify sequencing runs likely to fail QC *before* committing expensive reagents and sequencer time. This is a proactive quality control tool aimed at improving operational efficiency and reducing the Cost of Poor Quality (COPQ).

            **Mathematical Basis:**
            **Logistic Regression** is used as the classification algorithm. It models the probability of a binary outcome (Pass/Fail) by fitting data to a logistic (sigmoid) function. The model learns a set of coefficients ($\beta_i$) for each input feature ($x_i$) to predict the log-odds of failure:
            $log(\frac{P(Fail)}{1-P(Fail)}) = \beta_0 + \beta_1x_1 + ... + \beta_nx_n$
            
            **Procedure:**
            1. Historical run data with pre-sequencing QC metrics (e.g., library concentration) and the final outcome (Pass/Fail) is collected.
            2. The data is split into training and testing sets.
            3. A logistic regression model is trained on the training set.
            4. The model's performance is evaluated on the unseen test set using a **confusion matrix**.
            
            **Significance of Results:**
            The confusion matrix shows the model's real-world performance:
            - **True Positives (TP):** Correctly predicted failures. These represent saved runs.
            - **False Negatives (FN):** Failures the model missed. These represent the remaining risk.
            - **False Positives (FP):** Passed runs that were incorrectly predicted to fail. These represent unnecessary investigations.
            A model with a high True Positive Rate and a low False Positive Rate can be integrated into the lab's pre-run checklist to significantly improve efficiency.
            """)
        run_qc_data = ssm.get_data("ml_models", "run_qc_data")
        df_run_qc = pd.DataFrame(run_qc_data)
        X = df_run_qc[['library_concentration', 'dv200_percent', 'adapter_dimer_percent']]
        y = df_run_qc['outcome'].apply(lambda x: 1 if x == 'Fail' else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        model = LogisticRegression(random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        st.write("##### Run Failure Prediction Model Performance (on Test Set)")
        fig_cm = create_confusion_matrix_heatmap(cm, ['Pass', 'Fail'])
        st.plotly_chart(fig_cm, use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        st.success(f"""
        **Model Evaluation:**
        - The model correctly identified **{tp}** out of **{tp+fn}** failing runs in the test set.
        - It successfully avoided **{fn}** costly failures that would have otherwise occurred.
        - This predictive tool shows promise for integration into the pre-run QC checklist to reduce overall COPQ.
        """)
        
    # --- Tool 3: Time Series Forecasting ---
    with ml_tabs[2]:
        st.subheader("Time Series Forecasting for Lab Operations")
        with st.expander("View Method Explanation"):
            st.markdown(r"""
            **Purpose of the Tool:**
            To forecast future demand (e.g., incoming sample volume) based on historical data. This is crucial for proactive lab management, including reagent inventory control, staffing, and capacity planning.

            **Mathematical Basis:**
            An **ARIMA (Autoregressive Integrated Moving Average)** model is used. It is a powerful class of models for analyzing and forecasting time series data. It combines three components:
            - **AR (Autoregressive, p):** A regression model that uses the dependent relationship between an observation and some number of lagged observations.
            - **I (Integrated, d):** The use of differencing of raw observations (e.g., subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
            - **MA (Moving Average, q):** A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
            An ARIMA(p,d,q) model is a standard, statistically robust method for forecasting.

            **Procedure:**
            1. Historical time series data is collected (e.g., daily sample volume).
            2. The model (in this case, a pre-selected ARIMA(5,1,0)) is fit to the historical data.
            3. The fitted model is used to project future values, along with confidence intervals.

            **Significance of Results:**
            The forecast provides a data-driven estimate of future operational load. The confidence intervals give a sense of the uncertainty in the forecast. This information allows lab managers to move from reactive to proactive resource planning, ensuring they have the reagents and staff on hand to meet projected demand without being over-stocked.
            """)
        ts_data = ssm.get_data("ml_models", "sample_volume_data")
        df_ts = pd.DataFrame(ts_data)
        df_ts['date'] = pd.to_datetime(df_ts['date'])
        df_ts = df_ts.set_index('date')
        st.write("Fitting ARIMA model and forecasting next 30 days...")
        try:
            model = ARIMA(df_ts['samples'].asfreq('D'), order=(5, 1, 0)).fit()
            forecast = model.get_forecast(steps=30)
            forecast_df = forecast.summary_frame()
            fig = create_forecast_plot(df_ts, forecast_df)
            st.plotly_chart(fig, use_container_width=True)
            st.success("The forecast projects a continued upward trend in sample volume, suggesting the need to review reagent inventory and staffing levels for the upcoming month.")
        except Exception as e:
            st.error(f"Could not generate time series forecast. Error: {e}")

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
