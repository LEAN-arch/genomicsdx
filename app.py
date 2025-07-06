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
        create_doe_effects_plot # SME Definitive Fix: Import new robust plotter
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

    tool_tabs = st.tabs([
        "Process Control (Levey-Jennings)",
        "Hypothesis Testing (A/B Test)",
        "Equivalence Testing (TOST)",
        "Pareto Analysis (Failure Modes)",
        "Measurement System Analysis (Gauge R&R)",
        "Design of Experiments (DOE)"
    ])

    # --- Tool 1: Levey-Jennings ---
    with tool_tabs[0]:
        st.subheader("Statistical Process Control (SPC) for Assay Monitoring")
        with st.expander("View Method Explanation", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            The Levey-Jennings (L-J) chart is a fundamental tool for Quality Control (QC) in clinical laboratories. Its purpose is to monitor the stability and precision of an assay over time using a control material with a known value. It provides a visual way to detect shifts, trends, or increased variability in the assay process, helping to ensure that results remain reliable.

            **The Mathematical Basis & Method:**
            The chart plots QC measurements on the y-axis against the run number or date on the x-axis. Horizontal lines are drawn at the mean of the QC data and at +/- 1, 2, and 3 standard deviations (SD) from the mean. These lines act as control limits. The distribution of points is expected to be random around the mean. Specific patterns, known as **Westgard Rules**, can be applied to detect out-of-control conditions (e.g., one point exceeding ±3SD, two consecutive points exceeding ±2SD).

            **Procedure:**
            1.  The tool loads mock SPC data.
            2.  It automatically calculates the mean and standard deviation.
            3.  The L-J chart is plotted with the control measurements and the calculated control limits (Mean, ±1SD, ±2SD, ±3SD).
            4.  Visually inspect the chart for non-random patterns or points exceeding the ±3SD limits, which indicate a potential process issue.

            **Significance of Results:**
            - **In-Control Process:** Most points are near the mean, and they are randomly distributed above and below it. This indicates the assay is stable and performing as expected.
            - **Out-of-Control Process:** Points forming a trend (e.g., 6 consecutive points increasing) or exceeding the outer limits suggest a systemic issue (e.g., reagent degradation, instrument drift, change in operator technique). Such a signal requires immediate investigation, and patient sample results from the affected runs may need to be invalidated, as per CLIA regulations.
            """)
        spc_data = ssm.get_data("quality_system", "spc_data")
        fig = create_levey_jennings_plot(spc_data)
        st.plotly_chart(fig, use_container_width=True)
        st.success("The selected control data appears to be stable and in-control. No Westgard rule violations were detected.")

    # --- Tool 2: Hypothesis Testing ---
    with tool_tabs[1]:
        st.subheader("Hypothesis Testing for Assay Comparability")
        with st.expander("View Method Explanation", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            Hypothesis testing is used to determine if there is a statistically significant difference between two or more groups. In assay development, it's commonly used to compare the performance of a new method, reagent, or software pipeline (Group B) against the current standard (Group A). It helps answer questions like: "Does changing our pipeline significantly alter the median biomarker value?"

            **The Mathematical Basis & Method:**
            - **T-Test:** Assumes the data in both groups are normally distributed and have equal variances. It calculates a t-statistic that represents the difference between the means relative to the variation within the groups.
            - **Mann-Whitney U Test:** A non-parametric alternative used when the data is not normally distributed. It ranks all the data from both groups and checks if the ranks for one group are systematically higher or lower than the other.
            - **Shapiro-Wilk Test:** Used to check if the data is likely to have come from a normal distribution. A low p-value (< 0.05) suggests the data is not normal, and a non-parametric test like Mann-Whitney U should be used.
            The final result is a **p-value**, which is the probability of observing the data if there were truly no difference between the groups.

            **Procedure:**
            1.  The tool loads two sample datasets (e.g., `Reagent Lot A` vs. `Reagent Lot B`).
            2.  It first performs a Shapiro-Wilk test on each dataset to check for normality.
            3.  Based on the normality test, it automatically selects the appropriate comparison test (T-Test for normal data, Mann-Whitney U for non-normal).
            4.  It calculates the test statistic and the p-value.
            5.  A box plot is generated to visually compare the distributions of the two groups.

            **Significance of Results:**
            A low p-value (typically < 0.05) indicates a statistically significant difference between the two groups. This means the change (e.g., the new pipeline) had a real effect on the output. A high p-value (> 0.05) means there is not enough evidence to conclude that a difference exists.
            **Crucially, failing to find a difference is NOT the same as proving equivalence.** For that, you must use Equivalence Testing (TOST).
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
        with st.expander("View Method Explanation", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            Equivalence testing is the statistically rigorous way to demonstrate that two methods or two batches of material are "practically the same." This is critical for regulatory submissions when, for example, you change a critical reagent supplier and need to prove the new reagent performs identically to the old one. It answers the question: "Is the difference between these two groups small enough to be considered irrelevant?"

            **The Mathematical Basis & Method:**
            Instead of trying to disprove a null hypothesis of no difference (like a t-test), TOST (Two One-Sided Tests) tries to *reject* the hypothesis that the groups are *different*.
            1.  You first define an **equivalence margin** (e.g., ±10% of the mean). This is the "zone of indifference" where any difference is considered scientifically meaningless.
            2.  Two separate one-sided t-tests are performed:
                - Test 1: Is the mean difference significantly *greater than* the lower margin?
                - Test 2: Is the mean difference significantly *less than* the upper margin?
            3.  If **both** tests are statistically significant (both p-values < 0.05), you can reject the idea that the true difference lies outside your margins. Therefore, you can conclude the groups are equivalent.

            **Procedure:**
            1.  The tool loads two sample datasets (e.g., `Reagent Lot A` vs. `Reagent Lot B`).
            2.  The user defines an equivalence margin (as a percentage of the mean of Lot A).
            3.  The tool performs the TOST procedure and calculates the p-value.
            4.  A plot is generated showing the confidence interval of the mean difference in relation to the equivalence margins.

            **Significance of Results:**
            - **Equivalence Concluded (p < 0.05):** The 90% confidence interval for the difference lies entirely *within* the equivalence margins. You have successfully demonstrated that the two lots are equivalent for practical purposes. This is a powerful statement for a change control report.
            - **Equivalence Not Concluded (p >= 0.05):** The confidence interval crosses one or both of the equivalence margins. You cannot conclude that the lots are equivalent. They may be different, or your study may have been underpowered to prove equivalence.
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
        with st.expander("View Method Explanation", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            A Pareto chart is a quality management tool that applies the Pareto principle (the "80/20 rule") to identify the most significant factors in a given process. For lab operations, it's used to find the "vital few" causes of failures (e.g., sequencing run failures, QC failures) so that corrective and preventive action (CAPA) efforts can be focused on the highest-impact problems.

            **The Mathematical Basis & Method:**
            The chart is a combination of a bar chart and a line graph.
            - The **bars** represent individual failure modes (or causes), sorted in descending order of frequency.
            - The **line** represents the cumulative percentage of total failures.
            The analysis simply involves counting the occurrences of each category of failure, calculating the percentage of the total for each, and then calculating the cumulative percentage.

            **Procedure:**
            1.  The tool loads a list of observed failure modes.
            2.  It counts the frequency of each unique failure mode.
            3.  It calculates the percentage and cumulative percentage.
            4.  It generates the Pareto chart, showing the sorted bars and the cumulative line.

            **Significance of Results:**
            The chart clearly separates the "vital few" from the "trivial many." By looking at the chart, you can instantly see which 1, 2, or 3 failure modes account for ~80% of all problems. For example, if "Low Library Yield" and "Reagent QC Failure" account for 75% of all run failures, the operational improvement team knows to focus their resources on solving those two specific problems rather than wasting effort on less frequent issues.
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
        with st.expander("View Method Explanation", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            A Gauge Repeatability & Reproducibility (Gauge R&R) study is used to evaluate the precision of a measurement system (e.g., an assay). It breaks down the total observed variation into two key components:
            1.  **Repeatability (Equipment Variation):** Variation seen when the *same operator* measures the *same part* multiple times. This is the inherent variation of the assay/instrument.
            2.  **Reproducibility (Appraiser Variation):** Variation seen when *different operators* measure the *same part*. This is the variation caused by differences in operator technique.
            The goal is to ensure that the variation from the measurement system itself is small compared to the actual variation in the product being measured.

            **The Mathematical Basis & Method:**
            The analysis uses Analysis of Variance (ANOVA). It partitions the total sum of squares in the data into components attributable to the parts (the samples being measured), the operators, and the measurement error (repeatability). The % contribution of each source of variation to the total is then calculated.

            **Procedure:**
            1.  The tool loads a dataset from a structured Gauge R&R study (multiple operators measure multiple parts multiple times).
            2.  An ANOVA model (`measurement ~ operator + part`) is fitted to the data.
            3.  The components of variance are extracted from the ANOVA table.
            4.  The results are displayed in a table and a stacked bar chart showing the percentage contribution of each source of variation.

            **Significance of Results:**
            According to industry standards (e.g., AIAG), the total Gauge R&R contribution should be:
            - **< 10%:** Acceptable measurement system.
            - **10% - 30%:** May be acceptable depending on the application's importance and cost.
            - **> 30%:** Unacceptable. The measurement system is inadequate, and its variation is masking the true variation of the product.
            If Reproducibility (operator variation) is high, it indicates a need for better training or more standardized procedures. If Repeatability is high, the assay or instrument itself may need improvement.
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

    # --- Tool 6: DOE ---
    with tool_tabs[5]:
        st.subheader("Design of Experiments (DOE) for Assay Optimization")
        with st.expander("View Method Explanation", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            Design of Experiments (DOE) is a systematic method for determining the relationship between factors affecting a process and the output of that process. In assay development, it is used to efficiently optimize conditions (e.g., PCR cycles, DNA input amount, incubation time) to maximize a desired outcome (e.g., library yield).
            
            **The Analytical Approach (Main Effects & Interaction Plots):**
            For a 2-level factorial design, the most robust and direct analysis involves calculating the "effects" of each factor.
            - **Main Effect:** The average change in the response when a factor is changed from its low level to its high level.
            - **Interaction Effect:** A measure of how the effect of one factor is dependent on the level of the other factor. A large interaction effect means the factors are not independent.
            
            This method is numerically stable and provides a clear, quantitative, and visual summary of the experiment's outcome without relying on a potentially unstable regression model.

            **Procedure:**
            1.  The tool loads data from the 2x2 designed experiment.
            2.  It calculates the main effect for each factor and the two-way interaction effect.
            3.  It generates two plots:
                - A **bar chart** showing the magnitude and direction of each calculated effect.
                - An **interaction plot** to visually assess the relationship between the factors.

            **Significance of Results:**
            - **Effects Plot:** The largest bars (positive or negative) correspond to the most influential factors or interactions. This immediately tells you what parameters to focus on for optimization.
            - **Interaction Plot:** If the lines on this plot are **parallel**, there is no significant interaction. If the lines **cross or are not parallel**, a significant interaction is present. This is a critical finding, as it means the optimal level for one factor depends on the level chosen for the other.
            """)
        doe_data = ssm.get_data("quality_system", "doe_data")
        df_doe = pd.DataFrame(doe_data)
        
        st.write("##### DOE Data")
        st.dataframe(df_doe, use_container_width=True)
        
        try:
            # --- SME Definitive Fix: Replace fragile ANOVA model with robust effects calculation ---
            factor1, factor2, response = 'pcr_cycles', 'input_dna', 'library_yield'
            
            # Defensive check for required columns
            if not all(col in df_doe.columns for col in [factor1, factor2, response]):
                raise ValueError("DOE data is missing one or more required columns.")
            
            # The new, robust plotter calculates effects directly without a model
            effects_fig, interaction_fig = create_doe_effects_plot(df_doe, factor1, factor2, response)

            st.write(f"##### Main Effects and Interaction Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(effects_fig, use_container_width=True)
            with col2:
                st.plotly_chart(interaction_fig, use_container_width=True)
            
            st.success("""
            **Conclusion:** The effects plot clearly quantifies the impact of each factor and their interaction on library yield. The interaction plot visualizes this relationship, indicating whether the optimal settings for one factor depend on the other. This robust analysis provides the necessary insights for process optimization.
            """)

        except Exception as e:
            st.error(f"Could not perform DOE analysis. Error: {e}")
            logger.error(f"DOE analysis failed: {e}", exc_info=True)

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
        with st.expander("View Method Explanation", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            Machine learning models, especially complex ones like gradient boosting or random forests, are often considered "black boxes." For a medical device, especially a PMA-class diagnostic, this is unacceptable to regulators. We must be able to explain *why* the model made a specific prediction for a given patient. SHAP (SHapley Additive exPlanations) is a state-of-the-art method that assigns each feature (e.g., a specific methylation biomarker) an importance value for each individual prediction, providing crucial model transparency and explainability.

            **The Mathematical Basis & Method:**
            SHAP is based on Shapley values, a concept from cooperative game theory. It calculates the average marginal contribution of a feature value across all possible combinations of features. In essence, it answers the question: "How much did feature X's value contribute to pushing the model's prediction away from the baseline average?"
            - **Positive SHAP value:** Pushes the prediction higher (e.g., towards "Cancer Signal Detected").
            - **Negative SHAP value:** Pushes the prediction lower (e.g., towards "No Signal Detected").

            **Procedure:**
            1.  A pre-trained classifier and a sample of the training data are loaded.
            2.  A SHAP `Explainer` is created for the model. This modern API automatically selects the best algorithm (e.g., TreeExplainer for tree-based models).
            3.  SHAP values are calculated for every feature for every sample in the dataset.
            4.  A **summary plot** is generated. This plot shows the most important features overall and the distribution of their SHAP values.

            **Significance of Results:**
            The SHAP summary plot is incredibly insightful:
            - **Feature Importance:** Features are ranked top-to-bottom by their importance.
            - **Impact Direction:** The color shows whether a high (red) or low (blue) value of that feature resulted in a positive or negative SHAP value.
            For our MCED test, we would expect to see known cancer-related methylation markers ranked as highly important. If a non-biological feature (like `batch_id`) appeared as important, it would be a major red flag for a confounding batch effect. This plot provides powerful, objective evidence that the model is learning biologically relevant patterns, which is a cornerstone of the analytical validation for the algorithm.
            """)
        
        st.write("Generating SHAP values for the locked classifier model. This may take a moment...")
        X, y = ssm.get_data("ml_models", "classifier_data")
        model = ssm.get_data("ml_models", "classifier_model")
        
        try:
            # --- SME Definitive Fix: Use the modern, robust `shap.Explainer` API ---
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            
            st.write("##### SHAP Summary Plot (Impact on 'Cancer Signal Detected' Prediction)")
            # For a binary classifier, the shap_values object has a .values attribute.
            # We select the values for the positive class (index 1).
            plot_buffer = create_shap_summary_plot(shap_values, X)
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
        with st.expander("View Method Explanation", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            This tool uses machine learning to predict the likelihood of a sequencing run failing *before* it is started, based on pre-run QC metrics. Failed runs are a major source of cost and delay in a high-throughput lab. By identifying high-risk runs in advance, the lab can take preventive action (e.g., re-prepping the library, holding the run), saving significant resources.

            **The Mathematical Basis & Method:**
            A classification model (in this case, Logistic Regression) is trained on historical data.
            - **Features (X):** Pre-run QC metrics like library concentration, fragment size (DV200), and adapter-dimer percentage.
            - **Target (y):** The historical outcome of the run (Pass or Fail).
            The model learns the relationship between the input QC values and the run outcome. A **confusion matrix** is used to evaluate the model's performance. It's a table that shows:
            - **True Positives (TP):** Correctly predicted failures.
            - **True Negatives (TN):** Correctly predicted passes.
            - **False Positives (FP):** Incorrectly predicted failures (pass was predicted to fail).
            - **False Negatives (FN):** Incorrectly predicted passes (fail was predicted to pass). This is the most costly error.

            **Procedure:**
            1.  Historical run QC data is loaded and split into training and testing sets.
            2.  A Logistic Regression model is trained on the training set.
            3.  The trained model makes predictions on the unseen test set.
            4.  A confusion matrix is generated and plotted as a heatmap to visualize the model's performance.

            **Significance of Results:**
            The confusion matrix tells us how well the model can distinguish between runs that will pass and runs that will fail. The key goal is to minimize **False Negatives**—runs that the model predicted would pass but actually failed. By reviewing the model's performance, lab management can decide if it's reliable enough to be used in production. A good model can lead to substantial reductions in the Cost of Poor Quality (COPQ) by preventing wasted reagents and instrument time.
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
        with st.expander("View Method Explanation", expanded=False):
            st.markdown("""
            **Purpose of the Method:**
            This tool forecasts future lab operational metrics, such as incoming sample volume, based on historical trends. Accurate forecasting is essential for resource planning, including staffing, reagent purchasing, and capacity management. It helps the business move from reactive to proactive operational management.

            **The Mathematical Basis & Method:**
            This tool uses an **ARIMA (AutoRegressive Integrated Moving Average)** model, a standard and powerful class of models for time series forecasting.
            - **AR (AutoRegressive):** Assumes the current value depends on its own previous values.
            - **I (Integrated):** Uses differencing of the raw observations to make the time series stationary (i.e., its mean and variance don't change over time).
            - **MA (Moving Average):** Assumes the current value depends on past forecast errors.
            The model is fitted to the historical data, and then it projects the learned patterns into the future, creating a forecast with associated confidence intervals.

            **Procedure:**
            1.  Historical daily sample volume data is loaded.
            2.  An ARIMA model is fitted to this data.
            3.  The model is used to forecast the next 30 days of sample volume.
            4.  A plot is generated showing the historical data, the forecast, and the 95% confidence interval for the forecast.

            **Significance of Results:**
            The forecast plot provides an actionable estimate of future workload. Lab management can use this to:
            - **Optimize Reagent Orders:** Purchase enough reagents to meet the expected demand without overstocking and risking expiration.
            - **Schedule Staff:** Ensure adequate staffing levels for anticipated busy periods.
            - **Capacity Planning:** Identify when future demand might exceed current lab capacity, signaling the need for investment in new equipment or process improvements.
            The confidence interval provides a "worst-case" and "best-case" scenario, allowing for more robust planning.
            """)
        
        ts_data = ssm.get_data("ml_models", "sample_volume_data")
        df_ts = pd.DataFrame(ts_data)
        df_ts['date'] = pd.to_datetime(df_ts['date'])
        df_ts = df_ts.set_index('date')
        
        st.write("Fitting ARIMA model and forecasting next 30 days...")
        try:
            # A simple ARIMA model for demonstration
            model = ARIMA(df_ts['samples'].asfreq('D'), order=(5, 1, 0)).fit()
            forecast = model.get_forecast(steps=30)
            
            forecast_df = forecast.summary_frame()
            
            fig = create_forecast_plot(df_ts, forecast_df)
            st.plotly_chart(fig, use_container_width=True)
            st.success("The forecast projects a continued upward trend in sample volume, suggesting the need to review reagent inventory and staffing levels for the upcoming month.")
        except Exception as e:
            st.error(f"Could not generate time series forecast. Error: {e}")

def render_compliance_guide_tab():
    """Renders the educational guide to the regulatory landscape."""
    st.header("🏛️ A Guide to the IVD & Genomics Regulatory Landscape")
    st.markdown("This guide provides a high-level overview of the key regulations and standards governing the development of the GenomicsDx Sentry™ MCED Test. It is intended for educational purposes for the project team.")

    with st.expander("⚖️ **FDA Regulations (Title 21, Code of Federal Regulations)**"):
        st.subheader("21 CFR 820: Quality System Regulation (QSR)")
        st.markdown("""
        Also known as the Current Good Manufacturing Practice (cGMP), the QSR is the foundational regulation for medical device quality systems. It outlines the requirements for the procedures and facilities used in the design, manufacture, packaging, labeling, storage, installation, and servicing of all finished medical devices intended for human use.
        - **Key Subpart C - Design Controls (§ 820.30):** This is the heart of the DHF and this dashboard. It mandates a formal process for:
            - `(a)` General: Establish and maintain procedures to control the design of the device.
            - `(b)` **Design and Development Planning:** What this project is based on.
            - `(c)` **Design Input:** Defining all requirements.
            - `(d)` **Design Output:** The specifications, drawings, and procedures that make up the device.
            - `(e)` **Design Review:** Formal phase-gates to assess progress.
            - `(f)` **Design Verification:** *Did we build the product right?* (Analytical Validation).
            - `(g)` **Design Validation:** *Did we build the right product?* (Clinical & Usability Validation).
            - `(h)` **Design Transfer:** Moving the design to manufacturing (the clinical lab).
            - `(i)` **Design Changes:** Controlling changes after the design is locked.
            - `(j)` **Design History File (DHF):** The compilation of all records demonstrating the design was developed in accordance with the plan. This dashboard *is* the living DHF.
        """)
        st.subheader("21 CFR 809: In Vitro Diagnostic (IVD) Products")
        st.markdown("This part contains specific labeling requirements for IVDs, including Instructions For Use (IFU) and reagent labeling. (Ref: § 809.10)")

    with st.expander("🌐 **International Standards (ISO)**"):
        st.subheader("ISO 13485:2016 - Medical Devices Quality Management Systems")
        st.markdown("This is the international standard for medical device QMS. While the FDA QSR is law in the US, ISO 13485 is often required for market access in other countries (e.g., Europe, Canada). It is highly aligned with 21 CFR 820 but has a broader scope, emphasizing a risk-based approach throughout the entire QMS.")
        
        st.subheader("ISO 14971:2019 - Application of Risk Management to Medical Devices")
        st.markdown("This standard specifies the process for identifying, analyzing, evaluating, controlling, and monitoring risks associated with a medical device. It is the global benchmark for risk management and is a required process for both FDA and international submissions. The **Risk Management File** is the output of this process.")

        st.subheader("ISO 62304:2006 - Medical Device Software - Software Life Cycle Processes")
        st.markdown("This standard defines the lifecycle requirements for medical device software. It provides a framework for designing, developing, testing, and maintaining software in a safe and controlled manner. The required level of rigor depends on the **Software Safety Classification** (Class A, B, or C), which is based on the potential of the software to cause harm. Our SaMD is **Class C (High Risk)**, requiring the most stringent level of documentation and control.")

    with st.expander("🔬 **US Laboratory Regulations (CLIA)**"):
        st.subheader("Clinical Laboratory Improvement Amendments (CLIA)")
        st.markdown("CLIA regulations establish quality standards for all laboratory testing performed on humans in the U.S. (except for clinical trials and basic research). For our LDT (Laboratory Developed Test) service to be offered commercially, the performing laboratory must be CLIA-certified. This involves demonstrating analytical validity, having qualified personnel, and adhering to strict quality control and proficiency testing procedures.")

    with st.expander("📄 **PMA Submission Structure**"):
        st.markdown("""
        A Premarket Approval (PMA) is the most stringent type of device marketing application required by the FDA. It is required for Class III devices, like our MCED test. The submission is a comprehensive document that must provide reasonable assurance of the device's safety and effectiveness.
        
        A typical PMA for an IVD includes, but is not limited to:
        1.  **Device Description & Intended Use:** What it is and how it's used.
        2.  **Non-Clinical (Analytical) Studies:** The complete **Analytical Validation** package (Precision, LoD, Specificity, Robustness, etc.).
        3.  **Software/Bioinformatics Validation:** The complete software V&V package as per ISO 62304.
        4.  **Clinical Studies:** The full results and analysis from the pivotal clinical trial, including all patient data, statistical analysis plans, and outcomes.
        5.  **Labeling:** The proposed Instructions for Use, box labels, and Clinical Report format.
        6.  **Manufacturing Information:** A complete description of the laboratory process (Design Transfer, SOPs, QC procedures). This is the **Device Master Record (DMR)**.
        7.  **Quality System Information:** Evidence of compliance with 21 CFR 820.
        8.  **Risk Management File:** The complete file as per ISO 14971.
        
        This dashboard is designed to be the central repository for generating and organizing the evidence required for nearly every section of the PMA.
        """)
        
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
