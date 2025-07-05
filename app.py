# --- SME-Revised, PMA-Ready, and Unabridged Enhanced Version ---
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
from datetime import timedelta
from typing import Any, Dict, List, Tuple
import hashlib  # For deterministic seeding and data integrity checks

# --- Third-party Imports ---
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt

# --- Robust Path Correction Block ---
# Ensures modules are found, critical for structured applications.
try:
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception as e:
    st.warning(f"Could not adjust system path. Module imports may fail. Error: {e}")
# --- End of Path Correction Block ---

# --- Local Application Imports (with error handling) ---
try:
    from dhf_dashboard.analytics.action_item_tracker import render_action_item_tracker
    from dhf_dashboard.analytics.traceability_matrix import render_traceability_matrix
    from dhf_dashboard.dhf_sections import (
        design_changes, design_inputs, design_outputs, design_plan, design_reviews,
        design_risk_management, design_transfer, design_validation,
        design_verification, human_factors
    )
    # Ref: ISO 62304 / 21 CFR 820.30 - These utilities support V&V and project planning.
    from dhf_dashboard.utils.critical_path_utils import find_critical_path
    from dhf_dashboard.utils.plot_utils import (
        _RISK_CONFIG,
        create_action_item_chart, create_progress_donut, create_risk_profile_chart,
        create_roc_curve, create_levey_jennings_plot, create_lod_probit_plot, create_bland_altman_plot
    )
    from dhf_dashboard.utils.session_state_manager import SessionStateManager
except ImportError as e:
    st.error(f"Fatal Error: A required local module could not be imported: {e}. "
             "Please ensure the application is run from the correct directory and all submodules exist.")
    logging.critical(f"Fatal module import error: {e}", exc_info=True)
    st.stop()


# --- Setup Logging (Ref: 21 CFR Part 11 / ISO 13485:2016 Sec 4.2.5 - Records) ---
# Centralized logging configuration for audit trails and debugging.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# --- Module-Level Constants ---
# Centralizes DHF navigation, directly mapping to 21 CFR 820.30 sections.
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
# --- DATA PRE-PROCESSING & CACHING (Ref: ISO 62304 - Software Lifecycle) ---
# ==============================================================================

@st.cache_data
def preprocess_task_data(tasks_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Processes raw project task data into a DataFrame for Gantt chart plotting.
    Caching this expensive operation ensures a responsive UI. The tasks represent
    key milestones in the PMA submission journey (e.g., 'Lock LoD Study').
    """
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

    tasks_df['display_text'] = "<b>" + tasks_df['name'].fillna('').astype(str) + "</b> (" + \
                               tasks_df['completion_pct'].fillna(0).astype(int).astype(str) + "%)"
    return tasks_df

@st.cache_data
def get_cached_df(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generic, cached function to create DataFrames, improving performance."""
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


# ==============================================================================
# --- DASHBOARD DEEP-DIVE COMPONENT FUNCTIONS ---
# ==============================================================================

def render_dhf_completeness_panel(ssm: SessionStateManager, tasks_df: pd.DataFrame, docs_by_phase: Dict[str, pd.DataFrame]) -> None:
    """
    Renders the DHF completeness and gate readiness panel.
    Displays DHF phases as subheaders and a project timeline Gantt chart.
    Ref: 21 CFR 820.30(i) - Design History File
    """
    st.subheader("1. DHF Completeness & Phase Gate Readiness")
    st.markdown("Monitor the flow of Design Controls from inputs to outputs, including cross-functional sign-offs and DHF document status.")

    try:
        tasks_raw = ssm.get_data("project_management", "tasks")

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
                phase_docs = docs_by_phase.get(task_name)
                if phase_docs is not None and not phase_docs.empty:
                    st.dataframe(phase_docs[['id', 'title', 'status']], use_container_width=True, hide_index=True)
                else:
                    st.caption("No documents for this phase yet.")
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
            gantt_fig = px.timeline(
                tasks_df, x_start="start_date", x_end="end_date", y="name",
                color="color", color_discrete_map="identity",
                title="<b>Project Timeline and Critical Path to PMA Submission</b>",
                hover_name="name", custom_data=['status', 'completion_pct']
            )
            gantt_fig.update_traces(
                text=tasks_df['display_text'], textposition='inside', insidetextanchor='middle',
                marker_line_color=tasks_df['line_color'], marker_line_width=tasks_df['line_width'],
                hovertemplate="<b>%{hover_name}</b><br>Status: %{customdata[0]}<br>Complete: %{customdata[1]}%<extra></extra>"
            )
            gantt_fig.update_layout(
                showlegend=False, title_x=0.5, xaxis_title="Date", yaxis_title="DHF Phase / Major Milestone", height=400,
                yaxis_categoryorder='array', yaxis_categoryarray=tasks_df.sort_values("start_date", ascending=False)["name"].tolist()
            )
            st.plotly_chart(gantt_fig, use_container_width=True)
            legend_html = """
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-top: 15px; font-size: 0.9em;">
                <span><span style="display:inline-block; width:15px; height:15px; background-color:#2ca02c; margin-right: 5px; vertical-align: middle;"></span>Completed</span>
                <span><span style="display:inline-block; width:15px; height:15px; background-color:#1f77b4; margin-right: 5px; vertical-align: middle;"></span>In Progress</span>
                <span><span style="display:inline-block; width:15px; height:15px; background-color:#d62728; margin-right: 5px; vertical-align: middle;"></span>At Risk</span>
                <span><span style="display:inline-block; width:15px; height:15px; background-color:#7f7f7f; margin-right: 5px; vertical-align: middle;"></span>Not Started</span>
                <span><span style="display:inline-block; width:11px; height:11px; border: 2px solid red; margin-right: 5px; vertical-align: middle;"></span>On Critical Path</span>
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)
    except Exception as e:
        st.error("Could not render DHF Completeness Panel. Data may be missing or malformed.")
        logger.error(f"Error in render_dhf_completeness_panel: {e}", exc_info=True)

def render_risk_and_fmea_dashboard(ssm: SessionStateManager) -> None:
    """
    Renders the risk analysis dashboard, focused on patient harm from incorrect
    diagnostic results, as per ISO 14971.
    """
    st.subheader("2. DHF Risk Artifacts (ISO 14971)")
    st.markdown("Analyze the diagnostic's risk profile, focusing on mitigating potential patient harm from incorrect results (False Positives/Negatives).")

    # Ref: ISO 14971 - Risk management is a total product lifecycle process.
    risk_tabs = st.tabs(["Risk Mitigation Flow (System Level)", "Assay FMEA", "Software & Service FMEA"])
    with risk_tabs[0]:
        try:
            hazards_data = ssm.get_data("risk_management_file", "hazards")
            if not hazards_data:
                st.warning("No hazard analysis data available.")
                return
            df = get_cached_df(hazards_data)

            risk_config = _RISK_CONFIG
            get_level = lambda s, o: risk_config['levels'].get((s, o), 'High')
            df['initial_level'] = df.apply(lambda x: get_level(x.get('initial_S'), x.get('initial_O')), axis=1)
            df['final_level'] = df.apply(lambda x: get_level(x.get('final_S'), x.get('final_O')), axis=1)

            all_nodes = [f"Initial {level}" for level in risk_config['order']] + [f"Residual {level}" for level in risk_config['order']]
            node_map = {name: i for i, name in enumerate(all_nodes)}
            node_colors = [risk_config['colors'][name.split(' ')[1]] for name in all_nodes]

            links = df.groupby(['initial_level', 'final_level', 'hazard_id']).size().reset_index(name='count')
            sankey_data = links.groupby(['initial_level', 'final_level']).agg(count=('count', 'sum'), hazards=('hazard_id', lambda x: ', '.join(x))).reset_index()

            sankey_fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color=node_colors),
                link=dict(
                    source=[node_map.get(f"Initial {row['initial_level']}") for _, row in sankey_data.iterrows()],
                    target=[node_map.get(f"Residual {row['final_level']}") for _, row in sankey_data.iterrows()],
                    value=[row['count'] for _, row in sankey_data.iterrows()],
                    color=[risk_config['colors'][row['final_level']] for _, row in sankey_data.iterrows()],
                    customdata=[f"<b>{row['count']} risk(s)</b> moved from {row['initial_level']} to {row['final_level']}:<br>{row['hazards']}" for _, row in sankey_data.iterrows()],
                    hovertemplate='%{customdata}<extra></extra>'
                ))])
            sankey_fig.update_layout(title_text="<b>Risk Mitigation Flow: Initial vs. Residual Patient Harm</b>", font_size=12, height=500, title_x=0.5)
            st.plotly_chart(sankey_fig, use_container_width=True)
        except Exception as e:
            st.error("Could not render Risk Mitigation Flow. Data may be missing or malformed.")
            logger.error(f"Error in render_risk_and_fmea_dashboard (Sankey): {e}", exc_info=True)

    def render_fmea_risk_matrix_plot(fmea_data: List[Dict[str, Any]], title: str) -> None:
        """Renders an interactive Risk Matrix Bubble Chart for FMEA data."""
        st.info(f"""
        **How to read this chart:** This is a professional risk analysis tool for our diagnostic service.
        - **X-axis (Severity):** Impact of failure on patient safety/diagnosis. 1=Minor, 5=Catastrophic (e.g., missed cancer).
        - **Y-axis (Occurrence):** Likelihood of the failure mode. 1=Rare, 5=Frequent.
        - **Bubble Size (RPN):** Overall risk score (S x O x D). Bigger bubbles are higher priority.
        - **Bubble Color (Detection):** How likely are we to detect the failure *before* a result is released? **Bright red bubbles are hard-to-detect risks** and are extremely dangerous.
        
        **Your Priority:** Address items in the **top-right red zone** first. These are high-impact, high-frequency risks. Then, investigate any large, bright red bubbles regardless of their position.
        """, icon="💡")

        try:
            if not fmea_data:
                st.warning(f"No {title} data available.")
                return

            df = pd.DataFrame(fmea_data)
            df['RPN'] = df['S'] * df['O'] * df['D']

            rng = np.random.default_rng(0) # Deterministic seeding
            df['S_jitter'] = df['S'] + rng.uniform(-0.1, 0.1, len(df))
            df['O_jitter'] = df['O'] + rng.uniform(-0.1, 0.1, len(df))

            fig = go.Figure()
            # Background risk zones
            fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=5.5, y1=5.5, line=dict(width=0), fillcolor='rgba(44, 160, 44, 0.1)', layer='below')
            fig.add_shape(type="rect", x0=2.5, y0=2.5, x1=5.5, y1=5.5, line=dict(width=0), fillcolor='rgba(255, 215, 0, 0.15)', layer='below')
            fig.add_shape(type="rect", x0=3.5, y0=3.5, x1=5.5, y1=5.5, line=dict(width=0), fillcolor='rgba(255, 127, 14, 0.15)', layer='below')
            fig.add_shape(type="rect", x0=4.5, y0=4.5, x1=5.5, y1=5.5, line=dict(width=0), fillcolor='rgba(214, 39, 40, 0.15)', layer='below')

            fig.add_trace(go.Scatter(
                x=df['S_jitter'], y=df['O_jitter'],
                mode='markers+text', text=df['id'], textposition='top center', textfont=dict(size=9, color='#444'),
                marker=dict(
                    size=df['RPN'], sizemode='area', sizeref=2. * max(df['RPN']) / (40.**2), sizemin=4,
                    color=df['D'], colorscale='YlOrRd', colorbar=dict(title='Detection'),
                    showscale=True, line_width=1, line_color='black'
                ),
                customdata=df[['failure_mode', 'potential_effect', 'S', 'O', 'D', 'RPN', 'mitigation']],
                hovertemplate="""<b>%{customdata[0]}</b><br>--------------------------------<br><b>Effect:</b> %{customdata[1]}<br><b>S:</b> %{customdata[2]} | <b>O:</b> %{customdata[3]} | <b>D:</b> %{customdata[4]}<br><b>RPN: %{customdata[5]}</b><br><b>Mitigation:</b> %{customdata[6]}<extra></extra>"""
            ))

            fig.update_layout(
                title=f"<b>{title} Risk Landscape</b>", xaxis_title="Severity (S) of Patient Harm", yaxis_title="Occurrence (O) of Failure",
                xaxis=dict(range=[0.5, 5.5], tickvals=list(range(1, 6))), yaxis=dict(range=[0.5, 5.5], tickvals=list(range(1, 6))),
                height=600, title_x=0.5, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        except (KeyError, TypeError) as e:
            st.error(f"Could not render {title} Risk Matrix. Data may be malformed or missing S, O, D columns.")
            logger.error(f"Error in render_fmea_risk_matrix_plot for {title}: {e}", exc_info=True)
    with risk_tabs[1]:
        render_fmea_risk_matrix_plot(ssm.get_data("risk_management_file", "assay_fmea"), "Assay FMEA (Wet Lab)")
    with risk_tabs[2]:
        render_fmea_risk_matrix_plot(ssm.get_data("risk_management_file", "service_fmea"), "Software & Service FMEA (Dry Lab & Ops)")

def render_assay_and_ops_readiness_panel(ssm: SessionStateManager) -> None:
    """Renders the Assay Performance and Lab Operations readiness panel."""
    st.subheader("3. Assay & Lab Operations Readiness")
    st.markdown("This section tracks key activities bridging R&D with a robust, scalable, and CLIA-compliant diagnostic service.")
    qbd_tabs = st.tabs(["Analytical Performance & Controls", "CLIA Lab & Ops Readiness"])
    with qbd_tabs[0]:
        st.markdown("**Tracking Critical Assay Parameters (CAPs) & Performance**")
        st.caption("Monitoring the key assay characteristics that ensure robust and reliable performance.")
        try:
            assay_params = ssm.get_data("assay_performance", "parameters")
            if not assay_params: st.warning("No Critical Assay Parameters have been defined.")
            else:
                for param in assay_params:
                    st.subheader(f"CAP: {param.get('parameter', 'N/A')}")
                    st.caption(f"(Links to Requirement: {param.get('links_to_req', 'N/A')})")
                    st.markdown(f"**Associated Control Metric:** `{param.get('control_metric', 'N/A')}`")
                    st.markdown(f"**Acceptance Criteria:** `{param.get('acceptance_criteria', 'N/A')}`")
                    st.divider()
            st.info("💡 A well-understood relationship between CAPs and the final test result is the foundation of a robust assay, as required by 21 CFR 820.30 and ISO 13485.", icon="💡")
        except Exception as e:
            st.error("Could not render Analytical Performance panel."); logger.error(f"Error in render_assay_and_ops_readiness_panel (Assay): {e}", exc_info=True)
    with qbd_tabs[1]:
        st.markdown("**Tracking Key Lab Operations & Validation Status**")
        st.caption("Ensuring the laboratory environment is validated and ready for high-throughput clinical testing.")
        try:
            lab_ops_data = ssm.get_data("lab_operations", "readiness")
            if not lab_ops_data: st.warning("No Lab Operations readiness data available.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Reagent Lot Qualification**")
                    lot_qual = lab_ops_data.get('reagent_lot_qualification', {})
                    total, passed = lot_qual.get('total', 0), lot_qual.get('passed', 0)
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
        except Exception as e:
            st.error("Could not render CLIA Lab readiness panel."); logger.error(f"Error in render_assay_and_ops_readiness_panel (Lab Ops): {e}", exc_info=True)


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
        except Exception as e:
            st.error("Could not render Audit Readiness Scorecard."); logger.error(f"Error in render_audit_and_improvement_dashboard (Scorecard): {e}", exc_info=True)
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
                    meas = np.array(spc_data['measurements']); usl, lsl = spc_data['usl'], spc_data['lsl']
                    mu, sigma = meas.mean(), meas.std()
                    cpk = min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma)) if sigma > 0 else 0
                    st.metric("Process Capability (Cpk)", f"{cpk:.2f}", delta=f"{cpk-1.33:.2f} vs. target 1.33", delta_color="normal", help="A Cpk > 1.33 indicates a capable process for this control metric. Calculated from live SPC data.")
                else: st.metric("Process Capability (Cpk)", "N/A", help="SPC data missing.")
                st.caption("Increased Cpk from process optimization (DOE) directly reduces failed runs and COPQ.")
        except Exception as e:
            st.error("Could not render Assay Performance & COPQ Dashboard."); logger.error(f"Error in render_audit_and_improvement_dashboard (COPQ): {e}", exc_info=True)

# ==============================================================================
# --- TAB RENDERING FUNCTIONS ---
# ==============================================================================

def render_health_dashboard_tab(ssm: SessionStateManager, tasks_df: pd.DataFrame, docs_by_phase: Dict[str, pd.DataFrame]):
    """
    Renders the main DHF Health Dashboard tab, tailored for executive assessment
    of the MCED diagnostic's journey to PMA approval.
    """
    st.header("Executive Health Summary")

    # --- Health Score & KHI Calculation (PMA-Focused) ---
    schedule_score = 0
    if not tasks_df.empty:
        today = pd.Timestamp.now().floor('D')
        overdue_in_progress = tasks_df[(tasks_df['status'] == 'In Progress') & (tasks_df['end_date'] < today)]
        total_in_progress = tasks_df[tasks_df['status'] == 'In Progress']
        schedule_score = (1 - (len(overdue_in_progress) / len(total_in_progress))) * 100 if not total_in_progress.empty else 100

    hazards_df = get_cached_df(ssm.get_data("risk_management_file", "hazards")); risk_score = 0
    if not hazards_df.empty and all(c in hazards_df.columns for c in ['initial_S', 'initial_O', 'initial_D', 'final_S', 'final_O', 'final_D']):
        hazards_df['initial_rpn'] = hazards_df['initial_S'] * hazards_df['initial_O'] * hazards_df['initial_D']
        hazards_df['final_rpn'] = hazards_df['final_S'] * hazards_df['final_O'] * hazards_df['final_D']
        initial_rpn_sum = hazards_df['initial_rpn'].sum(); final_rpn_sum = hazards_df['final_rpn'].sum()
        risk_reduction_pct = ((initial_rpn_sum - final_rpn_sum) / initial_rpn_sum) * 100 if initial_rpn_sum > 0 else 100
        risk_score = max(0, risk_reduction_pct)

    reviews_data = ssm.get_data("design_reviews", "reviews")
    original_action_items = [item for r in reviews_data for item in r.get("action_items", [])]
    action_items_df = get_cached_df(original_action_items)

    execution_score = 100
    if not action_items_df.empty:
        open_items = action_items_df[action_items_df['status'] != 'Completed']
        if not open_items.empty:
            overdue_items_count = len(open_items[open_items['status'] == 'Overdue'])
            execution_score = (1 - (overdue_items_count / len(open_items))) * 100

    weights = {'schedule': 0.4, 'quality': 0.4, 'execution': 0.2}
    overall_health_score = (schedule_score * weights['schedule']) + (risk_score * weights['quality']) + (execution_score * weights['execution'])

    # --- MCED-SPECIFIC KHIs ---
    ver_tests_df = get_cached_df(ssm.get_data("design_verification", "tests"))
    val_studies_df = get_cached_df(ssm.get_data("design_validation", "studies"))
    total_av = len(ver_tests_df)
    passed_av = len(ver_tests_df[ver_tests_df['status'] == 'Completed'])
    av_pass_rate = (passed_av / total_av) * 100 if total_av > 0 else 0

    reqs_df = get_cached_df(ssm.get_data("design_inputs", "requirements"))
    ver_tests_with_links = ver_tests_df.dropna(subset=['input_verified_id'])['input_verified_id'].nunique()
    total_reqs = reqs_df['id'].nunique()
    trace_coverage = (ver_tests_with_links / total_reqs) * 100 if total_reqs > 0 else 0

    study_df = get_cached_df(ssm.get_data("clinical_study", "enrollment"))
    enrollment_rate = 0
    if not study_df.empty:
        enrolled = study_df['enrolled'].sum()
        target = study_df['target'].sum()
        enrollment_rate = (enrolled / target) * 100 if target > 0 else 0
    
    overdue_actions_count = len(action_items_df[action_items_df['status'] == 'Overdue']) if not action_items_df.empty else 0

    # --- Render Dashboard ---
    col1, col2 = st.columns([1.5, 2])
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = overall_health_score, title = {'text': "<b>Overall Program Health Score</b>"},
            number = {'font': {'size': 48}}, domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "green" if overall_health_score > 80 else "orange" if overall_health_score > 60 else "red"},
                     'steps' : [{'range': [0, 60], 'color': "#fdecec"}, {'range': [60, 80], 'color': "#fef3e7"}, {'range': [80, 100], 'color': "#eaf5ea"}]}))
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
    with khi_col1:
        st.metric(label="Analytical Validation (AV) Rate", value=f"{av_pass_rate:.1f}%", help="Percentage of all planned Analytical Verification protocols that are complete and passing. (Ref: 21 CFR 820.30(f))"); st.progress(av_pass_rate / 100)
    with khi_col2:
        st.metric(label="Pivotal Study Enrollment", value=f"{enrollment_rate:.1f}%", help="Enrollment progress for the pivotal clinical trial required for PMA submission."); st.progress(enrollment_rate / 100)
    with khi_col3:
        st.metric(label="Requirement-to-V&V Traceability", value=f"{trace_coverage:.1f}%", help="Percentage of requirements traced to a verification or validation activity. (Ref: 21 CFR 820.30(g))"); st.progress(trace_coverage / 100)
    with khi_col4:
        st.metric(label="Overdue Action Items", value=overdue_actions_count, delta=overdue_actions_count, delta_color="inverse", help="Total number of action items from all design reviews that are past their due date.")
    st.divider()
    st.subheader("Action Item Health (Last 30 Days)")
    st.markdown("This chart shows the trend of open action items. A healthy project shows a downward or stable trend. A rising red area indicates a growing backlog of overdue work, which requires management attention.")

    @st.cache_data
    def generate_burndown_data(_reviews_data: Tuple, _action_items_data: Tuple):
        """Generates deterministic, cached burndown chart data from action items."""
        if not _action_items_data:
            return pd.DataFrame()
        action_items_list = [dict(fs) for fs in _action_items_data]
        reviews_list = [dict(fs) for fs in _reviews_data]
        df = pd.DataFrame(action_items_list)
        for review in reviews_list:
            review_date = pd.to_datetime(review.get('date'))
            action_items_in_review = [dict(item_fs) for item_fs in review.get("action_items", [])]
            for item in action_items_in_review:
                if 'id' in item:
                    df.loc[df['id'] == item['id'], 'review_date'] = review_date
        df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
        df['created_date'] = pd.to_datetime(df.get('review_date'), errors='coerce')
        df.dropna(subset=['created_date', 'due_date', 'id'], inplace=True)
        def get_deterministic_offset(item_id):
            return int(hashlib.md5(str(item_id).encode()).hexdigest(), 16) % 3
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
            else:
                overdue_count = 0; ontime_count = 0
            daily_counts.append({'date': day, 'Overdue': overdue_count, 'On-Time': ontime_count})
        return pd.DataFrame(daily_counts)

    if original_action_items:
        immutable_actions = tuple(frozenset(d.items()) for d in original_action_items)
        immutable_reviews = tuple(frozenset(
            (k, tuple(frozenset(i.items()) for i in v) if isinstance(v, list) else v)
            for k, v in r.items()
        ) for r in reviews_data)
        burndown_df = generate_burndown_data(immutable_reviews, immutable_actions)
        if not burndown_df.empty:
            fig = px.area(burndown_df, x='date', y=['On-Time', 'Overdue'],
                          color_discrete_map={'On-Time': 'seagreen', 'Overdue': 'crimson'},
                          title="Trend of Open Action Items by Status",
                          labels={'value': 'Number of Open Items', 'date': 'Date', 'variable': 'Status'})
            fig.update_layout(height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.caption("No action item data to generate a burn-down chart.")
    else:
        st.caption("No action item data to generate a burn-down chart.")
    
    st.divider()
    st.header("Deep Dives")
    with st.expander("Expand to see Phase Gate Readiness & Timeline Details"):
        render_dhf_completeness_panel(ssm, tasks_df, docs_by_phase)
    with st.expander("Expand to see Risk & FMEA Details"):
        render_risk_and_fmea_dashboard(ssm)
    with st.expander("Expand to see Assay Performance and Lab Operations Readiness Details"):
        render_assay_and_ops_readiness_panel(ssm)
    with st.expander("Expand to see Audit & Continuous Improvement Details"):
        render_audit_and_improvement_dashboard(ssm)

def render_dhf_explorer_tab(ssm: SessionStateManager):
    """Renders the DHF Sections Explorer tab and its sidebar navigation."""
    st.header("🗂️ Design History File Explorer")
    st.markdown("Select a DHF section from the sidebar to view its contents. Each section corresponds to a requirement under **21 CFR 820.30**.")
    with st.sidebar:
        st.header("DHF Section Navigation")
        dhf_selection = st.radio("Select a section to view:", DHF_EXPLORER_PAGES.keys(), key="sidebar_dhf_selection")
    st.divider()
    page_function = DHF_EXPLORER_PAGES[dhf_selection]
    page_function(ssm)

def render_advanced_analytics_tab(ssm: SessionStateManager):
    """Renders the Advanced Analytics tab."""
    st.header("🔬 Advanced Compliance & Project Analytics")
    analytics_tabs = st.tabs(["Traceability Matrix", "Action Item Tracker", "Project Task Editor"])
    with analytics_tabs[0]: render_traceability_matrix(ssm)
    with analytics_tabs[1]: render_action_item_tracker(ssm)
    with analytics_tabs[2]:
        st.subheader("Project Timeline and Task Editor")
        st.warning("Directly edit project timelines, statuses, and dependencies. All changes are logged and versioned under the QMS.", icon="⚠️")
        try:
            tasks_data_to_edit = ssm.get_data("project_management", "tasks")
            if not tasks_data_to_edit:
                st.info("No tasks to display or edit.")
                return

            tasks_df_to_edit = pd.DataFrame(tasks_data_to_edit)
            tasks_df_to_edit['start_date'] = pd.to_datetime(tasks_df_to_edit['start_date'], errors='coerce')
            tasks_df_to_edit['end_date'] = pd.to_datetime(tasks_df_to_edit['end_date'], errors='coerce')
            
            original_df = tasks_df_to_edit.copy()
            edited_df = st.data_editor(
                tasks_df_to_edit, key="main_task_editor", num_rows="dynamic", use_container_width=True,
                column_config={"start_date": st.column_config.DateColumn("Start Date", format="YYYY-MM-DD", required=True), "end_date": st.column_config.DateColumn("End Date", format="YYYY-MM-DD", required=True)})
            
            if not original_df.equals(edited_df):
                df_to_save = edited_df.copy()
                df_to_save['start_date'] = pd.to_datetime(df_to_save['start_date']).dt.strftime('%Y-%m-%d')
                df_to_save['end_date'] = pd.to_datetime(df_to_save['end_date']).dt.strftime('%Y-%m-%d')
                
                df_to_save = df_to_save.replace({pd.NaT: None})

                ssm.update_data(df_to_save.to_dict('records'), "project_management", "tasks")
                st.toast("Project tasks updated! Rerunning...", icon="✅")
                st.rerun()
        except Exception as e: 
            st.error("Could not load the Project Task Editor.")
            logger.error(f"Error in task editor: {e}", exc_info=True)

def render_statistical_tools_tab(ssm: SessionStateManager):
    """Renders the Statistical Workbench with tools relevant to genomic assay validation."""
    st.header("📈 Statistical Workbench for Assay & Lab Development")
    st.info("Utilize this interactive workbench for rigorous statistical analysis of assay performance, a cornerstone of the Analytical Validation required for a PMA.")
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from scipy.stats import shapiro, mannwhitneyu, chi2_contingency, pearsonr
    except ImportError:
        st.error("This tab requires `statsmodels` and `scipy`. Please install them (`pip install statsmodels scipy`) to enable statistical tools.", icon="🚨"); return
    
    tool_tabs = st.tabs([
        "Process Control (Levey-Jennings)", "Hypothesis Testing", "Pareto Analysis (Failure Modes)", "Design of Experiments (DOE)",
        "Gauge R&R (MSA)", "Chi-Squared Test", "Correlation Analysis", "Equivalence Test (TOST)"
    ])

    with tool_tabs[0]: # SPC
        st.subheader("Statistical Process Control (SPC) for Assay Monitoring")
        st.markdown("Monitor assay stability using Levey-Jennings charts for quality control samples, a key requirement under **CLIA** and **ISO 15189**.")
        with st.expander("The Purpose, Math Basis, Procedure, and Significance"):
            st.markdown("#### Purpose: Monitor and Control the Analytical Process")
            st.markdown("The purpose of SPC in a clinical lab is to ensure the analytical process (our assay) remains in a state of statistical control. It allows us to monitor the performance of control materials run-to-run, distinguishing normal process variation from 'special cause' variation that could indicate a problem with reagents, equipment, or operators, potentially compromising patient results.")
            st.markdown("#### The Mathematical Basis: Mean and Standard Deviation (Levey-Jennings)")
            st.markdown("A Levey-Jennings chart is a specific application of an SPC chart. For a given control material:\n- **Centerline (CL):** The established mean (μ) of the control material's measured value.\n- **Control Limits (UCL/LCL):** These are typically set at `μ ± 2σ` (warning limits) and `μ ± 3σ` (action limits). A point outside ±3σ is a strong signal of a process issue.")
            st.markdown("#### The Procedure: Charting and Westgard Rules")
            st.markdown("1.  **Establish Limits:** Run the control material multiple times (~20) under stable conditions to calculate its mean (μ) and standard deviation (σ).\n2.  **Ongoing Monitoring:** In each subsequent assay run, measure the control material and plot its value on the chart.\n3.  **Rule Application:** Apply a set of rules (like the **Westgard Rules**) to detect systematic errors or increased random error, even if limits are not breached.")
            st.markdown("#### Significance: In-Control vs. Out-of-Control")
            st.markdown("- **In-Control:** The run is valid. Patient results can be reported.\n- **Out-of-Control:** A rule is violated. The run is rejected. Patient results must be held. An investigation (CAPA) is initiated to find and fix the root cause before re-running the samples.")
        
        spc_data = ssm.get_data("quality_system", "spc_data")
        fig = create_levey_jennings_plot(spc_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with tool_tabs[1]: # Hypothesis Testing
        st.subheader("Hypothesis Testing for Assay Comparability")
        st.markdown("Rigorously determine if a statistically significant difference exists between two groups (e.g., comparing two bioinformatics pipeline versions).")
        with st.expander("The Purpose, Math Basis, Procedure, and Significance"):
            st.markdown("#### Purpose: Make Objective Decisions")
            st.markdown("Hypothesis testing provides a formal framework to objectively compare assay conditions or versions. It's used to decide if a change in the process (e.g., a new bioinformatics pipeline) has a real, statistically significant impact on a key output metric (e.g., Tumor Fraction). This is fundamental for justifying design changes under 21 CFR 820.30.")
            st.markdown("#### The Mathematical Basis: Null vs. Alternative Hypothesis")
            st.markdown("- **Null Hypothesis (H₀):** Assumes no difference exists (e.g., the mean Tumor Fraction from Pipeline A is equal to Pipeline B).\n- **Alternative Hypothesis (H₁):** The claim we want to prove (e.g., the means are not equal).")
            st.markdown("#### The Procedure: From Question to Conclusion")
            st.markdown("1.  **Formulate Hypotheses**.\n2.  **Check Assumptions:** Verify data normality. If not normal, use a non-parametric test (e.g., Mann-Whitney U).\n3.  **Calculate P-value:** The probability of observing our data if H₀ were true.")
            st.markdown("#### Significance: The P-Value and Alpha (α)")
            st.markdown("- **`p < α` (typically 0.05):** Statistically significant. We **reject H₀**. We conclude a real difference exists.\n- **`p ≥ α`:** Not statistically significant. We **fail to reject H₀**. We lack sufficient evidence to claim a difference.")
        try:
            ht_data = ssm.get_data("quality_system", "hypothesis_testing_data")
            if not ht_data:
                st.info("Displaying example data.", icon="ℹ️")
                rng = np.random.default_rng(0)
                ht_data = {
                    'pipeline_a': list(rng.normal(0.012, 0.005, 30)),
                    'pipeline_b': list(rng.normal(0.010, 0.005, 30))
                }
            if ht_data and all(k in ht_data for k in ['pipeline_a', 'pipeline_b']):
                line_a, line_b = ht_data['pipeline_a'], ht_data['pipeline_b']
                shapiro_a = shapiro(line_a); shapiro_b = shapiro(line_b); is_normal = shapiro_a.pvalue > 0.05 and shapiro_b.pvalue > 0.05
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**1. Assumption Check (Normality)**"); st.caption(f"Shapiro-Wilk p-value (Pipeline A): {shapiro_a.pvalue:.3f}"); st.caption(f"Shapiro-Wilk p-value (Pipeline B): {shapiro_b.pvalue:.3f}")
                    st.markdown("**2. Statistical Test Execution**")
                    if is_normal:
                        st.info("Data appears normal. Performing Welch's t-test.", icon="✅"); stat, p_value = stats.ttest_ind(line_a, line_b, equal_var=False); test_name = "Welch's t-test"
                    else:
                        st.warning("Data not normal. Switching to Mann-Whitney U test.", icon="⚠️"); stat, p_value = mannwhitneyu(line_a, line_b); test_name = "Mann-Whitney U test"
                    st.metric(f"{test_name} Statistic", f"{stat:.3f}"); st.metric("P-value", f"{p_value:.3f}")
                    st.markdown("**3. Conclusion**")
                    if p_value < 0.05:
                        st.success(f"**Conclusion:** A statistically significant difference exists between the two pipeline versions (p < 0.05).")
                    else:
                        st.warning(f"**Conclusion:** No statistically significant difference detected (p >= 0.05).")
                with col2:
                    df_ht = pd.concat([pd.DataFrame({'value': line_a, 'pipeline': 'Pipeline A'}), pd.DataFrame({'value': line_b, 'pipeline': 'Pipeline B'})])
                    fig = px.box(df_ht, x='pipeline', y='value', title="Distribution of Tumor Fraction Estimates", points="all", labels={'value': 'Tumor Fraction (TF)'}); st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Hypothesis testing data is incomplete or missing.")
        except Exception as e:
            st.error("Could not perform Hypothesis Test."); logger.error(f"Error in Hypothesis Testing tool: {e}", exc_info=True)
    
    with tool_tabs[2]: # Pareto Analysis
        st.subheader("Pareto Analysis of Sequencing Run Failures")
        st.markdown("Apply the 80/20 rule to identify the 'vital few' failure modes that cause the majority of costly failed sequencing runs.")
        with st.expander("The Purpose, Math Basis, Procedure, and Significance"):
            st.markdown("#### Purpose: Prioritize Lab Improvement Efforts")
            st.markdown("Pareto Analysis helps the lab operations team focus their resources. By identifying the 20% of failure modes that cause 80% of the problems (like failed runs, which have a high Cost of Poor Quality), we can maximize the impact of our corrective and preventive actions (CAPAs).")
        try:
            fmea_df = get_cached_df(ssm.get_data("lab_operations", "run_failures"))
            if not fmea_df.empty:
                pareto_data = fmea_df['failure_mode'].value_counts().reset_index()
                pareto_data.columns = ['failure_mode', 'count']
                pareto_data = pareto_data.sort_values('count', ascending=False)
                pareto_data['cumulative_pct'] = (pareto_data['count'].cumsum() / pareto_data['count'].sum()) * 100
                fig = go.Figure(); fig.add_trace(go.Bar(x=pareto_data['failure_mode'], y=pareto_data['count'], name='Failure Count', marker_color='#1f77b4'))
                fig.add_trace(go.Scatter(x=pareto_data['failure_mode'], y=pareto_data['cumulative_pct'], name='Cumulative %', yaxis='y2', line=dict(color='#d62728')))
                fig.update_layout(title="Pareto Chart: Root Causes of Assay Run Failures", yaxis=dict(title='Count of Failed Runs'), yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 105]), xaxis_title='Failure Mode', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No run failure data available for Pareto analysis.")
        except Exception as e:
            st.error("Could not generate Pareto chart."); logger.error(f"Error in Pareto Analysis tool: {e}", exc_info=True)
    
    with tool_tabs[3]: # Design of Experiments
        st.subheader("Design of Experiments (DOE) for Assay Optimization")
        st.markdown("Efficiently determine which assay parameters (**factors**) significantly impact a key performance metric (**response**), like library yield.")
        with st.expander("The Purpose, Math Basis, Procedure, and Significance"):
            st.markdown("#### Purpose: Understand and Optimize the Assay")
            st.markdown("DOE is a powerful statistical method to optimize our wet lab assay. It helps us understand how factors like `Input DNA Amount` and `PCR Cycles` (and their interactions) affect a critical response like `Library Yield`. This is far more efficient than 'one-factor-at-a-time' testing and is a cornerstone of robust assay development.")
            st.markdown("#### The Math Basis: Regression and ANOVA")
            st.markdown("We fit a mathematical model (`Response = f(FactorA, FactorB, ...)`) to the data. Analysis of Variance (ANOVA) then tells us which factors have a statistically significant effect on the response (i.e., a p-value < 0.05).")
            st.markdown("#### Significance: Finding the Optimal Recipe")
            st.markdown("The output helps us identify the key process drivers and create a 'response surface' map. This map guides us to the optimal settings to maximize yield and create a robust, reliable assay, which is a key goal of Design Controls (21 CFR 820.30).")
        try:
            doe_df = get_cached_df(ssm.get_data("quality_system", "doe_data"))
            if not doe_df.empty:
                formula = 'library_yield ~ pcr_cycles * input_dna'; model = ols(formula, data=doe_df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.markdown("**Analysis of Variance (ANOVA) Table**"); st.caption("This table shows which factors significantly impact Library Yield. Look for p-values (PR(>F)) < 0.05.")
                st.dataframe(anova_table.style.map(lambda x: 'background-color: #eaf5ea' if x < 0.05 else '', subset=['PR(>F)']))
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Main Effects Plot**"); st.caption("Average effect of changing each factor.")
                    main_effects_data = doe_df.melt(id_vars='library_yield', value_vars=['pcr_cycles', 'input_dna'], var_name='factor', value_name='level')
                    main_effects = main_effects_data.groupby(['factor', 'level'])['library_yield'].mean().reset_index()
                    fig = px.line(main_effects, x='level', y='library_yield', color='factor', title="Main Effects on Library Yield", markers=True, labels={'level': 'Factor Level', 'library_yield': 'Mean Library Yield (ng)'}); st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("**Response Surface Contour Plot**"); st.caption("Predicted yield across the design space.")
                    t_range, p_range = np.linspace(doe_df['pcr_cycles'].min(), doe_df['pcr_cycles'].max(), 30), np.linspace(doe_df['input_dna'].min(), doe_df['input_dna'].max(), 30); t_grid, p_grid = np.meshgrid(t_range, p_range)
                    grid = pd.DataFrame({'pcr_cycles': t_grid.ravel(), 'input_dna': p_grid.ravel()}); yield_grid = model.predict(grid).values.reshape(t_grid.shape)
                    opt_idx = np.unravel_index(np.argmax(yield_grid), yield_grid.shape)
                    opt_temp, opt_press = t_range[opt_idx[1]], p_range[opt_idx[0]]; opt_strength = yield_grid.max()
                    fig = go.Figure(data=[go.Contour(z=yield_grid, x=t_range, y=p_range, colorscale='Viridis')])
                    fig.add_trace(go.Scatter(x=doe_df['pcr_cycles'], y=doe_df['input_dna'], mode='markers', marker=dict(color='black', size=10, symbol='x'), name='DOE Runs'))
                    fig.add_trace(go.Scatter(x=[opt_temp], y=[opt_press], mode='markers+text', marker=dict(color='red', size=16, symbol='star'), text=[' Optimum'], textposition="top right", name='Predicted Optimum'))
                    fig.update_layout(xaxis_title="PCR Cycles", yaxis_title="Input DNA (ng)", title=f"Predicted Library Yield (Max: {opt_strength:.1f} ng)"); st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("DOE data is not available.")
        except Exception as e:
            st.error("Could not generate DOE plots."); logger.error(f"Error in DOE Analysis tool: {e}", exc_info=True)

    with tool_tabs[4]: # Gauge R&R
        st.subheader("Measurement System Analysis (Gauge R&R)")
        st.markdown("Quantify the precision of your assay by evaluating its **Repeatability** and **Reproducibility**. A core component of Analytical Validation.")
        with st.expander("The 'Why', 'How', and 'So What?'"):
            st.markdown("#### Purpose: Can You Trust Your Assay?")
            st.markdown("A Gauge R&R study determines how much of the variation in your results comes from the measurement system itself versus the actual biological variation in the samples. An imprecise assay is useless. This study is a non-negotiable part of the analytical validation package for the FDA.")
            st.markdown("#### The Math Basis: ANOVA on a Crossed Design")
            st.markdown("We use ANOVA to partition the total observed variation into its sources:\n- **Repeatability (Gauge):** Variation when one operator measures the same part multiple times.\n- **Reproducibility (Operator):** Variation when different operators measure the same part.\n- **Part-to-Part:** The true variation between samples, which we *want* to be the largest component.")
            st.markdown("#### Significance of the Results: Go / No-Go for the Assay")
            st.markdown("""Key metrics from AIAG guidelines:
            - **%Contribution (Gauge R&R):** The percentage of total variance from the measurement system.
              - **< 1%:** Excellent.
              - **1% - 9%:** Acceptable.
              - **> 9%:** Unacceptable. The assay protocol or training needs improvement.
            - **Number of Distinct Categories (ndc):** How many distinct groups the assay can reliably distinguish.
              - **`ndc` >= 5:** Acceptable. The assay is effective for its intended purpose.
            """)
        try:
            msa_data_list = ssm.get_data("quality_system", "msa_data")
            if not msa_data_list:
                st.info("Displaying example data.", icon="ℹ️")
                rng = np.random.default_rng(0)
                parts_mock = np.repeat(np.arange(1, 11), 6) # 10 samples
                operators_mock = np.tile(np.repeat(['Tech A', 'Tech B', 'Tech C'], 2), 10) # 3 techs, 2 reps
                part_means = np.linspace(0.01, 0.05, 10) # Different tumor fractions
                op_bias = {'Tech A': -0.002, 'Tech B': 0, 'Tech C': 0.003}
                measurements = []
                for i, part_id in enumerate(parts_mock):
                    op_name = operators_mock[i]
                    base_val = part_means[part_id - 1] + op_bias[op_name]
                    measurements.append(base_val + rng.normal(0, 0.005)) # 0.005 is assay error
                msa_data_list = pd.DataFrame({'sample': parts_mock, 'operator': operators_mock, 'measurement': measurements}).to_dict('records')

            if msa_data_list and all(k in msa_data_list[0] for k in ['sample', 'operator', 'measurement']):
                df = pd.DataFrame(msa_data_list)
                model = ols('measurement ~ C(sample) + C(operator) + C(sample):C(operator)', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                anova_table.columns = [col.lower().strip().replace('pr(>f)', 'p_value') for col in anova_table.columns]
                anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']
                
                ms_operator = anova_table.loc['C(operator)', 'mean_sq']; ms_part = anova_table.loc['C(sample)', 'mean_sq']
                ms_interact = anova_table.loc['C(sample):C(operator)', 'mean_sq']; ms_error = anova_table.loc['Residual', 'mean_sq']
                
                n_parts, n_ops = df['sample'].nunique(), df['operator'].nunique()
                n_reps = len(df) / (n_parts * n_ops) if (n_parts * n_ops) > 0 else 0

                var_repeat = ms_error
                var_reprod = max(0, (ms_operator - ms_interact) / (n_parts * n_reps))
                var_interact = max(0, (ms_interact - ms_error) / n_reps)
                var_gaugeRR = var_repeat + var_reprod + var_interact
                var_part = max(0, (ms_part - ms_interact) / (n_ops * n_reps))
                var_total = var_gaugeRR + var_part

                if var_total > 1e-9:
                    contrib_gauge = (var_gaugeRR / var_total) * 100; contrib_part = (var_part / var_total) * 100
                    ndc = int(1.41 * (np.sqrt(var_part) / np.sqrt(var_gaugeRR))) if var_gaugeRR > 1e-9 else float('inf')

                    st.markdown("**Assay R&R Results (Tumor Fraction Estimate)**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Gauge R&R % Contribution", f"{contrib_gauge:.2f}%", help="<9% is acceptable.")
                        st.metric("Number of Distinct Categories (ndc)", f"{ndc}", help=">=5 is acceptable.")
                        if contrib_gauge > 9 or ndc < 5:
                            st.error("**Conclusion: Assay precision is UNACCEPTABLE.**", icon="🚨")
                        else:
                            st.success("**Conclusion: Assay precision is ACCEPTABLE.**", icon="✅")
                    with col2:
                        var_df = pd.DataFrame({'Source': ['Assay R&R', 'Sample-to-Sample'], 'Contribution (%)': [contrib_gauge, contrib_part]})
                        fig = px.bar(var_df, x='Source', y='Contribution (%)', title="Variance Contribution", text_auto='.2f', color='Source', color_discrete_map={'Assay R&R': 'crimson', 'Sample-to-Sample': 'seagreen'})
                        fig.update_layout(showlegend=False); st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not calculate variance components.")
            else:
                st.warning("Gauge R&R data (`msa_data`) is missing or incomplete.")
        except Exception as e:
            st.error(f"Could not perform Gauge R&R Analysis. Error: {e}"); logger.error(f"Error in Gauge R&R tool: {e}", exc_info=True)

    with tool_tabs[5]: # Chi-Squared
        st.subheader("Chi-Squared Test of Independence")
        st.markdown("Test for an association between two categorical variables (e.g., Sequencing Flow Cell vs. Run Pass/Fail Rate).")
        with st.expander("The 'Why', 'How', and 'So What?'"):
            st.markdown("#### Purpose: Finding Relationships in Categorical Lab Data")
            st.markdown("This test helps us find hidden relationships in our lab operations. For example, is one particular sequencer or flow cell associated with a higher run failure rate? This is a key tool for troubleshooting and root cause analysis in a CLIA environment.")
        try:
            chi_data = ssm.get_data("quality_system", "chi_squared_data")
            if not chi_data:
                st.info("Displaying example data.", icon="ℹ️")
                rng = np.random.default_rng(1)
                data = []
                for _ in range(100): data.append({'flow_cell': 'FC-100A', 'outcome': rng.choice(['Pass', 'Fail'], p=[0.95, 0.05])})
                for _ in range(100): data.append({'flow_cell': 'FC-200B', 'outcome': rng.choice(['Pass', 'Fail'], p=[0.85, 0.15])})
                chi_data = data
            if chi_data and all(k in chi_data[0] for k in ['flow_cell', 'outcome']):
                df = pd.DataFrame(chi_data)
                contingency_table = pd.crosstab(df['flow_cell'], df['outcome'])
                st.markdown("**Contingency Table (Observed Run Outcomes)**"); st.dataframe(contingency_table)
                if contingency_table.size > 1:
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Chi-Squared Statistic", f"{chi2:.3f}"); st.metric("P-value", f"{p:.4f}")
                        if p < 0.05:
                            st.success("**Conclusion:** A significant association exists between Flow Cell and Run Outcome (p < 0.05). Investigate!", icon="✅")
                        else:
                            st.warning("**Conclusion:** No significant association detected (p >= 0.05).", icon="⚠️")
                    with col2:
                        ct_percent = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
                        fig = px.imshow(ct_percent, text_auto='.1f', aspect="auto", title="Heatmap of Outcomes by Flow Cell (%)", labels=dict(x="Outcome", y="Flow Cell", color="% of Row Total"), color_continuous_scale=px.colors.sequential.Greens)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data to form a valid contingency table.")
            else:
                st.warning("Chi-Squared data (`chi_squared_data`) is missing or incomplete.")
        except Exception as e:
            st.error(f"Could not perform Chi-Squared Test. Error: {e}"); logger.error(f"Error in Chi-Squared tool: {e}", exc_info=True)

    with tool_tabs[6]: # Correlation
        st.subheader("Correlation Analysis")
        st.markdown("Explore the linear relationship between two continuous assay variables (e.g., DNA Quality vs. Library Complexity).")
        with st.expander("The 'Why', 'How', and 'So What?'"):
            st.markdown("#### Purpose: Quantifying Assay Relationships")
            st.markdown("Correlation analysis helps us understand how different parts of our assay affect each other. A strong correlation between a pre-analytical metric (like DNA quality) and a key performance metric (like library complexity) can be used to set sample acceptance criteria and predict run success.")
        try:
            corr_data_dict = ssm.get_data("quality_system", "correlation_data")
            if not corr_data_dict:
                st.info("Displaying example data.", icon="ℹ️")
                rng = np.random.default_rng(42)
                dv200 = np.linspace(50, 95, 50)
                complexity = 1e6 + 2e4 * dv200 + rng.normal(0, 1e5, 50)
                corr_data_dict = {'dv200_pct': list(dv200), 'library_complexity': list(complexity)}

            if corr_data_dict and all(k in corr_data_dict for k in ['dv200_pct', 'library_complexity']):
                df = pd.DataFrame(corr_data_dict)
                if len(df) > 2:
                    r, p = pearsonr(df['dv200_pct'], df['library_complexity'])
                    fig = px.scatter(df, x='dv200_pct', y='library_complexity', title=f"Correlation Analysis (r = {r:.3f})", trendline="ols", trendline_color_override="red", labels={'dv200_pct': 'DNA Quality (DV200 %)', 'library_complexity': 'Library Complexity'})
                    st.plotly_chart(fig, use_container_width=True)
                    st.metric("Pearson Correlation (r)", f"{r:.4f}"); st.metric("P-value", f"{p:.4f}")
                    if p < 0.05:
                        st.success(f"**Conclusion:** A statistically significant {'positive' if r > 0 else 'negative'} correlation exists.", icon="✅")
                    else:
                        st.warning("**Conclusion:** The observed correlation is not statistically significant.", icon="⚠️")
                else:
                    st.warning("Need at least 3 data points for correlation analysis.")
            else:
                st.warning("Correlation data (`correlation_data`) is missing or incomplete.")
        except Exception as e:
            st.error(f"Could not perform Correlation Analysis. Error: {e}"); logger.error(f"Error in Correlation tool: {e}", exc_info=True)

    with tool_tabs[7]: # Equivalence
        st.subheader("Equivalence Testing (TOST) for Change Control")
        st.markdown("Rigorously prove that two processes are 'practically the same,' essential for validating a new reagent lot or a minor process change.")
        with st.expander("The 'Why', 'How', and 'So What?'"):
            st.markdown("#### Purpose: Proving Sameness for Regulatory Changes")
            st.markdown("When we make a change, like qualifying a new lot of critical reagents, we don't want to prove it's *different*; we need to prove it's *the same* for all practical purposes. A standard t-test cannot do this. **Equivalence Testing (TOST)** is the correct statistical method and is expected by regulators for justifying such changes under a documented change control system (Ref: 21 CFR 820.30(i)).")
            st.markdown("#### Significance: A Defensible Change")
            st.markdown("- **`p < 0.05`:** Claim equivalence. You have strong evidence the change did not negatively impact assay performance beyond your pre-defined acceptable margin.\n- **`p >= 0.05`:** Cannot claim equivalence. The change might have introduced an unacceptable shift in performance. Do not implement the change without further investigation.")
        try:
            ht_data = ssm.get_data("quality_system", "equivalence_data")
            if not ht_data:
                rng = np.random.default_rng(10)
                ht_data = {'reagent_lot_a': list(rng.normal(0.85, 0.05, 30)), 'reagent_lot_b': list(rng.normal(0.86, 0.05, 30))}
            
            if ht_data and all(k in ht_data for k in ['reagent_lot_a', 'reagent_lot_b']):
                line_a, line_b = np.array(ht_data['reagent_lot_a']), np.array(ht_data['reagent_lot_b'])
                st.markdown("**1. Define Equivalence Margin (Based on Risk Assessment)**")
                delta = st.number_input("Enter equivalence margin for Bisulfite Conversion Rate (delta, δ):", min_value=0.0, value=0.05, step=0.01, help="Max difference in performance still considered 'equivalent'.")

                n1, n2 = len(line_a), len(line_b)
                mean_diff = np.mean(line_a) - np.mean(line_b)
                s1, s2 = np.var(line_a, ddof=1), np.var(line_b, ddof=1)
                
                if n1 + n2 > 2:
                    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
                    se_diff = pooled_sd * np.sqrt(1/n1 + 1/n2) if (n1 > 0 and n2 > 0) else 0

                    if se_diff > 0:
                        t_stat_lower = (mean_diff - (-delta)) / se_diff; t_stat_upper = (mean_diff - delta) / se_diff
                        dof = n1 + n2 - 2
                        p_lower = stats.t.sf(t_stat_lower, df=dof); p_upper = stats.t.cdf(t_stat_upper, df=dof)
                        tost_p_value = max(p_lower, p_upper)
                        ci_90 = stats.t.interval(0.90, df=dof, loc=mean_diff, scale=se_diff)

                        st.markdown("**2. Test Results (Reagent Lot A vs. Lot B)**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Difference in Means (A - B)", f"{mean_diff:.3f}")
                            st.metric("TOST P-Value", f"{tost_p_value:.4f}")
                            st.markdown(f"**90% CI of Difference:** `[{ci_90[0]:.3f}, {ci_90[1]:.3f}]`")
                        with col2:
                            st.markdown("**3. Conclusion**"); is_equivalent = tost_p_value < 0.05
                            if is_equivalent:
                                st.success(f"**Conclusion: The lots ARE statistically equivalent** within ±{delta}.", icon="✅")
                            else:
                                st.error(f"**Conclusion: Equivalence CANNOT be claimed** within ±{delta}.", icon="🚨")
                        fig = go.Figure()
                        fig.add_shape(type="rect", x0=-delta, y0=0, x1=delta, y1=1, line=dict(width=0), fillcolor="rgba(44, 160, 44, 0.2)", layer="below", name="Equivalence Zone")
                        fig.add_trace(go.Scatter(x=[ci_90[0], ci_90[1]], y=[0.5, 0.5], mode="lines", line=dict(color="blue", width=4), name="90% CI of Difference"))
                        fig.add_trace(go.Scatter(x=[mean_diff], y=[0.5], mode="markers", marker=dict(color="blue", size=12, symbol="x"), name="Observed Difference"))
                        fig.update_layout(title="Equivalence Test Visualization", xaxis_title="Difference in Bisulfite Conversion Rate", yaxis=dict(showticklabels=False, range=[0,1]), shapes=[dict(type='line', x0=0, y0=0, x1=0, y1=1, line=dict(color='black', dash='dash'))])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not perform test. Check data.")
                else:
                    st.warning("Not enough data points to perform the test.")
            else:
                st.warning("Equivalence testing data is incomplete or missing.")
        except Exception as e:
            st.error(f"Could not perform Equivalence Test. Error: {e}"); logger.error(f"Error in TOST tool: {e}", exc_info=True)


def render_machine_learning_lab_tab(ssm: SessionStateManager):
    """
    Renders the Machine Learning Lab, focusing on the validation and explainability
    of models used in the diagnostic service, as required by ISO 62304 and FDA AI/ML guidance.
    """
    st.header("🤖 Machine Learning & Bioinformatics Lab")
    st.info("Utilize and validate predictive models for operational efficiency and explore the classifier's behavior. Model explainability is key for regulatory review.")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        import shap
    except ImportError:
        st.error("This tab requires `scikit-learn` and `shap`. Please install them (`pip install scikit-learn shap`) to enable ML features.", icon="🚨")
        return

    ml_tabs = st.tabs([
        "Classifier Explainability (SHAP)", "Predictive Ops (Run Failure)", "Time Series Forecasting (Samples)"
    ])

    with ml_tabs[0]:
        st.subheader("Cancer Classifier Explainability (SHAP)")
        st.markdown("This tool uses SHAP to explain the output of the final cancer classification model. Understanding *why* the model makes a certain prediction is critical for validation, troubleshooting, and regulatory submission.")

        with st.expander("The Purpose, Math Basis, Procedure, and Significance for a PMA"):
            st.markdown("#### Purpose: Unlock the Black Box")
            st.markdown("For a high-risk diagnostic device, a 'black box' model is unacceptable to regulators. The purpose of this analysis is to provide deep, scientifically-grounded explanations for the model's predictions. This allows us to verify that the model is learning biologically relevant patterns and to debug cases where it makes an incorrect prediction.")
            st.markdown("#### The Mathematical Basis & Method: Random Forest and SHAP")
            st.markdown("- **Random Forest:** A powerful ensemble model used for classification. In our case, it takes hundreds of genomic features (e.g., methylation levels at specific CpG sites, fragmentation patterns) as input to predict 'Cancer Detected' or 'No Cancer Detected'.\n- **SHAP (SHapley Additive exPlanations):** SHAP assigns each feature an 'importance' value for a particular prediction. It tells us how much each genomic feature contributed to pushing the model's prediction from a baseline to its final output.")
            st.markdown("#### Significance for PMA Submission & Validation")
            st.markdown("- **Algorithm Validation:** Demonstrates to the FDA that we understand our model's logic and that it's not relying on spurious artifacts.\n- **Clinical Utility:** Helps biologists and clinicians understand the key genomic drivers behind a 'Cancer Detected' signal.\n- **Feature Importance Plot:** Globally, which genomic regions or features are most important for the classifier?\n- **SHAP Summary Plot:** Provides deep insights. It shows *not only* which features are important but *how* their values affect the outcome (e.g., 'High methylation at `cg01234567` strongly pushes the prediction towards 'Cancer Detected''). This is crucial for scientific validation.")

        @st.cache_data
        def get_classifier_model_and_data():
            np.random.seed(42); n_samples = 500
            # Mock genomic features
            data = {'cg01': np.random.normal(0.2, 0.1, n_samples), 'cg02': np.random.normal(0.8, 0.1, n_samples), 'frag_len': np.random.normal(150, 10, n_samples)}
            df = pd.DataFrame(data)
            # Cancer signal is high methylation in cg01, low in cg02, and short fragments
            cancer_conditions = (df['cg01'] > 0.3) & (df['cg02'] < 0.75) & (df['frag_len'] < 145)
            df['status'] = np.where(cancer_conditions, 'Cancer Detected', 'No Cancer Detected'); df['status_code'] = df['status'].apply(lambda x: 1 if x == 'Cancer Detected' else 0)
            X = df[['cg01', 'cg02', 'frag_len']]; y = df['status_code']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            model = RandomForestClassifier(n_estimators=100, random_state=42); model.fit(X_train, y_train)
            return model, X_test, y_test

        @st.cache_data
        def get_shap_explanation(_model, _X_test):
            explainer = shap.TreeExplainer(_model)
            shap_explanation = explainer(_X_test)
            return shap_explanation

        model, X_test, y_test = get_classifier_model_and_data()
        shap_explanation = get_shap_explanation(model, X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model Performance (Holdout Test Set)**")
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
            annotations = [[f"{labels[i][j]}<br>{cm[i][j]}<br>({cm_percent[i][j]:.2%})" for j in range(2)] for i in range(2)]
            fig_cm = go.Figure(data=go.Heatmap(
                   z=cm, x=['Predicted No Cancer', 'Predicted Cancer'], y=['Actual No Cancer', 'Actual Cancer'],
                   hoverongaps=False, colorscale='Blues', showscale=False,
                   text=annotations, texttemplate="%{text}"))
            fig_cm.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), title_x=0.5, title_text="<b>Confusion Matrix</b>", title_font_size=16)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            st.markdown("**Overall Feature Importance**")
            shap_values_cancer = shap_explanation[:, :, 1]
            mean_abs_shap = np.abs(shap_values_cancer.values).mean(axis=0)
            importance_df = pd.DataFrame({'feature': X_test.columns, 'importance': mean_abs_shap}).sort_values('importance')
            fig_bar = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Average Impact on Model Output', text_auto='.3f')
            fig_bar.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), title_font_size=16, xaxis_title="mean |SHAP value|")
            fig_bar.update_traces(marker_color='#1f77b4'); st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Deep Dive: How Genomic Features Drive Cancer Prediction")
        st.markdown("Each point is a sample. Red means a high feature value, blue a low one. Points on the right were pushed towards 'Cancer Detected'.")
        fig_summary = go.Figure()
        for i, feature in enumerate(X_test.columns):
            norm_vals = (shap_explanation.data[:, i] - shap_explanation.data[:, i].min()) / (shap_explanation.data[:, i].max() - shap_explanation.data[:, i].min())
            jitter = np.random.uniform(-0.15, 0.15, len(shap_values_cancer.values))
            fig_summary.add_trace(go.Scatter(
                x=shap_values_cancer.values[:, i], y=np.full(len(shap_values_cancer.values), i) + jitter, mode='markers',
                marker=dict(color=norm_vals, colorscale='RdBu', reversescale=True, showscale=True, colorbar=dict(title='Feature Value', x=1.15, tickvals=[0,1], ticktext=['Low', 'High'])),
                customdata=shap_explanation.data[:, i], hovertemplate=f"<b>{feature}</b><br>SHAP Value: %{{x:.3f}}<br>Feature Value: %{{customdata:.2f}}<extra></extra>", name=feature
            ))
        fig_summary.update_layout(title="<b>SHAP Summary Plot: Impact on 'Cancer Detected' Prediction</b>", xaxis_title="SHAP Value", showlegend=False,
            yaxis=dict(tickvals=list(range(len(X_test.columns))), ticktext=X_test.columns, title='Genomic Feature'), height=400)
        st.plotly_chart(fig_summary, use_container_width=True)

    with ml_tabs[1]:
        st.subheader("Predictive Operations: Sequencing Run Failure")
        st.markdown("This model predicts whether a sequencing run will **Pass** or **Fail** based on pre-run QC metrics, enabling proactive intervention.")
        with st.expander("The Purpose, Math Basis, Procedure, and Significance"):
            st.markdown("#### Purpose: Proactive Lab Operations")
            st.markdown("The goal is to reduce the Cost of Poor Quality (COPQ) from failed sequencing runs (wasted reagents, sequencer time, and labor). By predicting failure *before* committing a batch to the sequencer, we can flag high-risk plates for review or remediation, improving lab efficiency and turnaround time.")
            st.markdown("#### Significance: From 'What' to 'Why'")
            st.markdown("- **Risk Probability Forecast:** Identifies which assay plates are at highest risk of failure, allowing for targeted intervention.\n- **Risk Factor Contribution Plot (Drill-Down):** Shows the lab manager *why* a plate is high-risk (e.g., low DNA input, poor library quality), enabling effective troubleshooting.")
        
        @st.cache_data
        def train_and_predict_run_failure(tasks: Tuple):
            df = pd.DataFrame([dict(fs) for fs in tasks])
            train_df = df[df['status'].isin(['Pass', 'Fail'])].copy(); train_df['target'] = (train_df['status'] == 'Fail').astype(int)
            if len(train_df['target'].unique()) < 2: return None, None, None
            features = ['input_dna_ng', 'library_yield_ng', 'dv200_pct']; X_train = train_df[features]; y_train = train_df['target']
            model = LogisticRegression(random_state=42, class_weight='balanced'); model.fit(X_train, y_train)
            predict_df = df[df['status'] == 'Pending'].copy()
            if predict_df.empty: return None, None, None
            X_predict = predict_df[features]; predict_df['risk_probability'] = model.predict_proba(X_predict)[:, 1]
            return predict_df, model, features

        run_qc_data = ssm.get_data("lab_operations", "run_qc_history")
        immutable_runs = tuple(frozenset(d.items()) for d in run_qc_data)
        risk_predictions_df, risk_model, risk_features = train_and_predict_run_failure(immutable_runs)

        if risk_predictions_df is not None:
            st.markdown("**Forecasted Failure Probability for Pending Assay Plates**")
            sorted_risk_df = risk_predictions_df.sort_values('risk_probability', ascending=False)
            fig = px.bar(sorted_risk_df, x='risk_probability', y='plate_id', orientation='h', title="Forecasted Failure Risk for Pending Plates", labels={'risk_probability': 'Probability of Failure', 'plate_id': 'Plate ID'}, color='risk_probability', color_continuous_scale=px.colors.sequential.Reds, text='risk_probability')
            fig.update_traces(texttemplate='%{text:.0%}', textposition='outside'); fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'}); st.plotly_chart(fig, use_container_width=True)
            
            st.divider(); st.subheader("Drill-Down: Analyze a Specific Plate's Risk Factors")
            high_risk_plates = sorted_risk_df[sorted_risk_df['risk_probability'] > 0.5]['plate_id'].tolist()
            if not high_risk_plates:
                st.info("No plates are currently predicted to be at high risk (>50% probability).")
            else:
                selected_plate = st.selectbox("Select a high-risk plate to analyze:", options=high_risk_plates)
                task_data = risk_predictions_df[risk_predictions_df['plate_id'] == selected_plate].iloc[0]
                contributions = task_data[risk_features].values * risk_model.coef_[0]
                contribution_df = pd.DataFrame({'feature': risk_features, 'contribution': contributions}).sort_values('contribution', ascending=True)
                fig_contrib = px.bar(contribution_df, x='contribution', y='feature', orientation='h', title=f'Risk Factor Contributions for "{selected_plate}"', labels={'contribution': 'Impact on Risk (Log-Odds)', 'feature': 'QC Metric'}, color='contribution', color_continuous_scale=px.colors.sequential.RdBu_r, text_auto='.2f')
                fig_contrib.update_layout(showlegend=False, coloraxis_showscale=False); st.plotly_chart(fig_contrib, use_container_width=True)
        else:
            st.info("Not enough historical run data (Pass/Fail) to train a predictive model yet.")
    
    with ml_tabs[2]: # Time Series
        st.subheader("Time Series Forecasting for Lab Operations")
        st.markdown("Predict future sample receipt volume to aid in capacity planning, staffing, and reagent ordering.")
        try:
            from statsmodels.tsa.arima.model import ARIMA
            @st.cache_data
            def generate_ts_data():
                t = np.arange(100); trend = 2 * t; seasonality = 50 * np.sin(2 * np.pi * t / 12)
                noise = np.random.normal(0, 20, 100); series = 200 + trend + seasonality + noise
                dates = pd.date_range(start='2021-01-01', periods=100, freq='W')
                return pd.Series(series, index=dates)
            ts_data = generate_ts_data()
            st.markdown("**1. Historical Data (Weekly Sample Receipts)**"); st.line_chart(ts_data)
            st.markdown("**2. Fit Model and Generate Forecast**"); n_forecast = st.slider("Select number of weeks to forecast:", min_value=4, max_value=52, value=26)
            model = ARIMA(ts_data, order=(5,1,0)).fit() # Simplified ARIMA
            forecast = model.get_forecast(steps=n_forecast); forecast_mean = forecast.predicted_mean; forecast_ci = forecast.conf_int()
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Historical Data'))
            fig_ts.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Forecast', line=dict(dash='dash', color='red')))
            fig_ts.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0], mode='lines', name='Lower CI', line=dict(width=0), showlegend=False))
            fig_ts.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1], mode='lines', name='95% Confidence Interval', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)'))
            fig_ts.update_layout(title="Time Series Forecast for Sample Receipt Volume", yaxis_title="Number of Samples / Week"); st.plotly_chart(fig_ts, use_container_width=True)
        except ImportError:
            st.error("This tool requires `statsmodels`.")
        except Exception as e:
            st.error(f"Could not generate time series forecast: {e}")

def render_compliance_guide_tab():
    """Renders the educational content for the Compliance Guide tab, focused on IVDs."""
    st.header("🏛️ A Guide to the IVD & Genomics Regulatory Landscape")
    st.markdown("This section provides a high-level overview of the key regulations and standards governing the development of a PMA-class genomic diagnostic.")
    st.subheader("Navigating the Regulatory Maze for a Genomic IVD Service")
    st.info("Our MCED test is a service-based In Vitro Diagnostic (IVD), which means we are regulated both as a device manufacturer and as a clinical laboratory.")
    with st.expander("▶️ **21 CFR 820: The Quality System Regulation (QSR)**"):
        st.markdown("This is the FDA's rulebook for medical device design and manufacturing. It applies to all components we develop and control.\n- **Applies to:** The sample collection kit, reagents we manufacture, and all our software (bioinformatics pipeline, LIMS, classifier).\n- **Key Principle:** The Design Controls section (`820.30`) is the foundation of this entire application. It mandates a systematic, traceable process from user needs to a validated device. The DHF is the ultimate proof of this process.")
    with st.expander("▶️ **ISO 13485:2016: Quality Management Systems (Global Standard)**"):
        st.markdown("ISO 13485 is the international standard for a medical device QMS. Compliance is essential for market access outside the US (e.g., Europe, Canada).\n- **Relationship to QSR:** Very similar, but with a stronger emphasis on **risk management** integrated throughout the entire QMS. Our QMS is harmonized to meet both QSR and ISO 13485 requirements.")
    with st.expander("▶️ **CLIA (42 CFR 493) & ISO 15189: Clinical Laboratory Standards**"):
        st.markdown("Since we operate a laboratory that provides patient-specific results, we must also comply with standards for clinical testing.\n- **CLIA (Clinical Laboratory Improvement Amendments):** US federal regulations governing all laboratory testing on humans. We must have a CLIA certificate for high-complexity testing.\n- **ISO 15189:** The international standard for the quality and competence of medical laboratories. It covers everything from sample receipt and personnel competency to quality assurance and report generation.\n- **Our System:** The 'Assay & Lab Operations' dashboard tracks metrics critical for both CLIA and ISO 15189 compliance.")
    with st.expander("▶️ **ISO 14971:2019: Risk Management for Medical Devices**"):
        st.markdown("This is the global standard for risk management. For our IVD, the primary risk is not physical harm, but the harm caused by an **incorrect test result** (a false positive or false negative). Our risk management file documents how we identify, evaluate, and control these risks throughout the assay and software.")
    with st.expander("▶️ **ISO 62304 & FDA Software Guidance: Medical Device Software**"):
        st.markdown("Our bioinformatics pipeline, classifier model, and LIMS are all considered Medical Device Software. They must be developed under a rigorous, documented lifecycle.\n- **Key Principle:** We must prove with objective evidence that the software reliably meets its specified requirements. This involves requirements analysis, architectural design, coding standards, verification (unit/integration tests), and validation (testing the final, locked software). The 'Software & Service FMEA' is a key artifact.")
    
    st.divider(); st.subheader("Visualizing the Process: The V-Model for a Genomic Assay")
    st.markdown("The V-Model visualizes the Design Controls process, linking the design definition (left side) to the testing and validation activities (right side).")
    try:
        # Assuming a v-model image exists, relabeled for genomics
        v_model_image_path = os.path.join(project_root, "dhf_dashboard", "v_model_diagram_genomics.png") # A new, specific diagram
        if os.path.exists(v_model_image_path):
            _, img_col, _ = st.columns([1, 2, 1])
            img_col.image(v_model_image_path, caption="The V-Model for a genomic diagnostic service, linking requirements to V&V.", width=600)
        else:
            st.info("V-Model Diagram Placeholder: The diagram below illustrates the concept.")
    except Exception as e:
        st.error("An error occurred while trying to display the V-Model image."); logger.error(f"Error loading V-Model image: {e}", exc_info=True)
    
    col1, col2 = st.columns(2)
    with col1: 
        st.subheader("Left Side: Decomposition (What we will build)")
        st.markdown("- **Clinical Needs & Intended Use**\n- **Design Inputs** (System & Assay Requirements)\n- **Architectural Design** (Wet Lab & Dry Lab)\n- **Detailed Design** (Oligo sequences, Software functions)")
    with col2: 
        st.subheader("Right Side: Integration & Testing (Proof it works)")
        st.markdown("- **Unit Verification** (Reagent QC, Code tests)\n- **Integration Verification** (Pipeline runs sample->result)\n- **System Verification (Analytical Validation)**\n- **Clinical Validation**")
    
    st.success("""#### The Core Principle: Verification vs. Validation
- **Verification:** *Did we build the assay right?* (Does the assay meet its performance specifications? E.g., LoD, precision).
- **Validation:** *Did we build the right assay?* (Does the assay meet the clinical needs in the target population? E.g., sensitivity/specificity for Stage II lung cancer).""")

# ==============================================================================
# --- MAIN APPLICATION LOGIC ---
# ==============================================================================
def main() -> None:
    """Main function to configure and run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="GenomicsDx Command Center", page_icon="🧬")
    
    try:
        ssm = SessionStateManager()
        logger.info("Application initialized. Session State Manager loaded.")
    except Exception as e:
        st.error("Fatal Error: Could not initialize Session State. The application cannot continue.")
        logger.critical(f"Failed to instantiate SessionStateManager: {e}", exc_info=True)
        st.stop()

    try:
        tasks_raw = ssm.get_data("project_management", "tasks")
        tasks_df_processed = preprocess_task_data(tasks_raw)
        
        docs_df = get_cached_df(ssm.get_data("design_outputs", "documents"))
        if 'phase' in docs_df.columns:
            docs_by_phase = {phase: data for phase, data in docs_df.groupby('phase')}
        else:
            docs_by_phase = {}

    except Exception as e:
        st.error("Failed to process initial project data for dashboard.")
        logger.error(f"Error during initial data pre-processing: {e}", exc_info=True)
        tasks_df_processed = pd.DataFrame()
        docs_by_phase = {}

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
