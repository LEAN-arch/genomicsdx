# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Renders the Risk Management File (RMF) section of the DHF dashboard.

This module provides a comprehensive, multi-tabbed UI for documenting the
entire risk management process according to ISO 14971, including planning,
hazard analysis, FMEAs, and overall risk-benefit analysis, all tailored for
a genomic diagnostic service.
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List

# --- Third-party Imports ---
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- Local Application Imports ---
from ..utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def render_design_risk_management(ssm: SessionStateManager) -> None:
    """
    Renders the UI for the complete Risk Management File (RMF).
    """
    st.header("2. Risk Management File (ISO 14971:2019)")
    st.markdown("""
    This section constitutes the Risk Management File (RMF) for the diagnostic service. It documents the systematic application of management policies, procedures, and practices to the tasks of analyzing, evaluating, controlling, and monitoring risk associated with the device. The primary harms for this IVD are related to incorrect test results.
    """)
    st.info("Use the tabs to navigate through the risk management process, from planning to final analysis. Changes are saved automatically.", icon="ℹ️")

    try:
        # --- 1. Load All Relevant Data ---
        rmf_data: Dict[str, Any] = ssm.get_data("risk_management_file")
        inputs_data: List[Dict[str, Any]] = ssm.get_data("design_inputs", "requirements")
        v_and_v_data: List[Dict[str, Any]] = ssm.get_data("design_verification", "tests") + ssm.get_data("clinical_study", "hf_studies")
        
        # --- Prepare Dependency Lists for Dropdowns ---
        risk_control_reqs = [req for req in inputs_data if req.get('is_risk_control')]
        risk_control_ids: List[str] = [""] + sorted([req.get('id', '') for req in risk_control_reqs if req.get('id')])
        
        vv_protocol_ids: List[str] = [""] + sorted([p.get('id', '') for p in v_and_v_data if p.get('id')])

        # --- 2. Define Tabs for Each RMF Section ---
        tab_titles = [
            "1. Risk Plan & Acceptability",
            "2. Hazard Analysis",
            "3. Assay FMEA (aFMEA)",
            "4. Software & Service FMEA (s/p-FMEA)",
            "5. Overall Residual Risk Analysis"
        ]
        tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

        # --- Tab 1: Risk Plan & Acceptability Matrix ---
        with tab1:
            st.subheader("Risk Management Plan Summary & Acceptability Criteria")
            
            # Use st.form to batch updates for this tab
            with st.form("risk_plan_form"):
                plan_scope_val = st.text_area(
                    "**Risk Management Plan Scope & Objectives**",
                    value=rmf_data.get("plan_scope", ""),
                    height=150,
                    help="Define the scope of risk management activities for the entire product lifecycle."
                )
                
                st.markdown("**Risk Acceptability Matrix**")
                st.caption("This matrix defines the policy for risk acceptability. It is the basis for evaluating all identified risks.")
                
                # Displaying the risk matrix visually
                s_labels = ['5: Catastrophic', '4: Critical', '3: Serious', '2: Minor', '1: Negligible']
                o_labels = ['1: Improbable', '2: Remote', '3: Occasional', '4: Probable', '5: Frequent']
                
                matrix_data = [
                    ['Acceptable', 'Acceptable', 'Acceptable', 'Review', 'Unacceptable'],
                    ['Acceptable', 'Acceptable', 'Review', 'Unacceptable', 'Unacceptable'],
                    ['Acceptable', 'Review', 'Unacceptable', 'Unacceptable', 'Unacceptable'],
                    ['Review', 'Unacceptable', 'Unacceptable', 'Unacceptable', 'Unacceptable'],
                    ['Unacceptable', 'Unacceptable', 'Unacceptable', 'Unacceptable', 'Unacceptable']
                ]
                
                color_map = {'Acceptable': 'rgba(44, 160, 44, 0.6)', 'Review': 'rgba(255, 215, 0, 0.6)', 'Unacceptable': 'rgba(214, 39, 40, 0.6)'}
                z_color = [[color_map[cell] for cell in row] for row in matrix_data]

                fig = go.Figure(data=go.Heatmap(
                    z=[[1,1,2,3,3], [1,2,2,3,3], [2,2,3,3,3], [2,3,3,3,3], [3,3,3,3,3]],
                    x=o_labels, y=s_labels,
                    text=matrix_data,
                    texttemplate="%{text}",
                    textfont={"size":12},
                    colorscale=[[0, 'rgba(44, 160, 44, 0.6)'], [0.5, 'rgba(255, 215, 0, 0.6)'], [1, 'rgba(214, 39, 40, 0.6)']],
                    showscale=False
                ))
                fig.update_layout(title="Severity vs. Occurrence", xaxis_title="Occurrence", yaxis_title="Severity of Harm")
                st.plotly_chart(fig, use_container_width=True)

                submitted = st.form_submit_button("Save Plan Scope")
                if submitted:
                    rmf_data["plan_scope"] = plan_scope_val
                    ssm.update_data(rmf_data, "risk_management_file")
                    st.toast("Risk Management Plan scope saved!", icon="✅")


        # --- Helper Function for Risk Tables ---
        def render_risk_table(table_title: str, table_key: str, df: pd.DataFrame, column_config: dict):
            st.subheader(table_title)
            # Make a copy to compare against for changes
            original_data = df.to_dict('records')
            
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                key=f"risk_editor_{table_key}",
                column_config=column_config,
                hide_index=True
            )
            
            # Persist changes
            if edited_df.to_dict('records') != original_data:
                rmf_data[table_key] = edited_df.to_dict('records')
                ssm.update_data(rmf_data, "risk_management_file")
                st.toast(f"{table_title} data updated!", icon="✅")
                st.rerun()

        # --- Tab 2: Hazard Analysis ---
        with tab2:
            hazards_df = pd.DataFrame(rmf_data.get("hazards", []))
            render_risk_table(
                "Hazard Analysis and Risk Evaluation (Top-Down)",
                "hazards",
                hazards_df,
                {
                    "id": st.column_config.TextColumn("Hazard ID", required=True),
                    "description": st.column_config.TextColumn("Hazard Description", width="large", required=True),
                    "foreseeable_event": st.column_config.TextColumn("Foreseeable Sequence of Events", width="large", help="The chain of events leading from hazard to harm."),
                    "potential_harm": st.column_config.TextColumn("Potential Patient Harm", width="large", required=True),
                    "initial_S": st.column_config.NumberColumn("Initial S", min_value=1, max_value=5, required=True),
                    "initial_O": st.column_config.NumberColumn("Initial O", min_value=1, max_value=5, required=True),
                    "risk_control_measure": st.column_config.TextColumn("Risk Control Measure(s)"),
                    "final_S": st.column_config.NumberColumn("Final S", min_value=1, max_value=5),
                    "final_O": st.column_config.NumberColumn("Final O", min_value=1, max_value=5),
                    "verification_link": st.column_config.SelectboxColumn("V&V Link", options=vv_protocol_ids, help="Link to V&V protocol proving control is effective.")
                }
            )

        # --- Tab 3: Assay FMEA ---
        with tab3:
            afmea_df = pd.DataFrame(rmf_data.get("assay_fmea", []))
            render_risk_table(
                "Assay Failure Mode and Effects Analysis (aFMEA)",
                "assay_fmea",
                afmea_df,
                {
                    "id": st.column_config.TextColumn("aFMEA ID", required=True),
                    "process_step": st.column_config.TextColumn("Assay Process Step", required=True, width="medium"),
                    "failure_mode": st.column_config.TextColumn("Potential Failure Mode", width="large", required=True),
                    "potential_effect": st.column_config.TextColumn("Effect of Failure", width="large"),
                    "S": st.column_config.NumberColumn("S", min_value=1, max_value=5),
                    "O": st.column_config.NumberColumn("O", min_value=1, max_value=5),
                    "D": st.column_config.NumberColumn("D", min_value=1, max_value=5),
                    "mitigation": st.column_config.TextColumn("Mitigation / Control"),
                    "verification_link": st.column_config.SelectboxColumn("V&V Link", options=vv_protocol_ids)
                }
            )
            
        # --- Tab 4: Software FMEA ---
        with tab4:
            sfmea_df = pd.DataFrame(rmf_data.get("service_fmea", []))
            render_risk_table(
                "Software & Service Failure Mode and Effects Analysis (s/p-FMEA)",
                "service_fmea",
                sfmea_df,
                {
                    "id": st.column_config.TextColumn("sFMEA ID", required=True),
                    "process_step": st.column_config.TextColumn("Software/Service Component", required=True, width="medium"),
                    "failure_mode": st.column_config.TextColumn("Potential Failure Mode", width="large", required=True),
                    "potential_effect": st.column_config.TextColumn("Effect of Failure", width="large"),
                    "S": st.column_config.NumberColumn("S", min_value=1, max_value=5),
                    "O": st.column_config.NumberColumn("O", min_value=1, max_value=5),
                    "D": st.column_config.NumberColumn("D", min_value=1, max_value=5),
                    "mitigation": st.column_config.TextColumn("Mitigation / Control"),
                    "verification_link": st.column_config.SelectboxColumn("V&V Link", options=vv_protocol_ids)
                }
            )

        # --- Tab 5: Overall Residual Risk ---
        with tab5:
            st.subheader("Overall Residual Risk-Benefit Analysis & Conclusion")
            st.markdown("This is the final conclusion of the risk management process, required by ISO 14971. It should be a formal statement declaring whether the overall residual risk (the sum of all individual residual risks) is acceptable in relation to the documented clinical benefits of the device.")
            
            with st.form("risk_benefit_form"):
                analysis_val = st.text_area(
                    "**Risk-Benefit Analysis Statement:**",
                    value=rmf_data.get("overall_risk_benefit_analysis", ""),
                    height=200,
                    help="Example: 'The overall residual risk of the GenomicsDx Sentry™ Test, considering all identified hazards and the effectiveness of the implemented risk controls, is judged to be acceptable in relation to the substantial clinical benefit of early cancer detection...'"
                )
                submitted = st.form_submit_button("Save Risk-Benefit Statement")
                if submitted:
                    rmf_data["overall_risk_benefit_analysis"] = analysis_val
                    ssm.update_data(rmf_data, "risk_management_file")
                    st.toast("Risk-Benefit statement saved!", icon="✅")

    except Exception as e:
        st.error("An error occurred while displaying the Risk Management section. The data may be malformed.")
        logger.error(f"Failed to render design risk management: {e}", exc_info=True)
