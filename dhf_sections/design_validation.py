# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Renders the Design Validation section of the DHF dashboard.

This module provides a comprehensive UI for documenting design validation,
which confirms that the diagnostic service meets user needs and intended uses.
It is structured to manage evidence from pivotal clinical trials and human
factors (usability) studies, as required by 21 CFR 820.30(g) and FDA guidance.
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List

# --- Third-party Imports ---
import pandas as pd
import streamlit as st

# --- Local Application Imports ---
from ..utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def render_design_validation(ssm: SessionStateManager) -> None:
    """
    Renders the UI for the Design Validation section, focusing on clinical
    and usability validation evidence.

    Args:
        ssm (SessionStateManager): The session state manager to access DHF data.
    """
    st.header("8. Design Validation")
    st.markdown("""
    *As per 21 CFR 820.30(g), ISO 13485:2016 Section 7.3.6, and IEC 62366.*

    Validation ensures the final, complete diagnostic service meets the defined **user needs** and **intended uses**. This is accomplished through objective evidence from clinical trials and usability studies conducted under actual or simulated use conditions. It answers the ultimate question: **"Did we build the right product for our users and patients?"**
    """)
    st.info("Use the tabs to manage evidence from the pivotal clinical study and all human factors validation activities.", icon="✅")
    st.divider()

    try:
        # --- 1. Load Data and Prepare Dependencies ---
        validation_data: Dict[str, Any] = ssm.get_data("clinical_study")
        inputs_data: List[Dict[str, Any]] = ssm.get_data("design_inputs", "requirements")
        logger.info("Loaded design validation and clinical study data.")

        user_needs = [req for req in inputs_data if req.get('type') == 'User Need']
        user_need_options: List[str] = [""] + sorted([f"{un.get('id', '')}: {un.get('description', '')}" for un in user_needs])
        
        # --- 2. Define Tabs for Each Validation Stream ---
        tab_titles = [
            "1. Pivotal Clinical Trial Dashboard",
            "2. Human Factors (Usability) Validation"
        ]
        tab1, tab2 = st.tabs(tab_titles)

        # --- Tab 1: Pivotal Clinical Trial ---
        with tab1:
            st.subheader("Pivotal Clinical Trial Management")
            st.caption("Track the status, enrollment, and performance of the main clinical study providing safety and effectiveness data for the PMA submission.")

            # --- Study Metadata ---
            st.markdown("##### **Study Overview**")
            cols = st.columns(3)
            cols[0].text_input("**NCT ID**", value=validation_data.get("nct_id", ""), key="val_nct_id")
            cols[1].text_input("**Study Phase**", value=validation_data.get("phase", "Pivotal"), key="val_phase")
            cols[2].text_input("**Primary Completion Date (Planned)**", value=validation_data.get("planned_completion_date", ""), key="val_plan_date")
            
            st.text_area("**Primary Endpoints**", value=validation_data.get("primary_endpoints", ""), key="val_endpoints", height=100, help="E.g., Co-primary endpoints are sensitivity and specificity for detection of cancer signal.")
            
            # --- Enrollment Tracking ---
            st.markdown("##### **Enrollment Progress**")
            enrollment_df = pd.DataFrame(validation_data.get("enrollment", []))
            if not enrollment_df.empty:
                total_enrolled = enrollment_df['enrolled'].sum()
                total_target = enrollment_df['target'].sum()
                overall_progress = (total_enrolled / total_target) * 100 if total_target > 0 else 0
                
                st.metric("Overall Enrollment", f"{total_enrolled:,} / {total_target:,}", f"{overall_progress:.1f}% Complete")
                st.progress(overall_progress / 100)
                
                st.dataframe(enrollment_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No enrollment data defined.")

            # --- Interim Performance Metrics ---
            st.markdown("##### **Interim Clinical Performance Metrics (Based on Locked Algorithm)**")
            perf_df = pd.DataFrame(validation_data.get("performance_metrics", []))
            if not perf_df.empty:
                st.dataframe(perf_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No performance metrics available yet.")

        # --- Tab 2: Human Factors Validation ---
        with tab2:
            st.subheader("Human Factors & Usability Validation Studies (IEC 62366)")
            st.caption("Document summative usability studies that validate the user interface is safe, effective, and that use-related risks have been controlled.")

            hf_studies_data = validation_data.get("hf_studies", [])
            hf_studies_df = pd.DataFrame(hf_studies_data)
            
            edited_hf_df = st.data_editor(
                hf_studies_df,
                num_rows="dynamic",
                use_container_width=True,
                key="hf_validation_editor",
                column_config={
                    "id": st.column_config.TextColumn("Study ID", help="Unique ID (e.g., HF-VAL-001).", required=True),
                    "study_name": st.column_config.TextColumn("Study/Protocol Name", width="large", required=True),
                    "user_interface_validated": st.column_config.SelectboxColumn("User Interface Validated", options=["Sample Collection Kit & IFU", "Clinical Report", "LIMS/Portal"], required=True),
                    "user_need_validated": st.column_config.SelectboxColumn("Links to User Need", options=user_need_options, required=True),
                    "confirms_risk_control": st.column_config.CheckboxColumn("Confirms Risk Control?", default=False),
                    "status": st.column_config.SelectboxColumn("Status", options=["Not Started", "In Progress", "Completed"], required=True),
                    "report_link": st.column_config.LinkColumn("Link to Final Report"),
                },
                hide_index=True
            )
            
            if edited_hf_df.to_dict('records') != hf_studies_data:
                validation_data["hf_studies"] = edited_hf_df.to_dict('records')
                ssm.update_data(validation_data, "clinical_study")
                st.toast("Human Factors validation studies updated!", icon="✅")
                st.rerun()

        # --- Persist any changes made in the clinical trial tab ---
        # This approach is simpler since the widgets are not in a form
        current_data = ssm.get_data("clinical_study")
        
        # Check and update each field
        changed = False
        for key in ["nct_id", "phase", "planned_completion_date", "primary_endpoints"]:
            if st.session_state[f"val_{key}"] != current_data.get(key, ""):
                current_data[key] = st.session_state[f"val_{key}"]
                changed = True
        
        if changed:
            ssm.update_data(current_data, "clinical_study")
            logger.info("Clinical study metadata updated.")
            st.toast("Clinical study details saved!", icon="✅")
            # No rerun needed for text inputs

    except Exception as e:
        st.error("An error occurred while displaying the Design Validation section. The data may be malformed.")
        logger.error(f"Failed to render design validation: {e}", exc_info=True)
