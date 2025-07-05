# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Renders the Design Change Control section of the DHF dashboard.

This module provides a structured UI for documenting and managing formal
design change control workflows, as required by 21 CFR 820.30(i). It ensures
a comprehensive impact assessment and traceable linkage to validation activities.
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List, Optional

# --- Third-party Imports ---
import pandas as pd
import streamlit as st

# --- Local Application Imports ---
from ..utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def render_design_changes(ssm: SessionStateManager) -> None:
    """
    Renders the UI for the Design Change Control section.

    This function displays a summary table of all Design Change Requests (DCRs)
    and provides a detailed, form-based interface for creating and editing
    individual records. It enforces a compliant workflow, including structured
    impact analysis and planning for required V&V activities.

    Args:
        ssm (SessionStateManager): The session state manager to access DHF data.
    """
    st.header("10. Design Change Control")
    st.markdown("""
    *As per 21 CFR 820.30(i) and ISO 13485:2016, Section 7.3.7.*

    This section documents the formal control of all design changes made after initial design approval.
    Each change must be identified, documented, reviewed, and approved before implementation.
    The process must include a re-assessment of risk and a determination of the effects of the change
    on the constituent parts of the diagnostic service (assay, software, kit, etc.).
    """)
    st.info("Select a change from the table to view/edit its details, or log a new change request.", icon="ℹ️")

    try:
        # --- 1. Load Data & Initialize State ---
        changes_data: List[Dict[str, Any]] = ssm.get_data("design_changes", "changes")
        if "selected_dcr_id" not in st.session_state:
            st.session_state.selected_dcr_id = None
        
        # --- 2. Display Summary Table of DCRs ---
        st.subheader("Design Change Request (DCR) Log")
        
        if not changes_data:
            st.warning("No design change records have been logged yet.")
        else:
            changes_df = pd.DataFrame(changes_data)
            changes_df['approval_date'] = pd.to_datetime(changes_df['approval_date'], errors='coerce').dt.date
            
            st.dataframe(
                changes_df[['id', 'description', 'approval_status', 'approval_date']],
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="dcr_selection_table"
            )

            # Capture selection from the dataframe
            if st.session_state.dcr_selection_table['selection']['rows']:
                selected_index = st.session_state.dcr_selection_table['selection']['rows'][0]
                st.session_state.selected_dcr_id = changes_df.iloc[selected_index]['id']
            
        # --- 3. DCR Creation/Editing Form ---
        st.divider()

        # Determine if we are editing an existing DCR or creating a new one
        if st.session_state.selected_dcr_id:
            st.subheader(f"Editing DCR: {st.session_state.selected_dcr_id}")
            # Find the full dictionary for the selected DCR
            dcr_to_edit_list = [dcr for dcr in changes_data if dcr.get('id') == st.session_state.selected_dcr_id]
            if dcr_to_edit_list:
                dcr_to_edit = dcr_to_edit_list[0]
            else: # selection is stale
                st.session_state.selected_dcr_id = None
                dcr_to_edit = {}
        else:
            st.subheader("Log a New Design Change Request")
            dcr_to_edit = {} # Start with an empty dict for a new DCR

        if st.button("Log New DCR", use_container_width=True):
            st.session_state.selected_dcr_id = None
            st.rerun()

        with st.form(key="dcr_form"):
            # --- DCR Details ---
            dcr_id = st.text_input("**Change Request ID**", value=dcr_to_edit.get("id", ""), disabled=bool(st.session_state.selected_dcr_id))
            description = st.text_area("**Change Description**", value=dcr_to_edit.get("description", ""), height=100)
            reason = st.text_area("**Reason for Change**", value=dcr_to_edit.get("reason", ""), height=100, help="Justification, e.g., 'Corrective action from CAPA-012', 'Improved LoD from R&D experiment', 'Reagent supplier change'.")

            # --- Structured Impact Analysis (Critical for Compliance) ---
            st.markdown("**Impact Analysis (Required)**")
            impact_cols = st.columns(2)
            impact_clinical = impact_cols[0].checkbox("Clinical/Regulatory Impact", value=dcr_to_edit.get("impact_clinical", False), help="Does this change affect the Intended Use, Indications for Use, or require a new regulatory submission/supplement?")
            impact_analytical = impact_cols[1].checkbox("Analytical Performance Impact", value=dcr_to_edit.get("impact_analytical", False), help="Does this change affect Sensitivity, Specificity, LoD, Precision, etc.?")
            impact_software = impact_cols[0].checkbox("Software/Bioinformatics Impact", value=dcr_to_edit.get("impact_software", False), help="Does this change affect the classifier algorithm, pipeline, or data integrity? (Ref: ISO 62304)")
            impact_lab_ops = impact_cols[1].checkbox("Lab Operations Impact", value=dcr_to_edit.get("impact_lab_ops", False), help="Does this change affect SOPs, LIMS, reagents, or personnel training? (Ref: CLIA/ISO 15189)")
            impact_risk = impact_cols[0].checkbox("Risk Management Impact", value=dcr_to_edit.get("impact_risk", False), help="Does this change introduce new hazards or affect existing risk controls? (Ref: ISO 14971)")
            impact_analysis_details = st.text_area("Impact Analysis Details", value=dcr_to_edit.get("impact_analysis_details", ""), height=150, help="Provide a detailed summary of all checked impacts.")

            # --- V&V Planning ---
            st.markdown("**Required Verification & Validation Activities**")
            vv_plan = st.text_area("V&V Plan", value=dcr_to_edit.get("vv_plan", ""), height=150, help="List the specific tests and validation studies required to approve this change. E.g., 'Execute VER-105: Regression test on pipeline', 'Execute VAL-021: Bridging study for new reagent'.")

            # --- Approval Section ---
            st.markdown("**Approval Status**")
            approval_cols = st.columns(2)
            status_options = ["Pending", "Approved", "Rejected", "Implementation Pending", "Closed"]
            current_status = dcr_to_edit.get("approval_status", "Pending")
            approval_status = approval_cols[0].selectbox("Status", options=status_options, index=status_options.index(current_status))
            
            approval_date_val = pd.to_datetime(dcr_to_edit.get("approval_date"), errors='coerce')
            approval_date = approval_cols[1].date_input("Approval Date", value=approval_date_val)
            
            if st.form_submit_button("Save Design Change Record", use_container_width=True):
                # --- 4. Validate and Persist Data ---
                if not dcr_id or not description or not reason:
                    st.error("DCR ID, Description, and Reason are required fields.")
                else:
                    new_dcr_data = {
                        "id": dcr_id,
                        "description": description,
                        "reason": reason,
                        "impact_clinical": impact_clinical,
                        "impact_analytical": impact_analytical,
                        "impact_software": impact_software,
                        "impact_lab_ops": impact_lab_ops,
                        "impact_risk": impact_risk,
                        "impact_analysis_details": impact_analysis_details,
                        "vv_plan": vv_plan,
                        "approval_status": approval_status,
                        "approval_date": str(approval_date) if approval_date else None,
                        "action_items": dcr_to_edit.get("action_items", []) # Preserve existing action items
                    }
                    
                    # Find and update existing DCR or append new one
                    found = False
                    for i, dcr in enumerate(changes_data):
                        if dcr.get('id') == new_dcr_data['id']:
                            changes_data[i] = new_dcr_data
                            found = True
                            break
                    if not found:
                        changes_data.append(new_dcr_data)

                    ssm.update_data(changes_data, "design_changes", "changes")
                    logger.info(f"Design change record '{dcr_id}' saved/updated.")
                    st.toast(f"DCR '{dcr_id}' saved successfully!", icon="✅")
                    st.session_state.selected_dcr_id = None # Clear selection
                    st.rerun()

    except Exception as e:
        st.error("An error occurred while displaying the Design Changes section. The data may be malformed.")
        logger.error(f"Failed to render design changes: {e}", exc_info=True)
