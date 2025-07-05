# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Renders the Human Factors & Usability Engineering section of the DHF dashboard.

This module provides a structured UI for documenting the complete Usability
Engineering process in alignment with IEC 62366 and FDA guidance. It includes
defining user profiles, conducting a Use-Related Risk Analysis (URRA) for each
user interface, and linking risks to controls and validation activities.
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


def render_human_factors(ssm: SessionStateManager) -> None:
    """
    Renders the UI for the Human Factors & Usability Engineering section.
    """
    st.header("3. Human Factors & Usability Engineering (IEC 62366)")
    st.markdown("""
    This section documents the usability engineering process to ensure the diagnostic service can be used safely and effectively by its intended users in the intended use environments. The process focuses on identifying and mitigating use-related risks that could lead to patient harm (e.g., from a sample collection error or a misinterpreted result).
    """)
    st.info("Use the tabs to define user profiles and conduct a Use-Related Risk Analysis (URRA) for each user interface. Changes are saved automatically.", icon="👥")
    st.divider()

    try:
        # --- 1. Load Data and Prepare Dependencies ---
        hf_data: Dict[str, Any] = ssm.get_data("human_factors")
        rmf_data: Dict[str, Any] = ssm.get_data("risk_management_file")
        val_data: Dict[str, Any] = ssm.get_data("clinical_study")
        logger.info("Loaded human factors, risk, and validation data.")

        hazards: List[Dict[str, Any]] = rmf_data.get("hazards", [])
        hazard_ids: List[str] = [""] + sorted([h.get('id', '') for h in hazards if h.get('id')])

        hf_val_studies: List[Dict[str, Any]] = val_data.get("hf_studies", [])
        hf_val_ids: List[str] = [""] + sorted([s.get('id', '') for s in hf_val_studies if s.get('id')])
        
        # --- 2. Define Tabs for HFE Process ---
        tab_titles = [
            "1. User Profiles & Use Environments",
            "2. URRA: Sample Collection (Phlebotomist)",
            "3. URRA: Laboratory Processing (Lab Tech)",
            "4. URRA: Clinical Report (Oncologist)"
        ]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        # --- Helper Function for Rendering Tables ---
        def render_editor_tab(table_key: str, df: pd.DataFrame, column_config: dict):
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                key=f"hf_editor_{table_key}",
                column_config=column_config,
                hide_index=True
            )
            
            if edited_df.to_dict('records') != df.to_dict('records'):
                hf_data[table_key] = edited_df.to_dict('records')
                ssm.update_data(hf_data, "human_factors")
                st.toast(f"{table_key.replace('_', ' ').title()} data updated!", icon="✅")
                st.rerun()

        # --- Tab 1: User Profiles ---
        with tab1:
            st.subheader("Intended Users and Use Environments")
            st.caption("Define the characteristics of each user group and the context in which they will interact with the diagnostic service. This is a foundational step for all usability analysis.")
            user_profiles_df = pd.DataFrame(hf_data.get("user_profiles", []))
            render_editor_tab(
                "user_profiles",
                user_profiles_df,
                {
                    "user_group": st.column_config.TextColumn("User Group", required=True),
                    "description": st.column_config.TextColumn("User Profile Description", width="large", help="Describe key characteristics: education, technical skills, potential disabilities.", required=True),
                    "training": st.column_config.TextColumn("Assumed Training/Knowledge"),
                    "use_environment": st.column_config.TextColumn("Intended Use Environment", width="large", help="E.g., 'Busy outpatient phlebotomy clinic', 'High-throughput clinical lab', 'Oncologist's office during patient consultation'.")
                }
            )

        # --- Common Column Config for URRA Tables ---
        urra_column_config = {
            "id": st.column_config.TextColumn("URRA ID", required=True),
            "user_task": st.column_config.TextColumn("Critical User Task", width="large", required=True),
            "potential_use_error": st.column_config.TextColumn("Potential Use Error", width="large", help="How could the user perform the task incorrectly?"),
            "potential_harm": st.column_config.TextColumn("Resulting Potential Harm", width="large"),
            "related_hazard_id": st.column_config.SelectboxColumn("Links to System Hazard", options=hazard_ids, help="Link this use error to a formal system hazard."),
            "risk_control_measure": st.column_config.TextColumn("Risk Control Measure", width="large", help="Describe the design mitigation for this error (e.g., simplified IFU, warning in software)."),
            "validation_link": st.column_config.SelectboxColumn("HFE Validation Link", options=hf_val_ids, help="Link to the summative usability study that validates this control.")
        }

        # --- Tab 2: URRA - Sample Collection ---
        with tab2:
            st.subheader("Use-Related Risk Analysis: Sample Collection Kit & IFU")
            st.caption("Identify tasks, potential errors, and resulting harms related to the phlebotomist's interaction with the sample collection kit and Instructions For Use (IFU).")
            kit_urra_df = pd.DataFrame(hf_data.get("kit_urra", []))
            render_editor_tab("kit_urra", kit_urra_df, urra_column_config)

        # --- Tab 3: URRA - Lab Processing ---
        with tab3:
            st.subheader("Use-Related Risk Analysis: Laboratory Processing")
            st.caption("Identify tasks, potential errors, and resulting harms related to the laboratory technologist's interaction with the samples, reagents, instruments, and LIMS.")
            lab_urra_df = pd.DataFrame(hf_data.get("lab_urra", []))
            render_editor_tab("lab_urra", lab_urra_df, urra_column_config)

        # --- Tab 4: URRA - Clinical Report ---
        with tab4:
            st.subheader("Use-Related Risk Analysis: Clinical Test Report")
            st.caption("Identify tasks, potential errors, and resulting harms related to the oncologist's interpretation of the final test report.")
            report_urra_df = pd.DataFrame(hf_data.get("report_urra", []))
            render_editor_tab("report_urra", report_urra_df, urra_column_config)

    except Exception as e:
        st.error("An error occurred while displaying the Human Factors section. The data may be malformed.")
        logger.error(f"Failed to render human factors: {e}", exc_info=True)
