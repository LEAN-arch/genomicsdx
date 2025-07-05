# --- SME OVERHAUL: Definitive, Compliance-Focused Version (Corrected) ---
"""
Renders the Design and Development Plan section of the DHF dashboard.

This module provides a comprehensive UI for creating and editing the foundational
planning document for the MCED diagnostic program, as required by 21 CFR 820.30(b)
and ISO 13485:2016, Section 7.3.1.
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


def render_design_plan(ssm: SessionStateManager) -> None:
    """
    Renders the UI for the Design and Development Plan section.
    """
    st.header("1. Design and Development Plan")
    st.markdown("""
    *As per 21 CFR 820.30(b) and ISO 13485:2016, Section 7.3.*

    This is the master plan that describes and references all design and development activities. It outlines the project scope, team, regulatory pathway, and the strategies for proving the safety and effectiveness of the **GenomicsDx Sentry™ MCED Test**. This living document will be updated as the project evolves.
    """)
    st.info("Use the sections below to manage the plan. Use the 'Save Plan Changes' button at the bottom to commit all edits.", icon="ℹ️")
    st.divider()

    try:
        # --- 1. Load Data ---
        plan_data: Dict[str, Any] = ssm.get_data("design_plan")
        logger.info("Loaded design plan data.")

        # --- 2. Render UI Sections within a single form for atomic updates ---
        with st.form("design_plan_form"):
            with st.expander("**1.1 Project Overview & Scope**", expanded=True):
                project_name_val = st.text_input(
                    "**Project Name**",
                    value=plan_data.get("project_name", ""),
                )
                scope_val = st.text_area(
                    "**Project Scope**",
                    value=plan_data.get("scope", ""),
                    height=120,
                )
                # --- SYNTAX CORRECTION: Terminated the string literal ---
                intended_use_val = st.text_area(
                    "**Intended Use / Indications for Use Statement**",
                    value=plan_data.get("intended_use", ""),
                    height=150,
                    help="This is a formal regulatory statement. E.g., 'The GenomicsDx Sentry™ Test is a qualitative, blood-based, multi-cancer early detection (MCED) screening test...'"
                )
                # --- END CORRECTION ---

            with st.expander("**1.2 Regulatory Strategy & Compliance Framework**", expanded=True):
                cols = st.columns(3)
                cols[0].text_input("**Device Classification**", value="Class III", disabled=True)
                cols[1].text_input("**Submission Pathway**", value="Premarket Approval (PMA)", disabled=True)
                cols[2].text_input("**Special Designations**", value="Breakthrough Device Designation", disabled=True)
                
                st.markdown("**Applicable Standards & Guidance Documents**")
                standards_df = pd.DataFrame(plan_data.get("standards", []))
                edited_standards_df = st.data_editor(
                    standards_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "id": st.column_config.TextColumn("Document ID", required=True),
                        "title": st.column_config.TextColumn("Title/Description", required=True, width="large"),
                        "category": st.column_config.SelectboxColumn("Category", options=["Regulation", "ISO Standard", "FDA Guidance", "CLSI Guideline"], required=True)
                    },
                    key="dp_standards_editor"
                )

            with st.expander("**1.3 Clinical & Analytical Development Strategy**", expanded=True):
                clinical_dev_plan_val = st.text_area(
                    "**Clinical Strategy Summary**",
                    value=plan_data.get("clinical_dev_plan", ""),
                    height=150,
                )
                av_master_plan_ref_val = st.text_input(
                    "**AV Master Plan Document ID**",
                    value=plan_data.get("av_master_plan_ref", ""),
                )
            
            with st.expander("**1.4 Software & Bioinformatics Development Plan (ISO 62304)**", expanded=True):
                loc_options = ["Major", "Moderate", "Minor"]
                current_loc = plan_data.get("software_level_of_concern", "Major")
                loc_index = loc_options.index(current_loc) if current_loc in loc_options else 0
                software_level_of_concern_val = st.selectbox(
                    "**Software Level of Concern (per FDA Guidance):**",
                    options=loc_options,
                    index=loc_index,
                )
                st.text_input("**Software as a Medical Device (SaMD) Classification**", value="Class C (High Risk) per IMDRF", disabled=True)
                sw_dev_plan_val = st.text_area(
                    "**Bioinformatics & Software Development Lifecycle (SDLC) Summary**",
                    value=plan_data.get("sw_dev_plan", ""),
                    height=150,
                )

            with st.expander("**1.5 Risk Management Plan (ISO 14971)**", expanded=True):
                risk_management_plan_ref_val = st.text_input(
                    "**Risk Management Plan Document ID**",
                    value=plan_data.get("risk_management_plan_ref", "RMP-001"),
                )
            
            with st.expander("**1.6 Design Review Plan**", expanded=True):
                st.markdown("Define the major, formal Design Reviews that serve as project phase-gates.")
                
                reviews_df = pd.DataFrame(plan_data.get("design_review_plan", []))
                if 'planned_date' in reviews_df.columns:
                    reviews_df['planned_date'] = pd.to_datetime(reviews_df['planned_date'], errors='coerce')

                edited_reviews_df = st.data_editor(
                    reviews_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "phase_name": st.column_config.TextColumn("Phase/Gate Name", required=True),
                        "description": st.column_config.TextColumn("Description", width="large"),
                        "planned_date": st.column_config.DateColumn("Planned Date", format="YYYY-MM-DD")
                    },
                    key="dp_reviews_editor"
                )

            with st.expander("**1.7 Team and Responsibilities**", expanded=True):
                st.markdown("Define roles, assign team members, and outline their key responsibilities for the program.")
                team_df = pd.DataFrame(plan_data.get("team_members", []))
                edited_team_df = st.data_editor(
                    team_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "role": st.column_config.TextColumn("Role", required=True),
                        "name": st.column_config.TextColumn("Assigned Member", required=True),
                        "responsibility": st.column_config.TextColumn("Key Responsibilities", width="large"),
                    },
                    key="design_plan_team_editor",
                    hide_index=True
                )
            
            submitted = st.form_submit_button("Save Plan Changes", use_container_width=True, type="primary")
            if submitted:
                plan_data["project_name"] = project_name_val
                plan_data["scope"] = scope_val
                plan_data["intended_use"] = intended_use_val
                plan_data["standards"] = edited_standards_df.to_dict('records')
                plan_data["clinical_dev_plan"] = clinical_dev_plan_val
                plan_data["av_master_plan_ref"] = av_master_plan_ref_val
                plan_data["software_level_of_concern"] = software_level_of_concern_val
                plan_data["sw_dev_plan"] = sw_dev_plan_val
                plan_data["risk_management_plan_ref"] = risk_management_plan_ref_val
                
                reviews_to_save_df = edited_reviews_df.copy()
                if 'planned_date' in reviews_to_save_df.columns:
                     reviews_to_save_df['planned_date'] = reviews_to_save_df['planned_date'].dt.strftime('%Y-%m-%d').replace({pd.NaT: None})
                plan_data["design_review_plan"] = reviews_to_save_df.to_dict('records')

                plan_data["team_members"] = edited_team_df.to_dict('records')

                ssm.update_data(plan_data, "design_plan")
                logger.info("Design plan data updated in session state.")
                st.toast("Design & Development Plan saved!", icon="✅")
                st.rerun()

    except Exception as e:
        st.error("An error occurred while displaying the Design Plan section. The data may be malformed.")
        logger.error(f"Failed to render design plan: {e}", exc_info=True)
