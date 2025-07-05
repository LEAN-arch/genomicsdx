# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Renders the Design and Development Plan section of the DHF dashboard.

This module provides a comprehensive UI for creating and editing the foundational
planning document for the MCED diagnostic program, as required by 21 CFR 820.30(b)
and ISO 13485:2016, Section 7.3.1.
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict

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

    This function displays a structured form to capture all aspects of the
    design plan, including scope, team responsibilities, regulatory strategy,
    and key planning elements for clinical, analytical, and software validation.
    All changes are saved back to the session state.

    Args:
        ssm (SessionStateManager): The session state manager to access DHF data.
    """
    st.header("1. Design and Development Plan")
    st.markdown("""
    *As per 21 CFR 820.30(b) and ISO 13485:2016, Section 7.3.*

    This is the master plan that describes and references all design and development activities. It outlines the project scope, team, regulatory pathway, and the strategies for proving the safety and effectiveness of the **GenomicsDx Sentry™ MCED Test**. This living document will be updated as the project evolves.
    """)
    st.info("Changes made on this page are saved automatically upon interaction.", icon="ℹ️")
    st.divider()

    try:
        # --- 1. Load Data ---
        plan_data: Dict[str, Any] = ssm.get_data("design_plan")
        logger.info("Loaded design plan data.")

        # --- 2. Render UI Sections ---
        with st.expander("**1.1 Project Overview & Scope**", expanded=True):
            plan_data["project_name"] = st.text_input(
                "**Project Name**",
                value=plan_data.get("project_name", ""),
                key="dp_project_name",
                help="The official name of the MCED diagnostic program."
            )
            plan_data["scope"] = st.text_area(
                "**Project Scope**",
                value=plan_data.get("scope", ""),
                key="dp_scope",
                height=120,
                help="Describe the overall goals of the project, including the development of the assay, software, and operational infrastructure."
            )
            plan_data["intended_use"] = st.text_area(
                "**Intended Use / Indications for Use Statement**",
                value=plan_data.get("intended_use", ""),
                key="dp_intended_use",
                height=150,
                help="This is a formal regulatory statement. E.g., 'The GenomicsDx Sentry™ Test is a qualitative, blood-based, multi-cancer early detection (MCED) screening test for individuals of average risk, aged 50-79...'"
            )

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
            plan_data["standards"] = edited_standards_df.to_dict('records')

        with st.expander("**1.3 Clinical & Analytical Development Strategy**", expanded=True):
            st.markdown("**Pivotal Clinical Development Plan**")
            plan_data["clinical_dev_plan"] = st.text_area(
                "Clinical Strategy Summary",
                value=plan_data.get("clinical_dev_plan", ""),
                key="dp_clinical_plan",
                height=150,
                help="Summarize the strategy for the pivotal clinical trial, including study design (e.g., prospective cohort), target population, primary/secondary endpoints, and sample size justification."
            )

            st.markdown("**Analytical Validation (AV) Master Plan**")
            plan_data["av_master_plan_ref"] = st.text_input(
                "AV Master Plan Document ID",
                value=plan_data.get("av_master_plan_ref", ""),
                key="dp_av_plan_ref",
                help="Reference to the formal AV Master Plan document outlining all required analytical performance studies (LoD, Precision, etc.) based on CLSI guidelines."
            )
            
        with st.expander("**1.4 Software & Bioinformatics Development Plan (ISO 62304)**", expanded=True):
            loc_options = ["Major", "Moderate", "Minor"]
            current_loc = plan_data.get("software_level_of_concern", "Major")
            loc_index = loc_options.index(current_loc) if current_loc in loc_options else 0
            
            plan_data["software_level_of_concern"] = st.selectbox(
                "**Software Level of Concern (per FDA Guidance):**",
                options=loc_options,
                index=loc_index,
                key="dp_sw_loc",
                help="Determines the required rigor of software documentation. For a life-saving diagnostic, this is almost always 'Major'."
            )
            st.text_input("**Software as a Medical Device (SaMD) Classification**", value="Class C (High Risk) per IMDRF", disabled=True)
            plan_data["sw_dev_plan"] = st.text_area(
                "Bioinformatics & Software Development Lifecycle (SDLC) Summary",
                value=plan_data.get("sw_dev_plan", ""),
                key="dp_sw_plan",
                height=150,
                help="Summarize the SDLC model (e.g., Agile with phase-gates), coding standards, version control strategy (Git), and the plan for software verification and validation, including the criteria for final algorithm lock."
            )

        with st.expander("**1.5 Risk Management Plan (ISO 14971)**", expanded=True):
            plan_data["risk_management_plan_ref"] = st.text_input(
                "**Risk Management Plan Document ID**",
                value=plan_data.get("risk_management_plan_ref", "RMP-001"),
                key="dp_rmp_ref",
                help="Reference to the main Risk Management Plan document governing all risk management activities."
            )
        
        with st.expander("**1.6 Design Review Plan**", expanded=True):
            st.markdown("Define the major, formal Design Reviews that serve as project phase-gates.")
            reviews_df = pd.DataFrame(plan_data.get("design_review_plan", []))
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
            plan_data["design_review_plan"] = edited_reviews_df.to_dict('records')

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
            plan_data["team_members"] = edited_team_df.to_dict('records')

        # --- 3. Persist All Changes ---
        # A single update call at the end ensures atomicity for this section's edits if this were a single form,
        # but with st.data_editor and individual widgets, changes are handled per widget interaction.
        # We need to explicitly check and update for text/selectbox widgets.
        if ssm.get_data("design_plan") != plan_data:
            ssm.update_data(plan_data, "design_plan")
            logger.debug("Design plan data updated in session state.")
            # A toast can be added here, but it might be too frequent.
            # st.toast("Design plan updated!")

    except Exception as e:
        st.error("An error occurred while displaying the Design Plan section. The data may be malformed.")
        logger.error(f"Failed to render design plan: {e}", exc_info=True)
