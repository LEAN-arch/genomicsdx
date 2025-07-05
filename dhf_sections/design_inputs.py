# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Renders the Design Inputs section of the DHF dashboard.

This module provides a structured, hierarchical UI for managing all product
requirements, which serve as the foundation for the DHF. It categorizes inputs
into clinical, system, software, and physical domains, as required for a
complex genomic diagnostic service under 21 CFR 820.30(c).
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


def render_design_inputs(ssm: SessionStateManager) -> None:
    """
    Renders the hierarchical UI for managing all Design Inputs.

    This function displays requirements in categorized tabs for clarity and
    manages traceability between requirement levels (e.g., System -> Software)
    and to risk controls.

    Args:
        ssm (SessionStateManager): The session state manager to access DHF data.
    """
    st.header("4. Design Inputs")
    st.markdown("""
    *As per 21 CFR 820.30(c), ISO 13485:2016 Section 7.3.3, and ISO 14971.*

    This section captures all requirements for the diagnostic service. This includes high-level **User Needs**, which are decomposed into testable **System, Assay, Software, and Kit Requirements**. Requirements derived from risk analysis (**Risk Controls**) are also managed here. Each requirement must be unambiguous, verifiable, and traceable.
    """)
    st.info("Use the tabs below to manage requirements for each domain. Changes are saved automatically.", icon="ℹ️")

    try:
        # --- 1. Load All Necessary Data ---
        inputs_data: List[Dict[str, Any]] = ssm.get_data("design_inputs", "requirements")
        rmf_data: Dict[str, Any] = ssm.get_data("risk_management_file", {})
        hazards: List[Dict[str, Any]] = rmf_data.get("hazards", [])
        hazard_ids: List[str] = [""] + sorted([h.get('id', '') for h in hazards if h.get('id')])

        # Create a full list of requirement IDs for parent-child linking
        all_req_ids = [""] + sorted([req.get("id") for req in inputs_data if req.get("id")])

        # --- 2. Define Tabs for Each Requirement Category ---
        tab_titles = [
            "User Needs & Intended Use",
            "System & Assay Requirements",
            "Software Requirements",
            "Sample Kit & Labeling Requirements"
        ]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        # --- Helper function for rendering editor tabs ---
        def render_editor_tab(
            title: str,
            df: pd.DataFrame,
            key: str,
            column_config: Dict,
            help_text: str
        ):
            st.subheader(title)
            st.caption(help_text)

            # Isolate original data for this type
            original_tab_data = df.to_dict('records')

            edited_tab_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                key=key,
                column_config=column_config,
                hide_index=True
            )

            # Compare and update if changed
            if edited_tab_df.to_dict('records') != original_tab_data:
                # Merge back with other types and save
                other_reqs = [req for req in inputs_data if req.get('type') not in df['type'].unique()]
                updated_all_reqs = other_reqs + edited_tab_df.to_dict('records')
                ssm.update_data(updated_all_reqs, "design_inputs", "requirements")
                logger.info(f"Design inputs for '{title}' updated.")
                st.toast(f"{title} saved!", icon="✅")
                st.rerun()

        # --- 3. Render Each Tab ---
        df_all = pd.DataFrame(inputs_data) if inputs_data else pd.DataFrame(columns=['id', 'type', 'description', 'parent_id', 'status', 'is_risk_control', 'related_hazard_id'])

        with tab1: # User Needs
            un_df = df_all[df_all['type'] == 'User Need'].copy()
            render_editor_tab(
                "User Needs & Intended Use", un_df, "un_editor",
                {
                    "id": st.column_config.TextColumn("ID", help="E.g., UN-001", required=True),
                    "type": st.column_config.SelectboxColumn("Type", options=["User Need"], default="User Need", required=True),
                    "description": st.column_config.TextColumn("Description", width="large", help="High-level need from a user perspective.", required=True),
                    "status": st.column_config.SelectboxColumn("Status", options=["Active", "Obsolete"], default="Active", required=True),
                    "parent_id": None, "is_risk_control": None, "related_hazard_id": None # Hide irrelevant columns
                },
                "Define the highest-level clinical and user needs. These are the inputs to the Design Validation process."
            )

        with tab2: # System & Assay Requirements
            sa_df = df_all[df_all['type'].isin(['System', 'Assay'])].copy()
            render_editor_tab(
                "System & Assay Requirements", sa_df, "sa_editor",
                {
                    "id": st.column_config.TextColumn("ID", help="E.g., SYS-001, ASY-001", required=True),
                    "type": st.column_config.SelectboxColumn("Type", options=["System", "Assay"], required=True),
                    "description": st.column_config.TextColumn("Description", width="large", help="Testable performance requirement.", required=True),
                    "parent_id": st.column_config.SelectboxColumn("Traces to User Need", options=all_req_ids, help="Which User Need does this requirement satisfy?"),
                    "status": st.column_config.SelectboxColumn("Status", options=["Active", "Obsolete"], default="Active", required=True),
                    "is_risk_control": st.column_config.CheckboxColumn("Is Risk Control?", default=False),
                    "related_hazard_id": st.column_config.SelectboxColumn("Mitigates Hazard ID", options=hazard_ids)
                },
                "Define the top-level, quantitative performance requirements for the entire service and the core assay. These are the inputs to the Analytical Validation process."
            )

        with tab3: # Software Requirements
            sw_df = df_all[df_all['type'] == 'Software'].copy()
            render_editor_tab(
                "Software Requirements", sw_df, "sw_editor",
                {
                    "id": st.column_config.TextColumn("ID", help="E.g., SW-001", required=True),
                    "type": st.column_config.SelectboxColumn("Type", options=["Software"], default="Software", required=True),
                    "description": st.column_config.TextColumn("Description", width="large", help="Specific, testable software function.", required=True),
                    "parent_id": st.column_config.SelectboxColumn("Traces to System Req.", options=all_req_ids, help="Which System/Assay requirement does this software function support?"),
                    "status": st.column_config.SelectboxColumn("Status", options=["Active", "Obsolete"], default="Active", required=True),
                    "is_risk_control": st.column_config.CheckboxColumn("Is Risk Control?", default=False),
                    "related_hazard_id": st.column_config.SelectboxColumn("Mitigates Hazard ID", options=hazard_ids)
                },
                "Define the detailed requirements for the bioinformatics pipeline, classifier, LIMS interface, and reporting tool. Governed by ISO 62304."
            )

        with tab4: # Kit & Labeling Requirements
            kl_df = df_all[df_all['type'].isin(['Kit', 'Labeling'])].copy()
            render_editor_tab(
                "Sample Kit & Labeling Requirements", kl_df, "kl_editor",
                {
                    "id": st.column_config.TextColumn("ID", help="E.g., KIT-001, LBL-001", required=True),
                    "type": st.column_config.SelectboxColumn("Type", options=["Kit", "Labeling"], required=True),
                    "description": st.column_config.TextColumn("Description", width="large", help="Requirement for physical kit or user-facing text.", required=True),
                    "parent_id": st.column_config.SelectboxColumn("Traces to System Req.", options=all_req_ids, help="Which System requirement does this support?"),
                    "status": st.column_config.SelectboxColumn("Status", options=["Active", "Obsolete"], default="Active", required=True),
                    "is_risk_control": st.column_config.CheckboxColumn("Is Risk Control?", default=False),
                    "related_hazard_id": st.column_config.SelectboxColumn("Mitigates Hazard ID", options=hazard_ids)
                },
                "Define requirements for the physical sample collection kit (e.g., tube, packaging) and all labeling (e.g., IFU, Report Text). Governed by 21 CFR 809.10 and Human Factors Engineering."
            )

    except Exception as e:
        st.error("An error occurred while displaying the Design Inputs section. The data may be malformed.")
        logger.error(f"Failed to render design inputs: {e}", exc_info=True)
