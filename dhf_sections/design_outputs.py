# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Renders the Design Outputs section of the DHF dashboard.

This module provides a structured, categorized UI for managing all tangible
outputs of the design process. It ensures each output (specifications, code,
procedures, etc.) is version-controlled, its status is tracked, and it is
traceably linked to a design input, as required by 21 CFR 820.30(d).
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


def render_design_outputs(ssm: SessionStateManager) -> None:
    """
    Renders the UI for managing categorized Design Outputs.

    This function displays outputs in separate tabs for clarity (Assay, Software,
    Hardware, Labeling). It provides an editable table within each tab to manage
    the outputs, including their version, status, and the critical traceability
    link back to a specific design input requirement.

    Args:
        ssm (SessionStateManager): The session state manager to access DHF data.
    """
    st.header("5. Design Outputs")
    st.markdown("""
    *As per 21 CFR 820.30(d).*

    Design Outputs are the full set of specifications, procedures, and artifacts that define the entire diagnostic service. They are the tangible results of the design process and form the basis for the Device Master Record (DMR). **Crucially, each output must be traceable to a design input, be approved, and be placed under formal version control.**
    """)
    st.info("Use the tabs below to manage the outputs for each part of the system. Every output must be linked to an input requirement. Changes are saved automatically.", icon="ℹ️")

    try:
        # --- 1. Load Data and Prepare Dependencies ---
        outputs_data: List[Dict[str, Any]] = ssm.get_data("design_outputs", "documents")
        inputs_data: List[Dict[str, Any]] = ssm.get_data("design_inputs", "requirements")
        logger.info(f"Loaded {len(outputs_data)} design output records and {len(inputs_data)} input records.")

        if not inputs_data:
            st.warning("⚠️ No Design Inputs found. Please add requirements in '4. Design Inputs' before creating outputs.", icon="❗")
            return
        
        # Create a user-friendly mapping for the requirement selection dropdown
        req_options = [f"{req.get('id', '')}: {req.get('description', '')[:70]}..." for req in inputs_data]
        req_map = {option: req.get('id', '') for option, req in zip(req_options, inputs_data)}
        reverse_req_map = {v: k for k, v in req_map.items()}

        # --- 2. Define Tabs for Each Output Category ---
        tab_titles = [
            "Assay & Reagent Specifications",
            "Software & Algorithm Outputs",
            "Hardware & Kit Specifications",
            "Labeling & Procedures"
        ]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        # --- Helper function for rendering editor tabs ---
        def render_editor_tab(
            title: str,
            df: pd.DataFrame,
            key_suffix: str,
            column_config: Dict,
            help_text: str
        ):
            st.subheader(title)
            st.caption(help_text)
            
            # Use a copy for safe manipulation and comparison
            df_display = df.copy()
            
            # Map the raw ID to the descriptive text for display in the editor
            if 'linked_input_id' in df_display.columns:
                df_display['linked_input_descriptive'] = df_display['linked_input_id'].map(reverse_req_map)
            else:
                df_display['linked_input_descriptive'] = pd.Series(dtype=str)

            # Ensure the required columns exist before editing
            for col, config in column_config.items():
                if col not in df_display.columns and config is not None:
                    # Initialize with a default value appropriate for the column type
                    if isinstance(config, (st.column_config.TextColumn, st.column_config.LinkColumn)):
                        df_display[col] = ""
                    elif isinstance(config, st.column_config.SelectboxColumn):
                        df_display[col] = config.default if config.default is not None else (config.options[0] if config.options else None)
                    else:
                        df_display[col] = pd.NA
            
            # Define a consistent column order for the editor
            column_order = [
                "id", "type", "title", "version", "status", "approval_date", 
                "linked_input_descriptive", "link_to_artifact"
            ]
            # Filter order to only columns present in the config
            display_columns = [col for col in column_order if col in column_config and column_config[col] is not None]

            edited_tab_df = st.data_editor(
                df_display,
                num_rows="dynamic",
                use_container_width=True,
                key=f"output_editor_{key_suffix}",
                column_config=column_config,
                column_order=display_columns,
                hide_index=True
            )

            if edited_tab_df.to_dict('records') != df_display.to_dict('records'):
                # Map the descriptive text back to the raw ID for storage
                if 'linked_input_descriptive' in edited_tab_df.columns:
                    valid_rows = edited_tab_df['linked_input_descriptive'].notna()
                    edited_tab_df.loc[valid_rows, 'linked_input_id'] = edited_tab_df.loc[valid_rows, 'linked_input_descriptive'].map(req_map)
                
                # Drop the temporary display column before saving
                edited_tab_df.drop(columns=['linked_input_descriptive'], inplace=True, errors='ignore')

                # Merge back with other types and save
                other_outputs = [out for out in outputs_data if out.get('type') not in df['type'].unique()]
                updated_all_outputs = other_outputs + edited_tab_df.to_dict('records')
                ssm.update_data(updated_all_outputs, "design_outputs", "documents")
                logger.info(f"Design outputs for '{title}' updated.")
                st.toast(f"{title} saved!", icon="✅")
                st.rerun()

        # --- 3. Render Each Tab with Specific Configurations ---
        df_all = pd.DataFrame(outputs_data) if outputs_data else pd.DataFrame()
        status_options = ["Draft", "In Review", "Approved", "Obsolete"]

        with tab1: # Assay & Reagent
            assay_df = df_all[df_all['type'] == 'Assay Spec'].copy()
            render_editor_tab(
                "Assay & Reagent Specifications", assay_df, "assay",
                {
                    "id": st.column_config.TextColumn("Spec ID", required=True),
                    "type": st.column_config.SelectboxColumn("Type", options=["Assay Spec"], default="Assay Spec", required=True),
                    "title": st.column_config.TextColumn("Specification Title", width="large", required=True),
                    "version": st.column_config.TextColumn("Version", required=True, help="E.g., 1.0, 1.1, 2.0"),
                    "status": st.column_config.SelectboxColumn("Status", options=status_options, required=True),
                    "approval_date": st.column_config.DateColumn("Approval Date", format="YYYY-MM-DD"),
                    "linked_input_descriptive": st.column_config.SelectboxColumn("Traces to Requirement", options=req_options, required=True),
                    "link_to_artifact": st.column_config.LinkColumn("Link to Spec", help="Link to controlled document in eQMS.")
                },
                "Define the specifications for all wet-lab components, including oligo sequences, buffer formulations, and reagent QC tests."
            )

        with tab2: # Software & Algorithm
            sw_df = df_all[df_all['type'].isin(['Software Spec', 'Algorithm'])].copy()
            render_editor_tab(
                "Software & Algorithm Outputs", sw_df, "software",
                {
                    "id": st.column_config.TextColumn("Artifact ID", required=True),
                    "type": st.column_config.SelectboxColumn("Type", options=["Software Spec", "Algorithm"], required=True),
                    "title": st.column_config.TextColumn("Artifact Title", width="large", required=True),
                    "version": st.column_config.TextColumn("Version/Commit Hash", required=True, help="E.g., 2.1.3 or a Git commit hash."),
                    "status": st.column_config.SelectboxColumn("Status", options=status_options, required=True),
                    "approval_date": st.column_config.DateColumn("Approval Date", format="YYYY-MM-DD"),
                    "linked_input_descriptive": st.column_config.SelectboxColumn("Traces to Requirement", options=req_options, required=True),
                    "link_to_artifact": st.column_config.LinkColumn("Link to Repo/Doc", help="Link to Git repo, SDS, or locked model file.")
                },
                "List the Software Design Specifications (SDS), locked algorithm models, and source code repositories that define the bioinformatics pipeline and classifier."
            )

        with tab3: # Hardware & Kit
            kit_df = df_all[df_all['type'] == 'Hardware Spec'].copy()
            render_editor_tab(
                "Hardware & Kit Specifications", kit_df, "kit",
                {
                    "id": st.column_config.TextColumn("Spec ID", required=True),
                    "type": st.column_config.SelectboxColumn("Type", options=["Hardware Spec"], default="Hardware Spec", required=True),
                    "title": st.column_config.TextColumn("Specification Title", width="large", required=True),
                    "version": st.column_config.TextColumn("Version", required=True),
                    "status": st.column_config.SelectboxColumn("Status", options=status_options, required=True),
                    "approval_date": st.column_config.DateColumn("Approval Date", format="YYYY-MM-DD"),
                    "linked_input_descriptive": st.column_config.SelectboxColumn("Traces to Requirement", options=req_options, required=True),
                    "link_to_artifact": st.column_config.LinkColumn("Link to Drawing/Spec", help="Link to CAD drawing or material spec.")
                },
                "Define the specifications for the physical components of the service, such as the blood collection tube, packaging, and shipping materials."
            )

        with tab4: # Labeling & Procedures
            proc_df = df_all[df_all['type'].isin(['Labeling', 'Procedure'])].copy()
            render_editor_tab(
                "Labeling & Procedures", proc_df, "proc",
                {
                    "id": st.column_config.TextColumn("Doc ID", required=True),
                    "type": st.column_config.SelectboxColumn("Type", options=["Labeling", "Procedure"], required=True),
                    "title": st.column_config.TextColumn("Document Title", width="large", required=True),
                    "version": st.column_config.TextColumn("Version", required=True),
                    "status": st.column_config.SelectboxColumn("Status", options=status_options, required=True),
                    "approval_date": st.column_config.DateColumn("Approval Date", format="YYYY-MM-DD"),
                    "linked_input_descriptive": st.column_config.SelectboxColumn("Traces to Requirement", options=req_options, required=True),
                    "link_to_artifact": st.column_config.LinkColumn("Link to Document", help="Link to controlled document in eQMS (e.g., IFU, SOP, Report Template).")
                },
                "List all controlled documents that will be part of the final service, including the Instructions for Use (IFU), lab SOPs, and the Clinical Report template."
            )

    except Exception as e:
        st.error("An error occurred while displaying the Design Outputs section. The data may be malformed.")
        logger.error(f"Failed to render design outputs: {e}", exc_info=True)
