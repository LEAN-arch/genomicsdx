# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Renders the Design Verification / Analytical Validation section of the DHF.

This module provides a comprehensive dashboard for managing the extensive
Analytical Validation (AV) program required for a PMA-class genomic diagnostic.
It confirms that design outputs meet design inputs, as per 21 CFR 820.30(f),
by tracking the specific performance characteristic studies recommended by
CLSI guidelines.
"""

# --- Standard Library Imports ---
# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
# ... (docstring)
# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List
# --- Third-party Imports ---
import pandas as pd
import streamlit as st
# --- Local Application Imports (CORRECTED) ---
from ..utils.session_state_manager import SessionStateManager
# ... (rest of file is unchanged) ...

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def render_design_verification(ssm: SessionStateManager) -> None:
    """
    Renders the UI for the Design Verification / Analytical Validation section.

    This function displays the AV program in categorized tabs, allowing for
    detailed management of studies for precision, sensitivity, specificity, etc.
    It ensures full traceability from tests back to requirements and outputs.

    Args:
        ssm (SessionStateManager): The session state manager to access DHF data.
    """
    st.header("7. Design Verification (Analytical Validation)")
    st.markdown("""
    *As per 21 CFR 820.30(f) and relevant CLSI Guidelines (e.g., EP05, EP17, EP12).*

    Verification confirms through objective evidence that design outputs meet design inputs. For this IVD, this is primarily achieved through a comprehensive **Analytical Validation (AV)** program that characterizes the performance of the assay and bioinformatics pipeline. It answers the question: **"Did we build the assay and system right?"**
    """)
    st.info("Use the tabs below to manage the test protocols and reports for each required analytical performance study.", icon="🔬")
    st.divider()

    try:
        # --- 1. Load Data and Prepare Dependencies ---
        verification_data: List[Dict[str, Any]] = ssm.get_data("design_verification", "tests")
        outputs_data: List[Dict[str, Any]] = ssm.get_data("design_outputs", "documents")
        inputs_data: List[Dict[str, Any]] = ssm.get_data("design_inputs", "requirements")
        logger.info(f"Loaded {len(verification_data)} verification tests.")

        # Prepare lists for traceability dropdowns
        output_ids: List[str] = [""] + sorted([doc.get('id', '') for doc in outputs_data if doc.get('id')])
        input_ids: List[str] = [""] + sorted([req.get('id', '') for req in inputs_data if req.get('id')])
        
        # --- 2. High-Level AV Program KPIs ---
        st.subheader("Analytical Validation Program Status")
        if verification_data:
            av_df = pd.DataFrame(verification_data)
            total_tests = len(av_df)
            completed_tests = len(av_df[av_df['status'] == 'Completed'])
            passing_tests = len(av_df[av_df['result'] == 'Pass'])
            
            progress = (completed_tests / total_tests) * 100 if total_tests > 0 else 0
            pass_rate = (passing_tests / completed_tests) * 100 if completed_tests > 0 else 0

            kpi_cols = st.columns(3)
            kpi_cols[0].metric("AV Program Completion", f"{progress:.1f}%", f"{completed_tests}/{total_tests} Protocols Executed")
            kpi_cols[1].metric("AV Pass Rate", f"{pass_rate:.1f}%", f"{passing_tests}/{completed_tests} Passing")
            st.progress(progress / 100)
        else:
            st.warning("No verification activities have been logged yet.")
        st.divider()

        # --- 3. Define Tabs for Each AV Study Category ---
        tab_titles = [
            "Precision & Reproducibility",
            "Analytical Sensitivity (LoB/LoD/LoQ)",
            "Analytical Specificity & Interference",
            "Assay Robustness & Stability",
            "Software & System Verification"
        ]
        tabs = st.tabs(tab_titles)

        # --- Helper function for rendering editor tabs ---
        def render_av_tab(tab_name: str, test_types: List[str], help_text: str):
            st.caption(help_text)
            
            tab_data = [t for t in verification_data if t.get('test_type') in test_types]
            tab_df = pd.DataFrame(tab_data)

            edited_df = st.data_editor(
                tab_df,
                num_rows="dynamic",
                use_container_width=True,
                key=f"verification_editor_{tab_name}",
                column_config={
                    "id": st.column_config.TextColumn("Protocol ID", required=True),
                    "test_type": st.column_config.SelectboxColumn("Test Type", options=test_types, required=True),
                    "test_name": st.column_config.TextColumn("Test/Protocol Name", width="large", required=True),
                    "input_verified_id": st.column_config.SelectboxColumn("Verifies Requirement", options=input_ids, required=True),
                    "output_verified_id": st.column_config.SelectboxColumn("Tests Output", options=output_ids, required=True),
                    "status": st.column_config.SelectboxColumn("Status", options=["Not Started", "In Progress", "Completed"], required=True),
                    "result": st.column_config.SelectboxColumn("Result", options=["Pending", "Pass", "Fail"], required=True),
                    "report_link": st.column_config.LinkColumn("Link to Report", help="Link to the final, approved test report in the eQMS.")
                },
                hide_index=True
            )
            
            if edited_df.to_dict('records') != tab_data:
                # Merge back with other types and save
                other_tests = [t for t in verification_data if t.get('test_type') not in test_types]
                updated_all_tests = other_tests + edited_df.to_dict('records')
                ssm.update_data(updated_all_tests, "design_verification", "tests")
                st.toast(f"{tab_name} studies updated!", icon="✅")
                st.rerun()

        # --- Render each tab ---
        with tabs[0]:
            render_av_tab("Precision", ["Repeatability", "Intermediate Precision", "Reproducibility"],
                          "Characterize the agreement between replicate measurements under various conditions (same run, different days, different operators, different labs). Key metrics include SD and %CV.")
        with tabs[1]:
            render_av_tab("Sensitivity", ["Limit of Blank (LoB)", "Limit of Detection (LoD)", "Limit of Quantitation (LoQ)"],
                          "Determine the lowest measure of the analyte (e.g., tumor fraction) that can be reliably detected and/or quantified.")
        with tabs[2]:
            render_av_tab("Specificity", ["Interference", "Cross-Reactivity"],
                          "Demonstrate that the assay result is not affected by interfering substances (e.g., hemoglobin, bilirubin) or by cross-reacting analytes from non-cancerous conditions.")
        with tabs[3]:
            render_av_tab("Robustness", ["Robustness", "Sample Stability", "Reagent Stability"],
                          "Assess the assay's performance when small, deliberate changes are made to method parameters, and determine the stability of samples and reagents over time and temperature.")
        with tabs[4]:
            render_av_tab("Software", ["SW Verification", "System Verification"],
                          "Verify the performance of the bioinformatics pipeline and other software components against their requirements (Ref: ISO 62304), and verify the integrated system performance.")

    except Exception as e:
        st.error("An error occurred while displaying the Design Verification section. The data may be malformed.")
        logger.error(f"Failed to render design verification: {e}", exc_info=True)
