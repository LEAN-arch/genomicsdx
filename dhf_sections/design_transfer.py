# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Module for rendering the Assay Transfer & Lab Operations Readiness section.

This component provides a structured dashboard to manage and document activities
related to transferring the validated assay and software from R&D into a
CLIA/CAP-certified clinical laboratory environment for commercial operation, as
required by 21 CFR 820.30(h) and 21 CFR 820.170.
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


def render_design_transfer(ssm: SessionStateManager) -> None:
    """
    Renders a comprehensive dashboard for Assay Transfer & Lab Operations Readiness.
    """
    st.header("9. Assay Transfer & Lab Operations Readiness")
    st.markdown("""
    *As per 21 CFR 820.30(h) Design Transfer & 21 CFR 820.170 Production and Process Controls.*

    This section documents the activities that ensure the validated diagnostic service design is correctly translated into production procedures and specifications for the clinical laboratory. This is the critical bridge between R&D and a scalable, compliant commercial testing service under **CLIA** and **ISO 15189**.
    """)
    st.info("This dashboard provides a real-time view of operational readiness for commercial launch.", icon="🚀")
    st.divider()

    try:
        # --- 1. Load Data ---
        transfer_data: Dict[str, Any] = ssm.get_data("lab_operations")
        logger.info("Loaded lab operations and design transfer data.")

        # --- 2. High-Level Readiness KPIs ---
        st.subheader("Launch Readiness Dashboard")
        
        sop_data = transfer_data.get("sops", [])
        infra_data = transfer_data.get("infrastructure", [])
        ppq_data = transfer_data.get("ppq_runs", [])

        sop_total = len(sop_data)
        sop_approved = len([s for s in sop_data if s.get('status') == 'Approved'])
        sop_progress = (sop_approved / sop_total) * 100 if sop_total > 0 else 0

        infra_total = len(infra_data)
        infra_qualified = len([i for i in infra_data if i.get('status') == 'PQ Complete'])
        infra_progress = (infra_qualified / infra_total) * 100 if infra_total > 0 else 0
        
        ppq_total = len(ppq_data)
        ppq_passed = len([p for p in ppq_data if p.get('result') == 'Pass'])
        ppq_progress = (ppq_passed / ppq_total) * 100 if ppq_total > 0 else 0

        kpi_cols = st.columns(3)
        with kpi_cols[0]:
            st.metric("SOP Readiness", f"{sop_progress:.0f}%")
            st.progress(sop_progress / 100)
        with kpi_cols[1]:
            st.metric("Infrastructure Qualification", f"{infra_progress:.0f}%")
            st.progress(infra_progress / 100)
        with kpi_cols[2]:
            st.metric("PPQ Run Pass Rate", f"{ppq_progress:.0f}%")
            st.progress(ppq_progress / 100)

        st.divider()

        # --- 3. Define Tabs for Each Transfer Stream ---
        tab_titles = [
            "1. Process & SOP Transfer",
            "2. Lab Infrastructure & LIMS Validation",
            "3. Bioinformatics & Software Deployment",
            "4. Process Performance Qualification (PPQ)"
        ]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        # --- Helper Function for Rendering Tables ---
        def render_editor_tab(df_key: str, column_config: Dict, help_text: str):
            st.caption(help_text)
            
            data = transfer_data.get(df_key, [])
            df = pd.DataFrame(data)
            
            # Ensure date columns are in the correct format for the editor
            for col, config in column_config.items():
                if isinstance(config, st.column_config.DateColumn) and col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                key=f"transfer_editor_{df_key}",
                column_config=column_config,
                hide_index=True
            )
            
            if edited_df.to_dict('records') != df.to_dict('records'):
                # Convert date columns back to string format for JSON serialization
                df_to_save = edited_df.copy()
                for col, config in column_config.items():
                    if isinstance(config, st.column_config.DateColumn) and col in df_to_save.columns:
                         df_to_save[col] = df_to_save[col].dt.strftime('%Y-%m-%d').replace({pd.NaT: None})

                ssm.update_data(df_to_save.to_dict('records'), "lab_operations", df_key)
                st.toast(f"{df_key.replace('_', ' ').title()} updated!", icon="✅")
                st.rerun()

        # --- Tab 1: SOPs and Training ---
        with tab1:
            st.subheader("Standard Operating Procedures (SOPs) & Personnel Training")
            render_editor_tab(
                "sops",
                {
                    "doc_id": st.column_config.TextColumn("SOP ID", required=True),
                    "title": st.column_config.TextColumn("SOP Title", width="large", required=True),
                    "version": st.column_config.TextColumn("Version", required=True),
                    "status": st.column_config.SelectboxColumn("Status", options=["Draft", "In Review", "Approved"], required=True),
                    "training_records_link": st.column_config.LinkColumn("Training Records", help="Link to training completion records for lab personnel.")
                },
                "Track the transfer of R&D processes into formal, version-controlled SOPs for the clinical lab and the training of lab technologists."
            )
            
        # --- Tab 2: Infrastructure & LIMS ---
        with tab2:
            st.subheader("Laboratory Equipment & LIMS Qualification")
            render_editor_tab(
                "infrastructure",
                {
                    "asset_id": st.column_config.TextColumn("Asset ID", required=True),
                    "equipment_type": st.column_config.TextColumn("Equipment/System", required=True),
                    "status": st.column_config.SelectboxColumn("Qualification Status", options=["Pending", "IQ Complete", "OQ Complete", "PQ Complete"], required=True),
                    "qualification_report_link": st.column_config.LinkColumn("IQ/OQ/PQ Report", help="Link to final qualification report.")
                },
                "Document the Installation, Operational, and Performance Qualification (IQ/OQ/PQ) of all critical laboratory instruments and the validation of the Laboratory Information Management System (LIMS)."
            )

        # --- Tab 3: Software Deployment ---
        with tab3:
            st.subheader("Bioinformatics Pipeline & Classifier Deployment (ISO 62304)")
            render_editor_tab(
                "software_deployment",
                {
                    "component": st.column_config.TextColumn("Software Component", required=True),
                    "version": st.column_config.TextColumn("Deployed Version/Hash", required=True),
                    "deployment_date": st.column_config.DateColumn("Deployment Date", format="YYYY-MM-DD", required=True),
                    "validation_protocol": st.column_config.TextColumn("Validation Protocol ID"),
                    "validation_report_link": st.column_config.LinkColumn("Validation Report", help="Link to report verifying successful deployment and performance in the production environment.")
                },
                "Track the formal, controlled deployment of the locked bioinformatics pipeline and classifier algorithm from the development environment to the validated production infrastructure."
            )

        # --- Tab 4: Process Performance Qualification (PPQ) ---
        with tab4:
            st.subheader("Process Performance Qualification (PPQ) Runs")
            render_editor_tab(
                "ppq_runs",
                {
                    "run_id": st.column_config.TextColumn("PPQ Run ID", required=True),
                    "description": st.column_config.TextColumn("Run Description", width="large", help="E.g., '3 non-consecutive runs by 2 operators'", required=True),
                    "run_date": st.column_config.DateColumn("Run Date", format="YYYY-MM-DD"),
                    "result": st.column_config.SelectboxColumn("Result", options=["Not Started", "In Progress", "Pass", "Pass with Discrepancies", "Fail"], required=True),
                    "summary_report_link": st.column_config.LinkColumn("Summary Report", help="Link to the final PPQ summary report submitted to regulatory bodies.")
                },
                "Document the capstone PPQ runs. These end-to-end runs demonstrate that the entire transferred process is robust, reproducible, and consistently meets all performance specifications at scale."
            )

    except Exception as e:
        st.error("An error occurred while displaying the Design Transfer section. The data may be malformed.")
        logger.error(f"Failed to render design transfer: {e}", exc_info=True)
