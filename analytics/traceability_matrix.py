# --- SME OVERHAUL: Definitive, Consolidated, and Unabridged Final Version ---
"""
Module for rendering the Multi-Level DHF Traceability Command Center.

This module provides the critical, multi-faceted traceability views required
for a PMA-class genomic diagnostic. It demonstrates linkage from clinical needs
down to analytical and software verification, in alignment with 21 CFR 820.30,
ISO 13485, ISO 14971, and ISO 62304. This is the definitive traceability
module for the project.
"""

# --- Standard Library Imports ---
import logging
from typing import Dict, List, Tuple

# --- Third-party Imports ---
import pandas as pd
import streamlit as st

# --- Local Application Imports ---
from ..utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)

# --- Styling and Configuration ---
def style_trace_cell(val: str, **kwargs) -> str:
    """Applies CSS styling to a cell based on its status icon."""
    val_str = str(val)
    color = "inherit"
    bg_color = "#f8f9fa"  # Default light grey for empty cells
    font_weight = "normal"
    
    if '✅' in val_str:
        bg_color = 'rgba(44, 160, 44, 0.15)'  # Green for Pass
        color = '#1E4620'
        font_weight = "bold"
    elif '❌' in val_str:
        bg_color = 'rgba(214, 39, 40, 0.2)'  # Red for Fail
        color = '#5A0001'
        font_weight = "bold"
    elif '⏳' in val_str:
        bg_color = 'rgba(255, 127, 14, 0.2)' # Orange for In Progress
        color = '#663000'
        font_weight = "bold"
    elif '🔵' in val_str:
        bg_color = 'rgba(31, 119, 180, 0.2)' # Blue for Risk Control
        color = '#003366'
        font_weight = "bold"
    elif 'N/A' in val_str:
        color = '#6c757d'

    return f'background-color: {bg_color}; color: {color}; font-weight: {font_weight}; text-align: center;'

@st.cache_data
def generate_all_trace_data(
    requirements: List[Dict],
    verifications: List[Dict],
    validations: List[Dict]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pre-processes and merges all data for all traceability matrices.
    This expensive operation is cached for performance.
    """
    if not requirements:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    reqs_df = pd.DataFrame(requirements)
    ver_df = pd.DataFrame(verifications) if verifications else pd.DataFrame(columns=['id', 'result', 'input_verified_id'])
    val_df = pd.DataFrame(validations) if validations else pd.DataFrame(columns=['id', 'status', 'user_need_validated'])

    # --- 1. Clinical Traceability Data (User Needs -> Validation) ---
    user_needs_df = reqs_df[reqs_df['type'] == 'User Need'].copy()
    clinical_matrix = user_needs_df[['id', 'description']].set_index(['id', 'description'])
    if not val_df.empty and 'user_need_validated' in val_df.columns:
        val_status_map = val_df.set_index('user_need_validated')['status'].to_dict()
        val_id_map = val_df.set_index('user_need_validated')['id'].to_dict()
        
        status_icon_map = {'Pass': '✅', 'Fail': '❌', 'In Progress': '⏳', 'Not Started': '⏳'}
        
        # Map IDs and create trace string
        clinical_matrix['Link'] = clinical_matrix.index.get_level_values('id').map(val_id_map)
        clinical_matrix['Status'] = clinical_matrix.index.get_level_values('id').map(val_status_map)
        clinical_matrix['Trace'] = clinical_matrix['Status'].map(status_icon_map) + ' ' + clinical_matrix['Link'].fillna('')
        clinical_matrix['Trace'] = clinical_matrix['Trace'].fillna('❌')
    else:
        clinical_matrix['Trace'] = '❌'
    
    # --- Helper Function for Wide Matrices ---
    def create_wide_matrix(req_subset_df, ver_df):
        if ver_df.empty or req_subset_df.empty or 'input_verified_id' not in ver_df.columns:
            return pd.DataFrame(index=pd.MultiIndex.from_frame(req_subset_df[['id', 'description', 'is_risk_control']]))

        status_icon_map = {'Pass': '✅', 'Fail': '❌', 'In Progress': '⏳', 'Not Started': '⏳'}
        ver_df['status_icon'] = ver_df['result'].map(status_icon_map).fillna('⏳')
        ver_df['trace_link'] = ver_df['status_icon'] + ' ' + ver_df['id']

        merged = pd.merge(
            req_subset_df[['id', 'description', 'is_risk_control']],
            ver_df[['input_verified_id', 'id', 'trace_link']],
            left_on='id',
            right_on='input_verified_id',
            how='left'
        )
        
        if merged['id_y'].notna().any():
            matrix = merged.pivot_table(
                index=['id_x', 'description', 'is_risk_control'],
                columns='id_y',
                values='trace_link',
                aggfunc='first'
            )
        else:
            matrix = pd.DataFrame(index=pd.MultiIndex.from_frame(req_subset_df[['id', 'description', 'is_risk_control']]))

        matrix = matrix.reindex(pd.MultiIndex.from_frame(req_subset_df[['id', 'description', 'is_risk_control']]))
        matrix.index.names = ['id', 'description', 'is_risk_control']
        return matrix

    # --- 2. Analytical Traceability Data ---
    assay_reqs_df = reqs_df[reqs_df['type'].isin(['Assay', 'System'])].copy()
    analytical_matrix = create_wide_matrix(assay_reqs_df, ver_df)

    # --- 3. Software Traceability Data ---
    sw_reqs_df = reqs_df[reqs_df['type'] == 'Software'].copy()
    sw_matrix = create_wide_matrix(sw_reqs_df, ver_df)

    return clinical_matrix, analytical_matrix, sw_matrix


def render_traceability_matrix(ssm: SessionStateManager):
    """
    Renders the multi-level traceability command center.
    """
    st.header("🔬 Multi-Level Traceability Command Center")
    st.markdown("This provides end-to-end, status-driven traceability across the entire DHF, a critical component for PMA submission and audit readiness.")

    legend_cols = st.columns(4)
    legend_cols[0].markdown("<span style='color: #1E4620;'>✅ **Pass**</span>", unsafe_allow_html=True)
    legend_cols[1].markdown("<span style='color: #5A0001;'>❌ **Fail / Untraced**</span>", unsafe_allow_html=True)
    legend_cols[2].markdown("<span style='color: #663000;'>⏳ **In Progress**</span>", unsafe_allow_html=True)
    legend_cols[3].markdown("<span style='color: #003366;'>🔵 **Risk Control Verified**</span>", unsafe_allow_html=True)
    st.divider()

    try:
        clinical_matrix, analytical_matrix, sw_matrix = generate_all_trace_data(
            ssm.get_data("design_inputs", "requirements"),
            ssm.get_data("design_verification", "tests"),
            ssm.get_data("clinical_study", "hf_studies") # Using hf_studies as stand-in for all validation
        )

        tab1, tab2, tab3 = st.tabs([
            "**1. Clinical Traceability (Needs → Validation)**",
            "**2. Analytical Traceability (Requirements → AV)**",
            "**3. Software Traceability (Requirements → SW V&V)**"
        ])

        with tab1:
            st.subheader("Clinical Needs vs. Clinical Validation Studies")
            st.caption("Answers: 'Are we building the right test for patients?' This matrix links each high-level Clinical Need to the validation study that proves it is met.")

            if not clinical_matrix.empty:
                total_needs = len(clinical_matrix)
                traced_needs = clinical_matrix[~clinical_matrix['Trace'].str.contains("❌", na=False)].shape[0]
                passing_needs = clinical_matrix[clinical_matrix['Trace'].str.contains("✅", na=False)].shape[0]

                kpi1, kpi2 = st.columns(2)
                kpi1.metric("Traceability Coverage", f"{traced_needs / total_needs:.1%}" if total_needs else "0.0%")
                kpi2.metric("Passing Coverage", f"{passing_needs / traced_needs:.1%}" if traced_needs else "0.0%", help="Of the needs with a trace, what percentage are passing?")

                st.dataframe(
                    clinical_matrix.reset_index()[['id', 'description', 'Trace']].style.apply(lambda s: s.map(lambda x: style_trace_cell(x)), subset=['Trace']),
                    use_container_width=True,
                    column_config={ "id": "Need ID", "description": "Clinical Need Description", "Trace": "Validation Study (Status & ID)" }
                )
            else:
                st.warning("No 'User Need' type requirements found to build this matrix.")

        def render_wide_matrix(title: str, caption: str, matrix_df: pd.DataFrame):
            st.subheader(title)
            st.caption(caption)
            if not matrix_df.empty:
                total_reqs = len(matrix_df)
                
                # A requirement is traced if any cell in its row is not empty/NaN
                traced_reqs = matrix_df.notna().any(axis=1).sum()
                
                # A requirement is passing if it has at least one 'Pass' and zero 'Fail' links
                has_pass = matrix_df.apply(lambda row: row.str.contains('✅', na=False).any(), axis=1)
                has_fail = matrix_df.apply(lambda row: row.str.contains('❌', na=False).any(), axis=1)
                passing_reqs = (has_pass & ~has_fail).sum()

                kpi1, kpi2 = st.columns(2)
                kpi1.metric("Traceability Coverage", f"{traced_reqs / total_reqs:.1%}" if total_reqs else "0.0%")
                kpi2.metric("Passing Coverage", f"{passing_reqs / traced_reqs:.1%}" if traced_reqs else "0.0%", help="Of traced requirements, what % have at least one passing test and no failing tests?")

                display_df = matrix_df.reset_index()
                
                def style_risk_control_row(row):
                    is_risk_control = row.get('is_risk_control', False)
                    base_style = 'color: #003366; font-weight: bold;'
                    return [base_style if is_risk_control else '' for _ in row]
                
                # Apply styling
                styler = display_df.style.apply(style_risk_control_row, subset=['id', 'description'], axis=1)
                
                trace_cols = [col for col in display_df.columns if col not in ['id', 'description', 'is_risk_control']]
                styler = styler.map(style_trace_cell, subset=trace_cols)

                st.dataframe(styler, use_container_width=True, hide_index=True)
            else:
                st.warning("No relevant requirements or verification tests found to build this matrix.")

        with tab2:
            render_wide_matrix(
                "Assay & System Requirements vs. Analytical Validation Protocols",
                "Answers: 'Did we build the assay correctly?' This matrix links each performance requirement to the AV protocol that verifies it.",
                analytical_matrix
            )

        with tab3:
            render_wide_matrix(
                "Software Requirements vs. Software V&V Tests",
                "Answers: 'Did we build the software correctly?' This matrix links each software requirement to the test proving its implementation (Ref: ISO 62304).",
                sw_matrix
            )

    except Exception as e:
        st.error("An error occurred while generating the traceability matrix. The data may be incomplete or malformed.")
        logger.error(f"Failed to render traceability matrix: {e}", exc_info=True)
