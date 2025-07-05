# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Renders the centralized Compliance Action & CAPA Management Tool.

This module provides the logic for aggregating all compliance-related actions
from across the DHF (e.g., from Design Reviews, Audits, Complaints, CAPAs)
into a single, interactive, and risk-prioritized view. This tool is designed
to meet the requirements of 21 CFR 820.100 (CAPA) and ISO 13485.
"""

# --- Standard Library Imports ---
import logging
from datetime import datetime
from typing import Any, Dict, List

# --- Third-party Imports ---
import pandas as pd
import plotly.express as px
import streamlit as st

# --- Local Application Imports ---
from ..utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)

def render_action_item_tracker(ssm: SessionStateManager) -> None:
    """
    Aggregates, analyzes, and displays all compliance actions from the QMS.

    This function fetches actions from various sources (Design Reviews, Changes,
    CAPAs, Audits), enriches the data (e.g., identifies overdue items),
    displays risk-based KPIs, and provides an interactive, filterable table for
    project-wide oversight and audit readiness.

    Args:
        ssm (SessionStateManager): The session state manager to access DHF data.
    """
    st.header("🗃️ Centralized Compliance Action & CAPA Tracker")
    st.markdown("This table consolidates all action items from Design Reviews, Change Controls, CAPAs, and Audits for project-wide oversight and compliance management.")
    st.info("This is a single source of truth for all required compliance actions. Prioritize items based on their risk level.", icon="🎯")

    try:
        # --- 1. Aggregate actions from all potential QMS sources ---
        all_actions: List[Dict[str, Any]] = []

        # Source 1: Design Reviews
        reviews = ssm.get_data("design_reviews", "reviews")
        for i, review in enumerate(reviews):
            if not isinstance(review, dict): continue
            for action in review.get("action_items", []):
                if not isinstance(action, dict): continue
                action_copy = action.copy()
                action_copy['source'] = f"Review: {review.get('type', f'Review on {review.get('date')}')}"
                action_copy['type'] = action.get('type', 'Action Item') # Default type
                all_actions.append(action_copy)

        # Source 2: Design Changes
        changes = ssm.get_data("design_changes", "changes")
        for change in changes:
            if not isinstance(change, dict): continue
            for action in change.get("action_items", []):
                if not isinstance(action, dict): continue
                action_copy = action.copy()
                action_copy['source'] = f"DCR-{change.get('id', 'N/A')}"
                action_copy['type'] = action.get('type', 'Action Item')
                all_actions.append(action_copy)

        # Source 3: CAPA Records (Ref: 21 CFR 820.100)
        capas = ssm.get_data("quality_system", "capa_records")
        for capa in capas:
             if not isinstance(capa, dict): continue
             for action in capa.get("action_plan", []):
                if not isinstance(action, dict): continue
                action_copy = action.copy()
                action_copy['source'] = f"CAPA-{capa.get('id', 'N/A')}"
                action_copy['type'] = 'CAPA Action' # Overwrite type for clarity
                all_actions.append(action_copy)

        logger.info(f"Aggregated a total of {len(all_actions)} action items from all sources.")

        if not all_actions:
            st.success("🎉 No compliance actions have been recorded in the QMS yet.")
            return

        actions_df = pd.DataFrame(all_actions)

        # --- 2. Enrich the data for better insights ---
        required_cols = ['id', 'description', 'owner', 'due_date', 'status', 'source', 'risk_priority', 'type', 'voe_status']
        for col in required_cols:
            if col not in actions_df.columns:
                actions_df[col] = None # Fill with None to ensure schema consistency

        actions_df['due_date'] = pd.to_datetime(actions_df['due_date'], errors='coerce')
        now = pd.to_datetime(datetime.now().date())
        is_overdue = (actions_df['due_date'].notna()) & (actions_df['due_date'] < now) & (actions_df['status'] != 'Completed')
        actions_df.loc[is_overdue, 'status'] = 'Overdue'

        # --- 3. Create insightful, risk-based KPIs ---
        total_items = len(actions_df)
        completed_items = len(actions_df[actions_df['status'] == 'Completed'])
        open_items = total_items - completed_items
        overdue_count = len(actions_df[actions_df['status'] == 'Overdue'])
        high_risk_open_count = len(actions_df[(actions_df['risk_priority'] == 'High') & (actions_df['status'] != 'Completed')])

        st.subheader("Compliance Health KPIs")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Total Open Actions", open_items)
        kpi_cols[1].metric("Overdue Actions", overdue_count)
        kpi_cols[2].metric("High-Risk Open Actions", high_risk_open_count, help="Open actions with 'High' risk priority. These require immediate attention.")
        kpi_cols[3].metric("VoE Pending", len(actions_df[actions_df['voe_status'] == 'Pending']), help="Completed CAPA actions awaiting Verification of Effectiveness.")

        # --- 4. Add advanced visualization ---
        st.subheader("Open Actions by Status and Risk")
        if open_items > 0:
            open_df = actions_df[actions_df['status'] != 'Completed'].copy()
            risk_status_summary = open_df.groupby(['status', 'risk_priority']).size().reset_index(name='count')
            status_order = ["Overdue", "In Progress", "Open"]
            risk_order = ["High", "Medium", "Low"]

            fig = px.bar(
                risk_status_summary,
                x="status",
                y="count",
                color="risk_priority",
                title="Breakdown of Open Compliance Actions",
                labels={'count': 'Number of Actions', 'status': 'Status', 'risk_priority': 'Risk Priority'},
                category_orders={"status": status_order, "risk_priority": risk_order},
                color_discrete_map={'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#1f77b4'},
                text_auto=True
            )
            fig.update_layout(barmode='stack', legend_title_text='Risk Priority')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No open actions to visualize.")

        # --- 5. Create interactive filters ---
        st.divider()
        st.subheader("Filter and Export Compliance Actions")
        filter_cols = st.columns(3)
        status_options = sorted(actions_df['status'].unique())
        owner_options = sorted(actions_df['owner'].dropna().unique())
        type_options = sorted(actions_df['type'].dropna().unique())
        default_status = [s for s in status_options if s != 'Completed']

        status_filter = filter_cols[0].multiselect("Filter by Status:", options=status_options, default=default_status)
        owner_filter = filter_cols[1].multiselect("Filter by Owner:", options=owner_options)
        type_filter = filter_cols[2].multiselect("Filter by Type:", options=type_options)

        filtered_df = actions_df.copy()
        if status_filter:
            filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
        if owner_filter:
            filtered_df = filtered_df[filtered_df['owner'].isin(owner_filter)]
        if type_filter:
            filtered_df = filtered_df[filtered_df['type'].isin(type_filter)]

        # --- 6. Display the styled, filtered DataFrame ---
        def style_row_by_risk(row: pd.Series) -> List[str]:
            """Applies a background color based on risk priority."""
            color = ''
            if row.risk_priority == 'High':
                color = 'background-color: #ffcccc'
            elif row.risk_priority == 'Medium':
                color = 'background-color: #fff0cc'
            return [color for _ in row]

        st.dataframe(
            filtered_df[['id', 'description', 'owner', 'due_date', 'status', 'risk_priority', 'type', 'source', 'voe_status']].style.apply(style_row_by_risk, axis=1),
            use_container_width=True,
            column_config={
                "id": "Action ID",
                "description": st.column_config.TextColumn("Description", width="large"),
                "owner": "Owner",
                "due_date": st.column_config.DateColumn("Due Date", format="YYYY-MM-DD"),
                "status": "Status",
                "risk_priority": st.column_config.TextColumn("Risk", help="Risk priority of the action item."),
                "type": st.column_config.TextColumn("Type", help="Category of the action (e.g., CAPA Action, Audit Finding)."),
                "source": st.column_config.TextColumn("Source", help="The QMS record this action originated from."),
                "voe_status": st.column_config.TextColumn("VoE Status", help="Verification of Effectiveness Status for CAPA actions.")
            },
            hide_index=True
        )

        # --- 7. Add an Export Button ---
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered View to CSV",
            data=csv,
            file_name="compliance_actions_export.csv",
            mime="text/csv",
            key="export_compliance_actions"
        )

    except Exception as e:
        st.error("An error occurred while generating the action item tracker. The data may be incomplete or malformed.")
        logger.error(f"Failed to render action item tracker: {e}", exc_info=True)
