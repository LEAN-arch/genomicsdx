# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Module for rendering the Design Reviews section of the DHF Explorer.

This component provides a structured, formal interface to document and manage
records from phase-gate, technical, and other formal design reviews, as
required by 21 CFR 820.30(e).
"""

# --- Standard Library Imports ---
import logging
from typing import Dict, List, Any

# --- Third-party Imports ---
import pandas as pd
import streamlit as st

# --- Local Application Imports ---
from ..utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def render_design_reviews(ssm: SessionStateManager) -> None:
    """
    Renders an editable view of all formal design review records.
    
    This function displays each design review in a separate, structured container,
    allowing for focused viewing and editing of its purpose, scope, attendees,
    outcome, and associated action items. It enforces the documentation of
    key compliance elements like the presence of an independent reviewer.
    """
    st.header("6. Design Reviews")
    st.markdown("""
    *As per 21 CFR 820.30(e) and ISO 13485:2016, Section 7.3.5.*

    Document and track formal reviews of the design at appropriate, planned stages of the project lifecycle. Each review must include an individual who does not have direct responsibility for the design stage being reviewed, as well as any specialists needed. The results, including the design review date, attendees, and any action items, must be documented in the DHF.
    """)
    st.info("Select a review from the dropdown to view its details, or log a new review record.", icon="ℹ️")
    st.divider()

    try:
        # --- 1. Load Data and Prepare Dependencies ---
        reviews = ssm.get_data("design_reviews", "reviews")
        team_members_data = ssm.get_data("design_plan", "team_members")
        owner_options = sorted([member.get('name') for member in team_members_data if member.get('name')])
        
        # Sort reviews by date, most recent first
        reviews.sort(key=lambda r: pd.to_datetime(r.get('date', '1900-01-01')), reverse=True)

        # --- 2. High-Level Summary & Selection ---
        if not reviews:
            st.warning("No design review records found.")
        else:
            review_options = {f"{pd.to_datetime(r.get('date')).strftime('%Y-%m-%d')}: {r.get('type', 'Review')} ({r.get('id')})": r.get('id') for r in reviews}
            selected_review_display = st.selectbox(
                "**Select a Design Review to View/Edit**",
                options=review_options.keys(),
                index=0
            )
            selected_review_id = review_options[selected_review_display]
            # Use an index to modify the list in place later
            selected_index = next((i for i, r in enumerate(reviews) if r.get('id') == selected_review_id), None)
            if selected_index is None:
                st.error("Selected review not found. Please refresh.")
                return
            review = reviews[selected_index]
        
        # Button to start a new review log
        if st.button("Log New Design Review", use_container_width=True):
            # Create a new, unique ID and default structure, then rerun
            new_id = f"DR-{len(reviews) + 1:03d}"
            new_review = {
                "id": new_id, "date": str(pd.Timestamp.now().date()), "type": "Technical", "phase": "N/A",
                "attendees": [], "independent_reviewer": "", "scope": "", "outcome": "Pending", "notes": "", "action_items": []
            }
            reviews.append(new_review)
            ssm.update_data(reviews, "design_reviews", "reviews")
            st.success(f"Created new draft review {new_id}. Please fill in the details below.")
            st.rerun()

        if not reviews or 'review' not in locals(): 
            return # Exit if still no reviews after button check

        # --- 3. Detailed View of Selected/New Review ---
        with st.form(key=f"review_form_{review['id']}"):
            st.subheader(f"Design Review Record: {review.get('id')}")

            # --- Review Metadata ---
            cols = st.columns(3)
            date_val = cols[0].date_input("**Date**", value=pd.to_datetime(review.get('date')))
            type_val = cols[1].selectbox("**Review Type**", options=["Phase-Gate", "Technical", "Risk", "Software", "Usability"], index=["Phase-Gate", "Technical", "Risk", "Software", "Usability"].index(review.get('type', 'Technical')))
            phase_val = cols[2].text_input("**Project Phase Reviewed**", value=review.get('phase', ''), help="E.g., 'Assay Freeze', 'Algorithm Lock'")
            
            scope_val = st.text_area("**Scope & Purpose of Review**", value=review.get('scope', ''), height=100, help="What specific design elements, documents, and data were under review?")

            # --- Attendees & Outcome ---
            st.markdown("**Attendees & Review Outcome**")
            att_cols = st.columns(3)
            attendees_val = att_cols[0].multiselect("**Attendees**", options=owner_options, default=review.get('attendees', []))
            
            current_reviewer = review.get('independent_reviewer', '')
            reviewer_index = owner_options.index(current_reviewer) if current_reviewer in owner_options else 0
            independent_reviewer_val = att_cols[1].selectbox("**Independent Reviewer (Required)**", options=owner_options, index=reviewer_index, help="Individual without direct responsibility for this design stage.")
            
            outcome_val = review.get('outcome', 'Pending')
            outcome_options = ["Go", "Go with Conditions", "No-Go", "Pending"]
            outcome_val = att_cols[2].selectbox("**Formal Outcome**", options=outcome_options, index=outcome_options.index(outcome_val))
            
            outcome_color_map = {"Go": "lightgreen", "Go with Conditions": "lightyellow", "No-Go": "lightcoral", "Pending": "lightgray"}
            st.markdown(f"<div style='background-color:{outcome_color_map[outcome_val]}; padding: 10px; border-radius: 5px; text-align: center;'><b>Outcome: {outcome_val}</b></div>", unsafe_allow_html=True)
            
            notes_val = st.text_area("**Summary & Notes**", value=review.get('notes', ''), height=150, help="Summarize the discussion, key decisions, and rationale for the outcome.")

            # --- Action Items Editor ---
            st.markdown("**Action Items**")
            action_items_df = pd.DataFrame(review.get("action_items", []))

            edited_actions_df = st.data_editor(
                action_items_df,
                num_rows="dynamic",
                use_container_width=True,
                key=f"action_item_editor_{review['id']}",
                column_config={
                    "id": st.column_config.TextColumn("ID", required=True, help="Unique ID, e.g., AI-DR-001-01"),
                    "description": st.column_config.TextColumn("Description", width="large", required=True),
                    "owner": st.column_config.SelectboxColumn("Owner", options=owner_options, required=True),
                    "due_date": st.column_config.DateColumn("Due Date", format="YYYY-MM-DD", required=True),
                    "status": st.column_config.SelectboxColumn("Status", options=["Open", "In Progress", "Completed", "Overdue"], required=True),
                    "risk_priority": st.column_config.SelectboxColumn("Priority", options=["High", "Medium", "Low"], default="Medium", required=True)
                },
                hide_index=True,
            )
            
            # --- Save Button ---
            submitted = st.form_submit_button(f"Save Changes to Review {review['id']}", use_container_width=True, type="primary")
            if submitted:
                # Convert back to storable format
                edited_actions_list = edited_actions_df.to_dict('records')
                for item in edited_actions_list:
                    item['due_date'] = str(pd.to_datetime(item['due_date']).date()) if pd.notna(item['due_date']) else None
                
                # Update the specific review in the list
                reviews[selected_index] = {
                    "id": review['id'],
                    "date": str(date_val),
                    "type": type_val,
                    "phase": phase_val,
                    "scope": scope_val,
                    "attendees": attendees_val,
                    "independent_reviewer": independent_reviewer_val,
                    "outcome": outcome_val,
                    "notes": notes_val,
                    "action_items": edited_actions_list
                }
                
                ssm.update_data(reviews, "design_reviews", "reviews")
                st.toast(f"Design Review {review['id']} updated!", icon="✅")
                st.rerun()

    except Exception as e:
        st.error("An error occurred while displaying the Design Reviews section. The data may be malformed.")
        logger.error(f"Failed to render design reviews: {e}", exc_info=True)
