# genomicsdx/dhf_sections/design_risk_management.py 
#--- SME OVERHAUL: Definitive, Compliance-Focused Version ---
# --- SME OVERHAUL: Definitive, Compliance-Focused Version ---
"""
Renders the Risk Management File (RMF) section of the DHF dashboard.

This module provides a comprehensive, multi-tabbed UI for documenting the
entire risk management process according to ISO 14971. It includes planning,
hazard analysis, FMEAs, and overall risk-benefit analysis, all tailored for
a genomic diagnostic service. It features powerful analytics including live
risk matrices, Sankey diagrams, and traceability gap analysis.
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List

# --- Third-party Imports ---
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- Local Application Imports ---
from ..utils.session_state_manager import SessionStateManager
from ..utils.plot_utils import _RISK_CONFIG

# --- Setup Logging ---
logger = logging.getLogger(__name__)

def render_design_risk_management(ssm: SessionStateManager) -> None:
    """Renders the UI for the complete Risk Management File (RMF)."""
    st.header("2. Risk Management File (ISO 14971:2019)")
    st.markdown("This section constitutes the Risk Management File (RMF) for the diagnostic service. It documents the systematic application of management policies, procedures, and practices to the tasks of analyzing, evaluating, controlling, and monitoring risk associated with the device. The primary harms for this IVD are related to incorrect test results.")
    
    try:
        # --- 1. Load All Relevant Data ---
        rmf_data: Dict[str, Any] = ssm.get_data("risk_management_file") or {}
        v_and_v_data: List[Dict[str, Any]] = (ssm.get_data("design_verification", "tests") or []) + (ssm.get_data("clinical_study", "hf_studies") or [])
        vv_protocol_ids: List[str] = [""] + sorted([p.get('id', '') for p in v_and_v_data if p.get('id')])

        # --- 2. Risk Analytics Dashboard ---
        st.subheader("Risk Posture Analytics")
        with st.container(border=True):
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                # Sankey Diagram for Risk Reduction
                hazards_data = rmf_data.get("hazards", [])
                if not hazards_data:
                    st.caption("No hazard data to plot.")
                else:
                    df = pd.DataFrame(hazards_data)
                    risk_config = _RISK_CONFIG
                    get_level = lambda s, o: risk_config['levels'].get((s, o), 'High')
                    df['initial_level'] = df.apply(lambda x: get_level(x.get('initial_S'), x.get('initial_O')), axis=1)
                    df['final_level'] = df.apply(lambda x: get_level(x.get('final_S'), x.get('final_O')), axis=1)
                    all_nodes = [f"Initial {level}" for level in risk_config['order']] + [f"Residual {level}" for level in risk_config['order']]
                    node_map = {name: i for i, name in enumerate(all_nodes)}
                    node_colors = [risk_config['colors'][name.split(' ')[1]] for name in all_nodes]
                    links = df.groupby(['initial_level', 'final_level']).size().reset_index(name='count')
                    
                    sankey_fig = go.Figure(data=[go.Sankey(
                        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color=node_colors),
                        link=dict(
                            source=[node_map.get(f"Initial {row['initial_level']}") for _, row in links.iterrows()],
                            target=[node_map.get(f"Residual {row['final_level']}") for _, row in links.iterrows()],
                            value=links['count'],
                            color=[risk_config['colors'][row['final_level']] for _, row in links.iterrows()]
                        ))])
                    sankey_fig.update_layout(title_text="<b>Risk Reduction Flow</b>", font_size=10, height=350, margin=dict(l=20,r=20,t=40,b=20))
                    st.plotly_chart(sankey_fig, use_container_width=True)
            with viz_col2:
                # Risk Control Traceability Gap Analysis
                st.markdown("**Risk Control Traceability**", help="Highlights risk controls that are not yet linked to a V&V protocol.")
                all_fmea_risks = rmf_data.get("assay_fmea", []) + rmf_data.get("service_fmea", [])
                all_risk_controls = hazards_data + all_fmea_risks
                
                untraced_controls = [
                    rc for rc in all_risk_controls 
                    if rc.get('mitigation') or rc.get('risk_control_measure')
                    and not rc.get('verification_link')
                ]
                
                total_controls = len([rc for rc in all_risk_controls if rc.get('mitigation') or rc.get('risk_control_measure')])
                untraced_count = len(untraced_controls)
                coverage = ((total_controls - untraced_count) / total_controls) * 100 if total_controls > 0 else 100
                
                st.metric("V&V Coverage of Risk Controls", f"{coverage:.1f}%", delta=f"-{untraced_count} Untraced", delta_color="inverse")
                with st.expander("View Untraced Risk Controls"):
                    if untraced_controls:
                        df_untraced = pd.DataFrame(untraced_controls)
                        display_cols = [col for col in ['id', 'risk_control_measure', 'mitigation'] if col in df_untraced.columns]
                        st.dataframe(df_untraced[display_cols], hide_index=True, use_container_width=True)
                    else:
                        st.success("All risk controls are traced to a V&V activity.")

        # --- 3. RMF Section Tabs ---
        st.info("Use the tabs to navigate the Risk Management File. Changes are saved automatically.", icon="🗂️")
        # SME Enhancement: Add new tab for advanced risk analysis tools
        tab_titles = ["1. Risk Plan & Acceptability", "2. Hazard Analysis", "3. Assay FMEA", "4. Software & Service FMEA", "5. Risk Analysis Tools", "6. Overall Residual Risk"]
        tab1, tab2, tab3, tab4, tab_tools, tab6 = st.tabs(tab_titles)

        with tab1:
            st.subheader("Risk Management Plan Summary & Acceptability Criteria")
            with st.form("risk_plan_form"):
                plan_scope_val = st.text_area("**Risk Management Plan Scope**", value=rmf_data.get("plan_scope", ""), height=150)
                
                st.markdown("**Risk Acceptability Matrix**")
                st.caption("This matrix defines the policy for risk acceptability. It is the basis for evaluating all identified risks.")
                
                s_labels = ['1: Negligible', '2: Minor', '3: Serious', '4: Critical', '5: Catastrophic']
                o_labels = ['1: Improbable', '2: Remote', '3: Occasional', '4: Probable', '5: Frequent']
                matrix_text = [
                    ['Acceptable', 'Acceptable', 'Review', 'Review', 'Unacceptable'],
                    ['Acceptable', 'Acceptable', 'Review', 'Unacceptable', 'Unacceptable'],
                    ['Acceptable', 'Review', 'Unacceptable', 'Unacceptable', 'Unacceptable'],
                    ['Review', 'Unacceptable', 'Unacceptable', 'Unacceptable', 'Unacceptable'],
                    ['Unacceptable', 'Unacceptable', 'Unacceptable', 'Unacceptable', 'Unacceptable']
                ]
                color_map_numeric = {'Acceptable': 1, 'Review': 2, 'Unacceptable': 3}
                z_numeric = [[color_map_numeric[cell] for cell in row] for row in matrix_text]

                fig = go.Figure(data=go.Heatmap(
                    z=z_numeric, x=o_labels, y=s_labels,
                    text=matrix_text,
                    texttemplate="%{text}",
                    textfont={"size":12},
                    colorscale=[[0, 'rgba(44, 160, 44, 0.7)'], [0.5, 'rgba(255, 215, 0, 0.7)'], [1, 'rgba(214, 39, 40, 0.7)']],
                    showscale=False
                ))
                fig.update_layout(title_text="<b>Severity of Harm vs. Probability of Occurrence</b>", title_x=0.5, yaxis_autorange='reversed')
                st.plotly_chart(fig, use_container_width=True)
                
                if st.form_submit_button("Save Plan Scope"):
                    rmf_data["plan_scope"] = plan_scope_val
                    ssm.update_data(rmf_data, "risk_management_file")
                    st.toast("Risk Management Plan scope saved!", icon="✅")

        def render_risk_table(table_key: str, df: pd.DataFrame, column_config: dict):
            if df.empty:
                st.info(f"No data available for {table_key.replace('_', ' ').title()}. You can add new records below.")
                df = pd.DataFrame([{}]) 
            
            original_data = df.to_dict('records')
            edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"risk_editor_{table_key}", column_config=column_config, hide_index=True)
            
            if edited_df.to_dict('records') != original_data:
                cleaned_records = [record for record in edited_df.to_dict('records') if any(val not in [None, ''] for val in record.values())]
                rmf_data[table_key] = cleaned_records
                ssm.update_data(rmf_data, "risk_management_file")
                st.toast(f"{table_key.replace('_', ' ').title()} data updated!", icon="✅"); st.rerun()

        def render_fmea_risk_matrix_plot(fmea_data: List[Dict[str, Any]], title: str):
            st.markdown(f"**Interactive Risk Matrix: {title}**")
            if not fmea_data: st.warning(f"No {title} data available to plot."); return
            df = pd.DataFrame(fmea_data).dropna(subset=['S', 'O', 'D'])
            if df.empty: st.warning(f"No valid S, O, D data in {title} to plot."); return

            df['RPN'] = df['S'] * df['O'] * df['D']
            df['S_jitter'] = df['S'] + np.random.uniform(-0.15, 0.15, len(df))
            df['O_jitter'] = df['O'] + np.random.uniform(-0.15, 0.15, len(df))

            fig = go.Figure()
            # Add colored risk zones
            fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=5.5, y1=5.5, line=dict(width=0), fillcolor='rgba(44, 160, 44, 0.1)', layer='below')
            fig.add_shape(type="rect", x0=3.5, y0=3.5, x1=5.5, y1=5.5, line=dict(width=0), fillcolor='rgba(255, 127, 14, 0.15)', layer='below')
            fig.add_shape(type="rect", x0=4.5, y0=4.5, x1=5.5, y1=5.5, line=dict(width=0), fillcolor='rgba(214, 39, 40, 0.15)', layer='below')

            fig.add_trace(go.Scatter(
                x=df['S_jitter'], y=df['O_jitter'], mode='markers+text', text=df['id'], textposition='top center', textfont=dict(size=9),
                marker=dict(size=df['RPN'], sizemode='area', sizeref=2.*max(1, df['RPN'].max())/(40.**2), sizemin=4,
                            color=df['D'], colorscale='YlOrRd', colorbar=dict(title='Detection'), showscale=True),
                customdata=df[['failure_mode', 'potential_effect', 'S', 'O', 'D', 'RPN', 'mitigation']],
                hovertemplate="<b>%{customdata[0]}</b><br><b>Effect:</b> %{customdata[1]}<br><b>S:</b> %{customdata[2]} | <b>O:</b> %{customdata[3]} | <b>D:</b> %{customdata[4]} | <b>RPN: %{customdata[5]}</b><br><b>Mitigation:</b> %{customdata[6]}<extra></extra>"
            ))
            fig.update_layout(title=f"<b>{title} Risk Landscape</b>", xaxis_title="Severity (S)", yaxis_title="Occurrence (O)", height=500, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Hazard Analysis and Risk Evaluation (Top-Down)")
            render_risk_table("hazards", pd.DataFrame(rmf_data.get("hazards", [])), {
                "id": st.column_config.TextColumn("Hazard ID", required=True), "description": st.column_config.TextColumn("Hazard", width="medium"),
                "foreseeable_event": st.column_config.TextColumn("Foreseeable Event", width="large"), "potential_harm": st.column_config.TextColumn("Potential Harm", width="large"),
                "initial_S": st.column_config.NumberColumn("S (i)"), "initial_O": st.column_config.NumberColumn("O (i)"),
                "risk_control_measure": st.column_config.TextColumn("Risk Control Measure"), "final_S": st.column_config.NumberColumn("S (f)"),
                "final_O": st.column_config.NumberColumn("O (f)"), "verification_link": st.column_config.SelectboxColumn("V&V Link", options=vv_protocol_ids)
            })

        with tab3:
            st.subheader("Assay Failure Mode and Effects Analysis (aFMEA)")
            render_fmea_risk_matrix_plot(rmf_data.get("assay_fmea", []), "Assay FMEA")
            render_risk_table("assay_fmea", pd.DataFrame(rmf_data.get("assay_fmea", [])), {
                "id": st.column_config.TextColumn("aFMEA ID", required=True), "process_step": st.column_config.TextColumn("Step"), "failure_mode": st.column_config.TextColumn("Failure Mode", width="large"),
                "potential_effect": st.column_config.TextColumn("Effect", width="large"), "S": st.column_config.NumberColumn("S"), "O": st.column_config.NumberColumn("O"),
                "D": st.column_config.NumberColumn("D"), "mitigation": st.column_config.TextColumn("Mitigation"), "verification_link": st.column_config.SelectboxColumn("V&V Link", options=vv_protocol_ids)
            })

        with tab4:
            st.subheader("Software & Service Failure Mode and Effects Analysis (s/p-FMEA)")
            render_fmea_risk_matrix_plot(rmf_data.get("service_fmea", []), "Software & Service FMEA")
            render_risk_table("service_fmea", pd.DataFrame(rmf_data.get("service_fmea", [])), {
                "id": st.column_config.TextColumn("sFMEA ID", required=True), "process_step": st.column_config.TextColumn("Component"), "failure_mode": st.column_config.TextColumn("Failure Mode", width="large"),
                "potential_effect": st.column_config.TextColumn("Effect", width="large"), "S": st.column_config.NumberColumn("S"), "O": st.column_config.NumberColumn("O"),
                "D": st.column_config.NumberColumn("D"), "mitigation": st.column_config.TextColumn("Mitigation"), "verification_link": st.column_config.SelectboxColumn("V&V Link", options=vv_protocol_ids)
            })
            
        with tab_tools:
            st.subheader("Risk Analysis Tools")
            
            # --- RPN Significance Chart ---
            st.markdown("#### RPN Significance Chart")
            st.caption("This chart defines the action levels for Risk Priority Numbers (RPN) calculated in the FMEAs (where RPN = Severity × Occurrence × Detection). It provides a guideline for prioritizing risk mitigation activities.")

            rpn_data = {
                'RPN Range': ['100 - 125', '60 - 99', '20 - 59', '1 - 19'],
                'Risk Level': ['High', 'Medium', 'Low', 'Very Low'],
                'Recommended Action': [
                    'Unacceptable. Mitigation is required to reduce RPN. Design changes are mandatory.',
                    'Undesirable. Mitigation should be implemented. Justification required if no action is taken.',
                    'Acceptable with review. Risk is potentially acceptable, but should be reviewed for possible improvements.',
                    'Acceptable. Risk is broadly acceptable. No mitigation required.'
                ]
            }
            rpn_df = pd.DataFrame(rpn_data)
            
            def style_risk_level(val):
                color_map = {
                    'High': 'background-color: #ff9999',
                    'Medium': 'background-color: #ffe8a1',
                    'Low': 'background-color: #d4edda',
                    'Very Low': 'background-color: #e2e3e5'
                }
                return color_map.get(val, '')

            st.dataframe(rpn_df.style.applymap(style_risk_level, subset=['Risk Level']), hide_index=True, use_container_width=True)
            st.divider()

            # --- RPN Summary Table ---
            st.markdown("#### RPN Summary Table")
            st.caption("This table aggregates all failure modes from the FMEAs and ranks them by their Risk Priority Number (RPN) to help prioritize mitigation efforts.")
            assay_fmea_df = pd.DataFrame(rmf_data.get("assay_fmea", []))
            service_fmea_df = pd.DataFrame(rmf_data.get("service_fmea", []))
            
            if not assay_fmea_df.empty and not service_fmea_df.empty:
                assay_fmea_df['Source'] = 'Assay FMEA'
                service_fmea_df['Source'] = 'Software/Service FMEA'
                
                # Ensure RPN column exists and is numeric
                for df in [assay_fmea_df, service_fmea_df]:
                    if all(col in df.columns for col in ['S', 'O', 'D']):
                        df['RPN'] = pd.to_numeric(df['S']) * pd.to_numeric(df['O']) * pd.to_numeric(df['D'])
                    else:
                        df['RPN'] = 0

                rpn_df_concat = pd.concat([assay_fmea_df, service_fmea_df], ignore_index=True)
                rpn_df_concat = rpn_df_concat.sort_values(by="RPN", ascending=False)
                
                # Apply color gradient to RPN column
                st.dataframe(
                    rpn_df_concat[['id', 'Source', 'failure_mode', 'S', 'O', 'D', 'RPN']].style.background_gradient(cmap='YlOrRd', subset=['RPN']),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("No FMEA data available to generate an RPN summary.")

            st.divider()
            
            # --- Fault Tree Analysis (FTA) Chart ---
            st.markdown("#### Fault Tree Analysis (FTA) - Illustrative Example")
            st.caption("This FTA provides a top-down analysis of how lower-level faults can combine to cause the critical hazard of a **False Negative Result**. The size of each box represents its contribution to the overall failure probability (illustrative).")

            # Prepare data for the treemap
            parents = ["", "False Negative", "False Negative", "False Negative", "Assay Failure", "Assay Failure", "Software Failure", "Software Failure"]
            labels = ["False Negative", "Assay Failure", "Pre-analytical Error", "Software Failure", "Incomplete Conversion", "Contamination", "Model Overfitting", "Data Corruption"]
            values = [100, 60, 15, 25, 30, 30, 15, 10] # Illustrative values
            
            fta_fig = go.Figure(go.Treemap(
                labels = labels,
                parents = parents,
                values = values,
                textinfo = "label+value",
                # SME Definitive Fix: Corrected invalid property 'marker_colorscalefast' to 'marker.colorscale'
                marker=dict(colorscale='Reds')
            ))
            fta_fig.update_layout(
                title_text="<b>FTA for Top-Level Hazard: False Negative Result</b>",
                height=500, margin = dict(t=50, l=25, r=25, b=25)
            )
            st.plotly_chart(fta_fig, use_container_width=True)

        with tab6:
            st.subheader("Overall Residual Risk-Benefit Analysis & Conclusion")
            st.markdown("This is the final conclusion of the risk management process, required by ISO 14971. It should be a formal statement declaring whether the overall residual risk (the sum of all individual residual risks) is acceptable in relation to the documented clinical benefits of the device.")
            with st.form("risk_benefit_form"):
                analysis_val = st.text_area("**Risk-Benefit Analysis Statement:**", value=rmf_data.get("overall_risk_benefit_analysis", ""), height=200)
                if st.form_submit_button("Save Risk-Benefit Statement"):
                    rmf_data["overall_risk_benefit_analysis"] = analysis_val
                    ssm.update_data(rmf_data, "risk_management_file")
                    st.toast("Risk-Benefit statement saved!", icon="✅")

    except Exception as e:
        st.error("An error occurred while displaying the Risk Management section. The data may be malformed.")
        logger.error(f"Failed to render design risk management: {e}", exc_info=True)
