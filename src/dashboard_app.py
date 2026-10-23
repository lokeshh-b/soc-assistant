"""src/dashboard_app.py

Streamlit interactive analyst dashboard module for the Explainable SOC Assistant.
Provides tier-1/tier-2 SOC analysts with:
1. Real-time prioritized alert feeds and noise reduction telemetry (NRR).
2. Tier 1 SHAP feature attribution charts for interpretable anomaly scoring.
3. Tier 2 GenAI structured threat summaries with MITRE ATT&CK mapping.
4. Human-in-the-loop (HITL) review capture with hallucination auditing for research.
"""

from datetime import datetime, timezone
import os
import sys
from typing import Any, Dict, List, Optional

import streamlit as st

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schemas import (
    AnalystReviewSubmission,
    IncidentRecord,
    ShapFeatureImpact,
    Tier2LLMAnalysis,
    WindowMetricsRecord,
)
from src.state_store import StateStore


# =============================================================================
# Database Connection & Helper Functions
# =============================================================================

@st.cache_resource
def get_state_store() -> StateStore:
    """Initialize and cache the SQLite StateStore instance."""
    store = StateStore()
    store.init_db()
    return store


def seed_demo_data(store: StateStore) -> None:
    """Populate realistic sample telemetry and alerts for demonstration & testing."""
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Telemetry windows (3 benign, 1 duplicate coalesced, 2 forwarded)
    windows = [
        WindowMetricsRecord(
            window_id="win_20261022_0101",
            timestamp_end=now_iso,
            total_raw_logs=120,
            risk_score=0.12,
            forwarded_to_llm=0,
            tier1_latency_ms=8.2,
            ground_truth_malicious=0,
        ),
        WindowMetricsRecord(
            window_id="win_20261022_0102",
            timestamp_end=now_iso,
            total_raw_logs=95,
            risk_score=0.18,
            forwarded_to_llm=0,
            tier1_latency_ms=7.9,
            ground_truth_malicious=0,
        ),
        WindowMetricsRecord(
            window_id="win_20261022_0103",
            timestamp_end=now_iso,
            total_raw_logs=140,
            risk_score=0.08,
            forwarded_to_llm=0,
            tier1_latency_ms=9.1,
            ground_truth_malicious=0,
        ),
        WindowMetricsRecord(
            window_id="win_20261022_0104",
            timestamp_end=now_iso,
            total_raw_logs=210,
            risk_score=0.45,
            forwarded_to_llm=2,
            tier1_latency_ms=11.4,
            ground_truth_malicious=0,
        ),
        WindowMetricsRecord(
            window_id="win_20261022_0042",
            timestamp_end=now_iso,
            total_raw_logs=350,
            risk_score=0.92,
            forwarded_to_llm=1,
            tier1_latency_ms=12.4,
            ground_truth_malicious=1,
            ground_truth_mitre="T1110.001",
        ),
        WindowMetricsRecord(
            window_id="win_20261022_0043",
            timestamp_end=now_iso,
            total_raw_logs=180,
            risk_score=0.81,
            forwarded_to_llm=1,
            tier1_latency_ms=10.8,
            ground_truth_malicious=1,
            ground_truth_mitre="T1078",
        ),
    ]
    for w in windows:
        store.save_window_metrics(w)

    # 2. Incident 1: SSH Brute Force
    inc1 = IncidentRecord(
        window_id="win_20261022_0042",
        created_at=now_iso,
        source_ip="192.168.1.105",
        target_host="srv-db-prod-01",
        occurrence_count=3,
        risk_score=0.92,
        shap_summary=[
            ShapFeatureImpact(feature="failed_auth_count_10s", value=48, shap_impact=0.41),
            ShapFeatureImpact(feature="distinct_usernames_tried", value=12, shap_impact=0.28),
            ShapFeatureImpact(feature="ssh_port_22_ratio", value=0.96, shap_impact=0.15),
            ShapFeatureImpact(feature="bytes_in_ratio", value=0.12, shap_impact=-0.04),
        ],
        raw_logs=[
            "Oct 22 21:10:01 srv-db-prod-01 sshd[4120]: Failed password for invalid user admin from 192.168.1.105 port 54122 ssh2",
            "Oct 22 21:10:04 srv-db-prod-01 sshd[4125]: Failed password for root from 192.168.1.105 port 54128 ssh2",
            "Oct 22 21:10:09 srv-db-prod-01 sshd[4199]: Accepted password for root from 192.168.1.105 port 54190 ssh2",
        ],
        llm_analysis=Tier2LLMAnalysis(
            evidence_status="Sufficient",
            incident_summary="Detected an automated credential stuffing brute-force attack originating from internal workstation 192.168.1.105 against target srv-db-prod-01. The pattern shows 48 failed authentications across 12 distinct usernames followed by an anomalous root login.",
            threat_category="Brute Force",
            mitre_attack_id="T1110.001",
            recommended_actions=[
                "Immediately revoke active SSH sessions for root on srv-db-prod-01.",
                "Isolate originating host 192.168.1.105 at the network firewall switch.",
                "Enforce multi-factor authentication (MFA) and disable direct SSH root logins.",
            ],
        ),
        tier2_latency_ms=1240.5,
        analyst_decision="PENDING",
        hallucination_flag=0,
    )
    store.save_incident(inc1)

    # 3. Incident 2: Suspicious Admin Escalation
    inc2 = IncidentRecord(
        window_id="win_20261022_0043",
        created_at=now_iso,
        source_ip="10.0.4.52",
        target_host="ad-controller-01",
        occurrence_count=1,
        risk_score=0.81,
        shap_summary=[
            ShapFeatureImpact(feature="privilege_escalation_calls", value=6, shap_impact=0.37),
            ShapFeatureImpact(feature="off_hours_activity_score", value=0.88, shap_impact=0.24),
            ShapFeatureImpact(feature="kerberos_ticket_requests", value=19, shap_impact=0.18),
        ],
        raw_logs=[
            "Oct 22 21:12:15 ad-controller-01 Security[4672]: Special privileges assigned to new logon user svc-backup",
            "Oct 22 21:12:18 ad-controller-01 Security[4673]: Sensitive privilege use detected for SeDebugPrivilege",
        ],
        llm_analysis=Tier2LLMAnalysis(
            evidence_status="Sufficient",
            incident_summary="Service account svc-backup requested high-privilege debug tokens outside standard operating maintenance windows. Feature attributions indicate anomalous elevation of privileges without corresponding change tickets.",
            threat_category="Privilege Escalation",
            mitre_attack_id="T1078",
            recommended_actions=[
                "Verify whether maintenance window was scheduled for svc-backup.",
                "Inspect process memory of invoking caller on ad-controller-01.",
            ],
        ),
        tier2_latency_ms=980.2,
        analyst_decision="PENDING",
        hallucination_flag=0,
    )
    store.save_incident(inc2)


# =============================================================================
# Dashboard Page Configuration & Cyber SOC Styling
# =============================================================================

def configure_page() -> None:
    """Set page layout, title, and dark theme styling."""
    st.set_page_config(
        page_title="AgentSOC | AI-Assisted Security Operations Console",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        /* Main background and fonts */
        .main {
            background-color: #0b0f19;
            color: #f1f5f9;
        }
        /* Top Telemetry Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        .metric-value {
            font-size: 26px;
            font-weight: 700;
            color: #38bdf8;
            margin-top: 4px;
        }
        .metric-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
        }
        /* Status Badges */
        .badge-pending {
            background-color: rgba(245, 158, 11, 0.2);
            color: #f59e0b;
            border: 1px solid #f59e0b;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-approved {
            background-color: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid #10b981;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-rejected {
            background-color: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border: 1px solid #ef4444;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-threat {
            background-color: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid #38bdf8;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }
        /* Card Containers */
        .panel-card {
            background-color: #131c2e;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0f172a;
        }
        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Main Application Flow
# =============================================================================

def main() -> None:
    """Render the Explainable SOC Assistant Streamlit dashboard."""
    configure_page()
    store = get_state_store()

    # -------------------------------------------------------------------------
    # Header & System Telemetry Bar
    # -------------------------------------------------------------------------
    st.title("🛡️ AgentSOC: Explainable AI-Assisted SOC Console")
    st.caption("Human-in-the-Loop Cybersecurity Alert Triage with SHAP Feature Attribution & GenAI Summaries")

    metrics = store.get_dashboard_metrics()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Noise Reduction (NRR)</div>
                <div class="metric-value">{metrics['noise_reduction_ratio'] * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Ingested Windows</div>
                <div class="metric-value">{metrics['total_windows']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Active Incidents</div>
                <div class="metric-value">{metrics['total_incidents']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Pending Review</div>
                <div class="metric-value" style="color: #f59e0b;">{metrics['pending_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Approved (True Pos)</div>
                <div class="metric-value" style="color: #10b981;">{metrics['approved_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Hallucination Flags</div>
                <div class="metric-value" style="color: #f43f5e;">{metrics['hallucination_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Sidebar Controls & Queue Filtering
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Controls & State")
        st.info(f"**Database**: SQLite WAL\n\n**Path**: `{store.db_path}`")

        filter_decision = st.selectbox(
            "Triage Status Filter",
            options=["ALL", "PENDING", "APPROVED", "REJECTED"],
            index=0,
            help="Filter the incident triage queue by human review status."
        )

        st.divider()
        st.subheader("🧪 Research & Simulation")
        if st.button("📥 Load Sample Alerts (Seed Data)", use_container_width=True):
            seed_demo_data(store)
            st.success("Sample telemetry and incidents loaded into SQLite!")
            st.rerun()

        if st.button("🔄 Refresh Data Feed", use_container_width=True):
            st.rerun()

    # -------------------------------------------------------------------------
    # Alert Queue & Incident Deep Dive
    # -------------------------------------------------------------------------
    decision_query = None if filter_decision == "ALL" else filter_decision
    incidents = store.get_incidents(decision=decision_query, limit=100)

    if not incidents:
        st.warning("No incidents match the selected filter. Click **'Load Sample Alerts (Seed Data)'** in the sidebar to populate test incidents.")
        return

    left_col, right_col = st.columns([5, 7], gap="medium")

    # Left: Prioritized Queue Feed
    with left_col:
        st.subheader("📋 Prioritized Incident Feed")

        incident_options = {
            f"[{inc.analyst_decision}] {inc.window_id} | Score: {inc.risk_score:.2f} | Host: {inc.target_host}": inc.window_id
            for inc in incidents
        }

        selected_label = st.radio(
            "Select Incident to Investigate:",
            options=list(incident_options.keys()),
            index=0,
            label_visibility="collapsed",
        )
        selected_window_id = incident_options[selected_label]
        active_incident = store.get_incident(selected_window_id)

    # Right: Incident Deep Dive & Human-in-the-Loop Review
    with right_col:
        if active_incident is None:
            st.error("Selected incident not found in database.")
            return

        st.subheader(f"🔍 Deep Dive: `{active_incident.window_id}`")

        # Entity Context Pill Bar
        badge_class = f"badge-{active_incident.analyst_decision.lower()}"
        st.markdown(
            f"""
            <div class="panel-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="{badge_class}">{active_incident.analyst_decision}</span>
                        <span class="badge-threat" style="margin-left: 8px;">Risk: {active_incident.risk_score:.2f}</span>
                        <span style="color: #94a3b8; font-size: 13px; margin-left: 12px;">Coalesced Count: {active_incident.occurrence_count}</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 12px;">{active_incident.created_at}</div>
                </div>
                <div style="margin-top: 10px; font-size: 14px;">
                    <strong>Source IP:</strong> <code style="color: #38bdf8;">{active_incident.source_ip}</code> &nbsp;⟶&nbsp;
                    <strong>Target Host:</strong> <code style="color: #38bdf8;">{active_incident.target_host}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_shap, tab_llm, tab_logs = st.tabs(["📊 Tier 1 Explainability (SHAP)", "🤖 Tier 2 GenAI Triage", "📜 Raw Log Evidence"])

        # Tab 1: SHAP Explanations
        with tab_shap:
            st.markdown("##### Behavioral Feature Attributions (Local SHAP Impact)")
            st.caption("Quantifies which behavioral log features drove the ML anomaly score.")

            if active_incident.shap_summary:
                try:
                    import plotly.graph_objects as go

                    features = [item.feature for item in active_incident.shap_summary]
                    # Parse numerical impacts safely
                    impacts = []
                    for item in active_incident.shap_summary:
                        try:
                            val = float(str(item.shap_impact).replace("+", ""))
                        except ValueError:
                            val = 0.0
                        impacts.append(val)

                    colors = ["#f43f5e" if imp > 0 else "#38bdf8" for imp in impacts]

                    fig = go.Figure(
                        go.Bar(
                            x=impacts,
                            y=features,
                            orientation="h",
                            marker=dict(color=colors),
                        )
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#131c2e",
                        plot_bgcolor="#131c2e",
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=260,
                        xaxis_title="SHAP Value (Threat Contribution)",
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except ImportError:
                    # Fallback visual table if plotly is not installed
                    for feat in active_incident.shap_summary:
                        st.markdown(f"- **`{feat.feature}`**: `{feat.value}` (SHAP: `{feat.shap_impact}`)")
            else:
                st.info("No SHAP feature attributions recorded for this window.")

        # Tab 2: GenAI Triage Narrative & Mitigation
        with tab_llm:
            if active_incident.llm_analysis:
                analysis = active_incident.llm_analysis

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Threat Category:** `{analysis.threat_category}`")
                    st.markdown(f"**MITRE ATT&CK ID:** `{analysis.mitre_attack_id}`")
                with col_b:
                    st.markdown(f"**Evidence Status:** `{analysis.evidence_status}`")
                    st.markdown(f"**GenAI Latency:** `{active_incident.tier2_latency_ms:.1f} ms`")

                st.markdown("##### Grounded Incident Summary")
                st.markdown(f"> *{analysis.incident_summary}*")

                st.markdown("##### Actionable Mitigation Recommendations")
                if analysis.recommended_actions:
                    for idx, action in enumerate(analysis.recommended_actions, 1):
                        st.markdown(f"**{idx}.** {action}")
                else:
                    st.info("No specific remediation actions generated.")
            else:
                st.warning("Tier 2 GenAI analysis has not been performed on this incident yet.")

        # Tab 3: Raw Logs
        with tab_logs:
            st.markdown("##### Ingested Raw Security Log Lines")
            if active_incident.raw_logs:
                st.code("\n".join(active_incident.raw_logs), language="text")
            else:
                st.info("No raw log snippets attached.")

        # ---------------------------------------------------------------------
        # Human-in-the-Loop Review Box
        # ---------------------------------------------------------------------
        st.divider()
        st.subheader("✍️ Human-in-the-Loop Analyst Decision")

        with st.form(key=f"review_form_{active_incident.window_id}"):
            current_decision_idx = 0 if active_incident.analyst_decision in ["PENDING", "APPROVED"] else 1
            decision_choice = st.radio(
                "Triage Verdict:",
                options=["APPROVED", "REJECTED"],
                index=current_decision_idx,
                horizontal=True,
                help="'APPROVED' confirms a genuine security incident. 'REJECTED' flags as false positive."
            )

            hallucination_flag = st.checkbox(
                "🚩 Flag as Hallucination / Unsupported Claim (Research Audit)",
                value=active_incident.is_hallucination,
                help="Check if the GenAI narrative claims actions or indicators unsupported by the raw logs and SHAP values."
            )

            analyst_notes = st.text_area(
                "Operational Notes / Justification:",
                value=active_incident.analyst_notes or "",
                placeholder="Document your review reasoning, ticket ID, or escalation details...",
                height=90,
            )

            submit_clicked = st.form_submit_button("💾 Submit Analyst Decision", use_container_width=True)

            if submit_clicked:
                review_payload = AnalystReviewSubmission(
                    window_id=active_incident.window_id,
                    decision=decision_choice,
                    hallucination_flag=hallucination_flag,
                    analyst_notes=analyst_notes.strip() if analyst_notes else None,
                )
                try:
                    updated_record = store.submit_review(review_payload)
                    st.success(f"Review submitted for {updated_record.window_id}! Status: {updated_record.analyst_decision}")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to record review: {ex}")


if __name__ == "__main__":
    main()
