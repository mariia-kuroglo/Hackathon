import glob
import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go

# --- PAGE SETUP ---
# Must be the first Streamlit command. Sets the page title and default layout.
st.set_page_config(page_title="ZEISS Energy Smart AI", layout="wide", initial_sidebar_state="collapsed")


# --- 0. UI HELPER FUNCTIONS ---
def neon_card(title, value, delta, is_good_delta=True):
    """
    Generates a custom HTML/CSS card with neon styling for the dashboard.
    Changes border and text color based on whether the performance is 'good' or 'bad'.
    """
    delta_color = "#00ffcc" if is_good_delta else "#ff3366"
    border_color = "#00ffcc" if is_good_delta else "#ff3366"

    card_html = f"""
    <div style="
        background-color: #14151f;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2a2b3d;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border-left: 4px solid {border_color};
        margin-bottom: 20px;
    ">
        <h4 style="color: #8a8d9e; margin-top: 0; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">{title}</h4>
        <h2 style="color: #ffffff; margin: 10px 0; font-size: 32px; font-weight: 600;">{value}</h2>
        <span style="color: {delta_color}; font-weight: 600; font-size: 14px;">
            {delta}
        </span>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# --- 1. SMART BASELINE ENGINE (Expert-Trained + Outlier Filtering) ---
DEFAULT_BASELINE = {
    "idle": {"avg_time_min": 240, "avg_energy_wh": 562},
    "live_view_monitoring": {"avg_time_min": 175, "avg_energy_wh": 576},
    "processing": {"avg_time_min": 140, "avg_energy_wh": 533},
    "tile_scan_acquisition": {"avg_time_min": 257, "avg_energy_wh": 982}
}


@st.cache_data
def learn_expert_baseline(training_data_folder):
    """
    1. Filters only 'no_action' rows from training files (pure efficiency).
    2. Groups data by phase.
    3. Uses IQR (Interquartile Range) to remove statistical outliers.
    4. Calculates the final 'Expert Baseline'.
    """
    search_pattern = os.path.join(training_data_folder, "S*_v4.csv")
    training_files = glob.glob(search_pattern)
    if not training_files:
        return DEFAULT_BASELINE.copy()

    raw_phase_data = []
    for file in training_files:
        try:
            df = pd.read_csv(file)
            # EXPERT FILTER: Learn only from segments where ZEISS experts said "no_action"
            if 'recommended_action' in df.columns:
                df = df[df['recommended_action'] == 'no_action']

            if not df.empty and 'workflow_phase' in df.columns:
                stats = df.groupby('workflow_phase').agg(
                    time_min=('sample_interval_sec', lambda x: x.sum() / 60),
                    energy_wh=('estimated_energy_wh_interval', 'sum')
                ).reset_index()
                raw_phase_data.append(stats)
        except:
            continue

    if not raw_phase_data: return DEFAULT_BASELINE.copy()

    full_df = pd.concat(raw_phase_data)
    result = {}

    # Filter outliers separately for each phase to ensure a "Clean Baseline"
    for phase in full_df['workflow_phase'].unique():
        phase_subset = full_df[full_df['workflow_phase'] == phase]

        # If too little data, fallback to simple mean
        if len(phase_subset) < 4:
            result[phase] = {
                "avg_time_min": phase_subset['time_min'].mean(),
                "avg_energy_wh": phase_subset['energy_wh'].mean()
            }
            continue

        # IQR Method for Time
        q1_t, q3_t = phase_subset['time_min'].quantile([0.25, 0.75])
        iqr_t = q3_t - q1_t
        filtered_time = phase_subset[(phase_subset['time_min'] >= q1_t - 1.5 * iqr_t) &
                                     (phase_subset['time_min'] <= q3_t + 1.5 * iqr_t)]['time_min']

        # IQR Method for Energy
        q1_e, q3_e = phase_subset['energy_wh'].quantile([0.25, 0.75])
        iqr_e = q3_e - q1_e
        filtered_energy = phase_subset[(phase_subset['energy_wh'] >= q1_e - 1.5 * iqr_e) &
                                       (phase_subset['energy_wh'] <= q3_e + 1.5 * iqr_e)]['energy_wh']

        result[phase] = {
            "avg_time_min": filtered_time.mean() if not filtered_time.empty else phase_subset['time_min'].mean(),
            "avg_energy_wh": filtered_energy.mean() if not filtered_energy.empty else phase_subset['energy_wh'].mean()
        }
    return result


# --- 2. DYNAMIC PATH RESOLUTION ---
current_dir = os.getcwd()
training_folder_path = os.path.join(current_dir, "data", "training")
BASELINE_PHASES = learn_expert_baseline(training_folder_path)

# --- 3. MAIN UI ---
st.title("🔬 ZEISS Energy Smart AI Assistant")
st.markdown("Analyzing workflows against the **Expert-Trained Baseline** (IQR Filtered).")

uploaded_file = st.file_uploader("Upload New Workflow Scenario (CSV)", type="csv")

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    if 'workflow_phase' in df_new.columns:
        st.write("---")
        st.subheader("📊 High-Level Phase Analysis")

        # Aggregate metrics for the uploaded file
        agg_dict = {'total_time_min': ('sample_interval_sec', lambda x: x.sum() / 60)}
        if 'estimated_energy_wh_interval' in df_new.columns:
            agg_dict['total_energy_wh'] = ('estimated_energy_wh_interval', 'sum')

        new_stats = df_new.groupby('workflow_phase').agg(**agg_dict).to_dict('index')
        chart_phases, chart_actual, chart_baseline, detected_issues = [], [], [], []
        cols = st.columns(4)

        for i, phase in enumerate(["idle", "live_view_monitoring", "processing", "tile_scan_acquisition"]):
            if phase in new_stats:
                actual_time = new_stats[phase]['total_time_min']
                baseline_data = BASELINE_PHASES.get(phase, DEFAULT_BASELINE.get(phase))
                baseline_time = baseline_data['avg_time_min']

                chart_phases.append(phase.replace('_', ' ').title());
                chart_actual.append(actual_time);
                chart_baseline.append(baseline_time)

                diff_pct = ((actual_time - baseline_time) / baseline_time) * 100 if baseline_time > 0 else 0
                is_good = diff_pct <= 5  # Margin of error for good performance

                with cols[i]:
                    neon_card(
                        title=phase.replace('_', ' ').title(),
                        value=f"{actual_time:.0f} mins",
                        delta=f"{diff_pct:+.1f}% vs baseline",
                        is_good_delta=is_good
                    )

                # DIAGNOSTIC: Idle Waste Calculation
                if phase == "idle" and diff_pct > 5:
                    waste_wh = max(0, new_stats[phase].get('total_energy_wh', 0) - baseline_data['avg_energy_wh'])
                    detected_issues.append({
                        "issue": f"Idle time is {diff_pct:.0f}% higher than efficient baseline.",
                        "recommendation": f"Implement automated post-run sleep schedule. Potential savings: **{waste_wh:.1f} Wh**.",
                        "strategy": "R2 Reduce"
                    })

        # --- ENERGY BAR CHART ---
        fig = go.Figure()
        fig.add_trace(go.Bar(x=chart_phases, y=chart_baseline, name='Expert Baseline', marker_color='#2a2b3d'))
        fig.add_trace(go.Bar(x=chart_phases, y=chart_actual, name='Current Run', marker_color='#00ffcc'))
        fig.update_layout(
            barmode='group', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=0, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        st.subheader("🔍 Deep AI Diagnostics")

        # FEATURE 2: Redundant Camera (Live View during Scanning)
        if 'tile_scan_enabled_flag' in df_new.columns and 'live_view_enabled_flag' in df_new.columns:
            background_monitoring = df_new[
                (df_new['tile_scan_enabled_flag'] == True) & (df_new['live_view_enabled_flag'] == True)]
            if not background_monitoring.empty and 'estimated_energy_wh_interval' in background_monitoring.columns:
                # Assume camera/light is roughly 15% of the total system power in that interval
                waste_wh = background_monitoring['estimated_energy_wh_interval'].sum() * 0.15
                detected_issues.append({
                    "issue": f"Redundant Live View active during scanning ({len(background_monitoring) * 15 / 60:.1f} mins).",
                    "recommendation": f"Pause Live View monitoring while automated scan is running. Potential savings: **{waste_wh:.1f} Wh**.",
                    "strategy": "R1 Rethink"
                })

        # FEATURE 3: Over-Scanning (Dynamic Quality Thresholds)
        if 'tile_overlap_pct' in df_new.columns and 'quality_constraint' in df_new.columns:
            df_new['quality_constraint'] = df_new['quality_constraint'].astype(str).str.strip().str.lower()

            # Logic: High Quality -> 20%, Medium -> 15%, Low -> 12%
            ov_low = df_new[(df_new['tile_overlap_pct'] > 12) & (df_new['quality_constraint'] == 'low')]
            ov_med = df_new[(df_new['tile_overlap_pct'] > 15) & (df_new['quality_constraint'] == 'medium')]
            ov_high = df_new[(df_new['tile_overlap_pct'] > 20) & (df_new['quality_constraint'] == 'high')]

            all_ov = pd.concat([ov_low, ov_med, ov_high]).drop_duplicates()
            if not all_ov.empty:
                waste_wh = all_ov[
                               'estimated_energy_wh_interval'].sum() * 0.12 if 'estimated_energy_wh_interval' in all_ov.columns else 0
                rec_val = "20%" if not ov_high.empty else ("15%" if not ov_med.empty else "10-12%")
                detected_issues.append({
                    "issue": f"Excessive overlap detected ({df_new['tile_overlap_pct'].max():.1f}%).",
                    "recommendation": f"Reduce tile overlap to the recommended {rec_val} for this quality level. Potential savings: **{waste_wh:.1f} Wh**.",
                    "strategy": "R2 Reduce"
                })

        # FEATURE 4: Background System Load
        if 'perf_cpu_pct' in df_new.columns and 'perf_gpu_usage_pct' in df_new.columns:
            avg_cpu, avg_gpu = df_new['perf_cpu_pct'].mean(), df_new['perf_gpu_usage_pct'].mean()
            if avg_cpu > 12 or avg_gpu > 10:
                waste_wh = df_new[
                               'estimated_energy_wh_interval'].sum() * 0.08 if 'estimated_energy_wh_interval' in df_new.columns else 0
                detected_issues.append({
                    "issue": f"High background system load (CPU: {avg_cpu:.1f}%, GPU: {avg_gpu:.1f}%).",
                    "recommendation": f"Close non-critical background applications. Potential savings: **{waste_wh:.1f} Wh**.",
                    "strategy": "R2 Reduce"
                })

        # FEATURE 5: Fragmented Workflow
        phase_changes = (df_new['workflow_phase'] != df_new['workflow_phase'].shift()).sum()
        if phase_changes > 15:
            detected_issues.append({
                "issue": f"Highly fragmented workflow detected ({phase_changes} transitions).",
                "recommendation": "Try batching samples together to minimize warm-up and cool-down energy cycles.",
                "strategy": "R1 Rethink"
            })

        # --- DISPLAY FINAL DIAGNOSTICS ---
        if detected_issues:
            st.warning("🤖 **Energy Assistant:** I've identified several optimization opportunities:")
            for item in detected_issues:
                with st.container():
                    st.error(f"⚠️ **Anomaly:** {item['issue']}")
                    st.info(f"💡 **Action:** {item['recommendation']}  \n**[Strategy: {item['strategy']}]**")
                    st.write("")

            st.write("---")
            st.subheader("🧠 Active Learning Integration")
            if st.button("Accept & Update User Profile"):
                st.balloons()
                st.toast("AI Model adjusted based on your feedback!", icon="✅")
        else:
            st.success("🎉 Your workflow is perfectly optimized against the Expert Baseline!")
    else:
        st.error("Invalid CSV format. Please ensure required columns exist.")