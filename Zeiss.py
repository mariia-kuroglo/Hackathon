import glob
import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="ZEISS Energy Smart AI", layout="wide", initial_sidebar_state="collapsed")


# --- 0. UI HELPER FUNCTIONS ---
def neon_card(title, value, delta, is_good_delta=True):
    """Generates a custom HTML card with neon styling"""
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


# --- 1. BASELINE ENGINE ---
DEFAULT_BASELINE = {
    "idle": {"avg_time_min": 240, "avg_energy_wh": 562},
    "live_view_monitoring": {"avg_time_min": 175, "avg_energy_wh": 576},
    "processing": {"avg_time_min": 140, "avg_energy_wh": 533},
    "tile_scan_acquisition": {"avg_time_min": 257, "avg_energy_wh": 982}
}


@st.cache_data
def learn_efficient_baseline(training_data_folder):
    search_pattern = os.path.join(training_data_folder, "S*_v4.csv")
    training_files = glob.glob(search_pattern)
    if not training_files:
        return DEFAULT_BASELINE.copy()
    all_data = []
    for file in training_files:
        try:
            df = pd.read_csv(file)
            if 'workflow_phase' in df.columns and 'estimated_energy_wh_interval' in df.columns:
                phase_stats = df.groupby('workflow_phase').agg(
                    time_min=('sample_interval_sec', lambda x: x.sum() / 60),
                    energy_wh=('estimated_energy_wh_interval', 'sum')
                ).reset_index()
                all_data.append(phase_stats)
        except:
            continue
    if not all_data: return DEFAULT_BASELINE.copy()
    combined_df = pd.concat(all_data)
    baseline_df = combined_df.groupby('workflow_phase').mean().to_dict('index')
    result = DEFAULT_BASELINE.copy()
    for phase, stats in baseline_df.items():
        result[phase] = {"avg_time_min": stats['time_min'], "avg_energy_wh": stats['energy_wh']}
    return result


# --- 2. DYNAMIC PATHS ---
current_dir = os.getcwd()
training_folder_path = os.path.join(current_dir, "data", "training")
BASELINE_PHASES = learn_efficient_baseline(training_folder_path)

# --- 3. MAIN UI ---
st.title("🔬 ZEISS Energy Smart AI Assistant")
st.markdown(
    "Upload a new microscope workflow. The AI will analyze usage patterns and recommend **9R Strategy** optimizations.")

uploaded_file = st.file_uploader("Upload Test Scenario (CSV)", type="csv")

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    if 'workflow_phase' in df_new.columns:
        st.write("---")
        st.subheader("📊 High-Level Phase Analysis")

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
                with cols[i]:
                    neon_card(title=phase.replace('_', ' ').title(), value=f"{actual_time:.0f} mins",
                              delta=f"{diff_pct:+.1f}% vs reference", is_good_delta=(diff_pct <= 0))

                if phase == "idle" and diff_pct > 10 and 'total_energy_wh' in new_stats[phase]:
                    waste_wh = max(0, new_stats[phase]['total_energy_wh'] - baseline_data['avg_energy_wh'])
                    detected_issues.append({
                        "issue": f"Idle time was {diff_pct:.0f}% higher than baseline.",
                        "recommendation": f"Implement automated post-run sleep schedule. This will help reduce energy consumption by **{waste_wh:.1f} Wh**.",
                        "strategy": "R2 Reduce"
                    })

        # Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=chart_phases, y=chart_baseline, name='Baseline', marker_color='#2a2b3d'))
        fig.add_trace(go.Bar(x=chart_phases, y=chart_actual, name='Actual', marker_color='#00ffcc'))
        fig.update_layout(barmode='group', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        st.subheader("🔍 Deep AI Diagnostics")

        # FEATURE 2: Live View Camera
        if 'live_view_enabled_flag' in df_new.columns and 'tile_scan_enabled_flag' in df_new.columns:
            overlap = df_new[(df_new['live_view_enabled_flag'] == True) & (df_new['tile_scan_enabled_flag'] == True)]
            if not overlap.empty and 'estimated_energy_wh_interval' in overlap.columns:
                waste_wh = overlap['estimated_energy_wh_interval'].sum() * 0.15
                detected_issues.append({
                    "issue": "Live-view camera active during automated scan.",
                    "recommendation": f"Disable live-view monitoring during scans. This will help reduce energy consumption by **{waste_wh:.1f} Wh**.",
                    "strategy": "R1 Rethink"
                })

        # FEATURE 3: Over-Scanning (NEW THRESHOLDS)
        if 'tile_overlap_pct' in df_new.columns and 'quality_constraint' in df_new.columns:
            df_new['quality_constraint'] = df_new['quality_constraint'].astype(str).str.strip().str.lower()

            # Logic according to your requirements
            ov_low = df_new[(df_new['tile_overlap_pct'] > 12) & (df_new['quality_constraint'] == 'low')]
            ov_med = df_new[(df_new['tile_overlap_pct'] > 15) & (df_new['quality_constraint'] == 'medium')]
            ov_high = df_new[(df_new['tile_overlap_pct'] > 20) & (df_new['quality_constraint'] == 'high')]

            all_ov = pd.concat([ov_low, ov_med, ov_high]).drop_duplicates()
            if not all_ov.empty:
                waste_wh = all_ov[
                               'estimated_energy_wh_interval'].sum() * 0.12 if 'estimated_energy_wh_interval' in all_ov.columns else 0

                # Determine specific text based on the highest violation
                if not ov_high.empty:
                    rec_val = "20%"
                elif not ov_med.empty:
                    rec_val = "15%"
                else:
                    rec_val = "10-12%"

                detected_issues.append({
                    "issue": f"Excessive overlap detected ({df_new['tile_overlap_pct'].max():.1f}%).",
                    "recommendation": f"Reduce tile overlap to the recommended {rec_val} for this quality level. This will help reduce energy consumption by **{waste_wh:.1f} Wh**.",
                    "strategy": "R2 Reduce"
                })

        # FEATURE 4: Background Load
        if 'perf_cpu_pct' in df_new.columns and 'perf_gpu_usage_pct' in df_new.columns:
            avg_cpu, avg_gpu = df_new['perf_cpu_pct'].mean(), df_new['perf_gpu_usage_pct'].mean()
            if avg_cpu > 12 or avg_gpu > 10:
                waste_wh = df_new[
                               'estimated_energy_wh_interval'].sum() * 0.08 if 'estimated_energy_wh_interval' in df_new.columns else 0
                detected_issues.append({
                    "issue": f"High background load (CPU: {avg_cpu:.1f}%, GPU: {avg_gpu:.1f}%).",
                    "recommendation": f"Close non-critical background processes. This will help reduce energy consumption by **{waste_wh:.1f} Wh**.",
                    "strategy": "R2 Reduce"
                })

        # FEATURE 5: Fragmented Workflow
        phase_changes = (df_new['workflow_phase'] != df_new['workflow_phase'].shift()).sum()
        if phase_changes > 15:
            detected_issues.append({
                "issue": f"Highly fragmented workflow ({phase_changes} transitions).",
                "recommendation": "Batch samples to save warm-up/cool-down energy.",
                "strategy": "R1 Rethink"
            })

        # FINAL RESULTS
        if detected_issues:
            st.warning("🤖 **Energy Assistant:** Analysis complete.")
            for item in detected_issues:
                with st.container():
                    st.error(f"⚠️ **Anomaly Detected:** {item['issue']}")
                    st.info(
                        f"💡 **Recommendation:** {item['recommendation']}  \n**[Mapped Strategy: {item['strategy']}]**")
                    st.write("")

                    # --- TEACH THE AI ---
            st.write("---")
            st.subheader("🧠 Teach the AI")
            memory_file = "microscope_memory.txt"
            if st.button("Accept & Save Preferences to Profile"):
                with open(memory_file, "w") as f: f.write(f"User acknowledged {detected_issues[0]['strategy']}.")
                st.balloons();
                st.toast("AI Memory Updated!", icon="✅")
        else:
            st.success("🎉 Your workflow is running efficiently.")