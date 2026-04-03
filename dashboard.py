"""
Milestone 4: Advanced Streamlit Dashboard - Azure Capacity Intelligence
Professional dashboard matching Looker Studio style with multiple tabs, filters, and advanced analytics.
Uses default dataset.csv or forecast_output.csv for data.
"""

import json
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Azure Capacity Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #e6edf3;
    }
    
    .stMetric {
        background-color: #161b22;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #58a6ff;
    }
    
    .dashboard-header {
        color: #58a6ff;
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #8b949e;
        font-size: 0.9em;
        margin-bottom: 1.5rem;
    }
    
    .tab-title {
        color: #58a6ff;
        font-size: 1.3em;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data
def load_default_data():
    """Load default dataset.csv file."""
    csv_files = ['forecast_output.csv', 'dataset.csv', 'new_data.csv']
    
    for csv_file in csv_files:
        if Path(csv_file).exists():
            try:
                df = pd.read_csv(csv_file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Ensure required columns exist
                if 'usage_units' not in df.columns and 'predicted_usage' in df.columns:
                    df['usage_units'] = df['predicted_usage']
                
                return df, csv_file
            except Exception as e:
                continue
    
    return None, None


def validate_dashboard_data(df: pd.DataFrame):
    """Validate essential columns for dashboard operation."""
    required_columns = ["timestamp", "region", "service_type"]
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


@st.cache_data
def load_monitoring_metrics(metrics_file: str = "model_metrics.json"):
    """Load latest model monitoring metrics when available."""
    path = Path(metrics_file)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        if isinstance(metrics, list) and metrics:
            return metrics[-1]
    except Exception:
        return None
    return None


@st.cache_resource
def load_model_feature_importance(model_path: str = "artifacts/tuned_xgboost_model.pkl"):
    """Load feature importance from trained model if supported."""
    path = Path(model_path)
    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            artifact = pickle.load(f)

        model = artifact.get("model") if isinstance(artifact, dict) else artifact
        feature_columns = artifact.get("feature_columns") if isinstance(artifact, dict) else None

        if model is None:
            return None

        if hasattr(model, "feature_importances_"):
            importances = list(model.feature_importances_)
        elif hasattr(model, "get_booster"):
            booster = model.get_booster()
            score_map = booster.get_score(importance_type="gain")
            if not score_map:
                return None

            if feature_columns:
                ordered_names = list(feature_columns)
            elif hasattr(booster, "feature_names") and booster.feature_names:
                ordered_names = list(booster.feature_names)
            else:
                ordered_names = sorted(score_map.keys())

            importance_lookup = {name: float(score_map.get(name, 0.0)) for name in ordered_names}
            return pd.DataFrame({
                "Feature": list(importance_lookup.keys()),
                "Importance": list(importance_lookup.values()),
            }).sort_values("Importance", ascending=False)
        else:
            return None

        if feature_columns and len(feature_columns) == len(importances):
            feature_names = list(feature_columns)
        elif hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        elif hasattr(model, "get_booster") and getattr(model.get_booster(), "feature_names", None):
            feature_names = list(model.get_booster().feature_names)
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False)
    except Exception:
        return None


def calculate_advanced_metrics(df, selected_regions=None, selected_services=None):
    """Calculate advanced KPIs with filtering."""
    # Apply filters
    filtered_df = df.copy()
    
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    if selected_services:
        filtered_df = filtered_df[filtered_df['service_type'].isin(selected_services)]
    
    metrics = {}
    
    # Capacity risk (high utilization)
    if 'provisioned_capacity' in filtered_df.columns and 'usage_units' in filtered_df.columns:
        utilization = (filtered_df['usage_units'] / filtered_df['provisioned_capacity'] * 100)
        metrics['capacity_risk_events'] = (utilization > 80).sum()
        metrics['avg_utilization'] = utilization.mean()
        metrics['max_utilization'] = utilization.max()
    else:
        metrics['capacity_risk_events'] = 0
        metrics['avg_utilization'] = 0
        metrics['max_utilization'] = 0
    
    # Underutilized flags (low utilization)
    if 'provisioned_capacity' in filtered_df.columns and 'usage_units' in filtered_df.columns:
        underutil = (utilization < 30).sum()
        metrics['underutilized_flags'] = underutil
    else:
        metrics['underutilized_flags'] = 0
    
    # Headroom (available capacity)
    if 'provisioned_capacity' in filtered_df.columns and 'usage_units' in filtered_df.columns:
        headroom = (filtered_df['provisioned_capacity'] - filtered_df['usage_units']).mean()
        metrics['avg_headroom'] = max(0, headroom)
    else:
        metrics['avg_headroom'] = 0
    
    # Growth rate
    if 'usage_units' in filtered_df.columns:
        sorted_df = filtered_df.sort_values('timestamp')
        if len(sorted_df) > 1:
            first_period = sorted_df['usage_units'].iloc[:len(sorted_df)//2].mean()
            last_period = sorted_df['usage_units'].iloc[len(sorted_df)//2:].mean()
            if first_period > 0:
                growth_rate = ((last_period - first_period) / first_period * 100) / (len(sorted_df) / 30)
                metrics['daily_growth_rate'] = growth_rate
            else:
                metrics['daily_growth_rate'] = 0
        else:
            metrics['daily_growth_rate'] = 0
    else:
        metrics['daily_growth_rate'] = 0
    
    # Cost metrics
    if 'cost_usd' in filtered_df.columns:
        metrics['total_cost'] = filtered_df['cost_usd'].sum()
        metrics['avg_cost'] = filtered_df['cost_usd'].mean()
    else:
        metrics['total_cost'] = 0
        metrics['avg_cost'] = 0
    
    return metrics


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="dashboard-header">🚀 Azure Capacity Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Milestone 4 - Forecast Integration & Capacity Planning Dashboard</div>', unsafe_allow_html=True)

    # Top-level controls
    controls_col1, controls_col2 = st.columns([1, 3])
    with controls_col1:
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    with controls_col2:
        st.caption("Use Refresh Data after scheduler or batch prediction updates the files.")
    
    # Load data
    df, source_file = load_default_data()
    
    if df is None:
        st.error("No data files found. Please ensure forecast_output.csv or dataset.csv exists.")
        return

    is_valid, missing_cols = validate_dashboard_data(df)
    if not is_valid:
        st.error(
            "Data is missing required columns for dashboard rendering: "
            f"{', '.join(missing_cols)}"
        )
        st.info("Expected minimum columns: timestamp, region, service_type")
        return
    
    st.success(f"✓ Loaded {len(df)} records from {source_file}")

    # Monitoring status panel
    latest_metrics = load_monitoring_metrics()
    if latest_metrics:
        drift_label = "Drift Detected" if latest_metrics.get("is_drifted") else "Healthy"
        drift_delta = f"RMSE Δ: {latest_metrics.get('rmse_increase_percent', 0):.1f}%"
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Model Status", drift_label, drift_delta)
        with k2:
            st.metric("Latest RMSE", f"{latest_metrics.get('rmse', 0):.3f}")
        with k3:
            st.metric("Directional Accuracy", f"{latest_metrics.get('directional_accuracy', 0):.1f}%")
    else:
        st.info("Monitoring metrics not found yet. Run model monitoring to populate model_metrics.json.")
    
    # Sidebar: Filters
    st.sidebar.markdown("## 🎛️ Filters")
    
    # Region filter
    available_regions = sorted(df['region'].unique())
    selected_regions = st.sidebar.multiselect(
        "Select Regions:",
        options=available_regions,
        default=available_regions,
        key="region_filter"
    )
    
    # Service type filter
    available_services = sorted(df['service_type'].unique())
    selected_services = st.sidebar.multiselect(
        "Select Service Types:",
        options=available_services,
        default=available_services,
        key="service_filter"
    )
    
    # Date range filter
    st.sidebar.markdown("### Date Range")
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    date_range = st.sidebar.slider(
        "Select Date Range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        key="date_filter"
    )
    
    # Apply all filters
    filtered_df = df[
        (df['region'].isin(selected_regions)) &
        (df['service_type'].isin(selected_services)) &
        (df['timestamp'].dt.date >= date_range[0]) &
        (df['timestamp'].dt.date <= date_range[1])
    ]

    if filtered_df.empty:
        st.warning("No records match the selected filters. Adjust region, service, or date range.")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Records shown:** {len(filtered_df)} / {len(df)}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 KPI Overview",
        "📈 Demand Trends",
        "🌍 Regional Analysis",
        "🤖 Model & Forecast",
        "⚠️ Risk Alerts"
    ])
    
    # ========== TAB 1: KPI Overview ==========
    with tab1:
        st.markdown('<div class="tab-title">Executive KPIs</div>', unsafe_allow_html=True)
        
        metrics = calculate_advanced_metrics(filtered_df, selected_regions, selected_services)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🚨 Capacity Risk Events",
                f"{metrics['capacity_risk_events']}",
                f"{((metrics['capacity_risk_events'] / max(1, len(filtered_df))) * 100):.1f}% of records"
            )
        
        with col2:
            st.metric(
                "⚠️ Underutilized Flags",
                f"{metrics['underutilized_flags']}",
                f"{((metrics['underutilized_flags'] / max(1, len(filtered_df))) * 100):.1f}% of records"
            )
        
        with col3:
            st.metric(
                "📊 Avg Headroom (Units)",
                f"{metrics['avg_headroom']:,.0f}",
                "Available buffer"
            )
        
        with col4:
            st.metric(
                "📈 Avg Daily Growth Rate",
                f"{metrics['daily_growth_rate']:.3f}%",
                "Per day, all regions"
            )
        
        st.markdown("---")
        
        # More metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Total Cost (USD)",
                f"${metrics['total_cost']:,.0f}",
                "Filtered period"
            )
        
        with col2:
            st.metric(
                "📉 Avg Utilization",
                f"{metrics['avg_utilization']:.1f}%",
                "Average across records"
            )
        
        with col3:
            st.metric(
                "⬆️ Max Utilization",
                f"{metrics['max_utilization']:.1f}%",
                "Highest utilization"
            )
        
        with col4:
            st.metric(
                "💵 Avg Cost Per Record",
                f"${metrics['avg_cost']:.2f}",
                "Average unit cost"
            )
        
        st.markdown("---")
        
        # Cost composition
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Cost Efficiency Breakdown")
            
            if 'service_type' in filtered_df.columns and 'cost_usd' in filtered_df.columns:
                cost_by_service = filtered_df.groupby('service_type')['cost_usd'].sum()
                
                fig = go.Figure(data=[go.Pie(
                    labels=cost_by_service.index,
                    values=cost_by_service.values,
                    hole=.3,
                    marker=dict(colors=['#58a6ff', '#79c0ff', '#1f6feb'])
                )])
                
                fig.update_layout(
                    height=350,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor='rgba(22, 27, 34, 0.5)',
                    font=dict(color='#e6edf3')
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Utilization Distribution")
            
            if 'provisioned_capacity' in filtered_df.columns and 'usage_units' in filtered_df.columns:
                utilization = (filtered_df['usage_units'] / filtered_df['provisioned_capacity'] * 100)
                
                fig = go.Figure(data=[go.Histogram(
                    x=utilization,
                    nbinsx=20,
                    marker=dict(color='#58a6ff'),
                    name='Utilization %'
                )])
                
                fig.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="Risk Threshold (80%)")
                fig.add_vline(x=30, line_dash="dash", line_color="orange", annotation_text="Underutil. (30%)")
                
                fig.update_layout(
                    title_text="Utilization Distribution",
                    xaxis_title="Utilization %",
                    yaxis_title="Count",
                    height=350,
                    margin=dict(t=40, b=0, l=0, r=0),
                    paper_bgcolor='rgba(22, 27, 34, 0.5)',
                    font=dict(color='#e6edf3'),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 2: Demand Trends ==========
    with tab2:
        st.markdown('<div class="tab-title">Demand Trends Over Time</div>', unsafe_allow_html=True)
        
        # Time series
        df_sorted = filtered_df.sort_values('timestamp')
        
        if 'usage_units' in df_sorted.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_sorted['timestamp'],
                y=df_sorted['usage_units'],
                mode='lines',
                name='Usage Units',
                line=dict(color='#58a6ff', width=2),
                fill='tozeroy',
                fillcolor='rgba(88, 166, 255, 0.1)'
            ))
            
            if 'provisioned_capacity' in df_sorted.columns:
                fig.add_trace(go.Scatter(
                    x=df_sorted['timestamp'],
                    y=df_sorted['provisioned_capacity'],
                    mode='lines',
                    name='Provisioned Capacity',
                    line=dict(color='#79c0ff', width=2, dash='dash'),
                    fill=None
                ))
            
            fig.update_layout(
                title="Usage vs Provisioned Capacity (Demand Trends)",
                xaxis_title="Date",
                yaxis_title="Units",
                height=400,
                hovermode='x unified',
                paper_bgcolor='rgba(22, 27, 34, 0.5)',
                font=dict(color='#e6edf3'),
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Daily/service breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Usage by Service Type (Over Time)")
            if 'service_type' in filtered_df.columns and 'usage_units' in filtered_df.columns:
                service_over_time = filtered_df.groupby(['timestamp', 'service_type'])['usage_units'].sum().reset_index()
                
                fig = px.bar(
                    service_over_time,
                    x='timestamp',
                    y='usage_units',
                    color='service_type',
                    barmode='stack',
                    height=350,
                    title='Stacked Usage by Service Type'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(22, 27, 34, 0.5)',
                    font=dict(color='#e6edf3'),
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Cost Trend")
            if 'cost_usd' in filtered_df.columns:
                cost_over_time = filtered_df.sort_values('timestamp').groupby('timestamp')['cost_usd'].sum().reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cost_over_time['timestamp'],
                    y=cost_over_time['cost_usd'],
                    mode='lines',
                    line=dict(color='#1f6feb', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(31, 111, 235, 0.1)',
                    name='Cost'
                ))
                
                fig.update_layout(
                    title="Daily Cost Trend",
                    xaxis_title="Date",
                    yaxis_title="Cost ($)",
                    height=350,
                    hovermode='x unified',
                    paper_bgcolor='rgba(22, 27, 34, 0.5)',
                    font=dict(color='#e6edf3'),
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 3: Regional Analysis ==========
    with tab3:
        st.markdown('<div class="tab-title">Regional Capacity Breakdown</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top Regions by Cost")
            if 'region' in filtered_df.columns and 'cost_usd' in filtered_df.columns:
                region_cost = filtered_df.groupby('region')['cost_usd'].sum().sort_values(ascending=False).head(10)
                
                fig = go.Figure(data=[go.Bar(
                    y=region_cost.index,
                    x=region_cost.values,
                    orientation='h',
                    marker=dict(color='#58a6ff')
                )])
                
                fig.update_layout(
                    title="Top 10 Regions by Cost",
                    xaxis_title="Cost ($)",
                    yaxis_title="Region",
                    height=350,
                    paper_bgcolor='rgba(22, 27, 34, 0.5)',
                    font=dict(color='#e6edf3'),
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Utilization by Region")
            if 'region' in filtered_df.columns and 'provisioned_capacity' in filtered_df.columns:
                region_util = filtered_df.groupby('region').apply(
                    lambda x: (x['usage_units'].sum() / x['provisioned_capacity'].sum() * 100)
                    if x['provisioned_capacity'].sum() > 0 else 0
                ).sort_values(ascending=False).head(10)
                
                fig = go.Figure(data=[go.Bar(
                    y=region_util.index,
                    x=region_util.values,
                    orientation='h',
                    marker=dict(color='#79c0ff'),
                    name='Utilization %'
                )])
                
                fig.add_vline(x=80, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    title="Avg Utilization % by Region",
                    xaxis_title="Utilization %",
                    yaxis_title="Region",
                    height=350,
                    paper_bgcolor='rgba(22, 27, 34, 0.5)',
                    font=dict(color='#e6edf3'),
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Regional summary table
        st.markdown("### Regional Summary Table")
        
        if 'region' in filtered_df.columns:
            regional_summary = filtered_df.groupby('region').agg({
                'usage_units': ['sum', 'mean'],
                'provisioned_capacity': ['sum', 'mean'],
                'cost_usd': 'sum'
            }).round(2)
            
            regional_summary.columns = ['Total Usage', 'Avg Usage', 'Total Capacity', 'Avg Capacity', 'Total Cost']
            regional_summary = regional_summary.sort_values('Total Cost', ascending=False)
            
            st.dataframe(regional_summary, use_container_width=True)
    
    # ========== TAB 4: Model & Forecast ==========
    with tab4:
        st.markdown('<div class="tab-title">XGBoost Model Status & Forecast</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Feature Importance (Top 15)")
            importance_df = load_model_feature_importance()
            if importance_df is not None and not importance_df.empty:
                top_importance = importance_df.head(15).sort_values('Importance', ascending=True)

                fig = go.Figure(data=[go.Bar(
                    x=top_importance['Importance'],
                    y=top_importance['Feature'],
                    orientation='h',
                    marker=dict(color='#58a6ff')
                )])

                fig.update_layout(
                    title="Feature Importance Scores",
                    xaxis_title="Importance Score",
                    height=400,
                    paper_bgcolor='rgba(22, 27, 34, 0.5)',
                    font=dict(color='#e6edf3'),
                    template='plotly_dark'
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature importance unavailable. Ensure model artifact exists and supports feature_importances_.")
        
        with col2:
            st.markdown("### XGBoost Model Status")
            model_path = Path("artifacts/tuned_xgboost_model.pkl")
            model_status = "LOADED" if model_path.exists() else "NOT FOUND"
            st.markdown(f"""
            {'✅' if model_path.exists() else '❌'} **{model_status}**

            **Algorithm:** XGBoost Regressor  
            **Target:** usage_units (demand)  
            **Model Path:** {model_path}  
            **Source Data:** {source_file}
            """)

            if latest_metrics:
                st.markdown("**Latest Monitoring Snapshot**")
                st.write(f"- Batch ID: {latest_metrics.get('batch_id', 'N/A')}")
                st.write(f"- RMSE: {latest_metrics.get('rmse', 0):.3f}")
                st.write(f"- MAE: {latest_metrics.get('mae', 0):.3f}")
                st.write(f"- Drift: {'Yes' if latest_metrics.get('is_drifted') else 'No'}")
        
        st.markdown("---")
        
        # Forecast
        st.markdown("### Actual vs Forecast (Available Records)")

        if 'usage_units' in filtered_df.columns:
            forecast_frame = filtered_df.sort_values('timestamp').copy()
            if 'predicted_usage' not in forecast_frame.columns:
                forecast_frame['predicted_usage'] = np.nan

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_frame['timestamp'],
                y=forecast_frame['usage_units'],
                mode='lines',
                name='Actual / Usage',
                line=dict(color='#58a6ff', width=2)
            ))

            if forecast_frame['predicted_usage'].notna().any():
                fig.add_trace(go.Scatter(
                    x=forecast_frame['timestamp'],
                    y=forecast_frame['predicted_usage'],
                    mode='lines',
                    name='Predicted Usage',
                    line=dict(color='#1f6feb', width=2, dash='dash')
                ))
            else:
                st.info("No predicted_usage column found in current source. Run batch prediction for forecast overlay.")

            fig.update_layout(
                title="Demand Curve from Real Data",
                xaxis_title="Date",
                yaxis_title="Usage Units",
                height=400,
                hovermode='x unified',
                paper_bgcolor='rgba(22, 27, 34, 0.5)',
                font=dict(color='#e6edf3'),
                template='plotly_dark'
            )

            st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 5: Risk Alerts ==========
    with tab5:
        st.markdown('<div class="tab-title">Risk Alerts & Anomalies</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### High Risk Events (>80% Utilization)")
            
            if 'provisioned_capacity' in filtered_df.columns and 'usage_units' in filtered_df.columns:
                utilization = (filtered_df['usage_units'] / filtered_df['provisioned_capacity'] * 100)
                high_risk = filtered_df[utilization > 80].copy()
                high_risk['utilization_pct'] = utilization[utilization > 80].values
                
                if len(high_risk) > 0:
                    high_risk_summary = high_risk.groupby('region').size().sort_values(ascending=False)
                    
                    fig = go.Figure(data=[go.Bar(
                        x=high_risk_summary.values,
                        y=high_risk_summary.index,
                        orientation='h',
                        marker=dict(color='#ff7b72')
                    )])
                    
                    fig.update_layout(
                        title="High Risk Events by Region",
                        xaxis_title="Count",
                        yaxis_title="Region",
                        height=300,
                        paper_bgcolor='rgba(22, 27, 34, 0.5)',
                        font=dict(color='#e6edf3'),
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(f"⚠️ **Total High Risk Records:** {len(high_risk)}")
                else:
                    st.info("✅ No high risk events detected")
        
        with col2:
            st.markdown("### Low Utilization Events (<30%)")
            
            if 'provisioned_capacity' in filtered_df.columns and 'usage_units' in filtered_df.columns:
                utilization = (filtered_df['usage_units'] / filtered_df['provisioned_capacity'] * 100)
                low_util = filtered_df[utilization < 30].copy()
                low_util['utilization_pct'] = utilization[utilization < 30].values
                
                if len(low_util) > 0:
                    low_util_summary = low_util.groupby('region').size().sort_values(ascending=False)
                    
                    fig = go.Figure(data=[go.Bar(
                        x=low_util_summary.values,
                        y=low_util_summary.index,
                        orientation='h',
                        marker=dict(color='#ffa657')
                    )])
                    
                    fig.update_layout(
                        title="Underutilization Events by Region",
                        xaxis_title="Count",
                        yaxis_title="Region",
                        height=300,
                        paper_bgcolor='rgba(22, 27, 34, 0.5)',
                        font=dict(color='#e6edf3'),
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(f"⚠️ **Total Underutilized Records:** {len(low_util)}")
                else:
                    st.info("✅ No underutilization events detected")
        
        st.markdown("---")
        
        # Risk summary
        st.markdown("### Risk Summary Table")
        
        if 'region' in filtered_df.columns and 'service_type' in filtered_df.columns:
            risk_data = []
            
            for region in filtered_df['region'].unique():
                for service in filtered_df['service_type'].unique():
                    subset = filtered_df[(filtered_df['region'] == region) & (filtered_df['service_type'] == service)]
                    
                    if len(subset) > 0 and 'provisioned_capacity' in subset.columns:
                        util = (subset['usage_units'] / subset['provisioned_capacity'] * 100)
                        high_risk_count = (util > 80).sum()
                        low_util_count = (util < 30).sum()
                        
                        risk_data.append({
                            'Region': region,
                            'Service': service,
                            'High Risk (>80%)': high_risk_count,
                            'Low Util (<30%)': low_util_count,
                            'Total Records': len(subset),
                            'Risk Score': (high_risk_count * 10 + low_util_count * 5) / max(1, len(subset))
                        })
            
            risk_df = pd.DataFrame(risk_data).sort_values('Risk Score', ascending=False)
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
    **Data Source:** {source_file}  
    **Records:** {len(df)} total | {len(filtered_df)} filtered  
    **System:** Azure Demand Forecasting M4
    """)


if __name__ == "__main__":
    main()
