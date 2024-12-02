import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.community_analyzer import CommunityEngagementAnalyzer

# App setup
st.title("Community Engagement Analyzer")
st.write("Explore how customer segments interact with community channels based on engagement data.")

# Load or generate data
@st.cache_data
def load_data():
    analyzer = CommunityEngagementAnalyzer()
    data = analyzer.prepare_sample_data()
    X_scaled, processed_data = analyzer.preprocess_data(data)
    clusters = analyzer.perform_clustering(X_scaled, n_clusters=5)
    processed_data['Cluster'] = clusters

    # Map clusters to predefined customer segments
    cluster_to_segment = {
        0: "Indie Hackers and Solopreneurs",
        1: "Web Agencies and Freelancers",
        2: "Enterprise IT Teams",
        3: "Startups and SMBs",
        4: "No-Code Enthusiasts and Educators"
    }
    processed_data['Customer Segment'] = processed_data['Cluster'].map(cluster_to_segment)
    
    return processed_data

processed_data = load_data()

# Filters for customer segments
selected_segments = st.multiselect(
    "Select Customer Segments:",
    options=processed_data['Customer Segment'].unique(),
    default=processed_data['Customer Segment'].unique()
)

# Filter data based on selected segments
filtered_data = processed_data[processed_data['Customer Segment'].isin(selected_segments)]

# Predefined Metrics for Quick Insights
st.sidebar.subheader("Quick Insights")
if st.sidebar.button("View Top Engaged Segments"):
    top_segments = filtered_data.groupby("Customer Segment").sum()
    st.sidebar.write(top_segments.sort_values("discord_messages", ascending=False).head(3))

# Aggregated Data for Bar Chart
agg_data = filtered_data.melt(
    id_vars=['Customer Segment'],
    value_vars=['forum_posts', 'discord_messages', 'webinar_attendance',
                'office_hours_attendance', 'documentation_views', 'templates_used'],
    var_name='Community Channel',
    value_name='Engagement'
).groupby(['Customer Segment', 'Community Channel']).sum().reset_index()

# Visualization 1: Bar Chart
st.subheader("Total Engagement by Customer Segment and Community Channel")
bar_chart = px.bar(
    agg_data,
    x="Community Channel",
    y="Engagement",
    color="Customer Segment",
    barmode="group",
    title="Community Engagement Distribution",
    labels={"Community Channel": "Community Channel", "Engagement": "Total Engagement"}
)
st.plotly_chart(bar_chart)

# Scatter Plot for Metric Comparison
st.subheader("Explore Engagement Relationships")
x_axis = st.selectbox(
    "Select X-Axis Metric:",
    options=['forum_posts', 'discord_messages', 'webinar_attendance',
             'office_hours_attendance', 'documentation_views', 'templates_used'],
    index=1
)

y_axis = st.selectbox(
    "Select Y-Axis Metric:",
    options=['forum_posts', 'discord_messages', 'webinar_attendance',
             'office_hours_attendance', 'documentation_views', 'templates_used'],
    index=2
)

scatter_plot = px.scatter(
    filtered_data,
    x=x_axis,
    y=y_axis,
    color="Customer Segment",
    size="templates_used",  # Use bubble size to represent additional engagement
    hover_data=['forum_posts', 'discord_messages', 'webinar_attendance',
                'office_hours_attendance', 'documentation_views', 'templates_used'],
    title=f"Customer Segments: {x_axis.replace('_', ' ').title()} vs {y_axis.replace('_', ' ').title()}",
    labels={x_axis: x_axis.replace("_", " ").title(), y_axis: y_axis.replace("_", " ").title()}
)

# Add contextual benchmarks (average lines)
x_mean = filtered_data[x_axis].mean()
y_mean = filtered_data[y_axis].mean()
scatter_plot.add_shape(
    type="line",
    x0=x_mean,
    x1=x_mean,
    y0=0,
    y1=filtered_data[y_axis].max(),
    line=dict(color="red", dash="dash"),
    name="X-Axis Mean"
)
scatter_plot.add_shape(
    type="line",
    x0=0,
    x1=filtered_data[x_axis].max(),
    y0=y_mean,
    y1=y_mean,
    line=dict(color="blue", dash="dash"),
    name="Y-Axis Mean"
)
st.plotly_chart(scatter_plot)

# Heatmap for Community Channel Performance
st.subheader("Heatmap of Engagement by Segment and Channel")
heatmap_data = agg_data.pivot(
    index='Customer Segment',
    columns='Community Channel',
    values='Engagement'
).fillna(0)

heatmap = px.imshow(
    heatmap_data,
    title="Engagement Heatmap: Customer Segments vs Channels",
    labels={"x": "Community Channel", "y": "Customer Segment", "color": "Total Engagement"},
    color_continuous_scale="Viridis"
)
st.plotly_chart(heatmap)

# Key Insights
st.subheader("Key Insights")
top_segment = filtered_data.groupby("Customer Segment").sum().idxmax()['discord_messages']
st.write(f"The most active segment on Discord is **{top_segment}**.")

top_channel = agg_data.groupby("Community Channel").sum().idxmax()['Engagement']
st.write(f"The highest-engaging channel overall is **{top_channel}**.")
