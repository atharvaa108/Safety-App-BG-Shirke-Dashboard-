import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="BG Shirke Safety Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# iOS-inspired Glassmorphism CSS
st.markdown("""
<style>
    /* Import SF Pro font (iOS style) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 20px;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom headers */
    h1, h2, h3 {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: scale(1.05);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
    }
    
    /* DataFrame styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
    }
    
    /* Metric value styling */
    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 32px !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    # Replace with your actual file path
    df = pd.read_csv('BGShirke_safety_actionables.csv')
    return df

# Load the data
try:
    df = load_data()
except:
    st.error("⚠️ Please upload the BGShirke_safety_actionables.csv file")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ Safety Dashboard")
    st.markdown("---")
    
    # Project filter
    projects = ['All Projects'] + sorted(df['Project Name'].unique().tolist())
    selected_project = st.selectbox("📍 Select Project", projects)
    
    # Risk level filter
    risk_levels = ['All Levels'] + sorted(df['Risk Level'].unique().tolist())
    selected_risk = st.selectbox("⚠️ Risk Level", risk_levels)
    
    # Status filter
    statuses = ['All Status'] + sorted(df['Status'].unique().tolist())
    selected_status = st.selectbox("📊 Status", statuses)
    
    st.markdown("---")
    st.markdown("### 📅 Data Info")
    st.info(f"Total Records: {len(df)}")
    st.info(f"Projects: {df['Project Name'].nunique()}")

# Filter data
filtered_df = df.copy()
if selected_project != 'All Projects':
    filtered_df = filtered_df[filtered_df['Project Name'] == selected_project]
if selected_risk != 'All Levels':
    filtered_df = filtered_df[filtered_df['Risk Level'] == selected_risk]
if selected_status != 'All Status':
    filtered_df = filtered_df[filtered_df['Status'] == selected_status]

# Main Dashboard
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🛡️ BG Shirke Safety Analytics Dashboard</h1>", unsafe_allow_html=True)

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_observations = len(filtered_df)
    st.metric("Total Observations", total_observations, delta=None)

with col2:
    high_risk = len(filtered_df[filtered_df['Risk Level'] == 'High'])
    st.metric("High Risk", high_risk, delta=f"{(high_risk/total_observations*100):.1f}%" if total_observations > 0 else "0%")

with col3:
    open_issues = len(filtered_df[filtered_df['Status'] == 'OPEN'])
    st.metric("Open Issues", open_issues, delta=f"-{open_issues}" if open_issues > 0 else "0")

with col4:
    closed_rate = len(filtered_df[filtered_df['Status'] == 'CLOSED']) / total_observations * 100 if total_observations > 0 else 0
    st.metric("Closure Rate", f"{closed_rate:.1f}%", delta=f"+{closed_rate:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Charts Row 1
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Risk Distribution by Project")
    risk_by_project = filtered_df.groupby(['Project Name', 'Risk Level']).size().reset_index(name='Count')
    
    fig1 = px.bar(
        risk_by_project,
        x='Project Name',
        y='Count',
        color='Risk Level',
        color_discrete_map={'High': '#FF6B6B', 'Medium': '#FFB84D', 'Low': '#4ECB71'},
        barmode='group',
        template='plotly_dark'
    )
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### 🎯 Status Distribution")
    status_data = filtered_df['Status'].value_counts().reset_index()
    status_data.columns = ['Status', 'Count']
    
    colors = {'OPEN': '#FF6B6B', 'CLOSED': '#4ECB71', 'IN PROGRESS': '#FFB84D'}
    status_data['Color'] = status_data['Status'].map(colors)
    
    fig2 = px.pie(
        status_data,
        values='Count',
        names='Status',
        color='Status',
        color_discrete_map=colors,
        hole=0.5,
        template='plotly_dark'
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        showlegend=True
    )
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)

# Charts Row 2 - Only show when specific project is selected
if selected_project != 'All Projects':
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏗️ Main Area of Observation")
        area_data = filtered_df['Main Area of Observation'].value_counts().head(10).reset_index()
        area_data.columns = ['Area', 'Count']
        
        fig3 = px.bar(
            area_data,
            x='Count',
            y='Area',
            orientation='h',
            template='plotly_dark',
            color='Count',
            color_continuous_scale='Sunset'
        )
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=False),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown("### 📂 Category Breakdown")
        # Split categories by semicolon and explode to count each category separately
        category_data = filtered_df['Category'].str.split(';').explode()
        def space_hunter(x):
            return x.strip()
        category_data = category_data.apply(space_hunter).value_counts().reset_index(name = 'Count')
        category_data.columns = ['Category', 'Count']
        
        fig4 = go.Figure(data=[go.Bar(
            x=category_data['Category'],
            y=category_data['Count'],
            marker=dict(
                color=category_data['Count'],
                colorscale='Viridis',
                showscale=False
            ),
            text=category_data['Count'],
            textposition='outside'
        )])
        
        fig4.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            height=400
        )
        st.plotly_chart(fig4, use_container_width=True)
else:
    # Show only Category Breakdown when All Projects is selected
    st.markdown("### 📂 Category Breakdown")
    # Split categories by semicolon and explode to count each category separately
    category_data = filtered_df['Category'].str.split(';').explode()
    def space_hunter(x):
        return x.strip()
    category_data = category_data.apply(space_hunter).value_counts().reset_index(name = 'Count')
    category_data.columns = ['Category', 'Count']
    
    fig4 = go.Figure(data=[go.Bar(
        x=category_data['Category'],
        y=category_data['Count'],
        marker=dict(
            color=category_data['Count'],
            colorscale='Viridis',
            showscale=False
        ),
        text=category_data['Count'],
        textposition='outside'
    )])
    
    fig4.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )
    st.plotly_chart(fig4, use_container_width=True)

# High Risk Issues Section - Only show when specific project is selected
if selected_project != 'All Projects':
    st.markdown("### 🚨 Critical High-Risk Issues Requiring Attention")

    high_risk_open = filtered_df[(filtered_df['Risk Level'] == 'High') & (filtered_df['Status'] == 'OPEN')]

    if len(high_risk_open) > 0:
        st.markdown(f"**⚠️ {len(high_risk_open)} Open High-Risk Issues Found**")
        
        # Display summary by project
        high_risk_summary = high_risk_open.groupby('Project Name').size().reset_index(name='Open High-Risk Count')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show details in expandable sections
            for idx, row in high_risk_open.iterrows():
                with st.expander(f"🔴 {row['Project Name']} - {row['Safety Observation ID']}"):
                    st.markdown(f"**Category:** {row['Category']}")
                    st.markdown(f"**Location:** {row['Location of Breach']}")
                    st.markdown(f"**Main Area:** {row['Main Area of Observation']}")
                    st.markdown(f"**Details:** {row['Details']}")
                    st.markdown(f"**Recommendations:** {row['Recommendations Given']}")
        
        with col2:
            # Summary chart
            fig5 = px.bar(
                high_risk_summary,
                x='Project Name',
                y='Open High-Risk Count',
                template='plotly_dark',
                color='Open High-Risk Count',
                color_continuous_scale='Reds'
            )
            fig5.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig5, use_container_width=True)
    else:
        st.success("✅ No open high-risk issues! Great job!")

# Detailed Data Table
st.markdown("### 📋 Detailed Safety Observations")

display_columns = ['Project Name', 'Safety Observation ID', 'Risk Level', 'Status', 
                   'Category', 'Main Area of Observation', 'Details', 'Recommendations Given']

st.dataframe(
    filtered_df[display_columns].style.apply(
        lambda x: ['background-color: rgba(255,107,107,0.3)' if v == 'High' 
                   else 'background-color: rgba(255,184,77,0.3)' if v == 'Medium'
                   else 'background-color: rgba(78,203,113,0.3)' if v == 'Low'
                   else '' for v in x], 
        subset=['Risk Level']
    ),
    use_container_width=True,
    height=400
)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: rgba(255,255,255,0.7);'>BG Shirke Safety Analytics Dashboard © 2025</p>",
    unsafe_allow_html=True
)