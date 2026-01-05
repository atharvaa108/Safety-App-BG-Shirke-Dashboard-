import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Safety Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# iOS-inspired Glassmorphism CSS (optimized)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
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
    
    .metric-card:hover { transform: translateY(-5px); }
    #MainMenu, footer, header { visibility: hidden; }
    h1, h2, h3 { color: white !important; font-weight: 600 !important; }
    
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stSidebar"] * { color: white !important; }
    
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
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
    }
    
    .dataframe {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
    }
    
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

# Define risk level severity order and colors
RISK_SEVERITY_ORDER = [
    'Life Threat', 'Life threat', 'LIFE THREAT',
    'Critical', 'CRITICAL',
    'Severe', 'SEVERE',
    'High', 'HIGH',
    'Medium', 'MEDIUM',
    'Low', 'LOW',
    'Minor', 'MINOR',
    'Negligible', 'NEGLIGIBLE'
]

RISK_COLOR_MAP = {
    'Life Threat': '#8B0000', 'Life threat': '#8B0000', 'LIFE THREAT': '#8B0000',
    'Critical': '#B22222', 'CRITICAL': '#B22222',
    'Severe': '#DC143C', 'SEVERE': '#DC143C',
    'High': '#FF6B6B', 'HIGH': '#FF6B6B',
    'Medium': '#FFB84D', 'MEDIUM': '#FFB84D',
    'Low': '#4ECB71', 'LOW': '#4ECB71',
    'Minor': '#90EE90', 'MINOR': '#90EE90',
    'Negligible': '#98FB98', 'NEGLIGIBLE': '#98FB98'
}

def get_risk_order(risk_levels):
    """Create ordered list of risk levels based on severity"""
    # Get unique risk levels from data
    unique_risks = list(set(risk_levels))
    
    # Sort based on RISK_SEVERITY_ORDER
    ordered_risks = []
    for severity in RISK_SEVERITY_ORDER:
        if severity in unique_risks:
            ordered_risks.append(severity)
    
    # Add any risk levels not in our predefined order at the end
    for risk in unique_risks:
        if risk not in ordered_risks:
            ordered_risks.append(risk)
    
    return ordered_risks

def is_valid_category(category_value):
    """Check if a category value is meaningful"""
    if pd.isna(category_value):
        return False
    
    category_str = str(category_value).strip().lower()
    
    # Invalid category indicators
    invalid_indicators = [
        'not applicable', 'n/a', 'na', 'not available',
        '-', '--', '---', '_', 'none', 'nil', 'null'
    ]
    
    # Check if empty or too short
    if len(category_str) < 2:
        return False
    
    # Check if matches invalid indicators
    if category_str in invalid_indicators:
        return False
    
    return True

def has_valid_categories(df):
    """Check if dataframe has any valid categories"""
    if 'Category' not in df.columns:
        return False
    
    # Check if any row has a valid category
    return df['Category'].apply(is_valid_category).any()

# Optimized data cleaning function
def clean_data(df):
    """Clean the uploaded dataset with robust error handling"""
    try:
        df = df.copy()  # Work on a copy to avoid modifying original
        
        # Extract numeric ID from SA No
        if 'SA No' in df.columns:
            df['id'] = df['SA No'].astype(str).str.extract(r'(\d+)')[0]
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
        
        # Sort by Project Name and id
        sort_cols = []
        if 'Project Name' in df.columns:
            sort_cols.append('Project Name')
        if 'id' in df.columns:
            sort_cols.append('id')
        
        if sort_cols:
            df = df.sort_values(by=sort_cols, ascending=True)
        
        # Clean observation area columns
        text_cols = ['Main Area Of Observation', 'Sub Area Of Observation']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('\n', ' ', regex=False).str.strip()
        
        # Parse and create datetime columns
        if 'Created Date / Time' in df.columns:
            df['Created Date / Time'] = pd.to_datetime(df['Created Date / Time'], errors='coerce')
            df['Calculated Month'] = df['Created Date / Time'].dt.month
            df['Calculated Year'] = df['Created Date / Time'].dt.year
        
        # Clean category column
        if 'Category' in df.columns:
            df['Category'] = df['Category'].astype(str).str.strip()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    except Exception as e:
        st.error(f"Error during data cleaning: {str(e)}")
        return df

# Optimized data loading with proper encoding detection
@st.cache_data(ttl=3600)
def load_data_from_file(file_path):
    """Load data from file with automatic encoding detection"""
    encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if len(df.columns) > 5:  # Sanity check
                df = clean_data(df)
                return df, encoding
        except:
            continue
    
    raise ValueError("Could not read file with any supported encoding")

def read_uploaded_file(uploaded_file):
    """Read uploaded file with proper encoding handling"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            # Try multiple encodings for CSV
            encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    # Validate the dataframe
                    if len(df.columns) > 5 and 'SA No' in df.columns:
                        st.success(f"✅ CSV file read successfully with {encoding} encoding!")
                        return df, encoding
                except Exception as e:
                    continue
            
            raise ValueError("Could not read CSV file with any supported encoding")
            
        elif file_extension in ['xlsx', 'xls']:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            st.success("✅ Excel file read successfully!")
            return df, 'excel'
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None, None

# Create color-coded dataframe style
def color_risk_level(df):
    """Apply color coding to risk level column"""
    def highlight_risk(val):
        # Use the color map if available
        for risk_name, color in RISK_COLOR_MAP.items():
            if str(val).strip() == risk_name:
                # Convert hex to rgba
                return f'background-color: {color}40'  # Add transparency
        return ''
    
    return df.style.applymap(highlight_risk, subset=['Risk Level'])

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'encoding' not in st.session_state:
    st.session_state.encoding = None

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload your Safety Dataset (CSV or Excel)", 
    type=['csv', 'xlsx', 'xls'],
    help="Upload your safety observation dataset to visualize analytics"
)

if uploaded_file:
    with st.spinner('📊 Processing your file...'):
        df_raw, encoding = read_uploaded_file(uploaded_file)
        
        if df_raw is not None:
            # Store raw data in session state
            st.session_state.raw_df = df_raw.copy()
            
            # Clean the data
            with st.spinner('🔄 Cleaning and validating data...'):
                df = clean_data(df_raw)
                
                # Validate required columns
                required_cols = ['Project Name', 'Risk Level', 'Status']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.stop()
                
                st.session_state.df = df
                st.session_state.file_name = uploaded_file.name.rsplit('.', 1)[0]
                st.session_state.encoding = encoding
                
                st.success(f"✅ Data processed successfully! ({len(df)} records)")
                
                # Show data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"📊 Rows: {len(df)}")
                with col2:
                    st.info(f"📋 Columns: {len(df.columns)}")
                with col3:
                    st.info(f"🔤 Encoding: {encoding}")
                
                # Download cleaned data
                cleaned_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Data (CSV)",
                    data=cleaned_csv,
                    file_name=f"Cleaned_{st.session_state.file_name}.csv",
                    mime="text/csv",
                )

# Check if we have data to display
if st.session_state.df is None:
    st.warning("⚠️ Please upload a dataset to get started!")
    st.info("💡 **Tip:** Your CSV should contain columns like 'Project Name', 'Risk Level', 'Status', etc.")
    st.stop()

df = st.session_state.df

# Get risk level ordering for the dataset
risk_order = get_risk_order(df['Risk Level'].dropna().unique())

# Sidebar filters
with st.sidebar:
    st.markdown("### 🛡️ Safety Dashboard")
    st.markdown("---")
    
    # Project filter
    projects = ['All Projects'] + sorted(df['Project Name'].dropna().unique().tolist())
    selected_project = st.selectbox("📍 Select Project", projects)
    
    # Risk level filter - ordered by severity
    risk_levels = ['All Levels'] + risk_order
    selected_risk = st.selectbox("⚠️ Risk Level", risk_levels)
    
    # Status filter
    statuses = ['All Status'] + sorted(df['Status'].dropna().unique().tolist())
    selected_status = st.selectbox("📊 Status", statuses)
    
    st.markdown("---")
    st.markdown("### 📅 Data Info")
    st.info(f"Total Records: {len(df):,}")
    st.info(f"Projects: {df['Project Name'].nunique()}")
    
    if 'Calculated Month' in df.columns:
        valid_months = df['Calculated Month'].dropna()
        if len(valid_months) > 0:
            st.info(f"Date Range: {int(valid_months.min())}-{int(valid_months.max())}")

# Apply filters
filtered_df = df.copy()
if selected_project != 'All Projects':
    filtered_df = filtered_df[filtered_df['Project Name'] == selected_project]
if selected_risk != 'All Levels':
    filtered_df = filtered_df[filtered_df['Risk Level'] == selected_risk]
if selected_status != 'All Status':
    filtered_df = filtered_df[filtered_df['Status'] == selected_status]

# Get the highest severity risk level for metrics
highest_risk = risk_order[0] if risk_order else 'High'

# Main Dashboard Header
st.markdown(
    f"<h1 style='text-align: center; margin-bottom: 30px;'>🛡️ {st.session_state.file_name} Dashboard</h1>", 
    unsafe_allow_html=True
)

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

total_observations = len(filtered_df)
high_risk = len(filtered_df[filtered_df['Risk Level'] == highest_risk])
high_risk_count = len(filtered_df[filtered_df['Risk Level'].isin(['High', 'HIGH'])])
open_issues = len(filtered_df[filtered_df['Status'] == 'OPEN'])
closed_count = len(filtered_df[filtered_df['Status'] == 'CLOSED'])
closed_rate = (closed_count / total_observations * 100) if total_observations > 0 else 0

with col1:
    st.metric("Total Observations", f"{total_observations:,}")

with col2:
    st.metric(
        f"{highest_risk} Risk", 
        high_risk, 
        delta=f"{(high_risk/total_observations*100):.1f}%" if total_observations > 0 else "0%"
    )

with col3:
    st.metric(
        "High Risk Issues", 
        high_risk_count, 
        delta=f"{(high_risk_count/total_observations*100):.1f}%" if total_observations > 0 else "0%"
    )

with col4:
    st.metric("Open Issues", open_issues, delta=f"-{open_issues}" if open_issues > 0 else "0")

with col5:
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
        color_discrete_map=RISK_COLOR_MAP,
        barmode='group',
        template='plotly_dark',
        category_orders={'Risk Level': risk_order}
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

# Category Analysis Section - Only show if valid categories exist
if has_valid_categories(filtered_df):
    st.markdown("### 📂 Category Breakdown")
    
    # Filter only valid categories
    valid_categories = filtered_df[filtered_df['Category'].apply(is_valid_category)].copy()
    
    if len(valid_categories) > 0:
        # Split and clean categories
        category_data = valid_categories['Category'].str.split(',').explode()
        category_data = category_data.str.strip()
        
        # Filter out invalid categories after split
        category_data = category_data[category_data.apply(is_valid_category)]
        category_data = category_data.value_counts().reset_index()
        category_data.columns = ['Category', 'Count']
        
        if len(category_data) > 0:
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
            
            # Category details selector
            st.markdown("### 🔍 Category Details")
            selected_category = st.selectbox(
                "Select a category to view details:",
                options=['-- Select Category --'] + category_data['Category'].tolist()
            )
            
            if selected_category != '-- Select Category --':
                category_mask = valid_categories['Category'].str.contains(selected_category, case=False, na=False)
                category_details = valid_categories[category_mask][
                    ['Project Name', 'Category', 'Details', 'Risk Level', 
                     'Main Area Of Observation', 'Sub Area Of Observation', 'Status']
                ].copy()
                
                if len(category_details) > 0:
                    st.markdown(f"**📊 Showing {len(category_details)} observations for: {selected_category}**")
                    
                    st.dataframe(
                        color_risk_level(category_details),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Category metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Projects", category_details['Project Name'].nunique())
                    
                    # Show metrics for each risk level present in the data
                    metric_cols = [col2, col3, col4]
                    for i, risk_level in enumerate(risk_order[:3]):  # Show top 3 severity levels
                        if i < len(metric_cols) and risk_level in category_details['Risk Level'].values:
                            with metric_cols[i]:
                                st.metric(f"{risk_level} Risk", 
                                        len(category_details[category_details['Risk Level'] == risk_level]))
                else:
                    st.info("No data found for this category.")
        else:
            st.info("ℹ️ No valid categories available in the dataset. Categories contain only placeholder values (e.g., 'Not Applicable', '-', etc.)")
    else:
        st.info("ℹ️ No valid categories available in the dataset. Categories contain only placeholder values (e.g., 'Not Applicable', '-', etc.)")
else:
    st.info("ℹ️ No valid categories available in the dataset. Categories contain only placeholder values (e.g., 'Not Applicable', '-', etc.)")

# Project-specific analysis
if selected_project != 'All Projects':
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏗️ Main Area of Observation")
        if 'Main Area Of Observation' in filtered_df.columns:
            area_data = filtered_df.groupby(['Main Area Of Observation','Risk Level']).size().reset_index()
            area_data.columns = ['Area', 'Risk Level','Count']
            
            fig3 = px.bar(
                area_data,
                x='Count',
                y='Area',
                orientation='h',
                template='plotly_dark',
                color='Risk Level',
                color_discrete_map=RISK_COLOR_MAP,
                category_orders={'Risk Level': risk_order}
            )
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=False),
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Check if Life Threat exists in the data
        life_threat_variations = ['Life Threat', 'Life threat', 'LIFE THREAT']
        has_life_threat = any(lt in filtered_df['Risk Level'].values for lt in life_threat_variations)
        
        if has_life_threat:
            # Show both Life Threat and High risk open issues
            st.markdown("### 🚨 Critical Risk Issues (Life Threat & High)")
            critical_risk_open = filtered_df[
                (filtered_df['Risk Level'].isin(life_threat_variations + ['High', 'HIGH'])) & 
                (filtered_df['Status'] == 'OPEN')
            ]
            risk_label = "Life Threat & High Risk"
        else:
            # Show only High risk open issues
            st.markdown("### 🚨 High-Risk Issues")
            critical_risk_open = filtered_df[
                (filtered_df['Risk Level'].isin(['High', 'HIGH'])) & 
                (filtered_df['Status'] == 'OPEN')
            ]
            risk_label = "High Risk"

        if len(critical_risk_open) > 0:
            st.markdown(f"**⚠️ {len(critical_risk_open)} Open {risk_label} Issues**")
            
            # Sort by risk level (Life Threat first if it exists)
            critical_risk_open = critical_risk_open.sort_values(
                by='Risk Level',
                key=lambda x: x.map({lt: 0 for lt in life_threat_variations} | {'High': 1, 'HIGH': 1})
            )
            
            for idx, row in critical_risk_open.head(5).iterrows():
                risk_emoji = "🔴" if row['Risk Level'] in life_threat_variations else "🟠"
                with st.expander(f"{risk_emoji} {row.get('Safety Observation ID', 'N/A')} - {row['Risk Level']}"):
                    st.markdown(f"**Category:** {row.get('Category', 'N/A')}")
                    st.markdown(f"**Location:** {row.get('Location Of Breach', 'N/A')}")
                    st.markdown(f"**Details:** {row.get('Details', 'N/A')}")
        else:
            st.success(f"✅ No open {risk_label.lower()} issues!")

# Monthly Analysis
if 'Calculated Month' in filtered_df.columns and selected_project != 'All Projects':
    # Get raw data from session state
    if st.session_state.raw_df is not None:
        # Create a clean copy with calculated month
        monthly_raw_df = st.session_state.raw_df.copy()
        monthly_raw_df['Calculated Month'] = pd.to_datetime(monthly_raw_df['Created Date / Time'], errors='coerce').dt.month
        
        # Filter by selected project only
        monthly_raw_df = monthly_raw_df[monthly_raw_df['Project Name'] == selected_project]
        
        # Get only rows that have month data
        monthly_raw_df = monthly_raw_df[monthly_raw_df['Calculated Month'].notna()].copy()
        
        if len(monthly_raw_df) > 0:
            st.markdown("---")
            st.markdown("## 📅 Monthly Analysis")
            
            # Month names dictionary
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

            # Show which months have data
            available_months = sorted(monthly_raw_df['Calculated Month'].unique().astype(int).tolist())
            month_list = ', '.join([month_names.get(m, str(m)) for m in available_months])
            st.info(f"📊 Showing data for months: **{month_list}**")
            
            # Chart 1: Risk Distribution by Month
            st.markdown("### 📊 Risk Distribution Per Month")

            risk_month_data = monthly_raw_df.groupby(['Calculated Month','Risk Level']).size().reset_index(name="Count")
            
            fig_month1 = px.bar(
                risk_month_data,
                x='Calculated Month',
                y='Count',
                color='Risk Level',
                color_discrete_map=RISK_COLOR_MAP,
                barmode='group',
                template='plotly_dark',
                category_orders={'Risk Level': risk_order}
            )
            fig_month1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, title='Month'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                height=400
            )
            st.plotly_chart(fig_month1, use_container_width=True)
        
            col1, col2 = st.columns(2)
            
            # Chart 2: Total Risks per Month
            with col1:
                st.markdown("### 📈 Total Risks per Month")
                total_per_month = monthly_raw_df.groupby('Calculated Month').size().reset_index(name='Total Risks')
                
                fig_month2 = px.line(
                    total_per_month,
                    x='Calculated Month',
                    y='Total Risks',
                    markers=True,
                    template='plotly_dark'
                )
                fig_month2.update_traces(line_color='#FFB84D', marker=dict(size=10))
                fig_month2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False, title='Month'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    height=400
                )
                st.plotly_chart(fig_month2, use_container_width=True)
            
            # Chart 3: Status per Month
            with col2:
                st.markdown("### 🎯 Status per Month")
                status_month_data = monthly_raw_df.groupby(['Calculated Month','Status']).size().reset_index(name="Count")
                
                fig_month3 = px.bar(
                    status_month_data,
                    x='Calculated Month',
                    y='Count',
                    color='Status',
                    color_discrete_map={'OPEN': '#FF6B6B', 'CLOSED': '#4ECB71', 'IN PROGRESS': '#FFB84D'},
                    barmode='group',
                    template='plotly_dark'
                )
                fig_month3.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False, title='Month'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    height=400
                )
                st.plotly_chart(fig_month3, use_container_width=True)
            
            # Detailed monthly table
            st.markdown("### 📋 Detailed Monthly Risk Information")
            
            detail_cols = ['Calculated Month','Risk Level','Type','Details',
                          'Main Area Of Observation','Sub Area Of Observation','Status']
            available_cols = [col for col in detail_cols if col in monthly_raw_df.columns]
            
            if available_cols:
                monthly_details = monthly_raw_df[available_cols].copy()
                monthly_details['Month Name'] = monthly_details['Calculated Month'].map(month_names)
                
                # Reorder columns to put Month Name first
                cols = ['Month Name'] + [col for col in monthly_details.columns if col != 'Month Name']
                monthly_details = monthly_details[cols].sort_values(by='Calculated Month')
                
                st.dataframe(
                    color_risk_level(monthly_details),
                    use_container_width=True,
                    height=400
                )
        else:
            st.info("📅 No monthly data available for the selected project.")
    else:
        st.info("📅 No raw data available. Please re-upload your file.")

# Detailed Data Table
st.markdown("---")
st.markdown("### 📋 Detailed Safety Observations")

display_columns = ['Project Name', 'Safety Observation ID', 'Risk Level', 'Status', 
                   'Category', 'Main Area Of Observation', 'Details']
display_columns = [col for col in display_columns if col in filtered_df.columns]

st.dataframe(
    color_risk_level(filtered_df[display_columns]),
    use_container_width=True,
    height=400
)

# Footer
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: rgba(255,255,255,0.7);'>{st.session_state.file_name} Dashboard © 2025</p>",
    unsafe_allow_html=True
)
