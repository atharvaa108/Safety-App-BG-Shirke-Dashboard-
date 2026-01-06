import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# Optimized CSS - Fixed sidebar issue
st.markdown("""
<style>
    #MainMenu, footer { visibility: hidden; }
    
    /* Smooth transitions */
    .stMetric { 
        transition: all 0.3s ease;
    }
    
    /* Better spacing */
    .block-container {
        padding-top: 2rem;
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

STATUS_COLOR_MAP = {
    'OPEN': '#FF6B6B', 
    'CLOSED': '#4ECB71', 
    'IN PROGRESS': '#FFB84D'
}

MONTH_NAMES = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

# Utility Functions
def get_risk_order(risk_levels):
    """Create ordered list of risk levels based on severity"""
    unique_risks = list(set(risk_levels))
    ordered_risks = [risk for risk in RISK_SEVERITY_ORDER if risk in unique_risks]
    # Add any unrecognized risk levels at the end
    ordered_risks.extend([risk for risk in unique_risks if risk not in ordered_risks])
    return ordered_risks

def is_valid_category(category_value):
    """Check if a category value is meaningful"""
    if pd.isna(category_value):
        return False
    category_str = str(category_value).strip().lower()
    invalid_indicators = [
        'not applicable', 'n/a', 'na', 'not available',
        '-', '--', '---', '_', 'none', 'nil', 'null'
    ]
    return len(category_str) >= 2 and category_str not in invalid_indicators

def has_valid_categories(df):
    """Check if dataframe has any valid categories"""
    return 'Category' in df.columns and df['Category'].apply(is_valid_category).any()

def clean_data(df):
    """Clean the uploaded dataset with robust error handling"""
    try:
        df = df.copy()
        
        # Extract numeric ID from SA No if available
        if 'SA No' in df.columns:
            df['id'] = df['SA No'].astype(str).str.extract(r'(\d+)')[0]
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
        
        # Sort by Project Name and ID
        sort_cols = [col for col in ['Project Name', 'id'] if col in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols, ascending=True)
        
        # Clean text columns
        text_cols = ['Main Area Of Observation', 'Sub Area Of Observation']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('\n', ' ', regex=False).str.strip()
        
        # Parse dates and extract month/year
        if 'Created Date / Time' in df.columns:
            df['Created Date / Time'] = pd.to_datetime(df['Created Date / Time'], errors='coerce')
            df['Calculated Month'] = df['Created Date / Time'].dt.month
            df['Calculated Year'] = df['Created Date / Time'].dt.year
        
        # Clean category column
        if 'Category' in df.columns:
            df['Category'] = df['Category'].astype(str).str.strip()
        
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error during data cleaning: {str(e)}")
        return df

def read_uploaded_file(uploaded_file):
    """Read uploaded file with proper encoding handling"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            # Try multiple encodings for CSV files
            encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    if len(df.columns) > 5 and 'SA No' in df.columns:
                        return df, encoding
                except:
                    continue
            raise ValueError("Could not read CSV file with any supported encoding")
        
        elif file_extension in ['xlsx', 'xls']:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            return df, 'excel'
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None, None

def color_risk_level(df):
    """Apply color coding to risk level column"""
    def highlight_risk(val):
        color = RISK_COLOR_MAP.get(str(val).strip())
        return f'background-color: {color}40' if color else ''
    
    return df.style.applymap(highlight_risk, subset=['Risk Level'])

def create_chart_layout():
    """Common chart layout settings"""
    return dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )

# Initialize session state
for key in ['df', 'raw_df', 'file_name', 'encoding']:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================================
# WELCOME SCREEN - Show if no data is loaded
# ============================================================================
if st.session_state.df is None:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 30px;'>🛡️ Safety Analytics Dashboard</h1>", 
        unsafe_allow_html=True
    )
    
    st.markdown("### 📁 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose your CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your safety observation dataset"
    )
    
    if uploaded_file:
        with st.spinner('📊 Processing your file...'):
            df_raw, encoding = read_uploaded_file(uploaded_file)
            
            if df_raw is not None:
                st.session_state.raw_df = df_raw.copy()
                
                with st.spinner('🔄 Cleaning and validating data...'):
                    df = clean_data(df_raw)
                    
                    # Validate required columns
                    required_cols = ['Project Name', 'Risk Level', 'Status']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                        st.stop()
                    
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.file_name = uploaded_file.name.rsplit('.', 1)[0]
                    st.session_state.encoding = encoding
                    
                    st.success(f"✅ Data loaded successfully! ({len(df)} records)")
                    st.rerun()
    else:
        st.info("👆 Upload your dataset using the file picker above")
        st.markdown("---")
        st.markdown("### Expected Data Format")
        st.markdown("""
        Your dataset should include columns such as:
        - **Project Name** - Name of the project
        - **Risk Level** - Risk severity (e.g., High, Medium, Low)
        - **Status** - Current status (e.g., OPEN, CLOSED)
        - **Category** - Observation category
        - **Created Date / Time** - Timestamp of the observation
        - **Details** - Description of the observation
        """)
    
    st.stop()

# ============================================================================
# MAIN DASHBOARD - Data is loaded
# ============================================================================
df = st.session_state.df
risk_order = get_risk_order(df['Risk Level'].dropna().unique())

# Data Processing Summary
st.success(f"✅ Data processed successfully! ({len(df)} records)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info(f"📊 Rows: {len(df):,}")
with col2:
    st.info(f"📋 Columns: {len(df.columns)}")
with col3:
    st.info(f"🔤 Encoding: {st.session_state.encoding}")
with col4:
    cleaned_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned",
        data=cleaned_csv,
        file_name=f"Cleaned_{st.session_state.file_name}.csv",
        mime="text/csv"
    )

st.markdown("---")

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================
with st.sidebar:
    st.markdown("### 🛡️ Safety Dashboard")
    st.markdown("---")
    st.markdown("### 🎛️ Filters")
    
    # Project filter
    projects = ['All Projects'] + sorted(df['Project Name'].dropna().unique().tolist())
    selected_project = st.selectbox("📍 Select Project", projects)
    
    # Risk level filter
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
            st.info(f"Month Range: {int(valid_months.min())}-{int(valid_months.max())}")

# Apply filters
filtered_df = df.copy()
if selected_project != 'All Projects':
    filtered_df = filtered_df[filtered_df['Project Name'] == selected_project]
if selected_risk != 'All Levels':
    filtered_df = filtered_df[filtered_df['Risk Level'] == selected_risk]
if selected_status != 'All Status':
    filtered_df = filtered_df[filtered_df['Status'] == selected_status]

# ============================================================================
# MAIN DASHBOARD HEADER & KEY METRICS
# ============================================================================
st.markdown(
    f"<h1 style='text-align: center; margin-bottom: 30px;'>🛡️ {st.session_state.file_name} Dashboard</h1>", 
    unsafe_allow_html=True
)

# Calculate metrics
total_observations = len(filtered_df)
highest_risk = risk_order[0] if risk_order else 'High'
highest_risk_count = len(filtered_df[filtered_df['Risk Level'] == highest_risk])
high_risk_count = len(filtered_df[filtered_df['Risk Level'].isin(['High', 'HIGH'])])
open_issues = len(filtered_df[filtered_df['Status'] == 'OPEN'])
closed_count = len(filtered_df[filtered_df['Status'] == 'CLOSED'])
closed_rate = (closed_count / total_observations * 100) if total_observations > 0 else 0

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Observations", f"{total_observations:,}")
with col2:
    delta_pct = f"{(highest_risk_count/total_observations*100):.1f}%" if total_observations > 0 else "0%"
    st.metric(f"{highest_risk} Risk", highest_risk_count, delta=delta_pct)
with col3:
    delta_pct = f"{(high_risk_count/total_observations*100):.1f}%" if total_observations > 0 else "0%"
    st.metric("High Risk Issues", high_risk_count, delta=delta_pct)
with col4:
    st.metric("Open Issues", open_issues, delta=f"-{open_issues}" if open_issues > 0 else "0")
with col5:
    st.metric("Closure Rate", f"{closed_rate:.1f}%", delta=f"+{closed_rate:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# CHARTS ROW 1: Risk Distribution & Status
# ============================================================================
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
    fig1.update_layout(**create_chart_layout())
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### 🎯 Status Distribution")
    status_data = filtered_df['Status'].value_counts().reset_index()
    status_data.columns = ['Status', 'Count']
    fig2 = px.pie(
        status_data,
        values='Count',
        names='Status',
        color='Status',
        color_discrete_map=STATUS_COLOR_MAP,
        hole=0.5,
        template='plotly_dark'
    )
    layout = create_chart_layout()
    layout['showlegend'] = True
    fig2.update_layout(**layout)
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# CATEGORY ANALYSIS
# ============================================================================
if has_valid_categories(filtered_df):
    st.markdown("### 📂 Category Breakdown")
    valid_categories = filtered_df[filtered_df['Category'].apply(is_valid_category)].copy()
    
    if len(valid_categories) > 0:
        # Split and count categories
        if ',' in valid_categories['Category'].values:
            category_data = valid_categories['Category'].str.split(',').explode()
        else:
            category_data = valid_categories['Category'].str.split(';').explode()
        category_data = category_data.str.strip()
        category_data = category_data[category_data.apply(is_valid_category)]
        category_data = category_data.value_counts().reset_index()
        category_data.columns = ['Category', 'Count']
        
        if len(category_data) > 0:
            # Category bar chart
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
            fig4.update_layout(**create_chart_layout())
            fig4.update_xaxes(tickangle=-45)
            st.plotly_chart(fig4, use_container_width=True)
            
            # Category details selector
            st.markdown("### 🔍 Category Details")
            selected_category = st.selectbox(
                "Select a category to view details:",
                options=['-- Select Category --'] + category_data['Category'].tolist()
            )
            
            if selected_category != '-- Select Category --':
                category_mask = valid_categories['Category'].str.contains(
                    selected_category, case=False, na=False
                )
                display_cols = [
                    'Project Name', 'Category', 'Details', 'Risk Level',
                    'Main Area Of Observation', 'Sub Area Of Observation', 'Status'
                ]
                display_cols = [col for col in display_cols if col in valid_categories.columns]
                category_details = valid_categories[category_mask][display_cols].copy()
                
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
                    
                    metric_cols = [col2, col3, col4]
                    for i, risk_level in enumerate(risk_order[:3]):
                        if i < len(metric_cols) and risk_level in category_details['Risk Level'].values:
                            with metric_cols[i]:
                                count = len(category_details[category_details['Risk Level'] == risk_level])
                                st.metric(f"{risk_level} Risk", count)
                else:
                    st.info("No data found for this category.")

# ============================================================================
# PROJECT-SPECIFIC ANALYSIS
# ============================================================================
if selected_project != 'All Projects':
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏗️ Main Area of Observation")
        if 'Main Area Of Observation' in filtered_df.columns:
            area_data = filtered_df.groupby(['Main Area Of Observation', 'Risk Level']).size().reset_index()
            area_data.columns = ['Area', 'Risk Level', 'Count']
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
            layout = create_chart_layout()
            layout['xaxis']['showgrid'] = True
            layout['yaxis']['showgrid'] = False
            fig3.update_layout(**layout)
            st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Determine critical risk levels
        life_threat_variations = ['Life Threat', 'Life threat', 'LIFE THREAT']
        has_life_threat = any(lt in filtered_df['Risk Level'].values for lt in life_threat_variations)
        
        if has_life_threat:
            st.markdown("### 🚨 Critical Risk Issues (Life Threat & High)")
            critical_risks = life_threat_variations + ['High', 'HIGH']
            risk_label = "Life Threat & High Risk"
        else:
            st.markdown("### 🚨 High-Risk Issues")
            critical_risks = ['High', 'HIGH']
            risk_label = "High Risk"

        critical_risk_open = filtered_df[
            (filtered_df['Risk Level'].isin(critical_risks)) & 
            (filtered_df['Status'] == 'OPEN')
        ]

        if len(critical_risk_open) > 0:
            st.markdown(f"**⚠️ {len(critical_risk_open)} Open {risk_label} Issues**")
            
            # Sort by risk severity
            risk_priority = {lt: 0 for lt in life_threat_variations}
            risk_priority.update({'High': 1, 'HIGH': 1})
            critical_risk_open = critical_risk_open.sort_values(
                by='Risk Level',
                key=lambda x: x.map(risk_priority)
            )
            
            # Show top 5 critical issues
            for idx, row in critical_risk_open.head(5).iterrows():
                risk_emoji = "🔴" if row['Risk Level'] in life_threat_variations else "🟠"
                obs_id = row.get('Safety Observation ID', row.get('SA No', 'N/A'))
                
                with st.expander(f"{risk_emoji} {obs_id} - {row['Risk Level']}"):
                    st.markdown(f"**Category:** {row.get('Category', 'N/A')}")
                    st.markdown(f"**Location:** {row.get('Location Of Breach', 'N/A')}")
                    st.markdown(f"**Details:** {row.get('Details', 'N/A')}")
        else:
            st.success(f"✅ No open {risk_label.lower()} issues!")

# ============================================================================
# MONTHLY ANALYSIS (for selected projects)
# ============================================================================
if 'Calculated Month' in filtered_df.columns and selected_project != 'All Projects':
    if st.session_state.raw_df is not None:
        monthly_raw_df = st.session_state.raw_df.copy()
        monthly_raw_df['Calculated Month'] = pd.to_datetime(
            monthly_raw_df['Created Date / Time'], errors='coerce'
        ).dt.month
        monthly_raw_df = monthly_raw_df[monthly_raw_df['Project Name'] == selected_project]
        monthly_raw_df = monthly_raw_df[monthly_raw_df['Calculated Month'].notna()].copy()
        
        if len(monthly_raw_df) > 0:
            st.markdown("---")
            st.markdown("## 📅 Monthly Analysis")
            
            # Show available months
            available_months = sorted(monthly_raw_df['Calculated Month'].unique().astype(int).tolist())
            month_list = ', '.join([MONTH_NAMES.get(m, str(m)) for m in available_months])
            st.info(f"📊 Showing data for months: **{month_list}**")
            
            # Risk Distribution Per Month
            st.markdown("### 📊 Risk Distribution Per Month")
            risk_month_data = monthly_raw_df.groupby(
                ['Calculated Month', 'Risk Level']
            ).size().reset_index(name="Count")
            
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
            layout = create_chart_layout()
            layout['xaxis']['title'] = 'Month'
            fig_month1.update_layout(**layout)
            st.plotly_chart(fig_month1, use_container_width=True)
        
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Total Risks per Month")
                total_per_month = monthly_raw_df.groupby(
                    'Calculated Month'
                ).size().reset_index(name='Total Risks')
                
                fig_month2 = px.line(
                    total_per_month,
                    x='Calculated Month',
                    y='Total Risks',
                    markers=True,
                    template='plotly_dark'
                )
                fig_month2.update_traces(line_color='#FFB84D', marker=dict(size=10))
                layout = create_chart_layout()
                layout['xaxis']['title'] = 'Month'
                fig_month2.update_layout(**layout)
                st.plotly_chart(fig_month2, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Status per Month")
                status_month_data = monthly_raw_df.groupby(
                    ['Calculated Month', 'Status']
                ).size().reset_index(name="Count")
                
                fig_month3 = px.bar(
                    status_month_data,
                    x='Calculated Month',
                    y='Count',
                    color='Status',
                    color_discrete_map=STATUS_COLOR_MAP,
                    barmode='group',
                    template='plotly_dark'
                )
                layout = create_chart_layout()
                layout['xaxis']['title'] = 'Month'
                fig_month3.update_layout(**layout)
                st.plotly_chart(fig_month3, use_container_width=True)
            
            # Detailed Monthly Table
            st.markdown("### 📋 Detailed Monthly Risk Information")
            detail_cols = [
                'Calculated Month', 'Risk Level', 'Type', 'Details',
                'Main Area Of Observation', 'Sub Area Of Observation', 'Status'
            ]
            available_cols = [col for col in detail_cols if col in monthly_raw_df.columns]
            
            if available_cols:
                monthly_details = monthly_raw_df[available_cols].copy()
                monthly_details['Month Name'] = monthly_details['Calculated Month'].map(MONTH_NAMES)
                
                # Reorder columns to show month name first
                cols = ['Month Name'] + [col for col in monthly_details.columns if col != 'Month Name']
                monthly_details = monthly_details[cols].sort_values(by='Calculated Month')
                
                st.dataframe(
                    color_risk_level(monthly_details),
                    use_container_width=True,
                    height=400
                )

# ============================================================================
# DETAILED DATA TABLE
# ============================================================================
st.markdown("---")
st.markdown("### 📋 Detailed Safety Observations")

display_columns = [
    'Project Name', 'Safety Observation ID', 'Risk Level', 'Status',
    'Category', 'Main Area Of Observation', 'Details'
]
display_columns = [col for col in display_columns if col in filtered_df.columns]

st.dataframe(
    color_risk_level(filtered_df[display_columns]),
    use_container_width=True,
    height=400
)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: rgba(255,255,255,0.3);'>"
    f"{st.session_state.file_name} Dashboard © 2025 | "
    f"Powered by Streamlit"
    f"</p>",
    unsafe_allow_html=True
)