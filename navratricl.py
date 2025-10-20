import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Navratri Techno-Garba Dashboard",
    page_icon="🕺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: white;
        margin: 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0;
    }
    .takeaway {
        background-color: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        border-left: 4px solid #4ECDC4;
        font-style: italic;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_synthetic_data():
    """Generate synthetic Navratri event data for demonstration"""
    np.random.seed(42)
    
    states = ['Gujarat', 'Maharashtra', 'Rajasthan', 'Uttar Pradesh', 'Delhi', 'Karnataka', 'Tamil Nadu', 'West Bengal']
    cities = {
        'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot'],
        'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Nashik'],
        'Rajasthan': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota'],
        'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Varanasi', 'Agra'],
        'Delhi': ['New Delhi', 'Gurgaon', 'Noida', 'Faridabad'],
        'Karnataka': ['Bangalore', 'Mysore', 'Hubli', 'Mangalore'],
        'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem'],
        'West Bengal': ['Kolkata', 'Howrah', 'Durgapur', 'Siliguri']
    }
    
    event_types = ['Traditional Garba', 'Techno Garba', 'Dandiya Night', 'Cultural Program', 'Competition']
    categories = ['Goddess Devi & Her Nine Forms', 'Indian Culture & Traditions', 'Women Empowerment', 
                 'Social & Economic Impact', 'Spiritual Significance', 'Cultural & Global Influence']
    
    # Generate dates for Navratri period (9 days)
    start_date = datetime(2025, 9, 15)
    dates = [start_date + timedelta(days=i) for i in range(9)]
    
    data = []
    event_id = 1
    
    for _ in range(300):  # Generate 300 events
        state = np.random.choice(states)
        city = np.random.choice(cities[state])
        event_type = np.random.choice(event_types)
        category = np.random.choice(categories)
        date = np.random.choice(dates)
        
        # Generate realistic participant numbers based on event type and state
        base_participants = {
            'Traditional Garba': 150,
            'Techno Garba': 300,
            'Dandiya Night': 200,
            'Cultural Program': 100,
            'Competition': 80
        }
        
        state_multiplier = {
            'Gujarat': 1.5, 'Maharashtra': 1.3, 'Rajasthan': 1.1,
            'Uttar Pradesh': 1.0, 'Delhi': 1.2, 'Karnataka': 0.9,
            'Tamil Nadu': 0.8, 'West Bengal': 0.7
        }
        
        participants = int(base_participants[event_type] * state_multiplier[state] * np.random.uniform(0.5, 2.0))
        donations = participants * np.random.uniform(50, 200)  # Donations per participant
        
        data.append({
            'Event_ID': event_id,
            'Date': date.strftime('%Y-%m-%d'),
            'State': state,
            'City': city,
            'Event_Type': event_type,
            'Category': category,
            'Participants': participants,
            'Donations': round(donations, 2),
            'Duration_Hours': np.random.randint(2, 8),
            'Organizer': f"Local Committee {event_id}"
        })
        event_id += 1
    
    return pd.DataFrame(data)

@st.cache_data
def load_data(uploaded_file=None):
    """Load data from uploaded file or generate synthetic data"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return generate_synthetic_data()
    else:
        df = generate_synthetic_data()
        df['Date'] = pd.to_datetime(df['Date'])
        return df

def create_kpi_cards(df_filtered):
    """Create KPI cards for key metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_events = len(df_filtered)
    total_participants = df_filtered['Participants'].sum()
    avg_participants = df_filtered['Participants'].mean()
    total_donations = df_filtered['Donations'].sum()
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{total_events:,}</p>
            <p class="kpi-label">Total Events</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{total_participants:,}</p>
            <p class="kpi-label">Total Participants</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{avg_participants:.0f}</p>
            <p class="kpi-label">Avg Participants/Event</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">₹{total_donations:,.0f}</p>
            <p class="kpi-label">Total Donations</p>
        </div>
        """, unsafe_allow_html=True)

def create_time_series_chart(df_filtered):
    """Create time-series chart for participants over time"""
    daily_data = df_filtered.groupby('Date').agg({
        'Participants': 'sum',
        'Event_ID': 'count'
    }).reset_index()
    daily_data.rename(columns={'Event_ID': 'Events'}, inplace=True)
    
    fig = px.line(daily_data, x='Date', y='Participants', 
                  title='📈 Daily Participation Trends During Navratri',
                  hover_data=['Events'])
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Participants",
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_traces(
        line=dict(color='#4ECDC4', width=3),
        hovertemplate='<b>Date:</b> %{x}<br><b>Participants:</b> %{y:,}<br><b>Events:</b> %{customdata[0]}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="takeaway">💡 <strong>Takeaway:</strong> Peak participation typically occurs on weekends and during the final days of Navratri celebration.</div>', unsafe_allow_html=True)

def create_state_category_chart(df_filtered):
    """Create stacked bar chart for participants by category per state"""
    state_category = df_filtered.groupby(['State', 'Category'])['Participants'].sum().reset_index()
    
    fig = px.bar(state_category, x='State', y='Participants', color='Category',
                 title='🌟 Participation by State and Event Category',
                 color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_layout(
        xaxis_title="State",
        yaxis_title="Number of Participants",
        legend_title="Event Category",
        template='plotly_white',
        xaxis={'categoryorder': 'total descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="takeaway">💡 <strong>Takeaway:</strong> Gujarat and Maharashtra lead in participation, with "Goddess Devi & Her Nine Forms" being the most popular category.</div>', unsafe_allow_html=True)

def create_event_breakdown_chart(df_filtered):
    """Create treemap for event type breakdown"""
    event_breakdown = df_filtered.groupby(['Event_Type', 'State']).agg({
        'Participants': 'sum',
        'Event_ID': 'count'
    }).reset_index()
    
    fig = px.treemap(event_breakdown, 
                     path=['Event_Type', 'State'], 
                     values='Participants',
                     title='🎭 Event Type Distribution Across States',
                     color='Participants',
                     color_continuous_scale='Viridis')
    
    fig.update_layout(template='plotly_white')
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="takeaway">💡 <strong>Takeaway:</strong> Techno Garba events attract the highest participation, showing the blend of tradition with modern elements.</div>', unsafe_allow_html=True)

def create_top_cities_chart(df_filtered, top_n=10):
    """Create horizontal bar chart for top cities by participation"""
    city_data = df_filtered.groupby('City').agg({
        'Participants': 'sum',
        'Event_ID': 'count',
        'State': 'first'
    }).reset_index()
    city_data.rename(columns={'Event_ID': 'Events'}, inplace=True)
    city_data = city_data.nlargest(top_n, 'Participants')
    
    fig = px.bar(city_data, y='City', x='Participants', 
                 title=f'🏆 Top {top_n} Cities by Total Participation',
                 orientation='h',
                 color='Participants',
                 color_continuous_scale='plasma',
                 hover_data=['Events', 'State'])
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Number of Participants",
        yaxis_title="City",
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="takeaway">💡 <strong>Takeaway:</strong> Metropolitan cities show higher participation due to larger populations and better event organization infrastructure.</div>', unsafe_allow_html=True)

def create_interactive_table(df_filtered):
    """Create interactive data table with download functionality"""
    st.subheader("📊 Detailed Event Data")
    
    # Display summary statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Events Displayed", len(df_filtered))
    with col2:
        st.metric("Data Coverage", f"{df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}")
    
    # Format data for display
    display_df = df_filtered.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df['Donations'] = display_df['Donations'].apply(lambda x: f"₹{x:,.2f}")
    
    # Display the dataframe
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        column_config={
            "Event_ID": "Event ID",
            "Participants": st.column_config.NumberColumn("Participants", format="%d"),
            "Duration_Hours": st.column_config.NumberColumn("Duration (hrs)", format="%d")
        }
    )
    
    # Download button
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"navratri_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🕺 Vision 2.o Navratri Dashboard 💃</h1>
        <p>Interactive Analytics for Cultural Celebrations & Community Engagement</p>

    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for filters
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # File upload
    st.sidebar.subheader("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file", 
        type=['csv'],
        help="Upload a CSV file with columns: Date, State, City, Event_Type, Category, Participants, Donations"
    )
    
    if uploaded_file is None:
        st.sidebar.info("🎯 Using synthetic demo data. Upload your CSV for real analysis!")
    
    # Load data
    df = load_data(uploaded_file)
    
    # Filters
    st.sidebar.subheader("🔍 Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(df['Date'].min().date(), df['Date'].max().date()),
        min_value=df['Date'].min().date(),
        max_value=df['Date'].max().date()
    )
    
    # Multi-select filters
    states = st.sidebar.multiselect(
        "🌍 States",
        options=sorted(df['State'].unique()),
        default=sorted(df['State'].unique()),
        help="Select one or more states to analyze"
    )
    
    event_types = st.sidebar.multiselect(
        "🎪 Event Types",
        options=sorted(df['Event_Type'].unique()),
        default=sorted(df['Event_Type'].unique()),
        help="Select event types to include"
    )
    
    categories = st.sidebar.multiselect(
        "📋 Categories",
        options=sorted(df['Category'].unique()),
        default=sorted(df['Category'].unique()),
        help="Select event categories to analyze"
    )
    
    # Apply filters
    df_filtered = df[
        (df['Date'].dt.date >= date_range[0]) &
        (df['Date'].dt.date <= date_range[1]) &
        (df['State'].isin(states)) &
        (df['Event_Type'].isin(event_types)) &
        (df['Category'].isin(categories))
    ]
    
    # Main dashboard content
    if len(df_filtered) == 0:
        st.error("🚫 No data matches your selected filters. Please adjust your criteria.")
        return
    
    # KPI Cards
    st.subheader("📈 Key Performance Indicators")
    create_kpi_cards(df_filtered)
    
    # Charts section
    st.subheader("📊 Visual Analytics")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        create_time_series_chart(df_filtered)
    
    with col2:
        create_event_breakdown_chart(df_filtered)
    
    # Second row of charts
    col3, col4 = st.columns(2)
    
    with col3:
        create_state_category_chart(df_filtered)
    
    with col4:
        # Top N selector
        top_n = st.selectbox("Select top N cities:", [5, 10, 15, 20], index=1)
        create_top_cities_chart(df_filtered, top_n)
    
    # Data table
    create_interactive_table(df_filtered)
    
  
if __name__ == "__main__":
    main()