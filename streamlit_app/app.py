import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, List
import io
import plotly.io as pio

# Page configuration
st.set_page_config(
    page_title="AI Impact Trends Analysis",
    page_icon="📊",
    layout="wide"
)

# Get absolute path to data file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'Global_AI_Content_Impact_Dataset_with_Insights.csv')

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load and prepare the dataset for visualization."""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_kpis(df: pd.DataFrame) -> Tuple[int, float, str]:
    """Calculate main KPIs from the dataset."""
    if df.empty:
        return 0, 0.0, "N/A"
    total_categories = df['Industry'].nunique()
    avg_ai_usage = df['AI Adoption Rate (%)'].mean()
    yearly_growth = df.groupby('Year')['yearly_change_%'].mean()
    if yearly_growth.empty:
        max_growth_year = "N/A"
    else:
        max_growth_year = yearly_growth.idxmax()
    return total_categories, avg_ai_usage, max_growth_year

def create_line_plot(df: pd.DataFrame, selected_categories: List[str]) -> go.Figure:
    """Create line plot for average AI Usage % over time (per industry)."""
    # Group by Year and Industry, take mean
    avg_df = df.groupby(['Year', 'Industry'], as_index=False)['AI Adoption Rate (%)'].mean()
    fig = px.line(
        avg_df,
        x='Year',
        y='AI Adoption Rate (%)',
        color='Industry',
        title='Average AI Usage Trends Over Time',
        labels={'AI Adoption Rate (%)': 'AI Usage %', 'Year': 'Year'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=400, legend_title_text='Industry')
    return fig

def create_bar_plot(df: pd.DataFrame, selected_years: List[int]) -> go.Figure:
    """Create bar plot for AI Usage % by category."""
    filtered_df = df[df['Year'].isin(selected_years)]
    avg_by_category = filtered_df.groupby('Industry')['AI Adoption Rate (%)'].mean().reset_index()
    
    fig = px.bar(
        avg_by_category,
        x='Industry',
        y='AI Adoption Rate (%)',
        title='Average AI Usage by Industry',
        labels={'AI Adoption Rate (%)': 'AI Usage %', 'Industry': 'Industry'},
        color='Industry',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=400, legend_title_text='Industry', xaxis_tickangle=-45)
    return fig

def filter_insights_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Filter the DataFrame for rows where llm_insight contains the query (case-insensitive).
    Args:
        df (pd.DataFrame): The dataset.
        query (str): The search string.
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if not query:
        return df
    mask = df['llm_insight'].str.contains(query, case=False, na=False)
    return df[mask]

def create_comparison_plot(df: pd.DataFrame, entity1: str, entity2: str, comparison_type: str) -> go.Figure:
    """Create comparison plot between two countries or industries."""
    if comparison_type == "Country":
        df_filtered = df[df['Country'].isin([entity1, entity2])]
        color_by = 'Country'
    else:  # Industry
        df_filtered = df[df['Industry'].isin([entity1, entity2])]
        color_by = 'Industry'
    
    # Calculate average AI adoption rate by year and entity
    avg_df = df_filtered.groupby(['Year', color_by], as_index=False)['AI Adoption Rate (%)'].mean()
    
    fig = px.line(
        avg_df,
        x='Year',
        y='AI Adoption Rate (%)',
        color=color_by,
        title=f'AI Adoption Rate Comparison: {entity1} vs {entity2}',
        labels={'AI Adoption Rate (%)': 'AI Usage %', 'Year': 'Year'},
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    fig.update_layout(height=400, legend_title_text=color_by)
    return fig

def calculate_comparison_kpis(df: pd.DataFrame, entity1: str, entity2: str, comparison_type: str) -> tuple:
    """Calculate KPIs for comparison between two entities."""
    if comparison_type == "Country":
        df1 = df[df['Country'] == entity1]
        df2 = df[df['Country'] == entity2]
        entity_col = 'Country'
    else:
        df1 = df[df['Industry'] == entity1]
        df2 = df[df['Industry'] == entity2]
        entity_col = 'Industry'
    
    # Calculate KPIs
    avg1 = df1['AI Adoption Rate (%)'].mean()
    avg2 = df2['AI Adoption Rate (%)'].mean()
    max1 = df1['AI Adoption Rate (%)'].max()
    max2 = df2['AI Adoption Rate (%)'].max()
    growth1 = df1.groupby('Year')['AI Adoption Rate (%)'].mean().pct_change().mean() * 100
    growth2 = df2.groupby('Year')['AI Adoption Rate (%)'].mean().pct_change().mean() * 100
    
    return avg1, avg2, max1, max2, growth1, growth2

def create_insight_summary(df: pd.DataFrame) -> str:
    """Create a summary of key insights from the data."""
    avg_adoption = df['AI Adoption Rate (%)'].mean()
    top_industry = df.groupby('Industry')['AI Adoption Rate (%)'].mean().idxmax()
    top_country = df.groupby('Country')['AI Adoption Rate (%)'].mean().idxmax()
    yearly_growth = df.groupby('Year')['AI Adoption Rate (%)'].mean().pct_change().mean() * 100
    
    return f"""
    📊 **Key Insights Summary:**
    - Average AI adoption rate across all sectors: {avg_adoption:.1f}%
    - Leading industry: {top_industry}
    - Leading country: {top_country}
    - Average yearly growth: {yearly_growth:.1f}%
    """

def format_insight_card(row: pd.Series) -> str:
    """Format an insight into a compact card with relevant icons and colors."""
    # Determine trend icon and color based on metrics
    if row['yearly_change_%'] > 5:
        trend_icon = "📈"
        trend_color = "green"
    elif row['yearly_change_%'] < -5:
        trend_icon = "📉"
        trend_color = "red"
    else:
        trend_icon = "➡️"
        trend_color = "orange"
    
    return f"""
    <div style="padding: 10px; margin: 5px; border-radius: 5px; border: 1px solid #ddd; background-color: #f9f9f9;">
        <div style="color: {trend_color}; font-weight: bold;">
            {trend_icon} {row['Industry']} | {row['Country']} | {row['Year']}
        </div>
        <div style="color: #666; margin-top: 5px;">
            AI Usage: {row['AI Adoption Rate (%)']:.1f}% ({row['yearly_change_%']:+.1f}% YoY)
        </div>
        <div style="font-size: 0.9em; margin-top: 5px;">
            {row['llm_insight']}
        </div>
    </div>
    """

def main():
    """Main function to run the Streamlit app."""
    # Title & description
    st.title("AI Impact Trends Analysis 📊")
    st.markdown(
        """
        <div style='font-size:18px; color:#444; margin-bottom:16px;'>
        An interactive dashboard for analyzing the impact of Artificial Intelligence trends across global industries.<br>
        Select industries, years, or search for AI-generated insights, and discover business insights powered by data and LLMs.
        </div>
        """, unsafe_allow_html=True)

    # Load data
    df = load_data()
    if df.empty:
        st.stop()
    
    # Sidebar - Search bar always on top
    st.sidebar.title("Filters")
    
    # Sidebar filters
    countries = sorted(df['Country'].unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=countries,
        default=countries,
        help="Select countries to include in the analysis"
    )
    years = sorted(df['Year'].unique())
    categories = sorted(df['Industry'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Years",
        options=years,
        default=years,
        help="בחר שנים להצגה בגרפים ובתובנות"
    )
    selected_categories = st.sidebar.multiselect(
        "Select Industries",
        options=categories,
        default=categories[:3],
        help="בחר תעשיות להצגה בגרפים ובתובנות"
    )
    
    # סינון ראשי לפי כל הפילטרים
    df_filtered = df[
        (df['Country'].isin(selected_countries)) &
        (df['Year'].isin(selected_years)) &
        (df['Industry'].isin(selected_categories))
    ]

    # KPI Cards
    total_categories, avg_ai_usage, max_growth_year = calculate_kpis(df_filtered)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Industries", total_categories, help="Number of unique industries in the dataset")
    with col2:
        st.metric("Average AI Usage %", f"{avg_ai_usage:.1f}%", help="Average AI adoption rate across all selected industries")
    with col3:
        st.metric("Highest Growth Year", max_growth_year, help="The year with the highest average yearly change in AI adoption rate")
    
    # Visualizations
    st.subheader("Trend Analysis")
    
    if selected_categories and selected_countries:
        line_fig = create_line_plot(df_filtered, selected_categories)
        st.plotly_chart(line_fig, use_container_width=True)
        st.markdown("""
        **📈 Trend Analysis Over Time**  
        This line plot shows the AI adoption rate trends across different industries over time. 
        Each line represents an industry's progression, allowing you to identify:
        - Growth patterns in AI adoption
        - Industry-specific trends
        - Comparative analysis between sectors
        """)

    if selected_years and selected_countries:
        bar_fig = create_bar_plot(df_filtered, selected_years)
        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown("""
        **📊 Industry Comparison**  
        This bar chart compares AI adoption rates across different industries.
        The visualization helps identify:
        - Leading industries in AI adoption
        - Sectors with potential for growth
        - Overall industry distribution
        """)
    
    # Insights Section
    st.header("🤖 AI-Generated Insights")
    
    if not df_filtered.empty:
        # Add summary at the top
        st.markdown(create_insight_summary(df_filtered))
        
        # Show statistical overview
        avg_usage = df_filtered['AI Adoption Rate (%)'].mean()
        total_insights = len(df_filtered)
        st.info(f"Analysis based on {total_insights} data points with average AI adoption rate of {avg_usage:.1f}%")
        
        # Create tabs for different insight views
        tab1, tab2 = st.tabs(["📊 Key Insights", "📑 Detailed View"])
        
        with tab1:
            # Show top insights (significant changes)
            significant_changes = df_filtered[abs(df_filtered['yearly_change_%']) > 5].sort_values('yearly_change_%', ascending=False)
            
            if not significant_changes.empty:
                st.markdown("### Significant Changes")
                for _, row in significant_changes.head(5).iterrows():
                    st.markdown(format_insight_card(row), unsafe_allow_html=True)
            
        with tab2:
            # Show all insights in a more compact table format
            st.dataframe(
                df_filtered[['Country', 'Industry', 'Year', 'AI Adoption Rate (%)', 'yearly_change_%', 'llm_insight']]
                .sort_values('AI Adoption Rate (%)', ascending=False),
                use_container_width=True,
                height=400
            )
    
    else:
        st.warning("No data available for the selected filters.")

    # Data Table & Download
    st.subheader("Filtered Data Table")
    st.dataframe(df_filtered, use_container_width=True)
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download Insights as CSV", csv, "insights.csv", "text/csv")

    # Comparison Section
    st.header("📊 Comparative Analysis")
    
    # Choose comparison type
    comparison_type = st.radio(
        "Select comparison type:",
        ["Country", "Industry"],
        horizontal=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if comparison_type == "Country":
            entity1 = st.selectbox("Select first country", countries, index=0)
            entity2 = st.selectbox("Select second country", [c for c in countries if c != entity1], index=0)
        else:
            entity1 = st.selectbox("Select first industry", categories, index=0)
            entity2 = st.selectbox("Select second industry", [c for c in categories if c != entity1], index=0)

    # Create and display comparison plot
    comparison_fig = create_comparison_plot(df, entity1, entity2, comparison_type)
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Calculate and display KPIs
    avg1, avg2, max1, max2, growth1, growth2 = calculate_comparison_kpis(df, entity1, entity2, comparison_type)
    
    st.markdown("### Key Metrics Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average AI Usage",
            f"{avg1:.1f}% vs {avg2:.1f}%",
            f"{avg1 - avg2:+.1f}%",
            help=f"Average AI adoption rate comparison between {entity1} and {entity2}"
        )
    
    with col2:
        st.metric(
            "Highest AI Usage",
            f"{max1:.1f}% vs {max2:.1f}%",
            f"{max1 - max2:+.1f}%",
            help=f"Highest AI adoption rate comparison between {entity1} and {entity2}"
        )
    
    with col3:
        st.metric(
            "Yearly Growth Rate",
            f"{growth1:.1f}% vs {growth2:.1f}%",
            f"{growth1 - growth2:+.1f}%",
            help=f"Average yearly growth rate comparison between {entity1} and {entity2}"
        )

    st.markdown("""
    **📈 Comparison Analysis**  
    This visualization and metrics help you understand:
    - Relative AI adoption patterns between the selected entities
    - Growth rate differences and trends
    - Key performance gaps and opportunities
    """)

if __name__ == "__main__":
    main()
