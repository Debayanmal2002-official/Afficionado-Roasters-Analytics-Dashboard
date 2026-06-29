import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.stattools import adfuller, kpss
from prophet import Prophet
import xgboost as xgb
from src.ui_components import kpi_card, apply_custom_css


BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "data" / "Coffee_Clean.parquet"

@st.cache_data
def load_data(path):
    return pd.read_parquet(path)
df = load_data(file_path)


# MAIN PAGE CONFIG
st.set_page_config(
    page_title="Afficionado Coffee Roasters Dashboard",
    page_icon="☕",
    layout="wide"
)

# MAIN APP CONFIG

apply_custom_css()

## -- Hero Header Elements -- ##
st.markdown("""
<div class="hero-section">
    <div class="hero-title"> Afficionado Coffee Roasters Dashboard</div>
    <div class="hero-subtitle">
        Interactive Dashboard for Customer, Product, and Profitability Performance Analysis in Supply Chain Operations.
    </div>
</div>
""", unsafe_allow_html=True)


## -- Sidebar Filters -- ##
st.sidebar.header("Dashboard Filters")
top_n = st.sidebar.slider("Select number of top items to display", 5, 20, 5)
# 1. Your Multiselect
locations = sorted(df['store_location'].unique())
selected_locations = st.sidebar.multiselect("Select Locations",options=locations)
if not selected_locations:
    selected_locations = locations
else:
    selected_locations = selected_locations
selected_metric = st.sidebar.radio("Select Metric to View:", ['transaction_qty', 'revenue'], horizontal=True)

# Apply filter to the dataframe
main_df = df[ (df['store_location'].isin(selected_locations)) ]

## -- Calculations_1 -- ##
main_df['total_bill'] = main_df['unit_price'] * main_df['transaction_qty']
total_revenue = main_df['total_bill'].sum()
total_qty = main_df['transaction_qty'].sum()
avg_bill = main_df['total_bill'].mean()

## -- Feature Engineering -- ##
df_model = main_df.copy()
df_model['revenue'] = df_model['transaction_qty'] * df_model['unit_price']
df_hourly = df_model.groupby([
    'store_location',
    'product_category',
    'day_id',
    'hour'
]).agg({
    'transaction_qty': 'sum',
    'revenue': 'sum'
}).reset_index()
df_hourly = df_hourly.sort_values(['store_location', 'product_category', 'day_id', 'hour'])

# 1. Create unique lists of your dimensions
stores = df_hourly['store_location'].unique()
cats = df_hourly['product_category'].unique()
days = df_hourly['day_id'].unique()
min_hour = df_hourly['hour'].min()
max_hour = df_hourly['hour'].max()
hours = range(min_hour - 1, max_hour + 1)

# 2. Build the full template (the Cartesian product)
full_index = pd.MultiIndex.from_product([stores, cats, days, hours],
                                        names=['store_location', 'product_category', 'day_id', 'hour']
                                        )
df_full = pd.DataFrame(index=full_index).reset_index()

# 3. Merge your actual data into the template
df_final = pd.merge(
    df_full,
    df_hourly,
    on=['day_id', 'hour', 'store_location', 'product_category'],
    how='left'
)

date_mapping = main_df[['day_id', 'date']].drop_duplicates().set_index('day_id')['date']
df_final['date'] = df_final['day_id'].map(date_mapping)
cols = ['date'] + [c for c in df_final.columns if c != 'date']
df_final = df_final[cols]

# 4. Fill the gaps with 0
df_final['transaction_qty'] = df_final['transaction_qty'].fillna(0)
df_final['revenue'] = df_final['revenue'].fillna(0)

# 5. Re-sort to maintain the chronological order for your lags
df_final = df_final.sort_values(['store_location', 'product_category', 'day_id', 'hour'])

## -- Main App -- ##
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 Product & Category Performance",
    "📈 Historical Trend Analysis",
    "📊 Model Evaluation",
    "⏳ Future Forcasting",
    "📋 Business Report"
])

with tab1:
    st.header("Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Quantity Sold", f"{total_qty:,}")
    col3.metric("Average Transaction", f"${avg_bill:.2f}")

    st.markdown("---")

    # Aggregate for the TABLE (Includes Unit Price)
    product_stats = main_df.groupby(['product_category', 'product_type', 'product_detail', 'unit_price']).agg({'transaction_qty': 'sum','total_bill': 'sum'}).reset_index()
    product_stats.columns = ['Category', 'Type', 'Product Name', 'Unit Price', 'Units Sold', 'Total Revenue']
    product_stats['Display Name'] = (product_stats['Product Name'] +" ($" + product_stats['Unit Price'].astype(str) + ")")
    product_stats = product_stats.sort_values(by='Units Sold', ascending=False)

    # --- Layout: Top vs Bottom --- #
    col_1, col_2 = st.columns(2)

    with col_1:
        with st.container(border=True):
            st.subheader("🏆 Top 5 Performers (Volume)")
            top_5 = product_stats.head(5)
            fig_top = px.bar(
                top_5,
                x='Units Sold',
                y='Display Name',  # Use the unique display name here
                orientation='h',
                color_discrete_sequence=['#85BB65'],
                text='Units Sold',
                height=450
            )
            fig_top.update_layout(
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_top, width='content')

    with col_2:
        with st.container(border=True):
            st.subheader("⚠️ Bottom 5 Performers (Volume)")
            bottom_5 = product_stats.tail(5).sort_values(by='Units Sold', ascending=True)
            fig_bottom = px.bar(
                bottom_5,
                x='Units Sold',
                y='Display Name',
                orientation='h',
                color_discrete_sequence=['#FF4B33'],
                text='Units Sold',
                height=450
            )
            fig_bottom.update_layout(
                showlegend=False,
                yaxis={'categoryorder': 'total descending'},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_bottom, width='content')

    # --- Full Ranking Table --- #
    with st.expander("📄 View Full Product Ranking List"):
        table_df = product_stats.drop(columns=['Display Name'])
        styled_df = table_df.style.background_gradient(
            subset=['Units Sold'], cmap='Greens'
        ).background_gradient(
            subset=['Total Revenue'], cmap='YlOrBr'
        ).background_gradient(
            subset=['Unit Price'], cmap='YlOrBr'
        ).format({
            'Total Revenue': '${:,.2f}',
            'Units Sold': '{:,}',
            'Unit Price': '${:,.2f}'
        })
        st.dataframe(styled_df, width='stretch')

    st.markdown("---")

    # --- Revenue Analysis --- #
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader(f"Top {top_n} Categories by Revenue")
        category_revenue = main_df.groupby('product_category')['total_bill'].sum().reset_index()
        category_revenue = category_revenue.sort_values(by='total_bill', ascending=False)

        # Separate Top N from the rest
        top_slice = category_revenue.head(top_n)
        others_slice = category_revenue.iloc[top_n:]

        # Create 'Other' row if there are items remaining
        if not others_slice.empty:
            others_row = pd.DataFrame({
                'product_category': ['Other Products'],
                'total_bill': [others_slice['total_bill'].sum()]
            })
            plot_df = pd.concat([top_slice, others_row])
        else:
            plot_df = top_slice

        fig_type = px.pie(
            plot_df,
            values='total_bill',
            names='product_category',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold)
        fig_type.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_type, width='content')

    with right_col:
        st.subheader("Product Types Revenue Chart by Category")

        # Unique categories for the multiselect
        categories = sorted(main_df['product_category'].unique())
        # Use selectbox for a single choice
        selected_categories = st.selectbox("Select a Category to view details:",options=categories)
        main_df_1 = main_df[main_df['product_category'] == selected_categories]

        # Grouping every unique type in the selected categories
        type_grouped = main_df_1.groupby(['product_category', 'product_type'])['total_bill'].sum().reset_index()
        type_grouped = type_grouped.sort_values(['product_category', 'total_bill'], ascending=[True, False])

        # Bar chart showing ALL unique types
        fig_type_bar = px.bar(
            type_grouped,
            x='total_bill',
            y='product_type',  # Focus on types
            color='total_bill',
            color_continuous_scale='Tealgrn',  # Grouped by category color
            orientation='h',
            labels={'total_bill': 'Revenue ($)', 'product_type': 'Product Type'},
            height=300  # Height adjusted to fit many types
        )

        fig_type_bar.update_layout(
            showlegend=False,  # Removes the legend
            coloraxis_showscale=False,  # Removes the color bar on the right
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_type_bar, width='content')

    st.markdown("---")

    # --- Revenue Concentration & Menu Balance & Enginnering --- #

    st.header("⚖️ Revenue Concentration & Menu Balance")

    # 1. Prepare Data for Pareto
    pareto_df = product_stats.groupby('Display Name')['Total Revenue'].sum().reset_index()
    pareto_df = pareto_df.sort_values(by='Total Revenue', ascending=False)

    # 2. Calculate Cumulative Percentages
    pareto_df['Cumulative Revenue'] = pareto_df['Total Revenue'].cumsum()
    total_rev = pareto_df['Total Revenue'].sum()
    pareto_df['Cumulative %'] = (pareto_df['Cumulative Revenue'] / total_rev) * 100

    # 3. Categorize Products
    def categorize_pareto(pct):
        if pct <= 80: return "Revenue Anchor (Top 80%)"
        return "Long-tail (Remaining 20%)"

    pareto_df['Classification'] = pareto_df['Cumulative %'].apply(categorize_pareto)

    # --- Visualizing Concentration --- #
    # Calculate metrics for the summary
    anchors_count = len(pareto_df[pareto_df['Classification'] == "Revenue Anchor (Top 80%)"])
    total_products = len(pareto_df)
    anchor_ratio = (anchors_count / total_products) * 100

    fig_tree = px.treemap(
        pareto_df,
        path=['Classification', 'Display Name'],
        values='Total Revenue',
        color='Classification',
        color_discrete_map={
            "Revenue Anchor (Top 80%)": "#2E7D32",  # Deeper green for better readability
            "Long-tail (Remaining 20%)": "#424242"  # Dark grey to signal "secondary importance"
        }
    )

    fig_tree.update_traces(
        textinfo="label+value",
        texttemplate="<b>%{label}</b><br>$%{value:,.0f}",  # Bold name and formatted currency
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Contribution: %{percentParent:.1%}",
        marker_line_width=2,
        marker_line_color="#121212"  # Sharp borders between boxes
    )

    fig_tree.update_layout(
        margin=dict(t=30, l=10, r=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600
    )

    st.plotly_chart(fig_tree, width='stretch')

    st.markdown("---")

    col1, col2 = st.columns(2)

    col1.metric("Revenue Anchors", f"{anchors_count} Products")
    col2.metric("Concentration Ratio", f"{anchor_ratio:.1f}%")

    st.markdown("---")

    # Risk Assessment
    if anchor_ratio < 15:
        st.error("⚠️ **High Risk**: Revenue is too concentrated in very few products.")
    elif anchor_ratio > 25:
        st.success("✅ **Balanced**: Revenue is healthy and spread across the menu.")
    else:
        st.warning("⚖️ **Moderate**: Standard 80/20 distribution.")

    st.markdown("---")

    # 4. Long-tail Detail

    with st.expander("🔍 Identify Long-tail Products (Candidates for Removal)"):
        long_tail = pareto_df[pareto_df['Classification'] == "Long-tail (Remaining 20%)"]
        st.write(f"These {len(long_tail)} products contribute to only 20% of your total revenue.")
        st.dataframe(long_tail[['Display Name', 'Total Revenue', 'Cumulative %']], width='stretch')

    st.markdown("---")

    st.header("🎯 Menu Engineering: Popularity vs. Revenue")

    # 1. Define the Multiselect Filter
    all_categories = sorted(product_stats['Category'].unique().tolist())
    # Selectbox for category
    selected_categories_2 = st.multiselect(
        "Filter by Category:",
        options=all_categories,
        default=all_categories  # Start with everything selected
    )

    # 2. Filter the Data
    if selected_categories:
        filtered_df = product_stats[product_stats['Category'].isin(selected_categories_2)]
    else:
        filtered_df = product_stats

    avg_units = filtered_df['Units Sold'].mean()
    avg_rev = filtered_df['Total Revenue'].mean()

    # 2. Create the Scatter Plot
    fig_scatter = px.scatter(
        filtered_df,
        x='Units Sold',
        y='Total Revenue',
        color='Type',
        size='Total Revenue',
        hover_name='Display Name',
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Dark2
    )

    # 3. Add Quadrant Lines
    fig_scatter.add_vline(x=avg_units, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    fig_scatter.add_hline(y=avg_rev, line_dash="dash", line_color="rgba(255,255,255,0.5)")

    # 4. Add Quadrant Labels
    quadrant_labels = [
        dict(x=product_stats['Units Sold'].max() * 1.1, y=product_stats['Total Revenue'].max() * 1.05, text="⭐ STARS"),
        dict(x=product_stats['Units Sold'].min() * 1.0, y=product_stats['Total Revenue'].max() * 1.05, text="🧩 PUZZLES"),
        dict(x=product_stats['Units Sold'].max() * 1.05, y=product_stats['Total Revenue'].min() * 1.2,text="🐎 WORKHORSES"),
        dict(x=-300, y=2500, text="🐕 DOGS")
    ]

    for label in quadrant_labels:
        fig_scatter.add_annotation(
            x=label['x'], y=label['y'],
            text=label['text'],
            showarrow=False,
            font=dict(size=16, color="white", family="Arial Black"),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1
        )

    fig_scatter.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='White')))
    fig_scatter.update_layout(
        height=600,
        xaxis_title="Popularity (Units Sold)",
        yaxis_title="Profitability (Total Revenue $)",
        hovermode='closest',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_scatter, width='stretch')

with tab2:
    categories = sorted(df['product_category'].unique())
    selected_cats = st.multiselect("Select Categories to compare trends:", options=categories, default= categories[:3])
    if not selected_locations:
        selected_cats = categories
    else:
        selected_cats = selected_cats
    hour_df = df_final[df_final['product_category'].isin(selected_cats)]
    st.subheader("Peak Hour Analysis by Category")
    hourly_cat_rev = hour_df.groupby(['hour', 'product_category'])[selected_metric].sum().reset_index()

    fig_hour = px.line(
        hourly_cat_rev,
        x='hour',
        y=selected_metric,
        color='product_category',
        markers=True,
        line_shape='spline',
        labels={'transaction_qty': 'Total Sales', 'revenue': 'Revenue', 'hour': 'Hour of Day (24h)',
                'product_category': 'Category'},
        color_discrete_sequence=px.colors.qualitative.Light24
    )

    fig_hour.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_hour, width='stretch')

    st.markdown("---")

    st.subheader("Trend Analysis by Category")
    df_trend = df_final[df_final['product_category'].isin(selected_cats)]
    df_daily = df_trend.groupby(['date', 'product_category']).agg({selected_metric: 'sum'}).reset_index()
    # 2. Create the Plotly chart
    fig = px.line(
        df_daily,
        x='date',
        y=selected_metric,
        color='product_category',
        title=f"Daily Total {selected_metric.replace('_', ' ').title()} by Category",
        line_shape='spline',
        labels={'transaction_qty': 'Total Sales', 'revenue': 'Revenue', 'hour': 'Hour of Day (24h)','product_category': 'Category'},
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Light24
    )

    # 3. Clean up the layout
    fig.update_layout(
        xaxis_title="Time Period",
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    st.subheader("💡 Strategic Business Insights")

    # 1. Identify the Peak Hour (The "Rush")
    peak_hour = df_final.groupby('hour')['transaction_qty'].sum().idxmax()
    total_peak_vol = df_final.groupby('hour')['transaction_qty'].sum().max()

    # 2. Identify the top performing category
    top_cat = df_final.groupby('product_category')['transaction_qty'].sum().idxmax()

    # 3. Dynamic Text Insights
    st.markdown(f"""
    - **Peak Operational Window:** Your business hits its absolute maximum capacity at **{peak_hour}:00**. 
      - *Insight:* Ensure staffing levels are 20% higher at this hour to avoid lost sales or bottlenecking.
    - **Top Driver:** The **'{top_cat}'** category is your primary volume driver. 
      - *Insight:* Any supply chain disruption in this category will have an outsized impact on your total daily revenue.
    - **The 'Morning Rush' Phenomenon:** Your charts show a distinct **cliff at 11:00 AM**. 
      - *Insight:* You have a "morning-only" business model. You could potentially increase revenue by launching a "Midday Refresh" promotion specifically to flatten this curve after 11:00 AM.
    """)

    st.subheader("🥐 Cross-Sell Opportunities")

    # Group by hour to see if demand for Bakery and Drinks align
    # We calculate the correlation between Bakery and Drinks
    hourly_corr = df_final.pivot_table(index=['day_id', 'hour'],
                                       columns='product_category',
                                       values='transaction_qty',
                                       aggfunc='sum').fillna(0)

    # Check correlation: Do people buy Bakery when they buy Coffee?
    coffee_bakery_corr = hourly_corr['Coffee'].corr(hourly_corr['Bakery'])
    tea_bakery_corr = hourly_corr['Tea'].corr(hourly_corr['Bakery'])

    st.write(f"- **Coffee-Bakery Synergy:** Correlation is {coffee_bakery_corr:.2f}. " +
             (
                 "Strong combo! Highlight bakery near the coffee machine." if coffee_bakery_corr > 0.5 else "Opportunity to bundle."))

    st.write(f"- **Tea-Bakery Synergy:** Correlation is {tea_bakery_corr:.2f}. " +
             ("Strong combo! Cross-promote tea and bakery together." if tea_bakery_corr > 0.5 else "Weak synergy."))

    st.markdown('---')

    st.subheader("🚀 Growth Analysis Summary (6-Month View)")

    # 1. Calculate growth as we did before
    early_days = df_final['day_id'].min() + 30
    late_days = df_final['day_id'].max() - 30

    early_perf = df_final[df_final['day_id'] <= early_days].groupby('product_category')[selected_metric].mean()
    late_perf = df_final[df_final['day_id'] >= late_days].groupby('product_category')[selected_metric].mean()

    growth = ((late_perf - early_perf) / early_perf) * 100

    # 2. Build the DataFrame for the table
    df_growth = growth.reset_index()
    df_growth.columns = ['Category', 'Growth (%)']

    # 3. Categorize the performance (Logic inside the DataFrame)
    def get_status(val):
        if val > 10: return "Growth Engine"
        if val < -5: return "Cooling Down"
        return "Stable"
    def color_status(val):
        if val == "Growth Engine":
            return 'background-color: #d4edda; color: #155724'  # Light Green
        elif val == "Cooling Down":
            return 'background-color: #f8d7da; color: #721c24'  # Light Red
        else:
            return 'background-color: #fff3cd; color: #856404'  # Light Yellow
    df_growth['Status'] = df_growth['Growth (%)'].apply(get_status)
    styled_table = df_growth.style.map(color_status, subset=['Status']) \
                    .format({"Growth (%)": "{:.1f}%"})

    st.dataframe(styled_table, width='stretch')
    st.markdown("---")

    # 4. Success message
    ## Feature Engineering 2 ##
    df_final['qty_lag_1'] = df_final.groupby(['store_location', 'product_category', 'day_id'])['transaction_qty'].shift(
        1)
    df_final['qty_lag_24'] = df_final.groupby(['store_location', 'product_category'])['transaction_qty'].shift(16)
    df_final['qty_lag_168'] = df_final.groupby(['store_location', 'product_category'])['transaction_qty'].shift(112)

    df_final['rev_lag_1'] = df_final.groupby(['store_location', 'product_category', 'day_id'])['revenue'].shift(1)
    df_final['rev_lag_24'] = df_final.groupby(['store_location', 'product_category'])['revenue'].shift(16)
    df_final['rev_lag_168'] = df_final.groupby(['store_location', 'product_category'])['revenue'].shift(112)

    # Add your Rolling Average
    df_final['qty_roll_avg_3d'] = df_final.groupby(['store_location', 'product_category', 'hour'])[
        'transaction_qty'].transform(lambda x: x.rolling(window=3).mean())
    df_final['qty_roll_avg_7d'] = df_final.groupby(['store_location', 'product_category', 'hour'])[
        'transaction_qty'].transform(lambda x: x.rolling(window=7).mean())

    df_final['rev_roll_avg_3d'] = df_final.groupby(['store_location', 'product_category', 'hour'])['revenue'].transform(
        lambda x: x.rolling(window=3).mean())
    df_final['rev_roll_avg_7d'] = df_final.groupby(['store_location', 'product_category', 'hour'])['revenue'].transform(
        lambda x: x.rolling(window=7).mean())

    # 3. Drop the 'Cold Start' rows
    # Since we need 105 hours of history, we drop the first 105 rows
    df_final.dropna(inplace=True)

    st.subheader("Hourly Aggregated Data")
    st.success(f"Features engineered! Dataset size after dropping NaNs: {len(df_final)} rows.")
    with st.expander("View Full Engineered Dataset (df_final)"):
        st.write("Below is the full dataset after feature engineering, including all lags and rolling averages.")
        st.dataframe(df_final, width='stretch')
    st.markdown("---")


with tab3:
    cats = sorted(df_final['product_category'].unique().tolist())
    selected_cat_2 = st.sidebar.selectbox("Select Product Category for Baseline:", cats)
    df_eval = df_final[df_final['product_category'] == selected_cat_2].copy()
    lag_col = f"{'qty' if selected_metric == 'transaction_qty' else 'rev'}_lag_24"
    roll_col = f"{'qty' if selected_metric == 'transaction_qty' else 'rev'}_roll_avg_7d"
    df_bm = df_eval.copy()
    df_bm = df_bm.groupby('date')[[selected_metric, lag_col, roll_col]].sum().reset_index()
    df_bm = df_bm.set_index('date').asfreq('D').fillna(0)
    # 1. Transformations
    df_bm['diff'] = df_bm[selected_metric].diff().fillna(0)
    df_bm['diff_2'] = df_bm['diff'].diff().fillna(0)
    df_bm['sqrt'] = np.sqrt(df_bm[selected_metric])
    df_bm['log'] = np.log1p(df_bm[selected_metric])
    df_bm['day_of_week'] = df_bm.index.dayofweek


    # 2. Box-Cox Safety Check
    # Box-Cox requires data > 0. If you have 0s, we add a tiny constant.
    if (df_bm[selected_metric] >= 0).all():
        # Add a tiny epsilon if there are zeros to avoid math errors
        data_for_boxcox = df_bm[selected_metric] + 0.0001
        df_bm['boxcox'], lam = stats.boxcox(data_for_boxcox)
    else:
        st.warning("Box-Cox skipped: Data contains negative values.")

    def check_stationarity(series):
        # ADF Test
        adf_result = adfuller(series, autolag='AIC')
        # KPSS Test
        kpss_result = kpss(series, regression='c', nlags='auto')

        st.subheader("Stationarity Test Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**ADF Test**")
            st.write(f"Statistic: {adf_result[0]:.4f}")
            st.write(f"p-value: {adf_result[1]:.4f}")
        with col2:
            st.write("**KPSS Test**")
            st.write(f"Statistic: {kpss_result[0]:.4f}")
            st.write(f"p-value: {kpss_result[1]:.4f}")
        # Interpretation
        if adf_result[1] < 0.05 < kpss_result[1]:
            st.success("Result: The series is STATIONARY.")
        elif adf_result[1] >= 0.05 > kpss_result[1]:
            st.error("Result: The series is NON-STATIONARY (needs another transformation method).")
        else:
            st.warning("Result: The tests are contradictory (trend-stationary or complex).")


    st.write("### Testing Stationarity for Transformations")
    transformation = st.selectbox("Choose Transformation to Test", [selected_metric,'diff', 'diff_2','sqrt', 'log', 'boxcox'])
    check_stationarity(df_bm[transformation])


    def get_baseline_metrics(df, selected_metric):
        df = df.copy()
        split_idx = int(len(df) * 0.85)
        df = df.iloc[split_idx:].copy()
        y_true = df[selected_metric]

        # Naive Metrics
        y_pred_naive = df[lag_col]
        mae_nf = mean_absolute_error(y_true, y_pred_naive)
        rmse_nf = np.sqrt(mean_squared_error(y_true, y_pred_naive))
        mape_nf = np.mean(np.abs((y_true - y_pred_naive) / (y_true + 1))) * 100

        threshold = np.percentile(y_true, 90)
        rush_hours = df[df[selected_metric] > threshold]
        peak_err_nf = mean_absolute_error(rush_hours[selected_metric], rush_hours[lag_col])

        # Moving Avg Metrics
        y_pred_ma = df[roll_col]
        mae_ma = mean_absolute_error(y_true, y_pred_ma)
        rmse_ma = np.sqrt(mean_squared_error(y_true, y_pred_ma))
        mape_ma = np.mean(np.abs((y_true - y_pred_ma) / (y_true + 1))) * 100
        peak_err_ma = mean_absolute_error(rush_hours[selected_metric], rush_hours[roll_col])

        return (mae_nf, rmse_nf, mape_nf, peak_err_nf), (mae_ma, rmse_ma, mape_ma, peak_err_ma)


    naive_metrics, ma_metrics = get_baseline_metrics(df_bm, selected_metric)

    st.markdown("---")

    def evaluate_arima(df, selected_metric):
        # 1. Prepare Daily Data
        df = df.copy()
        exog = pd.get_dummies(df['day_of_week'], prefix='day', drop_first=True).astype(int)

        # 2. Split
        split_idx = int(len(df) * 0.85)
        train = df.iloc[:split_idx][selected_metric]
        test = df.iloc[split_idx:][selected_metric]

        train_exog = exog.iloc[:split_idx]
        test_exog = exog.iloc[split_idx:]

        # 3. Train Model
        model = SARIMAX(train,
                        exog=train_exog,
                        order=(2, 1, 1),
                        seasonal_order=(1, 0, 1, 30),
                        trend='c',
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        results = model.fit(disp=False)

        # 4. Forecast with explicit index alignment
        forecast = results.forecast(steps=len(test),exog=test_exog)
        # Ensure forecast has the same index as test
        y_pred = pd.Series(forecast.values, index=test.index)
        y_pred = y_pred.replace([np.inf, -np.inf], 0).fillna(0)

        # 5. Metrics
        mae = mean_absolute_error(test, y_pred)
        rmse = np.sqrt(mean_squared_error(test, y_pred))
        mape = np.mean(np.abs((test - y_pred) / (test + 1))) * 100

        threshold = np.percentile(test, 90)
        mask = test > threshold
        peak_err = mean_absolute_error(test[mask], y_pred[mask]) if mask.sum() > 0 else 0

        return mae, rmse, mape, peak_err

    def evaluate_ets(df, selected_metric):
        # 1. Split
        split_idx = int(len(df) * 0.85)
        train = df[selected_metric].iloc[:split_idx]
        test = df[selected_metric].iloc[split_idx:]

        # 2. Train Holt-Winters
        # trend='add': Handles your steady growth
        # seasonal='add': Handles your 30-day rhythm
        # seasonal_periods=30: Matches the cycle you discovered!
        model = ExponentialSmoothing(train,
                                     trend='add',
                                     damped_trend=True,
                                     seasonal='add',
                                     seasonal_periods=30)

        results = model.fit(optimized=True)

        # 3. Forecast
        y_pred = results.forecast(len(test))

        # 4. Metrics
        mae = mean_absolute_error(test, y_pred)
        rmse = np.sqrt(mean_squared_error(test, y_pred))
        mape = np.mean(np.abs((test - y_pred) / (test + 1))) * 100

        threshold = np.percentile(test, 90)
        mask = test > threshold
        peak_err = mean_absolute_error(test[mask], y_pred[mask]) if mask.sum() > 0 else 0

        return mae, rmse, mape, peak_err

    def evaluate_prophet(df, selected_metric):
        # 1. Prepare data for Prophet
        df_prophet = df.reset_index()[['date', selected_metric]]
        df_prophet.columns = ['ds', 'y']

        # 2. Split
        split_idx = int(len(df_prophet) * 0.85)
        train = df_prophet.iloc[:split_idx]
        test = df_prophet.iloc[split_idx:]

        # 3. Train
        # Prophet handles seasonality automatically!
        model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=True)
        model.fit(train)

        # 4. Forecast
        future = model.make_future_dataframe(periods=len(test), freq='D')
        forecast = model.predict(future)

        # Extract only the test period
        y_pred = forecast.iloc[split_idx:]['yhat'].values
        y_actual = test['y'].values

        # 5. Metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1))) * 100

        threshold = np.percentile(y_actual, 90)
        mask = y_actual > threshold
        peak_err = mean_absolute_error(y_actual[mask], y_pred[mask]) if mask.sum() > 0 else 0

        return mae, rmse, mape, peak_err


    def evaluate_xgboost(df, selected_metric):
        # 1. Feature Engineering: XGBoost needs 'features', not just a date index
        # It learns from these columns
        df = df.copy()
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month

        # Lag features (The "Memory")
        df['lag_1'] = df[selected_metric].shift(1)
        df['lag_7'] = df[selected_metric].shift(7)
        df = df.dropna()

        # 2. Split
        split_idx = int(len(df) * 0.85)
        features = ['day_of_week', 'day_of_year', 'month', 'lag_1', 'lag_7']

        X = df[features]
        y = df[selected_metric]

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 3. Train
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        model.fit(X_train, y_train)

        # 4. Forecast
        y_pred = model.predict(X_test)

        # 5. Metrics (Same as your standard block)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100

        threshold = np.percentile(y_test, 90)
        mask = y_test > threshold
        peak_err = mean_absolute_error(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0

        return mae, rmse, mape, peak_err


    arima_metrics = evaluate_arima(df_bm, selected_metric)
    expo_metrics = evaluate_ets(df_bm, selected_metric)
    prophet_metrics = evaluate_prophet(df_bm, selected_metric)
    xgboost_metrics = evaluate_xgboost(df_bm, selected_metric)

    results_data = [
        ["Naive Baseline", *naive_metrics],
        ["Moving Avg", *ma_metrics],
        ["ARIMA", *arima_metrics],
        ["Exponential Smoothing", *expo_metrics],
        ["Prophet", *prophet_metrics],
        ["xgboost", *xgboost_metrics]
    ]

    df_results = pd.DataFrame(results_data, columns=["Model", "MAE", "RMSE", "MAPE", "Peak Err"])


    def highlight_vs_baseline(row):
        # Create an empty list of styles, default to empty string
        styles = [''] * len(row)

        # Only run logic if this IS NOT a baseline model
        if row['Model'] not in ["Naive Baseline", "Moving Avg"]:
            naive_row = df_results.iloc[0]
            ma_row = df_results.iloc[1]
            # Loop ONLY through columns 1 to 4 (MAE, RMSE, MAPE, Peak Err)
            for i in range(1, len(row)):
                metric_name = row.index[i]
                val = row[metric_name]
                # Now we only compare numeric values!
                if val < naive_row[metric_name] and val < ma_row[metric_name]:
                    styles[i] = 'background-color: #28a745; color: white; font-weight: bold'  # Green

                elif val > naive_row[metric_name] and val > ma_row[metric_name]:
                    styles[i] = 'background-color: #dc3545; color: white; font-weight: bold'  # Red

                else:
                    # This captures the "in-between" cases
                    styles[i] = 'background-color: #007bff; color: white; font-weight: bold'  # Blue (Neutral/Average)

        return styles

    # When applying, ensure you use axis=1 and pass the styles to the dataframe
    styled_table = df_results.style.apply(highlight_vs_baseline, axis=1)
    st.subheader("Model Performance vs. Baseline", help = 'Red = Error bigger than both baseline models, Green = Error smaller than both baseline models, Blue = Error in between both models.')
    st.dataframe(styled_table, width='stretch')

    st.markdown('---')

    def plot_june_backtest_comparison(df, selected_metric, selected_models):
        fig = go.Figure()

        # --- 1. Common Data Prep ---
        split_idx = int(len(df) * 0.85)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()

        # Add Actuals
        fig.add_trace(go.Scatter(x=test.index, y=test[selected_metric], name='Actuals (June)',
                                 line=dict(color='white', width=2)))

        # --- 2. ETS ---
        if 'ETS' in selected_models:
            model = ExponentialSmoothing(train[selected_metric], trend='add', damped_trend=True,
                                         seasonal='add', seasonal_periods=30).fit(optimized=True)
            preds = model.forecast(len(test))
            fig.add_trace(go.Scatter(x=test.index, y=preds, name='ETS', line=dict(color='#28a745')))

        # --- 3. ARIMA (Using your exact exog logic) ---
        if 'ARIMA' in selected_models:
            # Recreate exog
            exog = pd.get_dummies(df['day_of_week'], prefix='day', drop_first=True).astype(int)
            train_exog, test_exog = exog.iloc[:split_idx], exog.iloc[split_idx:]

            model = SARIMAX(train[selected_metric], exog=train_exog, order=(2, 1, 1),
                            seasonal_order=(1, 0, 1, 30), trend='c',
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            preds = model.forecast(steps=len(test), exog=test_exog)
            fig.add_trace(go.Scatter(x=test.index, y=preds, name='ARIMA', line=dict(color='#dc3545', dash='dot')))

        # --- 4. Prophet (Using your exact prep) ---
        if 'Prophet' in selected_models:
            df_p = df.reset_index()[['date', selected_metric]]
            df_p.columns = ['ds', 'y']
            m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=True).fit(
                df_p.iloc[:split_idx])
            forecast = m.predict(m.make_future_dataframe(periods=len(test), freq='D'))
            preds = forecast.iloc[split_idx:]['yhat']
            fig.add_trace(go.Scatter(x=test.index, y=preds, name='Prophet', line=dict(color='#ffc107')))

        # --- 5. XGBoost (Using your exact feature engineering) ---
        if 'XGBoost' in selected_models:
            df_x = df.copy()
            df_x['day_of_week'] = df_x.index.dayofweek
            df_x['day_of_year'] = df_x.index.dayofyear
            df_x['month'] = df_x.index.month
            df_x['lag_1'] = df_x[selected_metric].shift(1)
            df_x['lag_7'] = df_x[selected_metric].shift(7)
            df_x = df_x.dropna()

            features = ['day_of_week', 'day_of_year', 'month', 'lag_1', 'lag_7']
            X_train, X_test = df_x[features].iloc[:split_idx], df_x[features].iloc[split_idx:]
            y_train = df_x[selected_metric].iloc[:split_idx]

            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5).fit(X_train, y_train)
            preds = model.predict(X_test)
            fig.add_trace(go.Scatter(x=test.index, y=preds, name='XGBoost', line=dict(color='#ff69b4', dash='dash')))

        # Final Layout
        fig.update_layout(title="June Back-test: Performance Comparison", template="plotly_dark",
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig)


    # --- Update your multiselect ---
    st.subheader('Model Comparison Chart')
    selected_models = st.multiselect("Select Models for June Comparison",
                                     ['ETS', 'Prophet', 'ARIMA', 'XGBoost'],
                                     default=['ETS', 'XGBoost'])

    plot_june_backtest_comparison(df_bm, selected_metric, selected_models)


with tab4:
    st.subheader("Model Accuracy KPIs")

    def plot_production_forecast(df, selected_metric, horizon, mape):
        df_clean = df.asfreq('D').ffill()

        # 2. Train ETS (The Champion Model)
        model = ExponentialSmoothing(df_clean[selected_metric],
                                     trend='add',
                                     damped_trend=True,
                                     seasonal='add',
                                     seasonal_periods=30)
        results = model.fit(optimized=True)

        # 3. Forecast
        # Calculate start date from the last index + 1 day
        start_date = df_clean.index[-1] + pd.Timedelta(days=1)
        forecast_dates = pd.date_range(start=start_date, periods=horizon)
        y_pred = results.forecast(horizon)

        # 4. Confidence Intervals (Using your MAPE)
        # Margin is prediction * (MAPE / 100)
        margin = y_pred * (mape / 100)
        upper = y_pred + margin
        lower = y_pred - margin

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                                 y=upper.tolist() + lower.tolist()[::-1],
                                 fill='toself', fillcolor='rgba(40, 167, 69, 0.15)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name='Confidence (MAPE)'))
        fig.add_trace(go.Scatter(x=forecast_dates, y=y_pred, name='ETS Forecast',
                                 line=dict(color='#28a745', width=3)))
        fig.update_layout(title=f"July Sales Forecast: Next {horizon} Days",
                          template="plotly_dark",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig)

    ## KPIs
    ets_mae, ets_rmse, ets_mape, ets_peak = evaluate_ets(df_bm, selected_metric)
    if selected_metric == "revenue":
        # MAE is already in Dollars
        error_label = "Revenue Error"
        error_value = f"${ets_mae:.2f}"
    else:
        # MAE is in Units
        error_label = "Unit Error"
        error_value = f"{ets_mae:.1f} Units"

    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("Accuracy", f"{100 - ets_mape:.1f}%")
    with col2:
        kpi_card("Peak Capture",
                 f"{(1 - (ets_peak / np.percentile(df_bm[selected_metric].iloc[int(len(df_bm) * 0.85):], 90))) * 100:.1f}%")
    with col3:
        kpi_card(error_label, error_value)
    st.markdown('---')

    st.subheader("Forecasting Chart")
    horizon = st.slider("Select Horizon (Days)", 7, 30, 30)
    plot_production_forecast(df_bm, selected_metric, horizon, expo_metrics[2])
    st.markdown('---')

    def get_hourly_fingerprint(df_eval,seleted_metric):
        # Ensure date/time are ready
        df = df_eval.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['day_name'] = df['date'].dt.day_name()

        # Calculate daily totals for every day in your history
        daily_totals = df.groupby('date')[seleted_metric].transform('sum')

        # Calculate the ratio for every hour
        df['hourly_ratio'] = df[seleted_metric] / daily_totals

        # Average the ratio for each day-of-week + hour
        fingerprint = df.groupby(['day_name', 'hour'])['hourly_ratio'].mean().reset_index()
        return fingerprint


    def plot_hourly_rush_hour(df_eval):
        # Get the fingerprint
        fingerprint = get_hourly_fingerprint(df_eval,selected_metric)

        # Pivot for the heatmap: Hour as Y, Day as X
        pivot_df = fingerprint.pivot(index='hour', columns='day_name', values='hourly_ratio')

        # Reorder columns so the week is in order
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_df = pivot_df[days_order]

        # Plotly Heatmap
        fig = px.imshow(pivot_df,
                        labels=dict(x="Day of Week", y="Hour of Day", color="Demand %"),
                        color_continuous_scale="RdYlGn_r",
                        aspect="auto")

        fig.update_layout(title="Rush-Hour Demand Heat Map",
                          template="plotly_dark",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          coloraxis_colorbar=dict(
                              title="Share of Daily Demand",
                              tickformat=".0%"  # This converts 0.15 to 15%
                          )
                          )

        st.plotly_chart(fig)

    st.subheader("Potential Heatmap")
    plot_hourly_rush_hour(df_eval)

with tab5:
    st.header("Strategic Business Overview")

    st.markdown("""
    ---
    ### Executive Summary: Performance & Strategic Outlook
    *Reporting Period: 6-Month Rolling Analysis (All Stores Aggregated)*

    #### 1. High-Level Performance
    The business is currently operating in a **strong growth phase**, characterized by a 6-month upward trend in transaction volume across core categories. 
    * **Operational Health:** The forecasting engine currently maintains a **92.0% accuracy rate**, demonstrating robust predictive capability for inventory planning.
    * **Core Revenue Drivers:** Coffee remains the primary volume driver, while the "Menu Engineering" analysis reveals a healthy distribution of "Star" products (high popularity, high revenue).

    #### 2. Operational Rhythm (The "Fingerprint")
    Our analysis of store-wide behavior confirms a distinct **Morning Peak Operational Window (08:00 – 11:00)**. 
    * **The "Midday Cliff":** Data shows a sharp decline in transaction volume post-11:00 AM. This presents a clear strategic opportunity to launch "Midday Refresh" promotions to smooth out the demand curve and improve capacity utilization.
    * **Weekend Dynamics:** While weekday mornings are driven by "speed-oriented" coffee demand, weekends exhibit a higher basket size and broader category participation, suggesting a shift from "functional" to "leisure" consumption.

    #### 3. Strategic Growth Opportunities
    * **Cross-Sell Potential:** Strong correlations between **Coffee-Bakery** and **Tea-Bakery** categories suggest that targeted "Bundle" promotions could increase the average transaction value.
    * **Menu Optimization:** The "Menu Engineering" matrix identifies a clear cluster of "Puzzles" (low-popularity/high-revenue) that require targeted marketing to drive traffic, and "Workhorses" that should be the focus of inventory stability efforts.

    #### 4. Forecasting & Risk Mitigation
    The "Peak Capture" rate of **90.2%** ensures that our inventory planning is successfully accounting for high-pressure rush hours. 
    * **Actionable Insight:** Inventory replenishment should be synchronized with the "Typical Weekly Fingerprint" to ensure that the 08:00–11:00 rush is never hampered by supply chain bottlenecks.
    ---
    """)

    # Interactive note for the user
    st.info(
        "💡 **Navigation Tip:** This summary represents an aggregate view of all locations. To analyze a specific store or category, please use the **Dashboard Filters** sidebar on the left.")

import streamlit.components.v1 as components

with st.sidebar:
    st.markdown("---")
    st.subheader("Reporting")

    # 1. Inject CSS to make the page print-friendly
    st.markdown("""
        <style>
        @media print {
            /* Force background colors to show in PDF */
            body, .main, .stApp {
                background-color: #1a1a1a !important; 
                color: white !important;
            }
            /* Ensure the content is visible */
            .stApp { visibility: visible; }
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. Updated Dark-Themed Button
    print_button = """
    <script>
        function printPage() { window.print(); }
    </script>
    <button onclick="printPage()" style="
        width: 100%; 
        padding: 12px; 
        cursor: pointer; 
        border-radius: 8px; 
        border: none; 
        background-color: #333333; 
        color: #ffffff;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        🖨️ Download Dashboard as PDF
    </button>
    """
    components.html(print_button, height=60)
