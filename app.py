import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import plotly.graph_objects as go

# == CONSTANTAS ==
NUM_BINS = 8
MIN_BIN_SIZE = 3
FIG_3D_FONT_SIZE = 9


df_full = pd.read_csv('models/full_data.csv')
with open('models/all_feature_importances.json', 'r') as f:
    importance_dict = json.load(f)

data_source_info = pd.read_csv('data/data_source_info.csv')
all_cols = list(df_full.columns)
plot_options = [col for col in all_cols if col not in ['Country Name', 'Country', 'Unnamed: 0']]

# == Begin App Layout ==
st.set_page_config(layout="wide")
st.title("Global Indicator Trends and Relationships by Country")
selected_column = st.selectbox("Select data to plot...", plot_options)


target_importances = importance_dict.get(selected_column)
df_features = pd.DataFrame(list(target_importances.items()), columns=["Feature", "Importance"])
df_features_sorted = df_features.sort_values(by='Importance', ascending=False)

df_full = df_full.round(1)
df_features_sorted = df_features_sorted.round(1)

col1, col2 = st.columns([3, 2])

with col1:

    # == Choropleth Graph ==
    fig_choro = px.choropleth(
    df_full,
    locations='Country',
    color=selected_column,
    hover_name='Country Name',
    color_continuous_scale="YlGnBu",
    projection='natural earth'
    )
    fig_choro.update_layout(
        autosize=True,
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.subheader(f"{selected_column} by Country")
    st.plotly_chart(fig_choro, use_container_width=True)
    st.caption('Countries without reliable data are excluded.')


    # == Bar Graph ==
    plot_options_2 = plot_options.copy()
    plot_options_2.remove(selected_column)
    selected_value_2 = st.selectbox('Select a variable to view relationship...',plot_options_2)

    x_min = df_full[selected_value_2].min()
    x_max = df_full[selected_value_2].max()
    bin_step = (x_max - x_min) / NUM_BINS

    bins = pd.cut(df_full[selected_value_2], bins=np.arange(x_min,x_max,bin_step))
    df_full['x_bin'] = bins

    bin_counts = df_full['x_bin'].value_counts().to_dict()

    df_filtered = df_full[df_full['x_bin'].map(bin_counts) >= MIN_BIN_SIZE]

    df_binned = df_filtered.groupby('x_bin')[selected_column].agg(['mean','count']).reset_index()
    df_binned.columns = [selected_value_2,f'{selected_column} mean','count']
    df_binned[selected_value_2] = df_binned[selected_value_2].astype(str)

    df_binned = df_binned[df_binned['count'] > 0]
    df_binned['x_label'] = df_binned.apply(
        lambda row: f"{row[selected_value_2]}<br>(n={row['count']})", axis=1
    )

    fig_bar = px.bar(
        df_binned,
        x='x_label',
        y=f'{selected_column} mean',
        labels={
            'x_label' : selected_value_2,
            f'{selected_column} mean': f'Mean {selected_column}'
        },
        title=f'Relationship between {selected_value_2} and {selected_column}'
    )
    fig_bar.update_traces(
        marker_color='skyblue',
        marker_line_color='black',
        marker_line_width=1
    )
    st.plotly_chart(fig_bar,use_container_width=True)
    st.caption(f'Bins with less than {MIN_BIN_SIZE} data points are excluded from the graphs.')


with col2:

    # == Feature Importance Graph
    fig_hist = px.histogram(
        df_features_sorted,
        x='Feature',
        y='Importance',
        title=f'Magnitude of Importance of Contributing Factors (Features) on {selected_column}',
    )
    fig_hist.update_layout(
        xaxis_title='Feature',
        yaxis_title='Magnitude of Feature Importance',
        font=dict(size=14)

    )
    fig_hist.update_traces(
        marker_color='skyblue',
        marker_line_color='black',
        marker_line_width=1
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(f"**This chart shows which factors most strongly influence {selected_column} " \
    f"across countries.** A Random Forest model was trained using other country-level variables to predict {selected_column}, "
    "and the bar heights represent each feature's relative importance in that prediction. Higher bars indicate greater " \
    "influence on the model’s output.")

    # == 3D Graph ==

    # align text boxes from sperate columns
    st.markdown("<div style='height:118px;'></div>", unsafe_allow_html=True)

    plot_options_3 = plot_options_2.copy()
    plot_options_3.remove(selected_value_2)
    selected_value_3 = st.selectbox('Select a third variable to view relationship...', plot_options_3)

    x_col, y_col, z_col = selected_value_2, selected_value_3, selected_column
    df_clean = df_full[[x_col, y_col, z_col]].dropna()

    x_min, x_max = df_clean[x_col].min(), df_clean[x_col].max()
    y_min, y_max = df_clean[y_col].min(), df_clean[y_col].max()

    x_bins = np.linspace(x_min, x_max, NUM_BINS + 1)
    y_bins = np.linspace(y_min, y_max, NUM_BINS + 1)

    df_clean['x_bin'] = pd.cut(df_clean[x_col], bins=x_bins)
    df_clean['y_bin'] = pd.cut(df_clean[y_col], bins=y_bins)

    grouped = df_clean.groupby(['x_bin','y_bin'])[z_col].agg(['mean','count']).reset_index()
    grouped = grouped[grouped['count'] >= MIN_BIN_SIZE]

    grouped['x_mid'] = grouped['x_bin'].apply(lambda i: i.mid)
    grouped['y_mid'] = grouped['y_bin'].apply(lambda i: i.mid)


    # Main point cloud (Z is the mean value per bin)
    points = go.Scatter3d(
        x=grouped['x_mid'],
        y=grouped['y_mid'],
        z=grouped['mean'],
        mode='markers',
        marker=dict(
            size=10,
            color=grouped['mean'],
            colorscale='YlGnBu',
            colorbar=dict(title=f'Mean {z_col}'),
            opacity=0.8
        ),
        text=grouped['count'].apply(lambda n: f'n={n}')
    )

    # === Vertical bars: lines from z=0 to z=mean
    bar_lines = []
    for _, row in grouped.iterrows():
        bar_lines.append(go.Scatter3d(
            x=[row['x_mid'], row['x_mid']],
            y=[row['y_mid'], row['y_mid']],
            z=[0, row['mean']],
            mode='lines',
            line=dict(color='gray', width=20),
            showlegend=False,
            hoverinfo='skip'
        ))

    # === Combine and plot
    fig = go.Figure(data=[points] + bar_lines)

    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=f'Mean {z_col}',
            xaxis=dict(titlefont=dict(size=FIG_3D_FONT_SIZE)),
            yaxis=dict(titlefont=dict(size=FIG_3D_FONT_SIZE)),
            zaxis=dict(titlefont=dict(size=FIG_3D_FONT_SIZE)),
        ),
        title=f'{z_col} Dependance on {x_col} and {y_col}'
    )

    st.plotly_chart(fig)

st.markdown(data_source_info.to_html(index=False), unsafe_allow_html=True)