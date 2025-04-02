import plotly.express as px
import plotly.graph_objects as go

CHANNELS_ORDER = [
    "F1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T3",
    "T5",
    "Fz",
    "Cz",
    "Pz",
    "Oz",
    "T6",
    "T4",
    "F8",
    "O2",
    "P4",
    "C4",
    "F4",
    "F2",
]


def tent_matrix_plot(tent_results):

    # Pivot the DataFrame to create a matrix
    pivot_df = tent_results.pivot(index="target", columns="source", values="Tent")
    pivot_df = pivot_df.reindex(index=CHANNELS_ORDER, columns=CHANNELS_ORDER)

    # Create the heatmap using Plotly
    fig = px.imshow(
        pivot_df,
        labels={"x": "Source", "y": "Target", "color": "Tent"},
        x=CHANNELS_ORDER,
        y=CHANNELS_ORDER,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        title="Tent Matrix: Target vs Source",
        xaxis_nticks=len(CHANNELS_ORDER),
        yaxis_nticks=len(CHANNELS_ORDER),
    )
    return fig


def flow_matrix_plot(tent_results):

    flow_results = tent_results.copy()
    # Pivot the DataFrame to create a matrix
    pivot_df = flow_results.pivot(index="target", columns="source", values="Flow")
    pivot_df = pivot_df.reindex(index=CHANNELS_ORDER, columns=CHANNELS_ORDER)

    # Create the heatmap using Plotly
    fig = px.imshow(
        pivot_df,
        labels={"x": "Source", "y": "Target", "color": "Flow"},
        x=CHANNELS_ORDER,
        y=CHANNELS_ORDER,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        title="Tent Matrix: Target vs Source",
        xaxis_nticks=len(CHANNELS_ORDER),
        yaxis_nticks=len(CHANNELS_ORDER),
    )
    return fig
