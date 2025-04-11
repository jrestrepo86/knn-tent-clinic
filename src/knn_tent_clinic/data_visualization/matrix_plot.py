import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from knn_tent_clinic import MATRIX_CHANNELS_ORDER as ch_order


def make_matrices_plot(df):

    data_array = []
    freq_bands = df["freq-band"].unique()
    cmin = df["tent_diff"].min()
    cmax = df["tent_diff"].max()
    for band in freq_bands:
        data_array.append(df[df["freq-band"] == band])
    fig = make_subplots(data_array[0])
    fig.frames = create_frames(data_array)
    fig = plot_layout(fig, cmin, cmax)
    fig = add_lines(fig)
    return fig


def df_to_matrix(df, measure):
    # Initialize the matrix with NaN values
    matrix = pd.DataFrame(np.nan, index=ch_order, columns=ch_order)

    # Populate the matrix symmetrically
    for _, row in df.iterrows():
        src = row["source"]
        tgt = row["target"]
        value = row[f"{measure}"]
        matrix.loc[src, tgt] = value
        matrix.loc[tgt, src] = value  # Ensure symmetry

    # Create a mask for the upper triangular matrix
    mask = np.triu(np.ones(matrix.shape, dtype=bool), k=0)
    upper_tri_matrix = matrix.where(mask)

    return upper_tri_matrix


def create_matrix(df, measure):

    mat_values = df_to_matrix(df, measure)
    coloraxis_val = "" if measure == "tent_diff" else 2
    heatmap = go.Heatmap(
        x=ch_order,
        y=ch_order,
        z=mat_values,
        # colorscale="Viridis",
        coloraxis=f"coloraxis{coloraxis_val}",
    )
    return heatmap


def create_frames(data_array):
    frames = []
    for data in data_array:
        freq_band = data["freq-band"].unique()
        tent_heatmap = create_matrix(data, "tent_diff")
        flow_heatmap = create_matrix(data, "flow")
        frame = go.Frame(
            data=[tent_heatmap, flow_heatmap],
            name=f"{freq_band}",
            traces=[0, 1],  # Associate with both heatmap traces
        )
        frames.append(frame)
    return frames


def make_subplots(df):
    fig = sp.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Tent Matrix", "Flow Matrix"),
        horizontal_spacing=0.15,
        column_widths=[0.5, 0.5],
    )
    # initial traces
    fig.add_trace(
        create_matrix(df, "tent_diff"),
        row=1,
        col=1,
    )
    fig.add_trace(
        create_matrix(df, "flow"),
        row=1,
        col=2,
    )
    return fig


def add_lines(fig):
    for col in [1, 2]:
        # Vertical lines
        fig.add_vline(
            x=7.5,
            line_width=3,
            line_dash="dash",
            line_color="RoyalBlue",
            row=1,
            col=col,
        )
        fig.add_vline(
            x=11.5,
            line_width=3,
            line_dash="dash",
            line_color="RoyalBlue",
            row=1,
            col=col,
        )

        # Horizontal lines
        fig.add_hline(
            y=7.5,
            line_width=3,
            line_dash="dash",
            line_color="RoyalBlue",
            row=1,
            col=col,
        )
        fig.add_hline(
            y=11.5,
            line_width=3,
            line_dash="dash",
            line_color="RoyalBlue",
            row=1,
            col=col,
        )

        # Diagonal line
        xref = f"x{col}" if col > 1 else "x"
        yref = f"y{col}" if col > 1 else "y"
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=-0.5,
            x1=len(ch_order) - 0.5,
            y1=len(ch_order) - 0.5,
            line=dict(color="RoyalBlue", width=3, dash="dot"),
            xref=xref,
            yref=yref,
        )
    return fig


def plot_layout(fig, cmin, cmax):

    fig.update_layout(
        coloraxis=dict(
            colorscale="sunsetdark",
            colorbar=dict(
                title="Tent",
                x=0.45,
                y=0.5,
                yanchor="middle",
                len=0.4,
            ),
            cmin=cmin,
            cmax=cmax,
        ),
        coloraxis2=dict(
            colorscale="sunsetdark",
            colorbar=dict(
                title="Flow",
                x=1.05,
                y=0.5,
                yanchor="middle",
                len=0.4,
            ),
            cmin=-1,
            cmax=1,
        ),
        xaxis=dict(
            tickvals=list(range(len(ch_order))),
            ticktext=ch_order,
            nticks=len(ch_order) + 1,
        ),
        yaxis=dict(
            tickvals=list(range(len(ch_order))),
            ticktext=ch_order,
            nticks=len(ch_order) + 1,
            autorange="reversed",
        ),
        xaxis2=dict(
            tickvals=list(range(len(ch_order))),
            ticktext=ch_order,
            nticks=len(ch_order) + 1,
        ),
        yaxis2=dict(
            tickvals=list(range(len(ch_order))),
            ticktext=ch_order,
            nticks=len(ch_order) + 1,
            autorange="reversed",
        ),
        title_text="Tent and Flow Matrices",
        title_x=0.5,
        showlegend=False,
        # Critical for maintaining color bars during animation
        coloraxis_showscale=True,
        coloraxis2_showscale=True,
    )

    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Frequency Band:",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 300},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": -0.3,
                "steps": [
                    {
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": f"{frame.name}",
                        "method": "animate",
                    }
                    for frame in fig.frames
                ],
            }
        ],
    )
    return fig
