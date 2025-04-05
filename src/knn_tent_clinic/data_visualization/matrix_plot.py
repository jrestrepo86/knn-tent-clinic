import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

from knn_tent_clinic import MATRIX_CHANNELS_ORDER

channels_order = MATRIX_CHANNELS_ORDER


def tent_matrix_plot(tent_results):

    # Pivot the DataFrame to create a matrix
    pivot_df = tent_results.pivot(index="source", columns="target", values="tent")
    pivot_df = pivot_df.reindex(index=channels_order, columns=channels_order)

    # Create the heatmap using Plotly
    fig = px.imshow(
        pivot_df,
        labels={"y": "Source", "x": "Target", "color": "tent"},
        x=channels_order,
        y=channels_order,
        color_continuous_scale="Viridis",
    )
    fig = plot_lines(fig)
    fig.update_layout(
        title="Tent Matrix: Source ⇨ Target",
        xaxis_nticks=len(channels_order) + 1,
        yaxis_nticks=len(channels_order) + 1,
    )
    return fig


def plot_lines(fig, color="RoyalBlue"):
    lw = 3
    fig.add_vline(x=7.5, line_width=lw, line_dash="dash", line_color=color)
    fig.add_vline(x=11.5, line_width=lw, line_dash="dash", line_color=color)
    fig.add_hline(y=7.5, line_width=lw, line_dash="dash", line_color=color)
    fig.add_hline(y=11.5, line_width=lw, line_dash="dash", line_color=color)
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=-0.5,
        x1=19.5,
        y1=19.5,
        line=dict(color=color, width=lw, dash="dot"),
    )
    return fig


def flow_matrix_plot(tent_results):

    flow_results = tent_results.copy()
    # Pivot the DataFrame to create a matrix
    pivot_df = flow_results.pivot(index="source", columns="target", values="flow")
    pivot_df = pivot_df.reindex(index=channels_order, columns=channels_order)

    # Create the heatmap using Plotly
    fig = px.imshow(
        pivot_df,
        labels={"y": "Source", "x": "Target", "color": "flow"},
        x=channels_order,
        y=channels_order,
        color_continuous_scale="Viridis",
    )
    fig = plot_lines(fig)
    fig.update_layout(
        title="Tent Matrix: Source ⇨ Target",
        xaxis_nticks=len(channels_order) + 1,
        yaxis_nticks=len(channels_order) + 1,
    )
    return fig


# def make_matrices_plots(tent_results):
#
#     # Generate individual figures
#     fig_tent = tent_matrix_plot(tent_results)
#     fig_flow = flow_matrix_plot(tent_results)
#
#     # Create subplots with independent color scales
#     fig = sp.make_subplots(
#         rows=1,
#         cols=2,
#         subplot_titles=("Tent Matrix: Source ⇨ Target", "Flow Matrix: Source ⇨ Target"),
#         horizontal_spacing=0.15,
#     )
#
#     # Add traces with separate color axes
#     fig.add_trace(fig_tent.data[0], row=1, col=1)
#     fig.add_trace(fig_flow.data[0], row=1, col=2)

# # Adjust shape references for subplots
# def adjust_shapes(shapes, xref, yref):
#     adjusted = []
#     for shape in shapes:
#         s = shape.to_plotly_json().copy()
#         if "xref" in s and s["xref"] == "x":
#             s["xref"] = xref
#         if "yref" in s and s["yref"] == "y":
#             s["yref"] = yref
#         adjusted.append(s)
#     return adjusted
#
# # Add shapes to appropriate subplots
# for shape in adjust_shapes(fig_tent.layout.shapes, "x1", "y1"):
#     fig.add_shape(shape, row=1, col=1)
# for shape in adjust_shapes(fig_flow.layout.shapes, "x2", "y2"):
#     fig.add_shape(shape, row=1, col=2)
#
# # Update layout with independent color bars
# fig.update_layout(
#     # Left plot settings
#     xaxis1=dict(title="Target", nticks=len(channels_order)) + 1,
#     yaxis1=dict(title="Source", nticks=len(channels_order)) + 1,
#     # Right plot settings
#     xaxis2=dict(title="Target", nticks=len(channels_order) + 1),
#     yaxis2=dict(title="Source", nticks=len(channels_order) + 1),
#     # Color axis settings
#     coloraxis1=dict(
#         colorbar=dict(title="tent", x=0.45, y=0.5), colorscale="Viridis"
#     ),
#     coloraxis2=dict(
#         colorbar=dict(title="flow", x=1.02, y=0.5), colorscale="Viridis"
#     ),
#     # Remove main title
#     title_text=None,
# )
#
# # Set independent color axes for each trace
# fig.data[0].coloraxis = "coloraxis1"
# fig.data[1].coloraxis = "coloraxis2"
#
# # Adjust annotation positions if needed
# fig.update_annotations(font_size=12)
#
# fig.update_layout(
#     title="Tent and Flow - Source ⇨ Target",
# )


# return fig
def make_matrices_plots(tent_results):
    # Generate pivot DataFrames
    tent_pivot = tent_results.pivot(
        index="source", columns="target", values="tent"
    ).reindex(index=channels_order, columns=channels_order)
    flow_pivot = tent_results.pivot(
        index="source", columns="target", values="flow"
    ).reindex(index=channels_order, columns=channels_order)

    # Create subplot figure
    fig = sp.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Tent Matrix: Source ⇨ Target", "Flow Matrix: Source ⇨ Target"),
        horizontal_spacing=0.15,
        column_widths=[0.5, 0.5],
    )

    # Add tent matrix heatmap
    fig.add_trace(
        go.Heatmap(
            x=channels_order,
            y=channels_order,
            z=tent_pivot.values,
            colorscale="Viridis",
            coloraxis="coloraxis",
            name="Tent",
        ),
        row=1,
        col=1,
    )

    # Add flow matrix heatmap
    fig.add_trace(
        go.Heatmap(
            x=channels_order,
            y=channels_order,
            z=flow_pivot.values,
            colorscale="Viridis",
            coloraxis="coloraxis2",
            name="Flow",
        ),
        row=1,
        col=2,
    )

    # Add lines to both subplots
    n_channels = len(channels_order)
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
            x1=n_channels - 0.5,
            y1=n_channels - 0.5,
            line=dict(color="RoyalBlue", width=3, dash="dot"),
            xref=xref,
            yref=yref,
        )

    # Update layout with dual colorbars and axis labels
    fig.update_layout(
        coloraxis=dict(
            colorbar=dict(title="Tent", x=0.45, y=0.5, yanchor="middle", len=0.4)
        ),
        coloraxis2=dict(
            colorbar=dict(title="Flow", x=1.05, y=0.5, yanchor="middle", len=0.4)
        ),
        xaxis=dict(
            tickvals=list(range(len(channels_order))),
            ticktext=channels_order,
            nticks=len(channels_order) + 1,
        ),
        yaxis=dict(
            tickvals=list(range(len(channels_order))),
            ticktext=channels_order,
            nticks=len(channels_order) + 1,
        ),
        xaxis2=dict(
            tickvals=list(range(len(channels_order))),
            ticktext=channels_order,
            nticks=len(channels_order) + 1,
        ),
        yaxis2=dict(
            tickvals=list(range(len(channels_order))),
            ticktext=channels_order,
            nticks=len(channels_order) + 1,
        ),
        title_text="Tent and Flow Matrices",
        title_x=0.5,
        showlegend=False,
    )

    return fig
