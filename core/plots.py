import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import VAR_COLORS

HEATMAP_METRICS = ['S̄', 'H̄', 'P̄ᵐ']

_LABEL_FONT = dict(color='#2c2c2c', size=18)

# Seafood / harvest: negative % = red (bad), positive % = green (good)
SEAFOOD_COLORSCALE = [[0.0, '#DC143C'], [0.5, '#FFFFFF'], [1.0, '#228B22']]
HARVEST_COLORSCALE = [[0.0, '#DC143C'], [0.5, '#FFFFFF'], [1.0, '#228B22']]
# Market price polarity inverted: higher price vs baseline = red
MARKET_PRICE_COLORSCALE = [[0.0, '#228B22'], [0.5, '#FFFFFF'], [1.0, '#DC143C']]
DEFAULT_DIVERGING_COLORSCALE = [[0.0, '#228B22'], [0.5, '#FFFFFF'], [1.0, '#DC143C']]

_HEATMAP_SERIES_KEY = {'S̄': 'Seafood', 'H̄': 'Harvest', 'P̄ᵐ': 'Market Price'}


def plot_four_variable_time_series(time_series_by_param, time_axis, param_values,
                                   param_label, title, colors=VAR_COLORS):
    num_columns = len(param_values)
    fig = make_subplots(
        rows=2, cols=num_columns,
        subplot_titles=[f'{param_label} = {v}' for v in param_values] + [''] * num_columns,
        shared_xaxes=True, vertical_spacing=0.10, horizontal_spacing=0.05,
    )
    row1_y_max = max(
        max(float(time_series_by_param[v]['Seafood'].max()),
            float(time_series_by_param[v]['Effort'].max()),
            float(time_series_by_param[v]['Harvest'].max()))
        for v in param_values
    )
    for column, param_value in enumerate(param_values, 1):
        series = time_series_by_param[param_value]
        show_legend = column == 1
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Seafood'], mode='lines',
            line=dict(color=colors['S'], width=1.5),
            name='Seafood (S)', legendgroup='S', showlegend=show_legend,
        ), row=1, col=column)
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Effort'], mode='lines',
            line=dict(color=colors['E'], width=1.5),
            name='Effort (E)', legendgroup='E', showlegend=show_legend,
        ), row=1, col=column)
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Harvest'], mode='lines',
            line=dict(color=colors['Harvest'], width=1.5),
            name='Harvest (H)', legendgroup='H', showlegend=show_legend,
        ), row=1, col=column)
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Fraudsters'], mode='lines',
            line=dict(color=colors['F'], width=1.5),
            name='Fraudsters (F)', legendgroup='F', showlegend=show_legend,
        ), row=2, col=column)
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Perception of Fraud'], mode='lines',
            line=dict(color=colors['FP'], width=1.5),
            name='Perception (FP)', legendgroup='FP', showlegend=show_legend,
        ), row=2, col=column)
    fig.update_yaxes(title_text='S / E / H', title_font=_LABEL_FONT, row=1, col=1)
    fig.update_yaxes(title_text='F / FP', title_font=_LABEL_FONT, row=2, col=1)
    fig.update_yaxes(rangemode='tozero')
    fig.update_yaxes(range=[0, row1_y_max * 1.05], row=1)
    fig.update_yaxes(range=[0, 1], row=2)
    fig.update_xaxes(title_text='Time', title_font=_LABEL_FONT, row=2)
    fig.update_annotations(font=_LABEL_FONT)
    fig.update_layout(
        height=600, title_text=title,
        title_y=1.0,
        title_font=dict(color='#1a1a1a', size=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.06, font=dict(color='#2c2c2c', size=14)),
        margin=dict(t=100, b=40),
    )
    return fig


def plot_time_series_with_economics(time_series_by_param, time_axis, param_values,
                                    param_label, title, colors=VAR_COLORS,
                                    show_per_effort=False):
    """Like plot_four_variable_time_series but with extra rows for economic variables:
    Row 1: S / E / H
    Row 2: F / FP
    Row 3: Market Price / Wholesale Price
    Row 4 (optional): Revenue per Effort / Cost per Effort  [show_per_effort=True]
    """
    num_columns = len(param_values)
    num_rows = 4 if show_per_effort else 3
    fig = make_subplots(
        rows=num_rows, cols=num_columns,
        subplot_titles=(
            [f'{param_label} = {v}' for v in param_values]
            + [''] * (num_rows - 1) * num_columns
        ),
        shared_xaxes=True, vertical_spacing=0.07, horizontal_spacing=0.05,
    )
    row1_y_max = max(
        max(float(time_series_by_param[v]['Seafood'].max()),
            float(time_series_by_param[v]['Effort'].max()),
            float(time_series_by_param[v]['Harvest'].max()))
        for v in param_values
    )
    for column, param_value in enumerate(param_values, 1):
        series = time_series_by_param[param_value]
        show_legend = column == 1
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Seafood'], mode='lines',
            line=dict(color=colors['S'], width=1.5),
            name='Seafood (S)', legendgroup='S', showlegend=show_legend,
        ), row=1, col=column)
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Effort'], mode='lines',
            line=dict(color=colors['E'], width=1.5),
            name='Effort (E)', legendgroup='E', showlegend=show_legend,
        ), row=1, col=column)
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Harvest'], mode='lines',
            line=dict(color=colors['Harvest'], width=1.5),
            name='Harvest (H)', legendgroup='H', showlegend=show_legend,
        ), row=1, col=column)
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Fraudsters'], mode='lines',
            line=dict(color=colors['F'], width=1.5),
            name='Fraudsters (F)', legendgroup='F', showlegend=show_legend,
        ), row=2, col=column)
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Perception of Fraud'], mode='lines',
            line=dict(color=colors['FP'], width=1.5),
            name='Perception (FP)', legendgroup='FP', showlegend=show_legend,
        ), row=2, col=column)
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Market Price'], mode='lines',
            line=dict(color=colors['Pm'], width=1.5),
            name='Market Price (Pₘ)', legendgroup='Pm', showlegend=show_legend,
        ), row=3, col=column)
        fig.add_trace(go.Scatter(
            x=time_axis, y=series['Wholesale Price'], mode='lines',
            line=dict(color=colors['Pw'], width=1.5),
            name='Wholesale Price (Pᵥ)', legendgroup='Pw', showlegend=show_legend,
        ), row=3, col=column)
        if show_per_effort:
            fig.add_trace(go.Scatter(
                x=time_axis, y=series['Revenue per Effort'], mode='lines',
                line=dict(color=colors['Rev'], width=1.5),
                name='Revenue / Effort', legendgroup='Rev', showlegend=show_legend,
            ), row=4, col=column)
            fig.add_trace(go.Scatter(
                x=time_axis, y=series['Cost per Effort'], mode='lines',
                line=dict(color=colors['Cost'], width=1.5),
                name='Cost / Effort', legendgroup='Cost', showlegend=show_legend,
            ), row=4, col=column)

    fig.update_yaxes(title_text='S / E / H', title_font=_LABEL_FONT, row=1, col=1)
    fig.update_yaxes(title_text='F / FP', title_font=_LABEL_FONT, row=2, col=1)
    fig.update_yaxes(title_text='Price', title_font=_LABEL_FONT, row=3, col=1)
    if show_per_effort:
        fig.update_yaxes(title_text='Per-Effort', title_font=_LABEL_FONT, row=4, col=1)
    fig.update_yaxes(rangemode='tozero')
    fig.update_yaxes(range=[0, row1_y_max * 1.05], row=1)
    fig.update_yaxes(range=[0, 1], row=2)
    fig.update_xaxes(title_text='Time', title_font=_LABEL_FONT, row=num_rows)
    fig.update_annotations(font=_LABEL_FONT)
    fig.update_layout(
        height=1000 if show_per_effort else 750,
        title_text=title,
        title_y=1.0,
        title_font=dict(color='#1a1a1a', size=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.04, font=dict(color='#2c2c2c', size=14)),
        margin=dict(t=100, b=40),
    )
    return fig


def plot_bifurcation(sweep_values, seafood_values, effort_values, fraudster_values,
                     perception_values, xlabel, title, vline_x=None, vline_label=None,
                     colors=VAR_COLORS):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Seafood S*', 'Effort E*', 'Fraudsters F*', 'Perception FP*'],
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )
    fig.add_trace(go.Scattergl(
        x=sweep_values, y=seafood_values, mode='markers',
        marker=dict(color=colors['S'], size=2, opacity=0.4), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=sweep_values, y=effort_values, mode='markers',
        marker=dict(color=colors['E'], size=2, opacity=0.4), showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scattergl(
        x=sweep_values, y=fraudster_values, mode='markers',
        marker=dict(color=colors['F'], size=2, opacity=0.4), showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scattergl(
        x=sweep_values, y=perception_values, mode='markers',
        marker=dict(color=colors['FP'], size=2, opacity=0.4), showlegend=False,
    ), row=2, col=2)
    if vline_x is not None:
        fig.add_vline(
            x=vline_x, line_dash='dash', line_color='gray',
            annotation_text=vline_label or '',
            annotation_position='top right', row=1, col=1,
        )
        fig.add_vline(x=vline_x, line_dash='dash', line_color='gray', row=1, col=2)
        fig.add_vline(x=vline_x, line_dash='dash', line_color='gray', row=2, col=1)
        fig.add_vline(x=vline_x, line_dash='dash', line_color='gray', row=2, col=2)
    fig.update_xaxes(title_text=xlabel, title_font=_LABEL_FONT, row=2)
    fig.update_yaxes(range=[0, 1], row=2)
    fig.update_annotations(font=_LABEL_FONT)
    fig.update_layout(height=750, title_text=title, margin=dict(t=60, b=40))
    return fig


def plot_poincare_maps(time_series_by_param, param_values, param_label,
                       burn_in_steps, colors=VAR_COLORS):
    num_columns = len(param_values)
    fig = make_subplots(
        rows=2, cols=num_columns,
        subplot_titles=[f'{param_label} = {v}' for v in param_values] + [''] * num_columns,
        vertical_spacing=0.12, horizontal_spacing=0.05,
    )
    for column, param_value in enumerate(param_values, 1):
        series = time_series_by_param[param_value]
        for row, (variable_name, color) in enumerate([
            ('Seafood', colors['S']), ('Effort', colors['E']),
        ], 1):
            values = series[variable_name]
            values_t = values[burn_in_steps:-1]
            values_t_plus_1 = values[burn_in_steps + 1:]
            fig.add_trace(go.Scattergl(
                x=values_t, y=values_t_plus_1, mode='markers',
                marker=dict(color=color, size=2, opacity=0.6), showlegend=False,
            ), row=row, col=column)
            axis_min = float(min(values_t.min(), values_t_plus_1.min())) * 0.9
            axis_max = float(max(values_t.max(), values_t_plus_1.max())) * 1.1
            fig.add_trace(go.Scatter(
                x=[axis_min, axis_max], y=[axis_min, axis_max], mode='lines',
                line=dict(color='black', width=0.8, dash='dash'), showlegend=False,
            ), row=row, col=column)
    fig.update_yaxes(title_text='S(t+1)', title_font=_LABEL_FONT, row=1, col=1)
    fig.update_yaxes(title_text='E(t+1)', title_font=_LABEL_FONT, row=2, col=1)
    fig.update_annotations(font=_LABEL_FONT)
    fig.update_layout(
        height=600,
        title_text='Poincare — x(t) vs x(t+1) (attractor only)',
        margin=dict(t=60, b=40),
    )
    return fig


def plot_time_series_heatmap(time_series_by_param, param_values, param_label,
                             active_metrics, burn_in_fraction=0.6,
                             baseline_dict=None):
    """One figure per active metric, stacked below the time series.

    If baseline_dict is provided, cell values show % change from baseline.
    Otherwise, shows signed % deviation from the row mean (within-scenario variation).

    Cell coloring (diverging scale):
    - Green: exceeds baseline (or row mean)
    - White: equals baseline (or row mean)
    - Red: falls below baseline (or row mean)
    """
    metrics_in_order = [m for m in HEATMAP_METRICS if m in active_metrics]
    if not metrics_in_order or not param_values:
        return None

    x_labels = [str(v) for v in param_values]
    figures = []

    for metric in metrics_in_order:
        series_key = _HEATMAP_SERIES_KEY[metric]

        post_burn_means = []
        for param_value in param_values:
            series = time_series_by_param[param_value][series_key]
            burn_in_steps = int(len(series) * burn_in_fraction)
            post_burn_means.append(float(np.mean(series[burn_in_steps:])))

        if baseline_dict and series_key in baseline_dict:
            baseline_value = baseline_dict[series_key]
            if baseline_value != 0:
                percent_changes = [
                    (value - baseline_value) / abs(baseline_value) * 100
                    for value in post_burn_means
                ]
            else:
                percent_changes = [0.0] * len(post_burn_means)
        else:
            row_mean = float(np.mean(post_burn_means)) or 1.0
            percent_changes = [
                (value - row_mean) / abs(row_mean) * 100
                for value in post_burn_means
            ]

        cell_text = [f"{percent:+.1f}%" for percent in percent_changes]

        # Fixed ±100% scale: -100% → 0.0, 0% → 0.5, +100% → 1.0; clamp beyond ±100%
        normalized_z = [
            max(0.0, min(1.0, (percent + 100) / 200))
            for percent in percent_changes
        ]

        if metric == 'S̄':
            colorscale = SEAFOOD_COLORSCALE
        elif metric == 'H̄':
            colorscale = HARVEST_COLORSCALE
        elif metric == 'P̄ᵐ':
            colorscale = MARKET_PRICE_COLORSCALE
        else:
            colorscale = DEFAULT_DIVERGING_COLORSCALE

        fig = go.Figure(data=go.Heatmap(
            z=[normalized_z],
            x=x_labels,
            y=[metric],
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            showscale=False,
            text=[cell_text],
            texttemplate="%{text}",
            textfont=dict(size=12),
            xgap=2,
            ygap=2,
            hovertemplate="%{y}  |  %{x} = %{text}<extra></extra>",
        ))

        fig.update_layout(
            height=120,
            margin=dict(t=5, b=65, l=80, r=80),
            yaxis=dict(tickfont=_LABEL_FONT),
            xaxis=dict(
                type='category',
                tickfont=_LABEL_FONT,
                title=param_label,
                title_font=_LABEL_FONT,
                title_standoff=12,
                side='bottom',
            ),
        )
        figures.append(fig)

    return figures if figures else None
