import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import VAR_COLORS

HEATMAP_METRICS = ['S̄', 'H̄', 'P̄ᵐ', 'φ_FP', 'φ_H']

_LABEL_FONT = dict(color='#2c2c2c', size=18)
MARKET_PRICE_COLORSCALE = [[0.0, '#228B22'], [0.5, '#FFFFFF'], [1.0, '#DC143C']]
DEFAULT_DIVERGING_COLORSCALE = [[0.0, '#228B22'], [0.5, '#FFFFFF'], [1.0, '#DC143C']]

_HEATMAP_COLORSCALE = {
    'S̄': [[0.0, '#DC143C'], [0.5, '#FFFFFF'], [1.0, '#228B22']],
    'H̄': [[0.0, '#DC143C'], [0.5, '#FFFFFF'], [1.0, '#228B22']],
    'P̄ᵐ': [[0.0, '#228B22'], [0.5, '#FFFFFF'], [1.0, '#DC143C']],
    'φ_FP': [[0.0, '#228B22'], [0.5, '#FFFFFF'], [1.0, '#DC143C']],
    'φ_H': [[0.0, '#228B22'], [0.5, '#FFFFFF'], [1.0, '#DC143C']],
}


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


def _continuation_line_segments(param_values, state_values, stable_flags):
    """Split a continuation branch into contiguous stable/unstable line segments."""
    segments = []
    num_points = len(param_values)
    segment_start = None
    segment_stable = None

    def flush(end_exclusive):
        if segment_start is None or end_exclusive - segment_start < 1:
            return
        segments.append({
            'param_values': param_values[segment_start:end_exclusive],
            'values': state_values[segment_start:end_exclusive],
            'stable': segment_stable,
        })

    for index in range(num_points):
        if not np.isfinite(state_values[index]) or not np.isfinite(param_values[index]):
            flush(index)
            segment_start = None
            segment_stable = None
            continue
        is_stable = bool(stable_flags[index])
        if segment_start is None:
            segment_start = index
            segment_stable = is_stable
        elif is_stable != segment_stable:
            flush(index)
            # Share an endpoint so adjacent segments meet visually at the switch.
            segment_start = max(index - 1, 0)
            segment_stable = is_stable
    flush(num_points)
    return segments


def plot_bifurcation(sweep_values, seafood_values, effort_values, fraudster_values,
                     perception_values, xlabel, title, vline_x=None, vline_label=None,
                     colors=VAR_COLORS, continuation_branch=None):
    """
    Plot a 2×2 attractor scatter, optionally overlaid with a fixed-point branch.

    Args:
        sweep_values: X-axis values for attractor samples.
        seafood_values: Attractor S samples.
        effort_values: Attractor E samples.
        fraudster_values: Attractor F samples.
        perception_values: Attractor FP samples.
        xlabel: X-axis label.
        title: Figure title.
        vline_x: Optional vertical reference line.
        vline_label: Label for that reference line.
        colors: Color map for attractor markers.
        continuation_branch: Optional dict from ``continue_fixed_point`` with
            ``param_values``, ``S``, ``E``, ``F``, ``FP``, ``stable``. Solid black
            = locally stable; dotted black = unstable.

    Returns:
        A Plotly figure.
    """
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

    if continuation_branch is not None and len(continuation_branch.get('param_values', [])):
        param_values = np.asarray(continuation_branch['param_values'], dtype=np.float64)
        stable_flags = np.asarray(continuation_branch['stable'], dtype=bool)
        state_panels = [
            (continuation_branch['S'], 1, 1),
            (continuation_branch['E'], 1, 2),
            (continuation_branch['F'], 2, 1),
            (continuation_branch['FP'], 2, 2),
        ]
        shown_stable_legend = False
        shown_unstable_legend = False
        for panel_index, (state_values, row, col) in enumerate(state_panels):
            state_values = np.asarray(state_values, dtype=np.float64)
            for segment in _continuation_line_segments(
                param_values, state_values, stable_flags,
            ):
                is_stable = segment['stable']
                show_legend = panel_index == 0 and (
                    (is_stable and not shown_stable_legend)
                    or ((not is_stable) and not shown_unstable_legend)
                )
                if is_stable:
                    shown_stable_legend = True
                else:
                    shown_unstable_legend = True
                fig.add_trace(go.Scatter(
                    x=segment['param_values'], y=segment['values'], mode='lines',
                    line=dict(
                        color='black', width=2.0,
                        dash='solid' if is_stable else 'dot',
                    ),
                    name=(
                        'Fixed point (stable)' if is_stable
                        else 'Fixed point (unstable)'
                    ),
                    legendgroup='fp_stable' if is_stable else 'fp_unstable',
                    showlegend=show_legend,
                    hovertemplate=(
                        'μ=%{x:.4f}<br>x*=%{y:.4f}'
                        f'<br>({"stable" if is_stable else "unstable"})<extra></extra>'
                    ),
                ), row=row, col=col)

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
    fig.update_layout(
        height=750, title_text=title, margin=dict(t=60, b=40),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            font=dict(color='#2c2c2c', size=13),
        ),
    )
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


def plot_time_series_heatmap(percent_by_metric, param_values, param_label,
                             active_metrics):
    """One figure per active metric from precomputed percent cell values.

    ``percent_by_metric`` maps each metric name to a list of percents aligned
    with ``param_values`` (from ``core.metrics.build_heatmap_display_rows``).

    Cell coloring (diverging scale, ±100% clamped):
    - S̄ / H̄: green = above baseline, red = below
    - P̄ᵐ / φ_FP / φ_H: red = above baseline (or positive price contribution)
    """
    metrics_in_order = [m for m in HEATMAP_METRICS if m in active_metrics]
    if not metrics_in_order or not param_values:
        return None

    x_labels = [str(v) for v in param_values]
    figures = []

    for metric in metrics_in_order:
        percent_changes = list(percent_by_metric[metric])
        if len(percent_changes) != len(param_values):
            raise ValueError(
                f"percent_by_metric[{metric!r}] length {len(percent_changes)} "
                f"!= len(param_values) {len(param_values)}"
            )

        cell_text = [f"{percent:+.1f}%" for percent in percent_changes]
        # Fixed ±100% scale: -100% → 0.0, 0% → 0.5, +100% → 1.0; clamp beyond ±100%
        normalized_z = [
            max(0.0, min(1.0, (percent + 100) / 200))
            for percent in percent_changes
        ]
        colorscale = _HEATMAP_COLORSCALE.get(metric, DEFAULT_DIVERGING_COLORSCALE)

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
