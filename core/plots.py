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


STORY_OUTCOME_METRICS = [
    ('Seafood', 'Seafood stock  S̄'),
    ('Harvest', 'Harvest  H̄'),
    ('Market Price', 'Market price  P̄ᵐ'),
]
STORY_FRAUD_METRICS = [
    ('Fraudsters', 'Fraudster share  F̄'),
    ('Perception of Fraud', 'Fraud perception  F̄P̄'),
]
_STORY_PERCENT_TICKS = [-98, -90, -75, -50, 0, 100, 300, 900]
_STORY_FRAUD_PERCENT_TICKS = [-99.9, -99, -90, -50, 0, 100, 300, 900, 1900]
_WORSE_BAND_COLOR = 'rgba(214, 52, 46, 0.06)'


def _story_fraud_summary_html(fraud_means_by_story, story_labels, story_colors):
    """One coloured 'Story: F̄ x · F̄P̄ y' line per story, for a plotly annotation."""
    return '<br>'.join(
        f"<span style='color:{color}'>{label.split('  ')[0]}</span>: "
        f"F̄ {fraud_means_by_story[label][0]:.2f} · F̄P̄ {fraud_means_by_story[label][1]:.2f}"
        for label, color in zip(story_labels, story_colors)
    )


def plot_story_outcome_curves(alpha_values, outcomes_by_story, story_labels, story_colors):
    """Post-transient outcomes vs α, one line per parameter story.

    Row 1: S̄, H̄, P̄ᵐ as ratios to each story's own no-fraud baseline, on a log
    axis whose ticks read as % change (shaded band = worse than baseline).
    Row 2: F̄ and F̄P̄ as ratios to each story's own α = 0 run (fraud without
    blasting), on the same kind of log % axis — the no-fraud baseline has
    F = FP = 0, so a % change against it is undefined.
    """
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{}, {}, {}], [{}, {}, None]],
        subplot_titles=[title for _, title in STORY_OUTCOME_METRICS + STORY_FRAUD_METRICS],
        horizontal_spacing=0.06, vertical_spacing=0.16,
    )
    tick_ratios = [1 + percent / 100 for percent in _STORY_PERCENT_TICKS]
    tick_labels = [f'{percent:+d}%' if percent else '0' for percent in _STORY_PERCENT_TICKS]
    ratio_floor, ratio_ceiling = 0.015, 12.0
    for column, (metric, _) in enumerate(STORY_OUTCOME_METRICS, 1):
        worse_low, worse_high = (1, ratio_ceiling) if metric == 'Market Price' else (ratio_floor, 1)
        fig.add_hrect(y0=worse_low, y1=worse_high, fillcolor=_WORSE_BAND_COLOR,
                      line_width=0, layer='below', row=1, col=column)
        fig.add_hline(y=1, line=dict(color='#888888', width=1), row=1, col=column)
        for label, color in zip(story_labels, story_colors):
            ratios = np.clip(np.asarray(outcomes_by_story[label][metric], dtype=float), 1e-3, None)
            fig.add_trace(go.Scatter(
                x=alpha_values, y=ratios, mode='lines',
                line=dict(color=color, width=2.5),
                name=label, legendgroup=label, showlegend=column == 1,
                customdata=(ratios - 1) * 100,
                hovertemplate='α = %{x:.2f}<br>%{customdata:+.1f}%<extra>' + label + '</extra>',
            ), row=1, col=column)
    fraud_ratio_floor = 8e-4
    fraud_ratio_ceiling = 2.0
    for column, (metric, _) in enumerate(STORY_FRAUD_METRICS, 1):
        fig.add_hline(y=1, line=dict(color='#888888', width=1), row=2, col=column)
        for label, color in zip(story_labels, story_colors):
            ratios = np.clip(np.asarray(outcomes_by_story[label][metric], dtype=float),
                             fraud_ratio_floor * 1.25, None)
            fraud_ratio_ceiling = max(fraud_ratio_ceiling, float(np.nanmax(ratios, initial=0)) * 1.2)
            fig.add_trace(go.Scatter(
                x=alpha_values, y=ratios, mode='lines',
                line=dict(color=color, width=2.5),
                name=label, legendgroup=label, showlegend=False,
                customdata=(ratios - 1) * 100,
                hovertemplate='α = %{x:.2f}<br>%{customdata:+.1f}%<extra>' + label + '</extra>',
            ), row=2, col=column)
    fig.update_yaxes(
        type='log', tickvals=tick_ratios, ticktext=tick_labels,
        range=[np.log10(ratio_floor), np.log10(ratio_ceiling)], gridcolor='#e5e5e5', row=1,
    )
    fig.update_yaxes(
        type='log',
        tickvals=[1 + percent / 100 for percent in _STORY_FRAUD_PERCENT_TICKS],
        ticktext=[f'{percent:+g}%' if percent else '0' for percent in _STORY_FRAUD_PERCENT_TICKS],
        range=[np.log10(fraud_ratio_floor), np.log10(fraud_ratio_ceiling)],
        gridcolor='#e5e5e5', row=2,
    )
    fig.update_yaxes(title_text='% change vs. own no-fraud baseline', title_font=_LABEL_FONT, row=1, col=1)
    fig.update_yaxes(title_text='% change vs. own α = 0 run', title_font=_LABEL_FONT, row=2, col=1)
    fig.update_xaxes(title_text='Blast intensity  α', title_font=_LABEL_FONT, range=[0, 1])
    fig.update_annotations(font=_LABEL_FONT)
    fig.update_layout(
        height=950,
        title_text='Outcomes vs blast intensity — one line per story (shaded = worse than baseline)',
        title_y=1.0,
        title_font=dict(color='#1a1a1a', size=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.04, font=dict(color='#2c2c2c', size=14)),
        margin=dict(t=110, b=40),
        hovermode='x unified',
    )
    return fig


def plot_story_time_series_strip(time_series_by_story, story_labels, story_colors,
                                 time_axis, burn_in_steps, outcome_badges, title,
                                 colors=VAR_COLORS):
    """One column per story at a shared α: S/E/H, F/FP and Pᵐ (log) over time.

    The transient window is shaded grey, dashed lines in the F / FP row mark the
    post-transient means F̄ and F̄P̄, and each column's x-axis title carries its
    outcome badge.
    """
    num_columns = len(story_labels)
    fig = make_subplots(
        rows=3, cols=num_columns,
        subplot_titles=list(story_labels) + [''] * 2 * num_columns,
        shared_xaxes=True, vertical_spacing=0.07, horizontal_spacing=0.04,
    )
    row1_y_max = max(
        max(float(time_series_by_story[label][variable].max())
            for variable in ('Seafood', 'Effort', 'Harvest'))
        for label in story_labels
    )
    traces_by_row = [
        (1, 'Seafood', 'S', 'Seafood (S)'),
        (1, 'Effort', 'E', 'Effort (E)'),
        (1, 'Harvest', 'Harvest', 'Harvest (H)'),
        (2, 'Fraudsters', 'F', 'Fraudsters (F)'),
        (2, 'Perception of Fraud', 'FP', 'Perception (FP)'),
        (3, 'Market Price', 'Pm', 'Market Price (Pₘ)'),
    ]
    for column, label in enumerate(story_labels, 1):
        series = time_series_by_story[label]
        for row, series_key, color_key, trace_name in traces_by_row:
            fig.add_trace(go.Scatter(
                x=time_axis, y=series[series_key], mode='lines',
                line=dict(color=colors[color_key], width=1.5),
                name=trace_name, legendgroup=color_key, showlegend=column == 1,
            ), row=row, col=column)
        for series_key, color_key, mean_name in [
            ('Fraudsters', 'F', 'Mean F̄ (post-transient)'),
            ('Perception of Fraud', 'FP', 'Mean F̄P̄ (post-transient)'),
        ]:
            post_transient_mean = float(np.mean(series[series_key][burn_in_steps:]))
            fig.add_trace(go.Scatter(
                x=[time_axis[burn_in_steps], time_axis[-1]],
                y=[post_transient_mean, post_transient_mean], mode='lines',
                line=dict(color=colors[color_key], width=2.5, dash='dash'),
                name=mean_name, legendgroup=f'{color_key}_mean', showlegend=column == 1,
                hovertemplate=f'{mean_name}: {post_transient_mean:.2f}<extra></extra>',
            ), row=2, col=column)
        fig.update_xaxes(title_text=f'Time<br>{outcome_badges[label]}',
                         title_font=dict(color='#2c2c2c', size=14), row=3, col=column)
    fig.add_vrect(x0=0, x1=burn_in_steps, fillcolor='#f2f2f2', line_width=0,
                  layer='below', row='all', col='all')
    fig.update_yaxes(title_text='S / E / H', title_font=_LABEL_FONT, row=1, col=1)
    fig.update_yaxes(title_text='F / FP', title_font=_LABEL_FONT, row=2, col=1)
    fig.update_yaxes(title_text='Pᵐ (log)', title_font=_LABEL_FONT, row=3, col=1)
    fig.update_yaxes(range=[0, row1_y_max * 1.05], row=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=2)
    market_prices = np.concatenate([
        np.asarray(time_series_by_story[label]['Market Price'], dtype=float)
        for label in story_labels
    ])
    positive_prices = market_prices[np.isfinite(market_prices) & (market_prices > 0)]
    if positive_prices.size:
        fig.update_yaxes(type='log', range=[np.log10(positive_prices.min()) - 0.1,
                                            np.log10(positive_prices.max()) + 0.1], row=3)
    else:
        fig.update_yaxes(type='log', row=3)
    fig.update_annotations(font=_LABEL_FONT)
    for annotation, color in zip(fig.layout.annotations, story_colors):
        annotation.font.color = color
    fig.update_layout(
        height=850,
        title_text=title,
        title_y=1.0,
        title_font=dict(color='#1a1a1a', size=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.04, font=dict(color='#2c2c2c', size=14)),
        margin=dict(t=100, b=110),
    )
    return fig


def plot_story_phase_portraits(trajectories_by_alpha, fraud_means_by_alpha, story_labels,
                               story_colors, equilibrium_point):
    """Post-transient S–E attractors of every story, one panel per α.

    ``trajectories_by_alpha`` maps α → {story label: (seafood, effort)} and
    ``fraud_means_by_alpha`` maps α → {story label: (F̄, F̄P̄)}, listed in each
    panel's corner. The no-fraud equilibrium is drawn as a star in every panel.
    """
    alpha_values = list(trajectories_by_alpha)
    fig = make_subplots(
        rows=1, cols=len(alpha_values),
        subplot_titles=[f'α = {alpha}' for alpha in alpha_values],
        horizontal_spacing=0.07,
    )
    for column, alpha in enumerate(alpha_values, 1):
        for label, color in zip(story_labels, story_colors):
            seafood, effort = trajectories_by_alpha[alpha][label]
            fig.add_trace(go.Scatter(
                x=seafood, y=effort, mode='lines', line=dict(color=color, width=2),
                name=label, legendgroup=label, showlegend=column == 1,
            ), row=1, col=column)
            fig.add_trace(go.Scatter(
                x=[seafood[-1]], y=[effort[-1]], mode='markers',
                marker=dict(color=color, size=9, line=dict(color='white', width=1.5)),
                legendgroup=label, showlegend=False, hoverinfo='skip',
            ), row=1, col=column)
        fig.add_trace(go.Scatter(
            x=[equilibrium_point[0]], y=[equilibrium_point[1]], mode='markers',
            marker=dict(color='#333333', size=16, symbol='star'),
            name='No-fraud equilibrium', legendgroup='equilibrium', showlegend=column == 1,
        ), row=1, col=column)
        fig.add_annotation(
            text=_story_fraud_summary_html(fraud_means_by_alpha[alpha], story_labels, story_colors),
            xref='x domain', yref='y domain', x=0.99, y=0.01,
            xanchor='right', yanchor='bottom', align='left', showarrow=False,
            font=dict(color='#2c2c2c', size=13),
            bgcolor='rgba(255, 255, 255, 0.85)', bordercolor='#d0d0d0', borderwidth=1,
            row=1, col=column,
        )
    fig.update_xaxes(title_text='Seafood stock  S', title_font=_LABEL_FONT, range=[0, 1.05])
    fig.update_yaxes(title_text='Fishing effort  E', title_font=_LABEL_FONT, row=1, col=1)
    fig.update_layout(
        height=560,
        title_text='Attractors in the S–E plane (post-transient, with mean F̄ / F̄P̄)',
        title_y=1.0,
        title_font=dict(color='#1a1a1a', size=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.06, font=dict(color='#2c2c2c', size=14)),
        margin=dict(t=100, b=40),
    )
    for annotation in fig.layout.annotations[:len(alpha_values)]:
        annotation.font = _LABEL_FONT
    return fig
