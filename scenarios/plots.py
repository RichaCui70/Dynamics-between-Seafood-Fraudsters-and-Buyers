import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import COLORS4, ECON_COLORS

HEATMAP_METRICS = ['S̄', 'H̄', 'P̄ᵐ']
_HEATMAP_KEY = {'S̄': 'Seafood', 'H̄': 'Harvest', 'P̄ᵐ': 'Market Price'}


def plot_4var_ts(ts_dict, t_arr, param_vals, param_label, title, colors=COLORS4):
    _N = len(param_vals)
    fig = make_subplots(
        rows=2, cols=_N,
        subplot_titles=[f'{param_label} = {v}' for v in param_vals] + [''] * _N,
        shared_xaxes=True, vertical_spacing=0.10, horizontal_spacing=0.05,
    )
    se_max = max(
        max(float(ts_dict[v]['Seafood'].max()), float(ts_dict[v]['Effort'].max()))
        for v in param_vals
    )
    for col, v in enumerate(param_vals, 1):
        d = ts_dict[v]
        show = col == 1
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Seafood'], mode='lines',
            line=dict(color=colors['S'], width=1.5),
            name='Seafood (S)', legendgroup='S', showlegend=show,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Effort'], mode='lines',
            line=dict(color=colors['E'], width=1.5),
            name='Effort (E)', legendgroup='E', showlegend=show,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Fraudsters'], mode='lines',
            line=dict(color=colors['F'], width=1.5),
            name='Fraudsters (F)', legendgroup='F', showlegend=show,
        ), row=2, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Perception of Fraud'], mode='lines',
            line=dict(color=colors['FP'], width=1.5),
            name='Perception (FP)', legendgroup='FP', showlegend=show,
        ), row=2, col=col)
    fig.update_yaxes(title_text='S / E', row=1, col=1)
    fig.update_yaxes(title_text='F / FP', row=2, col=1)
    fig.update_yaxes(rangemode='tozero')
    fig.update_yaxes(range=[0, se_max * 1.05], row=1)
    fig.update_xaxes(title_text='Time', row=2)
    fig.update_layout(
        height=600, title_text=title,
        title_y=1.0,
        legend=dict(orientation='h', yanchor='bottom', y=1.06),
        margin=dict(t=100, b=40),
    )
    return fig


def plot_4var_ts_fp_zoom(ts_dict, t_arr, param_vals, param_label, title,
                         fp_ylim=0.05, colors=COLORS4):
    """Like plot_4var_ts but with a third row showing only FP on a zoomed y-axis.
    Row 1: S / E
    Row 2: F / FP
    Row 3: FP (zoomed to [0, fp_ylim])
    """
    _N = len(param_vals)
    fig = make_subplots(
        rows=3, cols=_N,
        subplot_titles=[f'{param_label} = {v}' for v in param_vals] + [''] * 2 * _N,
        shared_xaxes=True, vertical_spacing=0.08, horizontal_spacing=0.05,
    )
    se_max = max(
        max(float(ts_dict[v]['Seafood'].max()), float(ts_dict[v]['Effort'].max()))
        for v in param_vals
    )
    for col, v in enumerate(param_vals, 1):
        d = ts_dict[v]
        show = col == 1
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Seafood'], mode='lines',
            line=dict(color=colors['S'], width=1.5),
            name='Seafood (S)', legendgroup='S', showlegend=show,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Effort'], mode='lines',
            line=dict(color=colors['E'], width=1.5),
            name='Effort (E)', legendgroup='E', showlegend=show,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Fraudsters'], mode='lines',
            line=dict(color=colors['F'], width=1.5),
            name='Fraudsters (F)', legendgroup='F', showlegend=show,
        ), row=2, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Perception of Fraud'], mode='lines',
            line=dict(color=colors['FP'], width=1.5),
            name='Perception (FP)', legendgroup='FP', showlegend=show,
        ), row=2, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Perception of Fraud'], mode='lines',
            line=dict(color=colors['FP'], width=1.5),
            name='FP (zoomed)', legendgroup='FPz', showlegend=show,
        ), row=3, col=col)

    fig.update_yaxes(title_text='S / E', row=1, col=1)
    fig.update_yaxes(title_text='F / FP', row=2, col=1)
    fig.update_yaxes(title_text='FP (zoom)', range=[0, fp_ylim], row=3, col=1)
    for col in range(2, _N + 1):
        fig.update_yaxes(range=[0, fp_ylim], row=3, col=col)
    fig.update_yaxes(rangemode='tozero', row=1)
    fig.update_yaxes(rangemode='tozero', row=2)
    fig.update_yaxes(range=[0, se_max * 1.05], row=1)
    fig.update_xaxes(title_text='Time', row=3)
    fig.update_layout(
        height=800, title_text=title,
        title_y=1.0,
        legend=dict(orientation='h', yanchor='bottom', y=1.06),
        margin=dict(t=100, b=40),
    )
    return fig


def plot_ts_with_economics(ts_dict, t_arr, param_vals, param_label, title,
                           colors=COLORS4, econ_colors=ECON_COLORS,
                           show_per_effort=False):
    """Like plot_4var_ts but with extra rows for economic variables:
    Row 1: S / E
    Row 2: F / FP
    Row 3: Market Price / Wholesale Price
    Row 4 (optional): Revenue per Effort / Cost per Effort  [show_per_effort=True]
    """
    _N = len(param_vals)
    n_rows = 4 if show_per_effort else 3
    fig = make_subplots(
        rows=n_rows, cols=_N,
        subplot_titles=[f'{param_label} = {v}' for v in param_vals] + [''] * (n_rows - 1) * _N,
        shared_xaxes=True, vertical_spacing=0.07, horizontal_spacing=0.05,
    )
    se_max = max(
        max(float(ts_dict[v]['Seafood'].max()), float(ts_dict[v]['Effort'].max()))
        for v in param_vals
    )
    for col, v in enumerate(param_vals, 1):
        d = ts_dict[v]
        show = col == 1
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Seafood'], mode='lines',
            line=dict(color=colors['S'], width=1.5),
            name='Seafood (S)', legendgroup='S', showlegend=show,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Effort'], mode='lines',
            line=dict(color=colors['E'], width=1.5),
            name='Effort (E)', legendgroup='E', showlegend=show,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Fraudsters'], mode='lines',
            line=dict(color=colors['F'], width=1.5),
            name='Fraudsters (F)', legendgroup='F', showlegend=show,
        ), row=2, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Perception of Fraud'], mode='lines',
            line=dict(color=colors['FP'], width=1.5),
            name='Perception (FP)', legendgroup='FP', showlegend=show,
        ), row=2, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Market Price'], mode='lines',
            line=dict(color=econ_colors['Pm'], width=1.5),
            name='Market Price (Pₘ)', legendgroup='Pm', showlegend=show,
        ), row=3, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Wholesale Price'], mode='lines',
            line=dict(color=econ_colors['Pw'], width=1.5),
            name='Wholesale Price (Pᵥ)', legendgroup='Pw', showlegend=show,
        ), row=3, col=col)
        if show_per_effort:
            fig.add_trace(go.Scatter(
                x=t_arr, y=d['Revenue per Effort'], mode='lines',
                line=dict(color=econ_colors['Rev'], width=1.5),
                name='Revenue / Effort', legendgroup='Rev', showlegend=show,
            ), row=4, col=col)
            fig.add_trace(go.Scatter(
                x=t_arr, y=d['Cost per Effort'], mode='lines',
                line=dict(color=econ_colors['Cost'], width=1.5),
                name='Cost / Effort', legendgroup='Cost', showlegend=show,
            ), row=4, col=col)

    fig.update_yaxes(title_text='S / E', row=1, col=1)
    fig.update_yaxes(title_text='F / FP', row=2, col=1)
    fig.update_yaxes(title_text='Price', row=3, col=1)
    if show_per_effort:
        fig.update_yaxes(title_text='Per-Effort', row=4, col=1)
    fig.update_yaxes(rangemode='tozero')
    fig.update_yaxes(range=[0, se_max * 1.05], row=1)
    fig.update_xaxes(title_text='Time', row=n_rows)
    fig.update_layout(
        height=1000 if show_per_effort else 750,
        title_text=title,
        title_y=1.0,
        legend=dict(orientation='h', yanchor='bottom', y=1.04),
        margin=dict(t=100, b=40),
    )
    return fig


def plot_bifurcation(x_data, s_data, e_data, xlabel, title,
                     vline_x=None, vline_label=None, colors=COLORS4):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Seafood S*', 'Effort E*'],
        horizontal_spacing=0.08,
    )
    fig.add_trace(go.Scattergl(
        x=x_data, y=s_data, mode='markers',
        marker=dict(color=colors['S'], size=2, opacity=0.4), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=x_data, y=e_data, mode='markers',
        marker=dict(color=colors['E'], size=2, opacity=0.4), showlegend=False,
    ), row=1, col=2)
    if vline_x is not None:
        fig.add_vline(
            x=vline_x, line_dash='dash', line_color='gray',
            annotation_text=vline_label or '',
            annotation_position='top right', row=1, col=1,
        )
        fig.add_vline(x=vline_x, line_dash='dash', line_color='gray', row=1, col=2)
    fig.update_xaxes(title_text=xlabel)
    fig.update_layout(height=600, title_text=title, margin=dict(t=60, b=40))
    return fig


def plot_return_maps(ts_dict, param_vals, param_label, burn, colors=COLORS4):
    _N = len(param_vals)
    fig = make_subplots(
        rows=2, cols=_N,
        subplot_titles=[f'{param_label} = {v}' for v in param_vals] + [''] * _N,
        vertical_spacing=0.12, horizontal_spacing=0.05,
    )
    for col, v in enumerate(param_vals, 1):
        d = ts_dict[v]
        for row, (var, clr) in enumerate([
            ('Seafood', colors['S']), ('Effort', colors['E']),
        ], 1):
            x = d[var]
            x_t, x_tp1 = x[burn:-1], x[burn + 1:]
            fig.add_trace(go.Scattergl(
                x=x_t, y=x_tp1, mode='markers',
                marker=dict(color=clr, size=2, opacity=0.6), showlegend=False,
            ), row=row, col=col)
            lo = float(min(x_t.min(), x_tp1.min())) * 0.9
            hi = float(max(x_t.max(), x_tp1.max())) * 1.1
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode='lines',
                line=dict(color='black', width=0.8, dash='dash'), showlegend=False,
            ), row=row, col=col)
    fig.update_yaxes(title_text='S(t+1)', row=1, col=1)
    fig.update_yaxes(title_text='E(t+1)', row=2, col=1)
    fig.update_layout(
        height=600,
        title_text='Poincare — x(t) vs x(t+1) (attractor only)',
        margin=dict(t=60, b=40),
    )
    return fig


# Seafood: negative % = red (bad), positive % = green (good)
_CS_SEAFOOD = [[0.0, '#DC143C'], [0.5, '#FFFFFF'], [1.0, '#228B22']]
# Harvest: negative % = red (bad), positive % = green (good)
_CS_HARVEST = [[0.0, '#DC143C'], [0.5, '#FFFFFF'], [1.0, '#228B22']]
# Harvest: negative % = red (bad), positive % = green (good)
_CS_MARKET_PRICE= [[0.0, '#228B22'], [0.5, '#FFFFFF'], [1.0, '#DC143C']]
# Harvest / Market Price: keep original blue scale
_CS_OTHERS  = [[0.0, '#228B22'], [0.5, '#FFFFFF'], [1.0, '#DC143C']]


def plot_ts_heatmap(ts_dict, param_vals, param_label, active_metrics, burn_frac=0.6, baseline_dict=None):
    """One figure per active metric, stacked below the time series.

    If baseline_dict is provided, cell values show % change from baseline.
    Otherwise, shows signed % deviation from the row mean (within-scenario variation).

    Cell coloring (diverging scale):
    - Green: exceeds baseline (or row mean)
    - White: equals baseline (or row mean)
    - Red: falls below baseline (or row mean)
    """
    y_order = [m for m in HEATMAP_METRICS if m in active_metrics]
    if not y_order or not param_vals:
        return None

    x_labels = [str(v) for v in param_vals]
    figs = []

    for metric in y_order:
        key = _HEATMAP_KEY[metric]

        # Post-burn averages
        avgs = []
        for pv in param_vals:
            arr = ts_dict[pv][key]
            n = len(arr)
            burn = int(n * burn_frac)
            avgs.append(float(np.mean(arr[burn:])))

        # Calculate % change (vs baseline or vs row mean)
        if baseline_dict and key in baseline_dict:
            baseline_val = baseline_dict[key]
            if baseline_val != 0:
                pcts = [(v - baseline_val) / abs(baseline_val) * 100 for v in avgs]
            else:
                pcts = [0.0] * len(avgs)
        else:
            # Fallback: signed % deviation from row mean
            row_mean = float(np.mean(avgs)) or 1.0
            pcts = [(v - row_mean) / abs(row_mean) * 100 for v in avgs]

        text  = [f"{p:+.1f}%" for p in pcts]

        # Fixed ±100% scale: -100% → 0.0, 0% → 0.5, +100% → 1.0; clamp beyond ±100%
        z_norm = [max(0.0, min(1.0, (p + 100) / 200)) for p in pcts]

        # Select colorscale based on metric
        if metric == 'S̄':
            colorscale = _CS_SEAFOOD
        elif metric == 'H̄':
            colorscale = _CS_HARVEST
        elif metric == 'P̄ᵐ':
            colorscale = _CS_MARKET_PRICE
        else:
            colorscale = _CS_OTHERS

        fig = go.Figure(data=go.Heatmap(
            z=[z_norm],
            x=x_labels,
            y=[metric],
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            showscale=False,
            text=[text],
            texttemplate="%{text}",
            textfont=dict(size=12),
            xgap=2,
            ygap=2,
            hovertemplate="%{y}  |  %{x} = %{text}<extra></extra>",
        ))

        fig.update_layout(
            height=120,
            margin=dict(t=5, b=50, l=80, r=80),
            yaxis=dict(tickfont=dict(size=11)),
            xaxis=dict(
                type='category',
                tickfont=dict(size=11),
                title=param_label,
                side='bottom',
            ),
        )
        figs.append(fig)

    return figs if figs else None
