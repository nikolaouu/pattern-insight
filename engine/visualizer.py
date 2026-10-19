import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class PatternVisualizer:
    """
    Plotly-based "macro" visualization layer for the results datacube.

    This class is the counterpart to ``Plotter`` (which handles micro,
    matplotlib-based views around individual patterns). Here we render
    *aggregate* views over the entire ``results_df`` produced by
    ``PatternEngine`` -- heatmaps, 3D surfaces, decay curves, risk/reward
    scatter, and a tactical summary table.

    Key conventions
    ---------------
    * Every method returns a ``plotly.graph_objects.Figure``. The caller
      decides whether to call ``.show()``, ``.write_html()``, or embed it.
      This is the opposite of ``Plotter``, which displays inline via
      ``plt.show()`` and returns ``None``.
    * The expected input is the datacube dataframe with one row per
      (interval, regime, session_open) cell and columns including
      ``win_rate``, ``sharpe_ratio``, ``p_value``, ``total``, etc.
    * ``regime_filter='ALL'`` means "do not filter by regime". Any other
      value keeps only rows whose ``regime`` column matches. This lets the
      same figure factory render both the "all regimes pooled" view and
      per-regime breakdowns.
    * Color scales follow a semantic convention:
        - "good high" metrics (win_rate, sharpe, sortino) -> RdYlGn
          (green = high, red = low).
        - p-value -> RdYlGn_r (inverted: low p-value = green = significant).
        - everything else -> Viridis (neutral sequential).
    """

    def __init__(self, template="plotly_dark"):
        """
        Args:
            template: Plotly template name. ``'plotly_dark'`` matches the
                rest of the project's dark aesthetic. Pass ``None`` to use
                the plotly default if a light theme is needed.
        """
        self.template = template

    # ------------------------------------------------------------------ #
    # Heatmap: interval x session_open, one metric
    # ------------------------------------------------------------------ #

    def plot_macro_heatmap(self, df: pd.DataFrame, metric: str = 'win_rate', regime_filter: str = 'ALL', title: str = None):
        """
        Render a 2D heatmap of ``metric`` across (interval, session_open).

        Args:
            df: Datacube dataframe.
            metric: Column to visualize. Expected to be numeric. Common
                values: ``win_rate``, ``sharpe_ratio``, ``p_value``,
                ``max_drawdown``.
            regime_filter: 'ALL' to pool all regimes, or a specific label
                to filter. When 'ALL' is passed but the dataframe has no
                'regime' column, no filtering is applied.
            title: Optional title override.

        Notes:
            * Values are aggregated with ``mean`` if duplicate cells exist
              (which can happen if the engine was run multiple times and
              results were concatenated).
            * Hover formatting uses percent for rate-like metrics and
              two-decimal floats otherwise.
            * The y-axis (``interval``) preserves the dataframe's row
              order, which is the insertion order produced by the engine's
              ``interval_labels`` list.
        """
        df_filtered = df.copy()
        if 'regime' in df_filtered.columns and regime_filter != 'ALL':
            df_filtered = df_filtered[df_filtered['regime'] == regime_filter]

        pivot_df = df_filtered.pivot_table(
            index='interval',
            columns='session_open',
            values=metric,
            aggfunc='mean'
        )

        # Percent formatting for rate-like metrics; float otherwise.
        text_template = "%{z:.1%}" if metric in ['win_rate', 'fail_rate', 'invalid_rate', 'max_drawdown'] else "%{z:.2f}"

        colorscale = "Viridis"

        # Semantic color overrides.
        if metric in ['win_rate', 'sharpe_ratio', 'sortino_ratio']:
            colorscale = "RdYlGn"

        elif metric == 'p_value':
            # Inverted: green = low p-value = statistically significant.
            colorscale = "RdYlGn_r"

        fig = px.imshow(
            pivot_df,
            labels=dict(x="Session Duration (Candles)", y="Interval / Regime", color=metric.upper()),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale=colorscale,
            aspect="auto",
            template=self.template,
            title=title or f"Macro Overview Heatmap: {metric.upper()} (Regime: {regime_filter})"
        )

        fig.update_traces(hovertemplate="Interval: %{y}<br>Session: %{x}<br>" + f"{metric.upper()}: " + text_template + "<extra></extra>")

        fig.update_layout(
            xaxis=dict(dtick=1),
            height=450,
            font=dict(size=12)
        )
        return fig

    # ------------------------------------------------------------------ #
    # 3D surface: interval x session_open x metric
    # ------------------------------------------------------------------ #

    def plot_macro_3d_surface(self, df: pd.DataFrame, metric: str = 'sharpe_ratio', regime_filter: str = 'ALL'):
        """
        Render a 3D surface of ``metric`` across (interval, session_open).

        The surface is a perspective view of the same pivot used by the
        heatmap: x = session_open, y = interval index, z = metric. Useful
        for spotting "ridges" -- e.g. a holding horizon where a strategy
        peaks across many intervals.

        Notes:
            * The y-axis is the *index* of each interval, not the interval
              label itself; tick labels are set via ``ticktext`` so the
              displayed labels are the interval names.
            * Colorscale is fixed to Viridis because a diverging scale
              would be misleading on a surface where sign is not
              guaranteed to be meaningful (e.g. Sharpe can be negative).
            * Missing cells (NaN in the pivot) appear as holes in the
              surface, which is the desired behavior -- it makes coverage
              gaps visible.
        """
        df_filtered = df.copy()

        if 'regime' in df_filtered.columns and regime_filter != 'ALL':
            df_filtered = df_filtered[df_filtered['regime'] == regime_filter]

        pivot_df = df_filtered.pivot_table(
            index='interval',
            columns='session_open',
            values=metric,
            aggfunc='mean'
        )

        x_sessions = pivot_df.columns.to_list()
        y_intervals = pivot_df.index.to_list()
        z_data = pivot_df.values

        fig = go.Figure(data=[go.Surface(
            z=z_data,
            x=x_sessions,
            y=np.arange(len(y_intervals)),
            colorscale='Viridis'
        )])

        fig.update_layout(
            title=f"3D Performance Topography: {metric.upper()}",
            scene=dict(
                xaxis_title='Session Duration',
                yaxis_title='Interval Index',
                zaxis_title=metric.upper(),
                # y-axis is numeric (interval position); override tick
                # labels to show the actual interval names.
                yaxis=dict(tickvals=np.arange(len(y_intervals)), ticktext=y_intervals)
            ),
            template=self.template,
            height=600
        )
        return fig

    # ------------------------------------------------------------------ #
    # Performance decay: metric vs holding horizon
    # ------------------------------------------------------------------ #

    def plot_performance_decay(self, df: pd.DataFrame, primary_metric: str = 'win_rate', show_p_value_threshold: bool = True):
        """
        Line chart of ``primary_metric`` vs ``session_open``, with an
        optional overlay of mean p-value on a secondary axis.

        This is the canonical "does the edge decay with holding time?"
        view. Aggregation is across all intervals and regimes pooled in
        ``df`` -- pass a pre-filtered ``df`` to get a specific slice.

        Args:
            df: Datacube dataframe.
            primary_metric: Metric to plot on the left axis. Values in
                ``['win_rate', 'fail_rate']`` are scaled by 100 to display
                as percentages; other metrics are plotted as-is.
            show_p_value_threshold: If True, overlay mean p-value on the
                right axis and draw the alpha=0.05 reference line.

        Notes:
            * ``df_grouped`` aggregates with ``mean`` for metrics and
              ``sum`` for ``total``. The ``total`` column is not plotted
              here but is included so callers can reuse the grouped frame
              if needed (e.g. for tooltips in a custom variant).
            * ``hovermode='x unified'`` groups tooltips by session_open,
              which is what you want when comparing metric vs p-value at
              the same horizon.
        """
        df_grouped = df.groupby('session_open').agg({
            primary_metric: 'mean',
            'p_value': 'mean',
            'total': 'sum',
            'sharpe_ratio': 'mean'
        }).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Percent-metric scaling: only for rate-like metrics.
        y_vals = df_grouped[primary_metric] * 100 if primary_metric in ['win_rate', 'fail_rate'] else df_grouped[primary_metric]

        fig.add_trace(
            go.Scatter(
                x=df_grouped['session_open'],
                y=y_vals,
                mode='lines+markers',
                name=primary_metric.upper(),
                line=dict(color='#00FFCC', width=3),
                marker=dict(size=6)
            ),
            secondary_y=False
        )

        if show_p_value_threshold:
            fig.add_trace(
                go.Scatter(
                    x=df_grouped['session_open'],
                    y=df_grouped['p_value'],
                    mode='lines',
                    name='p-value',
                    line=dict(color='#FF3366', width=1.5, dash='dash')
                ),
                secondary_y=True
            )

            # Alpha reference line on the secondary axis.
            fig.add_hline(
                y=0.05,
                line_dash="dot",
                line_color="red",
                annotation_text="α = 0.05 (Threshold)",
                secondary_y=True
            )

        fig.update_layout(
            title=f"Performance Decay Curve: {primary_metric.upper()} vs Session Duration",
            xaxis_title="Session Duration (Candles)",
            template=self.template,
            height=500,
            hovermode="x unified"
        )

        # p-value axis is bounded [0, 1] by definition.
        fig.update_yaxes(title_text=f"Mean {primary_metric.upper()}", secondary_y=False)
        fig.update_yaxes(title_text="Statistical Significance (p-value)", secondary_y=True, range=[0, 1])

        return fig

    # ------------------------------------------------------------------ #
    # Risk / reward scatter
    # ------------------------------------------------------------------ #

    def plot_mfe_vs_mae_scatter(self, df: pd.DataFrame, use_atr: bool = True):
        """
        Scatter of average MFE vs average MAE per cell, with marker size
        proportional to sample size and color encoding win rate.

        This is the "shape of the edge" view: cells that sit high on MFE
        and low on (absolute) MAE are attractive; cells on the diagonal
        are symmetric risk/reward; cells below the diagonal have adverse
        excursion exceeding favorable excursion.

        Args:
            df: Datacube dataframe.
            use_atr: If True, use ATR-normalized columns
                (``avg_mfe_atr`` / ``avg_mae_atr``), which are comparable
                across instruments and volatility regimes. If False, use
                raw percent columns (``avg_mfe`` / ``avg_mae``).

        Notes:
            * ``size='total'`` makes low-sample cells visually small, so
              that noisy estimates do not dominate the plot.
            * A dashed reference line y = -x is drawn through the origin.
              It represents the "MFE == |MAE|" boundary; points above it
              have favorable excursion exceeding adverse excursion.
            * Since MAE is stored as a positive magnitude (fraction), the
              y-axis is effectively plotted in "magnitude" space, and the
              reference line is negative-sloped to reflect that.
        """
        x_col = 'avg_mfe_atr' if use_atr else 'avg_mfe'
        y_col = 'avg_mae_atr' if use_atr else 'avg_mae'

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            size='total',
            color='win_rate',
            hover_data=['interval', 'session_open', 'sharpe_ratio', 'p_value'],
            color_continuous_scale='RdYlGn',
            labels={
                x_col: "Max Favorable Excursion (MFE)" + (" (ATR)" if use_atr else " (%)"),
                y_col: "Max Adverse Excursion (MAE)" + (" (ATR)" if use_atr else " (%)")
            },
            title="Risk / Reward Profile: MFE vs MAE per Session",
            template=self.template
        )

        # Reference line: MFE == |MAE| boundary. See docstring above for
        # why the slope is negative despite both axes being positive.
        max_val = max(df[x_col].max(), abs(df[y_col].min()))
        fig.add_shape(
            type="line", line=dict(dash='dash', color='gray'),
            x0=0, y0=0, x1=max_val, y1=-max_val
        )

        fig.update_layout(height=500)
        return fig

    # ------------------------------------------------------------------ #
    # Tactical table
    # ------------------------------------------------------------------ #

    def plot_tactical_table(self, df: pd.DataFrame, session_open=None, regime: str = 'ALL'):
        """
        Render a compact table of key metrics for a single session_open
        (and optionally a single regime).

        Intended as a "what does the edge look like right now?" snapshot
        for one holding horizon, e.g. when deciding which regime/session
        combination to trade.

        Args:
            df: Datacube dataframe.
            session_open: If provided, keep only rows with this exact
                session_open value. If None, all sessions are included
                (the table may become wide).
            regime: 'ALL' to pool all regimes, or a specific label to
                filter.

        Columns displayed:
            Interval, Session, Total, Win%, Return%, MFE(ATR), MAE(ATR),
            Sharpe, p-val

        Notes:
            * The table is rendered as a plotly ``go.Table`` trace (not a
              DataFrame display) so it can be embedded in dashboards and
              exported to HTML alongside the other figures.
            * All formatting (percent, rounding) is applied inline before
              being passed to the table; no interactivity beyond hover is
              provided.
            * ``fig.update_layout(height=350)`` is a fixed height; for
              many rows, consider ``height=None`` and letting plotly
              auto-size.
        """
        df_filtered = df.copy()

        if session_open is not None:
            df_filtered = df_filtered[df_filtered['session_open'] == session_open]

        if regime != 'ALL':
            df_filtered = df_filtered[df_filtered['regime'] == regime]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Interval", "Session", "Total", "Win%", "Return%", "MFE(ATR)", "MAE(ATR)", "Sharpe", "p-val"],
                fill_color='#1E1E1E',
                font=dict(color='white', size=12),
                align='center'
            ),
            cells=dict(
                values=[
                    df_filtered['interval'],
                    df_filtered['session_open'],
                    df_filtered['total'],
                    # Inline formatting: percentages rounded to 1 decimal,
                    # returns to 2 decimals, ratios to 2 decimals.
                    (df_filtered['win_rate'] * 100).round(1).astype(str) + '%',
                    (df_filtered['avg_return_succ'] * 100).round(2).astype(str) + '%',
                    df_filtered['avg_mfe_atr'].round(2),
                    df_filtered['avg_mae_atr'].round(2),
                    df_filtered['sharpe_ratio'].round(2),
                    df_filtered['p_value'].round(4)
                ],
                fill_color='#2D2D2D',
                font=dict(color='white', size=11),
                align='center'
            )
        )])

        fig.update_layout(
            title=f"Tactical Micro Table (Session={session_open}, Regime={regime})",
            height=350,
            template=self.template
        )
        return fig
