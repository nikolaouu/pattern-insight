import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class PatternVisualizer:


    def __init__(self, template="plotly_dark"):

        self.template = template

    def plot_macro_heatmap(self, df: pd.DataFrame, metric: str = 'win_rate', regime_filter: str = 'ALL', title: str = None):

        df_filtered = df.copy()
        if 'regime' in df_filtered.columns and regime_filter != 'ALL':
            df_filtered = df_filtered[df_filtered['regime'] == regime_filter]

        pivot_df = df_filtered.pivot_table(
            index='interval',
            columns='session_open',
            values=metric,
            aggfunc='mean'
        )

        values = pivot_df.values
        text_template = "%{z:.1%}" if metric in ['win_rate', 'fail_rate', 'invalid_rate', 'max_drawdown'] else "%{z:.2f}"

        colorscale = "Viridis"

        if metric in ['win_rate', 'sharpe_ratio', 'sortino_ratio']:
            colorscale = "RdYlGn"

        elif metric == 'p_value':
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

    def plot_macro_3d_surface(self, df: pd.DataFrame, metric: str = 'sharpe_ratio', regime_filter: str = 'ALL'):

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
                yaxis=dict(tickvals=np.arange(len(y_intervals)), ticktext=y_intervals)
            ),
            template=self.template,
            height=600
        )
        return fig

    def plot_performance_decay(self, df: pd.DataFrame, primary_metric: str = 'win_rate', show_p_value_threshold: bool = True):

        df_grouped = df.groupby('session_open').agg({
            primary_metric: 'mean',
            'p_value': 'mean',
            'total': 'sum',
            'sharpe_ratio': 'mean'
        }).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

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

        fig.update_yaxes(title_text=f"Mean {primary_metric.upper()}", secondary_y=False)
        fig.update_yaxes(title_text="Statistical Significance (p-value)", secondary_y=True, range=[0, 1])

        return fig

    def plot_mfe_vs_mae_scatter(self, df: pd.DataFrame, use_atr: bool = True):

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

        max_val = max(df[x_col].max(), abs(df[y_col].min()))
        fig.add_shape(
            type="line", line=dict(dash='dash', color='gray'),
            x0=0, y0=0, x1=max_val, y1=-max_val
        )

        fig.update_layout(height=500)
        return fig

    def plot_tactical_table(self, df: pd.DataFrame, session_open = None, regime: str = 'ALL'):

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
