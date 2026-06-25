import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


class CandleChartVisualizer:
    def __init__(self, use_volume_width: bool = True):
        self.use_volume_width = use_volume_width

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df[required_columns]

    def plot_candlestick(self, data: pd.DataFrame, title: str = "Candlestick Chart", actions: pd.Series = None, portfolio_values: pd.Series = None):
        data = self._prepare_data(data)

        widths = None
        if self.use_volume_width:
            volumes = data['volume']
            normalized = (volumes - volumes.min()) / (volumes.max() - volumes.min())
            widths = 0.2 + normalized * 0.8  # ширина от 0.2 до 1.0

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Candlestick(
            x=data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            whiskerwidth=0.2,
            line_width=widths if self.use_volume_width else 1.0,
            name='Candles'
        ), secondary_y=False)

        if actions is not None:
            if len(actions) != len(data):
                raise ValueError("Length of actions series must match data length")

            # Support both simple ('buy', 'sell') and sized ('buy_50%', 'sell_100%') formats
            buy_mask = actions.str.startswith('buy', na=False)
            sell_mask = actions.str.startswith('sell', na=False)

            buy_signals = data[buy_mask]
            sell_signals = data[sell_mask]
            
            buy_text = actions[buy_mask].apply(lambda x: x.split('_')[1] if '_' in str(x) else '')
            sell_text = actions[sell_mask].apply(lambda x: x.split('_')[1] if '_' in str(x) else '')

            fig.add_trace(go.Scatter(
                x=buy_signals['timestamp'],
                y=buy_signals['low'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='lime', size=10),
                name='Buy Signal',
                text='Size: ' + buy_text,
                hoverinfo='text+y'
            ), secondary_y=False)

            fig.add_trace(go.Scatter(
                x=sell_signals['timestamp'],
                y=sell_signals['high'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='Sell Signal',
                text='Size: ' + sell_text,
                hoverinfo='text+y'
            ), secondary_y=False)

        if portfolio_values is not None:
            if len(portfolio_values) != len(data):
                raise ValueError("Length of portfolio_values must match data length")
            
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=portfolio_values,
                mode='lines',
                line=dict(color='cyan', width=2),
                name='Deposit ($)',
                opacity=0.8
            ), secondary_y=True)

        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="Deposit ($)", secondary_y=True)

        fig.show()
