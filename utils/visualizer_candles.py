import plotly.graph_objects as go
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

    def plot_candlestick(self, data: pd.DataFrame, title: str = "Candlestick Chart"):
        data = self._prepare_data(data)

        widths = None
        if self.use_volume_width:
            volumes = data['volume']
            normalized = (volumes - volumes.min()) / (volumes.max() - volumes.min())
            widths = 0.2 + normalized * 0.8  # ширина от 0.2 до 1.0

        fig = go.Figure()

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
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )

        fig.show()