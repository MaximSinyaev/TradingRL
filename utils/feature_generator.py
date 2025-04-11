import pandas as pd
import numpy as np
from itertools import combinations

class FeatureGenerator:
    def __init__(self,
                 price_col: str = 'close',
                 volume_col: str = 'volume',
                 ema_span: int = 20,
                 extra_ema_spans: list = [50, 100],
                 rsi_period: int = 14,
                 volume_ema_span: int = 20,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                use_ema_diffs: bool = True):
        self.price_col = price_col
        self.volume_col = volume_col
        self.ema_span = ema_span
        self.extra_ema_spans = extra_ema_spans
        self.rsi_period = rsi_period
        self.volume_ema_span = volume_ema_span
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.use_ema_diffs = use_ema_diffs

    def compute_ema(self, series: pd.Series, span: int) -> pd.Series:
        ema = series.ewm(span=span, adjust=False).mean()
        # Следует еще обудмать необходимость замены значений до момента, когда EMA стабилизируется
        # На данный момент dataset подрезается от ema_span
        # ema.iloc[:span] = np.nan  # заменим на NaN до момента, когда EMA стабилизируется
        # print(f"Num nan values in EMA with span {span}: {ema.isna().sum()}")
        return ema

    def compute_log_return(self, series: pd.Series) -> pd.Series:
        return np.log(series / series.shift(1))

    def compute_rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def normalize_volume(self, volume: pd.Series, span: int) -> pd.Series:
        ema_vol = self.compute_ema(volume, span)
        return volume / (ema_vol + 1e-9)

    def compute_macd(self, series: pd.Series) -> pd.DataFrame:
        ema_fast = self.compute_ema(series, self.macd_fast)
        ema_slow = self.compute_ema(series, self.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.compute_ema(macd_line, self.macd_signal)
        macd_hist = macd_line - signal_line
        return pd.DataFrame({
            'macd_line': macd_line,
            'macd_signal': signal_line,
            'macd_hist': macd_hist
        })
        
    def compute_ema_diffs(self, df: pd.DataFrame) -> pd.DataFrame:
        all_spans = [self.ema_span] + self.extra_ema_spans
        pairs = list(combinations(all_spans, 2))
        for short, long in pairs:
            col_name = f'ema_diff_{short}_{long}'
            df[col_name] = (df[f'ema_{short}'] - df[f'ema_{long}']) / (df['ema_base'] + 1e-9)
        return df

    def normalize_price_features(self, df: pd.DataFrame, base_ema: pd.Series) -> pd.DataFrame:
        # Нормализуем open, high, low, close по базовой EMA
        # no_norm_mask = df['t'] < self.ema_span
        for col in ['open', 'high', 'low', 'close']:
            norm_col = f'{col}_over_ema_{self.ema_span}'
            df[norm_col] = df[col] / (base_ema + 1e-9)
        return df

    def add_extra_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        for span in self.extra_ema_spans:
            df[f'ema_{span}'] = self.compute_ema(df[self.price_col], span)
            df[f'ema_{span}_norm'] = df[f'ema_{span}'] / (df['ema_base'] + 1e-9)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['t'] = np.arange(1, len(df) + 1)

        price = df[self.price_col]
        volume = df[self.volume_col]

        # 1. EMA base
        df['ema_base'] = self.compute_ema(price, self.ema_span)

        # 2. Нормализуем все цены (open/high/low/close) по базовой EMA
        df = self.normalize_price_features(df, df['ema_base'])
        df[f'ema_{self.ema_span}'] = df['ema_base']  # чтобы не пропустить в парах

        # 3. Дополнительные EMA
        df = self.add_extra_emas(df)
        
        # 4. EMA разности
        if self.use_ema_diffs:
            df = self.compute_ema_diffs(df)

        # 4. Остальные фичи
        df['log_return'] = self.compute_log_return(price)
        df['rsi'] = self.compute_rsi(price, self.rsi_period)
        df['rsi_norm'] = df['rsi'] / 100
        df['normalized_volume'] = self.normalize_volume(volume, self.volume_ema_span)

        macd = self.compute_macd(price)
        df = df.join(macd)
        df['macd_line_norm'] = df['macd_line'] / (df['ema_base'] + 1e-9)
        df['macd_signal_norm'] = df['macd_signal'] / (df['ema_base'] + 1e-9)
        df['macd_hist_norm'] = df['macd_hist'] / (df['ema_base'] + 1e-9)

        # 5. Формируем state_vector
        base_features = [
            f'open_over_ema_{self.ema_span}', f'high_over_ema_{self.ema_span}',
            f'low_over_ema_{self.ema_span}', f'close_over_ema_{self.ema_span}',
            'log_return', 'rsi_norm', 'normalized_volume',
            'macd_line_norm', 'macd_signal_norm', 'macd_hist_norm',
        ]
        ema_features = [f'ema_{span}_norm' for span in self.extra_ema_spans]
        ema_diff_features = (
            [f'ema_diff_{s}_{l}' for s, l in combinations([self.ema_span] + self.extra_ema_spans, 2)]
            if self.use_ema_diffs else []
        )

        df['state_vector'] = df[base_features + ema_features + ema_diff_features].values.tolist()

        # 6. Убираем строки до ema_span
        df = df.iloc[self.ema_span:].reset_index(drop=True)

        return df