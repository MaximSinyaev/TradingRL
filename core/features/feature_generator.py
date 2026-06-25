import pandas as pd
import numpy as np
from itertools import combinations
import math
from typing import Optional

import joblib
from pathlib import Path

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
                 use_ema_diffs: bool = True,
                 d_frac: float = 0.5,
                 hmm_path: Optional[str] = "default"):
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
        self.d_frac = d_frac
        
        self.hmm_model = None
        self.hmm_scaler = None
        
        if hmm_path == "default":
            hmm_path = str(Path(__file__).parent.parent.parent / "models" / "hmm_model.pkl")
            
        if hmm_path is not None:
            hmm_file = Path(hmm_path)
            if not hmm_file.exists():
                raise FileNotFoundError(f"❌ HMM model file not found at {hmm_file}. "
                                        f"Please check the path or set hmm_path=None if the model does not use HMM.")
            try:
                data = joblib.load(hmm_file)
                self.hmm_model = data["model"]
                self.hmm_scaler = data["scaler"]
                print(f"✅ HMM Model loaded from {hmm_file}")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load HMM model from {hmm_file}. Error: {e}")

    def compute_ema(self, series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

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
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                norm_col = f'{col}_over_ema_{self.ema_span}'
                df[norm_col] = df[col] / (base_ema + 1e-9)
        return df

    def add_extra_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        for span in self.extra_ema_spans:
            df[f'ema_{span}'] = self.compute_ema(df[self.price_col], span)
            df[f'ema_{span}_norm'] = df[f'ema_{span}'] / (df['ema_base'] + 1e-9)
        return df

    def compute_garman_klass_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Вычисляет Garman-Klass Volatility, которая учитывает внутрисвечные движения
        (high, low, open, close), что дает более точную оценку волатильности,
        особенно на больших таймфреймах (4h, 1d).
        """
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        gk_vol = 0.5 * log_hl - (2 * math.log(2) - 1) * log_co
        # Применяем EMA для сглаживания волатильности
        return self.compute_ema(gk_vol, span=self.ema_span)

    def compute_fractional_diff(self, series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
        """
        Фракционное дифференцирование (наивная реализация с окном).
        Сохраняет память о тренде лучше, чем обычный log_return.
        Если реализация слишком медленная, можно заменить на просто log_return.
        """
        # Веса для фракционного дифференцирования
        w = [1.]
        for k in range(1, 100): # ограничим глубину истории для скорости
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
        w = np.array(w[::-1])
        
        # Применяем веса как фильтр
        # Используем rolling.apply для простоты
        def apply_weights(x):
            if len(x) < len(w):
                return np.nan
            return np.dot(x[-len(w):], w)
            
        return series.rolling(window=len(w)).apply(apply_weights, raw=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['t'] = np.arange(1, len(df) + 1)

        price = df[self.price_col]
        volume = df[self.volume_col]

        # 1. EMA base
        df['ema_base'] = self.compute_ema(price, self.ema_span)

        # 2. Нормализуем все цены (open/high/low/close) по базовой EMA
        df = self.normalize_price_features(df, df['ema_base'])
        df[f'ema_{self.ema_span}'] = df['ema_base']

        # 3. Дополнительные EMA и их нормализация
        df = self.add_extra_emas(df)
        
        # 4. EMA разности
        if self.use_ema_diffs:
            df = self.compute_ema_diffs(df)

        # 5. Базовые фичи
        df['log_return'] = self.compute_log_return(price)
        df['rsi'] = self.compute_rsi(price, self.rsi_period)
        df['rsi_norm'] = df['rsi'] / 100
        df['normalized_volume'] = self.normalize_volume(volume, self.volume_ema_span)

        macd = self.compute_macd(price)
        df = df.join(macd)
        df['macd_line_norm'] = df['macd_line'] / (df['ema_base'] + 1e-9)
        df['macd_signal_norm'] = df['macd_signal'] / (df['ema_base'] + 1e-9)
        df['macd_hist_norm'] = df['macd_hist'] / (df['ema_base'] + 1e-9)

        # 6. Новые фичи (Garman-Klass Volatility, Fractional Diff)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df['gk_volatility'] = self.compute_garman_klass_volatility(df)
            
        # Фракционное дифференцирование (логарифмированной цены для стабильности)
        df['frac_diff'] = self.compute_fractional_diff(np.log(price), d=self.d_frac)
        # Нормализуем z-score для frac_diff, так как масштаб может быть разный
        df['frac_diff_norm'] = (df['frac_diff'] - df['frac_diff'].rolling(self.ema_span).mean()) / (df['frac_diff'].rolling(self.ema_span).std() + 1e-9)

        # 7. Фичи фьючерсов (Funding Rate, Open Interest)
        if 'fundingRate' in df.columns:
            df['funding_rate'] = df['fundingRate']
            # Дельта фандинга: растет или падает
            df['funding_delta'] = df['fundingRate'].diff()
        else:
            df['funding_rate'] = 0.0
            df['funding_delta'] = 0.0

        if 'sumOpenInterest' in df.columns:
            # Относительное изменение Open Interest (pct_change)
            df['oi_delta'] = df['sumOpenInterest'].pct_change(fill_method=None)
        else:
            df['oi_delta'] = 0.0

        # Заполняем возможные NaN перед формированием вектора
        # Устанавливаем опцию, чтобы избежать FutureWarning
        pd.set_option('future.no_silent_downcasting', True)
        df = df.ffill()
        df = df.bfill()
        df = df.infer_objects(copy=False)

        # 7.5 Интеграция HMM
        hmm_features = []
        if self.hmm_model is not None and self.hmm_scaler is not None:
            obs_cols = ['log_return', 'gk_volatility', 'normalized_volume']
            if all(col in df.columns for col in obs_cols):
                X_hmm = df[obs_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
                try:
                    X_scaled = self.hmm_scaler.transform(X_hmm)
                    hmm_probs = self.hmm_model.predict_proba(X_scaled)
                    
                    for i in range(hmm_probs.shape[1]):
                        col_name = f'hmm_regime_{i}_prob'
                        df[col_name] = hmm_probs[:, i]
                        hmm_features.append(col_name)
                except Exception as e:
                    print(f"⚠️ HMM prediction failed: {e}")

        # 8. Формируем state_vector
        base_features = [
            f'open_over_ema_{self.ema_span}', f'high_over_ema_{self.ema_span}',
            f'low_over_ema_{self.ema_span}', f'close_over_ema_{self.ema_span}',
            'log_return', 'rsi_norm', 'normalized_volume',
            'macd_line_norm', 'macd_signal_norm', 'macd_hist_norm',
            'gk_volatility', 'frac_diff_norm',
            'funding_rate', 'funding_delta', 'oi_delta'
        ]
        
        # Защита: проверяем, что все нужные колонки есть
        available_base_features = [f for f in base_features if f in df.columns]
        
        ema_features = [f'ema_{span}_norm' for span in self.extra_ema_spans]
        ema_diff_features = (
            [f'ema_diff_{s}_{l}' for s, l in combinations([self.ema_span] + self.extra_ema_spans, 2)]
            if self.use_ema_diffs else []
        )

        final_features = available_base_features + ema_features + ema_diff_features + hmm_features
        df['state_vector'] = df[final_features].values.tolist()

        # 9. Убираем строки, где не успели стабилизироваться EMA и окно frac_diff
        cutoff = max(self.ema_span, 100) # 100 это глубина весов frac_diff
        df = df.iloc[cutoff:].reset_index(drop=True)

        return df