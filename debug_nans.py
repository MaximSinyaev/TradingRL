import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [np.nan, np.nan, 1, 2, 3],
    'B': [np.nan, 1, 2, 3, 4]
})

first_valid_idx = df.apply(lambda x: x.first_valid_index()).max()
print("First valid:", first_valid_idx)
df_trimmed = df.loc[first_valid_idx:]
print(df_trimmed)
