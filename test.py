import pandas as pd
import numpy as np

# Creating a sample dataframe
data = {'colA': ['Apple', 'Banana', 'Grape', 'Kiwi'],
        'colB': ['Pear', np.nan, 'Grape', 'Kiwi'],
        'colC': ['Pear', 'Banana', np.nan, 'Kiwi']}
df = pd.DataFrame(data)

# Applying the conditions and replacing values in colA
df.loc[(~df['colB'].isna()) & (~df['colC'].isna()) & (df['colB'] != df['colC']), 'colA'] = df['colB']

# Displaying the updated dataframe
print(df)
