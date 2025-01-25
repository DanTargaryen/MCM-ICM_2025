import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('summerOly_hosts.csv')

df['Host'] = df['Host'].str.split(',').str[1]

# 假设 df 是已经处理好的 DataFrame
df.to_csv('processed_host_data.csv', index=False)

# 输出提示消息
print("Data has been successfully exported to 'processed_data.csv'.")