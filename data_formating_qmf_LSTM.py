#%%
import pandas as pd 
import numpy as np
import glob

files = glob.glob('/home/fast-pc-2023/Téléchargements/python/data_hugo/* pred.csv')

df = {}
for f in files:
    df[f.split('/')[-1].strip('pred.csv')] = pd.read_csv(f)

df_pred = pd.concat(df).reset_index().drop('level_1', axis=1)
df_pred
#%%
df_pred.columns = ['country', 'date', 'pred']
#df.to_csv('all_preds.csv')
df_pred['date'] = pd.to_datetime(df_pred['date'])
df_pred['year'] = df_pred['date'].dt.year.astype(int)
df_pred = df_pred.groupby(['year', 'country']).last().reset_index()
df_pred['country'] = df_pred['country'].astype(str)
df_pred['country'] = [c.lower().strip() for c in df_pred['country']]
df_pred = df_pred.drop('date', axis=1)
df_pred

#%%

files = glob.glob('/home/fast-pc-2023/Téléchargements/python/data_hugo/*.xls')

df = {}
for f in files:
    df[f.split('/')[-1]] = pd.read_excel(f, header=3)

df = pd.concat(df).reset_index()
df =df.loc[df['Country Name'].isin(['France', 'New Zealand', 'China', 'Norway', 'Argentina'])]
df['full_index'] = df['level_0'] + ' ' + df['Country Name']
df = df.groupby('full_index').first()

df = df.sort_values('Country Name').iloc[:, 6:].T

countries = ['France', 'New Zealand', 'China', 'Norway', 'Argentina']
countries.sort()
df_dict = {}

i = 0
for c in countries:
    df_dict[c] = df.iloc[:,0+i:8+i]
    df_dict[c].columns = [c.split()[0] for c in df_dict[c].columns]

    i +=8

df = pd.concat(df_dict)
df = df.reset_index()
df.level_1 = df.level_1.astype(int) 
df.level_0 = df.level_0.astype(str)

col = ['country', 'year']

o_col = list(df.columns)
o_col.pop(0)
o_col.pop(0)
col.extend(o_col)

df.columns = col
print(df)
df['country'] = [c.lower().strip() for c in df['country']]

#%%
##df = df.set_index(['country', 'year'])
##df_pred = df_pred.set_index(['country', 'year'])
# %%
df = df.merge(df_pred, how='left',on=['year', 'country'] ).ffill().dropna()
print(df)
#%%
df.to_csv('final_all_data_and_pred.csv')
# %%
