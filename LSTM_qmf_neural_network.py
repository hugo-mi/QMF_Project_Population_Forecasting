#%%
import pandas as pd
import glob 
import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import matplotlib.pyplot as plt
#%%


df = pd.read_csv('/home/fast-pc-2023/final_all_data_and_pred.csv').drop('Unnamed: 0', axis=1)

df = df.sort_values('year')
df = df.dropna().set_index('year')
df = df.loc[df.country != 'china'] # remove China as the econometrics error are too different from the other countries to include it
df = df.loc[1996:] # remove early years as there was no prediction for them due to the econometric model lags and data quality issues
df, df.describe()

#%%

# ------------------------------------- Preprocessing / Normalization -------------------------------------------------

countries = ['argentina', 'norway', 'new zealand', 'france']
countries.sort()

to_standard_scale = ['birth_rate.xls', 'survival_age_65_female.xls',
       'mortality_rate_male.xls', 'death_rate.xls',
       'mortality_rate_female.xls', 'life_expectancy.xls',
       'survival_age_65_male.xls', 'fertility_rate_female.xls',]

to_label_encode = ['country']

for col in df.columns:


        if col in to_standard_scale:
            df[col] = RobustScaler().fit_transform(df[col].values.reshape(-1, 1))

        if col in to_label_encode:
            df[col] = LabelEncoder().fit_transform(df[col].values.reshape(-1, 1))


df['pred'] = df['pred'].values * 100 # Rescaling otherwise the model didn't managed to converge

#%%
# ----------------------- LSTM NEURAL NETWORK DATA FORMATTING ---------------------------------

from sklearn.preprocessing import OneHotEncoder

train_x, test_x, train_y, test_y, country_train, country_test = {}, {}, {}, {}, {}, {}

# Creating Train and Test Arrays making sure we are taking in the test the latest 5 years for each country

for c in range(len(countries)):
    train_x[c] = sliding_window_view(df.loc[df.country == c].drop('country', axis=1), 2, axis=1)[1:-5]
    test_x[c] = sliding_window_view(df.loc[df.country == c].drop('country', axis=1), 2, axis=1)[1:]
    train_y[c] = df.loc[df.country == c]['pred'].values[1:-5]
    test_y[c] = df.loc[df.country == c]['pred'].values[1:]
    country_train[c] = np.array([c] * len(train_y[c]))
    country_test[c] = np.array([c] * len(test_y[c]))

# Concatenate all Dicts 
all_dict = [train_x, test_x,train_y, test_y, country_train, country_test]

all_dict = [np.vstack(d.values()) for d in all_dict]


# Remapping dict to individual variables
train_x, test_x, train_y, test_y, country_train, country_test = all_dict[0], all_dict[1], all_dict[2], all_dict[3], all_dict[4], all_dict[5]

# Flatten all Arrays except the ones for the LSTM layer and One Hot Encode the country 
train_y, test_y, country_train, country_test = train_y.flatten(), test_y.flatten(), country_train.flatten(), country_test.flatten()
country_train, country_test = OneHotEncoder(sparse=False).fit_transform(country_train.reshape(-1,1)), OneHotEncoder(sparse=False).fit_transform(country_test.reshape(-1,1))

# All data should have the same number of rows
print(train_x.shape, country_train.shape, train_y.shape) 

# %%

# ---------------------------- BUILDING THE NEURAL NETWORK WITH TWO BRANCHES TO ACCOUNT FOR COUNTRY-FIXED EFFECTS -------------------

# One Hot Encoded Country Branch
input0 = Input(shape=(4,))
country_branch = Dense(5, activation='linear')(input0)

# All demographic variables at time t and t-1 Branch
input1 = Input((8, 2))
demographic = LSTM(20, activation='tanh')(input1)
demographic = Dense(16)(demographic)

# Concat both Branch and end with output layer
concat = tf.keras.layers.Concatenate()([country_branch, demographic])

final = Dense(16, activation='gelu')(concat)
out = Dense(1, activation='linear')(final)

model = tf.keras.models.Model(inputs=[input0, input1], outputs=out)
model.compile(loss='mse')
print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='loss' ,restore_best_weights=True, patience=20)

model.fit((country_train, train_x), train_y, epochs=1000,shuffle=False, callbacks=callback, use_multiprocessing=True)


# %%
# ----------------------- Creating Plots for each country ---------------------------------------

import plotly_express as xp


countries = ['Argentina', 'France', 'New Zealand', 'Norway']
figs = {}
for i in range(len(countries)):
    
    nn = model.predict((country_test[i*26:i*26+26], test_x[i*26:i*26+26])).squeeze()
    nn = np.insert(nn,0, np.nan) # To fix the length differential given the LSTM lag of 1
    curr_df = pd.DataFrame(nn)
    curr_df.columns = ['LSTM prediction']

    curr_df['Econometric Res'] = df.loc[df.country==i]['pred'].values
    curr_df.index = set(df.index)

    figs[countries[i]] = xp.line(curr_df, title='LSTM Neural Network Performance at predicting the residuals for {} (out-of-sample are from 2017 onwards)'.format(countries[i]))
    figs[countries[i]] = figs[countries[i]].add_vline(x=2017, line_dash='dash')

#%%
figs['Norway']
#%%
figs['Argentina']

#%%
figs['France']

# %%
figs['New Zealand']


