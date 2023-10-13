import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder

def get_grid_name(grid):
    name =[x for x in globals() if globals()[x] is grid][0]
    return name

def reduce_mem_usage(grid, verbose=True):
    grid = grid.copy()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = grid.memory_usage().sum() / 1024**2    
    for col in grid.columns:
        col_type = grid[col].dtypes
        if col_type in numerics:
            c_min = grid[col].min()
            c_max = grid[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    grid[col] = grid[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    grid[col] = grid[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    grid[col] = grid[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    grid[col] = grid[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    grid[col] = grid[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    grid[col] = grid[col].astype(np.float32)
                else:
                    grid[col] = grid[col].astype(np.float64)    
    end_mem = grid.memory_usage().sum() / 1024**2
    if verbose: 
        print(' Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return grid

# read the data
calendar = pd.read_csv('data/calendar.csv')
sell_prices = pd.read_csv('data/sell_prices.csv')
submission = pd.read_csv('data/sample_submission.csv')
grid = pd.read_csv('data/sales_train_evaluation.csv')

##### REMOVE RANDOM ID ONCE HAPPY ########
""" rnd_id = (random.sample(list(grid['id'].unique()), 1)[0]) # FOODS_1_058_WI_2_evaluation
grid = grid[grid['id'] == rnd_id] """

grid = reduce_mem_usage(grid)
sub_cols = ['id'] + [f'd_{i}' for i in range(1942, 1970)]
submission.columns = sub_cols
training_days = [f'd_{i}' for i in range(1600, 1942)]
req_cols = grid.columns[:6]
cols_to_keep = req_cols.tolist() + training_days

grid = grid[cols_to_keep]
grid = grid.join(submission.set_index('id'), on='id')
grid = grid.melt(id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                var_name='d', value_name='sales')
grid = grid.join(calendar.set_index('d'), on='d')
grid = grid.join(sell_prices.set_index(['store_id','item_id','wm_yr_wk']), on=['store_id', 'item_id', 'wm_yr_wk'])
grid['sell_price'] = grid['sell_price'].astype(np.float32)
grid['sell_price'] = grid['sell_price'].fillna(-1)

le = LabelEncoder()
cat_vars = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

for var in cat_vars:
    grid[var] = le.fit_transform(grid[var])
    grid[var] = grid[var].astype('int16')

grid['d'] = grid['d'].str.replace('d_', '').astype('int16')    

#print(grid[grid['id'] == 'HOUSEHOLD_2_467_WI_3_evaluation'][['id', 'd', 'sales']].sort_values(by=['d']).tail(50))
grid['date'] = grid['date'].astype('datetime64[ns]')
grid['tm_d'] = grid['date'].dt.day.astype(np.int8)
grid['tm_w'] = grid['date'].dt.weekday.astype(np.int8)
grid['tm_m'] = grid['date'].dt.month.astype(np.int8)
grid['tm_y'] = grid['date'].dt.year
grid['tm_y'] = (grid['tm_y'] - grid['tm_y'].min()).astype(np.int8)
grid['tm_dw'] = grid['date'].dt.dayofweek.astype(np.int8)
grid['tm_w_end'] = (grid['tm_dw'] >= 5).astype(np.int8)
grid = grid.drop(['wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'date'], axis=1)
grid = grid.sort_values(by=['id','d'], ascending=[True,True])
grid = reduce_mem_usage(grid)
grid.to_pickle('data/grid_no_features.pkl')