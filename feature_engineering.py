import pandas as pd
import numpy as np

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

grid = pd.read_pickle('data/grid_no_features.pkl')

print('************ ROLLING MEANS ************')
grp = grid.groupby(['id'], group_keys=False)['sales']
for roll in [7,14,30,60,180]:
    grid['rm_' + str(roll)] = grp.apply(lambda x: x.rolling(roll).mean())
   
print('************ ROLLING STATS ************')
for roll in [7,14,30,60,180]:
    grid['max_' + str(roll)] = grp.apply(lambda x: x.rolling(roll).max())
    grid['std_' + str(roll)] = grp.apply(lambda x: x.rolling(roll).std())

print('************ DIFF MEANS ************')
for l in [7,56,140]:
    grid['diff_rm_' + str(l)] = grp.apply(lambda x : x.diff().rolling(l).mean()) 

grp = grid.groupby(['id'], group_keys=False)['sales']
print('************ LAGS ************')
for lag in [0,1,2,3,4,5,6]:
    grid['lag_' + str(lag)] = grp.apply(lambda x: x.shift(lag))

print('************ ROLLING ZEROS ************')
for roll in [7,56,140]:
    grid['is_zero'] = [1 if sales == 0 else 0 for sales in grid['sales']]
    grp = grid.groupby(['id'], group_keys=False)['is_zero']
    grid['rolling_zero_' + str(roll)] = grp.apply(lambda x : x.rolling(roll).sum())
    grid = grid.drop('is_zero', axis = 1)   

grid = reduce_mem_usage(grid)
grid['sell_price'] = grid['sell_price'].astype(np.float32)
print('************ ROLLING PRICES ************')
# If we remove expanding all row values for a given id become the same
grp = grid.groupby(['id'], group_keys=False)['sell_price']
grid['price_max'] = grp.apply(lambda x : x.expanding().max())
grid['price_min'] = grp.apply(lambda x : x.expanding().min())
grid['price_std'] = grp.apply(lambda x : x.expanding().std())
grid['price_mean'] = grp.apply(lambda x : x.expanding().mean())
grid['price_norm'] = grid['sell_price']/grid['price_max']
grid['price_nunique'] = grid.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
grid['item_nunique'] = grid.groupby(['store_id','sell_price'])['item_id'].transform('nunique')

grid = reduce_mem_usage(grid)
grid.to_pickle('data/grid_features.pkl')