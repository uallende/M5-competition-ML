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

# Rolling Features
grid = pd.read_pickle('itermediate_dfs/no_feat.pkl')
grid = grid[['id','d','sales']]

zero_grid = grid.loc[:,['id','sales']]
zero_grid['is_zero'] = (grid['sales'] == 0).astype(int)
zero_grid = zero_grid.drop(['sales'], axis=1)
grid['is_zero'] = zero_grid['is_zero']

grp = grid.groupby(['id'], group_keys=False, observed=False)['sales']
grp_z = grid.groupby(['id'], group_keys=False, observed=False)['is_zero']

grid = reduce_mem_usage(grid)
print('************ ROLLING LAGS ************')
for roll in [7, 14, 30, 60, 180]:
    grid[f'rolling_zero_{roll}'] = grp_z.transform(lambda x: x.rolling(roll).sum())
    grid[f'rm_{roll}'] = grp.transform(lambda x: x.rolling(roll).mean())
    grid[f'std_{roll}'] = grp.transform(lambda x: x.rolling(roll).std())
    grid[f'diff_rm_{roll}'] = grp.transform(lambda x : x.diff().rolling(roll).mean()) 
    grid[f'max_{roll}'] = grp.transform(lambda x: x.rolling(roll).max())
    grid = reduce_mem_usage(grid)
del zero_grid

grid = reduce_mem_usage(grid)
print('************ LAGS ************')
for lag in np.arange(0, 15, 1):
    grid[f'lag_{lag}'] = grp.transform(lambda x: x.shift(lag))

grid = grid.drop(['is_zero', 'sales'], axis = 1)
ix_to_drop = grid[(grid['d'] <= 1941) & (grid.isna().any(axis=1))].index
grid.drop(index=ix_to_drop, inplace=True)
grid = reduce_mem_usage(grid)
grid.to_pickle('itermediate_dfs/lags.pkl')
del grid

# Mean Categorical encoding 
grid = pd.read_pickle('itermediate_dfs/no_feat.pkl')
grid = grid[['id','d','sales','item_id','dept_id','cat_id','store_id','state_id']]
grid = reduce_mem_usage(grid)
for col_name in ['cat_id', 'item_id', 'dept_id', 'store_id', 'store_id,cat_id', 'store_id,item_id', 'store_id,dept_id']:
    col_names = col_name.split(',')
    s_col_name = col_name.replace(',', '_')
    grid[f'{s_col_name}_enc'] = grid.groupby(col_names, observed=False)['sales'].transform(lambda x: x.expanding().mean())

print('************ CATEGORIES ENCODED ************')
# Memory reduction
grid = grid.drop(['sales','item_id','dept_id','cat_id','store_id','state_id'], axis=1)
grid = reduce_mem_usage(grid)
grid.to_pickle('itermediate_dfs/enc_feats.pkl')
del grid

# Categories Rolling means
grid = pd.read_pickle('itermediate_dfs/no_feat.pkl')
grid = grid[['id','d','store_id','item_id','dept_id','sales','cat_id']]

# For item-level features
item_feats = grid.groupby(['d', 'item_id'])['sales'].mean().reset_index()
grp = item_feats.groupby(['item_id'], observed=False)['sales']
for roll in [7, 30, 60]:
    item_feats[f'item_rm_{roll}'] = grp.transform(lambda x: x.rolling(roll).mean())
item_feats = item_feats.drop(['sales'], axis=1).dropna().reset_index(drop=True)
print(f'Items Completed')

grid = grid.merge(item_feats, on=['item_id','d'], how='left')
grid = grid.drop(['store_id','dept_id','item_id','cat_id','sales'], axis=1)
grid.dropna(inplace=True)
grid = reduce_mem_usage(grid)
grid.to_pickle(f'itermediate_dfs/dimension_feats.pkl')
# On promotion features
bare = pd.read_pickle('itermediate_dfs/bare_cal_price.pkl')
bare = bare.drop(['sales','dept_id','state_id', 'cat_id', 'release', 'd'], axis=1).drop_duplicates()
price = pd.read_pickle('data/prices.pkl')
price['on_promotion'] = price['sell_price'] < (price['price_mean']-price['price_std']*2)
price['ten_prc_promo'] = price['sell_price'] < 0.9 * price['price_mean']
price['prev_wk_promo'] = price['sell_price'] < 0.9 * price['prev_sell_price']
price = price.drop(['sell_price','price_max','price_min','price_std','price_mean','prev_sell_price'], axis=1)
price = price.merge(bare, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
price = price.drop(['store_id','item_id'], axis=1)
new_order = ['id', 'wm_yr_wk', 'on_promotion', 'ten_prc_promo', 'prev_wk_promo']
price = price[new_order]
price.to_pickle('itermediate_dfs/promo.pkl')
print(f'**** On-promotion features added ****')
price.to_pickle('itermediate_dfs/promo.pkl')
del price, bare