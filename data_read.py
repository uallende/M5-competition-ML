import pandas as pd
import numpy as np
import subprocess, psutil, os

from sklearn.preprocessing import LabelEncoder

def get_grid_name(grid):
    name =[x for x in globals() if globals()[x] is grid][0]
    return name

## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def reduce_mem_usage(grid, verbose=True):
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

def submit_to_kaggle(competition_name, submission_file, message):
    kaggle_path = "/root/miniconda3/envs/lightgbm/bin/kaggle"
    subprocess.run([kaggle_path, "competitions", "submit", "-c", competition_name, "-f", submission_file, "-m", message])

## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1    

# Prices
print('Prices')
price = pd.read_csv('data/sell_prices.csv')
calendar = pd.read_csv('data/calendar.csv')
calendar = reduce_mem_usage(calendar)

print(f'Price Stats')
grp = price.groupby(['store_id','item_id'])['sell_price']
price['price_max'] = grp.transform(lambda x: x.expanding().max()).reset_index(drop=True)
price['price_min'] = grp.transform(lambda x: x.expanding().min()).reset_index(drop=True)
price['price_std'] = grp.transform(lambda x: x.expanding().std()).reset_index(drop=True)
price['price_mean'] = grp.transform(lambda x: x.expanding().mean()).reset_index(drop=True)
price['prev_sell_price'] = grp.transform(lambda x: x.shift(1))
del grp
price = reduce_mem_usage(price)
print(price.columns)
price.to_pickle('data/prices.pkl')

# Price & Calendar features
TARGET = 'sales'         # Our main target
END_TRAIN = 1941         # Last day in train set
MAIN_INDEX = ['id','d']  # We can identify item by these columns

eva = pd.read_csv('data/sales_train_evaluation.csv')
print('Create Grid')
index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
grid = pd.melt(eva, 
                  id_vars = index_columns, 
                  var_name = 'd', 
                  value_name = TARGET)

print(f'Train rows. Wide: {len(eva)}, Deep: {len(grid)}')

add_grid = pd.DataFrame()
for i in range(1,29):
    temp_df = eva[index_columns]
    temp_df = temp_df.drop_duplicates()
    temp_df['d'] = 'd_'+ str(END_TRAIN+i)
    temp_df[TARGET] = np.nan
    add_grid = pd.concat([add_grid,temp_df])

grid = pd.concat([grid,add_grid])
grid = grid.reset_index(drop=True)

del temp_df, add_grid, eva
print("{:>20}: {:>8}".format('Original grid',sizeof_fmt(grid.memory_usage(index=True).sum())))

for col in index_columns:
    grid[col] = grid[col].astype('category')

print("{:>20}: {:>8}".format('Reduced grid',sizeof_fmt(grid.memory_usage(index=True).sum())))
grid = reduce_mem_usage(grid)

price = pd.read_csv('data/sell_prices.csv')
calendar = pd.read_csv('data/calendar.csv')
print('Release week')

release_df = price.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
release_df.columns = ['store_id','item_id','release']

grid = merge_by_concat(grid, release_df, ['store_id','item_id'])
del release_df

grid = merge_by_concat(grid, calendar[['wm_yr_wk','d']], ['d'])
grid = grid[grid['wm_yr_wk']>=grid['release']]
grid = grid.reset_index(drop=True)
grid = reduce_mem_usage(grid)

grid = merge_by_concat(grid, price, ['store_id','item_id','wm_yr_wk'])
grid = reduce_mem_usage(grid)
print(grid.columns)
del price, calendar
grid['release'] = grid['release'] - grid['release'].min()
grid['release'] = grid['release'].astype(np.int16)

price = pd.read_pickle('data/prices.pkl')
grid = grid.merge(price.drop(['sell_price'], axis=1), on = ['store_id','item_id','wm_yr_wk'], how='left')

calendar = pd.read_csv('data/calendar.csv')
grid = grid.merge(calendar.drop(['weekday','year','wday','month','wm_yr_wk'], axis=1), on = ['d'], how = 'left')

le = LabelEncoder()
cat_vars = ['item_id','store_id','dept_id','cat_id','state_id','event_name_1','event_type_1','event_name_2','event_type_2']
del price, calendar
for cat in cat_vars:
    grid[cat] = le.fit_transform(grid[cat])

grid['date'] = grid['date'].astype('datetime64[ns]')
grid['tm_d'] = grid['date'].dt.day.astype(np.int8)
grid['tm_w'] = grid['date'].dt.isocalendar().week.astype(np.int8)
grid['tm_m'] = grid['date'].dt.month.astype(np.int8)
grid['tm_y'] = grid['date'].dt.year
grid['tm_y'] = (grid['tm_y'] - grid['tm_y'].min()).astype(np.int8)
grid['tm_dw'] = grid['date'].dt.dayofweek.astype(np.int8)
grid['tm_w_end'] = (grid['tm_dw'] >= 5).astype(np.int8)
grid['d'] = grid['d'].str.replace('d_', '').astype('int16')
grid = reduce_mem_usage(grid)
grid.to_pickle('itermediate_dfs/no_feat.pkl')
del grid