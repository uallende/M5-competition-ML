import pandas as pd
import subprocess, time

from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping

def submit_to_kaggle(competition_name, submission_file, message):
    subprocess.run(["kaggle", "competitions", "submit", "-c", competition_name, "-f", submission_file, "-m", message])

if __name__ == "__main__":
       lgb_params = {
              'boosting_type': 'gbdt',
              'objective': 'tweedie',
              'tweedie_variance_power': 1.1,
              'metric': 'rmse',
              'subsample': 0.5,
              'device_type': 'gpu',
              'subsample_freq': 1,
              'min_child_weight': 1,
              'learning_rate': 0.03,
              'num_leaves': 2 ** 11 - 1,
              'min_data_in_leaf': 2 ** 12 - 1,
              'feature_fraction': 0.5,
              'max_bin': 100,
              'n_estimators': 1400, #1400
              'boost_from_average': False,
              'verbosity': -1
              }
       
       grid = pd.read_pickle('data/grid_features.pkl')
       horizon = 28
       
       STEPS = [4,8,12,16,20,24,28] # Training/Prediction every 4 days (Compromise between very granular and long time horizon)
       STEPS = [7,14,21,28] # Training/Prediction every 7 days (Compromise between very granular and long time horizon)
       STEPS = [14,28]
       VAL_DAYS, TEST_DAYS = STEPS[0], STEPS[0]
       STORES = grid.store_id.unique()
       DEPTS = grid.dept_id.unique()
       TARGET = ['sales']

       train_start = grid.d.min()
       train_end = 1941 - horizon
       first_val_day = train_end + 1
       last_val_day = 1941
       first_pred_day = 1941 + 1

       predictions = pd.DataFrame()
       remove_colums = ['id', 'store_id', 'state_id', 'd', 'sales', 'date']
       train_columns = grid.columns[~grid.columns.isin(remove_colums)]
       lags_columns = ['lag_7', 'lag_1', 'lag_2',
       'lag_3', 'lag_4', 'lag_5', 'lag_6']
       print(f'grid columns: {grid.columns}')
       print(f'train columns: {train_columns}')
       print(grid.shape)

       start_time = time.time()

       for store in STORES:
              print(f'************ Training Store {store+1} ************')
              
              #for dept in DEPTS:

              for step in STEPS:

                     grid = pd.read_pickle('data/grid_features.pkl')
                     grid = grid[(grid['store_id'] == store)] #& (grid['dept_id'] == dept)]
                     grid[lags_columns] = grid.groupby(['id'], observed=False)[lags_columns].shift(step)
                     grid = grid.dropna()

                     val_start = first_val_day + step - VAL_DAYS
                     val_end = first_val_day + step - 1
                     pred_start = first_pred_day + step - VAL_DAYS 
                     pred_end = first_pred_day + step - 1

                     trainX = grid[(grid['d'] <= train_end)][train_columns]
                     trainY = grid[(grid['d'] <= train_end)][TARGET]
                     valX = grid[(grid['d'] >= val_start) & (grid['d'] <= val_end)][train_columns]
                     valY = grid[(grid['d'] >= val_start) & (grid['d'] <= val_end)][TARGET]
                     testX = grid[(grid['d'] >= pred_start) & (grid['d'] <= pred_end)][train_columns]
                     print(f'Train shape: {trainX.shape}. Val shape: {valX.shape}. Test shape: {testX.shape}')
                     # rnd_id = (random.sample(list(grid['id'].unique()), 1)[0])
                     # print(grid[(grid['id'] == rnd_id) & (grid['d'] <= 1969)][['id', 'd', 'sales', 'lag_0', 'lag_1', 'lag_3', 'rm_7', 'rm_14']].sort_values(by=['d']).tail(35))

                     # Train
                     lgbm = LGBMRegressor(**lgb_params)
                     callbacks = [early_stopping(stopping_rounds=50, first_metric_only=False)]

                     lgbm.fit(trainX, trainY,
                            eval_set=[(valX, valY)],
                            eval_metric='rmse',
                            callbacks=callbacks)

                     # Predict
                     yhat = lgbm.predict(testX, num_iteration=lgbm.best_iteration_)
                     preds = grid[(grid['d'] >= pred_start) & (grid['d'] <= pred_end)][['id', 'd']]
                     preds['sales'] = yhat
                     predictions = pd.concat([predictions, preds], axis=0)

       # Submission
       end_time = time.time()
       time_taken = (end_time - start_time)/60
       print(f'Time taken to train the model: {time_taken:.2f} minutes')

       predictions.to_pickle(f'submissions/store_dpt_4days.pkl')
       submission = pd.read_csv('data/sample_submission.csv')
       predictions = predictions.pivot(index='id', columns='d', values='sales').reset_index()
       predictions.columns = submission.columns
       predictions = submission[['id']].merge(predictions, on='id', how='left').fillna(1)
       submission_file = "submissions/submission.csv"
       predictions.to_csv(f'{submission_file}', index=False)
       message = "Automated submission"
       competition_name = "m5-forecasting-accuracy"
       submit_to_kaggle(competition_name, submission_file, message)
