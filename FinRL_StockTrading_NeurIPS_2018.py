from finrl import config
from finrl import config_tickers
from finrl.main import check_and_make_directories
import os

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

# %matplotlib inline
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.finrl_meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint

import sys
sys.path.append("../FinRL-Library")
import itertools


from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
)

'''
Use check_and_make_directories() to replace the following
'''
DATA_SAVE_DIR='/dssg/home/acct-aemwx/aemwx-user1/wangyu/results/'
def save_model_results(model,df_account,df_action):
    model_name = list(dict(model=model).keys())[0]
    model.save(f'{DATA_SAVE_DIR}{model_name}')
    df_account.to_csv(f'{DATA_SAVE_DIR}{model_name}_account.csv')
    df_action.to_csv(f'{DATA_SAVE_DIR}{model_name}_action.csv')
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv(DATA_SAVE_DIR+f"{model_name}_perf_stats_all_"+now+'.csv')
    
# check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])



# df = YahooDownloader(start_date = '2009-01-01',
#                      end_date = '2021-10-31',
#                      ticker_list = config_tickers.DOW_30_TICKER).fetch_data()

df = pd.read_csv('/dssg/home/acct-aemwx/aemwx-user1/wangyu/FinRL/datasets/demo.csv',index_col=0)


print(f"config_tickers.DOW_30_TICKER: {config_tickers.DOW_30_TICKER}")


print(f"df.shape: {df.shape}")


df.sort_values(['date','tic'],ignore_index=True).head()



fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.INDICATORS,
                    use_vix=False,
                    use_turbulence=True,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)


list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)


processed_full.sort_values(['date','tic'],ignore_index=True).head(10)


train = data_split(processed_full, '2009-01-01','2020-07-01')
trade = data_split(processed_full, '2020-07-01','2021-10-31')
print(f"len(train): {len(train)}")
print(f"len(trade): {len(trade)}")

#

print(f"train.tail(): {train.tail()}")

#

print(f"trade.head(): {trade.head()}")

#

print(f"config.INDICATORS: {config.INDICATORS}")

#

stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(config.INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


#

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


e_train_gym = StockTradingEnv(df = train, **env_kwargs)

# md

## Environment for Training



#

env_train, _ = e_train_gym.get_sb_env()
print(f"type(env_train): {type(env_train)}")


agent = DRLAgent(env = env_train)


'''
Model Training: 5 models, A2C DDPG, PPO, TD3, SAC
'''

### Model 1: A2C


#

model_a2c = agent.get_model("a2c")

#

# trained_a2c = agent.train_model(model=model_a2c,
#                              tb_log_name='a2c',
#                              total_timesteps=50000)
trained_a2c = agent.train_model(model=model_a2c,
                             tb_log_name='a2c',
                             total_timesteps=50000)

### Model 2: DDPG

#

agent = DRLAgent(env = env_train)
model_ddpg = agent.get_model("ddpg")

#

trained_ddpg = agent.train_model(model=model_ddpg,
                             tb_log_name='ddpg',
                             total_timesteps=50000)

### Model 3: PPO


agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

#

trained_ppo = agent.train_model(model=model_ppo,
                             tb_log_name='ppo',
                             total_timesteps=50000)



### Model 4: TD3


agent = DRLAgent(env = env_train)
TD3_PARAMS = {"batch_size": 100,
              "buffer_size": 1000000,
              "learning_rate": 0.001}

model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)

#

trained_td3 = agent.train_model(model=model_td3,
                             tb_log_name='td3',
                             total_timesteps=30000)


### Model 5: SAC


agent = DRLAgent(env = env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 1000000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)


trained_sac = agent.train_model(model=model_sac,
                             tb_log_name='sac',
                             total_timesteps=60000)


'''
## Trading
Assume that we have $1,000,000 initial capital at 2020-07-01. We use the DDPG model to trade Dow jones 30 stocks.

# md

### Set turbulence threshold
Set the turbulence threshold to be greater than the maximum of insample turbulence data, if current turbulence index is greater than the threshold, then we assume that the current market is volatile
'''


data_risk_indicator = processed_full[(processed_full.date<'2020-07-01') & (processed_full.date>='2009-01-01')]
insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])


# insample_risk_indicator.vix.describe()


# insample_risk_indicator.vix.quantile(0.996)


insample_risk_indicator.turbulence.describe()


insample_risk_indicator.turbulence.quantile(0.996)


'''
### Trade

DRL model needs to update periodically in order to take full advantage of the data, ideally we need to retrain our model yearly, quarterly, or monthly. We also need to tune the parameters along the way, in this notebook I only use the in-sample data from 2009-01 to 2020-07 to tune the parameters once, so there is some alpha decay here as the length of trade date extends.

Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning process and are usually determined by testing some variations.

'''

#trade = data_split(processed_full, '2020-07-01','2021-10-31')
e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = None,risk_indicator_col='turbulence', **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()


# print(f"trade.head(): {trade.head()}")

for model in [trained_a2c,trained_ddpg,trained_ppo,trained_td3,trained_sac]:
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_a2c,
        environment = e_trade_gym)
    save_model_results(model,df_account_value,df_actions)

sys.exit()

# print(f"df_account_value.shape: {df_account_value.shape}")


# print(f"df_account_value.tail(): {df_account_value.tail()}")

# #

# print(f"df_actions.head(): {df_actions.head()}")

# md


'''
# # Part 7: Backtest Our Strategy
Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# md
'''

'''
# 7.1 BackTestStats
pass in df_account_value, this information is stored in env class
'''

#

print("==============Get Backtest Results===========")


#

#baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker="^DJI",
        start = df_account_value.loc[0,'date'],
        end = df_account_value.loc[len(df_account_value)-1,'date'])

stats = backtest_stats(baseline_df, value_col_name = 'close')


#

df_account_value.loc[0,'date']

#

df_account_value.loc[len(df_account_value)-1,'date']

# md
#
# <a id='6.2'></a>
# ## 7.2 BackTestPlot

df_account_value.to_csv('df_account_value.csv')

print("==============Compare to DJIA===========")
# %matplotlib inline
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(df_account_value,
             baseline_ticker = '^DJI',
             baseline_start = df_account_value.loc[0,'date'],
             baseline_end = df_account_value.loc[len(df_account_value)-1,'date'])



