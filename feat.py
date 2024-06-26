import os
import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):
    @classmethod
    def featureNames(cls):
        return [
            "ob_imb0",
            "ob_imb4",
            "ob_imb9",
            "trade_imb",
            "trade_imbema5",
            "lagret12",
            "trend_midpx",
            "trend_high",
            "trend_low",
            "trend_open",
            "trend_lastpx",
            "trend_tradeBuyQty",
            "trend_tradeSellQty",
            "diff_high_low",
            "diff_midpx_open",
            "diff_lastpx_midpx",
            "ratio_high_low",
            "ratio_open_close",
            "ratio_midpx_open",
            "ratio_tradeBuyQty_tradeSellQty",
            "log_midpx",
            "log_high",
            "log_lastpx",
            "log_tradeBuyQty",
            "log_tradeSellQty",
            "moving_avg_open_5",
            "moving_avg_lastpx_5",
            "rolling_std_midpx_5",
            "rolling_skew_midpx_5",
            "rolling_kurt_midpx_5",
            "max_midpx_last_N",
            "mean_midpx_last_N",
            "product_midpx_tradeBuyQty",
            "ratio_bid_ask",
            "ema_trade_imb",
            "cxlSellHigh",
            "cxlSellTurnover",
            "cxlSellQty",
            "ratio_high_bid0",
            "ratio_ask0_open",
            "spread_bid_ask",
            "spread_high_low",
            "volatility_high_low",
            "momentum_midpx",
            "momentum_trade_imb",
            "log_ask0",
            "log_bid0",
            "volatility_log_midpx",
            "momentum_log_midpx",
            "ratio_high_ask0",
            "ratio_low_bid0",
            "diff_bid0_ask0",
            "ratio_lastpx_bid0",
            "moving_avg_tradeBuyQty_5",
            "moving_avg_tradeSellQty_5",
            "rolling_mean_tradeBuyQty_5",
            "rolling_mean_tradeSellQty_5",
            "ema_midpx",
            "ema_high",
            "ema_low",
            "ema_open",
            "ema_lastpx",
        ]

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    def genFeatures(self, df):
        log.inf("Generating {} features from raw data...".format(len(self.featureNames())))

        # 按 symbol 分组并应用特征提取逻辑
        features_df_list = []
        for symbol, group in df.groupby('symbol'):
            features_df_list.append(self.generate_features_for_group(group))
        
        df = pd.concat(features_df_list)

        xdf = df[self.mcols + self.featureNames()].set_index(self.mcols)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
        
        return xdf.fillna(0), ydf.fillna(0)

    def generate_features_for_group(self, group):
        group.loc[:, "ob_imb0"] = (group["asize0"] - group["bsize0"]) / (group["asize0"] + group["bsize0"] + 1e-9)
        group.loc[:, "ob_imb4"] = (group["asize0_4"] - group["bsize0_4"]) / (group["asize0_4"] + group["bsize0_4"] + 1e-9)
        group.loc[:, "ob_imb9"] = (group["asize5_9"] - group["bsize5_9"]) / (group["asize5_9"] + group["bsize5_9"] + 1e-9)
        group.loc[:, "trade_imb"] = (group["tradeBuyQty"] - group["tradeSellQty"]) / (group["tradeBuyQty"] + group["tradeSellQty"] + 1e-9)
        group.loc[:, "trade_imbema5"] = group["trade_imb"].ewm(halflife=5).mean()
        group.loc[:, "bret12"] = (group["midpx"] - group["midpx"].shift(12)) / (group["midpx"].shift(12) + 1e-9)  # backward return
        
        # 按 interval 计算特征均值并合并
        cxbret = group.groupby("interval")[["bret12"]].mean().reset_index().rename(columns={"bret12": "cx_bret12"})
        group = group.merge(cxbret, on="interval", how="left")
        group.loc[:, "lagret12"] = group["bret12"] - group["cx_bret12"]

        # 新特征：对原特征取对数
        group.loc[:, "log_ob_imb0"] = np.log1p(group["ob_imb0"].replace([-np.inf, np.inf], 0).fillna(0))
        group.loc[:, "log_ob_imb4"] = np.log1p(group["ob_imb4"].replace([-np.inf, np.inf], 0).fillna(0))
        group.loc[:, "log_trade_imbema5"] = np.log1p(group["trade_imbema5"].replace([-np.inf, np.inf], 0).fillna(0))
        group.loc[:, "log_lagret12"] = np.log1p(group["lagret12"].replace([-np.inf, np.inf], 0).fillna(0))
        
        group.loc[:, "trend_midpx"] = group["midpx"].diff().fillna(0)
        group.loc[:, "trend_high"] = group["high"].diff().fillna(0)
        group.loc[:, "trend_low"] = group["low"].diff().fillna(0)
        group.loc[:, "trend_open"] = group["open"].diff().fillna(0)
        group.loc[:, "trend_lastpx"] = group["lastpx"].diff().fillna(0)
        group.loc[:, "trend_tradeBuyQty"] = group["tradeBuyQty"].diff().fillna(0)
        group.loc[:, "trend_tradeSellQty"] = group["tradeSellQty"].diff().fillna(0)

        # 差异（diff）特征
        group.loc[:, "diff_high_low"] = group["high"] - group["low"]
        group.loc[:, "diff_open_close"] = group["open"] - group["lastpx"]
        group.loc[:, "diff_midpx_open"] = group["midpx"] - group["open"]
        group.loc[:, "diff_lastpx_midpx"] = group["lastpx"] - group["midpx"]
        group.loc[:, "diff_bid0_ask0"] = group["bid0"] - group["ask0"]

        # 比率（ratio）特征
        group.loc[:, "ratio_high_low"] = group["high"] / (group["low"] + 1e-9)
        group.loc[:, "ratio_open_close"] = group["open"] / (group["lastpx"] + 1e-9)
        group.loc[:, "ratio_midpx_open"] = group["midpx"] / (group["open"] + 1e-9)
        group.loc[:, "ratio_lastpx_midpx"] = group["lastpx"] / (group["midpx"] + 1e-9)
        group.loc[:, "ratio_tradeBuyQty_tradeSellQty"] = group["tradeBuyQty"] / (group["tradeSellQty"] + 1e-9)
        group.loc[:, "ratio_high_bid0"] = group["high"] / (group["bid0"] + 1e-9)
        group.loc[:, "ratio_ask0_open"] = group["ask0"] / (group["open"] + 1e-9)
        group.loc[:, "ratio_lastpx_bid0"] = group["lastpx"] / (group["bid0"] + 1e-9)

        # 对数特征
        group.loc[:, "log_midpx"] = np.log1p(group["midpx"].replace([-np.inf, np.inf], 0).fillna(0))
        group.loc[:, "log_high"] = np.log1p(group["high"].replace([-np.inf, np.inf], 0).fillna(0))
        group.loc[:, "log_low"] = np.log1p(group["low"].replace([-np.inf, np.inf], 0).fillna(0))
        group.loc[:, "log_open"] = np.log1p(group["open"].replace([-np.inf, np.inf], 0).fillna(0))
        group.loc[:, "log_lastpx"] = np.log1p(group["lastpx"].replace([-np.inf, np.inf], 0).fillna(0))
        group.loc[:, "log_tradeBuyQty"] = np.log1p(group["tradeBuyQty"].replace([-np.inf, np.inf], 0).fillna(0))
        group.loc[:, "log_tradeSellQty"] = np.log1p(group["tradeSellQty"].replace([-np.inf, np.inf], 0).fillna(0))
        group.loc[:, "log_ask0"] = np.log1p(group["ask0"].replace([-np.inf, np.inf], 0).fillna(0))
        group.loc[:, "log_bid0"] = np.log1p(group["bid0"].replace([-np.inf, np.inf], 0).fillna(0))
        
        # 移动平均特征
        group.loc[:, "moving_avg_midpx_5"] = group["midpx"].rolling(window=5).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "moving_avg_high_5"] = group["high"].rolling(window=5).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "moving_avg_low_5"] = group["low"].rolling(window=5).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "moving_avg_open_5"] = group["open"].rolling(window=5).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "moving_avg_lastpx_5"] = group["lastpx"].rolling(window=5).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "moving_avg_tradeBuyQty_5"] = group["tradeBuyQty"].rolling(window=5).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "moving_avg_tradeSellQty_5"] = group["tradeSellQty"].rolling(window=5).mean().replace([-np.inf, np.inf], 0).fillna(0)
        
        # 滚动标准差特征
        group.loc[:, "volatility_midpx_5"] = group["midpx"].rolling(window=5).std().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "rolling_std_midpx_5"] = group["midpx"].rolling(window=5).std().replace([-np.inf, np.inf], 0).fillna(0)
        
        # 滚动均值、方差、偏度、峰度
        group.loc[:, "rolling_mean_midpx_5"] = group["midpx"].rolling(window=5).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "rolling_std_midpx_5"] = group["midpx"].rolling(window=5).std().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "rolling_skew_midpx_5"] = group["midpx"].rolling(window=5).skew().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "rolling_kurt_midpx_5"] = group["midpx"].rolling(window=5).kurt().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "volatility_log_midpx"] = group["log_midpx"].rolling(window=5).std().replace([-np.inf, np.inf], 0).fillna(0)
        
        # 前N分钟的最大/最小/平均价格
        N = 5
        group.loc[:, "max_midpx_last_N"] = group["midpx"].rolling(window=N).max().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "min_midpx_last_N"] = group["midpx"].rolling(window=N).min().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "mean_midpx_last_N"] = group["midpx"].rolling(window=N).mean().replace([-np.inf, np.inf], 0).fillna(0)

        # 特征交互
        group.loc[:, "product_midpx_tradeBuyQty"] = group["midpx"] * group["tradeBuyQty"]
        group.loc[:, "ratio_bid_ask"] = group["bid0"] / (group["ask0"] + 1e-9)  # 防止除零

        # EMA
        group.loc[:, "ema_trade_imb"] = group["trade_imb"].ewm(span=30).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "ema_midpx"] = group["midpx"].ewm(span=30).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "ema_high"] = group["high"].ewm(span=30).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "ema_low"] = group["low"].ewm(span=30).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "ema_open"] = group["open"].ewm(span=30).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "ema_lastpx"] = group["lastpx"].ewm(span=30).mean().replace([-np.inf, np.inf], 0).fillna(0)
        
        # 新增特征计算
        group.loc[:, "cxlSellHigh"] = group["cxlSellHigh"].fillna(0)
        group.loc[:, "cxlSellTurnover"] = group["cxlSellTurnover"].fillna(0)
        group.loc[:, "cxlSellQty"] = group["cxlSellQty"].fillna(0)
        group.loc[:, "spread_bid_ask"] = group["ask0"] - group["bid0"]
        group.loc[:, "spread_high_low"] = group["high"] - group["low"]
        group.loc[:, "volatility_high_low"] = group["high"].rolling(window=5).std() - group["low"].rolling(window=5).std()
        group.loc[:, "momentum_midpx"] = group["midpx"] - group["midpx"].shift(5)
        group.loc[:, "momentum_trade_imb"] = group["trade_imb"] - group["trade_imb"].shift(5)
        group.loc[:, "momentum_log_midpx"] = group["log_midpx"].diff().fillna(0)
        group.loc[:, "ratio_high_ask0"] = group["high"] / (group["ask0"] + 1e-9)
        group.loc[:, "ratio_low_bid0"] = group["low"] / (group["bid0"] + 1e-9)
        group.loc[:, "rolling_mean_tradeBuyQty_5"] = group["tradeBuyQty"].rolling(window=5).mean().replace([-np.inf, np.inf], 0).fillna(0)
        group.loc[:, "rolling_mean_tradeSellQty_5"] = group["tradeSellQty"].rolling(window=5).mean().replace([-np.inf, np.inf], 0).fillna(0)


        
        # 替换无穷大和NaN
        for col in group.columns:
            group[col] = group[col].replace([-np.inf, np.inf], 0).fillna(0)

        return group
