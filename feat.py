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
            "MA_10",
            "MA_50",
            "RSI_12",
            "upper_band",
            "lower_band",
            "volume_MA_10",
            "volume_MA_50",
            "volume_change_rate",
            "price_momentum_10",
            "price_momentum_50",
            "price_reversal_10",
            "price_reversal_50",
            "bid_ask_spread",
            "order_imbalance"
        ]

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    def genFeatures(self, df):
        # df：包含原始数据的dataframe
        # xdf：包含生成特征的数据 ydf：包含目标值的数据
        log.inf("Generating {} features from raw data...".format(len(self.featureNames())))
        df.loc[:, "ob_imb0"] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"])
        df.loc[:, "ob_imb4"] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"])
        df.loc[:, "ob_imb9"] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"])
        df.loc[:, "trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (df["tradeBuyQty"] + df["tradeSellQty"])
        df.loc[:, "trade_imbema5"] = df["trade_imb"].ewm(halflife=5).mean()
        df.loc[:, "bret12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12) # backward return
        cxbret = df.groupby("interval")[["bret12"]].mean().reset_index().rename(columns={"bret12": "cx_bret12"})
        df = df.merge(cxbret, on="interval", how="left")
        df.loc[:, "lagret12"] = df["bret12"] - df["cx_bret12"]
        
        # 技术指标
        # 1.指数移动平均线MA
        df.loc[:, "MA_10"] = df["midpx"].rolling(window=10).mean()
        df.loc[:, "MA_50"] = df["midpx"].rolling(window=50).mean()
        # 2.相对强弱指数RSI
        df.loc[:, "RSI_12"] = self.compute_rsi(df["midpx"])
        # 3.布林带
        df['stddev'] = df['midpx'].rolling(20).std()
        df['upper_band'] = df['midpx'].rolling(20).mean() + (df['stddev'] * 2)
        df['lower_band'] = df['midpx'].rolling(20).mean() - (df['stddev'] * 2)
        
        # 交易量特征
        # 1.成交量移动平均线
        df.loc[:, "volume_MA_10"] = df["tradeBuyQty"].rolling(window=10).mean()
        df.loc[:, "volume_MA_50"] = df["tradeBuyQty"].rolling(window=50).mean()
        # 2.成交量变化率
        df.loc[:, "volume_change_rate"] = df["tradeBuyQty"].pct_change()
        
        # 价格动量和反转
        # 1.价格动量
        df.loc[:, "price_momentum_10"] = df["midpx"].pct_change(periods=10)
        df.loc[:, "price_momentum_50"] = df["midpx"].pct_change(periods=50)
        # 2.价格反转
        df.loc[:, "price_reversal_10"] = df["midpx"].shift(10) / df["midpx"] - 1
        df.loc[:, "price_reversal_50"] = df["midpx"].shift(50) / df["midpx"] - 1
        
        # 市场微观结构特征
        # 1.买卖盘差
        df.loc[:, "bid_ask_spread"] = df["ask0"] - df["bid0"]
        # 2.订单不平衡
        df.loc[:, "order_imbalance"] = (df["bsize0_4"] - df["asize0_4"]) / (df["bsize0_4"] + df["asize0_4"])
        
        # 提取特征和目标值
        xdf = df[self.mcols + self.featureNames()].set_index(self.mcols)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
        return xdf.fillna(0), ydf.fillna(0)
    
    
    def compute_rsi(self, series, period = 12):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
