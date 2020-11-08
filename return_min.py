import pandas as pd
from pyspark.sql.types import TimestampType, FloatType, IntegerType
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp, concat, lit, hour, minute, lag, col, min, max, log
import matplotlib.pyplot as plt
from pyspark.sql.window import Window
import numpy as np

spark = SparkSession.builder.master("local").appName("Return").getOrCreate()
df = spark.read.csv(r'D:\LiBao\data_20200904\NSM_GlobalSelect_Nasdaq\NSM-2016-01-05-TAS-Data-1-of-1-a1.csv', header=True)
df_global_market = spark.read.csv(r'D:\LiBao\data_20200904\NMS-2\NMS-2016-01-05-TAS-Data-1-of-1-a1.csv', header=True)
df_capital_market = spark.read.csv(r'D:\LiBao\data_20200904\NAQ\NAQ-2016-01-05-TAS-Data-1-of-1-a1.csv', header=True)
df = df.union(df_global_market).union(df_capital_market)
# rename the first column and sequence number
df = df.withColumnRenamed('#RIC', 'Ticker')
df = df.withColumnRenamed('Seq. No.', 'SeqNo')
# change the type of the data
df = df.withColumn('Price', df['Price'].cast(FloatType()))
df = df.withColumn('Volume', df['Volume'].cast(FloatType()))
df = df.withColumn('SeqNo', df['SeqNo'].cast(IntegerType()))
# select the trade entry
trade_df = df.where("Type=='Trade'")
# convert trading time into hours and minutes
trade_df = trade_df.withColumn('Hour', hour(trade_df['Exch Time']))
trade_df = trade_df.withColumn('Minute', minute(trade_df['Exch Time']))
# substract the time with 870  to get the minute indicator
trade_df = trade_df.withColumn('MinuteIndicator', trade_df['Hour']*60 + trade_df['Minute']-870)
# select the data during trading hours, trading volume larger than 0 and drop any missing values on trade price
trade_df = trade_df.\
    filter((trade_df['MinuteIndicator'] >= 0) & (trade_df['MinuteIndicator'] <= 390) & (trade_df['Volume']>0)).\
    dropna(subset=('Price'))
# delete rows with ticker contains '![/', all these rows are minute-by-minute summary rather than real trade
trade_df = trade_df.filter(~trade_df['Ticker'].contains('![/'))
# drop duplicates minutes for each stock and
trade_4_computing_ret = trade_df.groupBy(['Ticker', 'MinuteIndicator']).max('SeqNo')\
                        .withColumnRenamed('max(SeqNo)', 'SeqNo')\
                        .join(trade_df, ['Ticker', 'SeqNo', 'MinuteIndicator'])\
                        .orderBy(['Ticker', 'MinuteIndicator'])\
                        .select('Ticker', 'MinuteIndicator', 'Price')
windowSpec = Window.partitionBy('Ticker').orderBy('MinuteIndicator')
trade_4_computing_ret = trade_4_computing_ret.withColumn('Lag Price', lag(trade_4_computing_ret['Price']).over(windowSpec))
trade_4_computing_ret = trade_4_computing_ret.withColumn('Ret',
                                                         log(trade_4_computing_ret['Price']/trade_4_computing_ret['Lag Price']))\
                                                         .dropna(subset=('Ret'))\
                                                         .orderBy('Ticker', 'MinuteIndicator')
final_df = pd.DataFrame(trade_4_computing_ret.collect(), columns=['Ticker', 'MinuteIndicator', 'Price', 'Lag Price', 'Ret'])
new = pd.DataFrame(final_df.Ticker.unique(), [range(1, 391)]*len(final_df.Ticker.unique()))\
                   .reset_index()\
                   .explode('index').rename(columns={'index': 'MinuteIndicator', 0: 'Ticker'})
df_for_plot = new.merge(final_df[['Ticker', 'MinuteIndicator', 'Ret']], how='left', on=['Ticker', 'MinuteIndicator'])\
                        .fillna(0)\
                        .groupby('MinuteIndicator').mean().reset_index()
# Plot
fig, ax = plt.subplots()
ax.grid()
ax.plot(df_for_plot['MinuteIndicator'], df_for_plot['Ret'], color='black', marker='d')
ax.set_xlabel("Minute of the Trading (0 = 9:30am)")
ax.set_ylabel("Ret")
plt.show()