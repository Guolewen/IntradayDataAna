import pandas as pd
from pyspark.sql.types import TimestampType, FloatType
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp, concat, lit
import matplotlib.pyplot as plt


spark = SparkSession.builder.master("local").appName("Bid Ask").getOrCreate()
df = spark.read.csv(r'D:\LiBao\data_20200904\NSM_GlobalSelect_Nasdaq\NSM-2016-01-05-TAS-Data-1-of-1-a1.csv', header=True)
df_global_market = spark.read.csv(r'D:\LiBao\data_20200904\NMS-2\NMS-2016-01-05-TAS-Data-1-of-1-a1.csv', header=True)
df_capital_market = spark.read.csv(r'D:\LiBao\data_20200904\NAQ\NAQ-2016-01-05-TAS-Data-1-of-1-a1.csv', header=True)
df = df.union(df_global_market).union(df_capital_market)
# merge date and time
df = df.withColumn('FullTime', concat(df['Date[G]'],lit(' '), df['Quote Time']))
# convert the full datetime to unix_timestamp in seconds
df = df.withColumn('FullTime', unix_timestamp(df['FullTime'], 'dd-MMM-yyyy HH:mm:ss.SSS'))
# create start unix_timestamp
df = df.withColumn('StartTime', lit('05-JAN-2016 14:30:00.000'))
df = df.withColumn('StartTime', unix_timestamp(df['StartTime'], 'dd-MMM-yyyy HH:mm:ss.SSS'))
df = df.withColumnRenamed('#RIC', 'Ticker')
cols = ['Bid Price', 'Ask Price']
for col_name in cols:
    df = df.withColumn(col_name, df[col_name].cast(FloatType()))
quotation_df = df.where("type=='Quote'")
quotation_df = quotation_df.filter(~quotation_df['Ticker'].contains('![/'))
# calculate the differences in seconds between quote time and opening time
quotation_df = quotation_df.withColumn('SecondIndicator', quotation_df['FullTime'] - quotation_df['StartTime'])
# select the quotation during trading hours: SecondIndicator between 0 and 23400
quotation_in_trading_df = quotation_df.filter((quotation_df['SecondIndicator'] >= 0) & (quotation_df['SecondIndicator'] < 23400))
# calculate percentage bid-ask spread for each entry
quotation_in_trading_df = quotation_in_trading_df.withColumn('Bid Ask Spread', (quotation_in_trading_df['Ask Price'] - quotation_in_trading_df['Bid Price'])/((quotation_in_trading_df['Ask Price'] + quotation_in_trading_df['Bid Price'])/2)*100)
# Groupby SecondIndicator to form an average second-by-second bid ask spread
avg = quotation_in_trading_df.groupBy('SecondIndicator').avg('Bid Ask Spread').collect()
avg_df = pd.DataFrame(avg, columns=['SecondIndicator', 'Avg Bid Ask Spread'])
# calculate
avg_df['MinuteIndicator'] = avg_df['SecondIndicator']//60
df_for_plot = avg_df.groupby(['MinuteIndicator']).mean('Avg Bid Ask Spread').reset_index()[['MinuteIndicator', 'Avg Bid Ask Spread']]
# plot
fig, ax = plt.subplots()
ax.grid()
ax.plot(df_for_plot['MinuteIndicator'], df_for_plot['Avg Bid Ask Spread'], color='black', marker='d', markersize=6)
ax.set_xlabel("Minute of the Trading (0 = 9:30am)")
ax.set_ylabel("% Spread")
plt.show()