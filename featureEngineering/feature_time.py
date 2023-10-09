import datetime
import time
import pandas as pd


print("------时间点列表-------")
# [年 月 日 一年中的第几天 一年中的第几周 周几]

# 将一个字符串形式的日期转换为pd的日期格式
car_sales = pd.DataFrame({"date_t": ['2023-01-20', '2023-06-22', '2023-06-23', '2023-06-24', '2023-07-24', '2023-07-25', '2023-07-26', '2023-07-27']})
car_sales['date'] = pd.to_datetime(car_sales['date_t'])
print(car_sales['date'].dtype, car_sales['date'][0])

# 取出几月份
car_sales.loc[:, 'month'] = car_sales['date'].dt.month
# 取出星期几
car_sales.loc[:, 'dow'] = car_sales['date'].dt.dayofweek
# 取出一年当中的第几天
car_sales.loc[:, 'doy'] = car_sales['date'].dt.dayofyear
# 取出来是几号
car_sales.loc[:, 'dom'] = car_sales['date'].dt.day
# 判断是否是周末
car_sales.loc[:, 'is_weekend'] = car_sales['dow'].apply(lambda x: 1 if (x == 0 or x == 6) else 0)
print(car_sales)


print("------时间段-------")
# 两个时间的差值(距离现在多少天)
time_start = '2022-10-21'
time_now = time.strftime('%Y-%m-%d', time.localtime(time.time()))

time1 = pd.to_datetime(time_start)
time2 = pd.to_datetime(time_now)

delta_time = time2 - time1  # Timedelta类型
print(delta_time)

print(delta_time.days)  # int

# 时间段再分桶
