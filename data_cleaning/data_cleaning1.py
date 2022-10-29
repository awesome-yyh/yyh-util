# import pandas as pd

# df = pd.read_csv('xxx.csv')

print("----缺失值----")
print(df.isna().any()) # 对各个列的数据，有缺失值对应True，否则对应False
df['Age'] = df['Age'].fillna(df['Age'].median()) # 按平均值填充缺失值
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) # 按众数填充缺失值
df = df.dropna() # 删掉含有缺失值的行，默认axis=0

print("----重复值----")
df.duplicated().sum() # 统计重复的样本个数
df.drop_duplicates(inplace = True) # 重复样本删除

print("----异常值----")
df['salary'].where(df.salary<=40, 40) # 将不符合条件的值替换掉成指定值

print("----文本处理----")
df["name"] = df["name"].astype(str) # 转str类型

df["name"] = df["name"].str.lower() # 全小写
df["name"] = df["name"].str.upper() # 全大写

df.Email.str.replace('com','cn') # 文本替换
df.Email.str.replace('(.*?)@','xxx@') # 使用正则表达式匹配旧内容
df.Email.str.replace('(.*?)@', lambda x:x.group().upper()) # 将匹配的内容代入函数

df["name"].str.split(',', expand=True, n=1) # 指定字符串作为拆分点，expand可以让拆分的内容扩展成单独一列，n是控制拆分成几列
df.Email.str.split('\@|\.',expand=True) # 使用正则表达式拆分

df.name.str.cat([df.level,df.Email],na_rep='*') # 多列拼接
