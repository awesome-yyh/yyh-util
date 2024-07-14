'''
Author: yyh owyangyahe@126.com
Date: 2024-07-14 21:22:01
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2024-07-14 21:22:07
FilePath: /mypython/yyh-util/machine_learning/s.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective='binary:logistic'
)
model.fit(X_train, y_train)