'''
Author: yyh owyangyahe@126.com
Date: 2024-07-14 21:23:00
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2024-07-14 21:23:04
FilePath: /mypython/yyh-util/machine_learning/catbo.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_feature_indices,
    verbose=False
)
model.fit(X_train, y_train)