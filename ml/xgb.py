# from .base_model import *
import xgboost as xgb

# 模型调优五部曲：
# Step 1：选择一组初始参数；
# Step 2：改变 max_depth 和 min_child_weight；
# Step 3：调节 gamma 降低模型过拟合风险；
# Step 4：调节 subsample 和 colsample_bytree 改变数据采样策略；
# Step 5：调节学习率 eta；

class XGBModel(BaseModel):
    def train(self):
        assert len(self.x_train) > 0
        assert len(self.y_train) > 0
        dtrain = xgb.DMatrix(self.x_train, self.y_train) # DMatrix是XGBoost使用的内部数据结构，它针对内存效率和训练速度进行了优化
        watch_list = [(dtrain, 'train')]
        params = {
            'booster': 'gbtree',  # 模型的求解方式，gbtree树型 gblinear线型
            'eval_metric': ['merror', 'mlogloss'],  # 评价指标
            
            # 'objective': 'rank:pairwise', # 排序任务，pairwise
            # 'objective': 'binary:logistic', # 二分类的逻辑回归，输出为概率
            # 'objective': 'multi:softmax',  # 目标，多分类问题，输出一维数据
            'objective': 'multi:softprob',  # 目标，多分类问题，输出 分类数*数据条数的概率矩阵，每行数据表示样本属于每个分类的概率
            'num_class': 4,  # 分类数
            
            'max_depth': 3,  # 树的深度，越大越容易过拟合
            'min_child_weight': 2,# 正则化参数，叶子节点中样本权重和的最小值，小于该值时停止树构建过程
            'gamma': 0.1, # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'subsample': 0.75,  # 对于每棵树，随机采样的比例，小了欠拟合，0.5代表平均采样，一般设置0.5-1
            'colsample_bytree': 0.75,  # 每棵随机采样的列数的占比，一般0.8左右
            'eta': 0.025,  # 迭代步长，如同学习率,[0, 1]
            
            'lambda': 2, # 控制模型复杂度的L2正则化项系数，参数越大，模型越不容易过拟合
            'seed': 0, # 随机数的种子
            'nthread': 8,  # 线程数，-1表示所有线程
            'silent': 0  # 0非静默模式，有输出；1静默模式，不输出运行信息
        }
        
        # num_boost_round：提升迭代的次数，也就是生成多少基模型
        # evals：这是一个列表，用于对训练过程中进行评估列表中的元素
        self.model = xgb.train(params, dtrain, 
                    num_boost_round=self.num_boost_round, evals=watch_list,
                    early_stopping_rounds=100)


    def save(self, path, features):
        importance = self.model.get_fscore()
        self.model.save_model(path)
        with open('%s.weights' % path, 'w+') as fo:
            for i, f in enumerate(features):
                fid = 'f%d' % i
                if fid not in importance:
                    fo.write('%s\t0\n' % f)
                    continue
                fo.write('%s\t%s\n' % (f, importance['f%d' % i]))

        with open('%s.fmap' % path, 'w+') as fo:
            for i, f in enumerate(features):
                fo.write('{0}\t{1}\tq\n'.format(i, f))
            fo.close()

    def predict(self, x_pred):
        self.pred = self.model.predict(xgb.DMatrix(x_pred))

    def load(self, path):
        self.model = xgb.Booster({'nthread': 8})
        self.model.load_model(path)


# plot feature importance manually
from numpy import loadtxt
from xgboost import XGBClassifier
from matplotlib import pyplot
from sklearn.datasets import load_iris
# load data
dataset = load_iris()
# split data into X and y
X = dataset.data
y = dataset.target
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# feature importance
print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
