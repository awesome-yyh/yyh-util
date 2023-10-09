import numpy as np
from sklearn import preprocessing


class DataProcessing():
    @classmethod
    def newton_norm_dec(cls, arr, factor):
        r"""时间衰减模型之牛顿冷却法
        发布时间等数据距现在的天数越大，重要度越低
        对此类数据调整并归一化（越大的数变得越小）
        
        Args:
            arr:
            factor:
        """
        newton_cool = np.exp(-factor * np.array(arr))  # 牛顿冷却法(减)

        norm = (newton_cool - np.min(newton_cool)) / (np.max(newton_cool) - np.min(newton_cool))  # 归一化

        return norm

    @classmethod
    def newton_norm_inc(cls, arr, factor):
        r"""时间衰减模型之牛顿冷却法
        转发量等数值越高，重要度越高
        对此类数据进行调整并归一化（越大的数变得越大）
        
        Args:
            arr:
            factor:
        """
        newton_cool = np.exp(factor * np.array(arr))  # 牛顿冷却法（增）

        norm = (newton_cool - np.min(newton_cool)) / (np.max(newton_cool) - np.min(newton_cool))  # 归一化

        return norm

    @classmethod
    def std(cls, arr, axis=0):
        r"""标准化
        
        Args:
            axis: 0是按列, 1是按行
        """
        print("mean: ", np.mean(arr, axis=axis))
        print("std: ", np.std(arr, axis=axis))
        stder = (arr - np.mean(arr, axis=axis)) / np.std(arr, axis=axis)
        print("*" * 10)
        print(stder)
        if axis == 0:
            sklearn_std = preprocessing.StandardScaler().fit_transform(arr)
        else:
            sklearn_std = preprocessing.StandardScaler().fit_transform(arr.T).T  # 默认是按列算的，转置2次为按行算
        print("*" * 10)
        print(sklearn_std)
        return sklearn_std

    @classmethod
    def norm_min_max(cls, arr, axis=0):
        r"""归一化
        
        Args:
            axis: 0是按列, 1是按行
        """
        print("min: ", np.min(arr, axis=axis))
        print("max: ", np.max(arr, axis=axis))
        norm = (arr - np.min(arr, axis=axis)) / (np.max(arr, axis=axis) - np.min(arr, axis=axis))
        print(norm)
        return norm

    @classmethod
    def norm(cls, arr, axis=0):
        r"""l2范数归一化
        也叫正则化
        
        Args:
            axis: 0是按列, 1是按行
        """
        print("l2 norm: ", np.linalg.norm(arr, axis=axis, ord=2))
        norm = arr / np.linalg.norm(arr, axis=axis, ord=2)
        print("*" * 20)
        print(norm)
        if axis == 0:
            sklearn_norm = preprocessing.Normalizer(norm="l2").fit_transform(arr.T).T  # 默认是按行算的，转置2次为按列算
        else:
            sklearn_norm = preprocessing.Normalizer(norm="l2").fit_transform(arr)
        print("p" * 10)
        print(sklearn_norm)


if __name__ == "__main__":
    arr = [1, 50, 100]
    print(arr)
    print("牛顿冷却法-减:")
    print(DataProcessing.newton_norm_dec(arr, 0.5))
    print("牛顿冷却法-增:")
    print(DataProcessing.newton_norm_inc(arr, 0.5))
    
    # arr = np.arange(12).reshape((3, 4))
    arr = np.array([arr])
    print("标准化:")
    DataProcessing.std(arr, axis=1)
    print("归一化:")
    DataProcessing.norm_min_max(arr, axis=1)
    print("l2范数归一化:")
    DataProcessing.norm(arr, axis=1)
    
