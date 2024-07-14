import tensorflow as tf


class MyMSE(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name='my_mse', **kwargs):
        super(MyMSE, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.reduce_mean(tf.math.square(y_true - y_pred))
        return loss


class RelativeError(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE,
                name='relative', **kwargs):
        super(RelativeError, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_upd = tf.where(y_true == 0.0, 1.0, y_true)
        y = tf.math.divide(y_pred, y_upd)
        loss = tf.math.reduce_mean(tf.abs(y - 1))
        return loss


class MaxAbsoluteDeviation(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE,
                name='my_mae', **kwargs):
        super(MaxAbsoluteDeviation, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.reduce_max(tf.math.abs(y_true - y_pred))
        return loss


# Non-differentiable
class InlierRatio(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, 
                name='inlier_ratio', treeshold=0.05, **kwargs):
        super(InlierRatio, self).__init__(reduction=reduction, name=name, **kwargs)
        self.treeshold = treeshold

    def __call__(self, y_true, y_pred, sample_weight=None):
        y = tf.math.divide((y_true - y_pred), tf.where(y_true == 0.0, 1.0, y_true))
        loss = tf.math.reduce_mean(tf.where(tf.abs(y) <= self.treeshold, 0.0, 1.0))
        return loss


class MaxDeviation(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, 
                name='max_deviation', treeshold=0.05, **kwargs):
        super(MaxDeviation, self).__init__(reduction=reduction, name=name, **kwargs)
        self.treeshold = treeshold

    def __call__(self, y_true, y_pred, sample_weight=None):
        y = tf.math.divide((y_true - y_pred), tf.where(y_true == 0, 1, y_true))
        loss = tf.math.reduce_max(tf.where(tf.abs(y) <= self.treeshold, 0.0, abs(y)))
        return loss


class MeanDeviation(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, 
                name='mean_deviation', treeshold=0.05, **kwargs):
        super(MeanDeviation, self).__init__(reduction=reduction, name=name, **kwargs)
        self.treeshold = treeshold

    def __call__(self, y_true, y_pred, sample_weight=None):
        y = tf.math.divide((y_true - y_pred), tf.where(y_true == 0, 1, y_true))
        loss = tf.math.reduce_mean(tf.where(tf.abs(y) <= self.treeshold, self.treeshold, abs(y)))
        return loss