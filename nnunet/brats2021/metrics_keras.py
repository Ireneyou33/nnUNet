import keras.backend as K
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    F1 score: https://en.wikipedia.org/wiki/F1_score
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def r_square(y_true, y_pred):
    '''
    mean_squared_error=np.sum((y_pred-y_true)**2)/len(y_true)
    1- mean_squared_error(y_true, y_pred)/ np.var(y_true)
    '''
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    SSR = K.mean(K.square(y_pred_f - K.mean(y_true_f)), axis=-1)
    SST = K.mean(K.square(y_true_f - K.mean(y_true_f)), axis=-1)
    # return SSR/SST
    return (1 - SSR/SST)

def r_square2(y_true, y_pred):
    '''
    mean_squared_error=np.sum((y_pred-y_true)**2)/len(y_true)
    1- mean_squared_error(y_true, y_pred)/ np.var(y_true)
    '''
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    SSR = K.mean(K.square(y_pred_f - y_true_f), axis=-1)
    SST = K.mean(K.square(y_true_f - K.mean(y_true_f)), axis=-1)
    return (1 - SSR/SST)

def r_square3(y_true, y_pred):
    '''
    mean_squared_error=np.sum((y_pred-y_true)**2)/len(y_true)
    1- mean_squared_error(y_true, y_pred)/ np.var(y_true)
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    SSR = K.mean(K.square(y_pred_f - K.mean(y_true_f)), axis=-1)
    SST = K.mean(K.square(y_true_f - K.mean(y_true_f)), axis=-1)
    return SSR/SST
    # return (1 - SSR/SST)


def rr_square(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    r = 1 - mean_squared_error(y_true_f, y_pred_f) / np.var(y_pred_f)
    
    



class Sensitivity:
    def __init__(self, thresh):
        self.thresh = 10 ** (-thresh)
        self.__name__ = 'sensi_' + str(thresh)

    def __call__(self, y_true, y_pred):
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        target = K.sum(y_true_f * y_pred_f, axis=-1) > self.thresh
        target_count = K.sum(K.cast(target, K.floatx()))
        total_area = K.sum(y_true_f, axis=-1) > K.epsilon()
        truth_count = K.sum(K.cast(total_area, K.floatx()))
        return target_count / truth_count
    
    
class Specificity:
    def __init__(self, thresh=2):
        self.thresh = 10 ** (-thresh)
        self.__name__ = 'speci' + str(thresh)

    def __call__(self, y_true, y_pred):
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        target = K.cast(K.sum(y_true_f + y_pred_f, axis=-1) < self.thresh, K.floatx())
        total_area = K.cast(K.sum(y_true_f, axis=-1) < K.epsilon(), K.floatx())

        target_count = K.sum(target)
        truth_count = K.sum(total_area)
        return target_count / truth_count


if __name__ == '__main__':
    # a = K.round(0.9)
    # print(K.eval(a))
    # a = acc([0.1], [0.2])
    a = precision(0.1, 0.2)
    print(K.eval(a))
