import numpy as np


def pred2label_one(preds, is_need_prob=False):
    index2label = {0: "label1", 1: "label2", 2: "label3"}
    preds = np.delete(preds, 3)
    
    index = np.where(preds >= 0)[0]
    label = [index2label[i] for i in index]
    if is_need_prob:
        prob = 1 / (1 + np.exp(-preds[index])) * 100
        return label, prob
    else:
        return (label,)


def pred2label(preds, is_need_prob=False):
    index2label = {0: "label1", 1: "label2", 2: "label3"}
    preds = np.delete(preds, 3, axis=1)
    
    label = [[] for _ in range(preds.shape[0])]
    index = np.where(preds >= 0)
    
    if is_need_prob:
        prob = [[] for _ in range(preds.shape[0])]
        prob_all = 1 / (1 + np.exp(-preds[index])) * 100
        
    for i in range(index[0].shape[0]):
        label[index[0][i]].append(index2label[index[1][i]])
        if is_need_prob:
            prob[index[0][i]].append(prob_all[i])
    
    if is_need_prob:
        return label, prob
    else:
        return label


if __name__ == "__main__":
    x = np.array([-4.492408, 5.3194633, 4.2618637, 7.9979463])
    print(pred2label_one(x, is_need_prob=True))
    print(pred2label_one(x))

    x = np.array([[-4.492408, -5.3194633, -4.2618637, -7.9979463], [-4.492408, 5.3194633, -4.2618637, 7.9979463], [4.492408, -5.3194633, -4.2618637, -7.9979463], [4.492408, 5.3194633, 4.2618637, 7.9979463]])
    print(pred2label(x, is_need_prob=True))
    print(pred2label(x))
