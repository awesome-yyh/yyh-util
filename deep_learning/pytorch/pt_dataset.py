import torch.utils.data as Data


class MyDataset(Data.Dataset):
    """构造Dataset"""
    def __init__(self, inputs, labels=None, transforms=None):
        self.inputs = inputs
        self.labels = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_i = self.inputs[index]
        # 如果数据以文件格式，可以在init做好文件路径list，在此处按index正式读取文件
        if self.transforms:
            input_i = self.transforms(input_i)
            # 特征工程(数值、类别、时间、文本(大小写、标点)、图像)

        if self.labels is not None:
            label = self.labels[index]
            return input_i, label
        else:
            return input_i
