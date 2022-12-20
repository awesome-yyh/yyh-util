import os
import re
import pandas as pd


def convert_raw_data_into_csv():
    # Getting the names of all the raw files
    train_pos_files = os.listdir("data/aclImdb/train/pos/")
    train_neg_files = os.listdir("data/aclImdb/train/neg/")
    test_pos_files = os.listdir("data/aclImdb/test/pos/")
    test_neg_files = os.listdir("data/aclImdb/test/neg/")

    para, sentiment, datatype = ([] for i in range(3))
    for file in train_pos_files:
        with open(os.path.join("data/aclImdb/train/pos/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("pos")
                datatype.append("train")
                
    for file in train_neg_files:
        with open(os.path.join("data/aclImdb/train/neg/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("neg")
                datatype.append("train")
                
    for file in test_pos_files:
        with open(os.path.join("data/aclImdb/test/pos/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("pos")
                datatype.append("test")
                
    for file in test_neg_files:
        with open(os.path.join("data/aclImdb/test/neg/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("neg")
                datatype.append("test")

    # Saving data to a csv file named as imdb_master.csv
    df = pd.DataFrame(columns=["review", "type", "label"])
    df["review"] = para
    df["type"] = datatype
    df["label"] = sentiment

    df.to_csv(os.path.join("data","imdb_master.csv"), index=False)
    return df


if __name__=='__main__':
    data_df = convert_raw_data_into_csv()
    print(data_df.shape)
    print(data_df.head(10))
    print(data_df['label'].value_counts(normalize=True)) # 该列各值的占比
    y = pd.get_dummies(data_df['label']).values
    print(y[-1])