import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import pipeline





if __name__ == '__main__':
    df = pd.read_csv('ElonTweets.csv')
    pipe = pipeline(model="roberta-large-mnli")
    print(pipe(df.iloc[1].Text))
    print(df.iloc[1].Text)