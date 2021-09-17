"""
This script takes the raw dataset downloaded from Kaggle,
takes a subset of it and saves it in the fraud/data directory
for use in the TFX pipeline.

This step simply allows us to obtain a smaller dataset so
that the pipeline runs much quicker
"""

import yaml
import pandas as pd

keep_cols = [
    'TARGET',
    'AMT_CREDIT',
    'AMT_INCOME_TOTAL',
    'NAME_CONTRACT_TYPE',
    'NAME_TYPE_SUITE',
    'FLAG_MOBIL',
    'CNT_CHILDREN',
]


def main():
    data = pd.read_csv('../data/application_data.csv', usecols=keep_cols)
    data.sample(n=10000).to_csv('../fraud/data/data.csv', index=False)


if __name__ == '__main__':
    main()
