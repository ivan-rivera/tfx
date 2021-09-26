"""
This script takes the raw dataset downloaded from Kaggle,
takes a subset of it and saves it in the fraud/data directory
for use in the TFX pipeline.

This step simply allows us to obtain a smaller dataset so
that the pipeline runs much quicker
"""

import pandas as pd
from sklearn.model_selection import train_test_split

keep_cols = [
    'TARGET',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'AMT_CREDIT',
    'OWN_CAR_AGE',
    'AMT_INCOME_TOTAL',
    'NAME_CONTRACT_TYPE',
    'OCCUPATION_TYPE',
    'ORGANIZATION_TYPE',
    'NAME_TYPE_SUITE',
    'FLAG_MOBIL',
    'CNT_CHILDREN',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
]


def main():
    data = pd.read_csv('../data/application_data.csv', usecols=keep_cols)
    train, test = train_test_split(data.sample(n=15_000), train_size=10_000)
    train.to_csv('../fraud/data/data.csv', index=False)
    test.to_csv('../data/test.csv', index=False)


if __name__ == '__main__':
    main()
