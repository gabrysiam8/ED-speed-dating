import pandas as pd
from IPython.display import display

if __name__ == '__main__':
    df = pd.read_csv('Speed Dating Data.csv', encoding="ISO-8859-1")

    column_names = ['iid', 'pid', 'gender', 'race', 'age', 'field_cd', 'career_c', 'int_corr', 'attr1_1', 'sinc1_1',
                    'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'attr7_2', 'sinc7_2', 'intel7_2', 'fun7_2', 'amb7_2',
                    'shar7_2', 'match']
    df = df[column_names]
    print(df)

    # outliers detection (for now only for interests correlation)
    for col in ['int_corr']:
        mean = df[col].mean()
        std = df[col].std()
        print(mean, std)
        outliers_df = pd.DataFrame()
        outliers_df['is_outlier'] = abs(df[col] - mean) > 3 * std
        with_outliers_df = pd.merge(df, outliers_df, left_index=True, right_index=True, how='left')
        display(with_outliers_df[with_outliers_df['is_outlier']])
        outlier_sum = with_outliers_df.is_outlier.sum()
        print('Number of outliers for ' + col + ': ' + str(outlier_sum))
        df = with_outliers_df[with_outliers_df['is_outlier'] == False].drop(['is_outlier'], axis=1)

    print(df)


