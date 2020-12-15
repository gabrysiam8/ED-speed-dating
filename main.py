import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


def merge_person_partner_data(f_df, m_df):
    columns_to_merge = ['gender', 'race', 'age', 'field_cd', 'career_c', 'int_corr', 'attr1_1', 'sinc1_1', 'intel1_1',
                        'fun1_1', 'amb1_1', 'shar1_1', 'attr7_2', 'sinc7_2', 'intel7_2', 'fun7_2', 'amb7_2', 'shar7_2']

    for index, row in f_df.iterrows():
        partner = m_df[(m_df['iid'] == row['pid']) & (m_df['pid'] == row['iid'])].iloc[0]
        for column in columns_to_merge:
            f_df.loc[index, 'p_' + column] = partner[column]

    return f_df.copy()


def detect_outliers(original_df, columns):
    # outliers detection (for now only for interests correlation)
    for col in columns:
        mean = original_df[col].mean()
        std = original_df[col].std()
        outliers_df = pd.DataFrame()
        outliers_df['is_outlier'] = abs(original_df[col] - mean) > 3 * std
        # print outliers
        print('Outlier values for ' + col + ': ')
        print(original_df[outliers_df.any(axis=1)])
        outlier_sum = outliers_df.is_outlier.sum()
        print('Number of outliers for ' + col + ': ' + str(outlier_sum))
        # remove outliers from original dataframe
        original_df = original_df[~outliers_df.any(axis=1)]

    return original_df


def replace_missing_values(original_df):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean = imp_mean.fit(original_df.values)
    return pd.DataFrame(data=imp_mean.transform(original_df.values), columns=column_names)


def two_components_pca(original_df):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(original_df.values)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'])
    return pca_df


if __name__ == '__main__':
    df = pd.read_csv('Speed Dating Data.csv', encoding="ISO-8859-1")

    column_names = ['iid', 'gender', 'race', 'age', 'field_cd', 'career_c', 'int_corr', 'attr1_1', 'sinc1_1',
                    'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'match', 'pid']
    df = df[column_names]
    print(df)

    df = detect_outliers(df, ['int_corr'])

    print(df)

    df_without_duplicates = df.drop_duplicates(subset=['iid'])
    race_stat = df_without_duplicates.groupby(['race']).size().rename("count").to_frame().reset_index()

    dict = {1: 'black', 2: 'white', 3: 'latino', 4: 'asian', 5: 'native', 6: 'other'}

    race_stat['value'] = race_stat['race'].map(dict)
    notclassified = df_without_duplicates.shape[0] - race_stat['count'].sum()
    for i in range(1, 6):
        if not (i in race_stat.race):
            race_stat = race_stat.append(
                pd.DataFrame([[i, 0, dict[i]]], columns=['race', 'count', 'value']))

    race_stat = race_stat.append(pd.DataFrame([[0, notclassified, 'notclassified']], columns=['race', 'count', 'value']))
    print(race_stat)

    # replace missing values with mean
    df = replace_missing_values(df)

    # Principal Component Analysis
    principal_df = two_components_pca(df)
    print(principal_df)

    # female_df = df[df['gender'] == 0].copy()
    # male_df = df[df['gender'] == 1].copy()
    # merged_df = merge_person_partner_data(female_df, male_df)
    #
    # print(merged_df)
