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

    df_without_duplicates = df.drop_duplicates(subset=['iid'])
    race_stat = df_without_duplicates.groupby(['race']).size().rename("count").to_frame().reset_index()
    field_stat = df_without_duplicates.groupby(['field_cd']).size().rename("count").to_frame().reset_index()

    dict = {1: 'black', 2:'white', 3:'latino', 4: 'asian', 5: 'native', 6: 'other'}

    race_stat['value'] = race_stat['race'].map(dict)
    notclassified = df_without_duplicates.shape[0] - race_stat['count'].sum()
    for i in range(1,6):
        if not (i in race_stat.race):
            race_stat = race_stat.append(
                pd.DataFrame([[i, 0, dict[i]]], columns=['race', 'count', 'value']))

    race_stat = race_stat.append(pd.DataFrame([[0, notclassified, 'notclassified']], columns=['race', 'count', 'value']))
    print(race_stat)

    dict_fields_of_study = {1: 'Law', 2:'Math', 3:'Social Science, Psychologist', 4: 'Medical Science, Pharmaceuticals, and Bio Tech',
                            5: 'Engineering', 6: 'English / Creative Writing / Journalism', 7: 'History / Religion / Philosophy',
                            8: 'Business / Econ / Finance', 9: 'Education, Academia', 10 : 'Biological Sciences / Chemistry / Physics',
                            11: 'Social Work', 12: 'Undergrad / undecided', 13: 'Political Science / International Affairs',
                            14: 'Film', 15: 'Fine Arts / Arts Administration', 16: 'Languages', 17: 'Architecture', 18: 'Other'}

    field_stat = df_without_duplicates.groupby(['field_cd']).size().rename("count").to_frame().reset_index()
    field_stat['value'] = field_stat['field_cd'].map(dict_fields_of_study)
    notclassified = df_without_duplicates.shape[0] - field_stat['count'].sum()
    for i in range(1,18):
        if not (i in field_stat.field_cd):
            field_stat = field_stat.append(
                pd.DataFrame([[i, 0, dict[i]]], columns=['field_cd', 'count', 'value']))

    field_stat = field_stat.append(pd.DataFrame([[0, notclassified, 'notclassified']], columns=['field_cd', 'count', 'value']))
    print(field_stat)