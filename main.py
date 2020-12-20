import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import seaborn as sns
from numpy import loadtxt


def merge_person_partner_data(f_df, m_df):
    columns_to_merge = ['gender', 'race', 'age', 'field_cd', 'career_c', 'int_corr', 'attr1_1', 'sinc1_1', 'intel1_1',
                        'fun1_1', 'amb1_1', 'shar1_1']

    for index, row in f_df.iterrows():
        (iid, pid) = index
        partner = m_df.loc[(pid, iid), :]
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
    return pd.DataFrame(data=imp_mean.transform(original_df.values), index=original_df.index, columns=original_df.columns)


def run_pca(n_components, original_df):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(original_df.values)
    pca_df = pd.DataFrame(data=principal_components, index=original_df.index)
    pca_df.rename({i: "PC{}".format(i) for i in range(n_components)}, axis=1, inplace=True)
    return pca_df


if __name__ == '__main__':
    df = pd.read_csv('Speed Dating Data.csv', encoding="ISO-8859-1")

    column_names = ['iid', 'pid', 'gender', 'race', 'age', 'field_cd', 'career_c', 'int_corr', 'attr1_1', 'sinc1_1',
                    'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'match']
    df = df[column_names]

    df_without_duplicates = df.drop_duplicates(subset=['iid'])
    race_stat = df_without_duplicates.groupby(['race']).size().rename("count").to_frame().reset_index()
    field_stat = df_without_duplicates.groupby(['field_cd']).size().rename("count").to_frame().reset_index()

    dict_races = {1: 'black', 2: 'white', 3: 'latino', 4: 'asian', 5: 'native', 6: 'other'}

    race_stat['value'] = race_stat['race'].map(dict_races)
    notclassified = df_without_duplicates.shape[0] - race_stat['count'].sum()
    for i in range(1, 6):
        if not (i in race_stat.race):
            race_stat = race_stat.append(
                pd.DataFrame([[i, 0, dict_races[i]]], columns=['race', 'count', 'value']))

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
    for i in range(1, 18):
        if not (i in field_stat.field_cd):
            field_stat = field_stat.append(
                pd.DataFrame([[i, 0, dict_fields_of_study[i]]], columns=['field_cd', 'count', 'value']))

    field_stat = field_stat.append(pd.DataFrame([[0, notclassified, 'notclassified']], columns=['field_cd', 'count', 'value']))
    print(field_stat)

    df = df.set_index(['iid', 'pid'])
    print(df)

    # outliers detection
    df = detect_outliers(df, ['int_corr'])

    # replace missing values with mean
    df = replace_missing_values(df)

    # clustering
    cols = loadtxt("args.txt", dtype=str, comments="#", delimiter=",", unpack=False)
    partial_df = df.loc[:, cols]

    x = "age"
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
    fig.suptitle(x, fontsize=20)


    variable = df[x].fillna(df[x].mean())
    breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    variable = variable[(variable > breaks[0]) & (variable <
                                                  breaks[10])]
    sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax)
    des = df[x].describe()
    ax.axvline(des["25%"], ls='--')
    ax.axvline(des["mean"], ls='--')
    ax.axvline(des["75%"], ls='--')
    ax.grid(True)
    des = round(des, 2).apply(lambda x: str(x))
    box = '\n'.join(("min: " + des["min"], "25%: " + des["25%"], "mean: " + des["mean"], "75%: " + des["75%"],
                     "max: " + des["max"]))
    ax.text(0.95, 0.95, box, transform=ax.transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))

    plt.show()

    # dendrogram = sch.dendrogram(sch.linkage(partial_df, method='ward'))
    # plt.show()

    # prepare models
    kmeans = KMeans(n_clusters=2).fit(partial_df)
    # data normalization
    normalized_vectors = preprocessing.normalize(partial_df)
    normalized_kmeans = KMeans(n_clusters=2).fit(normalized_vectors)

    # print results
    print('2 clusters')
    print('kmeans: {}'.format(silhouette_score(partial_df, kmeans.labels_, metric='euclidean')))
    print('Cosine kmeans:{}'.format(silhouette_score(normalized_vectors,
                                                     normalized_kmeans.labels_,
                                                     metric='cosine')))

    # prepare models
    kmeans = KMeans(n_clusters=3).fit(partial_df)
    # data normalization
    normalized_vectors = preprocessing.normalize(partial_df)
    normalized_kmeans = KMeans(n_clusters=3).fit(normalized_vectors)

    # print results
    print('3 clusters')
    print('kmeans: {}'.format(silhouette_score(partial_df, kmeans.labels_, metric='euclidean')))
    print('Cosine kmeans:{}'.format(silhouette_score(normalized_vectors,
                                                     normalized_kmeans.labels_,
                                                     metric='cosine')))

    # Principal Component Analysis
    pca_df = run_pca(2, partial_df)
    pca_df['labels'] = kmeans.labels_
    plt.title('kmeans')
    sns.scatterplot(x=pca_df.PC0, y=pca_df.PC1, hue=pca_df.labels, palette="Set2")
    plt.show()

    norm_pca_df = pca_df.copy()
    norm_pca_df['labels'] = normalized_kmeans.labels_
    plt.title('cosine kmeans')
    sns.scatterplot(x=norm_pca_df.PC0, y=norm_pca_df.PC1, hue=norm_pca_df.labels, palette="Set2")
    plt.show()

    # set all variables between 0 and 1
    scaler = preprocessing.MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(partial_df), columns=partial_df.columns)
    df_scaled['norm_kmeans'] = normalized_kmeans.labels_

    tidy = df_scaled.melt(id_vars='norm_kmeans')
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.barplot(x='norm_kmeans', y='value', hue='variable', data=tidy, palette='Set3')
    plt.legend([''])
    plt.show()

    # female_df = df[df['gender'] == 0].copy()
    # male_df = df[df['gender'] == 1].copy()
    # merged_df = merge_person_partner_data(female_df, male_df)
    #
    # print(merged_df)

    colors = {0: 'red', 1: 'black'}
    fig, ax = plt.subplots()
    grouped = df.groupby('gender')
    for key, group in grouped:
        if key % 1 == 0:
            group.plot(ax=ax, kind='scatter', x='attr1_1', y='age', label=key, color=colors[key], xlim=[0,60], ylim=[20,38])
    plt.show()
    # dla kobiet mniejsze znaczenie ma atrakcyjność

    colors = {0: 'red', 1: 'black'}
    fig, ax = plt.subplots()
    grouped = df.groupby('gender')
    for key, group in grouped:
        if key % 1 == 0:
            group.plot(ax=ax, kind='scatter', x='amb1_1', y='age', label=key, color=colors[key], xlim=[-0.1,20], ylim=[20,38])
    plt.show()
    # dla mężczyzn mniejsze znaczenie mają ambicje
