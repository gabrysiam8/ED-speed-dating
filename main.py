import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from pandas.plotting import parallel_coordinates
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
import seaborn as sns
from numpy import loadtxt
from sklearn.metrics import classification_report, confusion_matrix
import pylab as pl


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
        outliers_df = pd.DataFrame(index=original_df.index)
        outliers_df['is_outlier'] = abs(original_df[col] - mean) > 3 * std

        outlier_sum = outliers_df.is_outlier.sum()
        print('Number of outliers for ' + col + ': ' + str(outlier_sum))

        outliers = original_df[outliers_df.any(axis=1)]
        # remove outliers
        no_outliers = original_df[~outliers_df.any(axis=1)]

        # print outliers
        print('Outlier values for ' + col + ': ')
        print(outliers)

        # plot blue histogram with red outliers
        plt.hist(outliers[col], color='red')
        plt.hist(no_outliers[col], color='blue')
        plt.title(col)
        plt.show()

        # plot only outliers histogram
        plt.hist(outliers[col], color='red')
        plt.title('Outliers for ' + col)
        plt.show()

        original_df = no_outliers

    return original_df


def replace_missing_values(original_df):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean = imp_mean.fit(original_df.values)
    return pd.DataFrame(data=imp_mean.transform(original_df.values), index=original_df.index, columns=original_df.columns)


def run_pca(n_components, original_df):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(original_df.values)
    print('Eigenvectors for the projection space:')
    print(pca.components_)
    pca_df = pd.DataFrame(data=principal_components, index=original_df.index)
    pca_df.rename({i: "PC{}".format(i) for i in range(n_components)}, axis=1, inplace=True)
    return pca_df


def clustering(cluster_num, original_df, normalized_vectors):
    kmeans = KMeans(n_clusters=cluster_num).fit(original_df)
    normalized_kmeans = KMeans(n_clusters=cluster_num).fit(normalized_vectors)

    silhouette = silhouette_score(original_df, kmeans.labels_, metric='euclidean')
    silhouette_norm = silhouette_score(normalized_vectors, normalized_kmeans.labels_, metric='cosine')

    # print results
    print('{} clusters'.format(cluster_num))
    print('kmeans: {}'.format(silhouette))
    print('Cosine kmeans:{}'.format(silhouette_norm))

    return kmeans, normalized_kmeans, silhouette, silhouette_norm


def run_clustering(original_df):
    # data normalization
    normalized_vectors = preprocessing.normalize(original_df)

    kmeans = []
    normalized_kmeans = []
    silhouette = []
    normalized_silhouette = []
    cluster_num = range(2, 10)

    for i in cluster_num:
        kmeans_i, norm_kmeans_i, silhouette_i, silhouette_norm_i = clustering(i, original_df, normalized_vectors)
        norm_kmeans_i = KMeans(n_clusters=i).fit(normalized_vectors)

        kmeans.append(kmeans_i)
        normalized_kmeans.append(norm_kmeans_i)
        silhouette.append(silhouette_i)
        normalized_silhouette.append(silhouette_norm_i)

    plt.title('Silhouette score')
    sns.lineplot(x=cluster_num, y=normalized_silhouette)
    plt.show()

    max_value = max(normalized_silhouette)
    max_index = normalized_silhouette.index(max_value)
    print('Best cluster number: {}'.format(max_index + 2))
    return kmeans[max_index], normalized_kmeans[max_index]


def draw_cluster_barplot(original_df, model):
    # set all variables between 0 and 1
    scaler = preprocessing.MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(original_df), columns=original_df.columns)
    df_scaled['cluster'] = model.labels_

    tidy = df_scaled.melt(id_vars='cluster')
    plt.subplots(figsize=(15, 5))
    sns.barplot(x='cluster', y='value', hue='variable', data=tidy, palette='Set3')
    plt.legend([''])
    plt.show()


def plot_feature_dependency(clustered_df):
    # atrakcyjność, inteligencja
    sns.scatterplot(x=clustered_df.attr1_1, y=clustered_df.intel1_1, hue=clustered_df.labels,
                    palette="Set2")
    plt.show()

    # wspólne zainteresowania, inteligencja
    sns.scatterplot(x=clustered_df.shar1_1, y=clustered_df.intel1_1, hue=clustered_df.labels,
                    palette="Set2")
    plt.show()

    # atracyjność, ambicja
    sns.scatterplot(x=clustered_df.attr1_1, y=clustered_df.amb1_1, hue=clustered_df.labels, palette="Set2")
    plt.show()

    # atrakcyjność, szczerość
    sns.scatterplot(x=clustered_df.attr1_1, y=clustered_df.sinc1_1, hue=clustered_df.labels, palette="Set2")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('Speed Dating Data.csv', encoding="ISO-8859-1")

    column_names = ['iid', 'pid', 'gender', 'race', 'age', 'field_cd', 'int_corr', 'attr1_1', 'sinc1_1',
                    'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'match', 'age_o']
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

    df.hist(bins=30, figsize=(12, 12))
    pl.suptitle("Histogram for each numeric input variable")
    plt.show()

    # remove rows with nan race value
    df = df[df['race'].notna()]

    # outliers detection
    df = detect_outliers(df, ['int_corr', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1'])

    df.hist(bins=30, figsize=(12, 12))
    pl.suptitle("Histogram for each numeric input variable (after outliers detection)")
    plt.show()

    # replace missing values with mean
    df = replace_missing_values(df)

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

    # group attributes by gender
    colors = {0: 'red', 1: 'black'}
    gender = {0: 'female', 1: 'male'}
    fig, ax = plt.subplots()
    grouped = df.groupby('gender')
    for key, group in grouped:
        if key % 1 == 0:
            group.plot(ax=ax, kind='scatter', x='attr1_1', y='age', label=gender[key], color=colors[key], xlim=[0, 60],
                       ylim=[20, 38])
    plt.show()
    # dla kobiet mniejsze znaczenie ma atrakcyjność

    colors = {0: 'red', 1: 'black'}
    fig, ax = plt.subplots()
    grouped = df.groupby('gender')
    for key, group in grouped:
        if key % 1 == 0:
            group.plot(ax=ax, kind='scatter', x='amb1_1', y='age', label=gender[key], color=colors[key], xlim=[-0.1, 20],
                       ylim=[20, 38])
    plt.show()
    # dla mężczyzn mniejsze znaczenie mają ambicje


    # clustering (gender,attr1_1,sinc1_1,intel1_1,fun1_1,amb1_1,shar1_1)
    cols1, cols2 = loadtxt("args.txt", dtype=str, comments="#", delimiter=",", unpack=False)

    cols1 = [x for x in cols1 if x]
    print(cols1)
    df_to_clustering = df.loc[:, cols1]
    # dendrogram = dendrogram(linkage(partial_df, method='ward'))
    # plt.show()
    kmeans_model, norm_kmeans_model = run_clustering(df_to_clustering)

    # Principal Component Analysis
    pca_df = run_pca(2, df_to_clustering)
    plt.title('K-means (best cluster number)')
    sns.scatterplot(x=pca_df.PC0, y=pca_df.PC1, hue=kmeans_model.labels_, palette="Set2")
    plt.show()
    plt.title('Cosine K-means (2 clusters)')
    sns.scatterplot(x=pca_df.PC0, y=pca_df.PC1, hue=norm_kmeans_model.labels_, palette="Set2")
    plt.show()

    draw_cluster_barplot(df_to_clustering, norm_kmeans_model)

    df_to_clustering['labels'] = norm_kmeans_model.labels_

    plot_feature_dependency(df_to_clustering)

    df['gender'].hist(by=df_to_clustering['labels'])
    pl.suptitle('Gender')
    plt.show()

    size = df_to_clustering.groupby('labels').size()
    print(size)

    # Set all variables between 0 and 1
    scaler = preprocessing.MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_to_clustering), columns=df_to_clustering.columns)
    # Make the plot
    plt.figure(figsize=(15, 10))
    parallel_coordinates(df_scaled, class_column='labels', colormap=plt.get_cmap("Set1"))
    plt.xlabel("Features of data set")
    plt.ylabel("Importance")
    plt.show()

    # 3 clusters
    normalized_vectors = preprocessing.normalize(df_to_clustering)
    kmeans, norm_kmeans, silhouette, silhouette_norm = clustering(3, df_to_clustering, normalized_vectors)

    plt.title('Cosine K-means (3 clusters)')
    sns.scatterplot(x=pca_df.PC0, y=pca_df.PC1, hue=norm_kmeans.labels_, palette="Set2")
    plt.show()

    df_to_clustering['labels'] = norm_kmeans.labels_
    plot_feature_dependency(df_to_clustering)


    # clustering (gender, age, age_o)
    cols2 = [x for x in cols2 if x]
    print(cols2)
    df_to_clustering = df.loc[:, cols2]

    kmeans_model, norm_kmeans_model = run_clustering(df_to_clustering)

    # Principal Component Analysis
    pca_df = run_pca(2, df_to_clustering)
    plt.title('Cosine K-means (2 clusters)')
    sns.scatterplot(x=pca_df.PC0, y=pca_df.PC1, hue=norm_kmeans_model.labels_, palette="Set2")
    plt.show()

    draw_cluster_barplot(df_to_clustering, norm_kmeans_model)

    df_to_clustering['labels'] = norm_kmeans_model.labels_

    # wiek, wiek partnera
    sns.scatterplot(x=df_to_clustering.age, y=df_to_clustering.age_o, hue=df_to_clustering.labels,palette="Set2")
    plt.show()

    df['gender'].hist(by=df_to_clustering['labels'])
    pl.suptitle('Gender')
    plt.show()

    size = df_to_clustering.groupby('labels').size()
    print(size)


    # Classification
    feature_cols = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
    X = df.loc[:, feature_cols]
    y = df.gender
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
          .format(logreg.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
          .format(logreg.score(X_test, y_test)))

    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier().fit(X_train, y_train)
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knn.score(X_train, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
          .format(knn.score(X_test, y_test)))

    y_pred = knn.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.show()
    print(classification_report(y_test, y_pred))

    # female_df = df[df['gender'] == 0].copy()
    # male_df = df[df['gender'] == 1].copy()
    # merged_df = merge_person_partner_data(female_df, male_df)
    #
    # print(merged_df)
