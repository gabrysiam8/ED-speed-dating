import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import loadtxt
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


## for statistical tests


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
    return pd.DataFrame(data=imp_mean.transform(original_df.values), index=original_df.index,
                        columns=original_df.columns)


def run_pca(n_components, original_df):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(original_df.values)
    pca_df = pd.DataFrame(data=principal_components, index=original_df.index)
    pca_df.rename({i: "PC{}".format(i) for i in range(n_components)}, axis=1, inplace=True)
    return pca_df


def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"


if __name__ == '__main__':
    df = pd.read_csv('Speed Dating Data.csv', encoding="ISO-8859-1")

    column_names = ['iid', 'pid', 'gender', 'race', 'age', 'field_cd', 'career_c', 'int_corr', 'attr1_1', 'sinc1_1',
                    'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'match', 'dec_o']
    df = df[column_names]

    df_without_duplicates = df.drop_duplicates(subset=['iid'])
    race_stat = df_without_duplicates.groupby(['race']).size().rename("count").to_frame().reset_index()
    field_stat = df_without_duplicates.groupby(['field_cd']).size().rename("count").to_frame().reset_index()

    dicto = {1: 'black', 2: 'white', 3: 'latino', 4: 'asian', 5: 'native', 6: 'other'}

    race_stat['value'] = race_stat['race'].map(dicto)
    notclassified = df_without_duplicates.shape[0] - race_stat['count'].sum()
    for i in range(1, 6):
        if not (i in race_stat.race):
            race_stat = race_stat.append(
                pd.DataFrame([[i, 0, dicto[i]]], columns=['race', 'count', 'value']))

    race_stat = race_stat.append(
        pd.DataFrame([[0, notclassified, 'notclassified']], columns=['race', 'count', 'value']))
    print(race_stat)

    dict_fields_of_study = {1: 'Law', 2: 'Math', 3: 'Social Science, Psychologist',
                            4: 'Medical Science, Pharmaceuticals, and Bio Tech',
                            5: 'Engineering', 6: 'English / Creative Writing / Journalism',
                            7: 'History / Religion / Philosophy',
                            8: 'Business / Econ / Finance', 9: 'Education, Academia',
                            10: 'Biological Sciences / Chemistry / Physics',
                            11: 'Social Work', 12: 'Undergrad / undecided',
                            13: 'Political Science / International Affairs',
                            14: 'Film', 15: 'Fine Arts / Arts Administration', 16: 'Languages', 17: 'Architecture',
                            18: 'Other'}

    field_stat = df_without_duplicates.groupby(['field_cd']).size().rename("count").to_frame().reset_index()
    field_stat['value'] = field_stat['field_cd'].map(dict_fields_of_study)
    notclassified = df_without_duplicates.shape[0] - field_stat['count'].sum()
    for i in range(1, 18):
        if not (i in field_stat.field_cd):
            field_stat = field_stat.append(
                pd.DataFrame([[i, 0, dicto[i]]], columns=['field_cd', 'count', 'value']))

    field_stat = field_stat.append(
        pd.DataFrame([[0, notclassified, 'notclassified']], columns=['field_cd', 'count', 'value']))
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

    # dendrogram = sch.dendrogram(sch.linkage(partial_df, method='ward'))
    # plt.show()

    # # prepare models
    # kmeans = KMeans(n_clusters=2).fit(partial_df)
    # # data normalization
    # normalized_vectors = preprocessing.normalize(partial_df)
    # normalized_kmeans = KMeans(n_clusters=2).fit(normalized_vectors)
    #
    # # print results
    # print('2 clusters')
    # print('kmeans: {}'.format(silhouette_score(partial_df, kmeans.labels_, metric='euclidean')))
    # print('Cosine kmeans:{}'.format(silhouette_score(normalized_vectors,
    #                                                  normalized_kmeans.labels_,
    #                                                  metric='cosine')))
    #
    # # prepare models
    # kmeans = KMeans(n_clusters=3).fit(partial_df)
    # # data normalization
    # normalized_vectors = preprocessing.normalize(partial_df)
    # normalized_kmeans = KMeans(n_clusters=3).fit(normalized_vectors)
    #
    # # print results
    # print('3 clusters')
    # print('kmeans: {}'.format(silhouette_score(partial_df, kmeans.labels_, metric='euclidean')))
    # print('Cosine kmeans:{}'.format(silhouette_score(normalized_vectors,
    #                                                  normalized_kmeans.labels_,
    #                                                  metric='cosine')))

    # Principal Component Analysis
    # pca_df = run_pca(2, partial_df)
    # pca_df['labels'] = kmeans.labels_
    # plt.title('kmeans')
    # sns.scatterplot(x=pca_df.PC0, y=pca_df.PC1, hue=pca_df.labels, palette="Set2")
    # plt.show()
    #
    # norm_pca_df = pca_df.copy()
    # norm_pca_df['labels'] = normalized_kmeans.labels_
    # plt.title('cosine kmeans')
    # sns.scatterplot(x=norm_pca_df.PC0, y=norm_pca_df.PC1, hue=norm_pca_df.labels, palette="Set2")
    # plt.show()
    #
    # # set all variables between 0 and 1
    # scaler = preprocessing.MinMaxScaler()
    # df_scaled = pd.DataFrame(scaler.fit_transform(partial_df), columns=partial_df.columns)
    # df_scaled['norm_kmeans'] = normalized_kmeans.labels_
    #
    # tidy = df_scaled.melt(id_vars='norm_kmeans')
    # fig, ax = plt.subplots(figsize=(15, 5))
    # sns.barplot(x='norm_kmeans', y='value', hue='variable', data=tidy, palette='Set3')
    # plt.legend([''])
    # plt.show()

    # female_df = df[df['gender'] == 0].copy()
    # male_df = df[df['gender'] == 1].copy()
    # merged_df = merge_person_partner_data(female_df, male_df)
    #
    # print(merged_df)

    ##################################################
    dtf = pd.read_csv('Speed Dating Data.csv', encoding="ISO-8859-1")
    # dtf.head()
    #
    # dic_cols = {col: utils_recognize_type(dtf, col, max_cat=10) for col in dtf.columns}
    # heatmap = dtf.isnull()
    # for k, v in dic_cols.items():
    #     if v == "num":
    #         heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
    #     else:
    #         heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    # sns.heatmap(heatmap, cbar=False).set_title('Dataset Overview')
    # plt.show()
    # print("\033[1;37;40m Categerocial ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")
    #
    # dtf = dtf.set_index("iid")
    # dtf = dtf.rename(columns={"match": "Y"})
    #
    # y = "Y"
    # ax = dtf[y].value_counts().sort_values().plot(kind="barh")
    # totals = []
    # for i in ax.patches:
    #     totals.append(i.get_width())
    # total = sum(totals)
    # for i in ax.patches:
    #     ax.text(i.get_width() + .3, i.get_y() + .20,
    #             str(round((i.get_width() / total) * 100, 2)) + '%',
    #             fontsize=10, color='black')
    # ax.grid(axis="x")
    # plt.suptitle(y, fontsize=20)
    # plt.show()
    #
    # x = "age"
    # fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    # fig.suptitle(x, fontsize=20)
    # ### distribution
    # ax[0].title.set_text('distribution')
    # variable = dtf[x].fillna(dtf[x].mean())
    # breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    # variable = variable[(variable > breaks[0]) & (variable <
    #                                               breaks[10])]
    # sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
    # des = dtf[x].describe()
    # ax[0].axvline(des["25%"], ls='--')
    # ax[0].axvline(des["mean"], ls='--')
    # ax[0].axvline(des["75%"], ls='--')
    # ax[0].grid(True)
    # des = round(des, 2).apply(lambda x: str(x))
    # box = '\n'.join(("min: " + des["min"], "25%: " + des["25%"], "mean: " + des["mean"], "75%: " + des["75%"],
    #                  "max: " + des["max"]))
    # ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right",
    #            bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    # ### boxplot
    # ax[1].title.set_text('outliers (log scale)')
    # tmp_dtf = pd.DataFrame(dtf[x])
    # tmp_dtf[x] = np.log(tmp_dtf[x])
    # tmp_dtf.boxplot(column=x, ax=ax[1])
    # plt.show()
    # ## Create new column
    # dtf["Cabin_section"] = dtf["race"].apply(lambda x: str(x)[0])
    # ## Plot contingency table
    # cont_table = pd.crosstab(index=dtf["Cabin_section"],
    #                          columns=dtf["Y"], values=dtf["gender"], aggfunc="sum")
    # sns.heatmap(cont_table, annot=True, cmap="YlGnBu", fmt='.0f',
    #             linewidths=.5).set_title(
    #     'Cabin_section vs Pclass (filter: Y)')
    # plt.show()
    #
    # ## split data
    # dtf_train, dtf_test = model_selection.train_test_split(df, test_size=0.3)
    # ## print info
    # print("X_train shape:", dtf_train.drop("match", axis=1).shape, "| X_test shape:",
    #       dtf_test.drop("match", axis=1).shape)
    # print("y_train mean:", round(np.mean(dtf_train["match"]), 2), "| y_test mean:",
    #       round(np.mean(dtf_test["match"]), 2))
    # print(dtf_train.shape[1], "features:", dtf_train.drop("match", axis=1).columns.to_list())
    #
    # dic_cols = {col: utils_recognize_type(df, col, max_cat=20) for col in df.columns}
    # heatmap = df.isnull()
    # for k, v in dic_cols.items():
    #     if v == "num":
    #         heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
    #     else:
    #         heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    # sns.heatmap(heatmap, cbar=False).set_title('Dataset Overview')
    # plt.show()
    # print("\033[1;37;40m Categerocial ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")
    #
    # ## create dummy
    # dummy = pd.get_dummies(dtf_train["gender"],
    #                        prefix="gender", drop_first=True)
    # dtf_train = pd.concat([dtf_train, dummy], axis=1)
    # print(dtf_train.filter(like="gender", axis=1).head())
    # ## drop the original categorical column
    # dtf = dtf_train.drop("gender", axis=1)
    #
    # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # X = scaler.fit_transform(dtf_train.drop("attr1_1", axis=1))
    # dtf_scaled = pd.DataFrame(X, columns=dtf_train.drop("attr1_1", axis=1).columns, index=dtf_train.index)
    # dtf_scaled["attr1_1"] = dtf_train["attr1_1"]
    # dtf_scaled.head()
    #
    # corr_matrix = dtf.copy()
    # for col in corr_matrix.columns:
    #     if corr_matrix[col].dtype == "O":
    #         corr_matrix[col] = corr_matrix[col].factorize(sort=True)[0]
    # corr_matrix = corr_matrix.corr(method="pearson")
    # sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
    # plt.title("pearson correlation")
    #
    # print("DUPA")
    #
    # from sklearn.datasets import make_blobs
    # from matplotlib import pyplot
    # from pandas import DataFrame
    #
    # colors = {0: 'red', 1: 'black'}
    # fig, ax = pyplot.subplots()
    # grouped = df.groupby('gender')
    # for key, group in grouped:
    #     if key % 1 == 0:
    #         group.plot(ax=ax, kind='scatter', x='attr1_1', y='age', label=key, color=colors[key], xlim=[0,60], ylim=[20,38])
    # pyplot.show()
    # # dla kobiet mniejsze znaczenie ma atrakcyjność
    #
    # colors = {0: 'red', 1: 'black'}
    # fig, ax = pyplot.subplots()
    # grouped = df.groupby('gender')
    # for key, group in grouped:
    #     if key % 1 == 0:
    #         group.plot(ax=ax, kind='scatter', x='amb1_1', y='age', label=key, color=colors[key], xlim=[-0.1,20], ylim=[20,38])
    # pyplot.show()
    # # dla mężczyzn mniejsze znaczenie mają ambicje

    x = "age"
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
    fig.suptitle(x, fontsize=20)


    variable = dtf[x].fillna(dtf[x].mean())
    breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    variable = variable[(variable > breaks[0]) & (variable <
                                                  breaks[10])]
    sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax)
    des = dtf[x].describe()
    ax.axvline(des["25%"], ls='--')
    ax.axvline(des["mean"], ls='--')
    ax.axvline(des["75%"], ls='--')
    ax.grid(True)
    des = round(des, 2).apply(lambda x: str(x))
    box = '\n'.join(("min: " + des["min"], "25%: " + des["25%"], "mean: " + des["mean"], "75%: " + des["75%"],
                     "max: " + des["max"]))
    ax.text(0.95, 0.95, box, transform=ax.transAxes, fontsize=10, va='top', ha="right",
               bbox=dict(boxstyle='round', facecolor='white', alpha=1))

    plt.show()


    print(df_without_duplicates['race'].unique())
    print(dtf.groupby('dec_o').size())
    print(dtf.groupby('match').size())
    import seaborn as sns

    sns.countplot(df_without_duplicates['race'], label="Count")
    plt.show()

    import pylab as pl

    df.hist(bins=30, figsize=(9, 9))
    pl.suptitle("Histogram for each numeric input variable")
    plt.savefig('fruits_hist')
    plt.show()

    from matplotlib import cm

    feature_names = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
    X = df[feature_names]
    y = df['match']
    cmap = cm.get_cmap('gnuplot')

    from pandas.plotting import scatter_matrix

    df = pd.DataFrame(np.random.randn(1000, 4), columns=['A', 'B', 'C', 'D'])
    pd.plotting.scatter_matrix(df, alpha=0.2)

    # scatter = pd.plotting.scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)
    # plt.suptitle('Scatter-matrix for each input variable')
    # plt.savefig('fruits_scatter_matrix')

    from sklearn.model_selection import train_test_split

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

    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    pred = knn.predict(X_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches
    import matplotlib.patches as mpatches

    X = df[['mass', 'width', 'height', 'color_score']]
    y = df['fruit_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    X_mat = None
    y_mat = None

    def plot_fruit_knn(X, y, n_neighbors, weights):
        X_mat = X[['height', 'width']].as_matrix()
        y_mat = y.as_matrix()
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF'])


    clf = neighbors.KNeighborsClassifier(n_neighbors=2, weights=weights)
    clf.fit(X_mat, y_mat)
    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.

    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    patch0 = mpatches.Patch(color='#FF0000', label='apple')
    patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
    patch2 = mpatches.Patch(color='#0000FF', label='orange')
    patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
    plt.legend(handles=[patch0, patch1, patch2, patch3])
    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')
    plt.title("4-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
    plt.show()
    plot_fruit_knn(X_train, y_train, 5, 'uniform')