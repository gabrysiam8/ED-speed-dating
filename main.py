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
import seaborn as sns
from numpy import loadtxt
import matplotlib.cm as cm

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
    print(pca.components_)
    pca_df = pd.DataFrame(data=principal_components, index=original_df.index)
    pca_df.rename({i: "PC{}".format(i) for i in range(n_components)}, axis=1, inplace=True)
    return pca_df


def run_clustering(original_df):
    # data normalization
    normalized_vectors = preprocessing.normalize(partial_df)

    kmeans = []
    normalized_kmeans = []
    silhouette = []
    normalized_silhouette = []

    for i in range(2, 10):
        kmeans_i = KMeans(n_clusters=i).fit(partial_df)
        normalized_kmeans_i = KMeans(n_clusters=i).fit(normalized_vectors)

        silhouette_i = silhouette_score(partial_df, kmeans_i.labels_, metric='euclidean')
        silhouette_norm_i = silhouette_score(normalized_vectors, normalized_kmeans_i.labels_, metric='cosine')

        kmeans.append(kmeans_i)
        normalized_kmeans.append(normalized_kmeans_i)
        silhouette.append(silhouette_i)
        normalized_silhouette.append(silhouette_norm_i)

        # print results
        print('{} clusters'.format(i))
        print('kmeans: {}'.format(silhouette_i))
        print('Cosine kmeans:{}'.format(silhouette_norm_i))

    plt.title('Silhouette score')
    sns.lineplot(x=range(2, 10), y=normalized_silhouette)
    plt.show()

    max_value = max(normalized_silhouette)
    max_index = normalized_silhouette.index(max_value)
    print('Best cluster number: {}'.format(max_index + 2))
    return kmeans[max_index], normalized_kmeans[max_index]


def plot_kmeans_clusters(x_column, y_column, df, clusters_data):
    colors = cm.nipy_spectral(clusters_data.labels_.astype(float) / len(clusters_data.cluster_centers_))

    x = df[x_column]
    y = df[y_column]

    plt.scatter(x, y, marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusters_data.cluster_centers_
    # Draw white circles at cluster centers
    plt.scatter(centers[:, df.columns.get_loc(x_column)],
                centers[:, df.columns.get_loc(y_column)],
                marker='o', c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        plt.scatter(c[df.columns.get_loc(x_column)],
                    c[df.columns.get_loc(y_column)],
                    marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

    plt.xlabel(x_column)
    plt.ylabel(y_column)

    plt.show()

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


if __name__ == '__main__':
    df = pd.read_csv('Speed Dating Data.csv', encoding="ISO-8859-1")

    column_names = ['iid', 'pid', 'gender', 'race', 'age', 'field_cd', 'career_c', 'int_corr', 'attr1_1', 'sinc1_1',
                    'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'match', 'age_o', 'samerace']
    df = df[column_names]

    # for col in column_names[2:]:
    #     sns.displot(df, x=col)
    #     plt.show()

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
    race_stat = race_stat.set_index('value')
    print(race_stat)
    race_stat.plot.pie(autopct="%.1f%%", y='count')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.show()

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

    # # remove rows with nan race value
    # df = df[df['race'].notna()]
    #
    # # outliers detection
    # df = detect_outliers(df, ['int_corr', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1'])
    #
    # # replace missing values with mean
    # df = replace_missing_values(df)
    #
    # x = "age"
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
    # fig.suptitle(x, fontsize=20)
    #
    # variable = df[x].fillna(df[x].mean())
    # breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    # variable = variable[(variable > breaks[0]) & (variable <
    #                                               breaks[10])]
    # sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax)
    # des = df[x].describe()
    # ax.axvline(des["25%"], ls='--')
    # ax.axvline(des["mean"], ls='--')
    # ax.axvline(des["75%"], ls='--')
    # ax.grid(True)
    # des = round(des, 2).apply(lambda x: str(x))
    # box = '\n'.join(("min: " + des["min"], "25%: " + des["25%"], "mean: " + des["mean"], "75%: " + des["75%"],
    #                  "max: " + des["max"]))
    # ax.text(0.95, 0.95, box, transform=ax.transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    #
    # plt.show()
    #
    #
    # # clustering
    # cols1, cols2 = loadtxt("args.txt", dtype=str, comments="#", delimiter=",", unpack=False)
    #
    # cols1 = [x for x in cols1 if x]
    # print(cols1)
    # partial_df = df.loc[:, cols1]
    # # dendrogram = dendrogram(linkage(partial_df, method='ward'))
    # # plt.show()
    # kmeans_model, norm_kmeans_model = run_clustering(partial_df)
    #
    # # Principal Component Analysis
    # pca_df = run_pca(2, partial_df)
    # pca_df['labels'] = kmeans_model.labels_
    # plt.title('kmeans')
    # sns.scatterplot(x=pca_df.PC0, y=pca_df.PC1, hue=pca_df.labels, palette="Set2")
    # plt.show()
    #
    # norm_pca_df = pca_df.copy()
    # norm_pca_df['labels'] = norm_kmeans_model.labels_
    # plt.title('cosine kmeans')
    # sns.scatterplot(x=norm_pca_df.PC0, y=norm_pca_df.PC1, hue=norm_pca_df.labels, palette="Set2")
    # plt.show()
    #
    # partial_df['labels'] = norm_kmeans_model.labels_
    #
    # draw_cluster_barplot(partial_df, norm_kmeans_model)
    #
    # # atrakcyjność, inteligencja
    # sns.scatterplot(x=partial_df.attr1_1, y=partial_df.intel1_1, hue=pca_df.labels, palette="Set2")
    # plt.show()
    #
    # # atracyjność, wspólne zainteresowania
    # sns.scatterplot(x=partial_df.attr1_1, y=partial_df.shar1_1, hue=pca_df.labels, palette="Set2")
    # plt.show()
    #
    # # atracyjność, ambicja
    # sns.scatterplot(x=partial_df.attr1_1, y=partial_df.amb1_1, hue=pca_df.labels, palette="Set2")
    # plt.show()
    #
    # # atrakcyjność, szczerość
    # sns.scatterplot(x=partial_df.attr1_1, y=partial_df.sinc1_1, hue=pca_df.labels, palette="Set2")
    # plt.show()
    #
    # # Make the plot
    # plt.figure(figsize=(15, 10))
    # parallel_coordinates(partial_df, 'labels', colormap=plt.get_cmap("Set1"))
    # plt.xlabel("Features of data set")
    # plt.ylabel("Importance")
    # plt.show()
    #
    # # female_df = df[df['gender'] == 0].copy()
    # # male_df = df[df['gender'] == 1].copy()
    # # merged_df = merge_person_partner_data(female_df, male_df)
    # #
    # # print(merged_df)
    #
    # colors = {0: 'red', 1: 'black'}
    # fig, ax = plt.subplots()
    # grouped = df.groupby('gender')
    # for key, group in grouped:
    #     if key % 1 == 0:
    #         group.plot(ax=ax, kind='scatter', x='attr1_1', y='age', label=key, color=colors[key], xlim=[0,60], ylim=[20,38])
    # plt.show()
    # # dla kobiet mniejsze znaczenie ma atrakcyjność
    #
    # colors = {0: 'red', 1: 'black'}
    # fig, ax = plt.subplots()
    # grouped = df.groupby('gender')
    # for key, group in grouped:
    #     if key % 1 == 0:
    #         group.plot(ax=ax, kind='scatter', x='amb1_1', y='age', label=key, color=colors[key], xlim=[-0.1,20], ylim=[20,38])
    # plt.show()
    # # dla mężczyzn mniejsze znaczenie mają ambicje
    #
    # column_names = ['attr1_1', 'intel1_1', 'amb1_1', 'sinc1_1', 'fun1_1', 'gender']
    # df_without_duplicates = df_without_duplicates[column_names]
    # df_without_duplicates["amb1_1"] = df_without_duplicates["amb1_1"].fillna(df_without_duplicates["amb1_1"].mean())
    # df_without_duplicates["intel1_1"] = df_without_duplicates["intel1_1"].fillna(df_without_duplicates["intel1_1"].mean())
    # df_without_duplicates["attr1_1"] = df_without_duplicates["attr1_1"].fillna(df_without_duplicates["attr1_1"].mean())
    # df_without_duplicates["fun1_1"] = df_without_duplicates["fun1_1"].fillna(df_without_duplicates["fun1_1"].mean())
    # df_without_duplicates["sinc1_1"] = df_without_duplicates["sinc1_1"].fillna(df_without_duplicates["sinc1_1"].mean())
    # #df_without_duplicates["age"] = df_without_duplicates["age"].fillna(df_without_duplicates["age"].mean())
    # df = df_without_duplicates
    #
    # print(df['gender'].unique())
    # print(df['sinc1_1'].unique())
    #
    # from sklearn.preprocessing import MinMaxScaler
    # from sklearn.model_selection import train_test_split
    #
    # df_new = df[column_names]
    # df_new_form = pd.DataFrame(MinMaxScaler().fit_transform(df_new))
    # df_new_form.columns = df_new.columns
    # train, test = train_test_split(df_new_form, test_size=0.5, random_state=0)
    #
    # from sklearn.cluster import KMeans
    # from sklearn.neighbors import KNeighborsClassifier
    #
    # kmeans = KMeans(
    #     init="random",
    #     n_clusters=3,
    #     n_init=10,
    #     max_iter=300,
    #     random_state=43,
    # )
    #
    # cluster_data = kmeans.fit(df_new_form)
    # plot_kmeans_clusters("gender", "sinc1_1", df_new_form, cluster_data)
    #
    # neigh = KNeighborsClassifier(n_neighbors=3)
    # neigh.fit(train, cluster_data.labels_[train.index])
    # neigh.predict(test)
    #
    # print(neigh.score(test, cluster_data.labels_[test.index]))
    #
    # from sklearn import tree
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(train, cluster_data.labels_[train.index])
    # print(tree.plot_tree(clf, filled=True))


    from pandas.plotting import scatter_matrix
    from matplotlib import cm

    df = pd.read_csv('Speed Dating Data.csv', encoding="ISO-8859-1")
    feature_names = ['attr1_1', 'intel1_1', 'amb1_1', 'sinc1_1', 'fun1_1', 'gender', 'match']

    df["amb1_1"] = df["amb1_1"].fillna(df["amb1_1"].mean())
    df["intel1_1"] = df["intel1_1"].fillna(df["intel1_1"].mean())
    df["attr1_1"] = df["attr1_1"].fillna(df["attr1_1"].mean())
    df["fun1_1"] = df["fun1_1"].fillna(df["fun1_1"].mean())
    df["sinc1_1"] = df["sinc1_1"].fillna(df["sinc1_1"].mean())

    X = df[feature_names]
    y = df['match']
    cmap = cm.get_cmap('gnuplot')
    scatter = scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
    plt.suptitle('Scatter-matrix for each input variable')
    plt.savefig('fruits_scatter_matrix')

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knn.score(X_train, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
          .format(knn.score(X_test, y_test)))

    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches
    import matplotlib.patches as mpatches

    X = df[['attr1_1', 'intel1_1', 'amb1_1', 'sinc1_1', 'fun1_1', 'gender', 'match']]
    y = df['match']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


    from sklearn import neighbors

    def plot_fruit_knn(X, y, n_neighbors, weights):
        X_mat = X[['attr1_1', 'gender']].as_matrix()
        y_mat = y.as_matrix()
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF'])

        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
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
    plt.title("4-Class classification (k = %i, weights = '%s')" % (1, "weights"))
    plt.show()
    plot_fruit_knn(X_train, y_train, 5, 'uniform')

    k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0, 5, 10, 15, 20])

