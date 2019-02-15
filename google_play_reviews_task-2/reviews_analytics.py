import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from skopt import BayesSearchCV


def get_data():
    path_to_file = "./googleplaystore_user_reviews.csv"
    data = pd.read_csv(path_to_file, encoding="utf-8")
    return data


def sgd_classifier(X_train_tf, X_test_df, y_train, y_test):
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3,
                        random_state=42, max_iter=5, tol=None)
    clf = clf.fit(X_train_tf, y_train)
    scores = cross_val_score(clf, X_test_df, y_test)
    print("scores for SGDClassifier is {}".format(scores))
    y_pred = clf.predict(X_test_df)
    print(classification_report(y_test, y_pred, target_names=list(set(y_test.values))))


def naive_bayes(X_train_tf, X_test_df, y_train, y_test):
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train_tf, y_train)
    scores = cross_val_score(clf, X_test_df, y_test)
    print("Score for naive bayes is: {}".format(scores))
    y_pred = clf.predict(X_test_df)
    print(classification_report(y_test, y_pred, target_names=list(set(y_test.values))))
    # reviews = ["Very Useful in diabetes age 30. I need control sugar. thanks",
    #            "God health", "HEALTH SHOULD ALWAYS BE TOP PRIORITY. !!. ON MYSG5."]
    # X_new_counts = count_vect.transform(reviews)
    # X_new_tfids = tfid_transformer.transform(X_new_counts)
    # predicted = clf.predict(X_new_tfids)
    # for review, category in zip(reviews, predicted):
    #     print("{} => {}".format(review, category))


def linear_search_one_more_time(X_train_tf, X_test_df, y_train, y_test, data):
    from sklearn.metrics import confusion_matrix
    model = LinearSVC()
    model.fit(X_train_tf, y_train)
    y_pred = model.predict(X_test_df)
    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels=set(data.Sentiment.values),
                yticklabels=set(data.Sentiment.values))
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()


def grid_search_linearSVC(features, X_test_df, labels, y_test):
    from skopt import BayesSearchCV
    print("grid search start")
    opt = BayesSearchCV(
        LinearSVC(),
        {
            "C": (1e-6, 1e+6, "log-uniform")
        },
        n_iter=10
    )
    print("grid search begin")
    opt.fit(features, labels)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test_df, y_test))


def grid_search_SVC(features, X_test_df, labels, y_test):
    print("grid search start")
    opt = BayesSearchCV(
        SVC(),
        {
            "C": (1e-6, 1e+6, "log-uniform"),
            "gamma": (1e-6, 1e+1, "log-uniform"),
            "degree": (1, 3),
            "kernel": ["linear", "poly", "rbf"],
        },
        n_iter=30,
        random_state=1234
    )
    print("grid search begin")
    opt.fit(features, labels)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test_df, y_test))
    print("best_estimators: %s" % opt.best_params_)


bayes_cv_tuner = BayesSearchCV(
    estimator=xgb.XGBClassifier(
        n_jobs=1,
        objective='binary:logistic',
        eval_metric='auc',
        silent=True,
        tree_method='approx'
    ),
    search_spaces={
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },
    scoring='roc_auc',
    cv=StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs=3,
    n_iter=10,
    verbose=0,
    refit=True,
    random_state=42
)


def make_xgboost():
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.best_params_)

    # Get current parameters and the best parameters
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))


def make_plot_with_models(features, X_test_df, labels, y_test):
    models = [LinearSVC(),
              SVC(kernel="linear")]
    # RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    # MultinomialNB, LogisticRegression(random_state=0),
    CV = 2
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=["model_name", "fold_idx", "accuracy"])
    sns.boxplot(x="model_name", y="accuracy", data=cv_df)
    sns.stripplot(x="model_name", y="accuracy", data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    print(cv_df)
    print(cv_df.groupby("model_name").accuracy.mean())


def analyse_correlation(category_to_id, features, labels, tfidf):
    N = 2
    for Product, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(Product))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


def data_transformation(data):
    stat_dict = dict(data.values)
    feature = data["Translated_Review"]
    estimate = data["Sentiment"]
    # в начале трансформируем, потом сплитим.
    X_train, X_test, y_train, y_test = train_test_split(feature, estimate, train_size=0.8, random_state=0)
    """
    sublinear_df is set to True to use a logarithmic form for frequency.
    
    min_df is the minimum numbers of documents a word must be present in to be kept.
    
    norm is set to l2, to ensure all our feature vectors have a euclidian norm of 1.
    
    ngram_range is set to (1, 2) to indicate that we want to consider both unigrams and bigrams.
    
    stop_words is set to "english" to remove all 
    common pronouns ("a", "the", ...) to reduce the number of noisy features.
    """
    # tfidf - Term Frequency and Inverse Document Frequency
    tfid_vector = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.8,
                                  norm="l2", ngram_range=(1, 2), stop_words="english")
    X_train_tf, X_test_tf = tfid_vector.fit_transform(X_train), tfid_vector.transform(X_test)
    chi2score = chi2(X_train_tf, y_train)[0]
    plt.figure(figsize=(15, 10))
    wscores = zip(tfid_vector.get_feature_names(), chi2score)
    wchi2 = sorted(wscores, key=lambda x: x[1])
    topchi2 = list(zip(*wchi2[-20:]))
    x = range(len(topchi2[1]))
    labels = topchi2[0]
    plt.barh(x, topchi2[1], align="center", alpha=0.2)
    plt.plot(topchi2[1], x, "-o", markersize=5, alpha=0.8)
    plt.yticks(x, labels)
    plt.xlabel("$\chi^2$")
    # plt.show()
    return stat_dict, X_train_tf, X_test_tf, y_train, y_test


def reduce_dimensionality(X_train, X_test, y_train, y_test):
    # ch2_result = []
    # for n in np.arange(100, 9000, 100):
    #     ch2 = SelectKBest(chi2, k=n)
    #     x_train_chi2_selected = ch2.fit_transform(X_train, y_train)
    #     x_validation_ch2_selected = ch2.transform(X_test)
    #     clf = LinearSVC()
    #     clf.fit(x_train_chi2_selected, y_train)
    #     score = clf.score(x_validation_ch2_selected, y_test)
    #     ch2_result.append((score, n))
    #     print("chi2 feature selection evaluation calculated for {} features".format(n))
    #  ch2_result.sort(key=lambda x: x[0], reverse=True)
    #  print(ch2_result)
    # first set PCA to retain 95% of variety
    pca = TruncatedSVD(1500)
    pca.fit(X_train)
    X_train_k = pca.transform(X_train)
    X_test_k = pca.transform(X_test)
    return X_train_k, X_test_k, y_train, y_test
    # for k in range(100, 9000, 100):
    #     pca = TruncatedSVD(k)
    #     pca.fit(X_train)
    #     X_train_k = pca.transform(X_train)
    #     X_test_k = pca.transform(X_test)
    #     clf = LinearSVC()
    #     clf.fit(X_train_k, y_train)
    #     score = clf.score(X_test_k, y_test)
    #     print("Score for k={} and LinearSVC {}".format(k, score))
    # Best score has 1500 and 2000 features, so let's take 1500


def data_maker_without_nan(data):
    # I don't think that nan's will affect on final solution, but half of all data is nan.
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = data[["Translated_Review", "Sentiment"]]
    data = data[["Translated_Review", "Sentiment"]].drop_duplicates()
    stat_dict, X_train, X_test, y_train, y_test = data_transformation(data)
    # X_train, X_test, y_train, y_test = reduce_dimensionality(X_train, X_test, y_train, y_test)
    # result = bayes_cv_tuner.fit(X_train, y_train, callback=make_xgboost)
    # grid_search_SVC(X_train, X_test, y_train, y_test)
    # grid_search_linearSVC(X_train, X_test, y_train, y_test)
    make_plot_with_models(X_train, X_test, y_train, y_test)
    # naive_bayes(X_train_tf, X_test_df, y_train, y_test)
    # sgd_classifier(X_train_tf, X_test_df, y_train, y_test)


def main():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    data = get_data()
    data_maker_without_nan(data)


if __name__ == "__main__":
    # TODO make grid search for c-parameter linear svm model
    # Done - best score: 86,6%
    # TODO make grid search for poly(rbf) gamma parameter
    # In work: Need to reduce dimensionality to work with that function
    # comment for me
    # Robust, Nomaliser - глянуть
    # 1) Привести в порядок код.
    # 2) Прогнать данные через робуст, нормалайзер, ... (после тфидфтрансформера без редьюса дименшионов) -
    # для ядерных (нот линеар кернел).
    # 3) Доразобраться с сайкит-оптимайзером, иксджи-бустом.
    # 4) Посмотреть басс: Stemming, Lemmatization - мб для этого не нужны стоп-слова.
    # 5) Взять в аренду google cloud
    main()
