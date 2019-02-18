import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from skopt import BayesSearchCV

from Natural_language import LemmaTokenizer


def get_data():
    path_to_file = "./googleplaystore_user_reviews.csv"
    data = pd.read_csv(path_to_file, encoding="utf-8")
    return data


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


def grid_search_SVC(features, X_test, labels, y_test):
    print("grid search start")
    ch2 = SelectPercentile(percentile=5)
    x_train_chi2 = ch2.fit_transform(features, labels)
    x_valid = ch2.transform(X_test)
    C_range = 10.**np.arange(-3, 8)
    gamma_range = 10.**np.arange(-5, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(), param_grid=param_grid, n_jobs=-1, cv=5)
    grid.fit(x_train_chi2, labels)
    print("The best classifier is: ", grid.best_estimator_)


bayes_cv_tuner = BayesSearchCV(
    estimator=SVC(),
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
              SVC(kernel="linear"),
              RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
              MultinomialNB(),
              LogisticRegression(random_state=0)]
    CV = 5
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


def data_transformation(data):
    stat_dict = dict(data.values)
    feature = data["Translated_Review"]
    estimate = data["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(feature, estimate, train_size=0.8, random_state=0)
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


def linear_svc(X_train, X_test, y_train, y_test):
    # LinearSVC - 91,9%; SVC - 92% - using SelectPercentile(percentile=5)
    ch2 = SelectPercentile(percentile=5)
    x_train_chi2 = ch2.fit_transform(X_train, y_train)
    x_valid = ch2.transform(X_test)
    models = [LinearSVC(),
              SVC(C=0.01, kernel="linear"),
              SVC(C=0.001, kernel="linear"),
              SVC(C=10, kernel="linear"),
              SVC(C=100, kernel="linear")
              ]
    # RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    # MultinomialNB(),
    # LogisticRegression(random_state=0)
    for model in models:
        model.fit(x_train_chi2, y_train)
        print("Score for {} and SelectKBest: {}".format(model.__class__.__name__, model.score(x_valid, y_test)))
    # best score has SVC with kernel = linear: 91,5 %


def reduce_dimensionality(X_train, X_test, y_train, y_test):
    ch2_result = []
    for n in np.arange(100, 9000, 100):
        ch2 = SelectKBest(chi2, k=n)
        x_train_chi2_selected = ch2.fit_transform(X_train, y_train)
        x_validation_ch2_selected = ch2.transform(X_test)
        clf = LinearSVC()
        clf.fit(x_train_chi2_selected, y_train)
        score = clf.score(x_validation_ch2_selected, y_test)
        ch2_result.append((score, n))
        print(score, n)
        print("chi2 feature selection evaluation calculated for {} features".format(n))
    ch2_result.sort(key=lambda x: x[0], reverse=True)
    print(ch2_result)
    # first set PCA to retain 95% of variety
    # pca = TruncatedSVD(1500)
    # pca.fit(X_train)
    # X_train_k = pca.transform(X_train)
    # X_test_k = pca.transform(X_test)
    # return X_train_k, X_test_k, y_train, y_test
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


def data_lemmatization(data):
    # using LemmaTokenizer (WordNetLemmatizer) ans SelectKBest gives 91,4% (91,6%)
    feature = data["Translated_Review"]
    estimate = data["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(feature, estimate, train_size=0.8, random_state=0)
    vect = CountVectorizer(tokenizer=LemmaTokenizer())
    X_train, X_test = vect.fit_transform(X_train), vect.transform(X_test)
    return X_train, X_test, y_train, y_test


def data_maker_without_nan(data):
    # I don't think that nan's will affect on final solution, but half of all data is nan.
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = data[["Translated_Review", "Sentiment"]]
    data = data[["Translated_Review", "Sentiment"]].drop_duplicates()
    X_train, X_test, y_train, y_test = data_lemmatization(data)
    # stat_dict, X_train, X_test, y_train, y_test = data_transformation(data)
    # reduce_dimensionality(X_train, X_test, y_train, y_test)
    linear_svc(X_train, X_test, y_train, y_test)
    # result = bayes_cv_tuner.fit(X_train, y_train, callback=make_xgboost)
    # grid_search_SVC(X_train, X_test, y_train, y_test)
    # grid_search_linearSVC(X_train, X_test, y_train, y_test)
    # make_plot_with_models(X_train, X_test, y_train, y_test)
    # naive_bayes(X_train_tf, X_test_df, y_train, y_test)
    # sgd_classifier(X_train_tf, X_test_df, y_train, y_test)


def main():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    data = get_data()
    data_maker_without_nan(data)


if __name__ == "__main__":
    # comments for me
    """
    1) tfidftransformer не неужно проганять через скаллеры - это ухдшает точность получаемой модели
    2) Пока что лучший скаллер - CountVectorizer, c кастомным токенайзером
    3) Кастомный токенайзер WordNetLemmatizer
    4) scikit-optimizer - разобрался, но почему то выше 80% модель не удалось получить
    5) Градиент бустинг - смотри пункт 4
    6) Лучший на данный момент результат - SVC: 92% LinearSVC: 91,9%
    7) 
    """
    main()
