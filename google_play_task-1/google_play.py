import hashlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def get_data():
    path_to_file = "./googleplaystore.csv"
    data = pd.read_csv(path_to_file, encoding='utf-8')
    return data


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    # first approach was to separate data using hashes. Data is stable and there is no need in separating
    # data with this function. Instead of this one I used sklearn.model_selection.train_test_split.
    # May be for future project i will use this function, so i leave it here
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def make_randomforest_report(d):
    array = list(d.items())
    array.sort(key=lambda x: (x[1]['precision'], x[1]['recall']))
    total = 0
    # worst predictions of random forest classifier are for: entertainment(40%), education(65%),
    # family(84%), game(87%)
    for item in array:
        if item[0] == 'micro avg' or item[0] == 'weighted avg' or item[0] == 'macro avg':
            continue
        else:
            total += item[1]['support']
            print(item)
    print("total: ", total)


def make_a_plot():
    pass


def random_forest_classifier_for_all_data(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    random_forest = RandomForestClassifier(max_features=0.2, n_estimators=176, random_state=0, class_weight="balanced")
    random_forest.fit(X_train, y_train)
    print("All data result")
    print("Random forest classifier score %f" % random_forest.score(X_test, y_test))
    y_pred = random_forest.predict(X_test)
    d = classification_report(y_test, y_pred, output_dict=True)
    make_randomforest_report(d)
    print(random_forest.feature_importances_)


def random_forest_classifier(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    random_forest = RandomForestClassifier(max_features=0.2, n_estimators=100, random_state=0, oob_score=True,
                                           class_weight="balanced")
    random_forest.fit(X_train, y_train)
    print("Random forest classifier score %f" % random_forest.score(X_test, y_test))
    y_pred = random_forest.predict(X_test)
    d = classification_report(y_test, y_pred, output_dict=True)
    make_randomforest_report(d)
    print(random_forest.feature_importances_)


def logistic_regression_for_all_data(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(class_weight="balanced", multi_class="multinomial", solver="lbfgs")
    model.fit(X_train, y_train)
    print("Logistic regression score {}".format(model.score(X_test, y_test)))
    make_a_plot()


def logistic_regression(X_train, X_test, y_train, y_test):
    # почитать про разбиение если больше двух классов!
    # читать про параметры
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1.0, intercept_scaling=1,
                               dual=False, fit_intercept=True, penalty='l1', tol=0.0001, class_weight="balanced")
    model.fit(X_train, y_train)
    print("Logistic regression score {}".format(model.score(X_test, y_test)))
    make_a_plot()
    # y_pred = model.predict(X_test)


def kernel_svm(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)
    print("Kernel svm score is {}".format(model.score(X_test, y_test)))
    from sklearn.svm import LinearSVC
    model = LinearSVC()
    model.fit(X_train, y_train)
    print("Linear kernel score is {}".format(model.score(X_test, y_test)))
    # y_pred = model.predict(X_test)


def cross_val_score_implementation(X_train, X_test, y_train, y_test):
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=0.1).fit(X_train, y_train)
    scores = cross_val_score(clf, X_test, y_test, cv=10)
    print(scores)


# def make_all_classes_equal(data):
#     stat = {}
#     for item in data["Category"]:
#         carry = stat.get(item, 0) + 1
#         stat[item] = carry
#     min_value = min(stat.values())
#     columns = {}
#     for key in stat:
#         columns[key] = min_value
#     final_data = pd.DataFrame()
#     for item in range(len(data)):
#         carry = data.iloc[item]["Category"]
#         if columns.get(carry) > 0:
#             final_data = final_data.append(data.iloc[item], ignore_index=True)
#             columns[carry] -= 1
#         else:
#             continue
#     return final_data


def models_without_nan_and_result(all_data):
    all_data = all_data[all_data.Category != "1.9"]
    # all_data = make_all_classes_equal(all_data)
    subset = ["Type", "Content Rating", "Current Ver", "Android Ver"]
    all_data = all_data.drop(subset, axis=1)
    all_data = all_data.drop(["Rating"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(all_data.drop(["Category"], axis=1),
                                                        all_data["Category"], train_size=0.67, random_state=0)
    colT = ColumnTransformer(
        [("dummy_col", OneHotEncoder(handle_unknown='ignore'), ['Reviews', 'Size', 'Installs', 'Price',
                                                                'Genres', 'Last Updated'])])
    X_train = colT.fit_transform(X_train)
    X_test = colT.transform(X_test)
    print('scores for data without columns with missing values and rating column')
    # cross_val_score_implementation(X_train, X_test, y_train, y_test)
    logistic_regression(X_train, X_test, y_train, y_test)
    random_forest_classifier(X_train, X_test, y_train, y_test)
    # kernel_svm(X_train, X_test, y_train, y_test)


def models_with_nan_and_int_values(all_data):
    all_data.Rating = all_data.Rating.astype(float).fillna(0.0)
    all_data = all_data.fillna('')
    all_data = all_data[all_data.Category != "1.9"]
    # all_data = make_all_classes_equal(all_data)
    X_train, X_test, y_train, y_test = train_test_split(all_data.drop(["Category"], axis=1),
                                                        all_data["Category"], train_size=0.67, random_state=0)
    colT = ColumnTransformer(
        [("dummy_col", OneHotEncoder(handle_unknown='ignore'), ['Reviews', 'Size', 'Installs', 'Type', 'Price',
                                                                'Content Rating', 'Genres', 'Last Updated',
                                                                'Current Ver', 'Android Ver']),
         ("nummy_col", RobustScaler(), ["Rating"])])
    X_train = colT.fit_transform(X_train)
    X_test = colT.transform(X_test)
    print("scores for all data (including all missing values and rating column)")
    # cross_val_score_implementation(X_train, X_test, y_train, y_test)
    logistic_regression_for_all_data(X_train, X_test, y_train, y_test)
    random_forest_classifier_for_all_data(X_train, X_test, y_train, y_test)
    # kernel_svm(X_train, X_test, y_train, y_test)


def main():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    all_data = get_data()
    # cross_val_score_implementation(all_data)
    models_without_nan_and_result(all_data)
    models_with_nan_and_int_values(all_data)


if __name__ == "__main__":
    main()

