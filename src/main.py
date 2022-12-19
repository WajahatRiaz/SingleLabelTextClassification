

from __future__ import unicode_literals, print_function, division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import pandas as pd
import matplotlib.pyplot as plt

def read_file(filename):
    """Reads one csv file without header and two columns separated by a TAB"""
    return pd.read_csv("D:/AssignmentML/dataset/" + filename, 
                       header=None, sep='\t', 
                       names=['label', 'text'])

webKB_train_df = read_file('webkb-train-stemmed.txt')
webKB_test_df = read_file('webkb-test-stemmed.txt')

all_dfs = [webKB_train_df, webKB_test_df]
all_df_names = ["webKB_train_df", "webKB_test_df"]

# check that all files were read by printing the shape
# of the resulting dataframe and the first document
for (df, name) in zip(all_dfs, all_df_names):
    print(name, "(#rows, #columns):", df.shape)
    print(df.head(1))


def create_counts_df(df_train, df_test):
    """Receives two dataframes with train and test documents
    and returns a dataframe with the number of train, 
    test and total documents per class.
    Assumes both dataframes have the same classes."""
    counts_df = pd.concat([df_train["label"].value_counts(), 
                           df_test["label"].value_counts()], 
                          axis=1, keys=["# train docs", "# test docs"])
    counts_df["total # docs"] = counts_df.sum(axis=1)
    counts_df.loc["Total"] = counts_df.sum()
    return counts_df

webKB_counts = create_counts_df(webKB_train_df, webKB_test_df)
print(webKB_counts)

# do not plot totals (last column, last row in counts dataframes)
webKB_counts.iloc[:-1,:-1].plot.barh(stacked=True, figsize=(5, 5), title="Number of train and test documents per class for wbeKB")
plt.show()


def describe_dataset(name, df_train):
    print("Dataset:", name)

    # consider all words
    cvec = CountVectorizer()
    cvec.fit(df_train["text"])
    with_stop = len(cvec.get_feature_names_out())
    print("Number of features including stopwords:", with_stop)

    # remove english stopwords
    cvec = CountVectorizer(stop_words="english")
    cvec.fit(df_train["text"])
    without_stop = len(cvec.get_feature_names_out())
    print("Number of features excluding stopwords:", without_stop)
    
    print("Difference in number of features:", with_stop-without_stop)
    return with_stop, without_stop

webKB_with_stop, webKB_without_stop = describe_dataset("webKB", webKB_train_df.fillna(''))

features_df = pd.DataFrame({"webKB": [webKB_with_stop]},index=["# features \n (including stopwords)"])
features_df[["webKB"]].plot.barh(title="Number of features per dataset")
plt.show()

train_docs_df = pd.DataFrame({"webKB": webKB_counts.loc["Total"]["# train docs"]}, index=["# train docs"])
train_docs_df[["webKB"]].plot.barh(title="Number of train documents per dataset")
plt.show()

def to_tfidf(df_train, df_test):
    """Receives the train and test dataframes (columns: label, text) and 
    returns the X_train, y_train, X_test, y_test necessary to fit the models.
    The X_train and X_test feature matrices contain tfidf values."""
    vec = TfidfVectorizer(stop_words='english', smooth_idf=False)
    vec.fit(df_train["text"])
    X_train = vec.transform(df_train["text"])
    y_train = df_train["label"]
    X_test = vec.transform(df_test["text"])
    y_test = df_test["label"]
    return X_train, y_train, X_test, y_test
    
def apply_model(model, X_train, y_train, X_test, y_test):
    """Returns the Accuracy score of fitting the model with
    the train data and predicting for the test data."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print(f"Classification report for {model}:\n"
    f"{classification_report(y_test, y_pred)}\n")
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.figure_.suptitle(f"Confusion matrix {model}")
    print(f"Confusion matrix for the {model}:\n{disp.confusion_matrix}")

    Model_report = classification_report(y_test, y_pred, output_dict=True)
    model_df = pd.DataFrame(Model_report).transpose()
    model_df.to_excel(f"D:\\SingleLabelTextClassification\\docs\\Classifcation Reports\\{model}.xlsx") 
    plt.savefig(f"D:\\SingleLabelTextClassification\\docs\\Confusion Matrices\\{model}.jpg")
    plt.show()

    return score

def apply_models(models, df_train, df_test):
    """Returns a list of (model_name, accuracy) that results from 
    applying each model to the train and test dataframes.
    use_lsi indicates if it should be applied to the documents features
    before using the model; default False."""
    X_train, y_train, X_test, y_test = to_tfidf(df_train, df_test)

    return [(name, apply_model(model, X_train, y_train, X_test, y_test)) \
            for (name, model) in models]

webpage_models = (("Naive Bayes", MultinomialNB()),
                  ("SVM (Linear)", LinearSVC()),
                  ("Decision Tree", DecisionTreeClassifier()),
                  ("Random Forest", RandomForestClassifier()),
                 )

webKB_results = apply_models(webpage_models, webKB_train_df.fillna(''), webKB_test_df.fillna(''))

webpage_results_df = pd.DataFrame({"webKB": [result[1] for result in webKB_results]},index=[model[0] for model in webpage_models])
print(webpage_results_df[["webKB"]])