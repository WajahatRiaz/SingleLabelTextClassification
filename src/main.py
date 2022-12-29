from __future__ import unicode_literals, print_function, division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

TOTAL_CLASSES = 4
ITERATIONS = 4 # Number of iterations to split, train and evaluate the data set
classes = ['student', 'course', 'faculty', 'project']
tags_of_classes = [1,2,3,4]

classifiers = (("SVM (Linear)", LinearSVC()),
               ("Decision Tree", DecisionTreeClassifier()),
               ("Random Forest", RandomForestClassifier()))

filenames = ['D:\Again Assignment\dataset\webkb-train-stemmed.txt',
             'D:\Again Assignment\dataset\webkb-test-stemmed.txt']

with open('D:\Again Assignment\dataset\merge-stemmed.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())


file_contents = pd.read_csv("D:\Again Assignment\dataset\merge-stemmed.txt", header=None, sep='\t', 
                       names=['label', 'text'])

print(file_contents)

def count(data_frame):

    for i in range (TOTAL_CLASSES):
        print(f"Number of times {classes[i]} appeared", len(data_frame[data_frame.label==classes[i]]))
    
def tagging (data_frame):
    for i in range(TOTAL_CLASSES):
         data_frame.loc[data_frame["label"]==classes[i],"label"]=tags_of_classes[i]
    
    return data_frame

def to_tfidf(data_frame):


    df_y=data_frame["label"]
    df_x=data_frame["text"]

    tfidf = TfidfVectorizer(min_df=1, stop_words='english')
    df_x = tfidf.fit_transform(df_x.fillna('')).toarray()

    return df_x, df_y
   
def apply_models(model, data_frame, iteration):
    """Returns a list of (model_name, accuracy) that results from 
    applying each model to the train and test dataframes.
    use_lsi indicates if it should be applied to the documents features
    before using the model; default False."""

    tagged_data_frame = tagging(data_frame)

    print(tagged_data_frame)
    x, y = to_tfidf(tagged_data_frame)
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.3)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    list_f1_scores = []

    for (name,model) in classifiers:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        list_f1_scores.insert(i,f1_score(y_test, y_pred, average='micro'))
        print(f"Iteration %d:Classification report for {model}:\n"
        f"{classification_report(y_test, y_pred)}\n" %(iteration+1))
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        disp.figure_.suptitle(f"Iteration %d: Confusion matrix {model}"%(iteration+1))
        print(f"Iteration %d: Confusion matrix for the {model}:\n{disp.confusion_matrix}" %(iteration+1))

        Model_report = classification_report(y_test, y_pred, output_dict=True)
        model_df = pd.DataFrame(Model_report).transpose()
        model_df.to_excel(f"D:\\SingleLabelTextClassification\\docs\\Classifcation Reports\\{model}_%d.xlsx" %(iteration+1)) 
        plt.savefig(f"D:\\SingleLabelTextClassification\\docs\\Confusion Matrices\\{model}_%d.jpg" %(iteration+1))
   
    return np.array(list_f1_scores)

count(file_contents)
f1_score_matrix = np.eye(ITERATIONS, 3)

for i in range(ITERATIONS):
    f1_score_matrix[i,:] = apply_models(classifiers, file_contents, i)
    
macro_f1_scores = [mean(f1_score_matrix[:,i]) for i in range(3)]

print(macro_f1_scores)

def t_test_paired(x1,x2):

    print("Running t-test...\n")
    x1_bar, x2_bar = np.mean(x1), np.mean(x2)
    n1, n2 = len(x1), len(x2)
    var_x1, var_x2= np.var(x1, ddof=1), np.var(x2, ddof=1)
    var = ( ((n1-1)*var_x1) + ((n2-1)*var_x2) ) / (n1+n2-2)
    std_error = np.sqrt(var * (1.0 / n1 + 1.0 / n2))
  
    print("x1:",np.round(x1_bar,4))
    print("x2:",np.round(x2_bar,4))
    print("variance of first sample:",np.round(var_x1))
    print("variance of second sample:",np.round(var_x2,4))
    print("pooled sample variance:",var)
    print("standard error:",std_error)

    # calculate t statistics
    t = abs(x1_bar - x2_bar) / std_error
    print('t static:',t)
    # two-tailed critical value at alpha = 0.05
    t_c = stats.t.ppf(q=0.975, df=17)
    print("Critical value for t two tailed:",t_c)
 
    # one-tailed critical value at alpha = 0.05
    t_c = stats.t.ppf(q=0.95, df=12)
    print("Critical value for t one tailed:",t_c)
 
    # get two-tailed p value
    p_two = 2*(1-stats.t.cdf(x=t, df=12))
    print("p-value for two tailed:",p_two)
  
    # get one-tailed p value
    p_one = 1-stats.t.cdf(x=t, df=12)
    print("p-value for one tailed:",p_one)

t_test_paired(f1_score_matrix[:,0],f1_score_matrix[:,1])
t_test_paired(f1_score_matrix[:,0],f1_score_matrix[:,2])
t_test_paired(f1_score_matrix[:,1],f1_score_matrix[:,2])