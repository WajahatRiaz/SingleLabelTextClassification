# Single Label Text Classification using SCikit-learn

## 1. Introduction
Text Classification techniques are necessary to find relevant information in many different tasks that deal with large quantities of information in text form. Some of the most common tasks where these techniques are applied include: finding answers to similar questions that have been answered before; classifying news by subject or newsgroup; sorting spam from legitimate e-mail messages; finding web pages on a given subject, among others. In each case, the goal is to assign the appropriate class or label to each document that needs to be classified.

Depending on the application, documents can be classified into one or more classes. For instance, a piece of news regarding how the Prime Minister spent his holidays may be classified both as politics and in the social column. In other situations, however, documents can have only one classification, for example, when distinguishing between spam and legitimate e-mail messages. The focus of this work is where each document belongs to a single class, that is, single-label classification.

I will use multiple Machine Learning models and compare how well they perform on single-label text classification tasks using some well known public datasets discussed by Ana Cardoso that are actively used for research. The main goal is to reproduce part of her PhD work using state-of-the-art libraries in python consisting of sklearn, matplotlib, pandas, etc. I consider her work to be successful if I am able to reproduce the initial "related work" from her thesis. I expect results to be approximately the same as her published results.

## 2. WebKB

The documents in the WebKB are webpages collected by the World Wide Knowledge Base (Web->Kb) project of the CMU text learning group, and were downloaded from [The 4 Universities Data Set Homepage](http://www.google.com/url?q=http%3A%2F%2Fwww.cs.cmu.edu%2Fafs%2Fcs.cmu.edu%2Fproject%2Ftheo-20%2Fwww%2Fdata%2F&sa=D&sntz=1&usg=AOvVaw3sn3EgrlvyUavPg9zj8z8K).

These pages were collected from computer science departments of various universities in 1997, manually classified into seven different classes: student, faculty, staff, department, course, project, and other. The class other is a collection of pages that were not deemed the ``main page'' representing an instance of the previous six classes. For example, a particular faculty member may be represented by home page, a publications list, a vitae and several research interests pages. Only the faculty member's home page was placed in the faculty class. The publications list, vitae and research interests pages were all placed in the other category. For each class, the collection contains pages from four universities: Cornell, Texas, Washington, Wisconsin, and other miscellaneous pages collected from other universities. Some classes were discarded by Ana Cardoso such as Department and Staff because there were only a few pages from each university. She also discarded the class Other because pages were very different among this class. Because there is no standard train/test split for this dataset, and in order to be consistent with the previous ones, she  randomly chose two thirds of the documents for training and the remaining third for testing. For this particular split, the distribution of documents per class is the following:


<br><img width="257" alt="image" src="https://user-images.githubusercontent.com/61377755/208338911-90c3fb85-0a7b-4a06-bb1e-6af174bffb99.png">

More details can be found on the link: [Ana Cardoso Cachopo's Homepage](https://ana.cachopo.org/datasets-for-single-label-text-categorization)
