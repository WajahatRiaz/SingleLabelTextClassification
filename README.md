# Single Label Text Classification 


## WebKB

The documents in the WebKB are webpages collected by the World Wide Knowledge Base (Web->Kb) project of the CMU text learning group, and were downloaded from The 4 Universities Data Set Homepage. These pages were collected from computer science departments of various universities in 1997, manually classified into seven different classes: student, faculty, staff, department, course, project, and other. The class other is a collection of pages that were not deemed the ``main page'' representing an instance of the previous six classes. For example, a particular faculty member may be represented by home page, a publications list, a vitae and several research interests pages. Only the faculty member's home page was placed in the faculty class. The publications list, vitae and research interests pages were all placed in the other category. For each class, the collection contains pages from four universities: Cornell, Texas, Washington, Wisconsin, and other miscellaneous pages collected from other universities. Some classes were discarded by Ana Cardoso such as Department and Staff because there were only a few pages from each university. She also discarded the class Other because pages were very different among this class. Because there is no standard train/test split for this dataset, and in order to be consistent with the previous ones, she  randomly chose two thirds of the documents for training and the remaining third for testing. For this particular split, the distribution of documents per class is the following:



More details can be found on the link: [Homepage](https://ana.cachopo.org/datasets-for-single-label-text-categorization)
