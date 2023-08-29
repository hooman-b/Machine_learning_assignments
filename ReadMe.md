# Machine Learning Educational Notebook

## Program application
This repository is supposed to be used as a complementary document for Introductory machine learning courses.
 
## Program parts
This repository is based on the section and course materials of the Hanze University of applied Sciences machine learning course. Consequently, it contains three sections. The first section foceses on the foundation of unsupervised machine learning, The second part is centered around supervised learning, and the third part covers a smal part of deep learning methods. Each of the mentioned sections contains an educational notebook, and a notebook that tries to answer a research question related to the material of the educational notebook, which I call them research adventure. 

### Unsupervised Learning
Unspervised Learning folder contains three distinct assignments. the first folder (First_Assignment) explains the fundamentals of machine learning, and contains two tutorials related to unsupervised clustering methods. The second folder (Second_Assignment) foceses on word clustering and contains a comprehensive tutorial and a research adventure. The third folder (Third_Assignment) is centered around anomally detection consists of a tuterial and a research adventure.

#### **First_Assignment**:
This folder contains Three 'ipynb' notebooks. 'tutorial_cluster_scanpy_object' is a turorial about 'gene clustering' using Scanpy package, 'tutorial_Clustering_Methods' contain a tutorial about many important unsupervised clustering methods such as K-means, DBSCAN, and Spectral Clustering. 'Assignment1' notebook describes some of the most important and fundamental concepts of machine learning. The dataset used in this section is:

**wine data**:

Name: Wine Quality Data Set (Red & White Wine)
From: https://www.kaggle.com
link: [link][https://www.kaggle.com/datasets/ruthgn/wine-quality-data-set-red-white-wine]
Downloading tip: one can easily download the dataset from kaggle website and extract it into a proper folder.

#### **Second_Assignment**:
This folder contains two 'ipynb' notebooks. 'tutorial_clustering_words' explains non-negative matrix factorization (NMF) method, and many other important concepts such as Document-Term matrix (DTM), and a set of visualising and analysing methods used for this specific topic. This notebook has in-practice approach to teach all the important concepts by using data science pipeline, starting from describing the dataset, inspect and preprocess the dataset, implement the clustering method, and finally using analitical and visualizing techniques. this notebook uses a  collection of 200 clinical case report documents in plain text format as its dataset.

'Assignment2' notebook contains the research adventure of this section. it uses the same approach as the first notebook, but by using Quantum Physics articles on Arxiv 1994 to 2009 dataset. The aim of this research is to find the most popular quantum physics topics (hot topics) in the last 30 years based on the number of published article.

**Clinical dataset**:

Name: MACCROBAT2020
From: https://figshare.com
link: [link][https://figshare.com/articles/dataset/MACCROBAT2018/9764942]
Downloading tip: one can easily download the dataset from kaggle website and extract it into a proper folder.

**Quantum Physics dataset**:

Name: Quantum Physics articles on Arxiv 1994 to 2009
From: https://www.kaggle.com
link: [link][https://www.kaggle.com/datasets/louise2001/quantum-physics-articles-on-arxiv-1994-to-2009/discussion]
Downloading tip: one can easily download the dataset from kaggle website and extract it into a proper folder.

#### **Third_Assignment**:
This folder contains two 'ipynb' notebooks. 'Anomaly_Detection' contains an introduction on anomaly detection methods and models using pup sensor data. The aim of this notebook is to predict any abnormality in the performance of the water pomp based on the performance of itse 51 sensors.

'Assignment3' notebook, which is the reasearch advanture notebook, has exactly the dataset and purpose, but it is quite more in depth. Indeed, it compares the performance of different anomaly detection methods, uses a wide range of preprocessing techniques such as resampling, aggregation, smoothing techniques, to improve the performance of the anomaly detection models, and also introduce feature engineering techinques and utilizes SelectKBest technique to enhance the performance of the anomaly detection methods.

**Pupm Sensor dataset**:

Name: pump_sensor_data
From: https://www.kaggle.com
link: [link][https://www.kaggle.com/datasets/nphantawee/pump-sensor-data]
Downloading tip: one can easily download the dataset from kaggle website and extract it into a proper folder.

### Supervised Learning
The supervised_learning section contains four assignments, each of them foceses on an important concept in supervised machin learning. Fourth folder (Fourth_Assignment) contains an explanation about two fundamental concepts: Cost/loss function, and gradient descent. The fifth folder (Fifth_Assignment) describes a group of Supervised classification methods, all together with evaluation metrics, cross validation technique, and GridSearchCV. The sixth folder (Sixth_Assignment) focuses on Naive Bayes techniques and Decision Tree algorithm which can be counted as the most important biulding block of foundation ensemble methods. and finally, the seventh folder (Seventh_Assignment) explains one of the most important concepts in machine learning which is ensemble methods. It describes paralel methods, and sequential methods, and stacking methods.

#### **Fourth_Assignment**:
This folder contains two 'ipynb' notebooks. 'Study_notebook' contains logical and theoretical concepts behind loss/cost functions, and gradient decent method. Moreover, 'Assignment4' (research Advanture) puts all the theoretical aspects into practice. it uses an imaginary dataset called 'housing-data' that describes the trend of the house size based on its price. Actually, this dataset is a linear random dataset with a degree of error, but its equation is $y=x$. In this notebook, both cost function and gradient decent calculates manually using numpy package.

#### **Fifth_Assignment**:
This folder contains two 'ipynb' notebooks. 'E_LR_SVM' notebook describe some of the most important classification methods such as Logistic Regression, Linear Support Vector Classification (LinearSVC), C-Support Vector Classification (SVC) and General Algebraic Modeling System (GAMs) non-linear packages. Moreover, this notebook describes most of the important evaluation metrics, introduce Cross-validation technique, explain how to make and use a pipeline, and make a comprehensive use of GridSearchCV method along with an introduction on some preprocessing methods. Consequently, one can think of this notebook as a complete introduction on classification techniques. All the used datasets in this notebook are artificial.

'Assignment5', which is the research adventure of this section, uses the above approaches on breast cancer dataset to answer whether we can introduce a machine learning model that can predict the cancer based on the physical properties of a toumor. In this notebook beside the above material, some new preprocessing and data visualisation techniques are evaluated.

**Breast Cancer dataset**:

Name: Cancer Data
From: https://www.kaggle.com
link: [link][https://www.kaggle.com/datasets/erdemtaha/cancer-data]
Downloading tip: one can easily download the dataset from kaggle website and extract it into a proper folder.


#### **Sixth_Assignment**:
This folder contains two 'ipynb' notebooks. 'E_DT_NB' notebook contains the educational material related to Decision Tree algorithm and Naive Bayes methods. Decision Tree is one of the fundamental methods of ensemble methods, so it is explained in detail in this notebook along with the concept of Random Forest. Moreover, almost all the important Naive Bayes algorithms are explained. At the end the out come of the mentioned methods are compared to each other. The utilized dataset in this question is the famous iris dataset from sklearn datasets library.

Similar to the rest of this repository, 'Assignment6' notebook contains the research adventure. dataset in this question is the Titanic dataset, and this research aims at finding the rate of casualty of Titanic passenger based on different parameters such as gender and age. In fact this research intends to answer what parameters can help the passengers to survive from this ctastrophy. Beside the above methods, the focus in this assignment is mainly on data inspection and preprocessing, so one can find valuable insight into the art of inspecting a dataset in a proper way.

Name: The Complete Titanic Dataset
From: https://www.kaggle.com
link: [link][https://www.kaggle.com/datasets/vinicius150987/titanic3?resource=download]
Downloading tip: one can easily download the dataset from kaggle website and extract it into a proper folder.

#### **Seventh_Assignment**:
This folder contains two 'ipynb' notebooks. 'E_BAGGING_BOOSTING.ipynb' contains a comprehensive explanation about different ensemble methods. this notebook utilizes an artificial classification dataset to introduce and compare parallel, sequential, and stacking approaches. Bagging, Random Forest, Boosting, Stacking process are evaluated in this educational notebook.

'Assignment7' notebook contains research adventure of this part. In this assignment the research question of Assignment6 is evaluated again. The reason is that in the privious section the used models cannot show a promising prediction model, so in this assignment I want to inrease the prediction ability of them by using ensemble techniques to show the power of these techniques in camparison with solitary models.

Name: The Complete Titanic Dataset
From: https://www.kaggle.com
link: [link][https://www.kaggle.com/datasets/vinicius150987/titanic3?resource=download]
Downloading tip: one can easily download the dataset from kaggle website and extract it into a proper folder.

### Deep Learning
Finally, Deep Learning folder contains an introduction on neural network, tensors, and a guideline of using Keras package.

#### **Assignment_Neural_Network**:
This folder contains one 'ipynb' notebook. 'Assignment_Neural_Network' notebook describes the fundation of neural networks, mathematical background, some of the most famous algorithm, and Keras package. All the mentioned materials are explained by making a neural network for Titanic dataset. Consequently, this notebook is a proper first step to begin a journey in deep learning materials.

Name: The Complete Titanic Dataset
From: https://www.kaggle.com
link: [link][https://www.kaggle.com/datasets/vinicius150987/titanic3?resource=download]
Downloading tip: one can easily download the dataset from kaggle website and extract it into a proper folder.

## Challenges
During writting this educational repository I have faced some challenges:

1. First and for most, fighting with time. This project was quite time consuming for me since I was a student at the time at which I started to gather all of these materials together, so I do not dedicate enough time to this project.

2. I was learning about these materials while I am writing this notebook, so in order to present the correct material in this notebook, I had to study for a week, even for a small topic to present the correct information.

3. This field is a quite vast field, so I had to concentrate on a topic in this field as the main focuss, so I chose to work on classification and clustering rather than other section of machine learning.

4. Regarding technical challenges, finding the best visualization, analyzing, and evaluation techniques to present the trend and differences were quite challenging since I had to change the method in each section to cover a wider range of analysing materials.

5. Finding a cross section between programming and machine learning was another challenge since the focus of this notebook is machine learning; consequently, I had to use a simple programming format that is understandable for all people with elementary knowledge in the field of data science.
 
## Future Developements
I really like to develop this repository in the future, but if I could find a sponser for it or patienate audience who like to know about more machine learning methods. Two fields I would add to this repository in the future are manifold learning and regression sections, also I would add two to three chapters regarding deep learning section.

## Run and use the program
This program was written in Python 3.9.13 which is the diffult version of Anaconda environment. Also, 
the below packages were used in this program:

glob2                     0.7                pyhd3eb1b0_0
ipykernel                 6.15.2           py39haa95532_0
keras                     2.13.1                   pypi_0    pypi
matplotlib                3.2.0                    pypi_0    pypi
matplotlib-inline         0.1.6            py39haa95532_0
networkx                  2.8.4            py39haa95532_0
nltk                      3.7                pyhd3eb1b0_0
node2vec                  0.4.6                    pypi_0    pypi
numpy                     1.21.6                   pypi_0    pypi
opencv-python             4.7.0.72                 pypi_0    pypi
pandas                    2.0.3                    pypi_0    pypi
pathlib                   1.0.1            py39hcbf5309_7    conda-forge
pylint                    2.14.5           py39haa95532_0
pygam                     0.9.0                    pypi_0    pypi
pysptools                 0.15.0                   pypi_0    pypi
python                    3.9.13               h6244533_1
regex                     2022.7.9         py39h2bbff1b_0
scanpy                    1.9.4                    pypi_0    pypi
scikit-image              0.19.2           py39hf11a4ad_0
scikit-learn              1.2.2                    pypi_0    pypi
scikit-learn-intelex      2021.6.0         py39haa95532_0
scipy                     1.9.1                    pypi_0    pypi
seaborn                   0.12.2                   pypi_0    pypi
statsmodels               0.13.2           py39h2bbff1b_0
tensorflow                2.13.0                   pypi_0    pypi
tmtoolkit                 0.12.0                   pypi_0    pypi
wordcloud                 1.9.2                    pypi_0    pypi
yaml                      0.2.5                he774522_0    conda-forge

Regarding technical aspect of using this repository, one can download all the material on my github repository. To work with each folder, one can first download the relevant dataset, save it in a proper directory, and change the 'file_direction' in the config file to the mentioned directory. Also the name of the file in the config file ('file_name') should change to the name of the downloaded file. Then one can open a notebook and start to use it. 

In terms of educational usage, one should first study the tutorial notebook in detail, the go through each research adventure. However, I personally suggest to complete the research adventure by yourself, then have a look at its notebook.

## Credits
Each notebook has its own links and references, so one can find the references inside each notebook. I prefer not to mention any references here, and just mention the main link of this repo:

https://github.com/fenna/BFVM23DATASCNC5

To get access to this repository one can click on the following [link][https://github.com/hooman-b/Machine_learning_assignments]

## License
apache license 2.0

## Contact Information
Email: h.bahrdo@st.hanze.nl
Phone Number: +31681254428