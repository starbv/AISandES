#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа №3
# ### Выгрузка данных из ЛР №2 (вариант №8)
# ### ('comp.sys.mac.hardware', 'soc.religion.christian', 'talk.religion.misc')

# In[2]:


import warnings
from sklearn.datasets import fetch_20newsgroups
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[3]:


categories = ['comp.sys.mac.hardware', 'soc.religion.christian', 'talk.religion.misc']
remove = ('headers', 'footers', 'quotes')

twenty_train_full = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=remove)
twenty_test_full = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42, remove=remove)


# ### Применение стемминга

# In[27]:


import nltk
from nltk import word_tokenize
from nltk.stem import *

nltk.download('punkt')


# In[5]:


def stemming(data):
    porter_stemmer = PorterStemmer()
    stem = []
    for text in data:
        nltk_tokens = word_tokenize(text)
        line = ''.join([' ' + porter_stemmer.stem(word) for word in nltk_tokens])
        stem.append(line)
    return stem


# In[6]:


stem_train = stemming(twenty_train_full.data)
stem_test = stemming(twenty_test_full.data)


# ### Задание
# ### Вариант №8
# ### Методы: [SVM, DT, LR]

# In[7]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# In[8]:


stop_words = [None, 'english']
max_features_values = [100, 500, 1000, 5000, 10000]
use_idf = [True, False]


# In[9]:


dt_first = range(1, 5, 1)
dt_second = range(5, 100, 20)

decision_tree_max_depth = [*dt_first, *dt_second]


# In[10]:


parameters_svm = {
    'vect__max_features': max_features_values,
    'vect__stop_words': stop_words,
    'tfidf__use_idf': use_idf,
}

parameters_dt = {
    'vect__max_features': max_features_values,
    'vect__stop_words': stop_words,
    'tfidf__use_idf': use_idf,
    'clf__criterion': ('gini', 'entropy'),
    'clf__max_depth': decision_tree_max_depth,
}

parameters_lr = {
    'vect__max_features': max_features_values,
    'vect__stop_words': stop_words,
    'tfidf__use_idf': use_idf,
    'clf__solver': ['newton-cg', 'lbfgs', 'sag', 'liblinear'],
    'clf__penalty': ['l2']
}

parameters_lr_l1 = {
    'vect__max_features': max_features_values,
    'vect__stop_words': stop_words,
    'tfidf__use_idf': use_idf,
    'clf__solver': ['liblinear'],
    'clf__penalty': ['l1'],
}


# In[11]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# ### Метод опорных векторов (SVM)

# #### Без использования стемминга

# In[12]:


text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SVC())])
gscv_svm = GridSearchCV(text_clf_svm, param_grid=parameters_svm, n_jobs=-1)
gscv_svm.fit(twenty_train_full.data, twenty_train_full.target)


# #### С использованием стемминга

# In[13]:


text_clf_svm_stem = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', SVC(kernel='linear', C=1.0))])
gscv_svm_stem = GridSearchCV(text_clf_svm_stem, param_grid=parameters_svm, n_jobs=-1)
gscv_svm_stem.fit(stem_train, twenty_train_full.target)


# ### Дерево решений (DT)

# #### Без использования стемминга

# In[14]:


text_clf_dt = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', DecisionTreeClassifier())])
gscv_dt = GridSearchCV(text_clf_dt, param_grid=parameters_dt, n_jobs=-1)
gscv_dt.fit(twenty_train_full.data, twenty_train_full.target)


# #### С использованием стема

# In[15]:


text_clf_dt_stem = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', DecisionTreeClassifier())])
gscv_dt_stem = GridSearchCV(text_clf_dt_stem, param_grid=parameters_dt, n_jobs=-1)
gscv_dt_stem.fit(stem_train, twenty_train_full.target)


# ### Логистическая регрессия (LR)

# #### Без использования стемминга

# In[16]:


text_clf_lr = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression())])
gscv_lr = GridSearchCV(text_clf_lr, param_grid=parameters_lr, n_jobs=-1)
gscv_lr.fit(twenty_train_full.data, twenty_train_full.target)


text_clf_lr_l1 = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression())])
gscv_lr_l1 = GridSearchCV(text_clf_lr_l1, param_grid=parameters_lr_l1, n_jobs=-1)
gscv_lr_l1.fit(twenty_train_full.data, twenty_train_full.target)


# #### С использованием стемминга

# In[17]:


text_clf_lr_stem = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', LogisticRegression())])
gscv_lr_stem = GridSearchCV(text_clf_lr_stem, param_grid=parameters_lr, n_jobs=-1)
gscv_lr_stem.fit(stem_train, twenty_train_full.target)


text_clf_lr_l1_stem = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', LogisticRegression())])
gscv_lr_l1_stem = GridSearchCV(text_clf_lr_l1_stem, param_grid=parameters_lr_l1, n_jobs=-1)
gscv_lr_l1_stem.fit(stem_train, twenty_train_full.target)


# ### Вывод полученных результатов анализа

# In[18]:


from sklearn.metrics import classification_report


# In[19]:


predicted_svm = gscv_svm.predict(twenty_test_full.data)
print('Метод опорных векторов (SVM) без стемминга\n')
print(classification_report(twenty_test_full.target, predicted_svm, target_names=categories))
print(gscv_svm.best_params_)


# In[20]:


predicted_svm_stem = gscv_svm_stem.predict(twenty_test_full.data)
print('Метод опорных векторов (SVM) со стеммингом\n')
print(classification_report(twenty_test_full.target, predicted_svm_stem, target_names=categories))
print(gscv_svm_stem.best_params_)


# In[21]:


predicted_dt = gscv_dt.predict(twenty_test_full.data)
print('Дерево решений (DT) без стемминга\n')
print(classification_report(twenty_test_full.target, predicted_dt, target_names=categories))
print(gscv_dt.best_params_)


# In[22]:


predicted_dt_stem = gscv_dt_stem.predict(twenty_test_full.data)
print('Дерево решений (DT) со стеммингом\n')
print(classification_report(twenty_test_full.target, predicted_dt_stem, target_names=categories))
print(gscv_dt_stem.best_params_)


# In[23]:


predicted_lr = gscv_lr.predict(twenty_test_full.data)
print('Логистическая регрессия (LR) без стемминга\n')
print(classification_report(twenty_test_full.target, predicted_lr, target_names=categories))
print(gscv_lr.best_params_)

predicted_lr_l1 = gscv_lr_l1.predict(twenty_test_full.data)
print('Логистическая регрессия_l1 (LR) без стемминга\n')
print(classification_report(twenty_test_full.target, predicted_lr_l1, target_names=categories))
print(gscv_lr_l1.best_params_)


# In[24]:


predicted_lr_stem = gscv_lr_stem.predict(twenty_test_full.data)
print('Логистическая регрессия (LR) со стеммингом\n')
print(classification_report(twenty_test_full.target, predicted_lr_stem, target_names=categories))
print(gscv_lr_stem.best_params_)

predicted_lr_l1_stem = gscv_lr_l1_stem.predict(twenty_test_full.data)
print('Логистическая регрессия_l1 (LR) со стеммингом\n')
print(classification_report(twenty_test_full.target, predicted_lr_l1_stem, target_names=categories))
print(gscv_lr_l1_stem.best_params_)


# ### Сравнительная таблица

# In[25]:


import pandas as pd


# In[26]:


writer = pd.ExcelWriter('result.xlsx', engine='openpyxl')

# Метод опорных векторов (SVM) без стемминга
df1 = pd.DataFrame(classification_report(predicted_svm, twenty_test_full.target, output_dict=True))

# Метод опорных векторов (SVM) со стеммингом
df2 = pd.DataFrame(classification_report(predicted_svm_stem, twenty_test_full.target, output_dict=True))

# Дерево решений (DT) без стемминга
df3 = pd.DataFrame(classification_report(predicted_dt, twenty_test_full.target, output_dict=True))

# Дерево решений (DT) со стеммингом
df4 = pd.DataFrame(classification_report(predicted_dt_stem, twenty_test_full.target, output_dict=True))

# Логистическая регрессия (LR) без стемминга
df5 = pd.DataFrame(classification_report(predicted_lr, twenty_test_full.target, output_dict=True))

# Логистическая регрессия_l1 (LR) без стемминга
df6 = pd.DataFrame(classification_report(predicted_lr_l1, twenty_test_full.target, output_dict=True))

# Логистическая регрессия (LR) со стеммингом
df7 = pd.DataFrame(classification_report(predicted_lr_stem, twenty_test_full.target, output_dict=True))

# Логистическая регрессия_l1 (LR) со стеммингом
df8 = pd.DataFrame(classification_report(predicted_lr_l1_stem, twenty_test_full.target, output_dict=True))

df1.to_excel(writer, sheet_name='SVM без стемминга')
df2.to_excel(writer, sheet_name='SVM со стеммингом')

df3.to_excel(writer, sheet_name='DT без стемминга')
df4.to_excel(writer, sheet_name='DT со стеммингом')

df5.to_excel(writer, sheet_name='LR без стемминга')
df6.to_excel(writer, sheet_name='LR_l1 без стемминга')

df7.to_excel(writer, sheet_name='LR со стеммингом')
df8.to_excel(writer, sheet_name='LR_l1 со стеммингом')

writer.close()

