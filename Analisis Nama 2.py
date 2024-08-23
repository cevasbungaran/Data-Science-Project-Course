# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:37:47 2023

@author: cevas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Levenshtein
import regex
import pickle

df = pd.read_excel('dataset_gform.xlsx')
df_daftarNamaIslamP = pd.read_excel('NAMA.xlsx','ISLAM_P')
df_daftarNamaIslamL = pd.read_excel('NAMA.xlsx','ISLAM_L')
df_daftarNamaKristenP = pd.read_excel('NAMA.xlsx','KRISTEN_P')
df_daftarNamaKristenL = pd.read_excel('NAMA.xlsx','KRISTEN_L')
df_daftarNamaSunda = pd.read_excel('NAMA SUNDA.xlsx')
df_rw9 = pd.read_excel('Nama Rw9.xlsx')

#cleaning data
df = df.drop('Timestamp', axis=1)
indexDrop = [16,19,48,50,60,69,54,83,136]
df = df.drop(indexDrop)
df_rw9 = df_rw9.drop(['No', 'RT'], axis=1)
df_rw9 = df_rw9.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

# series_baru = df['Nama Ibu']
# series_baru = series_baru.append(df['Nama Ibu'], ignore_index=True)
# series_baru = series_baru.append(df['Nama Bapa'],  ignore_index=True)
# df_baru['nama'] = series_baru.str.lower()
# series_baru = df['Etnis Ibu']
# series_baru = series_baru.append(df['Etnis Ibu'], ignore_index=True)
# series_baru = series_baru.append(df['Etnis Bapa'],  ignore_index=True)
# df_baru['etnis'] = series_baru.str.lower()
# namaDepan = list(df_baru['nama'].str.split().str[0])
# df_baru['nama_depan'] = namaDepan
# df_baru.loc[268, 'nama_depan'] = 'abdul'

#export df
df_baru = pd.DataFrame()
df_baru.to_excel('Nama Baru.xlsx', index=False)
df_nama = pd.read_csv('Nama Baru Gender.csv')
df_nama= df_nama.apply(lambda x: x.str.lower() if x.dtype == "object" else x)



#memisahkan laki-laki
df_laki = pd.DataFrame()
series_baru = df_nama['nama'][df_nama['gender'] == 'l']
series_baru = series_baru.append(df['Nama Bapa'], ignore_index=True)
series_baru = series_baru.append(df_rw9['Nama'][df_rw9['Jenis Kelamin'] == 'laki-laki'], ignore_index=True)
df_laki['nama'] = series_baru.str.lower()

series_etnis = df_nama['etnis'][df_nama['gender']=='l']
series_etnis = series_etnis.append(df['Etnis Bapa'], ignore_index=True)
series_etnis = series_etnis.append(df_rw9['Etnis'][df_rw9['Jenis Kelamin'] == 'laki-laki'], ignore_index=True)
df_laki['etnis'] = series_etnis.str.lower()

#memisahkan perempuan 
df_perempuan = pd.DataFrame()
series_baru = df_nama['nama'][df_nama['gender'] == 'p']
series_baru = series_baru.append(df['Nama Ibu'], ignore_index=True)
series_baru = series_baru.append(df_rw9['Nama'][df_rw9['Jenis Kelamin'] == 'perempuan'], ignore_index=True)
df_perempuan['nama'] = series_baru.str.lower()

series_etnis = df_nama['etnis'][df_nama['gender']=='p']
series_etnis = series_etnis.append(df['Etnis Ibu'], ignore_index=True)
series_etnis = series_etnis.append(df_rw9['Etnis'][df_rw9['Jenis Kelamin'] == 'perempuan'], ignore_index=True)
df_perempuan['etnis'] = series_etnis.str.lower()

#lowercase daftar nama islam dan kristen
df_daftarNamaIslamL = df_daftarNamaIslamL.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
df_daftarNamaIslamP = df_daftarNamaIslamP.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
df_daftarNamaKristenL = df_daftarNamaKristenL.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
df_daftarNamaKristenP = df_daftarNamaKristenP.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
df_daftarNamaSunda = df_daftarNamaSunda.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

#laki tanpa pisah etnis
namaDepanLaki_Semua= list(df_laki['nama'].str.split().str[0])
df_laki['nama depan'] = namaDepanLaki_Semua

#perempuan tanpa pisah etnis
namaDepanPerempuan_Semua= list(df_perempuan['nama'].str.split().str[0])
df_perempuan['nama depan'] = namaDepanPerempuan_Semua


#eksplorasi data
df_gb = df_nama.groupby('etnis').count()
dfrw_gb = df_rw9.groupby('Etnis').count()
dflaki_gb = df_laki.groupby('etnis').count()
dfperem_gb = df_perempuan.groupby('etnis').count()
dfgab = pd.concat([df_laki,df_perempuan])
dfgab_gb = dfgab.groupby('etnis').count()

plt.figure(figsize=(10,8))
plt.barh(df_gb.index, df_gb['nama'])
plt.xlabel('jumlah', fontsize=16)
plt.ylabel('Etnis', fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.title('Sebaran data hasil survey berdasarkan etnis', fontsize=16)

plt.figure(figsize=(10,8))
plt.barh(dfrw_gb.index, dfrw_gb['Nama'])
plt.xlabel('jumlah', fontsize=16)
plt.ylabel('Etnis', fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.title('Sebaran data penduduk berdasarkan etnis', fontsize=16)

plt.figure(figsize=(10,8))
plt.barh(dflaki_gb.index, dflaki_gb['nama'])
plt.xlabel('jumlah', fontsize=16)
plt.ylabel('Etnis', fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.title('Sebaran data laki-laki berdasarkan etnis', fontsize=16)

plt.figure(figsize=(10,8))
plt.barh(dfperem_gb.index, dfperem_gb['nama'])
plt.xlabel('jumlah', fontsize=16)
plt.ylabel('Etnis', fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.title('Sebaran data perempuan berdasarkan etnis', fontsize=16)

plt.figure(figsize=(10,8))
plt.barh(dfgab_gb.index, dfgab_gb['nama'])
plt.xlabel('jumlah', fontsize=16)
plt.ylabel('Etnis', fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.title('Sebaran seluruh data berdasarkan etnis', fontsize=16)

#ANALISIS BARU NAMA DENGAN DATA NAMA ISLAM DAN KRISTEN
def calculate_levenshtein_distance(str1, str2):
    dist = Levenshtein.distance(str1, str2)
    return 1/(dist+1)

def hitungKemiripan(nama, df_sample):
    distance_tp = ()
    for index, row in df_sample.iterrows():
        name = row['nama depan']
        distance = calculate_levenshtein_distance(nama, name)
        distance_tp = (*distance_tp,tuple([distance,nama,name]))
    distance_tp = tuple(sorted(distance_tp, reverse=True))
    return distance_tp



def cariKemiripan(df_Nama, df_Sample):
    hasil_tuple = ()
    for index1, row1 in df_Nama.iterrows():
        name1 = row1['nama depan']
        hasil_tuple = (*hasil_tuple, hitungKemiripan(name1, df_Sample))
    return hasil_tuple

def cari_distance(df_data, df_sample, etnis,hasil_df):
    hasiltupleKristen = cariKemiripan(df_data, df_sample)
    nama = []
    nama_mirip = []
    distance= []
    for tuples in hasiltupleKristen:
        nama.append(tuples[0][1])
        nama_mirip.append(tuples[0][2])
        distance.append(tuples[0][0])  
    hasil_df['nama'] = nama
    hasil_df[f'nama {etnis} mirip'] = nama_mirip
    hasil_df[f'distance {etnis}'] = distance
    
hitungKemiripan('carriza', df_daftarNamaKristenP)


# cari_mirip = cariKemiripan(df_nama, df_daftarNamaKristenP)

hasil_laki_df = pd.DataFrame()
cari_distance(df_laki, df_daftarNamaIslamL, 'islam', hasil_laki_df)
cari_distance(df_laki, df_daftarNamaKristenL, 'kristen', hasil_laki_df)
cari_distance(df_laki, df_daftarNamaSunda, 'sunda', hasil_laki_df)
hasil_laki_df['etnis'] = df_laki['etnis']

laki_ml_df = pd.DataFrame()
laki_ml_df['distance_islam'] = hasil_laki_df['distance islam']
laki_ml_df['distance_kristen'] = hasil_laki_df['distance kristen']
laki_ml_df['distance_sunda'] = hasil_laki_df['distance sunda']
laki_ml_df['kelas'] = 0
for i in hasil_laki_df.index:
    if hasil_laki_df['etnis'][i] == 'sunda':
        laki_ml_df['kelas'][i] = 'sunda'
    else:
        laki_ml_df['kelas'][i] = 'non-sunda'

#PEREMPUAN
hasil_perempuan_df = pd.DataFrame()
cari_distance(df_perempuan, df_daftarNamaIslamP, 'islam', hasil_perempuan_df)
cari_distance(df_perempuan, df_daftarNamaKristenP, 'kristen', hasil_perempuan_df)
cari_distance(df_perempuan, df_daftarNamaSunda, 'sunda', hasil_perempuan_df)
hasil_perempuan_df['etnis'] = df_perempuan['etnis']

perempuan_ml_df = pd.DataFrame()
perempuan_ml_df['distance_islam'] = hasil_perempuan_df['distance islam']
perempuan_ml_df['distance_kristen'] = hasil_perempuan_df['distance kristen']
perempuan_ml_df['distance_sunda'] = hasil_perempuan_df['distance sunda']
perempuan_ml_df['kelas'] = 0
for i in hasil_perempuan_df.index:
    if hasil_perempuan_df['etnis'][i] == 'sunda':
        perempuan_ml_df['kelas'][i] = 'sunda'
    else:
        perempuan_ml_df['kelas'][i] = 'non-sunda'



##############################################################################      

#Memishakan nama belakang
namaBelakang = list()
df_belakang = pd.DataFrame()
for nama in df_nama['nama']:
    namasplit = nama.split()
    belakang = namasplit[len(namasplit)-1]
    namaBelakang.append(belakang)
df_belakang['nama_belakang'] = namaBelakang

df_belakang[['etnis', 'gender']] = df_nama[['etnis','gender']]

############################KLASIFIKASI#######################################
#Laki-Laki
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz


le = preprocessing.LabelEncoder()

##### INI BARU YANG DATANYA UDAH BALANCE SUNDA DAN NON SUNDA #####
############ KLASIFIKASI LAKI BARU ##############
extract2 = laki_ml_df[['distance_islam','distance_kristen','distance_sunda']] #ganti nama kolom menjadi tingkat kemiripan dengan kristen, islam, sunda
feature_np = np.array(extract2.values)

label= laki_ml_df[['kelas']]

#ubah etnis non sunda
def ubah_etnis(etnis):
    if etnis == 'sunda':
        return etnis
    else:
        return 'non_sunda'
# Menggunakan metode apply untuk mengubah nilai etnis
label['kelas'] = label['kelas'].apply(ubah_etnis)

laki_label_np = np.array(label)

label_np = laki_label_np.ravel()

label_le = le.fit_transform(label_np)

feature_np.shape
label_le.shape

X = feature_np
Y = label_np

selector = SelectKBest(score_func=chi2, k=3)
selector.fit(X,Y)

cols = selector.get_support(indices=True)
print(cols)
# Buat fitur dataframe dgn k kolom paling signifikan
df_features = extract2.iloc[:,cols]

feature_array = np.array(df_features.values)

X_train, X_test, y_train, y_test = train_test_split(feature_array, label_np, test_size=0.3)

#Model KNN
i=2
while i<50:
    KNN = KNeighborsClassifier(n_neighbors = i)
    KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)
    print("Akurasi model klasifikasi dgn K =",i, ':',metrics.accuracy_score(y_test, y_pred))
    i+=1
    
#Model dengan akurasi paling tinggi
KNN = KNeighborsClassifier(n_neighbors = 33)
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)
classes = label.kelas.unique()
print(classification_report( y_test, y_pred, target_names=classes))

#confusion matrix knn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

#NB
NaiveBayes = GaussianNB()
NaiveBayes.fit(X_train, y_train)
Naive_y_pred = NaiveBayes.predict(X_test)
classes = label.kelas.unique()
print(classification_report( y_test, Naive_y_pred, target_names=classes))

#confusion matrix nb
cm2 = confusion_matrix(y_test, Naive_y_pred)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot()

# Decision Tree
DT_model = tree.DecisionTreeClassifier(criterion='entropy') #model rank ipk
# Train Decision Tree Classifer using the 70% of the dataset
DT_model.fit(X_train,y_train)
#Predict the response for test dataset
Y_pred = DT_model.predict(X_test)
classes = label.kelas.unique()
print(classification_report( y_test, Y_pred, target_names=classes))

#confusion matrix dtree
cm2 = confusion_matrix(y_test, Y_pred)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot()

# Visualisasi Pohon
dot_data = export_graphviz(DT_model, feature_names=extract2.columns, class_names=DT_model.classes_, filled=True,rounded=True,special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data)
#Create and save the graph of tree as image (PNG format)
graph = graphviz.Source(dot_data)
graph.render("Dtree_model", format='png')

#Random Forest
RF = RandomForestClassifier(n_estimators=10)
RF.fit(X_train,y_train)
Y_pred_RF = RF.predict(X_test)
classes = label.kelas.unique()
print(classification_report( y_test, Y_pred_RF, target_names=classes))

# confusion matrix random forest
cm2 = confusion_matrix(y_test, Y_pred_RF)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot()


#save model knn
pkl_filename = "knn_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(KNN, file)

#save model naive bayes
pkl_filename = "nb_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(NaiveBayes, file)

#save model dtree
pkl_filename = "dtree_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(DT_model, file)

############ KLASIFIKASI PEREMPUAN BARU ##############
extract = perempuan_ml_df[['distance_islam','distance_kristen','distance_sunda']] #ganti nama kolom menjadi tingkat kemiripan dengan kristen, islam, sunda
feature_np2 = np.array(extract.values)

label2= perempuan_ml_df[['kelas']]

#ubah etnis non sunda
def ubah_etnis(etnis):
    if etnis == 'sunda':
        return etnis
    else:
        return 'non_sunda'
# Menggunakan metode apply untuk mengubah nilai etnis
label2['kelas'] = label2['kelas'].apply(ubah_etnis)

perempuan_label_np = np.array(label2)

label_np2 = perempuan_label_np.ravel()

label_le2 = le.fit_transform(label_np2)

feature_np2.shape
label_le2.shape

X2 = feature_np2
Y2 = label_np2

selector2 = SelectKBest(score_func=chi2, k=3)
selector2.fit(X2,Y2)

cols2 = selector2.get_support(indices=True)
print(cols2)
# Buat fitur dataframe dgn k kolom paling signifikan
df_features2 = extract.iloc[:,cols]

feature_array2 = np.array(df_features2.values)

X_train2, X_test2, y_train2, y_test2 = train_test_split(feature_array2, label_np2, test_size=0.3)

#Model KNN
i=2
while i<50:
    KNN = KNeighborsClassifier(n_neighbors = i)
    KNN.fit(X_train2, y_train2)
    y_pred2 = KNN.predict(X_test)
    print("Akurasi model klasifikasi dgn K =",i, ':',metrics.accuracy_score(y_test, y_pred))
    i+=1
    
#Model dengan akurasi paling tinggi
KNN2 = KNeighborsClassifier(n_neighbors = 7)
KNN2.fit(X_train2, y_train2)
y_pred2 = KNN2.predict(X_test2)
classes = label2.kelas.unique()
print(classification_report( y_test2, y_pred2, target_names=classes))

#confusion matrix knn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm2 = confusion_matrix(y_test2, y_pred2)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot()

#NB
NaiveBayes2 = GaussianNB()
NaiveBayes2.fit(X_train2, y_train2)
Naive_y_pred2 = NaiveBayes2.predict(X_test2)
classes2 = label2.kelas.unique()
print(classification_report( y_test2, Naive_y_pred2, target_names=classes))

#confusion matrix nb
cm3 = confusion_matrix(y_test2, Naive_y_pred2)
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3)
disp3.plot()

#DTree
DT_model2 = tree.DecisionTreeClassifier(criterion='entropy') #model rank ipk
# Train Decision Tree Classifer using the 70% of the dataset
DT_model2.fit(X_train2,y_train2)
#Predict the response for test dataset
Y_pred_perempuan = DT_model.predict(X_test2)
classes = label.kelas.unique()
print(classification_report( y_test2, Y_pred_perempuan, target_names=classes))

# confusion matrix decision tree
cm2 = confusion_matrix(y_test2, Y_pred_perempuan)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot()

# Visualisasi Pohon
dot_data = export_graphviz(DT_model2, feature_names=extract2.columns, class_names=DT_model.classes_, filled=True,rounded=True,special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data)
#Create and save the graph of tree as image (PNG format)
graph = graphviz.Source(dot_data)
graph.render("Dtree_model", format='png')

#Random Forest
RF2 = RandomForestClassifier(n_estimators=20)
RF2.fit(X_train2,y_train2)
Y_pred_RF2 = RF.predict(X_test2)
classes = label.kelas.unique()
print(classification_report( y_test2, Y_pred_RF2, target_names=classes))

# confusion matrix decision tree
cm2 = confusion_matrix(y_test2, Y_pred_RF2)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot()


#save model knn
pkl_filename = "knn_model_perempuan.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(KNN2, file)

#save model naive bayes
pkl_filename = "nb_model_perempuan.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(NaiveBayes2, file)
#save model RF
pkl_filename = "RF_model_perempuan.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(RF2, file)

#load model knn
import pickle 
pkl_filename = "knn_model.pkl"  
with open(pkl_filename, 'rb') as file:  
    loaded_model_knn = pickle.load(file)

#load model naive bayes
import pickle 
pkl_filename = "nb_model.pkl"  
with open(pkl_filename, 'rb') as file:  
    loaded_model_nb = pickle.load(file)
#    
import pickle 
pkl_filename = "nb_model.pkl"  
with open(pkl_filename, 'rb') as file:  
    loaded_model_nb = pickle.load(file)




















