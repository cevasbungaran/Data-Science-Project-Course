# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 20:07:09 2023

@author: Keluarga
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
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_levenshtein_ratio(str1, str2):
    return Levenshtein.ratio(str1, str2)

def hitungKemiripan(nama, df_sample):
    rasio_tp = ()
    for index, row in df_sample.iterrows():
        name = row['nama depan']
        distance = calculate_levenshtein_ratio(nama, name)
        rasio_tp = (*rasio_tp,tuple([distance,nama,name]))
    rasio_tp = tuple(sorted(rasio_tp, reverse=True))
    return rasio_tp


def cariKemiripan(df_Nama, df_Sample):
    hasil_tuple = ()
    for index1, row1 in df_Nama.iterrows():
        name1 = row1['nama depan']
        hasil_tuple = (*hasil_tuple, hitungKemiripan(name1, df_Sample))
    return hasil_tuple

def cari_ratio(df_data, df_sample, etnis,hasil_df):
    hasiltupleKristen = cariKemiripan(df_data, df_sample)
    nama = []
    nama_mirip = []
    ratio= []
    for tuples in hasiltupleKristen:
        nama.append(tuples[0][1])
        nama_mirip.append(tuples[0][2])
        ratio.append(tuples[0][0])
    hasil_df['nama'] = nama
    hasil_df[f'nama {etnis} mirip'] = nama_mirip
    hasil_df[f'ratio {etnis}'] = ratio



hasil_laki_df = pd.DataFrame()
cari_ratio(df_laki, df_daftarNamaIslamL, 'islam', hasil_laki_df)
cari_ratio(df_laki, df_daftarNamaKristenL, 'kristen', hasil_laki_df)
cari_ratio(df_laki, df_daftarNamaSunda, 'sunda', hasil_laki_df)
hasil_laki_df['etnis'] = df_laki['etnis']

laki_ml_df = pd.DataFrame()
laki_ml_df['ratio_islam'] = hasil_laki_df['ratio islam']
laki_ml_df['ratio_kristen'] = hasil_laki_df['ratio kristen']
laki_ml_df['ratio_sunda'] = hasil_laki_df['ratio sunda']
laki_ml_df['kelas'] = 0
for i in hasil_laki_df.index:
    if hasil_laki_df['etnis'][i] == 'sunda':
        laki_ml_df['kelas'][i] = 'sunda'
    else:
        laki_ml_df['kelas'][i] = 'non-sunda'




#PEREMPUAN
hasil_perempuan_df = pd.DataFrame()
cari_ratio(df_perempuan, df_daftarNamaIslamP, 'islam', hasil_perempuan_df)
cari_ratio(df_perempuan, df_daftarNamaKristenP, 'kristen', hasil_perempuan_df)
cari_ratio(df_perempuan, df_daftarNamaSunda, 'sunda', hasil_perempuan_df)
hasil_perempuan_df['ettnis'] = df_perempuan['etnis']



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
le = preprocessing.LabelEncoder()

extract = laki_ml_df[['ratio_islam','ratio_kristen','ratio_sunda']] #ganti nama kolom menjadi tingkat kemiripan dengan kristen, islam, sunda
feature_np = np.array(extract.values)

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
df_features = extract.iloc[:,cols]

feature_array = np.array(df_features.values)

X_train, X_test, y_train, y_test = train_test_split(feature_array, label_np, test_size=0.3)
X_train2, X_test2, y_train2, y_test2 = train_test_split(feature_array, label_np, test_size=0.3)

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
print(classification_report( y_test, y_pred))

#confusion matrix knn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

#NB
NaiveBayes = GaussianNB()
NaiveBayes.fit(X_train2, y_train2)
Naive_y_pred = NaiveBayes.predict(X_test2)
classes = label.kelas.unique()
print(classification_report( y_test2, Naive_y_pred, target_names=classes))

#confusion matrix nb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm2 = confusion_matrix(y_test2, Naive_y_pred)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp.plot()

#save model
pkl_filename = "knn_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(KNN, file)

#load model
import pickle 
pkl_filename = "knn_model.pkl"  
with open(pkl_filename, 'rb') as file:  
    loaded_model_knn = pickle.load(file)
    

#prediksi
df_nama_baru = pd.read_excel('data prediksi.xlsx')
df_nama_baru = df_nama_baru.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
df_laki_baru = pd.DataFrame()
series_baru = df_nama_baru['nama'][df_nama_baru['jenis kelamin'] == 'l']
df_laki_baru['nama'] = series_baru.str.lower()
namaDepan_baru = list(df_laki_baru['nama'].str.split().str[0])
df_namaDepan_laki_baru = pd.DataFrame()
df_namaDepan_laki_baru['nama depan'] = namaDepan_baru
df_namaDepan_laki_baru = df_namaDepan_laki_baru.drop_duplicates()

hasil_laki_baru = pd.DataFrame()
cari_ratio(df_namaDepan_laki_baru, df_daftarNamaKristenL, 'kristen', hasil_laki_baru)
cari_ratio(df_namaDepan_laki_baru, df_daftarNamaIslamL, 'islam', hasil_laki_baru)
cari_ratio(df_namaDepan_laki_baru, df_daftarNamaSunda, 'sunda', hasil_laki_baru)

fitur_df = hasil_laki_baru[['ratio islam','ratio kristen','ratio sunda']]
#Buat array Numpy utk features
fitur_np = np.array(fitur_df.values)

fitur_df.shape

#Lakukan prediksi
prediksi = KNN.predict(fitur_np)
print(prediksi)





















