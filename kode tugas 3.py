# -*- coding: utf-8 -*-
"""
Created on Wed May 24 08:18:24 2023

@author: cevas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_tIbadah = pd.read_excel('dataset_tempatIbadah.xlsx')
data_resto = pd.read_excel('dataset_restoran.xlsx')

#proporsi
grouped_data1 = data_tIbadah.groupby(['Kelurahan', 'Agama']).size().unstack().fillna(0)
proporsi1 = grouped_data1.div(grouped_data1.sum(axis=1), axis=0)
fig, ax = plt.subplots()
colors = ['red', 'green', 'blue', 'yellow', 'pink']
proporsi1.plot(kind='bar', stacked=True, color = colors, ax = ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Proporsi Tempat Ibadah berdasarkan Kelurahan")
plt.xlabel("Kelurahan")
plt.ylabel("Proporsi")
plt.show()

grouped_data2 = data_resto.groupby(['Kelurahan', 'Etnisitas']).size().unstack().fillna(0)
proporsi2 = grouped_data2.div(grouped_data2.sum(axis=1), axis=0)
fig, ax = plt.subplots()
colors = ['red', 'green', 'blue', 'yellow', 'pink', 'purple']
proporsi2.plot(kind='bar', stacked=True, color = colors, ax = ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Proporsi Etnisitas Restoran berdasarkan Kelurahan")
plt.xlabel("Kelurahan")
plt.ylabel("Proporsi")
plt.show()


#data mentah tempat ibadah
paskal_tIbadah = data_tIbadah[data_tIbadah['Kelurahan']=='Pasir Kaliki']
paskal_tIbadahGb = paskal_tIbadah.groupby('Agama').count()
balged_tIbadah = data_tIbadah[data_tIbadah['Kelurahan']=='Balonggede']
balged_tIbadahGb = balged_tIbadah.groupby('Agama').count()
cibadak_tIbadah = data_tIbadah[data_tIbadah['Kelurahan']=='Cibadak']
cibadak_tIbadahGb = cibadak_tIbadah.groupby('Agama').count()
jamika_tIbadah = data_tIbadah[data_tIbadah['Kelurahan']=='Jamika']
jamika_tIbadahGb = jamika_tIbadah.groupby('Agama').count()
ciroyom_tIbadah = data_tIbadah[data_tIbadah['Kelurahan']=='Ciroyom']
ciroyom_tIbadahGb = ciroyom_tIbadah.groupby('Agama').count()
bakmis_tIbadah = data_tIbadah[data_tIbadah['Kelurahan']=='Babakan Ciamis']
bakmis_tIbadahGb = bakmis_tIbadah.groupby('Agama').count()

colors = ['red', 'green', 'blue', 'yellow']
#barchart jumlah agama per kelurahan
#paskal
plt.bar(paskal_tIbadahGb.index, paskal_tIbadahGb['Kelurahan'], color = colors)
plt.title('Total Agama di Kelurahan Pasir Kaliki')
plt.xlabel('Agama')
plt.ylabel('Jumlah')
plt.show()
#balonggede
plt.bar(balged_tIbadahGb.index, balged_tIbadahGb['Kelurahan'], color = colors)
plt.title('Total Agama di Kelurahan Balonggede')
plt.xlabel('Agama')
plt.ylabel('Jumlah')
plt.show()
#cibadak
plt.bar(cibadak_tIbadahGb.index, cibadak_tIbadahGb['Kelurahan'], color = colors)
plt.title('Total Agama di Kelurahan Cibadak')
plt.xlabel('Agama')
plt.ylabel('Jumlah')
plt.show()
#jamika
plt.bar(jamika_tIbadahGb.index, jamika_tIbadahGb['Kelurahan'], color = colors)
plt.title('Total Agama di Kelurahan Jamika')
plt.xlabel('Agama')
plt.ylabel('Jumlah')
plt.show()
#ciroyom
plt.bar(ciroyom_tIbadahGb.index, ciroyom_tIbadahGb['Kelurahan'], color = colors)
plt.title('Total Agama di Kelurahan Ciroyom')
plt.xlabel('Agama')
plt.ylabel('Jumlah')
plt.show()
#babakan ciamis
plt.bar(bakmis_tIbadahGb.index, bakmis_tIbadahGb['Kelurahan'], color = colors)
plt.title('Total Agama di Kelurahan Babakan Ciamis')
plt.xlabel('Agama')
plt.ylabel('Jumlah')
plt.show()

#barchart rata-rata luas tempat ibadah per kelurahan
mean_paskal_tIbadahGb = paskal_tIbadah.groupby('Agama').mean()
mean_balged_tIbadahGb = balged_tIbadah.groupby('Agama').mean()
mean_cibadak_tIbadahGb = cibadak_tIbadah.groupby('Agama').mean()
mean_jamika_tIbadahGb = jamika_tIbadah.groupby('Agama').mean()
mean_ciroyom_tIbadahGb = ciroyom_tIbadah.groupby('Agama').mean()
mean_bakmis_tIbadahGb = bakmis_tIbadah.groupby('Agama').mean()
#paskal
plt.bar(mean_paskal_tIbadahGb.index, mean_paskal_tIbadahGb['Luas (m)'], color = colors)
plt.title('Rata-rata Luas Tempat Ibadah di Kelurahan Pasir Kaliki')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_paskal_tIbadahGb['Luas (m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()
#balonggede
plt.bar(mean_balged_tIbadahGb.index, mean_balged_tIbadahGb['Luas (m)'], color = colors)
plt.title('Rata-rata Luas Tempat Ibadah di Kelurahan Balonggede')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_balged_tIbadahGb['Luas (m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()
#cibadak
plt.bar(mean_cibadak_tIbadahGb.index, mean_cibadak_tIbadahGb['Luas (m)'], color = colors)
plt.title('Rata-rata Luas Tempat Ibadah di Kelurahan Cibadak')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_cibadak_tIbadahGb['Luas (m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()
#jamika
plt.bar(mean_jamika_tIbadahGb.index, mean_jamika_tIbadahGb['Luas (m)'], color = colors)
plt.title('Rata-rata Luas Tempat Ibadah di Kelurahan Jamika')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_jamika_tIbadahGb['Luas (m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()
#ciroyom
plt.bar(mean_ciroyom_tIbadahGb.index, mean_ciroyom_tIbadahGb['Luas (m)'], color = colors)
plt.title('Rata-rata Luas Tempat Ibadah di Kelurahan Ciroyom')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_ciroyom_tIbadahGb['Luas (m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()
#babakan ciamis
plt.bar(mean_bakmis_tIbadahGb.index, mean_bakmis_tIbadahGb['Luas (m)'], color = colors)
plt.title('Rata-rata Luas Tempat Ibadah di Kelurahan Babakan Ciamis')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_bakmis_tIbadahGb['Luas (m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()


#data mentah restoran
paskal_resto = data_resto[data_resto['Kelurahan']=='Pasir Kaliki']
paskal_restoGb = paskal_resto.groupby('Etnisitas').count()
balged_resto = data_resto[data_resto['Kelurahan']=='Balonggede']
balged_restoGb = balged_resto.groupby('Etnisitas').count()
cibadak_resto = data_resto[data_resto['Kelurahan']=='Cibadak']
cibadak_restoGb = cibadak_resto.groupby('Etnisitas').count()
jamika_resto = data_resto[data_resto['Kelurahan']=='Jamika']
jamika_restoGb = jamika_resto.groupby('Etnisitas').count()
ciroyom_resto = data_resto[data_resto['Kelurahan']=='Ciroyom']
ciroyom_restoGb = ciroyom_resto.groupby('Etnisitas').count()
bakmis_resto = data_resto[data_resto['Kelurahan']=='Babakan Ciamis']
bakmis_restoGb = bakmis_resto.groupby('Etnisitas').count()

colors = ['red', 'green', 'blue', 'yellow', 'purple']
#barchart jumlah etnis pada restoran per kelurahan
#paskal
plt.bar(paskal_restoGb.index, paskal_restoGb['Kelurahan'], color = colors)
plt.title('Total Etnisitas pada Restoran di Kelurahan Pasir Kaliki')
plt.xlabel('Etnisitas')
plt.ylabel('Jumlah')
plt.show()
#balonggede
plt.bar(balged_restoGb.index, balged_restoGb['Kelurahan'], color = colors)
plt.title('Total Etnisitas pada Restoran di Kelurahan Balonggede')
plt.xlabel('Etnisitas')
plt.ylabel('Jumlah')
plt.show()
#cibadak
plt.bar(cibadak_restoGb.index, cibadak_restoGb['Kelurahan'], color = colors)
plt.title('Total Etnisitas pada Restoran di Kelurahan Cibadak')
plt.xlabel('Etnisitas')
plt.ylabel('Jumlah')
plt.show()
#jamika
plt.bar(jamika_restoGb.index, jamika_restoGb['Kelurahan'], color = colors)
plt.title('Total Etnisitas pada Restoran di Kelurahan Jamika')
plt.xlabel('Etnisitas')
plt.ylabel('Jumlah')
plt.show()
#ciroyom
plt.bar(ciroyom_restoGb.index, ciroyom_restoGb['Kelurahan'], color = colors)
plt.title('Total Etnisitas pada Restoran di Kelurahan Ciroyom')
plt.xlabel('Etnisitas')
plt.ylabel('Jumlah')
plt.show()
#babakan ciamis
plt.bar(bakmis_restoGb.index, bakmis_restoGb['Kelurahan'], color = colors)
plt.title('Total Etnisitas pada Restoran di Kelurahan Babakan Ciamis')
plt.xlabel('Etnisitas')
plt.ylabel('Jumlah')
plt.show()

#barchart rata-rata luas tempat ibadah per kelurahan
mean_paskal_restoGb = paskal_resto.groupby('Etnisitas').mean()
mean_balged_restoGb = balged_resto.groupby('Etnisitas').mean()
mean_cibadak_restoGb = cibadak_resto.groupby('Etnisitas').mean()
mean_jamika_restoGb = jamika_resto.groupby('Etnisitas').mean()
mean_ciroyom_restoGb = ciroyom_resto.groupby('Etnisitas').mean()
mean_bakmis_restoGb = bakmis_resto.groupby('Etnisitas').mean()
#paskal
plt.bar(mean_paskal_restoGb.index, mean_paskal_restoGb['Luas(m)'], color = colors)
plt.title('Rata-rata Luas Restoran di Kelurahan Pasir Kaliki')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_paskal_restoGb['Luas(m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()
#balonggede
plt.bar(mean_balged_restoGb.index, mean_balged_restoGb['Luas(m)'], color = colors)
plt.title('Rata-rata Luas Restoran di Kelurahan Balonggede')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_balged_restoGb['Luas(m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()
#cibadak
plt.bar(mean_cibadak_restoGb.index, mean_cibadak_restoGb['Luas(m)'], color = colors)
plt.title('Rata-rata Luas Restoran di Kelurahan Cibadak')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_cibadak_restoGb['Luas(m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()
#jamika
plt.bar(mean_jamika_restoGb.index, mean_jamika_restoGb['Luas(m)'], color = colors)
plt.title('Rata-rata Luas Restoran di Kelurahan Jamika')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_jamika_restoGb['Luas(m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()
#ciroyom
plt.bar(mean_ciroyom_restoGb.index, mean_ciroyom_restoGb['Luas(m)'], color = colors)
plt.title('Rata-rata Luas Restoran di Kelurahan Ciroyom')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_ciroyom_restoGb['Luas(m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()
#babakan ciamis
plt.bar(mean_bakmis_restoGb.index, mean_bakmis_restoGb['Luas(m)'], color = colors)
plt.title('Rata-rata Luas Restoran di Kelurahan Babakan Ciamis')
plt.xlabel('Agama')
plt.ylabel('Luas')
for i, value in enumerate(mean_bakmis_restoGb['Luas(m)']):
    plt.text(i, value, str(round(value,2)), ha='center', va='bottom')
plt.show()








