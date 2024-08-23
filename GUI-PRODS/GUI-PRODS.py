#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:21:32 2023

@author: carizza
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from tkinter import ttk
import pickle
import Levenshtein
import numpy as np


class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PRODS")
        
        # pake notebook tkinter
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        self.page1 = ttk.Frame(self.notebook)
        self.notebook.add(self.page1, text="Prediksi Etnis")
        
        self.label1 = ttk.Label(self.page1, text="Masukkan Nama Lengkap Tanpa Gelar:")
        self.label1.pack(pady=10)
        
        self.input_string = tk.StringVar()
        self.entry1 = ttk.Entry(self.page1, textvariable=self.input_string)
        self.entry1.pack(pady=5)
        
        self.gender_label = ttk.Label(self.page1, text="Jenis Kelamin:")
        self.gender_label.pack()
        
        self.gender_var = tk.StringVar()
        self.gender_dropdown = ttk.Combobox(
            self.page1,
            textvariable=self.gender_var,
            state="readonly",
            values=["Perempuan", "Laki-laki"]
        )
        self.gender_dropdown.pack(pady=5)
        
        self.submit_button = ttk.Button(self.page1, text="Prediksi", command=self.process_string)
        self.submit_button.pack(pady=10)
        
        
        self.output_label = ttk.Label(self.page1, text="Hasil Prediksi Etnis:")
        self.output_label.pack()
        
        
        self.output_string = tk.StringVar()
        self.output_text = ttk.Label(self.page1, textvariable=self.output_string)
        self.output_text.pack()
        
        self.reset_button = ttk.Button(self.page1, text="Reset", command=reset_string_var(self))
        self.reset_button.pack(side='bottom',anchor='se',pady=20,padx=20)
        
        self.page2 = ttk.Frame(self.notebook)
        self.notebook.add(self.page2, text="Prediksi Etnis dari File Excel")
        
        self.import_label = ttk.Label(self.page2, text="Hanya dapat import file Excel (.xlsx) dengan 2 kolom berisi nama lengkap tanpa gelar dan jenis kelamin(l/p):")
        self.import_label.pack(pady=10)
    
        
        self.import_button = ttk.Button(self.page2, text="Import File", command=self.import_excel)
        self.import_button.pack()
        
        self.table_label = ttk.Label(self.page2, text="Tabel Hasil Prediksi Etnis:")
        self.table_label.pack()
        
        self.table = ttk.Treeview(self.page2, columns=("Nama Lengkap", "Jenis Kelamin", "Etnis"))
        self.table.heading("#1", text="Nama Lengkap")
        self.table.heading("#2", text="Jenis Kelamin")
        self.table.heading("#3", text="Etnis")
        self.table.pack()
    
    
    def process_string(self):
        input_text = self.input_string.get()
        output_text = input_text.lower()
        gender_choice = self.gender_var.get()
        self.output_string.set(prediksiNama(output_text, gender_choice))
        
    
    def import_excel(self):
        reset_table(self)
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if file_path:
            df = pd.read_excel(file_path)
            prediksitabel(df)
            for i, row in df.iterrows():
                self.table.insert("", "end", values=(row["Nama Lengkap"], row["Jenis Kelamin"], row["Etnis"]))
            return self.table
    
    
    

pkl_filename = "nb_model.pkl" 
with open(pkl_filename, 'rb') as file:  
    loaded_model_nb = pickle.load(file)
    
pkl_filename = "RF_model_perempuan.pkl" 
with open(pkl_filename, 'rb') as file:  
    loaded_model_nb_perempuan = pickle.load(file)
    
    
    def prediksiNama(nama, gender):
        gender = gender
        nama = nama
        nama = nama.lower()
        if gender == 'Laki-laki':
            prediksi = prediksiNamaLaki(nama)
            
            
        elif gender == 'Perempuan':
            prediksi = prediksiNamaPerempuan(nama)
        
        return prediksi
            

    def prediksiNamaLaki(nama):
        df_daftarNamaSunda = pd.read_excel('NAMA SUNDA.xlsx')
        df_daftarNamaIslamL = pd.read_excel('NAMA.xlsx','ISLAM_L')
        df_daftarNamaKristenL = pd.read_excel('NAMA.xlsx','KRISTEN_L')
        df_daftarNamaSunda = df_daftarNamaSunda.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df_daftarNamaIslamL = df_daftarNamaIslamL.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df_daftarNamaKristenL = df_daftarNamaKristenL.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        namaDepan_baru = pd.Series(nama.split()[0])
        df_namaBaru = pd.DataFrame()
        df_namaBaru['nama depan'] = namaDepan_baru        
        hasilIslam = hitungKemiripan(namaDepan_baru[0], df_daftarNamaIslamL)
        df_namaBaru['distance islam'] = hasilIslam[0][0]
        hasilKristen = hitungKemiripan(namaDepan_baru[0], df_daftarNamaKristenL)
        df_namaBaru['distance kristen'] = hasilKristen[0][0]
        hasilSunda = hitungKemiripan(namaDepan_baru[0], df_daftarNamaSunda)
        df_namaBaru['distance sunda'] = hasilSunda[0][0]
        feature_df = pd.DataFrame()
        feature_df['distance islam'] = df_namaBaru['distance islam']
        feature_df['distance kristen'] = df_namaBaru['distance kristen']
        feature_df['distance sunda'] = df_namaBaru['distance sunda']
        nb_pred = loaded_model_nb.predict(feature_df)
        if nb_pred[0] == 'sunda':
            return 'Sunda'
        elif nb_pred[0] == 'non_sunda':
            return 'Non Sunda'
        
    def prediksiNamaPerempuan(nama):
        df_daftarNamaSunda = pd.read_excel('NAMA SUNDA.xlsx')
        df_daftarNamaIslamP = pd.read_excel('NAMA.xlsx','ISLAM_P')
        df_daftarNamaKristenP = pd.read_excel('NAMA.xlsx','KRISTEN_P')
        df_daftarNamaSunda = df_daftarNamaSunda.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df_daftarNamaIslamP = df_daftarNamaIslamP.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df_daftarNamaKristenP = df_daftarNamaKristenP.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        namaDepan_baru = pd.Series(nama.split()[0])
        df_namaBaru = pd.DataFrame()
        df_namaBaru['nama depan'] = namaDepan_baru 
        hasilIslam = hitungKemiripan(namaDepan_baru[0], df_daftarNamaIslamP)
        df_namaBaru['distance islam'] = hasilIslam[0][0]
        hasilKristen = hitungKemiripan(namaDepan_baru[0], df_daftarNamaKristenP)
        df_namaBaru['distance kristen'] = hasilKristen[0][0]
        hasilSunda = hitungKemiripan(namaDepan_baru[0], df_daftarNamaSunda)
        df_namaBaru['distance sunda'] = hasilSunda[0][0]
        feature_df = pd.DataFrame()
        feature_df['distance islam'] = df_namaBaru['distance islam']
        feature_df['distance kristen'] = df_namaBaru['distance kristen']
        feature_df['distance sunda'] = df_namaBaru['distance sunda']
        nb_pred = loaded_model_nb_perempuan.predict(feature_df)
        if nb_pred[0] == 'sunda':
            return 'Sunda'
        elif nb_pred[0] == 'non_sunda':
            return 'Non Sunda'

    
    def prediksitabel(tabel):
        #prediksi
        
        df_daftarNamaSunda = pd.read_excel('NAMA SUNDA.xlsx')
        df_daftarNamaIslamP = pd.read_excel('NAMA.xlsx','ISLAM_P')
        df_daftarNamaIslamL = pd.read_excel('NAMA.xlsx','ISLAM_L')
        df_daftarNamaKristenP = pd.read_excel('NAMA.xlsx','KRISTEN_P')
        df_daftarNamaKristenL = pd.read_excel('NAMA.xlsx','KRISTEN_L')
        df_daftarNamaSunda = df_daftarNamaSunda.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df_daftarNamaIslamL = df_daftarNamaIslamL.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df_daftarNamaKristenL = df_daftarNamaKristenL.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df_daftarNamaIslamP = df_daftarNamaIslamP.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df_daftarNamaKristenP = df_daftarNamaKristenP.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df_nama_baru = tabel
        df_nama_baru = df_nama_baru.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df_laki_baru = pd.DataFrame()
        series_baru = df_nama_baru['Nama Lengkap'][df_nama_baru['Jenis Kelamin'] == 'l']
        df_laki_baru['Nama Lengkap'] = series_baru.str.lower()
        namaDepan_baru = list(df_laki_baru['Nama Lengkap'].str.split().str[0])
        df_namaDepan_laki_baru = pd.DataFrame()
        df_namaDepan_laki_baru['nama depan'] = namaDepan_baru
        df_namaDepan_laki_baru = df_namaDepan_laki_baru.drop_duplicates()
        
        hasil_laki_baru = pd.DataFrame()
        cari_distance(df_namaDepan_laki_baru, df_daftarNamaKristenL, 'kristen', hasil_laki_baru)
        cari_distance(df_namaDepan_laki_baru, df_daftarNamaIslamL, 'islam', hasil_laki_baru)
        cari_distance(df_namaDepan_laki_baru, df_daftarNamaSunda, 'sunda', hasil_laki_baru)
        
        fitur_df = hasil_laki_baru[['distance islam','distance kristen','distance sunda']]
        #Buat array Numpy utk features
        fitur_np = np.array(fitur_df.values)
        
        fitur_df.shape
        
        #Lakukan prediksi
        prediksi = loaded_model_nb.predict(fitur_np).tolist()
        df_nama_baru['Etnis'] = None
        for i, row in df_nama_baru.iterrows():
            if row['Jenis Kelamin'] == 'l':
                df_nama_baru.at[i, 'Etnis'] = prediksi.pop(0)
                    
        df_perempuan_baru = pd.DataFrame()
        series_baru = df_nama_baru['Nama Lengkap'][df_nama_baru['Jenis Kelamin'] == 'p']
        df_perempuan_baru['Nama Lengkap'] = series_baru.str.lower()
        namaDepan_baru = list(df_perempuan_baru['Nama Lengkap'].str.split().str[0])
        df_namaDepan_perempuan_baru = pd.DataFrame()
        df_namaDepan_perempuan_baru['nama depan'] = namaDepan_baru
        df_namaDepan_perempuan_baru = df_namaDepan_perempuan_baru.drop_duplicates()
        
        hasil_perempuan_df = pd.DataFrame()
        cari_distance(df_namaDepan_perempuan_baru, df_daftarNamaIslamP, 'islam', hasil_perempuan_df)
        cari_distance(df_namaDepan_perempuan_baru, df_daftarNamaKristenP, 'kristen', hasil_perempuan_df)
        cari_distance(df_namaDepan_perempuan_baru, df_daftarNamaSunda, 'sunda', hasil_perempuan_df)
        
        fitur_df2 = hasil_perempuan_df[['distance islam','distance kristen','distance sunda']]
        #Buat array Numpy utk features
        fitur_np2 = np.array(fitur_df2.values)
        
        fitur_df2.shape
        
        #Lakukan prediksi
        prediksi2 = loaded_model_nb_perempuan.predict(fitur_np2).tolist()
        for i, row in df_nama_baru.iterrows():
            if row['Jenis Kelamin'] == 'p':
                df_nama_baru.at[i, 'Etnis'] = prediksi2.pop(0)
                
        etnissr = []
        for etnis in df_nama_baru['Etnis']:
            if etnis == 'non_sunda':
                etnissr.append('Non Sunda')
            elif etnis == 'sunda':
                etnissr.append('Sunda')
        tabel['Etnis'] = etnissr 
        return tabel
        
        
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
        # if distance_tp[0][0] == 1000000:
        #     distance_list = list(distance_tp)
        #     distance_list[0] = list(distance_list[0])
        #     distance_list[0][0] = distance_list[0][0]*0
        #     distance_list[0] = tuple(distance_list[0])
        #     distance_tp = tuple(distance_list)
 
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
        
    def reset_table(self):
    # Clear existing items in the Treeview widget
        for item in self.table.get_children():
            self.table.delete(item)
    
    def reset_string_var(self):
    # Set a new value for the StringVar
        self.input_string.set("")
        entry1 = ttk.Entry(self.page1, textvariable=self.input_string)
        
        
if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    