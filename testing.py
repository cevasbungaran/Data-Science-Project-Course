# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 19:15:43 2023

@author: Madmierja Legacy
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import pickle
import Levenshtein
import numpy as np

root = tk.Tk()
root.attributes('-fullscreen', True)  # Tetapkan ukuran awal untuk root
root.title('PRODS')

def hide_indicators():
    etnis_indicate.config(bg='#097969')
    excel_indicate.config(bg='#097969')

def delete_pages():
    for frame in main_frame.winfo_children():
        frame.destroy()
        
def indicate(lb, page):
    hide_indicators()
    lb.config(bg='#5F9EA0')
    delete_pages()
    page()
    

def nama_page():
    nama_frame = tk.Frame(main_frame)
    nama_frame.pack()
    
    keterangan = tk.Label(nama_frame, text="Prediksi Etnis", font=('Arial Black', 50), fg='#097969', anchor='w')
    keterangan.config(width=200, height=1)
    keterangan.pack(pady=70, padx=40)
    
    label1 = tk.Label(nama_frame, text="Masukkan Nama Lengkap Tanpa Gelar", fg='#097969', anchor='w',
                      font=('Arial Black', 12))
    label1.config(width=200, height=1)
    label1.pack(padx=40)
    
    def on_entry_click(event):
        if input_string.get() == 'Masukkan nama di sini...':
            input_string.set('')
            entry1.config(foreground='black')

    def on_focusout(event):
        if not input_string.get():
            input_string.set('Masukkan nama di sini...')
            entry1.config(foreground='#097969')
    
    def process_string():
        input_text = input_string.get()
        output_text = input_text.lower()
        gender_choice = gender_var.get()
        output_string_text = prediksiNama(input_text, gender_choice)
        output_nama_text = 'Nama: \n' + input_text
        output_nama.set(output_nama_text)
        output_string_text = 'Etnis: \n' + output_string_text
        output_string.set(output_string_text)
        
    
    input_string = tk.StringVar()
    input_string.set('Masukkan nama di sini...')
    entry1 = tk.Entry(nama_frame, textvariable=input_string, fg='#097969')
    entry1.bind('<FocusIn>', on_entry_click)
    entry1.bind('<FocusOut>', on_focusout)
    entry1.config(width=200, font=('Arial', 12), justify="left")  # Set justify ke "left" untuk anchor teks ke kiri
    entry1.place(x=40, y=270, width=350, height=120)
    
    gender_label = tk.Label(nama_frame, text="Jenis Kelamin:", fg='#097969', anchor='w',
                      font=('Arial Black', 12))
    gender_label.config(width=200, height=1)
    gender_label.pack(padx=40, pady=120)
    
    gender_var = tk.StringVar()
    # Create radio buttons for each gender
    gender_radio_female = tk.Radiobutton(nama_frame, text="Perempuan", variable=gender_var, value="Perempuan",
                                         font=('Arial', 12), justify="left")
    gender_radio_male = tk.Radiobutton(nama_frame, text="Laki-laki", variable=gender_var, value="Laki-laki",
                                       font=('Arial', 12), justify="left")
    
    # Pack the radio buttons side by side
    gender_radio_female.pack(side="left", padx=10)
    gender_radio_female.config(font=('Arial', 12), justify="left")  # Set justify ke "left" untuk anchor teks ke kiri
    gender_radio_female.place(x=40, y=420, height=50)
    gender_radio_male.pack(side="left", padx=10)
    gender_radio_male.config(font=('Arial', 12), justify="left")  # Set justify ke "left" untuk anchor teks ke kiri
    gender_radio_male.place(x=150, y=420, height=50)
    gender_var.set(None)
    
    submit_button = tk.Button(nama_frame, text="Prediksi", bg='#097969', fg='white', command=process_string)
    submit_button.pack(pady=10)
    submit_button.config(width=200, font=('Arial', 12), justify="left")  # Set justify ke "left" untuk anchor teks ke kiri
    submit_button.place(x=40, y=490, width=200, height=50)
    
    output_label = tk.Label(nama_frame, text="Hasil Prediksi Etnis:",
                            fg='#097969', anchor='w')
    output_label.config(width=200, font=('Arial Black', 12), justify="left")  # Set justify ke "left" untuk anchor teks ke kiri
    output_label.pack(padx=40)
    # output_label.place(x=40, y=550, width=200)
    
    
    
    output_nama= tk.StringVar()
    output_text_nama = tk.Label(nama_frame, textvariable=output_nama)
    output_text_nama.config(textvariable=output_nama, width=200, fg='#097969',font=('Arial Black', 12), anchor='w',justify="left")
    output_text_nama.pack(padx=40, pady=10)
    
    output_string = tk.StringVar()
    output_text = tk.Label(nama_frame, textvariable=output_string)
    output_text.config(textvariable=output_string, width=200, fg='#097969',font=('Arial Black', 12), anchor='w',justify="left")
    output_text.place(y=425, width= 200)
    output_text.pack(padx=40)
    
def excel_page():
    excel_frame = tk.Frame(main_frame)
    excel_frame.pack()
    
    def import_excel():
        #reset_table(self)
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if file_path:
            df = pd.read_excel(file_path)
            prediksiTabel(df)
            for i, row in df.iterrows():
                table.insert("", "end", values=(row["Nama Lengkap"], row["Jenis Kelamin"], row["Etnis"]))
            return table
    
    keterangan_excel = tk.Label(excel_frame, text="Prediksi Etnis", font=('Arial Black', 50), fg='#097969')
    keterangan_excel.pack(pady=70)
    
    import_label = tk.Label(excel_frame, text="Hanya dapat import file Excel (.xlsx) dengan 2 kolom berisi nama lengkap tanpa gelar dan jenis kelamin(l/p):", 
                             font=('Arial Black', 12), foreground='#097969')
    import_label.pack()
    
    import_button = tk.Button(excel_frame, text="Import File", command=import_excel, bg='#097969', fg='white')
    import_button.config(width=15, height=2, font=('Arial', 12))
    import_button.pack(pady=15)
    
    table_label = tk.Label(excel_frame, text="Tabel Hasil Prediksi Etnis:", font=('Arial Black', 12), foreground='#097969')
    table_label.pack(pady=5)
    
    table_frame = ttk.Frame(excel_frame)
    table_frame.pack(pady=10)
    
    table = ttk.Treeview(table_frame, columns=("Nama Lengkap", "Jenis Kelamin", "Etnis"), height=20)
    table.heading("#1", text="Nama Lengkap")
    table.heading("#2", text="Jenis Kelamin")
    table.heading("#3", text="Etnis")
    
    
    # Add a vertical scrollbar
    yscrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
    table.configure(yscrollcommand=yscrollbar.set)
    
    # Pack the Treeview and Scrollbar
    table.pack(side="left", fill="both", expand=True)
    yscrollbar.pack(side="right", fill="y")
    
    return table



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

    
    def prediksiTabel(tabel):
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
        df_baru = pd.read_excel('cobagui.xlsx')
        df_baru = df_baru.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        df_laki_baru = pd.DataFrame()
        series_baru = df_baru['Nama Lengkap'][df_baru['Jenis Kelamin'] == 'l']
        df_laki_baru['Nama Lengkap'] = series_baru.str.lower()
        namaDepan_baru = list(df_laki_baru['Nama Lengkap'].str.split().str[0])
        df_namaDepan_laki_baru = pd.DataFrame()
        df_namaDepan_laki_baru['nama depan'] = namaDepan_baru
        
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
        df_baru['Etnis'] = None
        for i, row in df_baru.iterrows():
            if row['Jenis Kelamin'] == 'l':
                df_baru.at[i, 'Etnis'] = prediksi.pop(0)
                    
        df_perempuan_baru = pd.DataFrame()
        series_baru = df_baru['Nama Lengkap'][df_baru['Jenis Kelamin'] == 'p']
        df_perempuan_baru['Nama Lengkap'] = series_baru.str.lower()
        namaDepan_baru = list(df_perempuan_baru['Nama Lengkap'].str.split().str[0])
        df_namaDepan_perempuan_baru = pd.DataFrame()
        df_namaDepan_perempuan_baru['nama depan'] = namaDepan_baru
        
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
        for i, row in df_baru.iterrows():
            if row['Jenis Kelamin'] == 'p':
                df_baru.at[i, 'Etnis'] = prediksi2.pop(0)
                
        tabel['Etnis'] = tampilanEtnis(df_baru['Etnis'])
        return tabel
    
    def tampilanEtnis(tabel):
        etnissr = []
        for etnis in tabel:
            if etnis == 'non_sunda':
                etnissr.append('Non Sunda')
            elif etnis == 'sunda':
                etnissr.append('Sunda')
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






# NAV BAR SEBELAH KIRI
    
options_frame = tk.Frame(root, bg='#097969',highlightbackground='#5F9EA0',
                      highlightthickness=1)
options_frame.pack(side=tk.LEFT, fill=tk.Y)  # Gunakan fill=tk.Y untuk memastikan options_frame mengisi tinggi
options_frame.pack_propagate(False)
options_frame.configure(width=450)  # Hanya set width, karena height sudah diatur oleh root

main_frame = tk.Frame(root, highlightbackground='#5F9EA0',
                      highlightthickness=1)
main_frame.pack(fill=tk.BOTH, expand=True)  # Perubahan ini membuat main_frame mengisi seluruh ruang yang tersedia

#judul
prediksiLabel = tk.Label(options_frame, text="PRODS 2", font=('Arial Black', 50), bg='#097969', fg='white')
prediksiLabel.place(x=10, y=20)

    
#etnis
prediksiEtnis = tk.Button(options_frame, text='Prediksi Etnis', font=('Bold', 30)
                          , fg='white', bd=0, bg='#097969'
                          , command=lambda: indicate(etnis_indicate, nama_page))
prediksiEtnis.place(x=10, y=150)

etnis_indicate = tk.Label(options_frame,text='', bg='#097969')
etnis_indicate.place(x=3, y=150, width=5, height=75)

#excel
prediksiExcel = tk.Button(options_frame, 
                          text='Prediksi Etnis\ndari File Excel', 
                          font=('Bold', 30), anchor='w', 
                          fg='white', bd=0, bg='#097969'
                          , command=lambda: indicate(excel_indicate, excel_page))
prediksiExcel.place(x=20, y=240)

excel_indicate = tk.Label(options_frame,text='', bg='#097969')
excel_indicate.place(x=3, y=260, width=5, height=75)


root.mainloop()
