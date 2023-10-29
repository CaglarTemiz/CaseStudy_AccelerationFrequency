#Gerekli modüller
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import re

def yeni_veritabani(file, axis):
  
  #Eski veritabanını okuma
  dataset = pd.read_csv(file, sep= ';', header = 0)

  #Axis'e göre veri filtreleme
  #X_değerleri
  new_col_list = []

  for col_name in dataset.columns:
    check = re.search(axis,col_name)
    if(check != None):
     new_col_list.append(col_name)

  new_dataset = dataset[new_col_list]

  return new_dataset

# İvmeölçerlerin Hakim Frekansı'nı bulma
def frekans_imveolcer_listeleme(veritabani):
   
   #0'dan 180'e saniyede 200 örnek ise...
   time = np.linspace(0,180,36000)
   #Frekans verisi
   frequency = np.fft.fftfreq(len(time), d = 1/200)
  
   #İvmeölçer ve hakim frekans verilerini dictionary olarak kaydetme 
   dominantFreq_vs_accelerometer = dict()

   for col_name in veritabani.columns:
    #Ivme verilerinin fourier transform sayesinde hakim frekansını bulur
    acceleration_data = veritabani[col_name].values
    acceleration_fft = np.fft.fft(acceleration_data)
    amplitude = np.abs(acceleration_fft)
    dominant_frequency = abs(frequency[np.argmax(amplitude)])
    print("The dominant frequency of {} is {} Hz.".format(col_name, str(round(dominant_frequency, 2))))
    dominantFreq_vs_accelerometer.update({ col_name: dominant_frequency})
  
   return dominantFreq_vs_accelerometer

#Verilen sayıya en yakın sayıyı bulur 
def closest(lst, K):
     
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

#Sönümleme Fonksiyonu
def logaritmik_azalma(x, wn, damping):
  return np.exp(-x*wn*damping)

#Filtrelenmiş veritabanı üzerinde işlem yaparak istenilen çıktıyı verir
def filtrelenmis_veri(veritabani,dominantFreq_vs_accelerometer,hakim_frekans, n_degeri):

  hakim_frekans = closest(list(dominantFreq_vs_accelerometer.values()), hakim_frekans)
  
  for name in list(dominantFreq_vs_accelerometer.keys()):
    if dominantFreq_vs_accelerometer[name] == hakim_frekans:
      accelerometer_name = name
      break

  accelerations = veritabani[accelerometer_name].values
  time = np.linspace(0,180,36000)

  # Butterworth low-pass filtre tasarımı
  order = 4  # Filtrenin keskinliği
  cutoff_frequency = hakim_frekans/0.90 # 
  nyquist_frequency = 0.5 / (time[1] - time[0])
  normalized_cutoff = cutoff_frequency / nyquist_frequency
  b, a = butter(order, normalized_cutoff, btype='low')
  filtered_signal = filtfilt(b, a, accelerations)

  # Filtrelenmiş verilerin grafiğe dökümü
  fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
  line1 = ax.plot(time, abs(filtered_signal), color = "blue")

  #Filtrelenmiş sinyalin tepe noktalarını bulma
  peaks_signal, _ = find_peaks(filtered_signal, height = 0)
  time_peaks = time[peaks_signal]
  filtered_peaks = filtered_signal[peaks_signal]

  #Tepe noktasının %90 üzerinden yapılan işlemler
  acc90 = max(filtered_peaks)*0.9
  idx90 = np.where(filtered_peaks == closest(filtered_peaks,acc90))[0][0]
  delta = np.log(acc90/filtered_peaks[idx90 + n_degeri -1])
  damping = np.sqrt((delta**2)/(delta**2 + (2*np.pi)**2))

  #Logaritmik fonksiyonu fit ettirme
  popt, pcov = curve_fit(logaritmik_azalma,time_peaks,filtered_peaks)
  
  #Grafik işlemleri
  label_line = "Damping: " + str(100*round(damping,4)) + "%"
  line2 = ax.plot(time_peaks[(idx90 + n_degeri -1):],logaritmik_azalma(time_peaks[(idx90 + n_degeri -1):], popt[0], damping), color = "black", label = label_line)
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Filtered Acceleration (g)")
  title = "Filtered Accelerations from " + accelerometer_name
  ax.set_title(title)
  ax.legend()
  
  #Figürü kaydetme
  fig.savefig("FilteredAcceleration_vs_Time.png")
  fig.show()

  return fig

#######################################################################################################
# SCRIPT KODU

#Dosya konumu vev ivmeölçer bilgisiyle veritabanını hazırlar
veri_t = yeni_veritabani("C:/Users/Hp/Desktop/TDG/log-test1_200sps.csv",'x')

#Hakim frekans ve ivmeölçerleri listeler
List_fvsA= frekans_imveolcer_listeleme(veri_t)

#Ana çıktı olan figürü ve sönüm oranını verir
fig = filtrelenmis_veri(dominantFreq_vs_accelerometer= List_fvsA, veritabani= veri_t, hakim_frekans = 2.5, n_degeri = 5)






