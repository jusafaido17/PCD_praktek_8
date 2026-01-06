import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- KONFIGURASI UTAMA ---
FILENAME = 'uburubur.JPG'
# -------------------------

def segmentasi_pola_warna(filename):
    # 1. Membaca citra asli (BGR)
    img_bgr = cv2.imread(filename)
    if img_bgr is None:
        print(f"❌ ERROR: File '{filename}' tidak ditemukan.")
        return

    # Konversi ke RGB untuk tampilan Matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Konversi ruang warna dari RGB ke HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Memisahkan komponen H, S, dan V
    # Di OpenCV: H(0-179), S(0-255), V(0-255)
    H, S, V = cv2.split(img_hsv)
    
    # 3. Kuantisasi Warna (Sesuai logika IF-ELSE di modul)
    # Kita membuat salinan untuk diproses
    H_new = H.copy()
    
    # Catatan: Modul menggunakan skala 0-255 untuk Hue. 
    # Di OpenCV Hue maksimal 179. Maka 234/255 di MATLAB setara ~165 di OpenCV.
    
    # Logika segmentasi warna ungu/magenta (berdasarkan nilai Hue 234 di modul)
    # H_aksen akan menjadi mask biner
    # Kita cari nilai Hue yang mendekati 234 (skala 255) atau 165 (skala 180)
    target_hue = int(234 * 179 / 255)
    tolerance = 10 # Toleransi warna
    
    mask = cv2.inRange(H, target_hue - tolerance, target_hue + tolerance)
    
    # 4. Membersihkan latar belakang (Background jadi Putih)
    # Sesuai modul: R(~H_aksen) = 255; G(~H_aksen) = 255; B(~H_aksen) = 255;
    res_rgb = img_rgb.copy()
    res_rgb[mask == 0] = [255, 255, 255] # Mengubah selain warna target menjadi putih
    
    # --- VISUALISASI ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Praktek 8: Segmentasi Pola Warna (HSV)", fontsize=16)

    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Citra Asli (RGB)")
    axes[0].axis('off')

    # Menampilkan komponen Hue (yang digunakan untuk segmentasi)
    axes[1].imshow(H, cmap='hsv')
    axes[1].set_title("2. Komponen Hue")
    axes[1].axis('off')

    axes[2].imshow(res_rgb)
    axes[2].set_title("3. Hasil Segmentasi Warna")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# Jalankan fungsi
segmentasi_pola_warna(FILENAME)