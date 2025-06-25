import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

def ramsey_reset_test(model, X, y):
    """
    Melakukan uji linearitas Ramsey RESET secara manual.
    Hipotesis Nol (H0): Model linear.
    """
    n = X.shape[0]
    k = X.shape[1]
    
    # 1. Dapatkan R-squared dari model original
    r_squared_orig = model.score(X, y)
    
    # 2. Dapatkan nilai prediksi dari model original
    y_pred = model.predict(X)
    
    # 3. Buat variabel baru dari kuadrat nilai prediksi
    X_reset = X.copy()
    X_reset['y_pred_sq'] = y_pred**2
    
    # 4. Fit model baru (RESET model)
    reset_model = LinearRegression()
    reset_model.fit(X_reset, y)
    r_squared_reset = reset_model.score(X_reset, y)
    
    # 5. Hitung F-statistic
    q = X_reset.shape[1] - k # Jumlah variabel baru (dalam kasus ini, 1)
    f_stat = ((r_squared_reset - r_squared_orig) / q) / ((1 - r_squared_reset) / (n - k - q - 1))
    
    # 6. Hitung p-value dari distribusi F
    p_value = stats.f.sf(f_stat, dfn=q, dfd=n - k - q - 1)
    
    return f_stat, p_value


# --- 1. MEMUAT DATA DAN MENGAMBIL SAMPEL ---
try:
    df = pd.read_csv('datas/student_management_dataset.csv')
    df_sample = df.sample(n=150, random_state=42)
    print("Pengambilan sampel 150 data berhasil.\n")
except FileNotFoundError:
    print("File 'student_management_dataset.csv' tidak ditemukan. Mohon unggah file terlebih dahulu.")
    exit()

# Mendefinisikan variabel
Y = df_sample['GPA']
X = df_sample[['Study_Hours_per_Week', 'Attendance_Rate']]
# jadi kita TIDAK perlu menggunakan sm.add_constant(X) untuk model utama.

# --- 2. MEMBUAT TABEL STATISTIK DESKRIPTIF ---
print("--- Tabel 1: Statistik Deskriptif ---")
descriptive_stats = df_sample[['GPA', 'Study_Hours_per_Week', 'Attendance_Rate']].describe()
print(descriptive_stats)
print("\n" + "="*50 + "\n")


# --- 3. MEMBUAT TABEL HASIL UJI KORELASI ---
print("--- Tabel 2: Matriks Korelasi Pearson ---")
correlation_matrix = df_sample[['GPA', 'Study_Hours_per_Week', 'Attendance_Rate']].corr()
print(correlation_matrix)
print("\n" + "="*50 + "\n")


# --- 4. MEMBUAT MODEL REGRESI DAN UJI PRASYARAT ---
print("--- Tabel 3: Ringkasan Model Regresi dan Uji Prasyarat ---")

# Membuat dan melatih model
model = LinearRegression()
model.fit(X, Y)

# Menampilkan hasil utama dari model regresi
print("--- Hasil Model Regresi ---")
r_squared = model.score(X, Y)
intercept = model.intercept_
coefficients = model.coef_

print(f"Intercept (Konstanta)      : {intercept:.4f}")
print(f"Koefisien [Jam Belajar]    : {coefficients[0]:.4f}")
print(f"Koefisien [Tingkat Hadir]  : {coefficients[1]:.4f}")
print(f"R-squared                  : {r_squared:.4f}")
print("\n--- Hasil Uji Prasyarat/Asumsi ---")

# --- 2. MEMBUAT MODEL DENGAN SCIKIT-LEARN ---
# Scikit-learn secara otomatis menangani konstanta (intercept)
model = LinearRegression()
model.fit(X, Y)

# --- 3. PERHITUNGAN MANUAL UNTUK STATISTIK REGRESI ---
# Menambahkan konstanta ke X secara manual untuk perhitungan matriks
X_with_const = np.c_[np.ones(X.shape[0]), X]

# Menghitung prediksi dan residual
y_pred = model.predict(X)
residuals = Y - y_pred

# Menghitung parameter dasar
n = len(Y)  # Jumlah observasi
k = X.shape[1] # Jumlah variabel independen
dof_residuals = n - k - 1

# Menghitung R-squared dan Adjusted R-squared
ss_total = np.sum((Y - np.mean(Y))**2)
ss_residuals = np.sum(residuals**2)
r_squared = 1 - (ss_residuals / ss_total)
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / dof_residuals

# Menghitung Standard Error of the Regression (Residual Standard Error)
s_squared = ss_residuals / dof_residuals
s = np.sqrt(s_squared)

# Menghitung Standard Error, t-statistic, dan p-value untuk setiap koefisien
# Ini adalah bagian yang paling rumit, menggunakan aljabar linear
xtx_inv = np.linalg.inv(X_with_const.T @ X_with_const)
se_coeffs = np.sqrt(np.diag(xtx_inv) * s_squared)

# Menggabungkan intercept dan koefisien lainnya
coeffs = np.concatenate(([model.intercept_], model.coef_))

# Menghitung t-stats dan p-values
t_stats = coeffs / se_coeffs
p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=dof_residuals)) for t in t_stats]

# Menghitung F-statistic dan p-value-nya
ss_regression = ss_total - ss_residuals
ms_regression = ss_regression / k
ms_residuals = ss_residuals / dof_residuals
f_statistic = ms_regression / ms_residuals
f_p_value = 1 - stats.f.cdf(f_statistic, k, dof_residuals)


# --- 4. MEMBUAT TABEL RINGKASAN ---
print("--- Ringkasan Model Regresi (dihitung dengan Scikit-learn + Manual) ---")
print("="*60)
print(f"Variabel Dependen: GPA")
print(f"R-squared:         {r_squared:.4f}")
print(f"Adj. R-squared:    {adj_r_squared:.4f}")
print(f"F-statistic:       {f_statistic:.4f}")
print(f"Prob (F-statistic): {f_p_value:.4f}")
print(f"Jumlah Observasi:  {n}")
print("="*60)

# Membuat DataFrame untuk tabel koefisien
summary_table = pd.DataFrame({
    'Variabel': ['const'] + list(X.columns),
    'coef': coeffs,
    'std err': se_coeffs,
    't': t_stats,
    'P>|t|': p_values
})
print(summary_table.to_string(index=False))
print("="*60)

# Menghitung residual (nilai aktual - nilai prediksi) untuk pengujian
residuals = Y - model.predict(X)

# 1. Uji Linearitas
reset_f, reset_p = ramsey_reset_test(model, X, Y)
print("\n1. Uji Linearitas (Ramsey RESET Test):")
print(f"   - F-statistic = {reset_f:.3f}")
print(f"   - P-value = {reset_p:.3f}")
if reset_p > 0.05:
    print("   - Kesimpulan: P-value > 0.05, asumsi LINEARITAS terpenuhi.")
else:
    print("   - Kesimpulan: P-value < 0.05, hubungan mungkin NON-LINEAR.")
print("\n" + "="*50 + "\n")

# 2. Uji Normalitas Residual (Jarque-Bera)
jb_stat, jb_prob = stats.jarque_bera(residuals)
print(f"\n2. Uji Normalitas Residual (Jarque-Bera):")
print(f"   - Statistik Jarque-Bera = {jb_stat:.3f}")
print(f"   - Prob(JB) = {jb_prob:.3f}")
if jb_prob > 0.05:
    print("   - Kesimpulan: Prob(JB) > 0.05, residual terdistribusi NORMAL.")
else:
    print("   - Kesimpulan: Prob(JB) < 0.05, residual TIDAK terdistribusi normal.")

# 3. Uji Multikolinearitas (Variance Inflation Factor - VIF)
# Untuk VIF, perlu menambahkan konstanta secara manual ke data X
X_vif = pd.DataFrame(np.c_[np.ones(X.shape[0]), X], columns=['const'] + list(X.columns))
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns[1:] # Hanya tampilkan variabel independen
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])]
print("\n3. Uji Multikolinearitas (Variance Inflation Factor - VIF):")
print(vif_data)
print("   - Aturan umum: Nilai VIF < 10 menunjukkan tidak ada multikolinearitas yang serius.")

# 4. Uji Heteroskedastisitas (Breusch-Pagan)
# Fungsi het_breuschpagan membutuhkan residual dan data X (dengan konstanta)
X_bp = pd.DataFrame(np.c_[np.ones(X.shape[0]), X], columns=['const'] + list(X.columns))
bp_test = het_breuschpagan(residuals, X_bp)
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
bp_result = dict(zip(labels, bp_test))
print("\n4. Uji Heteroskedastisitas (Breusch-Pagan):")
print(f"   - P-value dari uji Breusch-Pagan = {bp_result['p-value']:.3f}")
if bp_result['p-value'] > 0.05:
    print("   - Kesimpulan: P-value > 0.05, tidak terdapat heteroskedastisitas (HOMOSKEDASTIS).")
else:
    print("   - Kesimpulan: P-value < 0.05, TERDAPAT heteroskedastisitas.")
print("\n" + "="*50 + "\n")

# 5. Uji Autokorelasi (Durbin-Watson)
dw_stat = durbin_watson(residuals)
print(f"1. Uji Autokorelasi (Durbin-Watson): Statistik D-W adalah {dw_stat:.3f}")
print("   - Aturan umum: Nilai antara 1.5 dan 2.5 menunjukkan tidak ada autokorelasi.")



# --- 5. MEMBUAT VISUALISASI DATA ---
print("--- Gambar dan Plot Visualisasi Data ---")
sns.set_theme(style="whitegrid")

# Plot 1: Histogram untuk setiap variabel (Tidak ada perubahan)
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.histplot(df_sample['GPA'], kde=True, color='blue')
plt.title('Distribusi GPA')
plt.subplot(1, 3, 2)
sns.histplot(df_sample['Study_Hours_per_Week'], kde=True, color='green')
plt.title('Distribusi Jam Belajar per Minggu')
plt.subplot(1, 3, 3)
sns.histplot(df_sample['Attendance_Rate'], kde=True, color='red')
plt.title('Distribusi Tingkat Kehadiran')
plt.tight_layout()
plt.savefig('plot1_histograms.png')
print("Plot 1 (Histogram) telah disimpan sebagai 'plot1_histograms.png'")

# Plot 2: Scatter plot untuk melihat hubungan variabel (Tidak ada perubahan)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.regplot(x='Study_Hours_per_Week', y='GPA', data=df_sample, scatter_kws={'alpha':0.6})
plt.title('Hubungan Jam Belajar vs GPA')
plt.subplot(1, 2, 2)
sns.regplot(x='Attendance_Rate', y='GPA', data=df_sample, scatter_kws={'alpha':0.6})
plt.title('Hubungan Tingkat Kehadiran vs GPA')
plt.tight_layout()
plt.savefig('plot2_scatterplots.png')
print("Plot 2 (Scatter Plot) telah disimpan sebagai 'plot2_scatterplots.png'")

# Plot 3: Residual Plot untuk Uji Linieritas dan Heteroskedastisitas
plt.figure(figsize=(8, 6))
fitted_vals = model.predict(X)
# 'residuals' sudah dihitung sebelumnya
sns.residplot(x=fitted_vals, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.title('Plot Residual vs Fitted Values')
plt.xlabel('Fitted (Predicted) Values')
plt.ylabel('Residuals')
plt.savefig('plot3_residual_plot.png')
print("Plot 3 (Residual Plot) telah disimpan sebagai 'plot3_residual_plot.png'")
print("\n--- Penjelasan Uji Linieritas dari Plot 3 ---")
print("   - Uji Linieritas: Perhatikan garis merah pada 'Plot Residual'. Jika garis merah tersebut cenderung lurus dan horizontal di sekitar angka 0, maka asumsi linieritas terpenuhi.")
print("   - Uji Heteroskedastisitas (Visual): Jika titik-titik residual menyebar secara acak tanpa membentuk pola tertentu (seperti corong), maka secara visual tidak ada masalah heteroskedastisitas.")