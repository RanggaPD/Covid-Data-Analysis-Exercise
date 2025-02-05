import pandas as pd
import matplotlib.pyplot as plt

# membaca data
df = pd.read_excel('/content/covid_kota_bandung.xlsx')

# mengubah kolom tanggal menjadi tipe data datetime
df['tanggal'] = pd.to_datetime(df['tanggal'], format='%Y-%m-%d')

# menambahkan kolom hari dengan menghitung selisih antara tanggal awal dan tanggal pada setiap baris
start_date = df['tanggal'].min()
df['hari'] = (df['tanggal'] - start_date).dt.days

df.head()

df = df.rename(columns={'konfirmasi_aktif': 'kasus_aktif'})

# memilih kolom hari dan kasus aktif untuk melakukan prediksi
data = df[['hari', 'kasus_aktif']]

data.head()

# memisahkan data menjadi data latih dan data uji
train_data = data.iloc[:len(data) - 7]
test_data = data.iloc[len(data) - 7:]

# membuat model prediksi menggunakan regresi linier
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train_data[['hari']], train_data['kasus_aktif'])

# membuat prediksi untuk data uji
predictions = model.predict(test_data[['hari']])
test_data['prediksi'] = predictions

# membuat grafik garis untuk menampilkan tren kasus aktif dan prediksi
sns.set_style('darkgrid')
plt.figure(figsize=(10, 6))
plt.plot(df['tanggal'], df['kasus_aktif'], label='Kasus Aktif', color='red')
plt.plot(test_data['tanggal'], test_data['prediksi'], label='Prediksi Kasus Aktif')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Kasus')
plt.title('Prediksi Kasus Aktif COVID-19')
plt.legend()
plt.show()
