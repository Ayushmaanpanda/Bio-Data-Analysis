from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load the data file
mat_data = loadmat('s0042lrem.mat')

data = mat_data['val']  
print("Data Shape:", data.shape)
print(data)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for i in range(5):  # Plot only the first 5 rows
    plt.plot(data[i, :], label=f'Patient {i+1}')

plt.title("Plot of First 5 Patients Over 10 sec")
plt.xlabel("Time [ms]")
plt.ylabel("ECG Value [mV]")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


import numpy as np

# Calculate the mean across rows 
mean_values = np.mean(data, axis=0)

# Plot the mean values
plt.figure(figsize=(12, 6))
plt.plot(mean_values, color='blue', label='Mean Across Data')
plt.title("Mean Value Across All Patient data Over 10 sec")
plt.xlabel("Time [ms]")
plt.ylabel("Mean Value [mV]")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(data[0, :], label='Patient 1')
plt.plot(data[1, :], label='Patient 2')
plt.title("Comparison of Patient 1 and Patient 2 Over 10 sec")
plt.xlabel("Time [ms]")
plt.ylabel("Value [mV]")
plt.legend()
plt.grid(True)
plt.show()


import seaborn as sns
# Plot heatmap
plt.figure(figsize=(15, 6))
sns.heatmap(data, cmap="viridis", cbar=True)
plt.title("Heatmap of Patients Over 10 sec")
plt.xlabel("Time [ms]")
plt.ylabel("Row")
plt.show()

correlation_matrix = np.corrcoef(data)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True, fmt=".2f")
plt.title("Correlation Matrix Between Rows")
plt.xlabel("Row Index")
plt.ylabel("Row Index")
plt.show()

from scipy.fft import fft, fftfreq
# Fourier Transform to the first row 
N = data.shape[0]  # N = 10000 
T = 1.0  

yf = fft(data[0, :]) # Perform FFT
xf = fftfreq(N, T)[:N//2]

# Plot FFT result
plt.figure(figsize=(10, 5))
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.title("Frequency Analysis (Fourier Transform) of Patient 1 Data")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Plot of all Signals 
plt.figure(figsize=(12, 8))
for i in range(data.shape[0]):
    plt.plot(data[i, :], label=f'Patient {i+1}')
plt.xlabel("Time [ms]")
plt.ylabel("Value [mV]")
plt.title("Signal Plot for Each Patient")
plt.legend(loc="upper right", ncol=3)
plt.grid(True)
plt.show()

# Statistical Analysis
print("Statistical Summary:")
mean_values = np.mean(data, axis=1)
std_dev_values = np.std(data, axis=1)
min_values = np.min(data, axis=1)
max_values = np.max(data, axis=1)

for i in range(data.shape[0]):
    print(f"Patient {i+1}: Mean = {mean_values[i]}, Std Dev = {std_dev_values[i]}, Min = {min_values[i]}, Max = {max_values[i]}")


