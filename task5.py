from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# task 1
file_name = 'G19.01_B7.ms_CH3OHvt1_292.51744GHz.image.pbcor_cutout_Hz_K.fits'

the_data = fits.open(file_name)[0].data
the_header = fits.open(file_name)[0].header

shape_of_data = np.shape(the_data)
print(shape_of_data)
number_of_z_channels, number_of_y_pixels, number_of_x_pixels = shape_of_data
spectrum = the_data[0:, number_of_y_pixels-1, number_of_x_pixels-1]
spectrumBell = the_data[0:, 49, 50]

# fig = plt.figure()
# plt.plot(spectrum)
# plt.show()

pixel_coordinates = [(0, 0), (number_of_x_pixels//2, number_of_y_pixels//2), (number_of_x_pixels-1, number_of_y_pixels-1)]
for x, y in pixel_coordinates:
    spectrum = the_data[0:, y, x]
    plt.figure(figsize=(10, 6))
    plt.plot(spectrum, label=f"Pixel ({x}, {y})")
    plt.xlabel('Channel Number')
    plt.ylabel('Intensity')
    plt.title(f'Spectrum at Pixel ({x}, {y})')
    plt.legend()
    plt.show()



# Task 2
CRVAL3 = the_header['CRVAL3']
# this is the frequency in Hz of the first z channel of the data
CDELT3 = the_header['CDELT3']
# this is the frequency in Hz of the separation of each channel
# of the spectrum in frequency space

channelFrequencies = np.zeros(number_of_z_channels)
for i in range(1, number_of_z_channels):
    channelFrequencies[i - 1] = CRVAL3 + (i - 1) * CDELT3

print("Channel frequencies:", channelFrequencies)


# Task 3
FRest = 292.51744e9 # in hertz
c = 3e8  # in meters per second ms^-1

# for loop
velocities = np.zeros(number_of_z_channels)
for i in range(number_of_z_channels):
    velocities[i] = (FRest - (CRVAL3 + i * CDELT3)) * c / FRest / 1000
print("Channel Velocities:", velocities)

#plot
plt.plot(velocities, the_data[:, 49, 50])
plt.ylabel("Brightness temperature of the spectrum (K)")
plt.xlabel("Channel Velocities (Km/s)")
plt.show()


# Task 4
A = 1
mu = 0
sigma = 1
def gaussian(x, A, mu, sigma):
    gaussian_value = A * np.exp(-((x - mu) ** 2) / (2 * (sigma**2)))
    return gaussian_value

initial_guesses = [7, 61, 0.5]

popt, pcov = curve_fit(gaussian, velocities, the_data[:, 49, 50], p0=initial_guesses)

# plot the gauss curve
plt.figure(figsize=(10, 6))
plt.plot(velocities, the_data[:, 49, 50], label="Observed Spectrum")
plt.plot(velocities, gaussian(velocities, *popt), label="Fitted Gaussian", linestyle='--')
plt.xlabel('Velocity (km/s)')
plt.ylabel('Intensity')
plt.title('Spectrum and Gaussian Fit')
plt.legend()
plt.show()

# Task 5
initial_guesses = [7, 61, 0.5]
pixel_coords = pd.read_csv('pixel_coordinates.csv')
x_coords, y_coords = pixel_coords['x pixel coordinate'], pixel_coords['y pixel coordinate']
numyMax, numxMax = 140, 140
velocity_map = np.full((numxMax, numyMax), np.nan)

for i in range(0, numxMax):
    velocities = np.zeros(number_of_z_channels-1)
    TheSpectrum = the_data[1:, y_coords[i], x_coords[i]]
    for j in range(1, number_of_z_channels-1):
        velocities[j-1] = (FRest-(CRVAL3 + (j-1)*CDELT3))*(c/(FRest*1e3))
    popt, pcov = curve_fit(gaussian, velocities, TheSpectrum, p0=initial_guesses)
    velocity_map[y_coords[i], x_coords[i]] = popt[1]

fig = plt.figure()
c = plt.imshow(velocity_map, origin='lower', cmap='RdBu_r')
plt.xlim([40,60])
plt.ylim([35,55])
plt.colorbar(label='Velocity (km/s)')
plt.xlabel('X Pixel Coordinate')
plt.ylabel('Y Pixel Coordinate')
plt.title('Velocity Map')
plt.show()