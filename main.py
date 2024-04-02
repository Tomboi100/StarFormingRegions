from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# task 1
#path = 'Users/Tommy/PycharmProjects/StarFormingRegions'
file_name = 'G19.01_B7.ms_CH3OHvt1_292.51744GHz.image.pbcor_cutout_Hz_K.fits'

the_data = fits.open(file_name)[0].data
the_header = fits.open(file_name)[0].header

shape_of_data = np.shape(the_data)
print(shape_of_data)
number_of_z_channels, number_of_y_pixels, number_of_x_pixels = shape_of_data
spectrum = the_data[0:, number_of_y_pixels-1, number_of_x_pixels-1]
spectrumBell = the_data[0:, 49, 50]

fig = plt.figure()
plt.plot(spectrum)
plt.show()

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


#task 2
CRVAL3 = the_header['CRVAL3'] # this is the frequency in Hz of the first z channel of the data
print(CRVAL3)
CDELT3 = the_header['CDELT3']  # this is the frequency in Hz of the separation of each channel of the spectrum in frequency space
print(CDELT3)

frequencies = np.zeros(number_of_z_channels)
for n in range(number_of_z_channels):
    frequencies[n] = CRVAL3 + n * CDELT3
print(frequencies)


# task 3
c = 3e8  # Speed of light in m/s
v_rest = 292.51744 * 1e9  # Rest frequency in Hz
velocities = (c * (v_rest - frequencies)) / v_rest
velocities_km_s = velocities / 1000
print(velocities_km_s)

# Assuming spectrum with the strongest signal from Assessed Task 1
plt.figure(figsize=(10, 6))
plt.plot(velocities_km_s, spectrumBell, label="Strongest Signal Spectrum")
plt.xlabel('Velocity (km/s)')
plt.ylabel('Intensity')
plt.title('Spectrum with Velocity Information')
plt.legend()
plt.show()

# task 4
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x-mu)**2)/(2*sigma**2))

initial_guesses = [max(spectrum), 2, 10]
popt, pcov = curve_fit(gaussian, velocities_km_s, spectrumBell, p0=initial_guesses)
#popt, pcov = curve_fit(gaussian, velocities, spectrum, p0=initial_guesses)

A_best, mu_best, sigma_best = popt
plt.figure(figsize=(10, 6))
plt.plot(velocities_km_s, spectrum, label="Observed Spectrum")
plt.plot(velocities_km_s, gaussian(velocities_km_s, *popt), label="Fitted Gaussian", linestyle='--')
plt.xlabel('Velocity (km/s)')
plt.ylabel('Intensity')
plt.title('Spectrum and Gaussian Fit')
plt.legend()
plt.show()
print(f"Best-fitting parameters:\nAmplitude (A): {A_best}\nCenter (mu): {mu_best} km/s\nWidth (sigma): {sigma_best}")

# task 5
pixel_coords = pd.read_csv('pixel_coordinates.csv')
x_coords, y_coords = pixel_coords['x pixel coordinate'], pixel_coords['y pixel coordinate']
velocity_map = np.full((number_of_y_pixels, number_of_x_pixels), np.nan)

# Iterate over the pixel coordinate pairs and conduct the Gaussian fitting
for x, y in zip(x_coords, y_coords):
    spectrum = the_data[:, y, x]
    initial_guesses = [np.max(spectrum), 2, 10]
    try:
        popt, _ = curve_fit(gaussian, velocities_km_s, spectrum, p0=initial_guesses)
        velocity_map[y, x] = popt[1]
    except RuntimeError:
        continue

fig = plt.figure(figsize=(10, 8))
plt.imshow(velocity_map, origin='lower', cmap='RdBu_r')
plt.colorbar(label='Velocity (km/s)')
plt.xlabel('X Pixel Coordinate')
plt.ylabel('Y Pixel Coordinate')
plt.title('Velocity Map')
plt.show()