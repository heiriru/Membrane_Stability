'''Azimuthal decomposition of eigenmodes on domains with a circular symmetry (ring geometry).'''
import numpy as np


def azimuthal_spectrum(f, center, radii, n_theta=256, m_max=8):
    '''
    Fourier amplitudes |f_m|, m = 0 ... m_max, of the scalar Function f on circles of the given radii around 'center',
    summed over the radii. f must allow point evaluation (f.set_allow_extrapolation(True) near the boundary).
    '''
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    amplitude = np.zeros(m_max + 1)
    for radius in radii:
        values = np.array([f(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)) for t in theta])
        coefficients = np.abs(np.fft.rfft(values)) / n_theta
        amplitude += coefficients[:m_max + 1] ** 2
    return np.sqrt(amplitude)


def dominant_m(f, center, radii, **kwargs):
    spectrum = azimuthal_spectrum(f, center, radii, **kwargs)
    return int(np.argmax(spectrum)), spectrum
