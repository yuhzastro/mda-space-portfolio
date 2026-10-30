"""Linear FM (chirp) pulses and matched filtering (pulse compression).

Transmit $s(t) = \\mathrm{rect}(t/T)\\,e^{i\\pi K t^2}$, bandwidth $B = |K|T$.
The matched filter $h(t) = s^*(-t)$ maximises output SNR for white noise; the
compressed pulse has $-3$ dB width $0.886/B$, peak sidelobe ratio $-13.26$ dB,
and SNR gain $TB$ (time-bandwidth product).

Spectrum by the principle of stationary phase (MATH 300): for $TB \\gg 1$,
$S(f) = \\int s(t)e^{-i2\\pi ft}dt \\approx \\frac{1}{\\sqrt{|K|}}\\,
e^{-i\\pi f^2/K}\\,e^{i\\pi/4}$ for $|f| < B/2$. The phase is stationary
where $\\frac{d}{dt}(\\pi Kt^2 - 2\\pi ft) = 0$, i.e. $t^* = f/K$, which is why
time and frequency are linearly mapped in a chirp.
"""
from __future__ import annotations
import numpy as np


def chirp(T, B, fs):
    """Baseband up-chirp sampled at fs, centred on t = 0. Returns (t, s)."""
    n = int(round(T * fs))
    t = (np.arange(n) - n / 2) / fs
    return t, np.exp(1j * np.pi * (B / T) * t ** 2)


def stationary_phase_spectrum(f, T, B):
    K = B / T
    return np.where(np.abs(f) < B / 2, np.exp(-1j * np.pi * f ** 2 / K + 1j * np.pi / 4) / np.sqrt(K), 0)


def matched_filter(x, ref, n_fft=None):
    """Fast correlation $y = x \\star \\mathrm{ref}$ via FFT, $Y = X\\,\\mathrm{REF}^*$.
    Output index k corresponds to a delay of k samples relative to the start of ref."""
    n = n_fft or int(2 ** np.ceil(np.log2(len(x) + len(ref))))
    return np.fft.ifft(np.fft.fft(x, n) * np.conj(np.fft.fft(ref, n)))[:len(x)]


def irf_metrics(y, oversample=16):
    """Impulse-response metrics of a 1-D compressed response: (-3 dB width in
    samples, PSLR dB, ISLR dB). Upsampled by zero-padding the spectrum."""
    n = len(y)
    Y = np.fft.fftshift(np.fft.fft(y))
    pad = np.zeros(n * oversample, complex)
    pad[(n * oversample - n) // 2:(n * oversample - n) // 2 + n] = Y
    z = np.abs(np.fft.ifft(np.fft.ifftshift(pad))) ** 2
    z /= z.max()
    k = int(np.argmax(z))
    hp = np.where(z >= 0.5)[0]
    # contiguous main-lobe half-power region around the peak
    lo = k
    while lo > 0 and z[lo - 1] >= 0.5:
        lo -= 1
    hi = k
    while hi < len(z) - 1 and z[hi + 1] >= 0.5:
        hi += 1
    # linear interpolation at the -3 dB crossings
    l = lo - 1 + (0.5 - z[lo - 1]) / (z[lo] - z[lo - 1])
    r = hi + (z[hi] - 0.5) / (z[hi] - z[hi + 1])
    width = (r - l) / oversample
    # main lobe = between first minima
    a = k
    while a > 0 and z[a - 1] < z[a]:
        a -= 1
    b = k
    while b < len(z) - 1 and z[b + 1] < z[b]:
        b += 1
    side = np.concatenate([z[:a], z[b + 1:]])
    pslr = 10 * np.log10(side.max())
    islr = 10 * np.log10(side.sum() / z[a:b + 1].sum())
    return width, pslr, islr
