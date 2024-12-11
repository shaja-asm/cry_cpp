#pragma once
#include <vector>

// FIR filter design and filter
std::vector<double> firwin_highpass(double cutoff, int sample_rate, int numtaps);
std::vector<double> fir_filter(const std::vector<double> &signal, const std::vector<double> &taps);

// Window functions
std::vector<double> hamming_window(int length);

// FFT computation (using FFTW)
std::vector<double> compute_fft_magnitude(const std::vector<double> &signal);

// MFCC computation
std::vector<float> compute_mfcc(const std::vector<float> &audio, int sample_rate, int n_mfcc, int max_length);
