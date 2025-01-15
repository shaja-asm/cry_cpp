#pragma once
#include <vector>

// FIR high-pass filter design using a Hamming window, replicating scipy.signal.firwin logic.
std::vector<double> firwin_highpass(double cutoff, int sample_rate, int numtaps);

// FIR filtering (equivalent to scipy.signal.lfilter(b, [1], x) for FIR filters).
std::vector<double> fir_filter(const std::vector<double> &signal, const std::vector<double> &taps);

// Hamming window function
std::vector<double> hamming_window(int length);

// Compute FFT magnitude using FFTW
std::vector<double> compute_fft_magnitude(const std::vector<double> &signal);

// Placeholder for MFCC computation
std::vector<float> compute_mfcc(const std::vector<float> &audio, int sample_rate, int n_mfcc, int max_length);
