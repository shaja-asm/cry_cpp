#include "utils.h"
#include <cmath>
#include <fftw3.h>
#include <stdexcept>
#include <algorithm>

// FIR high-pass filter design (simplified)
std::vector<double> firwin_highpass(double cutoff, int sample_rate, int numtaps) {
    std::vector<double> taps(numtaps, 0.0);
    double fc = cutoff/(sample_rate/2.0);
    int M = numtaps-1;
    for (int n=0; n<numtaps; n++) {
        double val;
        if(n == M/2) {
            val = (1.0 - 2.0*fc);
        } else {
            double x = (n - M/2.0)*M_PI;
            val = (std::sin(x)-2.0*fc*std::sin(x))/((x==0.0)?1.0:x);
        }
        double w = 0.54 - 0.46*std::cos(2*M_PI*n/M);
        taps[n] = val*w;
    }

    double sum = 0.0;
    for (auto v: taps) sum+=v;
    for (auto &v: taps) v/=sum;

    return taps;
}

std::vector<double> fir_filter(const std::vector<double> &signal, const std::vector<double> &taps) {
    std::vector<double> output(signal.size(),0.0);
    int nt = (int)taps.size();
    for (size_t n=0; n<signal.size(); n++) {
        double acc=0.0;
        for (int k=0; k<nt; k++) {
            if((int)n-k>=0) acc += signal[n-k]*taps[k];
        }
        output[n]=acc;
    }
    return output;
}

std::vector<double> hamming_window(int length) {
    std::vector<double> w(length);
    for (int i=0; i<length; i++) {
        w[i] = 0.54 - 0.46*std::cos(2*M_PI*i/(length-1));
    }
    return w;
}

std::vector<double> compute_fft_magnitude(const std::vector<double> &signal) {
    int N = (int)signal.size();
    std::vector<double> out(N/2+1,0.0);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(N/2+1));
    double* in = (double*)fftw_malloc(sizeof(double)*N);

    for (int i=0; i<N; i++) in[i]=signal[i];
    fftw_plan p = fftw_plan_dft_r2c_1d(N, in, fft_out, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i=0; i<(N/2+1); i++) {
        double re = fft_out[i][0];
        double im = fft_out[i][1];
        out[i]=std::sqrt(re*re+im*im);
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(fft_out);
    return out;
}

// MFCC computation (Placeholder)
std::vector<float> compute_mfcc(const std::vector<float> &audio, int sample_rate, int n_mfcc, int max_length) {
    // TODO: Implement full MFCC computation
    std::vector<float> mfcc(max_length*n_mfcc, 0.0f);
    return mfcc;
}
