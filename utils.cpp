#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <fftw3.h>
#include <vector>
#include <complex>
#include <limits>

// Helper sinc function matching np.sinc(x) = sin(pi*x)/(pi*x)
static inline double sinc(double x) {
    if (x == 0.0) return 1.0;
    double pix = M_PI * x;
    return std::sin(pix) / pix;
}

std::vector<double> firwin_highpass(double cutoff, int sample_rate, int numtaps) {
    double fs = (double)sample_rate;
    double nyq = fs/2.0;
    double fc = cutoff/nyq; // normalized cutoff in [0,1]

    if (numtaps < 1) {
        throw std::invalid_argument("numtaps must be > 0");
    }

    double left = fc;
    double right = 1.0;

    double alpha = (numtaps - 1)/2.0;
    std::vector<double> h(numtaps,0.0);

    // h[n] = right*sinc(right*m) - left*sinc(left*m) where m = n - alpha
    for (int n=0; n<numtaps; n++) {
        double m = n - alpha;
        double val = (right * sinc(m * right)) - (left * sinc(m * left));
        h[n] = val;
    }

    // Apply Hamming window
    int M = numtaps-1;
    for (int n=0; n<numtaps; n++) {
        double w = 0.54 - 0.46*std::cos((2.0*M_PI*n)/M);
        h[n] *= w;
    }

    // Scale so unity gain at Nyquist
    double scale_frequency = 1.0;
    double s = 0.0;
    for (int n=0; n<numtaps; n++) {
        double m = n - alpha;
        s += h[n]*std::cos(M_PI*m*scale_frequency);
    }
    for (int n=0; n<numtaps; n++) {
        h[n] /= s;
    }

    return h;
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
    int N = static_cast<int>(signal.size());
    if (N == 0) {
        throw std::invalid_argument("Input signal must not be empty.");
    }

    int out_size = (N/2) + 1;
    std::vector<double> magnitudes(out_size, 0.0);

    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*out_size);
    double* in = (double*)fftw_malloc(sizeof(double)*N);

    for (int i=0; i<N; i++) {
        in[i] = signal[i];
    }

    fftw_plan p = fftw_plan_dft_r2c_1d(N, in, fft_out, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i = 0; i < out_size; i++) {
        double re = fft_out[i][0];
        double im = fft_out[i][1];
        magnitudes[i] = std::sqrt(re*re + im*im);
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(fft_out);

    return magnitudes;
}

// DCT-II with norm='ortho'
static void dct_2d_norm_ortho(const std::vector<double>& input, int num_frames, int nfilt,
                              int n_mfcc, std::vector<double>& output) {
    output.resize(num_frames * n_mfcc);
    const int N = nfilt;
    double factor0 = std::sqrt(1.0 / (double)N);
    double factor = std::sqrt(2.0 / (double)N);

    for (int f = 0; f < num_frames; f++) {
        const double* in_row = &input[f * nfilt];
        for (int k = 0; k < n_mfcc; k++) {
            double sum_val = 0.0;
            for (int n = 0; n < N; n++) {
                sum_val += in_row[n] * std::cos(M_PI * (n + 0.5) * k / (double)N);
            }
            if (k == 0)
                sum_val *= factor0;
            else
                sum_val *= factor;
            output[f * n_mfcc + k] = sum_val;
        }
    }
}

// Compute MFCC features from an audio signal.
// If max_length == -1, returns all frames computed.
std::vector<float> compute_mfcc(const std::vector<float> &audio, int sr, int n_mfcc, int max_length) {
    double pre_emphasis = 0.97;
    double frame_size = 0.025;
    double frame_stride = 0.010;
    int frame_length = (int)std::round(frame_size * sr);
    int frame_step = (int)std::round(frame_stride * sr);

    // Pre-emphasis
    std::vector<double> emphasized_signal(audio.size());
    emphasized_signal[0] = audio[0];
    for (size_t i = 1; i < audio.size(); i++) {
        emphasized_signal[i] = audio[i] - pre_emphasis * audio[i - 1];
    }

    int signal_length = (int)emphasized_signal.size();
    int num_frames = (int)std::ceil((double)(signal_length - frame_length) / frame_step);

    int pad_signal_length = num_frames * frame_step + frame_length;
    std::vector<double> pad_signal = emphasized_signal;
    if (pad_signal_length > signal_length) {
        pad_signal.resize(pad_signal_length, 0.0);
    }

    // framing
    std::vector<double> frames_data(num_frames * frame_length);
    for (int i = 0; i < num_frames; i++) {
        int start = i * frame_step;
        for (int j = 0; j < frame_length; j++) {
            frames_data[i * frame_length + j] = pad_signal[start + j];
        }
    }

    // Hamming window
    std::vector<double> ham = hamming_window(frame_length);
    for (int i = 0; i < num_frames; i++) {
        for (int j = 0; j < frame_length; j++) {
            frames_data[i * frame_length + j] *= ham[j];
        }
    }

    int NFFT = 512;
    int out_size = (NFFT/2)+1;
    // Compute pow frames
    {
        fftw_plan p;
        double* in = (double*)fftw_malloc(sizeof(double)*NFFT);
        fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*out_size);

        p = fftw_plan_dft_r2c_1d(NFFT, in, out, FFTW_ESTIMATE);

        std::vector<double> pow_frames(num_frames * out_size, 0.0);

        for (int f = 0; f < num_frames; f++) {
            for (int n = 0; n < frame_length; n++) in[n] = frames_data[f*frame_length+n];
            for (int n = frame_length; n < NFFT; n++) in[n] = 0.0;

            fftw_execute(p);
            for (int k = 0; k < out_size; k++) {
                double re = out[k][0];
                double im = out[k][1];
                double mag = std::sqrt(re*re + im*im);
                double pow_val = (1.0/NFFT)*(mag*mag);
                pow_frames[f*out_size + k] = pow_val;
            }
        }

        fftw_destroy_plan(p);
        fftw_free(in);
        fftw_free(out);

        // Mel filter banks
        int nfilt = 40;
        double low_freq_mel = 0.0;
        double high_freq_mel = 2595.0 * std::log10(1.0 + (sr/2.0)/700.0);
        std::vector<double> mel_points(nfilt+2);
        for (int i = 0; i < nfilt+2; i++) {
            mel_points[i] = low_freq_mel + (i*(high_freq_mel - low_freq_mel)/(nfilt+1));
        }
        std::vector<double> hz_points(nfilt+2);
        for (int i = 0; i < nfilt+2; i++) {
            hz_points[i] = 700.0*(std::pow(10.0, mel_points[i]/2595.0)-1.0);
        }

        std::vector<int> bin(nfilt+2);
        for (int i = 0; i < nfilt+2; i++) {
            bin[i] = (int)std::floor((NFFT+1)*hz_points[i]/sr);
        }

        std::vector<double> fbank(nfilt*out_size, 0.0);
        for (int m = 1; m <= nfilt; m++) {
            int f_m_minus = bin[m-1];
            int f_m = bin[m];
            int f_m_plus = bin[m+1];

            for (int k = f_m_minus; k < f_m; k++) {
                fbank[(m-1)*out_size + k] = ((double)k - bin[m-1])/( (double)bin[m] - bin[m-1]);
            }
            for (int k = f_m; k < f_m_plus; k++) {
                fbank[(m-1)*out_size + k] = ((double)bin[m+1]-k)/( (double)bin[m+1]-bin[m]);
            }
        }

        std::vector<double> filter_banks_data(num_frames * nfilt, 0.0);
        for (int f = 0; f < num_frames; f++) {
            for (int m = 0; m < nfilt; m++) {
                double sum_val = 0.0;
                for (int k = 0; k < out_size; k++) {
                    sum_val += pow_frames[f*out_size + k] * fbank[m*out_size + k];
                }
                sum_val = 20*std::log10(std::max(sum_val, (double)std::numeric_limits<double>::epsilon())); // Python epsilon is around 1.19209e-07. C++ epsilon is around 2.220446e-16
                filter_banks_data[f*nfilt + m] = sum_val;
            }
        }

        // DCT to get MFCC
        std::vector<double> mfcc_data;
        dct_2d_norm_ortho(filter_banks_data, num_frames, nfilt, n_mfcc, mfcc_data);

        // Mean normalization
        for (int c = 0; c < n_mfcc; c++) {
            double sum_val = 0.0;
            for (int f = 0; f < num_frames; f++) {
                sum_val += mfcc_data[f*n_mfcc + c];
            }
            double mean_val = sum_val / num_frames + 1e-8;
            for (int f = 0; f < num_frames; f++) {
                mfcc_data[f*n_mfcc + c] -= mean_val;
            }
        }

        // Convert to float
        std::vector<float> mfcc_final(mfcc_data.size());
        for (size_t i = 0; i < mfcc_data.size(); i++) {
            mfcc_final[i] = (float)mfcc_data[i];
        }

        // If max_length is -1, we do not truncate or pad here.
        // The caller (preprocess_audio) handles padding/truncation.
        return mfcc_final;
    }
}
