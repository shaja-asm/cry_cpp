#include "VADSystem.h"
#include "utils.h"
#include <cmath>
#include <algorithm>

VADSystem::VADSystem(const std::vector<int16_t> &data, int sample_rate)
    : sample_rate_(sample_rate) {
    signal_.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        signal_[i] = (double)data[i];
    }
}

void VADSystem::process() {
    high_pass_filter();
    frame_signal();
    apply_window();
    compute_short_term_energy();
    compute_spectral_flatness();
    make_vad_decision();
}

bool VADSystem::has_voice() const {
    return std::any_of(vad_result_.begin(), vad_result_.end(), [](bool v){return v;});
}


void VADSystem::high_pass_filter(double cutoff) {
    auto taps = firwin_highpass(cutoff, sample_rate_, 101);
    signal_ = fir_filter(signal_, taps);
}

void VADSystem::frame_signal() {
    double FRAME_SIZE = 0.25;   
    double FRAME_STRIDE = 0.1;  
    frame_length_ = (int)std::round(FRAME_SIZE*sample_rate_);
    int frame_step = (int)std::round(FRAME_STRIDE*sample_rate_);
    int signal_length = (int)signal_.size();
    num_frames_ = (int)std::ceil((double)(signal_length - frame_length_)/frame_step) + 1;

    int pad_signal_length = num_frames_*frame_step + frame_length_;
    signal_.resize(pad_signal_length, 0.0);

    frames_.resize(num_frames_*frame_length_);
    for (int i = 0; i < num_frames_; i++) {
        for (int j = 0; j < frame_length_; j++) {
            frames_[i*frame_length_ + j] = signal_[i*frame_step + j];
        }
    }
}

void VADSystem::apply_window() {
    auto window = hamming_window(frame_length_);
    for (int i = 0; i < num_frames_; i++) {
        for (int j = 0; j < frame_length_; j++) {
            frames_[i*frame_length_ + j] *= window[j];
        }
    }
}

void VADSystem::compute_short_term_energy() {
    ste_.resize(num_frames_);
    for (int i = 0; i < num_frames_; i++) {
        double energy=0.0;
        for (int j=0; j<frame_length_; j++) {
            double v = frames_[i*frame_length_+j];
            energy += v*v;
        }
        energy /= frame_length_;
        ste_[i] = 10.0*std::log10(energy+1e-15);
    }
}

void VADSystem::compute_spectral_flatness() {
    int NFFT = 512;
    sf_db_.resize(num_frames_);

    double eps = 1e-15; // a small epsilon similar to np.finfo(float).eps

    for (int i = 0; i < num_frames_; i++) {
        // Extract frame
        std::vector<double> frame(frame_length_);
        for (int j = 0; j < frame_length_; j++) {
            frame[j] = frames_[i*frame_length_ + j];
        }
        // Zero-pad to NFFT
        frame.resize(NFFT, 0.0);

        // Compute magnitude spectrum (like mag_frames in Python)
        std::vector<double> spectrum = compute_fft_magnitude(frame); // size = NFFT/2+1

        // Replace zeros with eps to avoid log(0)
        for (double &val : spectrum) {
            if (val == 0.0) val = eps;
        }

        // Compute power spectrum: pow_frames = (1.0 / NFFT) * (mag_frames^2)
        for (double &val : spectrum) {
            val = (1.0 / NFFT) * (val * val);
        }

        // Compute geometric mean: exp(mean(log(pow_frames + eps)))
        double sum_log = 0.0;
        for (double val : spectrum) {
            sum_log += std::log(val + eps);
        }
        double geometric_mean = std::exp(sum_log / spectrum.size());

        // Compute arithmetic mean: mean(pow_frames) + eps
        double sum = 0.0;
        for (auto val : spectrum) {
            sum += val;
        }
        double arithmetic_mean = (sum / spectrum.size()) + eps;

        double sf = 10.0 * std::log10(geometric_mean / arithmetic_mean);
        sf_db_[i] = sf;
    }
}


void VADSystem::make_vad_decision() {
    double ENERGY_THRESHOLD = -40.0;
    double SF_THRESHOLD = -15.0;
    vad_result_.resize(num_frames_);

    std::vector<double> energy_thresh(num_frames_, ENERGY_THRESHOLD);
    std::vector<double> sf_thresh(num_frames_, SF_THRESHOLD);

    for (int i = 1; i < num_frames_; i++) {
        energy_thresh[i] = std::max(ENERGY_THRESHOLD, 0.7*ste_[i-1]);
        sf_thresh[i] = std::max(SF_THRESHOLD, 0.7*sf_db_[i-1]);
    }

    for (int i = 0; i < num_frames_; i++) {
        bool ste_dec = ste_[i] > energy_thresh[i];
        bool sf_dec = sf_db_[i] > sf_thresh[i];
        vad_result_[i] = (ste_dec && sf_dec);
    }
}
