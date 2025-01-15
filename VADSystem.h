#pragma once
#include <vector>
#include <cstdint>

class VADSystem {
public:
    VADSystem(const std::vector<int16_t> &data, int sample_rate);
    void process();
    bool has_voice() const;

private:
    void high_pass_filter(double cutoff=500.0);
    void frame_signal();
    void apply_window();
    void compute_short_term_energy();
    void compute_spectral_flatness();
    void make_vad_decision();

    std::vector<double> signal_;
    int sample_rate_;

    std::vector<double> frames_;
    int num_frames_ = 0;
    int frame_length_ = 0;

    std::vector<double> ste_;
    std::vector<double> sf_db_;
    std::vector<bool> vad_result_;
};
