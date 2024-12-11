#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include <deque>
#include <vector>
#include "CryAnnotation.h"
#include "PredictCry.h"

class CryDetector {
public:
    CryDetector();
    ~CryDetector();

    void start();
    void stop();
    CryAnnotation get_cry_state();

private:
    void capture_loop();
    void update_cry_state_periodically();
    void process_audio_segment(const std::vector<int16_t> &segment);
    bool init_audio_input();

    // Configuration constants (similar to Python)
    static constexpr double SEGMENT_DURATION = 0.5;
    static constexpr int SAMPLE_RATE = 22050;
    static constexpr int PERIOD_SIZE = 1024;
    static constexpr double HISTORY_DURATION = 1.0;   
    static constexpr double PREDICTION_DURATION = 5.0;
    static constexpr int CHANNELS = 2;

    std::atomic<bool> keep_running_;
    std::thread capture_thread_;
    std::thread state_update_thread_;

    std::mutex history_lock_;
    std::deque<std::vector<int16_t>> history_buffer_;

    std::mutex cry_state_lock_;
    CryAnnotation cry_state_;

    PredictCry predictor_;

    // Dynamic loud noise threshold
    double dynamic_loud_noise_threshold_ = 45.0;
    double previous_selected_channel_energy_ = -1.0;

    // ALSA related
    struct pcm_handle_deleter { void operator()(snd_pcm_t* p) const { if(p) snd_pcm_close(p); } };
    std::unique_ptr<snd_pcm_t, pcm_handle_deleter> pcm_handle_;

    bool loud_noise_detection(double energy_db, const std::vector<double> &selected_channel);
    bool voice_activity_detection(const std::vector<int16_t> &selected_channel);
    void update_dynamic_loud_noise_threshold(double current_energy);
    static double compute_energy_db(const std::vector<int16_t> &audio, int channels, std::vector<int16_t> &selected_channel_out);
};
