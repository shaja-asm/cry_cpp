#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include <deque>
#include <vector>
#include <memory>
#include <alsa/asoundlib.h>
#include "CryAnnotation.h"
#include "PredictCry.h"
#include "VADSystem.h"

class CryDetector {
public:
    CryDetector();
    ~CryDetector();

    // Delete copy constructor and copy assignment operator to prevent accidental copying
    CryDetector(const CryDetector&) = delete;
    CryDetector& operator=(const CryDetector&) = delete;

    // Public Methods
    void start();
    void stop();
    CryAnnotation get_cry_state() const;

private:
    // Private Methods
    void capture_loop();
    void update_cry_state_periodically();
    void process_audio_segment(const std::vector<int16_t> &segment);
    bool init_audio_input();

    // Configuration Constants
    static constexpr double SEGMENT_DURATION = 0.5;          // Duration of each audio segment in seconds
    static constexpr int SAMPLE_RATE = 22050;                // Sampling rate in Hz
    static constexpr int PERIOD_SIZE = 1024;                 // Number of frames per period
    static constexpr double HISTORY_DURATION = 1.0;          // Duration to keep in history buffer in seconds
    static constexpr double PREDICTION_DURATION = 5.0;       // Duration for making predictions in seconds
    static constexpr int CHANNELS = 2;                        // Number of audio channels

    // Atomic Flag for Thread Control
    std::atomic<bool> keep_running_;

    // Thread Objects
    std::thread capture_thread_;
    std::thread state_update_thread_;

    // Mutexes for Thread Safety
    mutable std::mutex history_lock_;           // Protects history_buffer_
    std::deque<std::vector<int16_t>> history_buffer_; // Audio history buffer

    mutable std::mutex cry_state_lock_;         // Protects cry_state_
    CryAnnotation cry_state_;                   // Current state of cry detection

    // Predictor for Machine Learning Model
    PredictCry predictor_;

    // Dynamic Loud Noise Threshold Management
    double dynamic_loud_noise_threshold_ = 45.0;
    double previous_selected_channel_energy_ = -1.0;

    // ALSA PCM Handle with Custom Deleter
    struct pcm_handle_deleter { 
        void operator()(snd_pcm_t* p) const { 
            if(p) snd_pcm_close(p); 
        } 
    };
    std::unique_ptr<snd_pcm_t, pcm_handle_deleter> pcm_handle_;

    // Private Helper Methods
    bool loud_noise_detection(double energy_db, const std::vector<double> &selected_channel);
    bool voice_activity_detection(const std::vector<int16_t> &selected_channel);
    void update_dynamic_loud_noise_threshold(double current_energy);

    // Static Method for Energy Computation
    static double compute_energy_db(const std::vector<int16_t> &audio, int channels, std::vector<int16_t> &selected_channel_out);
};
