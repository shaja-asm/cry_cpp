#include "CryDetector.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <thread>
#include <algorithm>
#include "VADSystem.h"

// ALSA headers
#include <alsa/asoundlib.h>

CryDetector::CryDetector()
    : keep_running_(false), cry_state_(CryAnnotation::UNKNOWN) {
    if (!init_audio_input()) {
        throw std::runtime_error("Failed to initialize audio input");
    }
}

CryDetector::~CryDetector() {
    stop();
}

bool CryDetector::init_audio_input() {
    int err = snd_pcm_open(&pcm_handle_, "default", SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        std::cerr << "Error opening PCM device: " << snd_strerror(err) << std::endl;
        return false;
    }

    snd_pcm_hw_params_t *params;
    snd_pcm_hw_params_alloca(&params);

    snd_pcm_hw_params_any(pcm_handle_.get(), params);
    snd_pcm_hw_params_set_access(pcm_handle_.get(), params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm_handle_.get(), params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(pcm_handle_.get(), params, CHANNELS);
    unsigned int rate = SAMPLE_RATE;
    snd_pcm_hw_params_set_rate_near(pcm_handle_.get(), params, &rate, nullptr);
    snd_pcm_hw_params_set_period_size_near(pcm_handle_.get(), params, (snd_pcm_uframes_t*)&PERIOD_SIZE, nullptr);

    err = snd_pcm_hw_params(pcm_handle_.get(), params);
    if (err < 0) {
        std::cerr << "Error setting HW params: " << snd_strerror(err) << std::endl;
        return false;
    }

    err = snd_pcm_prepare(pcm_handle_.get());
    if (err < 0) {
        std::cerr << "Error preparing PCM device: " << snd_strerror(err) << std::endl;
        return false;
    }

    return true;
}

void CryDetector::start() {
    keep_running_ = true;
    capture_thread_ = std::thread(&CryDetector::capture_loop, this);
    state_update_thread_ = std::thread(&CryDetector::update_cry_state_periodically, this);
}

void CryDetector::stop() {
    keep_running_ = false;
    if (capture_thread_.joinable()) capture_thread_.join();
    if (state_update_thread_.joinable()) state_update_thread_.join();
}

CryAnnotation CryDetector::get_cry_state() {
    std::lock_guard<std::mutex> lk(cry_state_lock_);
    return cry_state_;
}

void CryDetector::capture_loop() {
    int num_chunks = static_cast<int>(SAMPLE_RATE * SEGMENT_DURATION / PERIOD_SIZE);
    int16_t buffer[PERIOD_SIZE * CHANNELS];

    while (keep_running_) {
        std::vector<int16_t> audio_data;
        for (int i = 0; i < num_chunks; i++) {
            int frames = snd_pcm_readi(pcm_handle_.get(), buffer, PERIOD_SIZE);
            if (frames > 0) {
                audio_data.insert(audio_data.end(), buffer, buffer + frames * CHANNELS);
            } else {
                if (frames == -EPIPE) {
                    snd_pcm_prepare(pcm_handle_.get());
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        if (!audio_data.empty()) {
            process_audio_segment(audio_data);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void CryDetector::process_audio_segment(const std::vector<int16_t> &segment) {
    std::lock_guard<std::mutex> lk(history_lock_);
    history_buffer_.push_back(segment);

    // Ensure history does not exceed PREDICTION_DURATION
    double total_duration = 0.0;
    for (auto &seg : history_buffer_) {
        total_duration += static_cast<double>(seg.size()) / (SAMPLE_RATE * CHANNELS);
    }
    while (total_duration > PREDICTION_DURATION) {
        auto front_seg = history_buffer_.front();
        history_buffer_.pop_front();
        total_duration -= static_cast<double>(front_seg.size()) / (SAMPLE_RATE * CHANNELS);
    }
}

void CryDetector::update_cry_state_periodically() {
    while (keep_running_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        std::vector<int16_t> combined;
        {
            std::lock_guard<std::mutex> lk(history_lock_);
            if (history_buffer_.empty()) continue;
            // Combine all segments
            size_t total_size = 0;
            for (auto &seg : history_buffer_) total_size += seg.size();
            combined.reserve(total_size);
            for (auto &seg : history_buffer_) {
                combined.insert(combined.end(), seg.begin(), seg.end());
            }
        }

        std::vector<int16_t> selected_channel;
        double energy_db = compute_energy_db(combined, CHANNELS, selected_channel);

        if (energy_db < 0.0 || selected_channel.empty()) {
            // Error or no data
            std::lock_guard<std::mutex> lk(cry_state_lock_);
            cry_state_ = CryAnnotation::UNKNOWN;
            continue;
        }

        bool loud = loud_noise_detection(energy_db, std::vector<double>(selected_channel.begin(), selected_channel.end()));
        if (!loud) {
            // SILENT
            std::lock_guard<std::mutex> lk(cry_state_lock_);
            cry_state_ = CryAnnotation::SILENT;
            continue;
        }

        bool voice = voice_activity_detection(selected_channel);
        if (!voice) {
            // NOT_CRY
            std::lock_guard<std::mutex> lk(cry_state_lock_);
            cry_state_ = CryAnnotation::NOT_CRY;
            continue;
        }

        // Run ML prediction
        CryAnnotation prediction = predictor_.get_prediction(selected_channel);
        std::lock_guard<std::mutex> lk(cry_state_lock_);
        cry_state_ = prediction;
    }
}

bool CryDetector::loud_noise_detection(double energy_db, const std::vector<double> &selected_channel) {
    double current_channel_energy = 0.0;
    for (auto val : selected_channel) {
        current_channel_energy += val*val;
    }
    current_channel_energy /= selected_channel.size();

    update_dynamic_loud_noise_threshold(current_channel_energy);
    return energy_db > dynamic_loud_noise_threshold_;
}

bool CryDetector::voice_activity_detection(const std::vector<int16_t> &selected_channel) {
    VADSystem vad(selected_channel, SAMPLE_RATE);
    vad.process();
    return vad.has_voice();
}

void CryDetector::update_dynamic_loud_noise_threshold(double current_energy) {
    double current_energy_db = 10.0 * std::log10(current_energy + 1e-10);
    if (previous_selected_channel_energy_ < 0.0) {
        dynamic_loud_noise_threshold_ = std::max(current_energy_db, 45.0);
        previous_selected_channel_energy_ = current_energy_db * 0.6;
    } else {
        dynamic_loud_noise_threshold_ = std::max(previous_selected_channel_energy_, 45.0);
        previous_selected_channel_energy_ = current_energy_db * 0.6;
    }
}

double CryDetector::compute_energy_db(const std::vector<int16_t> &audio, int channels, std::vector<int16_t> &selected_channel_out) {
    if (audio.empty()) return -1.0;

    std::vector<int16_t> left;
    std::vector<int16_t> right;
    left.reserve(audio.size()/channels);
    right.reserve(audio.size()/channels);

    for (size_t i = 0; i < audio.size(); i+=channels) {
        left.push_back(audio[i]);
        if (channels == 2) right.push_back(audio[i+1]);
    }

    auto energy = [](const std::vector<int16_t> &ch) {
        double e=0.0;
        for (auto v: ch) e += (double)v*v;
        return e/(ch.size()+1e-15);
    };

    double e_left = energy(left);
    double e_right = energy(right);
    double avg_e = (e_left + e_right) / 2.0;
    double energy_db = 10.0 * std::log10(avg_e + 1e-15);

    if (e_left >= e_right) {
        selected_channel_out = left;
    } else {
        selected_channel_out = right;
    }

    return energy_db;
}
