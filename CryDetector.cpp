#include "CryDetector.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <thread>
#include <algorithm>
#include "VADSystem.h"

// Constructor: Initializes audio input and sets initial state
CryDetector::CryDetector()
    : keep_running_(false), cry_state_(CryAnnotation::UNKNOWN) {
    if (!init_audio_input()) {
        throw std::runtime_error("Failed to initialize audio input");
    }
}

// Destructor: Stops threads and cleans up resources
CryDetector::~CryDetector() {
    stop();
}

// Initializes the ALSA PCM device for audio capture
bool CryDetector::init_audio_input() {
    // Set ALSA configuration path
    setenv("ALSA_CONFIG_PATH", "/usr/share/alsa/alsa.conf", 1);

    snd_pcm_t* raw_pcm_handle = nullptr;
    int err = snd_pcm_open(&raw_pcm_handle, "default", SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        std::cerr << "Error opening PCM device: " << snd_strerror(err) << std::endl;
        return false;
    }

    // Assign the PCM handle to the unique_ptr with the custom deleter
    pcm_handle_.reset(raw_pcm_handle);

    snd_pcm_hw_params_t* params;
    snd_pcm_hw_params_alloca(&params);

    // Initialize hardware parameters
    err = snd_pcm_hw_params_any(pcm_handle_.get(), params);
    if (err < 0) {
        std::cerr << "Error initializing HW params: " << snd_strerror(err) << std::endl;
        return false;
    }

    // Set access type
    err = snd_pcm_hw_params_set_access(pcm_handle_.get(), params, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) {
        std::cerr << "Error setting PCM access: " << snd_strerror(err) << std::endl;
        return false;
    }

    // Set sample format
    err = snd_pcm_hw_params_set_format(pcm_handle_.get(), params, SND_PCM_FORMAT_S16_LE);
    if (err < 0) {
        std::cerr << "Error setting PCM format: " << snd_strerror(err) << std::endl;
        return false;
    }

    // Set number of channels
    err = snd_pcm_hw_params_set_channels(pcm_handle_.get(), params, CHANNELS);
    if (err < 0) {
        std::cerr << "Error setting channel count: " << snd_strerror(err) << std::endl;
        return false;
    }

    // Set sample rate
    unsigned int rate = SAMPLE_RATE;
    err = snd_pcm_hw_params_set_rate_near(pcm_handle_.get(), params, &rate, nullptr);
    if (err < 0) {
        std::cerr << "Error setting PCM rate: " << snd_strerror(err) << std::endl;
        return false;
    }

    // Set period size
    snd_pcm_uframes_t period_size = PERIOD_SIZE;
    err = snd_pcm_hw_params_set_period_size_near(pcm_handle_.get(), params, &period_size, nullptr);
    if (err < 0) {
        std::cerr << "Error setting period size: " << snd_strerror(err) << std::endl;
        return false;
    }

    // Apply hardware parameters
    err = snd_pcm_hw_params(pcm_handle_.get(), params);
    if (err < 0) {
        std::cerr << "Error setting HW params: " << snd_strerror(err) << std::endl;
        return false;
    }

    // Prepare PCM device
    err = snd_pcm_prepare(pcm_handle_.get());
    if (err < 0) {
        std::cerr << "Error preparing PCM device: " << snd_strerror(err) << std::endl;
        return false;
    }

    return true;
}

// Starts the capture and state update threads
void CryDetector::start() {
    keep_running_ = true;
    capture_thread_ = std::thread(&CryDetector::capture_loop, this);
    state_update_thread_ = std::thread(&CryDetector::update_cry_state_periodically, this);
}

// Stops the capture and state update threads
void CryDetector::stop() {
    keep_running_ = false;
    if (capture_thread_.joinable()) capture_thread_.join();
    if (state_update_thread_.joinable()) state_update_thread_.join();
}

// Retrieves the current cry state in a thread-safe manner
CryAnnotation CryDetector::get_cry_state() const {
    std::lock_guard<std::mutex> lk(cry_state_lock_);
    return cry_state_;
}

// Continuously reads audio data from ALSA and processes it
void CryDetector::capture_loop() {
    int num_chunks = static_cast<int>(SAMPLE_RATE * SEGMENT_DURATION / PERIOD_SIZE);
    // Allocate buffer on the heap to prevent stack overflow
    std::vector<int16_t> buffer(PERIOD_SIZE * CHANNELS);

    while (keep_running_) {
        std::vector<int16_t> audio_data;
        audio_data.reserve(num_chunks * PERIOD_SIZE * CHANNELS); // Reserve space to optimize

        for (int i = 0; i < num_chunks; i++) {
            int frames = snd_pcm_readi(pcm_handle_.get(), buffer.data(), PERIOD_SIZE);
            if (frames > 0) {
                // Ensure we only insert the actual frames read
                audio_data.insert(audio_data.end(), buffer.begin(), buffer.begin() + frames * CHANNELS);
            } else {
                if (frames == -EPIPE) {
                    // Buffer overrun
                    snd_pcm_prepare(pcm_handle_.get());
                } else {
                    std::cerr << "Error reading from PCM device: " << snd_strerror(frames) << std::endl;
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

// Processes an audio segment by adding it to the history buffer
void CryDetector::process_audio_segment(const std::vector<int16_t> &segment) {
    // std::cout << "Entering process_audio_segment" << std::endl;
    std::lock_guard<std::mutex> lk(history_lock_);
    history_buffer_.push_back(segment);

    // Ensure history < PREDICTION_DURATION
    double total_duration = 0.0;
    for (const auto &seg : history_buffer_) {
        total_duration += static_cast<double>(seg.size()) / (SAMPLE_RATE * CHANNELS);
    }
    while (total_duration > PREDICTION_DURATION && !history_buffer_.empty()) {
        auto front_seg = history_buffer_.front();
        history_buffer_.pop_front();
        total_duration -= static_cast<double>(front_seg.size()) / (SAMPLE_RATE * CHANNELS);
    }

    // std::cout << "Exiting process_audio_segment" << std::endl;
}

// Periodically updates the cry state
void CryDetector::update_cry_state_periodically() {
    while (keep_running_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        std::vector<int16_t> combined;
        {
            std::lock_guard<std::mutex> lk(history_lock_);
            if (history_buffer_.empty()) continue;
            // Combine all segments
            size_t total_size = 0;
            for (const auto &seg : history_buffer_) total_size += seg.size();
            combined.reserve(total_size);
            for (const auto &seg : history_buffer_) {
                combined.insert(combined.end(), seg.begin(), seg.end());
            }
        }

        std::vector<int16_t> selected_channel;
        double energy_db = compute_energy_db(combined, CHANNELS, selected_channel);
        std::cout << "energy_db: " << energy_db << std::endl;

        if (energy_db < 0.0 || selected_channel.empty()) {
            // Error or no data
            std::lock_guard<std::mutex> lk(cry_state_lock_);
            cry_state_ = CryAnnotation::UNKNOWN;
            continue;
        }

        // Convert selected_channel to double for loud_noise_detection
        std::vector<double> selected_channel_double(selected_channel.begin(), selected_channel.end());

        bool loud = loud_noise_detection(energy_db, selected_channel_double);
        if (!loud) {
            // SILENT
            std::lock_guard<std::mutex> lk(cry_state_lock_);
            cry_state_ = CryAnnotation::SILENT;
            std::cout << "Loud noise not detected" << std::endl;
            continue;
        }

        bool voice = voice_activity_detection(selected_channel);
        if (!voice) {
            // NOT_CRY
            std::lock_guard<std::mutex> lk(cry_state_lock_);
            cry_state_ = CryAnnotation::NOT_CRY;
            std::cout << "Voice activity not detected" << std::endl;
            continue;
        }

        // Run ML prediction
        // std::cout << "Running ML prediction" << std::endl;
        CryAnnotation prediction = predictor_.get_prediction(selected_channel);
        std::lock_guard<std::mutex> lk(cry_state_lock_);
        cry_state_ = prediction;
    }
}

// Detects if the audio is loud based on energy thresholds
bool CryDetector::loud_noise_detection(double energy_db, const std::vector<double> &selected_channel) {
    double current_channel_energy = 0.0;
    for (const auto& val : selected_channel) {
        current_channel_energy += val * val;
    }
    current_channel_energy /= selected_channel.size();

    update_dynamic_loud_noise_threshold(current_channel_energy);
    return energy_db > dynamic_loud_noise_threshold_;
}

// Uses VADSystem to detect voice activity in the selected channel
bool CryDetector::voice_activity_detection(const std::vector<int16_t> &selected_channel) {
    VADSystem vad(selected_channel, SAMPLE_RATE);
    vad.process();
    return vad.has_voice();
}

// Updates the dynamic loud noise threshold based on current energy
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

// Computes the energy in decibels and selects the channel with higher energy
double CryDetector::compute_energy_db(const std::vector<int16_t> &audio, int channels, std::vector<int16_t> &selected_channel_out) {
    if (audio.empty()) return -1.0;

    std::vector<int16_t> left;
    std::vector<int16_t> right;
    left.reserve(audio.size() / channels);
    right.reserve(audio.size() / channels);

    for (size_t i = 0; i < audio.size(); i += channels) {
        left.push_back(audio[i]);
        if (channels == 2 && (i + 1) < audio.size()) { // Added bounds check
            right.push_back(audio[i + 1]);
        }
    }

    auto energy = [](const std::vector<int16_t> &ch) -> double {
        double e = 0.0;
        for (auto v : ch) e += static_cast<double>(v) * v;
        return e / (ch.size() + 1e-15); // Avoid division by zero
    };

    double e_left = energy(left);
    double e_right = energy(right);
    double avg_e = (e_left + e_right) / 2.0;
    double energy_db = 10.0 * std::log10(avg_e + 1e-15); // Avoid log(0)

    if (e_left >= e_right) {
        selected_channel_out = left;
    } else {
        selected_channel_out = right;
    }

    return energy_db;
}
