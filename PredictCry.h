#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include "CryAnnotation.h"

// Include the TFLite C API header
#include "tensorflow/lite/c/c_api.h" // Ensure this path is correct based on your project structure

// Forward declarations for TFLite C API structs
struct TfLiteModel;
struct TfLiteInterpreter;
struct TfLiteInterpreterOptions;
struct TfLiteTensor;

// Custom deleters for TensorFlow Lite structures
struct TfLiteModelDeleter {
    void operator()(TfLiteModel* model) const {
        if (model) {
            TfLiteModelDelete(model);
        }
    }
};

struct TfLiteInterpreterOptionsDeleter {
    void operator()(TfLiteInterpreterOptions* options) const {
        if (options) {
            TfLiteInterpreterOptionsDelete(options);
        }
    }
};

struct TfLiteInterpreterDeleter {
    void operator()(TfLiteInterpreter* interpreter) const {
        if (interpreter) {
            TfLiteInterpreterDelete(interpreter);
        }
    }
};

class PredictCry {
public:
    PredictCry(int num_threads = 2, int sample_rate = 22050);
    ~PredictCry();

    // Delete copy constructor and copy assignment operator to prevent accidental copying
    PredictCry(const PredictCry&) = delete;
    PredictCry& operator=(const PredictCry&) = delete;

    // Implement move constructor and move assignment operator if needed
    PredictCry(PredictCry&&) noexcept = default;
    PredictCry& operator=(PredictCry&&) noexcept = default;

    // Public Method
    CryAnnotation get_prediction(const std::vector<int16_t> &audio) const;

private:
    // Private Methods
    CryAnnotation predict(const std::vector<int16_t> &audio) const; // Marked as const
    std::vector<float> preprocess_audio(const std::vector<int16_t> &audio) const; // Marked as const
    bool init_tflite();

    int sample_rate_;
    int num_threads_;

    // Smart pointers with custom deleters
    std::unique_ptr<TfLiteModel, TfLiteModelDeleter> model_;
    std::unique_ptr<TfLiteInterpreterOptions, TfLiteInterpreterOptionsDeleter> options_;
    std::unique_ptr<TfLiteInterpreter, TfLiteInterpreterDeleter> interpreter_;

    // Raw pointers to tensors managed by the interpreter
    TfLiteTensor* input_tensor_ = nullptr;
    TfLiteTensor* output_tensor_ = nullptr;

    float input_scale_ = 1.0f;
    int input_zero_point_ = 0;
    float output_scale_ = 1.0f;
    int output_zero_point_ = 0;

    int input_dtype_ = 0;
    int output_dtype_ = 0;
};
