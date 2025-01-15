#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include "CryAnnotation.h"
#include "tensorflow/lite/c/c_api.h"

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

    // Delete copy constructor and assignment operator
    PredictCry(const PredictCry&) = delete;
    PredictCry& operator=(const PredictCry&) = delete;

    // Default move operations
    PredictCry(PredictCry&&) noexcept = default;
    PredictCry& operator=(PredictCry&&) noexcept = default;

    // Removed const from get_prediction because it modifies internal state via predict()
    CryAnnotation get_prediction(const std::vector<int16_t> &audio);

private:
    // Removed const from predict() because it updates output_tensor_
    CryAnnotation predict(const std::vector<int16_t> &audio);
    std::vector<float> preprocess_audio(const std::vector<int16_t> &audio) const; 
    bool init_tflite();

    int sample_rate_;
    int num_threads_;

    // Store as const pointers, cast away const when copying data
    const TfLiteTensor* input_tensor_ = nullptr;
    const TfLiteTensor* output_tensor_ = nullptr;

    std::unique_ptr<TfLiteModel, TfLiteModelDeleter> model_;
    std::unique_ptr<TfLiteInterpreterOptions, TfLiteInterpreterOptionsDeleter> options_;
    std::unique_ptr<TfLiteInterpreter, TfLiteInterpreterDeleter> interpreter_;

    float input_scale_ = 1.0f;
    int input_zero_point_ = 0;
    float output_scale_ = 1.0f;
    int output_zero_point_ = 0;

    int input_dtype_ = 0;
    int output_dtype_ = 0;
};
