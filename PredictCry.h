#pragma once
#include <vector>
#include "CryAnnotation.h"

// Forward declarations for TFLite C API structs
struct TfLiteModel;
struct TfLiteInterpreter;
struct TfLiteInterpreterOptions;
struct TfLiteTensor;

class PredictCry {
public:
    PredictCry(int num_threads=2, int sample_rate=22050);
    ~PredictCry();

    CryAnnotation get_prediction(const std::vector<int16_t> &audio);

private:
    CryAnnotation predict(const std::vector<int16_t> &audio);
    std::vector<float> preprocess_audio(const std::vector<int16_t> &audio);
    bool init_tflite();

    int sample_rate_;
    int num_threads_;

    TfLiteModel* model_ = nullptr;
    TfLiteInterpreterOptions* options_ = nullptr;
    TfLiteInterpreter* interpreter_ = nullptr;
    TfLiteTensor* input_tensor_ = nullptr;
    TfLiteTensor* output_tensor_ = nullptr;

    float input_scale_ = 1.0f;
    int input_zero_point_ = 0;
    float output_scale_ = 1.0f;
    int output_zero_point_ = 0;

    int input_dtype_ = 0;
    int output_dtype_ = 0;
};
