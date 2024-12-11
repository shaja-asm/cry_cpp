#include "PredictCry.h"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <iostream>
#include "tensorflow/lite/c/c_api.h"
#include "utils.h" // For MFCC and related preprocessing

static const char* MODEL_PATH = "basestation/algorithm/cry_detection/cnn_cry_detection_model_quant_shifted.tflite";

PredictCry::PredictCry(int num_threads, int sample_rate)
    : sample_rate_(sample_rate), num_threads_(num_threads) {
    if(!init_tflite()) {
        throw std::runtime_error("Failed to initialize TFLite interpreter");
    }
}

PredictCry::~PredictCry() {
    if (interpreter_) TfLiteInterpreterDelete(interpreter_);
    if (options_) TfLiteInterpreterOptionsDelete(options_);
    if (model_) TfLiteModelDelete(model_);
}

bool PredictCry::init_tflite() {
    model_ = TfLiteModelCreateFromFile(MODEL_PATH);
    if(!model_) return false;

    options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options_, num_threads_);

    interpreter_ = TfLiteInterpreterCreate(model_, options_);
    if(!interpreter_) return false;

    if (TfLiteInterpreterAllocateTensors(interpreter_) != kTfLiteOk) return false;

    input_tensor_ = TfLiteInterpreterGetInputTensor(interpreter_, 0);
    output_tensor_ = TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    if(!input_tensor_ || !output_tensor_) return false;

    input_dtype_ = TfLiteTensorType(input_tensor_);
    output_dtype_ = TfLiteTensorType(output_tensor_);

    auto input_params = TfLiteTensorQuantizationParams(input_tensor_);
    input_scale_ = input_params.scale;
    input_zero_point_ = input_params.zero_point;

    auto output_params = TfLiteTensorQuantizationParams(output_tensor_);
    output_scale_ = output_params.scale;
    output_zero_point_ = output_params.zero_point;

    return true;
}

CryAnnotation PredictCry::get_prediction(const std::vector<int16_t> &audio) {
    return predict(audio);
}

CryAnnotation PredictCry::predict(const std::vector<int16_t> &audio) {
    std::vector<float> input_data = preprocess_audio(audio);
    const int TF_LITE_TYPE_FLOAT32 = 1;
    const int TF_LITE_TYPE_INT8 = 9;

    void* tensor_data_ptr = nullptr;
    int tensor_data_size = 0;

    if (input_dtype_ == TF_LITE_TYPE_INT8) {
        std::vector<int8_t> quantized_data(input_data.size());
        for (size_t i=0; i<input_data.size(); i++) {
            float val = input_data[i]/input_scale_ + input_zero_point_;
            val = std::round(val);
            if(val < -128) val = -128; 
            if(val > 127) val = 127;
            quantized_data[i] = static_cast<int8_t>(val);
        }
        tensor_data_ptr = quantized_data.data();
        tensor_data_size = (int)quantized_data.size();
        if (TfLiteTensorCopyFromBuffer(input_tensor_, tensor_data_ptr, tensor_data_size) != kTfLiteOk) {
            throw std::runtime_error("Failed to copy data to input tensor");
        }
    } else if (input_dtype_ == TF_LITE_TYPE_FLOAT32) {
        tensor_data_ptr = input_data.data();
        tensor_data_size = (int)(input_data.size()*sizeof(float));
        if (TfLiteTensorCopyFromBuffer(input_tensor_, tensor_data_ptr, tensor_data_size) != kTfLiteOk) {
            throw std::runtime_error("Failed to copy data to input tensor");
        }
    } else {
        throw std::runtime_error("Unsupported input tensor data type");
    }

    if (TfLiteInterpreterInvoke(interpreter_) != kTfLiteOk) {
        throw std::runtime_error("Failed to invoke TFLite interpreter");
    }

    int output_byte_size = (int)TfLiteTensorByteSize(output_tensor_);
    if (output_dtype_ == TF_LITE_TYPE_INT8) {
        std::vector<int8_t> output_int8(output_byte_size);
        if (TfLiteTensorCopyToBuffer(output_tensor_, output_int8.data(), output_int8.size()) != kTfLiteOk) {
            throw std::runtime_error("Failed to copy from output tensor");
        }
        float pred = (static_cast<float>(output_int8[0]) - output_zero_point_)*output_scale_;
        return pred > 0.5f ? CryAnnotation::CRY : CryAnnotation::NOT_CRY;

    } else if (output_dtype_ == TF_LITE_TYPE_FLOAT32) {
        std::vector<float> output_float(output_byte_size/sizeof(float));
        if (TfLiteTensorCopyToBuffer(output_tensor_, output_float.data(), output_byte_size) != kTfLiteOk) {
            throw std::runtime_error("Failed to copy from output tensor");
        }
        float pred = output_float[0];
        return pred > 0.5f ? CryAnnotation::CRY : CryAnnotation::NOT_CRY;
    } else {
        throw std::runtime_error("Unsupported output tensor type");
    }
}

std::vector<float> PredictCry::preprocess_audio(const std::vector<int16_t> &audio) {
    const int NUM_MFCC = 33;
    const int MAX_LENGTH = 499;

    std::vector<float> float_audio(audio.begin(), audio.end());
    // Normalize
    float max_val = 1e-8f;
    for (auto v: float_audio) {
        float abs_v = std::fabs(v);
        if(abs_v > max_val) max_val = abs_v;
    }
    for (auto &v: float_audio) {
        v /= max_val;
    }

    auto mfcc = compute_mfcc(float_audio, sample_rate_, NUM_MFCC, MAX_LENGTH);

    // Flatten (MAX_LENGTH, NUM_MFCC)
    std::vector<float> input_data;
    input_data.reserve(MAX_LENGTH * NUM_MFCC);
    for (int i = 0; i < MAX_LENGTH; i++) {
        for (int j = 0; j < NUM_MFCC; j++) {
            input_data.push_back(mfcc[i*NUM_MFCC + j]);
        }
    }

    return input_data;
}
