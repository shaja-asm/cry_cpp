#include "PredictCry.h"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <iostream>
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "utils.h"

static const char* MODEL_PATH = "/home/root/basestation/basestation/algorithm/cry_detection/cnn_cry_detection_model_quant_shifted.tflite";

PredictCry::PredictCry(int num_threads, int sample_rate)
    : sample_rate_(sample_rate), num_threads_(num_threads) {
    // std::cerr << "[DEBUG] Entering PredictCry constructor.\n";
    if (!init_tflite()) {
        std::cerr << "[ERROR] Failed to initialize TFLite interpreter.\n";
        throw std::runtime_error("Failed to initialize TFLite interpreter");
    }
    // std::cerr << "[DEBUG] Successfully initialized PredictCry.\n";
}

PredictCry::~PredictCry() {
    std::cerr << "[DEBUG] Destroying PredictCry object.\n";
}

bool PredictCry::init_tflite() {
    // std::cerr << "[DEBUG] Loading TFLite model from: " << MODEL_PATH << "\n";
    model_.reset(TfLiteModelCreateFromFile(MODEL_PATH));
    if (!model_) {
        std::cerr << "[ERROR] Failed to load TFLite model.\n";
        return false;
    }

    // std::cerr << "[DEBUG] Creating interpreter options.\n";
    options_.reset(TfLiteInterpreterOptionsCreate());
    TfLiteInterpreterOptionsSetNumThreads(options_.get(), num_threads_);

    // std::cerr << "[DEBUG] Creating interpreter.\n";
    interpreter_.reset(TfLiteInterpreterCreate(model_.get(), options_.get()));
    if (!interpreter_) {
        std::cerr << "[ERROR] Failed to create TFLite interpreter.\n";
        return false;
    }

    // std::cerr << "[DEBUG] Allocating TFLite tensors.\n";
    if (TfLiteInterpreterAllocateTensors(interpreter_.get()) != kTfLiteOk) {
        std::cerr << "[ERROR] Failed to allocate TFLite tensors.\n";
        return false;
    }

    // std::cerr << "[DEBUG] Getting input and output tensors.\n";
    input_tensor_ = TfLiteInterpreterGetInputTensor(interpreter_.get(), 0);
    output_tensor_ = TfLiteInterpreterGetOutputTensor(interpreter_.get(), 0);
    if (!input_tensor_ || !output_tensor_) {
        std::cerr << "[ERROR] Failed to get input/output tensors.\n";
        return false;
    }

    input_dtype_ = TfLiteTensorType(input_tensor_);
    output_dtype_ = TfLiteTensorType(output_tensor_);

    auto input_params = TfLiteTensorQuantizationParams(input_tensor_);
    input_scale_ = input_params.scale;
    input_zero_point_ = input_params.zero_point;

    auto output_params = TfLiteTensorQuantizationParams(output_tensor_);
    output_scale_ = output_params.scale;
    output_zero_point_ = output_params.zero_point;

    // std::cerr << "[DEBUG] Checking input tensor dimensions.\n";
    const int MAX_LENGTH = 499;
    const int NUM_MFCC = 33;

    const TfLiteIntArray* dims = input_tensor_->dims;
    if (dims->size != 4 ||
        dims->data[0] != 1 || dims->data[1] != MAX_LENGTH ||
        dims->data[2] != NUM_MFCC || dims->data[3] != 1) 
    {
        std::cerr << "[ERROR] Input tensor has unexpected shape: [";
        for (int i = 0; i < dims->size; i++) {
            std::cerr << dims->data[i] << (i < dims->size - 1 ? ", " : "");
        }
        std::cerr << "], expected [1, 499, 33, 1].\n";
        return false;
    }

    // std::cerr << "[DEBUG] init_tflite completed successfully.\n";
    return true;
}

CryAnnotation PredictCry::get_prediction(const std::vector<int16_t> &audio) {
    // std::cerr << "[DEBUG] Entering get_prediction.\n";
    CryAnnotation result = predict(audio);
    // std::cerr << "[DEBUG] Exiting get_prediction with result: " << static_cast<int>(result) << "\n";
    return result;
}

CryAnnotation PredictCry::predict(const std::vector<int16_t> &audio) {
    // std::cerr << "[DEBUG] Entering predict.\n";
    // std::cerr << "[DEBUG] Preprocessing audio.\n";
    std::vector<float> input_data = preprocess_audio(audio);
    // std::cerr << "[DEBUG] Preprocessing completed. Input data size: " << input_data.size() << "\n";

    const int TF_LITE_TYPE_FLOAT32 = 1;
    const int TF_LITE_TYPE_INT8 = 9;

    TfLiteTensor* mutable_input_tensor = const_cast<TfLiteTensor*>(input_tensor_);
    if (!mutable_input_tensor) {
        std::cerr << "[ERROR] mutable_input_tensor is null.\n";
        throw std::runtime_error("Input tensor is null");
    }

    // std::cerr << "[DEBUG] Copying input data to input tensor.\n";
    if (input_dtype_ == TF_LITE_TYPE_INT8) {
        std::vector<int8_t> quantized_data(input_data.size());
        for (size_t i = 0; i < input_data.size(); i++) {
            float val = (input_data[i] / input_scale_) + input_zero_point_;
            val = std::round(val);
            if (val < -128) val = -128;
            if (val > 127) val = 127;
            quantized_data[i] = static_cast<int8_t>(val);
        }

        if (TfLiteTensorCopyFromBuffer(mutable_input_tensor, quantized_data.data(), quantized_data.size()) != kTfLiteOk) {
            std::cerr << "[ERROR] Failed to copy quantized data to input tensor.\n";
            throw std::runtime_error("Failed to copy quantized data to input tensor");
        }

    } else if (input_dtype_ == TF_LITE_TYPE_FLOAT32) {
        size_t tensor_data_size = input_data.size() * sizeof(float);
        if (TfLiteTensorCopyFromBuffer(mutable_input_tensor, input_data.data(), tensor_data_size) != kTfLiteOk) {
            std::cerr << "[ERROR] Failed to copy float data to input tensor.\n";
            throw std::runtime_error("Failed to copy float data to input tensor");
        }
    } else {
        std::cerr << "[ERROR] Unsupported input tensor data type.\n";
        throw std::runtime_error("Unsupported input tensor data type");
    }

    // std::cerr << "[DEBUG] Invoking interpreter.\n";
    if (TfLiteInterpreterInvoke(interpreter_.get()) != kTfLiteOk) {
        std::cerr << "[ERROR] Failed to invoke TFLite interpreter.\n";
        throw std::runtime_error("Failed to invoke TFLite interpreter");
    }
    // std::cerr << "[DEBUG] Inference completed successfully.\n";

    // std::cerr << "[DEBUG] Re-fetching the output tensor.\n";
    output_tensor_ = TfLiteInterpreterGetOutputTensor(interpreter_.get(), 0);
    if (!output_tensor_) {
        std::cerr << "[ERROR] Failed to get output tensor after invocation.\n";
        throw std::runtime_error("Failed to get output tensor after invocation");
    }

    TfLiteTensor* mutable_output_tensor = const_cast<TfLiteTensor*>(output_tensor_);
    if (!mutable_output_tensor) {
        std::cerr << "[ERROR] mutable_output_tensor is null.\n";
        throw std::runtime_error("Output tensor is null");
    }

    int output_byte_size = static_cast<int>(TfLiteTensorByteSize(output_tensor_));
    // std::cerr << "[DEBUG] Output tensor byte size: " << output_byte_size << "\n";

    if (output_dtype_ == TF_LITE_TYPE_INT8) {
        // std::cerr << "[DEBUG] Reading int8 output.\n";
        std::vector<int8_t> output_int8(output_byte_size);
        if (TfLiteTensorCopyToBuffer(mutable_output_tensor, output_int8.data(), output_int8.size()) != kTfLiteOk) {
            std::cerr << "[ERROR] Failed to copy from output tensor (int8).\n";
            throw std::runtime_error("Failed to copy from output tensor");
        }
        float pred = (static_cast<float>(output_int8[0]) - output_zero_point_) * output_scale_;
        // std::cerr << "[DEBUG] Prediction value: " << pred << "\n";
        CryAnnotation result = (pred > 0.5f) ? CryAnnotation::CRY : CryAnnotation::NOT_CRY;
        std::cerr << "[DEBUG] Exiting predict with result: " << static_cast<int>(result) << "\n";
        return result;

    } else if (output_dtype_ == TF_LITE_TYPE_FLOAT32) {
        // std::cerr << "[DEBUG] Reading float32 output.\n";
        std::vector<float> output_float(output_byte_size / sizeof(float));
        if (TfLiteTensorCopyToBuffer(mutable_output_tensor, output_float.data(), output_byte_size) != kTfLiteOk) {
            std::cerr << "[ERROR] Failed to copy from output tensor (float32).\n";
            throw std::runtime_error("Failed to copy from output tensor");
        }
        float pred = output_float[0];
        // std::cerr << "[DEBUG] Prediction value: " << pred << "\n";
        CryAnnotation result = (pred > 0.5f) ? CryAnnotation::CRY : CryAnnotation::NOT_CRY;
        std::cerr << "[DEBUG] Exiting predict with result: " << static_cast<int>(result) << "\n";
        return result;

    } else {
        std::cerr << "[ERROR] Unsupported output tensor type.\n";
        throw std::runtime_error("Unsupported output tensor type");
    }
}

std::vector<float> PredictCry::preprocess_audio(const std::vector<int16_t> &audio) const {
    // std::cerr << "[DEBUG] Entering preprocess_audio.\n";
    const int NUM_MFCC = 33;
    const int MAX_LENGTH = 499;

    std::vector<float> float_audio(audio.begin(), audio.end());
    float max_val = 0.0f;
    for (auto v : float_audio) {
        float abs_v = std::fabs(v);
        if (abs_v > max_val) max_val = abs_v;
    }
    max_val += 1e-8f; 
    for (auto &v : float_audio) {
        v /= max_val;
    }

    // std::cerr << "[DEBUG] Computing MFCC.\n";
    std::vector<float> mfcc = compute_mfcc(float_audio, sample_rate_, NUM_MFCC, -1);
    int frames = static_cast<int>(mfcc.size() / NUM_MFCC);
    // std::cerr << "[DEBUG] MFCC frames: " << frames << "\n";

    if (frames < MAX_LENGTH) {
        mfcc.resize(MAX_LENGTH * NUM_MFCC, 0.0f);
        // std::cerr << "[DEBUG] Padded MFCC to " << MAX_LENGTH << " frames.\n";
    } else if (frames > MAX_LENGTH) {
        mfcc.resize(MAX_LENGTH * NUM_MFCC);
        // std::cerr << "[DEBUG] Truncated MFCC to " << MAX_LENGTH << " frames.\n";
    }

    // std::cerr << "[DEBUG] Exiting preprocess_audio.\n";
    return mfcc;
}
