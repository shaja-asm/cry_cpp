#include "PredictCry.h"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <iostream>
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "utils.h" // For MFCC and related preprocessing

static const char* MODEL_PATH = "/home/root/basestation/basestation/algorithm/cry_detection/cnn_cry_detection_model_quant_shifted.tflite";

PredictCry::PredictCry(int num_threads, int sample_rate)
    : sample_rate_(sample_rate), num_threads_(num_threads) {
    if (!init_tflite()) {
        throw std::runtime_error("Failed to initialize TFLite interpreter");
    }
}

PredictCry::~PredictCry() {
    // Smart pointers automatically delete the TFLite objects
}

bool PredictCry::init_tflite() {
    // Load the TFLite model
    model_.reset(TfLiteModelCreateFromFile(MODEL_PATH));
    if (!model_) {
        std::cerr << "Failed to load TFLite model from: " << MODEL_PATH << std::endl;
        return false;
    }

    // Create interpreter options
    options_.reset(TfLiteInterpreterOptionsCreate());
    TfLiteInterpreterOptionsSetNumThreads(options_.get(), num_threads_);

    // Create interpreter
    interpreter_.reset(TfLiteInterpreterCreate(model_.get(), options_.get()));
    if (!interpreter_) {
        std::cerr << "Failed to create TFLite interpreter." << std::endl;
        return false;
    }

    // Allocate tensors
    if (TfLiteInterpreterAllocateTensors(interpreter_.get()) != kTfLiteOk) {
        std::cerr << "Failed to allocate TFLite tensors." << std::endl;
        return false;
    }

    // Get input and output tensors
    const TfLiteTensor* input_tensor_ = TfLiteInterpreterGetInputTensor(interpreter_.get(), 0);
    const TfLiteTensor* output_tensor_ = TfLiteInterpreterGetOutputTensor(interpreter_.get(), 0);
    if (!input_tensor_ || !output_tensor_) {
        std::cerr << "Failed to get input/output tensors." << std::endl;
        return false;
    }

    // Get tensor data types
    input_dtype_ = TfLiteTensorType(input_tensor_);
    output_dtype_ = TfLiteTensorType(output_tensor_);

    // Get quantization parameters
    auto input_params = TfLiteTensorQuantizationParams(input_tensor_);
    input_scale_ = input_params.scale;
    input_zero_point_ = input_params.zero_point;

    auto output_params = TfLiteTensorQuantizationParams(output_tensor_);
    output_scale_ = output_params.scale;
    output_zero_point_ = output_params.zero_point;

    // Check input dimensions match expected shape
    // Expected shape: [1, 499, 33, 1]
    const int MAX_LENGTH = 499;
    const int NUM_MFCC = 33;

    const TfLiteIntArray* dims = input_tensor_->dims;
    if (dims->size != 4 ||
        dims->data[0] != 1 || dims->data[1] != MAX_LENGTH ||
        dims->data[2] != NUM_MFCC || dims->data[3] != 1) 
    {
        std::cerr << "Input tensor has unexpected shape: [";
        for (int i = 0; i < dims->size; i++) {
            std::cerr << dims->data[i] << (i < dims->size - 1 ? ", " : "");
        }
        std::cerr << "], expected [1, 499, 33, 1]." << std::endl;
        return false;
    }

    return true;
}

CryAnnotation PredictCry::get_prediction(const std::vector<int16_t> &audio) const {
    return predict(audio);
}

CryAnnotation PredictCry::predict(const std::vector<int16_t> &audio) const { // Marked as const
    std::vector<float> input_data = preprocess_audio(audio);
    // input_data now has size MAX_LENGTH * NUM_MFCC
    // Model expects shape: [1, MAX_LENGTH, NUM_MFCC, 1]
    // The total number of elements: 1*499*33*1 = 16467 floats.

    const int TF_LITE_TYPE_FLOAT32 = 1;
    const int TF_LITE_TYPE_INT8 = 9;

    // Copy data into the input tensor in the format the model expects.
    // The memory layout is already correct: we have a flat vector in row-major order.
    // Just quantize if needed and copy.

    if (input_dtype_ == TF_LITE_TYPE_INT8) {
        std::vector<int8_t> quantized_data(input_data.size());
        for (size_t i = 0; i < input_data.size(); i++) {
            float val = (input_data[i] / input_scale_) + input_zero_point_;
            val = std::round(val);
            if (val < -128) val = -128;
            if (val > 127) val = 127;
            quantized_data[i] = static_cast<int8_t>(val);
        }

        if (TfLiteTensorCopyFromBuffer(input_tensor_, quantized_data.data(), quantized_data.size()) != kTfLiteOk) {
            throw std::runtime_error("Failed to copy quantized data to input tensor");
        }

    } else if (input_dtype_ == TF_LITE_TYPE_FLOAT32) {
        // Directly copy float data
        size_t tensor_data_size = input_data.size() * sizeof(float);
        if (TfLiteTensorCopyFromBuffer(input_tensor_, input_data.data(), tensor_data_size) != kTfLiteOk) {
            throw std::runtime_error("Failed to copy float data to input tensor");
        }
    } else {
        throw std::runtime_error("Unsupported input tensor data type");
    }

    // Run inference
    if (TfLiteInterpreterInvoke(interpreter_.get()) != kTfLiteOk) {
        throw std::runtime_error("Failed to invoke TFLite interpreter");
    }

    int output_byte_size = static_cast<int>(TfLiteTensorByteSize(output_tensor_));
    if (output_dtype_ == TF_LITE_TYPE_INT8) {
        std::vector<int8_t> output_int8(output_byte_size);
        if (TfLiteTensorCopyToBuffer(output_tensor_, output_int8.data(), output_int8.size()) != kTfLiteOk) {
            throw std::runtime_error("Failed to copy from output tensor");
        }
        float pred = (static_cast<float>(output_int8[0]) - output_zero_point_) * output_scale_;
        return (pred > 0.5f) ? CryAnnotation::CRY : CryAnnotation::NOT_CRY;
    } else if (output_dtype_ == TF_LITE_TYPE_FLOAT32) {
        std::vector<float> output_float(output_byte_size / sizeof(float));
        if (TfLiteTensorCopyToBuffer(output_tensor_, output_float.data(), output_byte_size) != kTfLiteOk) {
            throw std::runtime_error("Failed to copy from output tensor");
        }
        float pred = output_float[0];
        return (pred > 0.5f) ? CryAnnotation::CRY : CryAnnotation::NOT_CRY;
    } else {
        throw std::runtime_error("Unsupported output tensor type");
    }
}

std::vector<float> PredictCry::preprocess_audio(const std::vector<int16_t> &audio) const { // Marked as const
    const int NUM_MFCC = 33;
    const int MAX_LENGTH = 499;

    // Convert int16_t audio to float
    std::vector<float> float_audio(audio.begin(), audio.end());

    // Normalize: y = y / (max(abs(y)) + 1e-8)
    float max_val = 0.0f;
    for (auto v : float_audio) {
        float abs_v = std::fabs(v);
        if (abs_v > max_val) max_val = abs_v;
    }
    max_val += 1e-8f; 
    for (auto &v : float_audio) {
        v /= max_val;
    }

    // Compute MFCC: returns a 1D vector of size (frames*NUM_MFCC)
    // Adjust compute_mfcc to allow variable frames if needed.
    std::vector<float> mfcc = compute_mfcc(float_audio, sample_rate_, NUM_MFCC, -1);
    int frames = static_cast<int>(mfcc.size() / NUM_MFCC);

    // Pad or truncate to MAX_LENGTH
    if (frames < MAX_LENGTH) {
        // Pad with zeros
        mfcc.resize(MAX_LENGTH * NUM_MFCC, 0.0f);
    } else if (frames > MAX_LENGTH) {
        // Truncate
        mfcc.resize(MAX_LENGTH * NUM_MFCC);
    }

    // Python code: mfcc = mfcc[..., np.newaxis]
    // This would add a channel dimension.
    // The model expects [1, MAX_LENGTH, NUM_MFCC, 1].
    // Our mfcc is currently [MAX_LENGTH*NUM_MFCC].
    // The memory layout is the same as [1, MAX_LENGTH, NUM_MFCC, 1].
    // We'll rely on the model and input dims we checked to interpret it correctly.

    return mfcc;
}
