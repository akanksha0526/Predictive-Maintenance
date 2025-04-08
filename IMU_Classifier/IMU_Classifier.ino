/*
  BMI270 Gyroscope LSTM Anomaly Detector
  This sketch reads gyroscope data from the BMI270 sensor on Arduino Nano 33 BLE Sense
  and detects anomalies using a pre-trained LSTM model provided in model.h.
  
  The circuit:
  - Arduino Nano 33 BLE Sense board with onboard BMI270 IMU

  This example code is in the public domain.
*/

#include <Arduino_BMI270_BMM150.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Include the model data header
#include "model.h"

// Parameters for data collection
const int sequenceLength = 100;     // Length of sequence for LSTM input
const float anomalyThreshold = 0.7; // Threshold for anomaly score (adjust based on your model)
const int samplingPeriod = 10;      // Sampling period in milliseconds

// Buffers for storing gyroscope data sequence
float gyroSequence[sequenceLength][3]; // [samples][x,y,z]
int sequenceIndex = 0;
bool bufferFilled = false;

// TensorFlow Lite setup
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;

// Create a static memory buffer for TFLM
constexpr int tensorArenaSize = 32 * 1024;
uint8_t tensorArena[tensorArenaSize] _attribute_((aligned(16)));

// Function prototypes
void addToSequence(float gX, float gY, float gZ);
float runInference();
void preprocessData();

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize the BMI270 IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Print sensor details
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");
//  Serial.print("Gyroscope range = ±");
//  Serial.print(IMU.gyroscopeRange());
//  Serial.println(" degrees/second");
  
  // Initialize TensorFlow Lite model
  Serial.println("Initializing TensorFlow Lite...");
  
  // Get the TFL representation of the model from the model array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Print information about input tensor to help with debugging
  TfLiteTensor* input = tflInterpreter->input(0);
  Serial.println("Model loaded successfully!");
  Serial.print("Input tensor dimensions: [");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print(",");
  }
  Serial.println("]");
  Serial.print("Input tensor type: ");
  Serial.println(input->type);
  
  Serial.println("LSTM Gyroscope Anomaly Detector Started");
  Serial.println("X-Gyro,Y-Gyro,Z-Gyro,Anomaly_Score");
  
  // Initialize sequence buffer
  for (int i = 0; i < sequenceLength; i++) {
    for (int j = 0; j < 3; j++) {
      gyroSequence[i][j] = 0.0;
    }
  }
}

void loop() {
  float gX, gY, gZ;
  
  // Check if gyroscope data is available
  if (IMU.gyroscopeAvailable()) {
    // Read the gyroscope data
    IMU.readGyroscope(gX, gY, gZ);
    
    // Add values to sequence
    addToSequence(gX, gY, gZ);
    
    // Output current values
    Serial.print(gX);
    Serial.print(",");
    Serial.print(gY);
    Serial.print(",");
    Serial.print(gZ);
    
    if (bufferFilled) {
      // Run inference after collecting a full sequence
      if (sequenceIndex % 10 == 0) {  // Run every 10 samples to reduce computation
        float anomalyScore = runInference();
        
        // Output anomaly score
        Serial.print(",");
        Serial.println(anomalyScore, 6);
        
        // Check if an anomaly was detected
        if (anomalyScore > anomalyThreshold) {
          Serial.println("ANOMALY DETECTED!");
          Serial.print("Anomaly score: ");
          Serial.println(anomalyScore, 6);
        }
      } else {
        Serial.println(",NA");
      }
    } else {
      Serial.println(",NA");
    }
    
    // Wait for the next sample
    delay(samplingPeriod);
  }
}

// Add a new value to the sequence buffer
void addToSequence(float gX, float gY, float gZ) {
  gyroSequence[sequenceIndex][0] = gX;
  gyroSequence[sequenceIndex][1] = gY;
  gyroSequence[sequenceIndex][2] = gZ;
  
  sequenceIndex = (sequenceIndex + 1) % sequenceLength;
  
  // Mark the buffer as filled when we've collected enough samples
  if (sequenceIndex == 0 && !bufferFilled) {
    bufferFilled = true;
  }
}

// Preprocess data for the LSTM model based on input tensor shape
void preprocessData() {
  // Get the input tensor
  TfLiteTensor* input = tflInterpreter->input(0);
  
  // Adapt the preprocessing based on the actual input tensor dimensions
  // This handles both 3D [1, sequence, features] and 2D [sequence, features] formats
  
  int seq_dim = (input->dims->size == 3) ? 1 : 0;
  int feature_dim = (input->dims->size == 3) ? 2 : 1;
  
  int seq_length = input->dims->data[seq_dim];
  int feature_count = input->dims->data[feature_dim];
  
  // Ensure we don't exceed tensor dimensions
  int actual_seq_length = min(seq_length, sequenceLength);
  int actual_feature_count = min(feature_count, 3); // x, y, z gyro
  
  // Find min/max for scaling
  float minVal[3] = {1000, 1000, 1000};
  float maxVal[3] = {-1000, -1000, -1000};
  
  // Find min/max values for normalization
  for (int i = 0; i < actual_seq_length; i++) {
    for (int j = 0; j < actual_feature_count; j++) {
      if (gyroSequence[i][j] < minVal[j]) minVal[j] = gyroSequence[i][j];
      if (gyroSequence[i][j] > maxVal[j]) maxVal[j] = gyroSequence[i][j];
    }
  }
  
  // Copy data to the input tensor with appropriate normalization
  for (int i = 0; i < actual_seq_length; i++) {
    for (int j = 0; j < actual_feature_count; j++) {
      float normalized_value;
      
      // Apply normalization - if min == max, avoid division by zero
      if (maxVal[j] == minVal[j]) {
        normalized_value = 0;
      } else {
        // Scale to [-1, 1]
        normalized_value = 2.0 * (gyroSequence[i][j] - minVal[j]) / (maxVal[j] - minVal[j]) - 1.0;
      }
      
      // Copy to input tensor based on its shape
      if (input->dims->size == 3) {
        // For 3D input: [batch, sequence, features]
        input->data.f[i * feature_count + j] = normalized_value;
      } else {
        // For 2D input: [sequence, features]
        input->data.f[i * feature_count + j] = normalized_value;
      }
    }
  }
}

// Run inference with the LSTM model
float runInference() {
  // Preprocess the data
  preprocessData();
  
  // Run inference
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return -1.0; // Error value
  }
  
  // Get anomaly score from output tensor
  TfLiteTensor* output = tflInterpreter->output(0);
  
  // Assuming the model outputs a single value representing anomaly score
  return output->data.f[0];
}