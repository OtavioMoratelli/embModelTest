#include <stdio.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "modelLite.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "model.c"

constexpr int kTensorArenaSize = 81920; //80kB
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

static tflite::MicroMutableOpResolver<4> resolver;

void setupHTTP(){
    
}

void setup(){
resolver.AddFullyConnected();
resolver.AddLogistic();

}
extern "C" void app_main(void){
    const tflite::Model* model = tflite::GetModel(modelLite_tflite);
    if(model->version() != TFLITE_SCHEMA_VERSION){
        printf("model mismatch\n");
        return;
    }

    setup();

    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    if(interpreter.AllocateTensors() != kTfLiteOk){
        printf("tensors failed\n");
        return;
    }

    TfLiteTensor* input = interpreter.input(0);


    float batch[100] = {0.84285714, 0.92857143, 0.75714286, 0.72857143, 0.51428571, 0.47142857, 0.37142857, 0.27142857, 0.44285714, 0.41428571, 0.37142857, 0.5, 0.45714286, 0.28571429, 0.28571429, 0.25714286, 0.45714286, 0.31428571, 0.32857143, 0.42857143, 0.45714286, 0.28571429, 0.28571429, 0.42857143, 0.44285714, 0.31428571, 0.38571429, 0.48571429, 0.5, 0.41428571, 0.24285714, 0.5, 0.28571429, 0.31428571, 0.38571429, 0.42857143, 0.27142857, 0.5, 0.21428571, 0.44285714, 0.37142857, 0.22857143, 0.45714286, 0.35714286, 0.34285714, 0.32857143, 0.41428571, 0.25714286, 0.41428571, 0.47142857, 0.31428571, 0.27142857, 0.21428571, 0.41428571, 0.34285714, 0.48571429, 0.45714286, 0.28571429, 0.47142857, 0.44285714, 0.22857143, 0.32857143, 0.44285714, 0.38571429, 0.32857143, 0.24285714, 0.31428571, 0.27142857, 0.21428571, 0.41428571, 0.24285714, 0.3, 0.21428571, 0.25714286, 0.3, 0.28571429, 0.38571429, 0.24285714, 0.34285714, 0.45714286, 0.25714286, 0.32857143, 0.47142857, 0.48571429, 0.3, 0.3, 0.4, 0.38571429, 0.44285714, 0.3, 0.38571429, 0.45714286, 0.27142857, 0.42857143, 0.25714286, 0.45714286, 0.4, 0.41428571, 0.47142857, 0.37142857};


    while(true){
        for (int i = 0; i < 100; i++){
            input->data.f[0] = batch[i];
        
        

            if(interpreter.Invoke() != kTfLiteOk){
                printf("invoke failed\n");
                continue;
            }

            TfLiteTensor* output = interpreter.output(0);
            float prediction = output->data.f[0];
            printf("input: %d, predicition: %f\n", i, prediction);
        }

        vTaskDelay(pdMS_TO_TICKS(10000));
    }


}