// ファイル: cpp/cuda/resize_kernel.cu
// 時間: 3-4時間 | 優先度: 🔴 最高

#include <cuda_runtime.h>
#include <cuda.h>

__global__ void resize_bilinear_kernel(
    const unsigned char* input,
    unsigned char* output,
    int input_width, int input_height,
    int output_width, int output_height,
    int channels
) {
    int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (output_x >= output_width || output_y >= output_height) return;
    
    // バイリニア補間
    float scale_x = (float)input_width / output_width;
    float scale_y = (float)input_height / output_height;
    
    float input_x = output_x * scale_x;
    float input_y = output_y * scale_y;
    
    int x1 = (int)input_x;
    int y1 = (int)input_y;
    int x2 = min(x1 + 1, input_width - 1);
    int y2 = min(y1 + 1, input_height - 1);
    
    float dx = input_x - x1;
    float dy = input_y - y1;
    
    for (int c = 0; c < channels; c++) {
        float val = 
            input[(y1 * input_width + x1) * channels + c] * (1-dx) * (1-dy) +
            input[(y1 * input_width + x2) * channels + c] * dx * (1-dy) +
            input[(y2 * input_width + x1) * channels + c] * (1-dx) * dy +
            input[(y2 * input_width + x2) * channels + c] * dx * dy;
            
        output[(output_y * output_width + output_x) * channels + c] = 
            (unsigned char)val;
    }
}

extern "C" {
    void resize_gpu(
        const unsigned char* input,
        unsigned char* output,
        int input_width, int input_height,
        int output_width, int output_height,
        int channels
    ) {
        dim3 blockSize(16, 16);
        dim3 gridSize(
            (output_width + blockSize.x - 1) / blockSize.x,
            (output_height + blockSize.y - 1) / blockSize.y
        );
        
        resize_bilinear_kernel<<<gridSize, blockSize>>>(
            input, output,
            input_width, input_height,
            output_width, output_height,
            channels
        );
        
        cudaDeviceSynchronize();
    }
}