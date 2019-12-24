#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16

using namespace std;
using namespace cv;

texture<unsigned char, 2, cudaReadModeElementType> inTexture;

__device__ inline void sortArr(float arr[], int left, int right) {
	int i = left, j = right;
	int tmp;
	int pivot = arr[(left + right) / 2];
	while (i <= j) {
		while (arr[i] < pivot)
			i++;
		while (arr[j] > pivot)
			j--;
		if (i <= j) {
			tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
			i++;
			j--;
		}
	};
	if (left < j)
		sortArr(arr, left, j);
	if (i < right)
		sortArr(arr, i, right);
}

__global__ void gpuCalculation(unsigned char* input, unsigned char* output, int width, int height) {

	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	float sortMas[9];

	if ((x < width) && (y < height))
	{
		int vector_counter = 0;
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				sortMas[vector_counter] = tex2D(inTexture, x + j, y + i);
				vector_counter++;
			}
		}
		sortArr(sortMas, 0, 8);
		output[y * width + x] = sortMas[4];
	}
}

void medianFilter(const Mat& input, Mat& output) {

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int gray_size = input.step * input.rows;

	size_t pitch;
	unsigned char* d_input = NULL;
	unsigned char* d_output;

	cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char) * input.step, input.rows);
	cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char) * input.step, sizeof(unsigned char) * input.step, input.rows, cudaMemcpyHostToDevice); 
	cudaBindTexture2D(0, inTexture, d_input, input.step, input.rows, pitch);
	cudaMalloc<unsigned char>(&d_output, gray_size);

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	cudaEventRecord(start, 0);

	gpuCalculation <<<grid, block >>> (d_input, d_output, input.cols, input.rows);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(output.ptr(), d_output, gray_size, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventElapsedTime(&time, start, stop);
	cout << "Time on GPU: " << time << " milliseconds" << endl;
}

int main() {

	Mat input = imread("image.bmp", IMREAD_GRAYSCALE);
	//Mat output_own(input.rows, input.cols, CV_8UC1);
	//medianFilter(input, output_own);
	imwrite("result.bmp", input);
	getchar();
	return 0;
}