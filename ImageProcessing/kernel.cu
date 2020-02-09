#include <iostream>
#include <math.h>
#include <inttypes.h>
#include "cuda_runtime.h"

#pragma pack(push)
#pragma pack(1)
typedef struct {
	uint8_t Blue;
	uint8_t Green;
	uint8_t Red;
} Pixel;
#pragma pack(pop)

void ParseByteArrayToPixelArray(Pixel *arr, uint8_t *byteArray, int length, int bytesPerPixel);
void ParsePixelArrayToByteArray(Pixel *arr, uint8_t *byteArray, int length, int bytesPerPixel);

__global__ void add(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}

__global__ void setBlue(Pixel *x, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n)
	{
		x[index].Blue = 0;
	}
}

__global__ void setGreen(Pixel *x, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n)
	{
		x[index].Green = 0;
	}
}

__global__ void setRed(Pixel *x, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n)
	{
		x[index].Red = 0;
	}
}

__global__ void blurBlue(Pixel *input, Pixel *output, float *mask, int height, int width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Check boundaries
	if (row <= 0 || row >= height - 1 || col <= 0 || col >= width -1)
	{
		return;
	}

	int sumBlue = 0;

	for (int k = -1; k <= 1; k++)
	{
		for (int j = -1; j <= 1; j++)
		{
			sumBlue += mask[(k + 1) * 3 + j + 1] * input[(row - k) * width + (col + j)].Blue;
		}
	}

	output[row * width + col].Blue = sumBlue;
}

__global__ void blurGreen(Pixel *input, Pixel *output, float *mask, int height, int width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Check boundaries
	if (row <= 0 || row >= height - 1 || col <= 0 || col >= width - 1)
	{
		return;
	}

	int sumGreen = 0;

	for (int k = -1; k <= 1; k++)
	{
		for (int j = -1; j <= 1; j++)
		{
			sumGreen += mask[(k + 1) * 3 + j + 1] * input[(row - k) * width + (col + j)].Green;
		}
	}

	output[row * width + col].Green = sumGreen;
}

__global__ void blurRed(Pixel *input, Pixel *output, float *mask, int height, int width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Check boundaries
	if (row <= 0 || row >= height - 1 || col <= 0 || col >= width - 1)
	{
		return;
	}

	int sumRed = 0;

	for (int k = -1; k <= 1; k++)
	{
		for (int j = -1; j <= 1; j++)
		{
			sumRed += mask[(k + 1) * 3 + j + 1] * input[(row - k) * width + (col + j)].Red;
		}
	}

	output[row * width + col].Red = sumRed;
}

extern "C" _declspec(dllexport) void __cdecl BlurImage(uint8_t *inputImage, uint8_t *outputImage, int length, int imageHeight, int imageWidth, int bytesPerPixel)
{
	int pixelCount = length / bytesPerPixel;
	Pixel *arr = new Pixel[pixelCount];
	Pixel *outputArr = new Pixel[pixelCount];
	memset(outputArr, 0, pixelCount * sizeof(Pixel));

	// Parse byte array data into pixel array
	ParseByteArrayToPixelArray(arr, inputImage, length, bytesPerPixel);

	float mask[9] = {
		1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
		1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
		1 / 9.0f, 1 / 9.0f, 1 / 9.0f
	};

	// Allocate device memory, copy array to device memory
	Pixel *x, *y;
	cudaMallocManaged(&x, pixelCount * sizeof(Pixel));
	cudaMallocManaged(&y, pixelCount * sizeof(Pixel));
	cudaMemcpy(x, arr, pixelCount * sizeof(Pixel), cudaMemcpyHostToDevice);

	float *maskArr;
	cudaMallocManaged(&maskArr, 9 * sizeof(float));
	cudaMemcpy(maskArr, mask, 9 * sizeof(float), cudaMemcpyHostToDevice);

	// Grid/Block/Thread configuration
	dim3 dimBlock(16, 16);
	int blockSize = 16 * 16;
	int numBlocks = (pixelCount + blockSize - 1) / blockSize;
	int blockGridDim = sqrt(numBlocks);
	dim3 dimGrid(blockGridDim + 1, blockGridDim + 1);

	// Call CUDA kernels
	blurBlue<<<dimGrid, dimBlock>>>(x, y, maskArr, imageHeight, imageWidth);
	blurGreen<<<dimGrid, dimBlock>>>(x, y, maskArr, imageHeight, imageWidth);
	blurRed<<<dimGrid, dimBlock>>>(x, y, maskArr, imageHeight, imageWidth);

	// Wait for CUDA to finish
	//cudaDeviceSynchronize(); // not needed

	// Copy calculated data back to host memory
	cudaMemcpy(outputArr, y, pixelCount * sizeof(Pixel), cudaMemcpyDeviceToHost);

	ParsePixelArrayToByteArray(outputArr, outputImage, length, bytesPerPixel);

	// Free memory
	cudaFree(x);
	cudaFree(y);
	cudaFree(maskArr);
	delete[] arr;
	delete[] outputArr;
}

extern "C" _declspec(dllexport) void __cdecl RemoveColorFromImage(uint8_t *imageData, int length, int imageHeight, int imageWidth, int bytesPerPixel, int color)
{
	int pixelCount = length / bytesPerPixel;
	Pixel *arr = new Pixel[pixelCount];

	// Parse byte array data into pixel array
	ParseByteArrayToPixelArray(arr, imageData, length, bytesPerPixel);

	// Allocate device memory, copy array to device memory
	Pixel *x;
	cudaMallocManaged(&x, imageHeight * imageWidth * sizeof(Pixel));
	cudaMemcpy(x, arr, pixelCount * sizeof(Pixel), cudaMemcpyHostToDevice);

	// Set grid/block/thread configuration
	int blockSize = 256;
	int numBlocks = (pixelCount + blockSize - 1) / blockSize;

	// Call CUDA kernels
	switch (color)
	{
	case 0:
		setBlue<<<numBlocks, blockSize>>>(x, pixelCount);
		break;
	case 1:
		setGreen<<<numBlocks, blockSize>>>(x, pixelCount);
		break;
	case 2:
		setRed<<<numBlocks, blockSize>>>(x, pixelCount);
		break;
	}

	// Wait for CUDA to finish
	cudaDeviceSynchronize();

	// Copy calculated data back to host memory
	cudaMemcpy(arr, x, pixelCount * sizeof(Pixel), cudaMemcpyDeviceToHost);

	// Parse back to byte array
	ParsePixelArrayToByteArray(arr, imageData, length, bytesPerPixel);

	// Free memory
	cudaFree(x);
	delete[] arr;
}

void ParseByteArrayToPixelArray(Pixel *arr, uint8_t *byteArray, int length, int bytesPerPixel)
{
	for (int i = 0, j = 0; i < length; i += bytesPerPixel, j++)
	{
		arr[j].Blue = byteArray[i];
		arr[j].Green = byteArray[i + 1];
		arr[j].Red = byteArray[i + 2];
	}
}

void ParsePixelArrayToByteArray(Pixel *arr, uint8_t *byteArray, int length, int bytesPerPixel)
{
	for (int i = 0, j = 0; i < length; i += bytesPerPixel, j++)
	{
		byteArray[i] = arr[j].Blue;
		byteArray[i + 1] = arr[j].Green;
		byteArray[i + 2] = arr[j].Red;
	}
}