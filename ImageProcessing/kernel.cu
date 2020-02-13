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

__global__ void applyMaskBlue(Pixel *input, Pixel *output, float *mask, int height, int width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int sumBlue = 0;

	for (int k = -1; k <= 1; k++)
	{
		for (int j = -1; j <= 1; j++)
		{
			if (row - k >= 0 && col + j >= 0)
				sumBlue += mask[(k + 1) * 3 + j + 1] * input[(row - k) * width + (col + j)].Blue;
		}
	}

	output[row * width + col].Blue = sumBlue;
}

__global__ void applyMaskGreen(Pixel *input, Pixel *output, float *mask, int height, int width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int sumGreen = 0;

	for (int k = -1; k <= 1; k++)
	{
		for (int j = -1; j <= 1; j++)
		{
			if (row - k >= 0 && col + j >= 0)
				sumGreen += mask[(k + 1) * 3 + j + 1] * input[(row - k) * width + (col + j)].Green;
		}
	}

	output[row * width + col].Green = sumGreen;
}

__global__ void applyMaskRed(Pixel *input, Pixel *output, float *mask, int height, int width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int sumRed = 0;

	for (int k = -1; k <= 1; k++)
	{
		for (int j = -1; j <= 1; j++)
		{
			if(row - k >= 0 && col + j >= 0)
				sumRed += mask[(k + 1) * 3 + j + 1] * input[(row - k) * width + (col + j)].Red;
		}
	}

	output[row * width + col].Red = sumRed;
}

__global__ void convertToGrayscale(Pixel *input, Pixel *output, int height, int width)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// Calculate colors
	int red = input[index].Red;
	int green = input[index].Green;
	int blue = input[index].Blue;
	int color = red * 0.3f + green * 0.59f + blue * 0.11f;

	// Set all three colors to the same value
	output[index].Red = color;
	output[index].Blue = color;
	output[index].Green = color;
}

__global__ void thresholdHalf(Pixel *input, Pixel *output, int height, int width)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (input[index].Green <= (UCHAR_MAX >> 1))
	{
		output[index].Red =	output[index].Green = output[index].Blue = 0x00;
	}
	else
	{
		output[index].Red = output[index].Green = output[index].Blue = 0xFF;
	}
}

__global__ void thresholdThird(Pixel *input, Pixel *output, int height, int width)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (input[index].Green <= 85)
	{
		output[index].Red = output[index].Green = output[index].Blue = 0x00;
	}
	else if (input[index].Green <= 170)
	{
		output[index].Red = output[index].Green = output[index].Blue = 0x7F;
	}
	else
	{
		output[index].Red = output[index].Green = output[index].Blue = 0xFF;
	}
}

__global__ void thresholdFourth(Pixel *input, Pixel *output, int height, int width)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (input[index].Green <= 63)
	{
		output[index].Red = output[index].Green = output[index].Blue = 0x00;
	}
	else if (input[index].Green <= 127)
	{
		output[index].Red = output[index].Green = output[index].Blue = 0x3F;
	}
	else if (input[index].Green <= 189)
	{
		output[index].Red = output[index].Green = output[index].Blue = 0x7F;
	}
	else
	{
		output[index].Red = output[index].Green = output[index].Blue = 0xFF;
	}
}

extern "C" _declspec(dllexport) void __cdecl ThresholdImage(uint8_t *inputImage, uint8_t *outputImage, int length, int imageHeight, int imageWidth, int bytesPerPixel, int parts)
{
	int pixelCount = length / bytesPerPixel;
	Pixel *arr = new Pixel[pixelCount];
	Pixel *outputArr = new Pixel[pixelCount];
	memset(outputArr, 0, pixelCount * sizeof(Pixel));

	// Parse byte array data into pixel array
	ParseByteArrayToPixelArray(arr, inputImage, length, bytesPerPixel);

	// Allocate device memory
	Pixel *x, *y, *z;
	cudaMallocManaged(&x, pixelCount * sizeof(Pixel));
	cudaMallocManaged(&y, pixelCount * sizeof(Pixel));
	cudaMallocManaged(&z, pixelCount * sizeof(Pixel));
	// Copy array to device memory
	cudaMemcpy(x, arr, pixelCount * sizeof(Pixel), cudaMemcpyHostToDevice);

	// Grid/Block/Thread configuration
	int blockSize = 256;
	int numBlocks = (pixelCount + blockSize - 1) / blockSize;

	// First convert to grayscale
	convertToGrayscale<<<numBlocks, blockSize>>>(x, y, imageHeight, imageWidth);

	// Then threshold the image
	switch (parts)
	{
	case 2:
		thresholdHalf<<<numBlocks, blockSize>>>(y, z, imageHeight, imageWidth);
		break;
	case 3:
		thresholdThird<<<numBlocks, blockSize>>>(y, z, imageHeight, imageWidth);
		break;
	case 4:
		thresholdFourth<<<numBlocks, blockSize>>>(y, z, imageHeight, imageWidth);
	}

	// Copy data to pixel array
	cudaMemcpy(outputArr, z, pixelCount * sizeof(Pixel), cudaMemcpyDeviceToHost);

	ParsePixelArrayToByteArray(outputArr, outputImage, length, bytesPerPixel);

	// Free memory
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
	delete[] arr;
	delete[] outputArr;
}

extern "C" _declspec(dllexport) void __cdecl ConvertImageToGrayscale(uint8_t *inputImage, uint8_t *outputImage, int length, int imageHeight, int imageWidth, int bytesPerPixel)
{
	int pixelCount = length / bytesPerPixel;
	Pixel *arr = new Pixel[pixelCount];
	Pixel *outputArr = new Pixel[pixelCount];
	memset(outputArr, 0, pixelCount * sizeof(Pixel));

	// Parse byte array data into pixel array
	ParseByteArrayToPixelArray(arr, inputImage, length, bytesPerPixel);

	// Allocate device memory, copy array to device memory
	Pixel *x, *y;
	cudaMallocManaged(&x, pixelCount * sizeof(Pixel));
	cudaMallocManaged(&y, pixelCount * sizeof(Pixel));
	cudaMemcpy(x, arr, pixelCount * sizeof(Pixel), cudaMemcpyHostToDevice);

	// Grid/Block/Thread configuration
	int blockSize = 256;
	int numBlocks = (pixelCount + blockSize - 1) / blockSize;

	convertToGrayscale<<<numBlocks, blockSize>>>(x, y, imageHeight, imageWidth);

	cudaMemcpy(outputArr, y, pixelCount * sizeof(Pixel), cudaMemcpyDeviceToHost);

	ParsePixelArrayToByteArray(outputArr, outputImage, length, bytesPerPixel);

	cudaFree(x);
	cudaFree(y);
	delete[] arr;
	delete[] outputArr;
}

extern "C" _declspec(dllexport) void __cdecl FilterImage(uint8_t *inputImage, uint8_t *outputImage, float *mask, int length, int imageHeight, int imageWidth, int bytesPerPixel)
{
	int pixelCount = length / bytesPerPixel;
	Pixel *arr = new Pixel[pixelCount];
	Pixel *outputArr = new Pixel[pixelCount];
	memset(outputArr, 0, pixelCount * sizeof(Pixel));

	// Parse byte array data into pixel array
	ParseByteArrayToPixelArray(arr, inputImage, length, bytesPerPixel);

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
	int gridDimX = sqrt(numBlocks * imageWidth / imageHeight);
	int gridDimY = sqrt(numBlocks * imageHeight / imageWidth);
	dim3 dimGrid(gridDimX + 1, gridDimY + 1); // +1 to be safe everything is processed

	// Call CUDA kernels
	applyMaskBlue<<<dimGrid, dimBlock>>>(x, y, maskArr, imageHeight, imageWidth);
	applyMaskGreen<<<dimGrid, dimBlock>>>(x, y, maskArr, imageHeight, imageWidth);
	applyMaskRed<<<dimGrid, dimBlock>>>(x, y, maskArr, imageHeight, imageWidth);

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