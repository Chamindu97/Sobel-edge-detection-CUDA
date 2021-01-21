#include <iostream>
#include <cuda_runtime.h>
#include<opencv2/opencv.hpp>


using namespace cv;
using namespace std;

// GPU constant memory to hold our kernels (extremely fast access time)
__constant__ float convolutionKernelStore[256];

__global__ void convolve(uchar *source, int width, int height, int paddingX, int paddingY, unsigned int kOffset, int kWidth, int kHeight, uchar *destination)
{
	// Calculate our pixel's location
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float sum = 0.0;
	int   pWidth = kWidth / 2;
	int   pHeight = kHeight / 2;

	//Only execute valid pixels
	if (x >= pWidth + paddingX && y >= pHeight + paddingY && x < (blockDim.x * gridDim.x) - pWidth - paddingX &&
		y < (blockDim.y * gridDim.y) - pHeight - paddingY)
	{
		for (int j = -pHeight; j <= pHeight; j++)
		{
			for (int i = -pWidth; i <= pWidth; i++)
			{
				// Sample the weight for this location
				int ki = (i + pWidth);
				int kj = (j + pHeight);
				float w = convolutionKernelStore[(kj * kWidth) + ki + kOffset];


				sum += w * float(source[((y + j) * width) + (x + i)]);
			}
		}
	}

	// Average sum
	destination[(y * width) + x] = (uchar)sum;
}

__global__ void pythagoras(uchar *a, uchar *b, uchar *c)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float af = float(a[idx]);
	float bf = float(b[idx]);

	c[idx] = (uchar)sqrtf(af*af + bf * bf);
}

// create image buffer
uchar* createImageBuffer(unsigned int bytes, uchar **devicePtr)
{
	uchar *ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}


int main(int argc, char** argv) {

	// Open the web camera
	VideoCapture cap("C:/Users/chami/Desktop/videos/wildlife_1080p.mp4");
	Mat frame;
	if (cap.isOpened() == false) {
		cout << "Could not initialize capturing...\n" << endl;
		return -1;
	}

	// Functions to get the execution time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Create gaussian kernel (sum = 159)
	const float gaussianKernel5x5[25] =
	{
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
	};
	cudaMemcpyToSymbol(convolutionKernelStore, gaussianKernel5x5, sizeof(gaussianKernel5x5), 0);
	const unsigned int gaussianKernel5x5Offset = 0;

	// Sobel gradient kernels
	const float sobelGradientX[9] =
	{
		-1.f,  0.f,  1.f,
		-2.f,  0.f,  2.f,
		-1.f,  0.f,  1.f,
	};
	const float sobelGradientY[9] =
	{
		 1.f,  2.f,  1.f,
		 0.f,  0.f,  0.f,
		-1.f, -2.f, -1.f,
	};

	cudaMemcpyToSymbol(convolutionKernelStore, sobelGradientX, sizeof(sobelGradientX), sizeof(gaussianKernel5x5));
	cudaMemcpyToSymbol(convolutionKernelStore, sobelGradientY, sizeof(sobelGradientY), sizeof(gaussianKernel5x5) + sizeof(sobelGradientX));
	const unsigned int sobelGradientXOffset = sizeof(gaussianKernel5x5) / sizeof(float);
	const unsigned int sobelGradientYOffset = sizeof(sobelGradientX) / sizeof(float) + sobelGradientXOffset;

	// Create shared CPU / GPU images
	cap >> frame;
	uchar *sourceDataDevice, *blurredDataDevice, *edgesDataDevice;
	Mat source(frame.size(), CV_8U, createImageBuffer(frame.cols * frame.rows, &sourceDataDevice));
	Mat blurred(frame.size(), CV_8U, createImageBuffer(frame.cols * frame.rows, &blurredDataDevice));
	Mat edges(frame.size(), CV_8U, createImageBuffer(frame.cols * frame.rows, &edgesDataDevice));

	// Create 2 temporary images (sobel gradients)
	uchar *deviceGradientX, *deviceGradientY;
	cudaMalloc(&deviceGradientX, frame.cols * frame.rows);
	cudaMalloc(&deviceGradientY, frame.cols * frame.rows);

	// Loop while capturing images
	while (1)
	{
		// Capture the image in grayscale
		cap >> frame;
		if (frame.empty())
			break;

		cvtColor(frame, source, COLOR_BGR2GRAY);

		// Record the time the process takes
		cudaEventRecord(start);
		{
			// convolution kernel  parametros
			dim3 cthreads(32, 32);
			dim3 cblocks(frame.cols / cthreads.x, frame.rows / cthreads.y);
			
			// pythagoran kernel parametros
			dim3 pthreads(1024,1);
			dim3 pblocks(frame.cols * frame.rows / 1024);
			
			//  gaussian blur (first kernel in store @ 0)
			convolve << < cblocks, cthreads >> > (sourceDataDevice, frame.cols, frame.rows, 0, 0, gaussianKernel5x5Offset, 5, 5, blurredDataDevice);

			// sobel gradient convolutions (x&y padding is now 2 because there is a border of 2 around a 5x5 gaussian filtered image)
			convolve << < cblocks, cthreads >> > (blurredDataDevice, frame.cols, frame.rows, 2, 2, sobelGradientXOffset, 3, 3, deviceGradientX);
			convolve << < cblocks, cthreads >> > (blurredDataDevice, frame.cols, frame.rows, 2, 2, sobelGradientYOffset, 3, 3, deviceGradientY);
			pythagoras << < pblocks, pthreads >> > (deviceGradientX, deviceGradientY, edgesDataDevice);

			cudaDeviceSynchronize();
		}
		cudaEventRecord(stop);

		// Sample run time
		float ms = 0.0f;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		cout << "Process time: " << ms << endl;
		
		namedWindow("Original", WINDOW_NORMAL);
		imshow("Original", frame);

		namedWindow("blurred", WINDOW_NORMAL);
		imshow("blurred", blurred);

		// Show results
		namedWindow("Sobel", WINDOW_NORMAL);
		imshow("Sobel", edges);

		// Spin
		int c = waitKey(1) & 0xff;
		if (c == 'q' || c == 'Q' || c == 27) {
			break;
		}	
	}

	// Exit
	cudaFreeHost(source.data);
	cudaFreeHost(blurred.data);
	cudaFreeHost(edges.data);
	cudaFree(deviceGradientX);
	cudaFree(deviceGradientY);

	return 0;
}