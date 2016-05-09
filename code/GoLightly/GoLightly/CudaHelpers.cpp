#include "CudaHelpers.h"

#include <iostream>
using namespace std;



extern "C" void Check(cudaError_t err = cudaSuccess)
{
	if (err == -1)
		err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		cerr << "CUDA call returned error: (" << (int)err << ") \"" << cudaGetErrorString(err) << "\"" << endl;
		exit(EXIT_FAILURE);
	}
}
