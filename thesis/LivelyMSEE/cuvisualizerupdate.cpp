__global__ void visualizerUpdatePreviewTexture(
	cudaSurfaceObject_t image
	, int imageWidth
	, int imageHeight
	, float *fieldData
	, int fieldWidth
	, int fieldHeight
	, float *materials
	)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int readX = (int)(x * fieldWidth * 1.f / imageWidth);
	int readY = (int)(y * fieldHeight * 1.f / imageHeight);
	float fieldValue = fieldData[readY * fieldWidth + readX];
	float cb = materials[readY * fieldWidth + readX];
	float4 color = make_float4(fieldValue, cb, 0, 1);	color.w = threadIdx.x == 0 || threadIdx.y == 0; 
	surf2Dwrite(color, image, x * sizeof(float4), y, cudaBoundaryModeClamp);
}
