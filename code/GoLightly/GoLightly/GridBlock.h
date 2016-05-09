#ifndef GRIDBLOCK_H
#define GRIDBLOCK_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
using namespace std;

class GridBlock
{
public:
	dim3 GridDim;
	dim3 BlockDim;
	dim3 GridOffset;

	GridBlock();
    GridBlock(int gridWidth, int gridHeight, int blockWidth, int blockHeight, int offsetX = 0, int offsetY = 0);

	static vector<GridBlock> Divide(dim3 fieldSize, dim3 updateRangeStart, dim3 updateRangeEnd, dim3 blockSize, bool preferLinear = false);

};

#endif