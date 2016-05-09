/// attempt to divide into roughly equal sections
//#define QUADRIFY_GRID

#include "GridBlock.h"

GridBlock::GridBlock() :
	GridDim(0,0,0)
	,BlockDim(0,0,0)
	,GridOffset(0,0,0)
{
}

GridBlock::GridBlock(int gridWidth, int gridHeight, int blockWidth, int blockHeight, int offsetX, int offsetY)
	:
	GridDim(gridWidth,gridHeight)
	,BlockDim(blockWidth,blockHeight)
	,GridOffset(offsetX,offsetY)
{
}


/*
---------------------
|         |         |
|         |         |
|   q0    |   q1    |
|         |         |
---------------------
|         |         |
|   q2    |   q3    |
|         |         |
|         |         |
---------------------
*/

#ifdef QUADRIFY_GRID

vector<GridBlock> GridBlock::Divide(dim3 fieldSize, dim3 updateRangeStart, dim3 updateRangeEnd, dim3 blockSize, bool preferLinear)
{
	vector<GridBlock> blocks;

	if (preferLinear)
	{
		unsigned int count = fieldSize.x;
		unsigned int maxGridWidth = 1024;
		unsigned int blockWidth = blockSize.x;
		unsigned int offset = 0;
		unsigned int delta = 0;
		while (count > 0)
		{
			if (count  / blockSize.x > 0)
			{
				unsigned int gridX = min<unsigned int>(count / blockSize.x,maxGridWidth);
				blocks.push_back(GridBlock(gridX,1,blockSize.x,1,offset,0));

				delta = gridX * blockSize.x;
			}
			else if (count % blockSize.x > 0)
			{
				blocks.push_back(GridBlock(1,1,delta,1,offset,0));
				delta = count % blockSize.x;
			}


			count -= delta;
			offset += delta;
		}

	}
	else
	{
		//auto left = updateRangeStart.x;
		//auto right = updateRangeEnd.x;
		//auto top = updateRangeStart.y;
		//auto bottom = updateRangeEnd.y;

		///// try and divide into 4ths
		//dim3 center(fieldSize.x / 2, fieldSize.y / 2);
		////dim3 center((updateRangeStart.x + updateRangeEnd.x) / 2, (updateRangeStart.y + updateRangeEnd.y) / 2);

		//blocks.push_back(GridBlock((center.x - left + 1) / blockSize.x, (center.y - top + 1) / blockSize.y, blockSize.x, blockSize.y, left, top));
		//blocks.push_back(GridBlock((right - center.x + 1) / blockSize.x, (center.y - top + 1) / blockSize.y, blockSize.x, blockSize.y, center.x + 1, top));
		//blocks.push_back(GridBlock((center.x - left + 1) / blockSize.x, (bottom - center.y + 1) / blockSize.y, blockSize.x, blockSize.y, left, center.y + 1));
		//blocks.push_back(GridBlock((right - center.x + 1) / blockSize.x, (bottom - center.y + 1) / blockSize.y, blockSize.x, blockSize.y, center.x + 1, center.y + 1));

		blocks.push_back(GridBlock(512 / 32, 512 / 32, 32, 32, 0,0));
		blocks.push_back(GridBlock(512 / 32, 512 / 32, 32, 32, 512,0));
		blocks.push_back(GridBlock(512 / 32, 512 / 32, 32, 32, 0,512));
		blocks.push_back(GridBlock(512 / 32, 512 / 32, 32, 32, 512,512));

	}

	return blocks;
}

#else

vector<GridBlock> GridBlock::Divide(dim3 fieldSize, dim3 updateRangeStart, dim3 updateRangeEnd, dim3 blockSize, bool preferLinear)
{
	auto left = updateRangeStart.x;
	auto right = updateRangeEnd.x;
	auto top = updateRangeStart.y;
	auto bottom = updateRangeEnd.y;

	auto width = right - left;
	auto height = bottom - top;

	/////                              o0
	/////         w0 = n / b           | w1 = n % b
	///// ------------------------------------
	///// |                            |     |
	///// |                            |     |
	///// |                            |     |
	///// |            a0              | a1  | h0 = n / b
	///// |                            |     |
	///// |                            |     |
	///// |                            |     |
	///// ------------------------------------ o1
	///// |                            |     |
	///// |            a2              | a3  | h1 = n % b
	///// |                            |     |
	///// ------------------------------------
	auto w0 = width  /  blockSize.x;
	auto o0 = w0	 *  blockSize.x;
	auto w1 = width  %  blockSize.x;
	auto h0 = height /  blockSize.y;
	auto o1 = h0	 *  blockSize.y;
	auto h1 = height %  blockSize.y;

	vector<GridBlock> blocks;
	if (w0 > 0 && h0 > 0)
		/// a0
	{
		blocks.push_back(GridBlock(w0, h0, blockSize.x, blockSize.y, left,  top));
	}

	/// a1
	if (w1 > 0 && h0 > 0)
	{
		blocks.push_back(GridBlock(1, h0, w1, blockSize.y, left + o0, top));
	}

	/// a2
	if (w0 > 0 && h1 > 0)
	{
		blocks.push_back(GridBlock(w0, h1, blockSize.x, 1, left, top + o1));
	}

	/// a3
	if (h1 > 0 && w1 > 0)
	{
		blocks.push_back(GridBlock(1, 1, w1, h1, left + o0, top + o1));
	}

	return blocks;
}

#endif