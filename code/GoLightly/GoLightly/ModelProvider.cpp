#define PI 3.1415926f

#include <iostream>
#include <set>
#include <vector>

#include <minmax.h>

using namespace std;

#include "ModelProvider.h"
#include "../Thirdparty/soil/SOIL2.h"


ModelProvider::ModelProvider(void)
{
}


ModelProvider::~ModelProvider(void)
{
}


void ModelProvider::Cylinder(FieldDescriptor &media, float n, float radius, int centerX, int centerY, float width)
{
	dim3 center(centerX,centerY);

	if (center.x < 0)
	{
		center.x = media.Size.x / 2;
	}

	if (center.y < 0)
	{
		center.y = media.Size.y / 2;
	}

	if (radius < 0)
	{
		radius = min(center.x,center.y);
	}

	if (width < 0)
		width = radius + 1;

	auto topLeft = center -  radius;
	auto bottomRight = center + radius;

	for(unsigned int j = topLeft.y; j < bottomRight.y; j++)
	{
		for(unsigned int i = topLeft.x; i < bottomRight.x; i++)
		{
			float r = length(dim3(center.x - i,center.y - j));
			if (r <= radius && r >= radius - width)
			{
				media.HostArray[j * media.Size.x + i] = n;
			}

		}
	}

}



/// <summary>
/// Creates a whispering gallery mode sensor and input waveguide
/// </summary>
/// <param name="media">The media container field (typically Cb)</param>
/// <param name="radiusQuotient">ration of dius of the WGM </param>
/// <param name="thickness">thickness in voxels of the shell</param>
int3 ModelProvider::WGMTest(FieldDescriptor &media, float n, float radiusQuotient, float guideWidth, float couplingDistance, float shellThickness)
{

	/// center of the simulation domain
	auto center = media.Size / 2;
	unsigned int radius = static_cast<unsigned int>(radiusQuotient * min(media.Size.x, media.Size.y) / 2.f);

	if (shellThickness < 0)
		shellThickness = guideWidth;
	else if (shellThickness < 0.1f)
		shellThickness = radius;

	auto topLeft = center - radius;
	auto bottomRight = center + radius;

	int count = 0;

	if (shellThickness > radius)
		shellThickness = radius + 1;

	Cylinder(media,n,radius,center.x,center.y,shellThickness);

	/// draw the hopefully-coupled fiber strand
	for(int j = bottomRight.y + couplingDistance; j < bottomRight.y + guideWidth + couplingDistance; ++j)
	{
		for(int i = 0; i < media.Size.x; ++i)
		{
			media.HostArray[j * media.Size.x + i] = n;
			count++;
		}
	}

	int3 sourcePosition;

	sourcePosition.x = 12;
	sourcePosition.y = (bottomRight.y + guideWidth / 2 + couplingDistance);
	sourcePosition.z = 0;

	cout << "Set " << count << " media cells to nu = " << n << endl;

	return sourcePosition;
}

void ModelProvider::FromImage(
	FieldDescriptor &media
 ,unsigned char *bytes
 ,unsigned int width
 ,unsigned int height
 ,unsigned int channels
 ,float n
 ,vector<unsigned int> &sourceOffsets
 ,unsigned int pmlLayers
 )
{

	for(int j = 0; j < media.Size.y; j++)
	{
		int sourceY = j * height / media.Size.y;

		for(int i = 0; i < media.Size.x; i++)
		{
			int sourceX = i * width / media.Size.x;
			unsigned int sourceOffset = channels * (sourceY * width + sourceX);
			unsigned int mediaOffset = j * media.Size.x + i;

			unsigned char red =   bytes[sourceOffset + 0];
			unsigned char green = bytes[sourceOffset + 1];
			unsigned char blue =  bytes[sourceOffset + 2];
			//unsigned char alpha = bytes[sourceOffset + 3];

			/// add a new source position if we find a red pixel.
			/// - only add if the source is NOT inside of a PML region
			if (red > 128)
			{
				sourceOffsets.push_back(mediaOffset);
			}

			/// fill default waveguide material (parameter n)
			if (green > 0)
			{
				// interpolate n based on green value.
				float m = n * green * 1.f / 255;
				media.HostArray[mediaOffset] = m;
			}

		}
	}


	if (sourceOffsets.size() == 0)
	{
		cerr << "No sources found\n";
		exit(EXIT_FAILURE);
	}
}

void ModelProvider::LoadModel(unsigned char **bytes, unsigned int *width, unsigned int *height, unsigned int *channels, const string &path)
{
	int w,h,c;
	*bytes = SOIL_load_image(path.c_str(),&w,&h,&c,0);

	cout << "Loading image \"" << path << "\"\n";

	cout << "SOIL result: \"" << SOIL_last_result() << "\"\n";


	*width =    static_cast<unsigned int>(w);
	*height =   static_cast<unsigned int>(h);
	*channels = static_cast<unsigned int>(c);

}

void ModelProvider::FromImage(FieldDescriptor &media, const string &path, float n, vector<unsigned int> &sources)
{
	unsigned char *bytes;
	unsigned int width, height, channels;

	LoadModel(&bytes,&width,&height,&channels,path);

	FromImage(media,bytes,width,height,channels,n, sources);
}
