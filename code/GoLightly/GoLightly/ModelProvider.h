#ifndef MODELPROVIDER_H
#define MODELPROVIDER_H

#include <string>
#include <vector>
using namespace std;

#include "FieldDescriptor.cuh"

class ModelProvider
{
public:
	ModelProvider(void);
	~ModelProvider(void);

	static int3 CrystalGuide(FieldDescriptor &target, float epsR = 9.f)
	{
		int3 sourcePosition;

		sourcePosition.x = target.Size.x / 2;
		sourcePosition.y = target.Size.y / 2;
		sourcePosition.z = target.Size.z / 2;

		int cellsX = 10;
		int cellsY = 10;
		for(int i = 0; i < cellsX; i++)
		{
			for(int j = 0; j < cellsY; j++)
			{
			}
		}

		return sourcePosition;
	}

	
	/// <summary>
	/// Creates a whispering gallery mode sensor and input waveguide
	/// </summary>
	/// <param name="media">The media.</param>
	static int3 WGMTest(FieldDescriptor &media, float n = 0.055f, float radiusQuotient = 0.5f, float guideWidth = 6, float couplingDistance = 3, float shellThickness = -1);

	static void FromImage(
		FieldDescriptor &media
		, const string &path
		, float n
		, vector<unsigned int> &sources
		);

	static void LoadModel(unsigned char **bytes, unsigned int *width, unsigned int *height, unsigned int *channels, const string &path);
	static void FromImage(
		FieldDescriptor &media
		,unsigned char *bytes
		,unsigned int width
		,unsigned int height
		,unsigned int channels
		,float n
		,vector<unsigned int> &sourceOffsets
		,unsigned int pmlLayers = 10
		);

	static void Cylinder(FieldDescriptor &media, float n, float radius, int centerX = -1, int centerY = -1, float width = -1);

	static void FixSources(vector<unsigned int> &sources, dim3 domainSize, unsigned int pmlLayers = 10);
};

#endif

