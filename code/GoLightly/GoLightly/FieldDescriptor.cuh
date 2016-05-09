#ifndef FIELDDESCRIPTOR_CUH
#define FIELDDESCRIPTOR_CUH

#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include "CudaHelpers.h"
#include "CudaHelper.cuh"

#include "GridBlock.h"


enum class FieldType
{
	None
	,
	Hx
	,
	Hy
	,
	Hz
	,
	Ex
	,
	Ey
	,
	Ez
	,
	Ca
	,
	Cb
	,
	Da
	,
	Db
	,
	/// boundary names
	Exy
	,Exz
	,Eyx
	,Eyz
	,Ezx
	,Ezy
	,Hxy
	,Hxz
	,Hyx
	,Hyz
	,Hzx
	,Hzy

};

string to_string(FieldType f);

enum class FieldDirection
{
	X
	,Y
	,Z
};

using namespace std;

struct DeviceFieldDescriptor
{
	FieldType Name;
	dim3 Size;
	dim3 UpdateRangeStart;
	dim3 UpdateRangeEnd;

	float *Data;

};

/// <summary>
/// Description of and references to field data for a primary (E or H) field 
/// </summary>
struct FieldDescriptor
{
	/// <summary>
	/// describes a split-field boundary region for PML 
	/// </summary>
	struct BoundaryDescriptor
	{
		BoundaryDescriptor() :
			Amp(nullptr)
			,Psi(nullptr)
			,Decay(nullptr)
			,AmpDecayLength(0)
			,m_cuda(nullptr)
		{
		}

		~BoundaryDescriptor()
		{
		}

		FieldType Name;
		FieldDirection Direction;

		/// CPU-resident fields
		float *Amp;
		float *Psi;
		float *Decay;

		BoundaryDescriptor *DeviceDescriptor;

		unsigned int AmpDecayLength;


	private:
		CudaHelper *m_cuda;
	};

	FieldDescriptor()
	{
		DeviceArray = nullptr;
		Size = dim3(0,0,0);

		UpdateRangeStart = Size;
		UpdateRangeEnd = Size;

		Name = FieldType::None;
		DefaultValue = 0.f;

	}

	~FieldDescriptor()
	{
	}

	float DefaultValue;
	FieldType Name;

	dim3 Size;
	dim3 UpdateRangeStart;
	dim3 UpdateRangeEnd;

	vector<float> HostArray;
	float *DeviceArray;

	DeviceFieldDescriptor *DeviceDescriptor;

	vector<GridBlock> GridBlocks;
	map<FieldType,shared_ptr<BoundaryDescriptor>> Boundaries;

};

#endif