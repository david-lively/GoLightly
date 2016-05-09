#ifndef MONITOR_H
#define MONITOR_H

#include <vector>
#include <algorithm>
#include <map>

using namespace std;

#include "CudaHelpers.h"
#include "FieldDescriptor.cuh"


/*
Defines an FDTD monitor
*/
struct Monitor
{
	/// unique ID for this monitor as read from the model source bitmap's blue channel.
	unsigned char Id;

	//position offsets for every location that the monitor covers.
	unsigned int *Index;

	//Length of the Index array
	size_t IndexCount;

	float MaxValue;
	float MinValue;
	/// value at the current frame

	//Device-side array containing the data for the monitor at each frame. This should probably periodically flushed to disk.
	float *Frames;

	//Number of frames in the collection.
	unsigned int FrameCount;

	//current frame for writing
	unsigned int CurrentFrame;
	vector<float> MagnitudeHistory;

	vector<float> RmsHistory;
	float RmsMaxValue;
	float RmsMinValue;


	Monitor *DeviceMonitor;

	vector<GridBlock> GridBlocks;

	Monitor() : 
		Id(0)
		, Index(nullptr)
		, IndexCount(0)
		, Frames(nullptr)
		, FrameCount(0)
		, CurrentFrame(0)
		, DeviceMonitor(nullptr)
		, MaxValue(numeric_limits<float>::min())
		, MinValue(numeric_limits<float>::max())
		, RmsMaxValue(numeric_limits<float>::min())
		, RmsMinValue(numeric_limits<float>::max())
		
	{
	}

	~Monitor()
	{
	}

	void Initialize(CudaHelper &memory, vector<unsigned int> &positions, const unsigned int framesToAllocate);

	void Update(DeviceFieldDescriptor *ez, DeviceFieldDescriptor *hx, DeviceFieldDescriptor *hy, float simulationTime, CudaHelper &cuda);

	void CopyToHost(vector<float> &target, unsigned int offset = 0);

	void UpdateMagnitudes(vector<float> &buffer);

    static void Save(const string &path, map<unsigned char,shared_ptr<Monitor>> &monitors, bool normalizeOutput = false);

private:
	CudaHelper *m_cuda;


};



#endif