#include "Monitor.cuh"
#include <fstream>

void Monitor::Initialize(CudaHelper &memory, vector<unsigned int> &positions, const unsigned int framesToAllocate)
{
	m_cuda = &memory;

	Frames = memory.Malloc<float>(framesToAllocate * positions.size());
	FrameCount = framesToAllocate;

	memory.Memset(Frames, positions.size(), 0);

	Index = memory.Malloc<unsigned int>(positions.size());

	memory.Memcpy(Index,positions.data(),positions.size());

	IndexCount = positions.size();


	GridBlocks = GridBlock::Divide(dim3(positions.size(),1,1),dim3(0,0,0),dim3(positions.size(),1,1),dim3(32,1),true);
	DeviceMonitor = memory.Malloc<Monitor>(1);

	memory.Memcpy(DeviceMonitor,this,1);
}

__global__ void UpdateMonitor(
	Monitor *m
	, DeviceFieldDescriptor *ez
	, unsigned int currentFrame
	, unsigned int xOffset
	)
{
	/// index into monitor pixel lookup table. (Effectively, pointer to the monitor element - green pixel from
	/// the source model image)
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x + xOffset;

	/// dereference index and get the current Ez value, then place it in the
	/// active monitor frame.
	m->Frames[m->IndexCount * m->CurrentFrame + index] = ez->Data[m->Index[index]];


}

void Monitor::Update(DeviceFieldDescriptor *ez, DeviceFieldDescriptor *hx, DeviceFieldDescriptor *hy, float simulationTime, CudaHelper &cuda)
{
	static vector<float> buffer(FrameCount * IndexCount,0);

	for(auto itt = begin(GridBlocks); itt != end(GridBlocks); ++itt)
	{
		GridBlock &gb = *itt;

		UpdateMonitor<<<
			gb.GridDim
			,gb.BlockDim
			,0
			,cuda.GetStream(gb)
			>>>
			(DeviceMonitor, ez, CurrentFrame, gb.GridOffset.x);
	}

	Check(cudaDeviceSynchronize());

	CurrentFrame = (CurrentFrame + 1) % FrameCount;

	if (CurrentFrame == 0)
	{
		CopyToHost(buffer,0);
		UpdateMagnitudes(buffer);

	}

}

void Monitor::CopyToHost(vector<float> &buffer, unsigned int offset)
{
	buffer.resize(FrameCount * IndexCount);
	m_cuda->MemcpyAsync(buffer.data(),Frames,FrameCount * IndexCount, cudaMemcpyDeviceToHost);
}

/// <summary>
/// calculate the e-field magnitude throughout the monitor area.
/// </summary>
void Monitor::UpdateMagnitudes(vector<float> &buffer)
{

	for(unsigned int frame = 0; frame < FrameCount; ++frame)
	{
		float sum = 0.f;

		size_t frameStart = frame * IndexCount;


		for(auto i = 0; i < IndexCount; ++i)
		{
			sum += pow(buffer[frameStart + i],2);
		}

		float val = sqrt(sum / IndexCount); 

		MaxValue = max(val,MaxValue);
		MinValue = min(val,MinValue);

		MagnitudeHistory.push_back(val);
	}

	RmsHistory.clear();
	/// RMS
	unsigned int windowSize = 100;
	unsigned int startFrame = max<unsigned int>(0,MagnitudeHistory.size() - windowSize);
	for(unsigned int frame = startFrame; frame < min<unsigned int>(startFrame + windowSize,MagnitudeHistory.size()); ++frame)
	{
		float sum = 0.f;
		unsigned int count = 0;

		for(unsigned int i = frame; i < min<unsigned int>(frame + windowSize, MagnitudeHistory.size()); ++i) 
		{
			sum += pow(MagnitudeHistory[i],2);
			count++;
		}

		float val = sqrt(sum / count); 

		RmsMaxValue = max(val,RmsMaxValue);
		RmsMinValue = min(val,RmsMinValue);

		RmsHistory.push_back(val);
	}


}

/// <summary>
/// Save the recorded monitor history as a CSV at the given path.
/// </summary>
/// <param name="path">The path.</param>
/// <param name="monitors">The monitors.</param>
void Monitor::Save(const string &path, map<unsigned char,shared_ptr<Monitor>> &monitors, bool normalizeOutput)
{
	cout << "Saving monitor output to \"" << path << "\"\n";

	if (monitors.size() == 0)
	{
		cout << "\t Nothing to save. No monitors are defined in input model.\n\tAdd pixels with a non-zero blue channel to define a monitor.\n";
		return;
	}


	unsigned int magnitudeLength = numeric_limits<unsigned int>::max();

	ofstream target(path);

	/// for output normalization: get min and max of all recorded monitor values and scale outputs before writing to the file.77
	float normalizeMin = numeric_limits<float>::max();
	float normalizeMax = numeric_limits<float>::min();

	for(auto it = begin(monitors); it != end(monitors); ++it)
	{
		Monitor &m = *it->second;

		normalizeMin = min(normalizeMin, m.MinValue);
		normalizeMax = max(normalizeMax, m.MaxValue);

		target << "monitor " << to_string(m.Id) << ",";

		magnitudeLength = min<size_t>(magnitudeLength,m.MagnitudeHistory.size());
	}

	target << endl;

	if (normalizeOutput)
		cout << "Normalizing output: min " << normalizeMin << " max " << normalizeMax << endl;

	for(auto i = 0; i < magnitudeLength; i++)
	{
		for(auto it = begin(monitors); it != end(monitors); ++it)
		{
			Monitor &m = *it->second;

			float outputValue = m.MagnitudeHistory[i];

			if (normalizeOutput)
				outputValue = (outputValue - normalizeMin) / (normalizeMax - normalizeMin);

			target << outputValue << ",";
		}

		target << endl;
	}

	target.close();

	cout << "Done\n";

}


