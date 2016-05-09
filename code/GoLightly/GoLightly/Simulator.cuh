#ifndef FDTD_CUH
#define FDTD_CUH

#include <vector>
#include <iostream>
#include <string>
#include <map>
#include <memory>

#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "FieldDescriptor.cuh"
#include "Configuration.h"
#include "CudaHelper.cuh"

#include "Monitor.cuh"

using namespace std;

/// <summary>
/// CUDA FDTD simulator
/// </summary>
class Simulator
{
public:
	Simulator();
	~Simulator();

	/// assumes Simulator already has a valid configuration
	void Initialize();

	void Initialize(const Configuration &config);
	void Update(float elapsedSeconds);
	void UpdateSource(float elapsedSeconds);
	void GetPreviewField(float** fieldPtr, float **materialsPtr, unsigned int *width, unsigned int *height);

	void Shutdown();

	Configuration &GetConfiguration()
	{
		return m_configuration;
	}

	void Save(FieldType name, const std::string &path);

	void TogglePause();

	float GetSimulatedTime()
	{
		return m_totalSimulatedTime;
	}

	unsigned int MonitorCount() { 
		return static_cast<unsigned int>(m_monitors.size()); 
	} 

	Monitor &GetMonitor(unsigned int i) 
	{
		unsigned int index = 0;

		for(auto it = begin(m_monitors); it != end(m_monitors); ++it, ++index)
		{
			if (index == i)
			{
				return *it->second;
			}
		}

		throw runtime_error("Monitor collection does not have at least " + to_string(i) + " items.");			
	}

	map<unsigned char, shared_ptr<Monitor>> &Monitors()
	{
		return m_monitors;
	}

	/// reset fields to initial values to restart the simulation
	void Reset();

	bool IsPaused() const { return m_paused; } 
private:

	Configuration m_configuration;
	CudaHelper m_cuda;

	unsigned int m_deviceId;
	float m_sourceValue;
	float m_sourceDelta;
	unsigned int m_sourceCount;

	float m_totalSimulatedTime;
	float m_totalTime;

	float m_sourceLambda;

	dim3 m_maxGridSize;
	dim3 m_maxThreadsDim;
	dim3 m_maxThreadsPerBlock;

	bool m_paused;

	map<FieldType,shared_ptr<FieldDescriptor>> m_fields;
	map<unsigned char, shared_ptr<Monitor>> m_monitors;

	template<typename T>
	size_t GetDimSize(T s)
	{
		return s.x * s.y * s.z;
	}

	template<typename T>
	void printArray(T &collection)
	{
		for(auto it = begin(collection); it != end(collection); ++it)
		{
			cout << *it << ",";
		}
	}

	template<typename T>
	void printArray(const std::string &title, T &collection)
	{
		cout << title << ": ";
		printArray(collection);
		cout << endl;
	}

	void InitCuda();
	void InitSimulation();
	void InitializeBoundaryData();

	void BuildMonitors(
		const unsigned char *bytes
		, const unsigned int width
		, const unsigned int height
		, const unsigned int channels
		, const unsigned int framestoAllocate = 1
		);

	FieldDescriptor &CreateFieldDescriptor(
		FieldType name
		, unsigned int width
		, unsigned int height
		, unsigned int marginX
		, unsigned int marginY
		, float defaultValue = 0.f
		//, unsigned int preferredBlockX = 32
		//, unsigned int preferredBlockY = 32
		);


	FieldDescriptor::BoundaryDescriptor &CreateBoundary(FieldDescriptor &field, FieldType boundaryName, FieldDirection dir);
};

#endif