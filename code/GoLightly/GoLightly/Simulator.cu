#define CA (1.f)
#define DA (1.f)
#define USE_PML
//#define USE_MAGNETIC_MATERIALS

#include <iostream>
#include <iomanip>
#include <fstream>
#include <set>
#include <vector>
#include <stack>

using namespace std;


#include "Simulator.cuh"
#include "CudaHelpers.h"
#include "ModelProvider.h"
#include "Monitor.cuh"




Simulator::Simulator()
{
	m_deviceId = 0;
	m_totalTime = 0;
	m_sourceValue = 0;
	m_totalSimulatedTime = 0.f;

	m_paused = false;

}

Simulator::~Simulator()
{
}

/* device-resident field descriptors, used to avoid passing these as parameters to every kernel launch */
__device__ DeviceFieldDescriptor				*Ez;
__device__ FieldDescriptor::BoundaryDescriptor	*Ezx;
__device__ FieldDescriptor::BoundaryDescriptor	*Ezy;

__device__ DeviceFieldDescriptor				*Hx;
__device__ FieldDescriptor::BoundaryDescriptor	*Hxy;

__device__ DeviceFieldDescriptor				*Hy;
__device__ FieldDescriptor::BoundaryDescriptor	*Hyx;

__device__ DeviceFieldDescriptor *Cb;


#ifdef USE_MAGNETIC_MATERIALS
__device__ DeviceFieldDescriptor *Db;
#else
__device__ float DbDefault;
#endif

__device__ class Configuration *Configuration;

__device__ unsigned int *SourceOffsets;
__device__ unsigned int SourceCount;

/*
Set GPU-resident field pointers
*/
__global__ void InitDevicePointers(
	DeviceFieldDescriptor *ez
	, DeviceFieldDescriptor *hx
	, DeviceFieldDescriptor *hy
	, FieldDescriptor::BoundaryDescriptor *ezx
	, FieldDescriptor::BoundaryDescriptor *ezy
	, FieldDescriptor::BoundaryDescriptor *hxy
	, FieldDescriptor::BoundaryDescriptor *hyx
	, DeviceFieldDescriptor *cb
#ifdef USE_MAGNETIC_MATERIALS
	,DeviceFieldDescriptor *db
#else
	, float dbDefault
#endif	
	, class Configuration *config
	, unsigned int *sourceOffsets
	, unsigned int sourceCount
	)
{
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
	{
		Ez = ez;
		Ezx = ezx;
		Ezy = ezy;

		Hx = hx;
		Hxy = hxy;

		Hy = hy;
		Hyx = hyx;

		Cb = cb;
#ifdef USE_MAGNETIC_MATERIALS
		Db = db;
#else
		DbDefault = dbDefault;
#endif
		Configuration = config;

		SourceOffsets = sourceOffsets;
		SourceCount = sourceCount;

	}
}

__global__ void UpdateSourcesEz(
	float sourceDelta
	, unsigned int indexOffset
	)
{

	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x + indexOffset;
	unsigned int sourceOffset = SourceOffsets[index];

	Ez->Data[sourceOffset] += sourceDelta;
}

void Simulator::UpdateSource(float simulationTime)
{
	float sourceValue = sin(simulationTime * m_configuration.Frequency);
	float sourceDelta = sourceValue - m_sourceValue;

	if (m_configuration.VerboseOutput)
		cout << "Source " << sourceValue << " delta " << sourceDelta << " sim time " << simulationTime << endl;

	m_sourceValue = sourceValue;

	if (m_sourceCount > 1024)
		throw runtime_error("Implementation does not support more than 1024 source pixels.");

	UpdateSourcesEz << <
		dim3(1, 1, 1)
		, dim3(m_sourceCount, 1, 1)
		, 0
		, m_cuda.GetStream(*this)
		>> >(sourceDelta, 0);
}



/// <summary>
/// Update the EZ field
/// </summary>
/// <param name="sourceOffset">Value offset within the EZ field of the source</param>
/// <param name="hx">change in the source value since the last update</param>
/// <returns></returns>
__global__ void UpdateEz(
	dim3 threadOffset
	)
{

	unsigned int x = threadOffset.x + blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = threadOffset.y + blockIdx.y * blockDim.y + threadIdx.y;

	if (y < 1 || x < 1)
		return;

	float cb = Cb->Data[y * Cb->Size.x + x];

	unsigned int center = y * Ez->Size.x + x;
	float hxBottom = Hx->Data[y * Hx->Size.x + x];
	float hxTop = Hx->Data[(y - 1) * Hx->Size.x + x];
	float dhx = (hxBottom - hxTop);

	float hyRight = Hy->Data[y * Hy->Size.x + x];
	float hyLeft = Hy->Data[y * Hy->Size.x + x - 1];
	float dhy = (hyLeft - hyRight);

	float ezxPsi = 0.f;
	float ezyPsi = 0.f;

	if (x < 10 || x > Ez->UpdateRangeEnd.x - 10 || y < 10 || y > Ez->UpdateRangeEnd.y - 10)
	{
		ezyPsi = Ezy->Decay[y] * Ezy->Psi[center] + Ezy->Amp[y] * dhx;
		Ezy->Psi[center] = ezyPsi;
		ezxPsi = Ezx->Decay[x] * Ezx->Psi[center] + Ezx->Amp[x] * dhy;
		Ezx->Psi[center] = ezxPsi;

	}

	Ez->Data[center] = CA * Ez->Data[center] + cb * (dhy - dhx) + cb * (ezxPsi - ezyPsi);
}

__global__ void UpdateHx(dim3 threadOffset)
{
	unsigned int x = threadOffset.x + blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = threadOffset.y + blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= Ez->Size.y - 1)
		return;



	unsigned int hxOffset = y * Hx->Size.x + x;
#ifdef USE_MAGNETIC_MATERIALS
	float db = Db->Data[y * Db->Size.x + x];
#else
	const float db = DbDefault;
#endif
	//float ezTop = Ez->Data[y * Ez->Size.x + x];
	//float ezBottom = Ez->Data[(y+1) * Ez->Size.x + x];

	float dEz = Ez->Data[(y + 1) * Ez->Size.x + x] - Ez->Data[y * Ez->Size.x + x];

	float hx = DA * Hx->Data[hxOffset] - db * dEz;


	if (y < 10 || y > Hx->UpdateRangeEnd.y - 10 || x < 10 || x > Hx->UpdateRangeEnd.x - 10)
	{
		/// update boundaries
		float decay = Hxy->Decay[y];
		float amp = Hxy->Amp[y];
		float psi = Hxy->Psi[hxOffset];

		psi = decay * psi + amp * dEz / Configuration->Dx;

		Hxy->Psi[hxOffset] = psi;
		hx = hx - db * Configuration->Dx * psi;
	}

	Hx->Data[hxOffset] = hx;

}

__global__ void UpdateHy(dim3 threadOffset)
{
	unsigned int x = threadOffset.x + blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = threadOffset.y + blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= Ez->Size.x - 1)
		return;

	unsigned int hyOffset = y * Hy->Size.x + x;

#ifdef USE_MAGNETIC_MATERIALS
	float db = Db->Data[y * Db->Size.x + x];
#else
	const float db = DbDefault;
#endif

	float ezLeft = Ez->Data[y * Ez->Size.x + x];
	float ezRight = Ez->Data[y * Ez->Size.x + x + 1];

	float dEz = ezRight - ezLeft;
	float hy = DA * Hy->Data[hyOffset] - db * (ezRight - ezLeft);

	if (x < 10 || y < 10 || x > Hy->UpdateRangeEnd.x - 10 || y > Hy->UpdateRangeEnd.y - 10)
	{

		float psi = Hyx->Psi[hyOffset];
		float decay = Hyx->Decay[x];
		float amp = Hyx->Amp[x];

		psi = decay * psi + amp * dEz / Configuration->Dx;

		hy = hy - db * Configuration->Dx * psi;

		Hyx->Psi[hyOffset] = psi;
	}

	Hy->Data[hyOffset] = hy;
}



// crops the effective domain size to a multiple of the default block (size (32,32)
//#define SINGLE_LAUNCH


void Simulator::Update(float elapsedSeconds)
{
	if (m_paused)
		return;

	Check(cudaDeviceSynchronize());

	/*
	NOTE: using the same grid and block size for all fields effectively shrinks the domain but a small amount (1 cell assuming the domain size is a multiple of the block size),
	but removes the need to check update bounds in the CUDA kernels.
	*/
	auto &ez = *m_fields[FieldType::Ez];
	auto &hx = *m_fields[FieldType::Hx];
	auto &hy = *m_fields[FieldType::Hy];

	for (auto it = ez.GridBlocks.rbegin(); it != ez.GridBlocks.rend(); ++it)
		//for(auto it = begin(ez.GridBlocks); it != end(ez.GridBlocks); ++it)
	{
		UpdateEz << <it->GridDim, it->BlockDim, 0, m_cuda.GetStream(*it) >> >
			(
			it->GridOffset
			);

	}


	Check(cudaDeviceSynchronize());

	//for(auto it = begin(hx.GridBlocks); it != end(hx.GridBlocks); ++it)
	for (auto it = hx.GridBlocks.rbegin(); it != hx.GridBlocks.rend(); ++it)
	{
		UpdateHx << <it->GridDim, it->BlockDim, 0, m_cuda.GetStream(*it) >> >(it->GridOffset);
	}

	//for(auto it = begin(hy.GridBlocks); it != end(hy.GridBlocks); ++it)
	for (auto it = hy.GridBlocks.rbegin(); it != hy.GridBlocks.rend(); ++it)
	{
		UpdateHy << <it->GridDim, it->BlockDim, 0, m_cuda.GetStream(*it) >> >(it->GridOffset);
	}

	for (auto it = begin(m_monitors); it != end(m_monitors); ++it)
	{
		Monitor &m = *it->second;

		m.Update(ez.DeviceDescriptor, hx.DeviceDescriptor, hy.DeviceDescriptor, m_totalSimulatedTime, m_cuda);
	}

	UpdateSource(m_totalSimulatedTime);

	m_totalTime += elapsedSeconds;
	m_totalSimulatedTime += m_configuration.Dt;

}

/// <summary>
/// Get a CUDA pointer to the field buffer to use for previews, as well as its dimensions
/// </summary>
/// <param name="fieldPtr">CUDA device pointer</param>
/// <param name="width">width of the buffer</param>
/// <param name="height">height of the buffer</param>
void Simulator::GetPreviewField(float** fieldPtr, float **materialsPtr, unsigned int *width, unsigned int *height)
{
	//auto fieldName = FieldType::Ez;
	//auto boundaryName = FieldType::Ezy;

	auto fieldName = FieldType::Ez;
	auto boundaryName = FieldType::Ezx;

	auto &fd = *m_fields[fieldName];
	auto &b = *fd.Boundaries[boundaryName];

	//*fieldPtr = b.Psi;
	*fieldPtr = fd.DeviceArray;
	*width = fd.Size.x;
	*height = fd.Size.y;

	FieldDescriptor &cb = *m_fields[FieldType::Cb];

	*materialsPtr = cb.DeviceArray;
}

void Simulator::Initialize()
{
	InitCuda();

	size_t freeMemory, totalMemory;
	m_cuda.GetMemoryInfo(freeMemory, totalMemory);
	float freeGB = freeMemory / (1024 * 1024 * 1024.f);

	cout << "\tAvailable GPU memory: " << std::setw(3) << std::setprecision(3) << freeMemory * 100.0 / totalMemory << "% or " << freeGB << "GB\n";

	InitSimulation();

	size_t newFreeMemory;
	m_cuda.GetMemoryInfo(newFreeMemory, totalMemory);

	float usedMB = (freeMemory - newFreeMemory) / (1024 * 1024);

	cout << "\nUsing " << std::setw(3) << std::setprecision(3) << usedMB << "MB for GPU-resident data\n\n";

	m_paused = m_configuration.StartPaused;

	cout << "Configuration:\n" << to_string(m_configuration) << endl;
}


void Simulator::Initialize(const class Configuration &config)
{
	m_configuration = config;

	Initialize();
}

FieldDescriptor &Simulator::CreateFieldDescriptor(
	FieldType name
	, unsigned int width
	, unsigned int height
	, unsigned int marginX
	, unsigned int marginY
	, float defaultValue
	)
{
	auto ptr = make_shared<FieldDescriptor>();

	m_fields[name] = ptr;

	FieldDescriptor &field = *ptr;

	field.DefaultValue = defaultValue;

	field.Name = name;
	field.Size = dim3(width, height, 1);
	field.UpdateRangeStart = dim3(marginX, marginY);
	field.UpdateRangeEnd = dim3(width - marginX - 1, height - marginY - 1);
	field.GridBlocks = GridBlock::Divide(field.Size, field.UpdateRangeStart, field.UpdateRangeEnd, dim3(32, 32));

	unsigned int count = static_cast<unsigned int>(GetDimSize(field.Size));
	field.HostArray = vector<float>(count, defaultValue);

	field.DeviceArray = m_cuda.Malloc<float>(count);

	m_cuda.Memcpy(field.DeviceArray, field.HostArray.data(), count);

	field.DeviceDescriptor = m_cuda.Malloc<DeviceFieldDescriptor>(1);

	DeviceFieldDescriptor fdc;
	fdc.Data = field.DeviceArray;
	fdc.Size = field.Size;
	fdc.Name = name;
	fdc.UpdateRangeEnd = field.UpdateRangeEnd;
	fdc.UpdateRangeStart = field.UpdateRangeStart;

	m_cuda.Memcpy(field.DeviceDescriptor, &fdc, 1);

	Check(cudaDeviceSynchronize());

	cout << "\nCreated field: \"" << to_string(name) << "\"\n";

	return field;
}

FieldDescriptor::BoundaryDescriptor &Simulator::CreateBoundary(FieldDescriptor &field, FieldType boundaryName, FieldDirection dir)
{
	/// make sure we don't already have a boundary if this type.
	if (field.Boundaries.find(boundaryName) != end(field.Boundaries))
		throw;

	auto ptr = make_shared<FieldDescriptor::BoundaryDescriptor>();
	field.Boundaries[boundaryName] = ptr;

	FieldDescriptor::BoundaryDescriptor &boundary = *ptr;

	boundary.Name = boundaryName;
	boundary.Direction = dir;

	switch (dir)
	{
	case FieldDirection::X:
		boundary.AmpDecayLength = field.Size.x;
		break;

	case FieldDirection::Y:
		boundary.AmpDecayLength = field.Size.y;
		break;

	case FieldDirection::Z:
		boundary.AmpDecayLength = field.Size.z;
		break;

	default:
		/// unhandled field direction
		throw;
	}

	auto hostArray = vector<float>(boundary.AmpDecayLength, 0.f);
	//unsigned int bytes = static_cast<unsigned int>(sizeof(float) * hostArray.size());


	///// amp and decay fields
	boundary.Amp = m_cuda.Malloc<float>(boundary.AmpDecayLength);
	m_cuda.Memcpy(boundary.Amp, hostArray.data(), hostArray.size());

	boundary.Decay = m_cuda.Malloc<float>(boundary.AmpDecayLength);
	m_cuda.Memcpy(boundary.Decay, hostArray.data(), hostArray.size());

	unsigned int count = field.Size.x * field.Size.y * field.Size.z;

	boundary.Psi = m_cuda.Malloc<float>(count);
	m_cuda.Memset(boundary.Psi, count, 0);

	/// allocate device-side boundary descriptor
	boundary.DeviceDescriptor = m_cuda.Malloc<FieldDescriptor::BoundaryDescriptor>(1);
	m_cuda.Memcpy(boundary.DeviceDescriptor, &boundary, 1);

	cout << "\nCreated Boundary \"" << to_string(boundaryName) << "\"\n";

	return boundary;
}

/// <summary>
/// initializes resources such as field and boundary arrays and sets material properties
/// </summary>
/// <remarks>
/// Example Yee grid: domain size is 4 x 4. 
/// Ez @ (0,0), size (5x5)
/// Hx @ (0,0), size (5x4)
/// Hy @ (0,0), size (4x5)
/// ez---hy---ez---hy---ez---hy---ez---hy---ez
/// |         |         |         |         |         
/// hx        hx        hx        hx        hx
/// |         |         |         |         |         
/// ez---hy---EZ---hy---EZ---hy---EZ---hy---ez
/// |         |         |         |         |         
/// hx        hx        hx        hx        hx
/// |         |         |         |         |         
/// ez---hy---EZ---hy---EZ---hy---EZ---hy---ez
/// |         |         |         |         |         
/// hx        hx        hx        hx        hx
/// |         |         |         |         |         
/// ez---hy---EZ---hy---EZ---hy---EZ---hy---ez
/// |         |         |         |         |         
/// hx        hx        hx        hx        hx
/// |         |         |         |         |         
/// ez---hy---ez---hy---ez---hy---ez---hy---ez
/// </remarks>
void Simulator::InitSimulation()
{
	bool loadModel = m_configuration.ModelPath != "";

	unsigned char *modelBytes = nullptr;
	unsigned int modelWidth, modelHeight, modelChannels;


	if (loadModel)
	{
		ModelProvider::LoadModel(&modelBytes, &modelWidth, &modelHeight, &modelChannels, m_configuration.ModelPath);
		m_configuration.DomainSize = dim3(modelWidth, modelHeight, 1);
	}

	dim3 domainSize = m_configuration.DomainSize;

	unsigned int count = static_cast<unsigned int>(GetDimSize(domainSize));

	/// create field and boundary descriptors
	auto &ez = CreateFieldDescriptor(FieldType::Ez, domainSize.x + 1, domainSize.y + 1, 1, 1);

	auto &ezx = CreateBoundary(ez, FieldType::Ezx, FieldDirection::X);
	auto &ezy = CreateBoundary(ez, FieldType::Ezy, FieldDirection::Y);

	auto &hx = CreateFieldDescriptor(FieldType::Hx, domainSize.x + 1, domainSize.y, 0, 1);
	auto &hxy = CreateBoundary(hx, FieldType::Hxy, FieldDirection::Y);

	auto &hy = CreateFieldDescriptor(FieldType::Hy, domainSize.x, domainSize.y + 1, 0, 1);
	auto &hyx = CreateBoundary(hy, FieldType::Hyx, FieldDirection::X);

	/// initialize materials
	const float eps0 = 1.f;
	float cbDefault = m_configuration.Dt / (eps0 * m_configuration.Dx);

	auto &cb = CreateFieldDescriptor(FieldType::Cb, domainSize.x + 1, domainSize.y + 1, 0, 0, cbDefault);

	const float mu0 = 1.f;
	float dbDefault = m_configuration.Dt / (mu0 * m_configuration.Dx);
#ifdef USE_MAGNETIC_MATERIALS
	auto &db = CreateFieldDescriptor(FieldType::Db,domainSize.x,		domainSize.y, 0, 0, dbDefault);
#else

#endif


	float dt = m_configuration.Dt;
	float dx = m_configuration.Dx;

	float epsOrMuR = m_configuration.EpsilonR;

	float n = dt / (dx * epsOrMuR);

	vector<unsigned int> sources;
	sources.reserve(domainSize.x * domainSize.y);


	if (modelBytes)
	{
		ModelProvider::FromImage(
			cb
			, modelBytes
			, modelWidth
			, modelHeight
			, modelChannels
			, n
			, sources
			, m_configuration.PmlLayers
			);

		BuildMonitors(modelBytes, modelWidth, modelHeight, modelChannels);
	}
	else
	{
		auto sourcePosition = ModelProvider::WGMTest(cb, n, 0.25f, 10, 5, 0.09);

		//for(unsigned int j = m_configuration.PmlLayers; j < ez.Size.y - m_configuration.PmlLayers; ++j)
		//{
		//	sources.push_back(j * ez.Size.x + 100);
		//}

		sources.push_back(sourcePosition.y * ez.Size.x + sourcePosition.x);
	}

	free(modelBytes);
	modelBytes = nullptr;

	size_t bytes = sizeof(float)* cb.HostArray.size();
	m_cuda.Memcpy(cb.DeviceArray, cb.HostArray.data(), cb.HostArray.size());

	InitializeBoundaryData();

	class Configuration *config;
	config = m_cuda.Malloc<class Configuration>(1);
	//Check(cudaMemcpy(config,&m_configuration,sizeof(class Configuration),cudaMemcpyHostToDevice));
	m_cuda.Memcpy(config, &m_configuration, 1);

	m_sourceCount = sources.size();

	unsigned int *deviceSources = nullptr;
	bytes = sources.size() * sizeof(unsigned int);

	deviceSources = m_cuda.Malloc<unsigned int>(sources.size());
	//Check(cudaMemcpy(deviceSources,sources.data(), bytes, cudaMemcpyHostToDevice));
	m_cuda.Memcpy(deviceSources, sources.data(), sources.size());


	InitDevicePointers << <dim3(1), dim3(1) >> >(
		ez.DeviceDescriptor
		, hx.DeviceDescriptor
		, hy.DeviceDescriptor
		, ezx.DeviceDescriptor
		, ezy.DeviceDescriptor
		, hxy.DeviceDescriptor
		, hyx.DeviceDescriptor
		, cb.DeviceDescriptor
#ifdef USE_MAGNETIC_MATERIALS
		,db.DeviceDescriptor
#else
		, dbDefault
#endif

		, config
		, deviceSources
		, m_sourceCount
		);

	Check(cudaDeviceSynchronize());

}

vector<float> Zeros(unsigned int length)
{
	return vector<float>(length, 0.f);
}

/// <summary>
/// calculate initial values for boundary arrays.
/// SEE: http://read.pudn.com/downloads169/sourcecode/math/780343/tmz_with_npml_scan.m__.htm
/// </summary>
void Simulator::InitializeBoundaryData()
{
	auto &ez = *m_fields[FieldType::Ez];
	auto &ezx = *ez.Boundaries[FieldType::Ezx];
	auto &ezy = *ez.Boundaries[FieldType::Ezy];

	auto &hx = *m_fields[FieldType::Hx];
	auto &hxy = *hx.Boundaries[FieldType::Hxy];

	auto &hy = *m_fields[FieldType::Hy];
	auto &hyx = *hy.Boundaries[FieldType::Hyx];

	auto ezxAmp = vector<float>(ezx.AmpDecayLength, 0.f);
	auto ezxDecay = vector<float>(ezx.AmpDecayLength, 0.f);

	auto ezyAmp = vector<float>(ezy.AmpDecayLength, 0.f);
	auto ezyDecay = vector<float>(ezy.AmpDecayLength, 0.f);

	auto hyxAmp = vector<float>(hyx.AmpDecayLength, 0.f);
	auto hyxDecay = vector<float>(hyx.AmpDecayLength, 0.f);

	auto hxyAmp = vector<float>(hxy.AmpDecayLength, 0.f);
	auto hxyDecay = vector<float>(hxy.AmpDecayLength, 0.f);


	auto layers = m_configuration.PmlLayers;
	auto sigmaMax = m_configuration.PmlSigmaMax;
	auto sigmaOrder = m_configuration.PmlSigmaOrder;
	auto dx = m_configuration.Dx;
	auto dt = m_configuration.Dt;

	float eps0 = 1.f;

	auto xmin = layers * dx;
	float invLayersDx = 1.f / (layers * dx);


	for (unsigned int i = 0; i < layers; i++)
	{
		float elength = i * dx;
		float hlength = (i + 0.5f) * dx;

		float esigma = sigmaMax * (float)pow((abs(elength - xmin) * invLayersDx), sigmaOrder);
		float hsigma = sigmaMax * (float)pow((abs(hlength - xmin) * invLayersDx), sigmaOrder);

		auto edecay = exp(-(dt * esigma) / eps0);

		auto hdecay = exp(-(dt * hsigma) / eps0);

		ezxDecay[i] = edecay;
		ezxDecay[ezxDecay.size() - i - 1] = edecay;

		ezxAmp[i] = edecay - 1;
		ezxAmp[ezxAmp.size() - i - 1] = edecay - 1;

		hxyDecay[i] = hdecay;
		hxyDecay[hxyDecay.size() - i - 1] = hdecay;

		hxyAmp[i] = hdecay - 1;
		hxyAmp[hxyAmp.size() - i - 1] = hdecay - 1;

		ezyDecay[i] = edecay;
		ezyDecay[ezyDecay.size() - i - 1] = edecay;

		ezyAmp[i] = edecay - 1;
		ezyAmp[ezyAmp.size() - i - 1] = edecay - 1;

		hyxDecay[i] = hdecay;
		hyxDecay[hyxDecay.size() - i - 1] = hdecay;

		hyxAmp[i] = hdecay - 1;
		hyxAmp[hyxAmp.size() - i - 1] = hdecay - 1;


	}

	/// copy fields to the GPU

	m_cuda.Memcpy(ezx.Amp, ezxAmp.data(), ezxAmp.size());
	m_cuda.Memcpy(ezx.Decay, ezxDecay.data(), ezxDecay.size());

	m_cuda.Memcpy(ezx.Amp, ezxAmp.data(), ezxAmp.size());
	m_cuda.Memcpy(ezx.Decay, ezxDecay.data(), ezxDecay.size());

	m_cuda.Memcpy(ezy.Amp, ezyAmp.data(), ezyAmp.size());
	m_cuda.Memcpy(ezy.Decay, ezyDecay.data(), ezyDecay.size());

	m_cuda.Memcpy(hxy.Amp, hxyAmp.data(), hxyAmp.size());
	m_cuda.Memcpy(hxy.Decay, hxyDecay.data(), hxyDecay.size());

	m_cuda.Memcpy(hyx.Amp, hyxAmp.data(), hyxAmp.size());
	m_cuda.Memcpy(hyx.Decay, hyxDecay.data(), hyxDecay.size());
}


void Simulator::Reset()
{
	throw runtime_error("Not implemented");
}


void Simulator::InitCuda()
{
	static bool isInitialized = false;

	cout << "Initializing CUDA\n";

	if (isInitialized)
	{
		cerr << "CUDA is already initialized." << endl;
		exit(EXIT_FAILURE);
	}

	int deviceCount = 0;

	Check(cudaGetDeviceCount(&deviceCount));

	if (deviceCount == 0)
	{
		cerr << "No CUDA devices found" << endl;
		exit(EXIT_FAILURE);
	}
	else
	{
		cout << "Found " << deviceCount << " devices" << endl;
	}

	m_deviceId = 0;

	cudaDeviceProp deviceProperties;
	Check(cudaGetDeviceProperties(&deviceProperties, m_deviceId));

	m_maxGridSize = make_uint3(deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1], deviceProperties.maxGridSize[2]);
	m_maxThreadsDim = make_uint3(deviceProperties.maxThreadsDim[0], deviceProperties.maxThreadsDim[1], deviceProperties.maxThreadsDim[2]);
	m_maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

	cout << "Using device """ << deviceProperties.name << """" << endl;
	printArray("\tMax grid", deviceProperties.maxGridSize);
	printArray("\tMax block", deviceProperties.maxThreadsDim);
	cout << "\tMax threads per block: " << deviceProperties.maxThreadsPerBlock << endl;
	cout << "\tTotal GPU Memory: " << deviceProperties.totalGlobalMem / (1024 * 1024 * 1024.f) << "GB" << endl;

}

void Simulator::Shutdown()
{
	m_cuda.Shutdown();

	/// reset the device to allow NVIDIA visual profiler to correctly capture the application timeline
	cudaDeviceReset();
}

void Simulator::TogglePause()
{
	m_paused = !m_paused;
}


#define SAVE_MONITORS
/// <summary>
/// Saves the specified field to a CSV file
/// </summary>
/// <param name="name">Field to save</param>
/// <param name="path">Output file</param>
void Simulator::Save(FieldType name, const std::string &path)
{
#ifdef SAVE_MONITORS

	Monitor::Save(path, m_monitors, m_configuration.NormalizeOutput);
#else
	FieldDescriptor &field = *m_fields[name];
	float *data;

	auto byteLength = sizeof(float) * field.Size.x * field.Size.y;
	data = static_cast<float*>(malloc(byteLength));

	//Check(cudaMemcpy(data,field.DeviceArray,byteLength,cudaMemcpyDeviceToHost));
	m_cuda.Memcpy(data, field.DeviceArray, field.Size.x * field.Size.y);

	cout << "Saving field to file \"" << path << "\"\n";

	ofstream target;
	target.open(path);

	for(unsigned int j = 0; j < field.Size.y; j++)
	{
		for(int i = 0; i < field.Size.x; i++)
		{
			target << data[j * field.Size.x + i];
			if (i != field.Size.x - 1)
				target << ",";
		}
		target << endl;
	}

	cout << "Done\n";
#endif

}

/*
Scan the image and generate a list of monitors.

Basically:
1. Loop through all pixels, looking for anything with a BLUE component, which indicates that the
pixel is part of a monitor. The value of the blue component is a unique identifier indicating to which monitor instance the pixel belongs
2. Add the linear offset of the given monitor pixel to the collection for that monitor (see map monitorPositions below)
3. Have the monitor Initialize itself with the collected positions.

*/
void Simulator::BuildMonitors(
	const unsigned char *bytes
	, const unsigned int width
	, const unsigned int height
	, const unsigned int channels
	, const unsigned int framesToAllocate
	)
{
	map<unsigned char, vector<unsigned int>> monitorPositions;

	auto layers = m_configuration.PmlLayers;

	for (unsigned int j = layers; j < height - 1 - layers; ++j)
	{
		for (unsigned int i = layers; i < width - 1 - layers; ++i)
		{
			unsigned int pixelOffset = channels  * (j * width + i);

			/// get third byte for this pixel (blue component), which is the monitor ID if this is a monitor cell.
			unsigned char monitorId = bytes[pixelOffset + 2];

			if (monitorId > 0)
			{
				auto &mp = monitorPositions[monitorId];

				unsigned int monitorOffset = j * m_fields[FieldType::Ez]->Size.x + i;
				mp.push_back(monitorOffset);
			}

		}
	}

	for (auto it = begin(monitorPositions); it != end(monitorPositions); ++it)
	{
		vector<unsigned int> &positions = it->second;

		auto ptr = make_shared<Monitor>();
		ptr->Id = it->first;

		m_monitors[it->first] = ptr;

		ptr->Initialize(m_cuda, positions, 1);
	}
}

