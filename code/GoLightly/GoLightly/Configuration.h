#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <string>
#include <map>

#include "CudaHelpers.h"

class Configuration;

namespace std
{
	std::string to_string(const Configuration &config);
}


/// <summary>
/// Simulation configuration data
/// </summary>
class Configuration
{
public:
	float Dt;
	float Dx;
	float Lambda;
	float Frequency;
	float EpsilonR;
	bool StartPaused;
	std::string FilenamePrefix;
	bool Benchmark;
	/// number of samples that make one period of the source
	unsigned int SamplesPerPeriod;

	unsigned int SourceDuration;
	float SourceDecayRate;

	std::string ModelPath;
	std::string OutputPath;
	unsigned int SimulationLength;

	bool EnableVisualizer;
	
	/// self-tune grid/block sizes, visualizer update freuqquency, etc.
	bool AutoTune;

	/// number of CUDA updates to execute per visualizer update.
	unsigned int VisualizerUpdateFrequency;

	/// normalize all output 
	bool NormalizeOutput;

	dim3 DomainSize;

	bool EnablePml;
	unsigned int PmlSigmaOrder;
	float PmlSigmaMax;
	unsigned int PmlLayers;

	/// emit some helpful statistics to the console while running
	bool VerboseOutput;


	Configuration();
	Configuration(dim3 domainSize, float lambda);
	~Configuration();

	void SetDefaults();
	void InitPML();

	std::string ToString() const;

private:



};




#endif