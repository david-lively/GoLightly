#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>

#include "Configuration.h"


namespace std
{
	string to_string(const Configuration &config)
	{
		return config.ToString();
	}

}

using namespace std;

Configuration::Configuration()
{
	SetDefaults();


	InitPML();

}

Configuration::Configuration(dim3 domainSize, float lambda)
{
	SetDefaults();

	Lambda = lambda;

	DomainSize = domainSize;

	Dx = lambda / 10.f;
	Dt = Dx / sqrt(2.f) * 0.95f;


	Frequency = 1.f / lambda;

	InitPML();
}


Configuration::~Configuration()
{

}

void Configuration::SetDefaults()
{
	AutoTune = false;
	NormalizeOutput = true;
	Frequency = 1;//0.166f;
	Lambda = 1.f / Frequency;
	SamplesPerPeriod = 10;
	DomainSize = dim3(512,512);
	EpsilonR = 9.f;

	Dx = Lambda / 10.f;
	Dt = Dx / sqrt(2.f) * 0.95;

	SimulationLength = 2048;
	SourceDuration = numeric_limits<unsigned int>::max();
	ModelPath = "";
	OutputPath = "";
	VerboseOutput = false;

	EnablePml = false;
	PmlSigmaOrder = 4;
	PmlSigmaMax = 1.f;
	PmlLayers = 10;


	EnableVisualizer = true;
	VisualizerUpdateFrequency = 1;

	StartPaused = false;

	FilenamePrefix = "";
}

void Configuration::InitPML()
{
	float mu0 = 1.f;
	float eps0 = 1.f;
	float epsR = 1.f;

	EnablePml = true;
	PmlSigmaOrder = 4;
	PmlSigmaMax = 1.f;
	PmlLayers = 10;
	PmlSigmaMax = 0.75f * (0.8f * (PmlSigmaOrder + 1) / (Dx * (float)pow(mu0 / (eps0 * epsR), 0.5f)));
}



string Configuration::ToString() const
{
	stringstream ss;

	ss << "\n\tDt         " << Dt << endl;
	ss << "\tDx         " << Dx << endl;
	ss << "\tLambda     " << Lambda << endl;
	ss << "\tSim length " << SimulationLength << endl;
	ss << "\tDomain     (" << DomainSize.x << "," << DomainSize.y << "," << DomainSize.z << ")\n";
	ss << "\tModel		\"" << ModelPath << "\"\n";
	ss << "\tVisualizer " << (EnableVisualizer ? "ON" : "OFF") << endl;

	if (EnableVisualizer)
	{
		ss << "\tVisualizer update frequency " << VisualizerUpdateFrequency << endl;
	}

	ss << "\tPml "<< (EnablePml ? "ON" : "OFF") << endl;

	if (EnablePml)
	{
		ss << "\t\tSigma Order " << PmlSigmaOrder << endl;
		ss << "\t\tSigma max   " << PmlSigmaMax << endl;
		ss << "\t\tLayers      " << PmlLayers << endl;
	}

	return ss.str();
}




