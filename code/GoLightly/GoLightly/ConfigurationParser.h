#ifndef CONFIGURATIONPARSER_H
#define CONFIGURATIONPARSER_H


#include <string>
#include <map>

using namespace std;


#include "Configuration.h"

enum class Token
{
	Unknown
	,
	ModelPath
	,
	OutputPath
	,
	Lambda /// source wavelength
	,
	RunLength
	,
	NoGL
	,
	UpdateFrequency
	,
	MediaCoefficient
	,
	NormalizeOutput
	,
	AutoTune
	,
	StartPaused
	,
	FilenamePrefix
	,
	DomainWidth
	,
	DomainHeight
	,
	Benchmark
};


class ConfigurationScanner
{
public:
	static void FromCommandLine(Configuration &config, int argc, char *argv[]);

private:

	static std::map<std::string,Token> CreateTokenMap()
	{
		std::map<std::string,Token> result;

		result["-nogl"  ] = Token::NoGL;
		result["-model"	] = Token::ModelPath;
		result["-lambda"] = Token::Lambda;
		result["-runlength"] = Token::RunLength;
		result["-rate"	] = Token::UpdateFrequency;
		result["-media"] = Token::MediaCoefficient;
		result["-output"] = Token::OutputPath;
		result["-normalize"] = Token::NormalizeOutput;
		result["-tune"] = Token::AutoTune;
		result["-paused"] = Token::StartPaused;
		result["-prefix"] = Token::FilenamePrefix;
		result["-width"]  = Token::DomainWidth;
		result["-height"] = Token::DomainHeight;
		result["-benchmark"] = Token::Benchmark;


		return result;
	}
	static std::map<std::string,Token> m_tokens;


	template<typename T>
	static T PeekValue(char *argv[], unsigned int index)
	{
		return static_cast<T>(string(argv[index]));
	}

	template<>
	static float PeekValue(char *argv[], unsigned int index)
	{
		return stof(argv[index]);
	}

	template<>
	static string PeekValue(char *argv[], unsigned int index)
	{
		return string(argv[index]);
	}

	template<>
	static int PeekValue(char *argv[], unsigned int index)
	{
		return stoi(string(argv[index]));
	}

	template<>
	static bool PeekValue(char *argv[], unsigned int index)
	{
		return stoi(string(argv[index]));
	}

	template<typename T>
	static T GetValue(char *argv[], unsigned int &index)
	{
		return PeekValue<T>(argv,index++);
	}


};


#endif
