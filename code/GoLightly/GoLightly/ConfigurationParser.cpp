#include <iostream>
#include <algorithm>
#include <string> 

using namespace std;

#include "ConfigurationParser.h"

map<string,Token> ConfigurationScanner::m_tokens = CreateTokenMap();

void ConfigurationScanner::FromCommandLine(Configuration &config, int argc, char *argv[])
{
	cout << "Parsing command line:\n";

	unsigned int i = 1;

	while(i < argc)
	{
		string word = GetValue<string>(argv,i);

		/// convert to lowercase
		transform(begin(word),end(word),begin(word),::tolower);

		/// get the token for this word
		auto token = m_tokens[word];

		/// commands that don't require a parameter
		switch(token)
		{
		case Token::NoGL:
			cout << "Disabling visualizer\n";
			continue;

		case Token::AutoTune:

			cout << "Auto-tuning enabled\n";
			config.AutoTune = true;
			continue;
		}

		/// switches requiring parameters
		if (i >= argc) 
		{
			cerr << "Switch \"" << word << "\" requires a parameter value which was not provided.\n";
			exit(EXIT_FAILURE);
		}

		switch(token)
		{
		case Token::NormalizeOutput:
			config.NormalizeOutput = GetValue<bool>(argv,i);
			cout << "Output normalization: " << (config.NormalizeOutput ? "enabled" : "disabled") << endl;
			break;

		case Token::ModelPath:
			config.ModelPath = GetValue<string>(argv,i);
			cout << "Using model \"" << config.ModelPath << "\"\n";
			break;

		case Token::OutputPath:
			config.OutputPath = GetValue<string>(argv,i);
			cout << "Output will be saved to \"" << config.OutputPath << "\"\n";
			break;


		case Token::Lambda:
			config.Lambda = GetValue<float>(argv,i);
			cout << "Lambda = " << config.Lambda << endl;
			break;


		case Token::RunLength:
			config.SimulationLength = GetValue<int>(argv,i);
			cout << "Simulation length = " << config.SimulationLength << endl;
			break;


		case Token::UpdateFrequency:
			config.VisualizerUpdateFrequency = GetValue<int>(argv,i);
			cout << "Visualizer update frequency = " << config.VisualizerUpdateFrequency << endl;
			break;

		case Token::MediaCoefficient:
			config.EpsilonR = GetValue<float>(argv,i);
			cout << "Epsilon_r = " << config.EpsilonR << endl;
			break;

		case Token::DomainWidth:
			config.DomainSize.x = GetValue<int>(argv, i);
			cout << "Domain width = " << config.DomainSize.x;
			break;

		case Token::DomainHeight:
			config.DomainSize.y = GetValue<int>(argv, i);
			cout << "Domain width = " << config.DomainSize.y;
			break;

		case Token::StartPaused:
			config.StartPaused = GetValue<bool>(argv,i);
			cout << "Start paused: " << (config.StartPaused ? "true" : "false") << endl;
			break;

		case Token::FilenamePrefix:
			config.FilenamePrefix = GetValue<string>(argv,i);
			cout << "Output filename prefix: \"" << config.FilenamePrefix << "\"\n";
			break;

		case Token::Benchmark:
			config.Benchmark = true;
			cout << "Benchmarking - save metrics after each run.\n";
			break;

		case Token::Unknown:
		default:
			cout << "Unknown command line option: \"" << word << "\"\n";
			exit(EXIT_FAILURE);
		}
	}
}

