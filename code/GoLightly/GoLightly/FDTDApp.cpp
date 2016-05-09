#include <array>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "ConfigurationParser.h"
#include "Common.h"
#include "FDTDApp.h"
#include "../Common/Files.h"
#include "Renderer.h"

#include "Vectors.h"

#define GRAPH_RMS

using namespace std;

FDTDApp::FDTDApp(void) :
	m_tuning(false)
	,m_tuneVisualizerUpdateMin(1)
	,m_tuneVisualizerUpdateMax(255)
	,m_tuneFrequency(128)
	,m_tuneVisualizerBestTime(numeric_limits<float>::max())
	,m_uiScrollFactor(0)
	,m_uiScrollDirection(0)
	,m_showHelp(false)
	,m_graphBottomMargin(0)
{
}

FDTDApp::~FDTDApp(void)
{

}


void FDTDApp::Run()
{
	auto startTicks = GetTickCount();

	App::Run();

	auto stopTicks = GetTickCount();

	float seconds = (stopTicks - startTicks) / 1000.f;
	float fps = FrameNumber() / seconds;

	auto& config = m_simulator.GetConfiguration();

	if (config.Benchmark)
	{
		string filename = config.FilenamePrefix + "benchmark.csv";
		cout << "Writing metrics to \"" << filename << "\"\n";

		ofstream outfile;

		outfile.open(filename, ios_base::app);
		outfile << config.DomainSize.x << "," << config.DomainSize.y << "," << fps << "," << seconds << endl;
		outfile.close();
	}

}


bool FDTDApp::Initialize(unsigned int width, unsigned int height, const std::string &title, bool noGL, int argc, char *argv[])
{
	/// Update the Configuration object based on command line parameters or script
	ConfigurationScanner::FromCommandLine(m_simulator.GetConfiguration(), argc, argv);
	m_simulator.GetConfiguration().EnableVisualizer = !noGL;

	cudaEventCreate(&m_tuneEventStart);
	cudaEventCreate(&m_tuneEventStop);

	bool result = App::Initialize(width,height,title,noGL,argc,argv);

	if (!result)
	{
		cerr << "Could not initialize application. Set a breakpoint and see what happened.\n";
		exit(EXIT_FAILURE);
	}

	/// rgb byte values for ROYGBIV
	array<unsigned char, 7 * 3> rainbowBytes = 
	{
		0xff,	0,		0		// red
		,
		0xff,	0x7f,	0		// orange
		,
		0xff,	0xff,	0		// yellow
		,
		0,		0xff,	0		// green
		,
		0,		0,		0xff	// blue
		,
		0x4b,	0,		0x82	// indigo
		,
		0x8b,	0,		0xff	// violet
	};

	if(!NoGL())
	{
		m_render.Initialize();

		/// just in case...
		m_rainbow.clear();

		/// generate rainbow color set
		float scale = 1.f / 255;

		for(unsigned int i = 0; i < rainbowBytes.size() - 3; i += 3)
		{
			m_rainbow.push_back(vec4(rainbowBytes[i] * scale, rainbowBytes[i + 1] * scale, rainbowBytes[i + 2] * scale, 1));
		}

	}

	return true;
}

void FDTDApp::UpdateSimulator(float elapsedSeconds)
{
	m_simulator.Update(elapsedSeconds);
}

void FDTDApp::UpdateVisualizer(float elapsedSeconds)
{
	if (!m_simulator.GetConfiguration().EnableVisualizer)
		return;


	bool fastChange = GetInputTransition(GLFW_KEY_LEFT_SHIFT) != InputTransition::None;
	float delta = fastChange ? 1000 : 100;

	if (GetInputTransition(GLFW_KEY_UP) != InputTransition::None)
		m_visualizer.ColorScale += delta * elapsedSeconds;
	
	if (GetInputTransition(GLFW_KEY_DOWN) != InputTransition::None)
		m_visualizer.ColorScale = max<float>(0,m_visualizer.ColorScale - delta * elapsedSeconds);

	if (GetInputTransition(GLFW_KEY_U) == InputTransition::Release)
	{
		/// show or hide UI elements
		m_uiScrollDirection = 1 - m_uiScrollDirection;		
	}

	// TODO: add "reset" command implementation
	//if (GetInputTransition(GLFW_KEY_R) == InputTransition::Release)
	//{
	//	m_simulator.Reset();
	//}

	if (GetInputTransition(GLFW_KEY_F1) == InputTransition::Release)
	{
		m_showHelp = !m_showHelp;
	}

	if (GetInputTransition(GLFW_KEY_F2) == InputTransition::Release)
	{
		string path = m_simulator.GetConfiguration().OutputPath;

		if (path == "")
		{
			path = Files::GenerateFilename("fdtd-",".csv");
		}

		cout << "Saving Ez to \"" << path << "\"\n";

		m_simulator.Save(FieldType::Ez,"\\fdtd_data\\" + path);
	}

	if (GetInputTransition(GLFW_KEY_P) == InputTransition::Release)
	{
		m_simulator.TogglePause();
	}

	if (FrameNumber() % m_simulator.GetConfiguration().VisualizerUpdateFrequency == 0)
	{
		float *field;
		float *materials;
		unsigned int width;
		unsigned int height;

		m_simulator.GetPreviewField(&field, &materials, &width, &height);

		m_visualizer.Update(elapsedSeconds, field, width, height, materials);
	}


	if (m_tuning)
	{
		cudaEventRecord(m_tuneEventStart);
		m_tuneFramesRemaining--;
	}

	static float tuneCurrentMs = 0;

	if (m_tuning)
	{
		float ms = 0;

		cudaEventRecord(m_tuneEventStop);
		cudaEventElapsedTime(&ms, m_tuneEventStart, m_tuneEventStop);

		tuneCurrentMs += ms;

		if (m_tuneFramesRemaining == 0 && m_tuneVisualizerUpdateMin != m_tuneVisualizerUpdateMax)
		{
			/// update visualizer framerate based on whether we are running faster or slower than previous guess
			/// TODO: extrapolate this into a separate tuning class that can tweak a specific value (visualizer update, grid/block)
			/// assume higher visualizer update frequency == slower

			if (tuneCurrentMs < m_tuneVisualizerBestTime)
			{
				// move frequency halfway between current and max
			}
			else
			{
				/// move frequency haflway between current and min

			}

		}
	}

}


void FDTDApp::Update(float elapsedSeconds)
{	
	App::Update(elapsedSeconds);

	UpdateSimulator(elapsedSeconds);

	if(!NoGL())
		UpdateVisualizer(elapsedSeconds);


	/// exit when we've reached the end of the simulation
	if (FrameNumber() == m_simulator.GetConfiguration().SimulationLength)
	{
		Close();

		string path = m_simulator.GetConfiguration().OutputPath;

		if (path == "")
			path = Files::GenerateFilename("fdtd-",".csv");


		cout << "Saving Ez to \"" << path << "\"\n";
		m_simulator.Save(FieldType::Ez,"\\fdtd_data\\" + path);
	}
}

string FDTDApp::GetUpdateStatusMessage(float elapsedSeconds)
{
	auto status = App::GetUpdateStatusMessage(elapsedSeconds);

	status = 
		"Frame " + to_string(FrameNumber()) 
		+ " " + status 
		+ " Time " + to_string(m_simulator.GetSimulatedTime()
		);

	if (m_simulator.IsPaused())
		status = "PAUSED " + status;
	
	static string helpMessage = 
		"\n"
		" F1    Show help\n"											
		" F2    Save monitor history\n"								
		" Up    Increase Contrast (hold SHIFT for fast change)\n"	
		" Down  Decrease Contrast (hold SHIFT for fast change)\n"	
		" U     Show/Hide UI\n"										
		" P     Pause simulation\n"									
		" ESC   Exit"												
		;

	if (m_simulator.GetConfiguration().EnableVisualizer)
	{
		if (m_showHelp)
			status += helpMessage;
		else
			status += "\nF1 for help";
	}

	return status;
}


void FDTDApp::Render(float elapsedSeconds)
{
	if (!m_visualizer.Enabled)
		return;

	App::Render(elapsedSeconds);



	m_visualizer.Render(elapsedSeconds);


	m_render.Print(GetUpdateStatusMessage(elapsedSeconds));

	static float mn = numeric_limits<float>::max();
	static float mx = numeric_limits<float>::min();


	/// draw a graph showing monitors status
	auto &monitors = m_simulator.Monitors();

	unsigned int height = 128;
	unsigned int padding = 20;
	unsigned int width = floor<int>(m_render.ViewportWidth() * 1.f / monitors.size()) - padding;
	
	unsigned int x = padding;
	//unsigned int y = m_render.ViewportHeight() - padding - height;
	unsigned int y = m_render.ViewportHeight() - height - padding;

	static float normalizeMin = numeric_limits<float>::max();
	static float normalizeMax = numeric_limits<float>::min();
	static unsigned int totalHeight = height + padding;
	static unsigned int totalWidth = (width + padding);

	unsigned int monitorIndex = 0;

	for(auto it = begin(monitors); it != end(monitors); ++it)
	{
		auto &monitor = *it->second;

		
		vec4 color = m_rainbow[monitorIndex % m_rainbow.size()];
		vec4 backgroundColor = m_rainbow[(monitorIndex + 3) % m_rainbow.size()];

#ifdef GRAPH_RMS
		normalizeMin = min<float>(normalizeMin,monitor.RmsMinValue);
		normalizeMax = max<float>(normalizeMax,monitor.RmsMaxValue);

		unsigned int displaySamples = min<unsigned int>(300,monitor.RmsHistory.size());

		m_render.DrawGraph(
			"Monitor " + to_string(it->first)
			,x 
			,y + m_uiScrollFactor * totalHeight
			,width
			,height
			,color
			,backgroundColor
			,monitor.RmsHistory
			,max<int>(0,monitor.RmsHistory.size() - displaySamples)
			,width
			,normalizeMin
			,normalizeMax
			);
#else
		normalizeMin = min<float>(normalizeMin,monitor.MinValue);
		normalizeMax = max<float>(normalizeMax,monitor.MaxValue);
		
		unsigned int displaySamples = min<unsigned int>(300,monitor.MagnitudeHistory.size());

		m_render.DrawGraph(
			"Monitor " + to_string(it->first)
			,x 
			,y + m_uiScrollFactor * totalHeight
			,width
			,height
			,color
			,backgroundColor
			,monitor.MagnitudeHistory
			,max<int>(0,monitor.MagnitudeHistory.size() - displaySamples)
			,width
			,normalizeMin
			,normalizeMax
			);
#endif
		x += width + padding;

		monitorIndex++;
	}

	//m_render.Print(0,0,"testing");

	switch(m_uiScrollDirection)
	{
	case 0 :
		break;
	case 1 :
		break;
	default:
		throw runtime_error("Invalid scroll direction value");
	}

	const float scrollSpeed = 0.1f;

	m_uiScrollFactor += (m_uiScrollDirection * 2 - 1) * scrollSpeed;
	m_uiScrollFactor = max(min(m_uiScrollFactor,1.f),0.f);

	m_render.Render(elapsedSeconds);
}

void FDTDApp::RenderUI(float elapsedSeconds)
{
}


void FDTDApp::OnReady()
{
	unsigned int windowWidth = 1280;
	unsigned int windowHeight = 720;
	unsigned int sideLength = 1024;

	dim3 domainSize(sideLength, sideLength);

	m_simulator.Initialize();
	m_visualizer.Enabled = !NoGL();
	m_visualizer.Ready();
	m_visualizer.Resize(domainSize.x,domainSize.y, windowWidth, windowHeight);

	InitializeUI();
}

void FDTDApp::InitializeUI()
{
	auto &monitors = m_simulator.Monitors();

	unsigned int count = monitors.size();

	/// generate a grid of monitor locations

	if (count == 0)
		return;

	// 



}

void FDTDApp::OnResize(unsigned int width, unsigned int height)
{
	m_render.Resize(width,height);
	cout << "Resize(" << width << "," << height << ")\n";
	auto domainSize = m_simulator.GetConfiguration().DomainSize;
	m_visualizer.Resize(domainSize.x, domainSize.y, width,height);
}

void FDTDApp::OnShutdown()
{
	if(!NoGL())
		m_render.Shutdown();
	m_visualizer.Shutdown();
	m_simulator.Shutdown();
	App::OnShutdown();
}

