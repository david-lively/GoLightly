#ifndef FDTDAPP_H
#define FDTDAPP_H

#include <vector>
#include <string>

#include "App.h"
#include "Simulator.cuh"
#include "Visualizer.cuh"
#include "CudaHelpers.h"
#include "Renderer.h"
#include "Vectors.h"

class FDTDApp :
	public App
{
public:
	FDTDApp(void);
	~FDTDApp(void);

	void Run() override;
	bool Initialize(unsigned int width = 1280, unsigned int height = 720, const std::string &title = "OpenGL App", bool noGL = false, int argc = 1, char *argv[] = nullptr) override;

	void Update(float elapsedSeconds) override;
	void Render(float elapsedSeconds) override;
	void RenderUI(float elapsedSeconds);

	std::string GetUpdateStatusMessage(float elapsedSeconds) override;

	void OnReady() override;
	void OnShutdown() override;
	void OnResize(unsigned int width, unsigned int height) override;

	void InitializeUI();

private:
	Simulator m_simulator;
	Visualizer m_visualizer;
	Renderer m_render;
	vector<vec4> m_rainbow;

	/// position and size of each graph
	map<unsigned char, vec4> m_graphLocations;

	///TODO: Move this stuff into a dedicated "Tuning" class.
	/// true if currently auto-tuning
	bool m_tuning;
	bool m_showHelp;

	unsigned int m_graphTransparencyMode;
	unsigned int m_graphBottomMargin;

	/// scroll-in/scroll-out transition for UI elements
	float m_uiScrollFactor;
	int m_uiScrollDirection;

	float m_tuneVisualizerBestTime;
	unsigned int m_tuneVisualizerUpdateMin;
	unsigned int m_tuneVisualizerUpdateMax;
	unsigned int m_tuneFrequency;
	unsigned int m_tuneFramesRemaining;

	cudaEvent_t m_tuneEventStart;
	cudaEvent_t m_tuneEventStop;

	void UpdateSimulator(float elapsedSeconds);
	void UpdateVisualizer(float elapsedSeconds);
};

#endif
