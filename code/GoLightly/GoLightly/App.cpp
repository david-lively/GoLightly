/// GLFW *must* be included before GL.h (or gl_core*.h in this case)
#include <iostream>
#include <vector>
#include <string>

using namespace std;

#include "Common.h"
#include "App.h"


/// <summary>
/// instance of App used for GLFW callbacks
/// </summary>
App *App::m_instance = nullptr;

App::App(void) :
	m_window(nullptr)
	,m_width(1280)
	,m_height(720)
	,m_noGL(false)
	,m_fps(0.f)
	,m_frameNumber(0)
	,m_timeHistoryTotal(0.f)
	,m_shouldClose(false)
{

	m_currentKeyboardState.fill(false);
	m_lastKeyboardState.fill(false);

	cout << "Press ESCAPE to exit" << endl;

	m_timeHistory.fill(0);
}

App::~App(void)
{
}

void App::OnReady()
{
}

void App::Render(float elapsedSeconds)
{
}

void App::Update(float elapsedSeconds)
{
	static int timeIndex = 0;

	m_timeHistoryTotal -= m_timeHistory[timeIndex];

	m_timeHistoryTotal += (m_timeHistory[timeIndex] = elapsedSeconds);

	m_fps = m_timeHistoryTotal > 0 ? m_timeHistory.size() / m_timeHistoryTotal : 0.f;

	timeIndex = (timeIndex + 1) % m_timeHistory.size();

	m_runTimeMs += elapsedSeconds * 1000.f;

	if (!m_noGL && GetInputTransition(GLFW_KEY_ESCAPE) == InputTransition::Release)
		glfwSetWindowShouldClose(m_window,1);

	/// only update status messages every 100 frames
	if (++m_frameNumber % 1000 != 0)
		return;

	auto updateMessage = GetUpdateStatusMessage(elapsedSeconds);


	if (m_noGL)
	{
		cout << updateMessage  << endl;
	}
	else
	{
		glfwSetWindowTitle(m_window,updateMessage.c_str());
	}
}

std::string App::GetUpdateStatusMessage(float elapsedSeconds)
{
	return "FPS " + to_string(m_fps) + " elapsed " + to_string(elapsedSeconds) + " total " + to_string(m_runTimeMs);
}


InputTransition App::GetInputTransition(int key)
{
	static std::array<InputTransition,4> transitions = { InputTransition::None, InputTransition::Release, InputTransition::Press, InputTransition::Hold };

	bool prev = m_lastKeyboardState[key];
	bool current = m_currentKeyboardState[key];

	InputTransition transition = transitions[current * 2 + prev];

	return transition;
}

void App::Run()
{
	DWORD startTicks = GetTickCount();

	if (m_noGL)
		RunWithoutGL();
	else
		RunWithGL();

	DWORD stopTicks = GetTickCount();

	float seconds = (stopTicks - startTicks) / 1000.f;

	cout << "Speed: " << m_frameNumber / seconds << " fps. " << m_frameNumber << " frames took " << seconds << " seconds\n";

}

void App::RunWithGL()
{
	float lastTime = static_cast<float>(glfwGetTime());
	float time = 0;


	m_frameNumber = 0;

	while(!glfwWindowShouldClose(m_window))
	{
		time = static_cast<float>(glfwGetTime());

		float elapsedSeconds = time - lastTime;

		copy(begin(m_currentKeyboardState),end(m_currentKeyboardState),begin(m_lastKeyboardState));

		glfwPollEvents();

		Update(elapsedSeconds);

		gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

		Render(elapsedSeconds);

		glfwSwapBuffers(m_window);

		lastTime = time;
	}

	OnShutdown();

	glfwTerminate();

	m_window = nullptr;

}

void App::RunWithoutGL()
{
	DWORD lastTicks;
	DWORD ticks = 0;

	m_frameNumber = 0;
	m_runTimeMs = 0;

	lastTicks = GetTickCount();

	while(!m_shouldClose)
	{
		ticks = GetTickCount(); 

		Update((ticks - lastTicks) / 1000.f);

		lastTicks = ticks;
	}

	OnShutdown();


}

void App::OnShutdown()
{
}


/// <summary>
/// return a pointer to a new, initialized App window
/// </summary>
/// <param name="window">The window.</param>
/// <returns></returns>
bool App::Initialize(unsigned int width, unsigned int height, const string &title, bool noGL, int argc, char *argv[])
{
	m_noGL = noGL;
	if (!noGL)
	{
		if (m_window || m_instance)
		{
			OutputDebugString("Already initialized. This implementation allows only one instance of the App class.");
			return false;
		}

		if(!glfwInit())
		{
			return false;
		}

		/// turn on some simple screen space anti-aliasing
		glfwWindowHint(GLFW_SAMPLES,16);
		m_window = glfwCreateWindow(width,height,title.c_str(),NULL,NULL);

		if(!m_window)
		{
			OutputDebugString("Could not create GLFW window");
			return false;
		}

		glfwMakeContextCurrent(m_window);

		if(!gl::sys::LoadFunctions())
		{
			glfwTerminate();
			OutputDebugString("Could not initialize App extensions");
			return false;
		}

		m_width = width;
		m_height = height;

		gl::Viewport(0,0,m_width,m_height);
		gl::Enable(gl::MULTISAMPLE);

		glfwSetWindowSizeCallback(m_window,App::WindowResizeCallback);
		glfwSetKeyCallback(m_window,App::KeyboardCallback);

		glfwSwapInterval(0);


		m_instance = this;
	}

	OnReady();

	return true;	
}

void App::Resize(unsigned int width, unsigned int height)
{
	m_width = width;
	m_height = height;
	gl::Viewport(0,0,width,height);

	OnResize(width,height);
}

void App::OnResize(unsigned int width, unsigned int height)
{
}


void App::WindowResizeCallback(GLFWwindow* window, int width, int height)
{
	if (window != m_instance->m_window)
		throw runtime_error("Received callback for the wrong window!");

	m_instance->Resize(width,height);
}

/// <summary>
/// Keyboard action callback from GLFW
/// </summary>
/// <param name="window">window that received the event</param>
/// <param name="key">keyboard key that was pressed or released</param>
/// <param name="scancode">system-specific scancode of the key</param>
/// <param name="action">GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT</param>
/// <param name="mods">Bit field describing which modifier keys were held down</param>
/// <remarks>
/// </remarks>
void App::KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	m_instance->m_currentKeyboardState[key] = action != 0;
}


