#ifndef OPENGL_H
#define OPENGL_H

#include <string>
#include <array>

#include "Common.h"
#include "Input.h"

/// <summary>
/// Provides App helper methods
/// </summary>
class App
{
public:
	App(void);
	virtual ~App(void) = 0;

	virtual bool Initialize(unsigned int width = 1280, unsigned int height = 720, const std::string &title = "OpenGL App", bool noGL = false, int argc = 1, char *argv[] = nullptr);

	virtual void Run();

	virtual void OnReady();
	virtual void OnShutdown();

	virtual std::string GetUpdateStatusMessage(float elapsedSeconds);
	virtual void Update(float elapsedSeconds);
	virtual void Render(float elapsedSeconds);
	void Resize(unsigned int width, unsigned int height);
	virtual void OnResize(unsigned int width, unsigned int height);

	int Width() { return m_width; }
	int Height() { return m_height; } 

	const int FrameNumber() const { return m_frameNumber; } 

	InputTransition GetInputTransition(int key);

	bool NoGL() { return m_noGL; }

	void Close() { m_shouldClose = true; } 
private:
	bool m_noGL;
	bool m_shouldClose;
	GLFWwindow *m_window;

	unsigned int m_width;
	unsigned int m_height;

	/// total update or render frames executed 
	int m_frameNumber;

	/// run loop execution time in milliseconds
	int m_runTimeMs; 

	/// calculated run speed 
	float m_fps;
	/// sum of all values currently in m_timeHistory
	float m_timeHistoryTotal;


	std::array<float,100> m_timeHistory;
	/// <summary>
	/// instance of App used for GLFW callbacks
	/// </summary>
	static App *m_instance;

	KeyboardState m_currentKeyboardState;
	KeyboardState m_lastKeyboardState;

	static void WindowResizeCallback(GLFWwindow*,int,int);
	static void KeyboardCallback(GLFWwindow*,int,int,int,int);

	void RunWithGL();
	void RunWithoutGL();


};

#endif