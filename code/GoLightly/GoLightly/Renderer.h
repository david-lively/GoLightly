#ifndef RENDERER_H
#define RENDERER_H

#include <vector>
#include <string>
#include "../Thirdparty/soil/SOIL2.h"
#include "Vectors.h"

using namespace std;

/// UI renderer for FDTD simulator
class Renderer
{
public:
	Renderer();
	~Renderer();

	void Initialize();

	void Shutdown();
	void Resize(unsigned int width, unsigned int height)
	{
		m_viewPortSizeX = width;
		m_viewPortSizeY = height;
	}

	void Print(float x, float y, const std::string &text);

	/// enqueue a string for deferred rendering, splitting into multiple lines where necessary
	void Print(const std::string &text);

	void Render(float elapsedSeconds);

	void MoveTo(unsigned int x, unsigned int y)
	{
		m_cursorX = x;
		m_cursorY = y;
	}

	unsigned int ViewportWidth() { return m_viewPortSizeX; }
	unsigned int ViewportHeight() { return m_viewPortSizeY; } 

	void DrawQuad(int x, int y, int width, int height, const vec4 &color);
	/// render a graph inline with the text
	void DrawGraph(
		const string &title
		,int x
		, int y
		, unsigned int width
		, unsigned int height
		, const vec4 &foregroundColor
		, const vec4 &backgroundColor
		, vector<float> &data
		, unsigned int first
		, unsigned int sampleCount
		, float normalizeMin
		, float normalizeMax);

private:
	vector<string> m_strings;
	//vector<tuple<vec2,string>> m_positionedStrings;

	/// text cursor X position
	unsigned int m_cursorX;
	/// text cursor Y position
	unsigned int m_cursorY;

	/// GL handle to a font texture
	GLuint m_font;

	int m_fontWidth, m_fontHeight;
	unsigned int m_viewPortSizeX, m_viewPortSizeY;

	/// text rendering shader program
	GLuint m_textProgram;
	GLuint m_textBuffer;

	/// graph rendering shader program
	GLuint m_screenPrimitiveProgram;
	GLuint m_graphProgram;


	GLuint m_quadVertexBuffer;
	GLuint m_graphDataBuffer;

	int GetActiveProgram()
	{
		int program = 0;
		gl::GetIntegerv(gl::CURRENT_PROGRAM, &program);

		return program;
	}

	void SetUniform(const string &name, unsigned int value)
	{
		int program = GetActiveProgram();;

		auto location = gl::GetUniformLocation(program, name.data());

		if (location >= 0)
		{
			gl::Uniform1i(location,value);
		}
	}

	void SetUniform(const string &name, float v0, float v1)
	{
		int program = GetActiveProgram();;

		auto location = gl::GetUniformLocation(program, name.data());

		if (location >= 0)
		{
			gl::Uniform2f(location,v0,v1);
		}
	}

	void SetUniform(const string &name, const vec4 &value)
	{
		int program = GetActiveProgram();;

		auto location = gl::GetUniformLocation(program, name.data());

		if (location >= 0)
		{
			gl::Uniform4f(location,value.x, value.y, value.z, value.w);
		}
	}
	void PrePrint();
	void DoPrint(GLint positionAttributeLocation, float x, float y, const string &text);
	void PostPrint();

};


#endif // !RENDERER_H
