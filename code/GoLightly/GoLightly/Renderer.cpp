#include <iostream>
#include <string>
#include <vector>
#include <minmax.h>

using namespace std;

#include "OpenGLHelpers.h"
#include "Renderer.h"
#include "Vectors.h"

Renderer::Renderer() : 
	m_font(0)
	,m_fontWidth(0)
	,m_fontHeight(0)
	,m_textBuffer(0)
	,m_viewPortSizeX(1280)
	,m_viewPortSizeY(720)
	,m_cursorX(0)
	,m_cursorY(0)
	,m_screenPrimitiveProgram(0)
	,m_quadVertexBuffer(0)
	,m_graphProgram(0)
{
}

Renderer::~Renderer()
{
}

void Renderer::Print(const string &text)
{
	auto start = 0;

	/// split string on newlines.
	for(auto i = 0; i < text.length(); ++i)
	{
		if (text[i] == '\n')
		{
			if (i < start)
				throw;

			m_strings.push_back(text.substr(start,i - start + 1));	
			start = i + 1;			
		}
	}
	if (start != text.length() - 1)
		m_strings.push_back(text.substr(start,text.size() - start));
}

void Renderer::Initialize()
{
#pragma region text resources
	// load texture containing our signed distance field font
	m_font = SOIL_load_OGL_texture(
		//"lucidagrande_sdf.jpg"
		"resources\\courier_sdf.png"
		,SOIL_LOAD_AUTO
		,SOIL_CREATE_NEW_ID
		//,SOIL_FLAG_INVERT_Y
		,0
		,&m_fontWidth
		,&m_fontHeight
		);

	check_gl_error();

	if (!m_font)
	{
		cerr << "Could not load font texture\n"
			<< "SOIL result: \"" << SOIL_last_result() 
			<< "\"\n";
		exit(EXIT_FAILURE);
	}

	char *requiredUniforms[] = {"Font"};

	unsigned int requiredUniformsCount = 1;

	m_textProgram = OpenGLHelpers::CreateProgram("Text",requiredUniforms,requiredUniformsCount);

	gl::GenBuffers(1,&m_textBuffer);

	check_gl_error();	

#pragma endregion

#pragma region graph resources
	m_screenPrimitiveProgram = OpenGLHelpers::CreateProgram("ScreenPrimitive");
	gl::GenBuffers(1,&m_quadVertexBuffer);

	// generate a triangle-strip quad definition
	vector<vec2> quadVertices;
	quadVertices.reserve(4);

	quadVertices.push_back(vec2(1 , 0));
	quadVertices.push_back(vec2(1 , 1));
	quadVertices.push_back(vec2(0 , 0));
	quadVertices.push_back(vec2(0 , 1));

	gl::BindBuffer(gl::ARRAY_BUFFER,m_quadVertexBuffer);
	gl::BufferData(gl::ARRAY_BUFFER, sizeof(vec2) * quadVertices.size(), quadVertices.data(), gl::STATIC_DRAW);
	gl::BindBuffer(gl::ARRAY_BUFFER, 0);

	gl::GenBuffers(1, &m_graphDataBuffer);

	m_graphProgram = OpenGLHelpers::CreateProgram("Graph");


#pragma endregion



}


/// private print, assumes current gl state (depth, culling, program and texture binding) is valid
void Renderer::DoPrint(GLint positionAttributeLocation, float x, float y, const string &text)
{
	SetUniform("StartPosition",x,y);

	gl::BufferData(gl::ARRAY_BUFFER, sizeof(unsigned char) * text.length(), text.data(), gl::DYNAMIC_DRAW);

	gl::DrawArrays(gl::POINTS,0,text.length());

}

/// immediately render a string at the given pixel coordinates
void Renderer::Print(float x, float y, const std::string &text)
{
	if (text.size() == 0)
		return;

	gl::Disable(gl::DEPTH_TEST);
	gl::Disable(gl::CULL_FACE);
	gl::PolygonMode(gl::FRONT_AND_BACK,gl::FILL);
	gl::Enable (gl::BLEND); 
	gl::BlendFunc (gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

	gl::UseProgram(m_textProgram);

	GLuint textureUnit = gl::TEXTURE1;
	gl::ActiveTexture(textureUnit);	
	gl::BindTexture(gl::TEXTURE_2D,m_font);

	SetUniform("Font",(int)textureUnit - (int)gl::TEXTURE0);
	SetUniform("ViewportSize",m_viewPortSizeX,m_viewPortSizeY);
	SetUniform("FontSize",40);
	SetUniform("StartPosition",x,y);

	gl::BindBuffer(gl::ARRAY_BUFFER,m_textBuffer);

	gl::BufferData(gl::ARRAY_BUFFER, sizeof(unsigned char) * text.length(), text.data(), gl::DYNAMIC_DRAW);

	GLint positionLoc = gl::GetAttribLocation(m_textProgram,"Character");

	if (positionLoc >= 0)
	{
		gl::EnableVertexAttribArray(positionLoc);

		gl::VertexAttribIPointer(positionLoc,1,gl::UNSIGNED_BYTE,1,0);

		gl::DrawArrays(gl::POINTS,0,text.length());

		gl::DisableVertexAttribArray(positionLoc);
	}


	gl::BindBuffer(gl::ARRAY_BUFFER,0);
	gl::BindTexture(gl::TEXTURE_2D,0);

	check_gl_error();
}

void Renderer::PrePrint()
{
	gl::Disable(gl::DEPTH_TEST);
	gl::Disable(gl::CULL_FACE);
	gl::PolygonMode(gl::FRONT_AND_BACK,gl::FILL);
	gl::Enable (gl::BLEND); 
	gl::BlendFunc (gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

	gl::UseProgram(m_textProgram);

	GLuint textureUnit = gl::TEXTURE1;
	gl::ActiveTexture(textureUnit);	
	gl::BindTexture(gl::TEXTURE_2D,m_font);

	SetUniform("Font",(int)textureUnit - (int)gl::TEXTURE0);
	SetUniform("ViewportSize",m_viewPortSizeX,m_viewPortSizeY);
	SetUniform("FontSize",40);

	gl::BindBuffer(gl::ARRAY_BUFFER,m_textBuffer);

	GLint positionLoc = gl::GetAttribLocation(m_textProgram,"Character");

	gl::EnableVertexAttribArray(positionLoc);
	gl::VertexAttribIPointer(positionLoc,1,gl::UNSIGNED_BYTE,1,0);

	check_gl_error();
}

void Renderer::PostPrint()
{
	GLint positionLoc = gl::GetAttribLocation(m_textProgram,"Character");

	if (positionLoc >= 0)
	{
		gl::DisableVertexAttribArray(positionLoc);

		gl::BindBuffer(gl::ARRAY_BUFFER,0);
		gl::BindTexture(gl::TEXTURE_2D,0);
	}

	gl::UseProgram(0);

}

void Renderer::Render(float elapsedSeconds)
{
	if (m_strings.size() == 0)
		return;

	PrePrint();

	const float fontSize = 40;

	float x = 0;
	float y = 0;

	GLint positionLoc = gl::GetAttribLocation(m_textProgram, "Character");

	for(auto it = begin(m_strings); it != end(m_strings); ++it)
	{
		auto text = *it;

		DoPrint(positionLoc,x,-1.f * y, *it);

		y ++;//= fontSize;
	}

	PostPrint();

	check_gl_error();

	m_strings.clear();
}



void Renderer::DrawGraph(
	const string &title
	,int x
	,int y
	,unsigned int width
	,unsigned int height
	,const vec4 &foregroundColor
	,const vec4 &backgroundColor
	,vector<float> &data
	,unsigned int first
	,unsigned int sampleCount
	,float normalizeMin
	,float normalizeMax
	)
{
	check_gl_error();
	unsigned int padding = 2;

	DrawQuad(x,y,width,height,backgroundColor);

	/// build the vertex buffer
	static vector<vec2> bufferData;
	static vector<float> rmsData;

	rmsData.clear();

	bufferData.clear();

	sampleCount = min(sampleCount, data.size() - first);

	if (sampleCount < 2)
		return;

	unsigned int windowSize = 100;

	bufferData.reserve(sampleCount * windowSize);

	for(unsigned int i = 0; i < sampleCount; i++)
	{
		unsigned int index = first + i;
		float val = data[index];

		bufferData.push_back(vec2(i,val));
	}


	gl::UseProgram(m_graphProgram);

	SetUniform("Size",width - 2 * padding,height - 2 * padding);
	SetUniform("Location",x + padding,y + padding);
	SetUniform("SampleCount",sampleCount);
	SetUniform("MinMax",normalizeMax, normalizeMin);
	SetUniform("ViewportSize",m_viewPortSizeX,m_viewPortSizeY);
	SetUniform("Color",foregroundColor);


	gl::BindBuffer(gl::ARRAY_BUFFER, m_graphDataBuffer);

	GLint positionLoc = gl::GetAttribLocation(m_graphProgram, "Position");

	gl::EnableVertexAttribArray(positionLoc);
	gl::VertexAttribPointer(positionLoc,2,gl::FLOAT,false,sizeof(vec2),0);
	gl::BufferData(gl::ARRAY_BUFFER, sizeof(vec2) * bufferData.size(), bufferData.data(), gl::DYNAMIC_DRAW);

	gl::DrawArrays(gl::LINE_STRIP, 0, bufferData.size());

	gl::BindBuffer(gl::ARRAY_BUFFER, 0);

	check_gl_error();


}


void Renderer::DrawQuad(int x, int y, int width, int height, const vec4 &color)
{
	gl::UseProgram(m_screenPrimitiveProgram);

	GLint positionLoc = gl::GetAttribLocation(m_screenPrimitiveProgram, "Position");

	if (positionLoc >= 0)
	{
		SetUniform("Location",x,y);
		SetUniform("Size",width,height);
		SetUniform("ViewportSize",m_viewPortSizeX,m_viewPortSizeY);
		SetUniform("Color",color);

		gl::BindBuffer(gl::ARRAY_BUFFER, m_quadVertexBuffer);
		gl::EnableVertexAttribArray(positionLoc);
		gl::VertexAttribPointer(positionLoc,2,gl::FLOAT,false,sizeof(vec2),0);

		gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);

		gl::DisableVertexAttribArray(positionLoc);
	}

	gl::UseProgram(0);
}


void Renderer::Shutdown()
{
	if (m_font)
	{
		gl::DeleteTextures(1,&m_font);
		check_gl_error();

		m_font = 0;
	}

	if (m_textProgram)
	{
		gl::DeleteProgram(m_textProgram);
		check_gl_error();
		m_textProgram = 0;
	}

	if (m_textBuffer)
	{
		gl::DeleteBuffers(1,&m_textBuffer);
		check_gl_error();
		m_textBuffer = 0;
	}

	if (m_screenPrimitiveProgram)
	{
		gl::DeleteProgram(m_screenPrimitiveProgram);
		check_gl_error();
		m_screenPrimitiveProgram = 0;
	}

	if (m_quadVertexBuffer)
	{
		gl::DeleteBuffers(1, &m_quadVertexBuffer);
		check_gl_error();
		m_quadVertexBuffer = 0;
	}

}

