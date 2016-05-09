#include "Raycaster.h"

#include <stdexcept>
#include <iostream>
#include <vector>

using namespace std;


Raycaster::Raycaster(void)
{
	m_texture = 0;

	m_textureTarget=gl::TEXTURE_2D;

	m_width = -1;
	m_height = -1;
}


Raycaster::~Raycaster(void)
{
	if (m_texture > 0)
	{
		gl::DeleteTextures(1, &m_texture);
		m_texture = 0;
	}

}

void Raycaster::InitCuda()
{
	m_cuda.InitDevice();
	m_cuda.InitNoise();
}

void Raycaster::Create(int width, int height)
{
	for(int i = 0; i < 2; i++)
	{
		Create(m_texture,width,height);
		m_cuda.CreateResource(m_texture,m_textureTarget);
	}

}
/// <summary>
/// Creates an RGBA32f texture of the given dimensions
/// </summary>
/// <param name="width">The width.</param>
/// <param name="height">The height.</param>
void Raycaster::Create(GLuint &handle, int width, int height)
{
	m_width = width;
	m_height = height;

	if (handle != 0)
	{
		gl::DeleteTextures(1, &handle);
		handle = 0;
	}

	check_gl_error();

	gl::GenTextures(1, &handle);

	gl::BindTexture(m_textureTarget,handle);

	check_gl_error();

	gl::TexImage2D(m_textureTarget,0,gl::RGBA32F,m_width,m_height,0,gl::RGBA,gl::FLOAT,0);

	check_gl_error();

	gl::TexParameterf(m_textureTarget,gl::TEXTURE_MIN_FILTER,gl::NEAREST);

	check_gl_error();

	gl::TexParameterf(m_textureTarget,gl::TEXTURE_MAG_FILTER,gl::NEAREST);

	check_gl_error();

	gl::BindTexture(m_textureTarget,0);

	check_gl_error();



}

/// <summary>
/// Call CUDA to update the texture contents
/// </summary>
/// <param name="elapsedSeconds">The elapsed seconds.</param>
void Raycaster::Update(float elapsedSeconds)
{
	m_cuda.Update(elapsedSeconds,m_width,m_height);
}

/// <summary>
/// Bind the completed texture for use by OpenGL
/// </summary>
void Raycaster::Bind(GLint uniformLocation, GLenum textureUnit )
{
	m_textureUnit = textureUnit;

	gl::ActiveTexture(textureUnit);
	gl::BindTexture(m_textureTarget,m_texture);

	gl::Uniform1i(uniformLocation, (int)textureUnit - (int)gl::TEXTURE0);

	check_gl_error();
}

/// <summary>
/// Unbind the texture
/// </summary>
void Raycaster::Unbind()
{
	gl::BindTexture(m_textureTarget, 0);
}

