#ifndef RAYCASTER_H
#define RAYCASTER_H

#include "Common.h"
#include "CudaRaycaster.cuh"

class Raycaster
{
public:
	Raycaster(void);
	~Raycaster(void);

	void Create(int width, int height);
	void Create(GLuint &handle, int width, int height);

	void Bind(GLint uniformLocation, GLenum textureUnit);
	void Unbind();

	void Update(float elapsedSeconds);
	/// CUDA stuff
	void InitCuda();

private:
	GLenum m_textureTarget;
	GLenum m_textureUnit;
	GLuint m_texture;

	unsigned int m_width;
	unsigned int m_height;

	CudaRaycaster m_cuda;

};

#endif
