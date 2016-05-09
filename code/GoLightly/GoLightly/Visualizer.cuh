#ifndef VISUALIZER_CUH
#define VISUALIZER_CUH

#include <array>

//#include "Common.h"

#include <GLFW/glfw3.h>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class Visualizer
{
public:
	Visualizer(void);
	~Visualizer(void);

	void Bind(GLuint uniformLocation, GLenum textureUnit);
	void Unbind();

	void Update(float elapsedSeconds, float *field, int width, int height, float *materials);
	void Render(float elapsedSeconds);

	void Resize(unsigned int domainWidth, unsigned int domainHeight, unsigned int width, unsigned int height, unsigned int bottomMargin = 0);

	void Ready();
	void Shutdown();

	void CreateGeometry();
	void CreateEffects();

	float ColorScale;

	bool Enabled;

private:
	unsigned int m_vertexCount;
	GLuint m_vertexBuffer;
	
	unsigned int m_indexCount;
	GLuint m_indexBuffer;
	GLuint m_program;

	GLenum m_textureTarget;
	GLenum m_textureUnit;
	GLuint m_texture;

	unsigned int m_width;
	unsigned int m_height;
	std::array<float,16> m_worldMatrix;
	std::array<float,2> m_textureSize;
	std::array<float,2> m_windowSize;

	cudaGraphicsResource *m_resource;
	cudaStream_t m_cudaStream;

	void RegisterCudaResources();

};

#endif
