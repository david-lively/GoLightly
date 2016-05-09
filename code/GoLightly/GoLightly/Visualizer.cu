#include "Visualizer.cuh"

#include <iostream>
#include <array>

using namespace std;

#include "CudaHelpers.h"
#include "OpenGLHelpers.h"
#include "GLError.h"
#include "../Common/Files.h"
#include "../Thirdparty/soil/SOIL2.h"


Visualizer::Visualizer(void)
{
	m_resource = nullptr;
	m_texture = 0;

	m_textureTarget = gl::TEXTURE_2D;

	m_width = 0;
	m_height = 0;

	ColorScale = 10.f;

	Enabled = true;

	m_textureSize.fill(1.f);
	m_windowSize.fill(1.f);
}


Visualizer::~Visualizer(void)
{
}


__global__ void visualizerUpdatePreviewTexture(
	cudaSurfaceObject_t image
	, int imageWidth
	, int imageHeight
	, float *fieldData
	, int fieldWidth
	, int fieldHeight
	, float *materials
	)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int readX = (int)(x * fieldWidth * 1.f / imageWidth);
	int readY = (int)(y * fieldHeight * 1.f / imageHeight);

	float fieldValue = fieldData[readY * fieldWidth + readX];
	float cb = materials[readY * fieldWidth + readX];

	float4 color = make_float4(fieldValue, cb, 0, 1);

	if (threadIdx.x == 0 || threadIdx.y == 0)
		color.w = 1;
	else
		color.w = 0;

	//color.x = x * 1.f / imageWidth;
	//color.y = y * 1.f / imageHeight;
	//color.z = 0.2f;
	//color.w = 1;

	surf2Dwrite(color, image, x * sizeof(float4), y, cudaBoundaryModeClamp);
}



/// <summary>
/// Generate a preview texture from the FDTD field contents
/// </summary>
/// <param name="elapsedSeconds">The elapsed seconds.</param>
/// <param name="field">The field.</param>
/// <param name="width">The width.</param>
/// <param name="height">The height.</param>
void Visualizer::Update(float elapsedSeconds, float *field, int width, int height, float *materials)
{
	static bool firstRun = true;

	if (!Enabled || !m_resource)
		return;

	Check(cudaGraphicsMapResources(1, &m_resource));

	cudaArray_t textureArray;
	Check(cudaGraphicsSubResourceGetMappedArray(&textureArray, m_resource, 0, 0));

	cudaResourceDesc desc;

	memset(&desc, 0, sizeof(desc));

	desc.resType = cudaResourceTypeArray;
	desc.res.array.array = textureArray;

	cudaSurfaceObject_t surface;

	Check(cudaCreateSurfaceObject(&surface, &desc));

	dim3 block(32, 32, 1);
	dim3 grid(m_width / block.x, m_height / block.y);


	visualizerUpdatePreviewTexture<<<grid, block, 0, m_cudaStream >>>(surface, m_width, m_height, field, width, height, materials);

	Check(cudaDeviceSynchronize());

	Check(cudaDestroySurfaceObject(surface));

	Check(cudaGraphicsUnmapResources(1, &m_resource));

	firstRun = false;
}

void Visualizer::Render(float elapsedSeconds)
{
	if (!Enabled)
		return;

	gl::Disable(gl::DEPTH_TEST);
	gl::Disable(gl::CULL_FACE);
	gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);

	/// render full-screen quad
	gl::UseProgram(m_program);

	auto textureSizeLocation = gl::GetUniformLocation(m_program, "TextureSize");
	if (textureSizeLocation >= 0)
	{
		gl::Uniform2fv(textureSizeLocation, 1, static_cast<const GLfloat*>(m_textureSize.data()));
	}

	auto windowSizeLocation = gl::GetUniformLocation(m_program, "WindowSize");
	if (windowSizeLocation >= 0)
	{
		gl::Uniform2fv(windowSizeLocation, 1, static_cast<const GLfloat*>(m_windowSize.data()));
	}

	GLint colorScaleLocation = gl::GetUniformLocation(m_program, "ColorScale");

	if (colorScaleLocation >= 0)
		gl::Uniform1f(colorScaleLocation, ColorScale);

	GLint textureLocation = gl::GetUniformLocation(m_program, "CudaTexture");

	if (textureLocation >= 0)
		Bind(textureLocation, gl::TEXTURE0);

	check_gl_error();

	gl::BindBuffer(gl::ARRAY_BUFFER, m_vertexBuffer);
	gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, m_indexBuffer);

	GLint positionLocation = gl::GetAttribLocation(m_program, "Position");

	if (positionLocation >= 0)
	{

		gl::EnableVertexAttribArray(positionLocation);
		gl::VertexAttribPointer(positionLocation, 3, gl::FLOAT, false, 3 * sizeof(GLfloat), (const GLvoid*)0);
		gl::DrawElements(gl::TRIANGLES, m_indexCount, gl::UNSIGNED_SHORT, 0);

		gl::BindBuffer(gl::ARRAY_BUFFER, 0);
		gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0);
	}
	if (textureLocation >= 0)
		Unbind();

	check_gl_error();
}



void Visualizer::Bind(GLuint uniformLocation, GLenum textureUnit)
{
	if (!Enabled)
		return;

	m_textureUnit = textureUnit;

	gl::ActiveTexture(textureUnit);
	gl::BindTexture(m_textureTarget, m_texture);

	gl::Uniform1i(uniformLocation, (int)textureUnit - (int)gl::TEXTURE0);

	check_gl_error();
}

void Visualizer::Unbind()
{
	if (!Enabled)
		return;

	gl::BindTexture(m_textureTarget, 0);
}

void Visualizer::Resize(unsigned int domainWidth, unsigned int domainHeight, unsigned int windowWidth, unsigned int windowHeight, unsigned int bottomMargin)
{
	if (!Enabled)
		return;

	if (windowWidth == 0 || windowHeight == 0)
		return;

	float aspect = domainWidth * 1.f / domainHeight;
	unsigned int textureWidth = 0, textureHeight = 0;

	/// fit texture to viewport, as tightly as possible
	if (domainWidth > domainHeight)
	{
		textureWidth = min(domainWidth, windowWidth);
		textureHeight = textureWidth * domainHeight / domainWidth;
	}
	else
	{
		textureHeight = min(domainHeight, windowHeight);
		textureWidth = textureHeight * domainWidth / domainHeight;
	}

	m_textureSize[0] = textureWidth;
	m_textureSize[1] = textureHeight;
	m_windowSize[0] = windowWidth;
	m_windowSize[1] = windowHeight;

	if (textureWidth % 32 != 0)
	{
		textureWidth = textureWidth / 32 * 32;
	}

	if (textureHeight % 32 != 0)
	{
		textureHeight = textureHeight / 32 * 32;
	}


	m_width = textureWidth;
	m_height = textureHeight;

	check_gl_error();

	if (m_texture != 0)
	{
		gl::DeleteTextures(1, &m_texture);
		m_texture = 0;
	}

	gl::GenTextures(1, &m_texture);
	gl::BindTexture(m_textureTarget, m_texture);

	gl::TexImage2D(m_textureTarget, 0, gl::RGBA32F, m_width, m_height, 0, gl::RGBA, gl::FLOAT, 0);
	gl::TexParameterf(m_textureTarget, gl::TEXTURE_MIN_FILTER, gl::NEAREST);
	gl::TexParameterf(m_textureTarget, gl::TEXTURE_MAG_FILTER, gl::NEAREST);

	gl::BindTexture(m_textureTarget, 0);

	RegisterCudaResources();
}

/// <summary>
/// Register the texture with CUDA if necessary
/// </summary>
void Visualizer::RegisterCudaResources()
{
	if (!Enabled)
		return;

	/// create graphics resource
	if (m_resource)
		cudaGraphicsUnregisterResource(m_resource);

	Check(
		cudaGraphicsGLRegisterImage(
		&m_resource
		, m_texture
		, m_textureTarget
		, cudaGraphicsRegisterFlagsSurfaceLoadStore
		)
		);
}

void Visualizer::Ready()
{
	if (!Enabled)
		return;

	gl::ClearColor(0, 0, 0, 1);

	CreateGeometry();
	CreateEffects();

	Check(cudaStreamCreate(&m_cudaStream));
}


void Visualizer::CreateEffects()
{
	if (!Enabled)
		return;

	m_program = OpenGLHelpers::CreateProgram("basic");

	if (gl::GetAttribLocation(m_program, "Position") < 0)
	{
		cerr << "Shader \"Basic\" does not accept required ""Position"" attribute" << endl;
		exit(EXIT_FAILURE);
	}
}

void Visualizer::CreateGeometry()
{
	/// generate a screen-aligned quad
	std::array<GLfloat, 3 * 4> vectors =
	{
		-1.f, 1.f, 0
		,
		1.f, 1.f, 0
		,
		1.f, -1.f, 0
		,
		-1.f, -1.f, 0
	};

	gl::GenBuffers(1, &m_vertexBuffer);
	gl::BindBuffer(gl::ARRAY_BUFFER, m_vertexBuffer);
	gl::BufferData(gl::ARRAY_BUFFER, sizeof(GLfloat)* vectors.size(), &vectors, gl::STATIC_DRAW);
	gl::BindBuffer(gl::ARRAY_BUFFER, 0);

	m_vertexCount = static_cast<unsigned int>(vectors.size());

	check_gl_error();

	std::array<GLushort, 3 * 2> indices =
	{
		0, 1, 2
		,
		2, 3, 0
	};

	gl::GenBuffers(1, &m_indexBuffer);
	gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, m_indexBuffer);
	gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, sizeof(GLushort)* indices.size(), &indices, gl::STATIC_DRAW);
	gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0);

	m_indexCount = static_cast<unsigned int>(indices.size());

	check_gl_error();

}

void Visualizer::Shutdown()
{
	if (!Enabled)
		return;

	if (m_vertexBuffer > 0)
	{
		gl::DeleteBuffers(1, &m_vertexBuffer);
		m_vertexBuffer = 0;
	}

	if (m_indexBuffer > 0)
	{
		gl::DeleteBuffers(1, &m_indexBuffer);
		m_indexBuffer = 0;
	}

	if (m_program > 0)
	{
		gl::DeleteProgram(m_program);
		m_program = 0;
	}
}


