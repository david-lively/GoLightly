#include <iostream>
#include <string>
#include <vector>

#include "OpenGLHelpers.h"
#include "../Common/Files.h"

using namespace std;



/// <summary>
/// Gets the OpenGL shader or program information log.
/// </summary>
/// <param name="shader">GL handle to the shader or program</param>
/// <returns></returns>
string OpenGLHelpers::GetOpenGLInfoLog(GLuint handle, bool isProgram)
{
	string log = "";
	GLint logLength;
	GLsizei returnedLength;


	if (isProgram)
	{
		GLint status;

		gl::GetProgramiv(handle, gl::LINK_STATUS, &status);

		if (status != GL_FALSE)
			return "";

		gl::GetProgramiv(handle,gl::INFO_LOG_LENGTH, &logLength);
		char *buffer = new char[logLength];

		gl::GetProgramInfoLog(handle,logLength,&returnedLength,buffer);
		log = string(buffer);

		delete[] buffer;

	}
	else
	{
		GLint status;
		gl::GetShaderiv(handle, gl::COMPILE_STATUS, &status);

		if (status != GL_FALSE)
			return "";

		gl::GetShaderiv(handle,gl::INFO_LOG_LENGTH, &logLength);

		char *buffer = new char[logLength];

		gl::GetShaderInfoLog(handle,logLength,&returnedLength,buffer);
		log = string(buffer);

		delete[] buffer;

	}

	return log;
}

GLuint OpenGLHelpers::CreateShader(GLenum type, const string &source)
{
	const char *c_str = source.c_str();

	GLuint shader = gl::CreateShader(type);

	gl::ShaderSource(shader,1,&c_str,nullptr);
	gl::CompileShader(shader);

	string log = GetOpenGLInfoLog(shader,false);
	if (log != "")
	{
		cerr << "Shader compilation errors" << endl;
		cerr << log;
		exit(EXIT_FAILURE);
	}

	return shader;
}

map<GLenum,string> InitializeShaderNameMap()
{
	map<GLenum,string> names;

	names[gl::VERTEX_SHADER]			= ".vert.glsl";
	names[gl::FRAGMENT_SHADER]			= ".frag.glsl";
	names[gl::GEOMETRY_SHADER]			= ".geom.glsl";
	names[gl::TESS_CONTROL_SHADER]		= ".tessc.glsl";
	names[gl::TESS_EVALUATION_SHADER]	= ".tesse.glsl";

	return names;
}

map<GLenum,string> OpenGLHelpers::m_shaderFileExtensions = InitializeShaderNameMap();


GLuint OpenGLHelpers::CreateProgram(const string &baseFilename, char **requiredUniforms, unsigned int requiredUniformCount)
{
	vector<GLuint> shaders;
	shaders.reserve(10);

	GLuint program = gl::CreateProgram();

	cout << "Compiling shaders for \"" << baseFilename << "\"\n";

	for(auto it = begin(m_shaderFileExtensions); it != end(m_shaderFileExtensions); ++it)
	{
		string filename = "resources\\" + baseFilename + it->second;

		/// see if we have a source file with this 
		if (!Files::Exists(filename))
			continue;

		string source = Files::Read(filename);

		cout << "Compiling shader \"" << filename << "\"\n";

		GLuint shader = CreateShader(it->first, source);

		shaders.push_back(shader);

		gl::AttachShader(program,shader);
	}

	if (shaders.size() == 0)
	{
		cerr << "No shader source files were found for filename \"" << baseFilename << "\".";
		exit(EXIT_FAILURE);
	}

	gl::LinkProgram(program);

	check_gl_error();

	string log = GetOpenGLInfoLog(program, true);
	if (log != "")
	{
		cerr << "Program errors for \"" << baseFilename  << "\"\n";
		cerr << log << endl;
		exit(EXIT_FAILURE);
	}

	for(auto it = begin(shaders); it != end(shaders); ++it)
	{
		gl::DeleteShader(*it);
	}

	check_gl_error();

	return program;
}

