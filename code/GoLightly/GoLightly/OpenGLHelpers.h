#ifndef OPENGLHELPERS_H
#define OPENGLHELPERS_H

#include <map>
#include <string>

#include "Common.h"

using namespace std;

class OpenGLHelpers
{
public:
	static string GetOpenGLInfoLog(GLuint handle, bool isProgram);
	static GLuint CreateShader(GLenum type, const string &source);
	static GLuint CreateProgram(const string &baseFilename, char **requiredAttribs = nullptr, unsigned int requiredAttributeCount = 0);

private:
	/// map of gl shader type to filename extension
	static map<GLenum,string> m_shaderFileExtensions;



};




#endif