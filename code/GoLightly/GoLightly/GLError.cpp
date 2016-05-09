/*
* see http://blog.nobel-joergensen.com/2013/01/29/debugging-opengl-using-glgeterror/
*/
#include "GLError.h"
#include <iostream>
#include <string>
#include <GL/gl_core_4_2.hpp>

//#ifdef WIN32
//#  include <GL/glew.h>
//#elif __APPLE__
//#  include <OpenGL/gl3.h>
//#else
//#  include <GL3/gl3.h>
//#endif

#include <GL/gl_core_4_2.hpp>

using namespace std;


void _check_gl_error(const char *file, int line) {
	GLenum err (gl::GetError());

	while(err!=gl::NO_ERROR_) {
                string error;

                switch(err) {
                        case gl::INVALID_OPERATION:      error="INVALID_OPERATION";      break;
                        case gl::INVALID_ENUM:           error="INVALID_ENUM";           break;
                        case gl::INVALID_VALUE:          error="INVALID_VALUE";          break;
                        case gl::OUT_OF_MEMORY:          error="OUT_OF_MEMORY";          break;
                        case gl::INVALID_FRAMEBUFFER_OPERATION:  error="INVALID_FRAMEBUFFER_OPERATION";  break;
                }

                cerr << "gl::" << error.c_str() <<" - "<<file<<":"<<line<<endl;
                err=gl::GetError();
        }
}
