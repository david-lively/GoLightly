#ifndef VECTORS_H
#define VECTORS_H

struct vec2
{
	float x,y;

	vec2() : x(0), y(0)
	{
	}

	vec2(float x, float y)
	{
		this->x = x;
		this->y = y;
	}


};

struct vec3 : vec2
{
	float z;

	vec3() : z(0)
	{
	}

	vec3(float x, float y, float z) : vec2(x,y)
	{
		this->z = z;
	}
};

struct vec4 : vec3
{
	float w;

	vec4() : w(1)
	{
	}

	vec4(float x, float y, float z, float w) : vec3(x,y,z)
	{
		this->w = w;
	}


};

#endif