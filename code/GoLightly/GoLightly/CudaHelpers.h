#ifndef CUDAHELPERS_CUH
#define CUDAHELPERS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" void Check(cudaError_t);

/// <summary>
/// Divide an int3, dim3 or other type that have .x, .y and .z fields by the given divisor. 
/// </summary>
/// <param name="i">The i.</param>
/// <param name="divisor">The divisor.</param>
/// <returns></returns>
template<typename T1, typename T2>
T1 operator/(const T1 &left, const T2 right)
{
	return T1(left.x / right, left.y / right, left.z / right);
}

template<typename T>
T operator-(const T &left, const int right)
{
	return T(left.x - right, left.y - right, left.z - right);
}

//template<typename T1, typename T2>
//T1 operator-(const T1 &left, const T2 &right)
//{
//	return T1(left.x - right.x, left.y - right.y, left.z - right.z);
//}

template<typename T>
T operator+(const T &left, const int right)
{
	return left - (-right);
}

template<typename T>
float length(const T& vec)
{
	return static_cast<float>(sqrt(vec.x * vec.x + vec.y * vec.y));
}

template<typename T>
T clamp(const T val, const T lower, const T upper)
{
	return min(max(val,lower),upper);
}

template<typename T1, typename T2>
T1 clamp(const T1& val, const T2 &lower, const T2& upper)
{
	T1 result;

	result.x = max(min(upper.x,val.x),lower.x);
	result.y = max(min(upper.y,val.y),lower.y);
	result.z = max(min(upper.z,val.z),lower.z);


	return result;
}


#endif