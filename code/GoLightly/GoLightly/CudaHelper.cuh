#ifndef CUDAHELPER_H
#define CUDAHELPER_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <map>
#include <locale>

using namespace std;

#include "CudaHelpers.h"

/*
wraps cuda runtime memory routines (cudaMalloc, cudaFree, etc.) 
to track and free resources
*/
class CudaHelper
{
public:
	CudaHelper()
	{
		/// arbitrarily reserve space for 20 pointers.
		m_allocated.reserve(20);
		m_totalBytes = 0;
	}

	~CudaHelper()
	{
	}

	template<typename T>
	cudaStream_t &GetStream(T &object)
	{
		if (!m_streams.count(&object))
		{
			cudaStream_t result;
			Check(cudaStreamCreate(&result));
			m_streams[&object] = result;
		}

		return m_streams[&object];
	}

	template<class T>
	std::string FormatWithCommas(T value)
	{
		std::stringstream ss;
		ss.imbue(std::locale(""));
		ss << std::fixed << value;
		return ss.str();
	}

	/*
	Allocate a device buffer to contain the given number of elements of the given type, and
	add it to the internal allocation table.
	*/
	template<typename T>
	T* Malloc(size_t numberOfElements)
	{
		size_t bytes = numberOfElements * sizeof(T);

		cout << "Allocating " << FormatWithCommas(bytes) << " bytes ";
		T* ptr = nullptr;

		Check(cudaMalloc<T>(&ptr, bytes));

		m_allocated.push_back(static_cast<void*>(ptr));
		m_blockSizes[static_cast<void*>(ptr)] = bytes;
		m_totalBytes += bytes;

		cout << "(" << FormatWithCommas(m_totalBytes) << ")\n";

		return ptr;
	}

	template<typename T>
	vector<void*>::iterator Free(T** devicePtr)
	{
		void *voidPtr = static_cast<void*>(*devicePtr);

		auto it = find(begin(m_allocated),end(m_allocated),voidPtr);
		if (it != end(m_allocated))
		{
			unsigned int bytes = m_blockSizes[voidPtr];
			cout << "Freeing " << FormatWithCommas(bytes) << " bytes ";

			Check(cudaFree(*devicePtr));

			it = m_allocated.erase(it);
			m_blockSizes.erase(voidPtr);

			m_totalBytes -= bytes;
			cout << "(" << m_totalBytes << ")\n";

			*devicePtr = nullptr;

		}
		else
			throw runtime_error("Pointer does not appear to be a valid CUDA device pointer");

		return it;
	}

	/*
	Copy COUNT elements of T from source to destination
	*/
	template<typename T>
	void Memcpy(void *destination, T *source, size_t count, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
	{
		size_t bytes = sizeof(T) * count;

		Check(cudaMemcpy(destination, source, bytes, kind));
	}


	/*
	Copy COUNT elements of T from source to destination
	*/
	template<typename T>
	void MemcpyAsync(void *destination, T *source, size_t count, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
	{
		size_t bytes = sizeof(T) * count;

		auto &stream = GetStream(*this);

		Check(cudaMemcpyAsync(destination, source, bytes, kind,stream));
	}


	template<typename T>
	void Memset(void *location, size_t count, T value)
	{
		Check(cudaMemset(location,value,sizeof(T) * count));
	}

	template<typename T>
	bool Exists(T* devicePtr)
	{
		return find(begin(m_allocated),end(m_allocated),static_cast<void*>(devicePtr)) != end(m_allocated);
	}

	void FreeAll()
	{
		auto it = begin(m_allocated);

		while(m_allocated.size() > 0)
		{
			void *ptr = *it;
			it = Free(&ptr);
		}

		//for(auto it = begin(m_allocated); it != end(m_allocated); ++it)
		//{
		//	void *ptr = *it;
		//	it = Free(&ptr);
		//}

		m_allocated.clear();
	}

	void Shutdown()
	{
		cout << "CudaHelper::Shutdown()\n";
		FreeAll();
	}

	size_t GetFreeMemory()
	{
		size_t totalMemory, freeMemory;

		Check(cudaMemGetInfo(&freeMemory,&totalMemory));

		return freeMemory;
	}

	size_t GetTotalMemory()
	{
		size_t totalMemory, freeMemory;

		Check(cudaMemGetInfo(&freeMemory, &totalMemory));

		return totalMemory;
	}

	void GetMemoryInfo(size_t &freeMemory, size_t &totalMemory)
	{
		Check(cudaMemGetInfo(&freeMemory, &totalMemory));
	}


private:
	/// total number of bytes that are in use
	unsigned long m_totalBytes;

	vector<void*> m_allocated;
	/// number of bytes allocated for each block, for reporting purposes
	map<void*,unsigned long> m_blockSizes;

	map<void*,cudaStream_t> m_streams;
};


#endif
