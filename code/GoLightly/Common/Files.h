#ifndef FILES_H
#define FILES_H

#include <string>

/// <summary>
/// provides file IO convenience routines
/// </summary>
class Files
{
private:
	Files() { }
	virtual ~Files() = 0;

public:
	static std::string Read(const std::string& path);
	static bool Exists(const std::string& path);
	static std::string GenerateFilename(const std::string &prefix, const std::string &suffix);
};


#endif