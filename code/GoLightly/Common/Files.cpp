#define _CRT_SECURE_NO_WARNINGS
#include "Files.h"

#include <iostream>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <ctime>

using namespace std;

/// <summary>
/// Reads the specified file.
/// </summary>
/// <param name="path">path to the file</param>
/// <returns></returns>
string Files::Read(const string& path)
{
	std::ifstream t(path);
	if(!t.is_open())
	{
		cerr << "Could not load file \"" << path << "\". Exiting.";
		exit(EXIT_FAILURE);
	}

	std::string str;

	t.seekg(0, std::ios::end);   
	str.reserve(t.tellg());
	t.seekg(0, std::ios::beg);

	str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

	return std::move(str);
}


/// <summary>
/// Determines if the file exists
/// </summary>
/// <param name="path">path to the file</param>
/// <returns></returns>
bool Files::Exists(const string& path)
{
	ifstream ifile(path.c_str());
	return ifile.is_open();
}



string Files::GenerateFilename(const string &prefix, const string &suffix )
{
   time_t now;

   now = time(NULL);

   if (now == -1)
	   throw runtime_error("time() did not return a valid value");


   /*
   %Y	yyyy four-digit year
   %m	mm	 two-digit month
   %d   dd   two-digit day-of-month
   %H	HH	 two-digit 24 hour 
   %M	mm	 two-digit minute
   %S	ss	 two-digit second
   */

   /// max length based on the selected format
   const unsigned int maxLength = 128;
   char dateStr[maxLength + 1];

   strftime(dateStr,maxLength,"%Y%m%d-%H%M%S",localtime(&now));

   return prefix + string(dateStr) + suffix;
}

