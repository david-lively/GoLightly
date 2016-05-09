#include <iostream>
#include <string>

using namespace std;

#include "Script.h"

/*
Generates bitmap model definitions for the GoLightly simulator from
Wavefront OBJ files and metadata.
*/
int main(int argc, char *argv[])
{
	cout << "GoLightly Model Processor\n";
	cout << "(c) 2014 David Lively\n";

	if (argc < 2)
	{
		cout << "Usage: modelProcessor.exe <model metadata file>\n";
		exit(EXIT_FAILURE);
	}

	string filename = argv[1];

	Script script;

	script.Build(filename);

	cout << "main() - Done\n";
	getchar();

}