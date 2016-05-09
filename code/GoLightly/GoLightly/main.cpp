#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

#include "FDTDApp.h"

int main(int argc, char *argv[])
{
	cout << "GoLightly FDTD Simulator" << endl;
	cout << "(c) 2014 David Lively" << endl;

	char** p = argv;
	vector<string> params;
	
	if (argc > 0)
	{
		params = vector<string>(p,p + argc);
	}

	bool nogl = find(begin(params),end(params),"-nogl") != end(params);

	FDTDApp app;

	if(!app.Initialize(1280,720,"FDTD",nogl,argc,argv))
	{
		cerr << "Could not initialize App." << endl;
		exit(EXIT_FAILURE);
	}

	app.Run();

}