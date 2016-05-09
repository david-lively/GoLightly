#ifndef Scanner_H
#define Scanner_H

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <regex>
#include <iostream>

using namespace std;

#include "Common.h"
#include "Token.h"

/// converts text into a token stream
class Scanner
{
public:
	Scanner() : m_source("")
	{
		Initialize();
	}

	void Scan(string &source, vector<Token> &tokens);

private:
	string m_source;


	/// regex for each token type
	map<TokenType,regex> m_tokenPatterns;

	void Initialize();
	void InitializeTokenMap();


};

#endif