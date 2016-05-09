/* 
GoLightly Model Processor - Script engine   
(c) 2014 David Lively davidlively@gmail.com 
*/
#ifndef SCRIPT_H
#define SCRIPT_H

#include <string>
#include <map>
#include <stack>
#include <vector>
#include <iostream>
#include <string>
#include <regex>
#include <functional>

using namespace std;

#include "Token.h"
#include "Scanner.h"
#include "Parser.h"

class Script
{
public:
	Script();
	~Script();

	void Build(const std::string &filename, int argc = 0, char *argv[] = nullptr);

private:

	Scanner m_scanner;
	Parser m_parser;


	/// script source
	string m_source;

	vector<Token> m_tokens;

	/// finite state machine for building abstract syntax tree from token stream


	Token PeekToken();
	Token GetToken();
	bool EndOfTokens();

};

#endif