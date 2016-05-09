#include <cctype>
#include <stack>
#include <functional>
#include <algorithm>
#include <cassert>
#include <array>
#include <iostream>

#include "Script.h"
#include "..\Common\Files.h"

#include "Token.h"

#include "PrintVisitor.h"

using namespace std;

Script::Script() 
{
}

Script::~Script()
{
}


void Script::Build(const std::string &filename, int argc, char *argv[])
{
	cout << "Processing model metadata from \"" << filename << "\"\n";

	if (!Files::Exists(filename))
	{
		Abort("Source file \"" + filename + "\" not found");
	}

	m_source = Files::Read(filename);

	vector<Token> tokens;
	tokens.reserve(1024);

	m_scanner.Scan(m_source,tokens);

	auto *program = m_parser.Parse(tokens);

	//for(auto token : tokens)
	//{
	//	cout << "[TOKEN]" << token.Type << " " <<  token.Text << " ";
	//	if (token.Type == TokenType::EndOfStatement)
	//	{
	//		cout << endl;
	//	}
	//}

	//cout << endl;


	if(!program)
		Abort("Parser return null program.");

	PrintVisitor printer;

	program->Accept(printer);

}
