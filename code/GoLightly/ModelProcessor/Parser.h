#ifndef PARSER_H
#define PARSER_H

#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <functional>
#include <map>


using namespace std;

#include "Common.h"
#include "Token.h"
#include "AST.h"

class Parser
{
public:
	Parser() 
	{
		cout << "Parser::Parser()\n";
	}

	Program* Parse(vector<Token>& tokens);

private:

	void Reset();

	/// shunting-yard phase
	void ConvertToRPN(vector<Token>& tokens);

	/// build an abstract syntax tree from an RPN-order token list
	Program* BuildSyntaxTree(vector<Token>& tokens);
};

template<typename T>
void clear(stack<T>& s)
{
	while(s.size() > 0)
		s.pop();
}

#endif