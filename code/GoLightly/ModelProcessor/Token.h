#ifndef TOKEN_H
#define TOKEN_H

#include <string>

using namespace std;

#include "Common.h"

/// Token TokenType returned from the scanner
enum struct TokenType
{
	Unknown
	,Comment	   
	,Error
	,Float
	,Integer
	,String	
	,Identifier
	,Operator
	,BinaryOperator
	,WhiteSpace   
	,Eof
	,ListStart
	,ListDelimiter
	,ListEnd
	,Call
	,Function
	/// wildcard match for states
	,Else		
	,BlockStart
	,BlockEnd
	,EndOfStatement

};


ostream &operator<<(ostream& os, const TokenType& tokenType);


struct Token
{
	TokenType Type;

	unsigned int Line;
	unsigned int Column;
	unsigned int Length;

	string Text;

	Token(TokenType type = TokenType::Unknown, const string &text = "", unsigned int line = 0, unsigned int column = 0) : 
		Line(line)
		,Column(column)
		,Length(0)
		,Type(type)
		,Text(text)
	{
	}
};

#endif