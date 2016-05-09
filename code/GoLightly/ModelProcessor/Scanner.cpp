#include <cassert>
#include <regex>
#include <iostream>

using namespace std;
using namespace std::tr1;

#include "Common.h"
#include "Script.h"

void Scanner::Initialize()
{
	InitializeTokenMap();
}

/*
Generate a token stream from the input text
*/
void Scanner::Scan(string &source, vector<Token> &tokens)
{
	TokenType current = TokenType::Unknown;

	m_source = source;

	cout << "Scanning source...\n";

	smatch matches;

	string::const_iterator currentPosition = m_source.begin();
	string::const_iterator endOfSource = m_source.end();

	while (currentPosition < endOfSource)
	{

		bool matchFound = false;
		string matchText = "";
		TokenType matchType = TokenType::Unknown;

		for(auto pit = begin(m_tokenPatterns); pit != end(m_tokenPatterns); ++pit)
		{
			if(regex_search(
				currentPosition
				,endOfSource
				,matches
				,pit->second
				,std::regex_constants::match_continuous
				))
			{
				if (matches[0].str().length() > matchText.length())
				{
					matchText = matches[0].str();

					if (matchText.find("\n") != string::npos)
					{
						matchType = TokenType::EndOfStatement;						
					}
					else
						matchType = pit->first;
				}

				matchFound = true;
			}
		}

		if (!matchFound)
		{
			Abort("No suitable token found");
			return;
		}

		if (matchType != TokenType::WhiteSpace)
		{
			//cout << matchType << " \t\t'" << matchText << "'\n";
			tokens.push_back(Token(matchType,matchText));
		}

		currentPosition += matchText.length();

	}

	tokens.push_back(Token(TokenType::Eof,"Eof"));

	cout << "Scan complete. Found " << tokens.size() << " tokens.\n";
}

void Scanner::InitializeTokenMap()
{
	map<TokenType,string> p;

	p[TokenType::Comment	   ] = "(/\\*([^*]|[\\r\\n]|(\\*+([^*/]|[\\r\\n])))*\\*+/)|(//.*)";
	p[TokenType::Float		   ] = "[0-9]+\\.[0-9^(A-Za-z)]*";
	p[TokenType::Integer	   ] = "[0-9]+";
	p[TokenType::String		   ] = "\\\"([^\\\"\\\\\\\\]|\\\\\\\\.)*\\\"";
	p[TokenType::Identifier	   ] = "[a-zA-Z_][a-zA-Z0-9_]*";
	p[TokenType::Operator	   ] = "\\^|\\*|\\/|\\+|\\-|\\=";
	p[TokenType::BinaryOperator] = "(\\=\\=)|(\\+\\=)|(\\-\\=)|(\\*\\=)|(\\/\\=)";
	p[TokenType::WhiteSpace    ] = "\\s+";

	p[TokenType::EndOfStatement] = ";";
	p[TokenType::ListStart     ] = "\\(";
	p[TokenType::ListDelimiter ] = "\\,";
	p[TokenType::ListEnd	   ] = "\\)";
	p[TokenType::BlockStart    ] = "\\{";
	p[TokenType::BlockEnd	   ] = "\\}";

	m_tokenPatterns.clear();

	for(auto it = begin(p); it != end(p); ++it)
	{
		m_tokenPatterns[it->first] = regex("^" + it->second);
	}

}
