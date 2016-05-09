#include "Token.h"

using namespace std;

map<TokenType,string> tokenNames;

#define ADD(tt) { result[TokenType::tt] = #tt; }

map<TokenType,string> CreateTokenNameMap()
{
	map<TokenType,string> result;

	ADD(Unknown           );
	ADD(Error			  );
	ADD(Float			  );
	ADD(Integer			  );
	ADD(String			  );
	ADD(Identifier		  );
	ADD(BinaryOperator	  );
	ADD(Operator		  );
	ADD(Comment	   		  );
	ADD(WhiteSpace   	  );
	ADD(EndOfStatement	  );
	ADD(Eof				  );
	ADD(ListStart         );
	ADD(ListDelimiter	  );
	ADD(ListEnd			  );
	ADD(Call              );
	ADD(BlockStart        );
	ADD(BlockEnd          );

	return result;
}



ostream &operator<<(ostream& os, const TokenType& tokenType)
{
	static map<TokenType,string> names = CreateTokenNameMap();

	os << names[tokenType];

	return os;
}

