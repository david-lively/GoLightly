#ifndef AST_H
#define AST_H

#include <vector>
#include <string>
#include <ostream>

using namespace std;

#include "Common.h"
#include "Token.h"

//#include "Visitor.h"
/// external struct
struct Visitor;

#define ACCEPT void Accept(Visitor& visitor);


enum class NodeType
{
	Unknown
	,Operation
	,Block
	,Integer
	,Float
	,String
	,Identifier
	,BinaryOperator
	,Call
	,Declaration
	,Expression
};



template<typename T>
inline void Indent(unsigned int level, const T& value)
{
	cout << string(level,'\t') << value << endl;
}

struct Node
{
	NodeType Type;

	Node()
	{
		Type=NodeType::Unknown;
	}

	/// virtual dtor just to enable polymorphism. Yay, C++.
	virtual ~Node() = 0
	{
	}


	virtual void Accept(Visitor& visitor);

};

struct Expression : public Node 
{
	Expression()
	{
		Type=NodeType::Expression;
	}
};

struct Constant : public Expression {};

template<typename T>
struct TypedConstant : public Constant
{
	T Value;

	TypedConstant()
	{

	}

	TypedConstant(const Token& token)
	{

	}

};

struct Integer : public TypedConstant<int> 
{
	Integer(const Token& token)
	{
		Value = atoi(token.Text.c_str());
		Type=NodeType::Integer;
	}

	ACCEPT;
};

struct Float : public TypedConstant<float> 
{
	Float(const Token& token)
	{
		Value = static_cast<float>(atof(token.Text.c_str()));
		Type=NodeType::Float;
	}

	ACCEPT;


};

struct String : public TypedConstant<string> 
{
	String(const Token& token) : TypedConstant(token)
	{
		Value = token.Text;
		Type=NodeType::String;
	}

	ACCEPT;

};

struct Identifier : public TypedConstant<string> 
{
	Identifier(const Token& token) : TypedConstant(token)
	{
		Value = token.Text;
		Type=NodeType::Identifier;
	}

	ACCEPT;

};

struct ListStart : public Node {};
struct ListEnd : public Node{};
struct BlockStart : public Node {};
struct BlockEnd : public Node {};

struct BinaryOperator : public Expression
{
	Node* Left;
	Operation Op;
	Node* Right;

	BinaryOperator() :
		Left(nullptr)
		,Right(nullptr)
		,Op(Operation::Unknown)
	{
		Type=NodeType::BinaryOperator;
	}

	~BinaryOperator()
	{
		if(Left)
			delete Left;
		if(Right)
			delete Right;
	}

	ACCEPT;

};

struct Call : public Expression
{
	Identifier* Callee;
	vector<Expression*> Parameters;

	Call() :
		Callee(nullptr)
	{
		Type=NodeType::Call;
	}

	~Call()
	{
		for(auto param : Parameters)
		{
			if(param)
				delete param;
		}

		Parameters.clear();
	}

	ACCEPT;


};

struct Block : public Node
{
	vector<Node*> Statements;

	Block()
	{
	}

	~Block()
	{
		for(auto statement : Statements)
		{
			if(statement)
				delete statement;
		}

		Statements.clear();
	}


	ACCEPT;



};

struct Function : public Call
{
	Block* Body;

	Function() : Call()
	{
		Type=NodeType::Declaration;
	}

	~Function()
	{
		if(Body)
			delete Body;
	}

	ACCEPT;

};

struct Program : public Block
{
	ACCEPT;
};


#endif