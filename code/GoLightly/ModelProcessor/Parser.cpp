#include <stack>
#include <vector>
#include <queue>
#include <deque>
#include <iostream>

using namespace std;

#include "Common.h"
#include "Parser.h"
#include "Token.h"
#include "AST.h"


Program* Parser::Parse(vector<Token>& tokens)
{
	Reset();

	ConvertToRPN(tokens);
	cout << "\n\n";

	return BuildSyntaxTree(tokens);
}

#define SHIFT { output.push(operators.top()); operators.pop(); } 
#define UNWIND_TO(tokenType) { while(operators.size() > 0 && operators.top().Type != TokenType::tokenType) SHIFT; }
#define EXPECT(tokenType) (operators.size() > 0 && operators.top().Type == TokenType::tokenType)



/// arrange token stream into RPN order via Shunting-Yard algorithm
/// see http://en.wikipedia.org/wiki/Shunting-yard_algorithm
void Parser::ConvertToRPN(vector<Token>& tokens)
{
	queue<Token> output;
	stack<Token> operators;

	for(auto current = begin(tokens); current != end(tokens); ++current)
	{
		switch(current->Type)
		{

		case TokenType::BlockStart:
			{
				/// if the top item on the output list is a call, then exchange it for a function declaration.
				if (output.back().Type == TokenType::Call)
				{
					output.back().Type = TokenType::Function;
				}
				else
					output.push(*current);

				operators.push(*current);
			}
			break;

		case TokenType::ListStart:
			operators.push(*current);
			break;

		case TokenType::Integer:
		case TokenType::Float:
		case TokenType::String:
			output.push(*current);
			break;

		case TokenType::Identifier:
			if ((current + 1) < end(tokens))
			{
				if ((current+1)->Type == TokenType::ListStart)
				{
					current->Type = TokenType::Call;
					operators.push(*current);
				}
				else
				{
					output.push(*current);
				}
			}

			break;


		case TokenType::EndOfStatement:
			/// complete statement, so unwind to output, until a block start, or all the way.
			UNWIND_TO(BlockStart);
			output.push(*current);
			break;


		case TokenType::ListDelimiter:
			UNWIND_TO(ListStart);

			if (operators.size() == 0)
			{
				Error("No opening parenthesis for comma-delimited list");
				continue;
			}

			break;

		case TokenType::Operator:
			{
				auto op = StringToOperation(current->Text);

				while(operators.size() > 0 && op < StringToOperation(operators.top().Text))
					SHIFT;

				operators.push(*current);
			}
			break;

		case TokenType::BlockEnd:
			{
				UNWIND_TO(BlockStart);

				if(!EXPECT(BlockStart))
				{
					Error("Unbalanced braces");
					continue;
				}

				operators.pop();
				output.push(*current);
			}
			break;

		case TokenType::ListEnd:
			{
				UNWIND_TO(ListStart);

				if(!EXPECT(ListStart))
				{
					Error("Unbalanced parenthesis");
					continue;
				}

				/// discard paren
				operators.pop();

				/// list preceeded by a function definition, so it's declaration or call.

				if (EXPECT(Call))
					SHIFT;
			}

			break;

		case TokenType::Eof:
			break;


		default:
			Warning("Unhandled token type");

		}

	}

	while(operators.size() > 0)
	{
		TokenType t = operators.top().Type;

		if (t == TokenType::ListStart || t == TokenType::ListEnd)
		{
			Error("Unbalanced parenthesis");
		}

		output.push(operators.top());
		operators.pop();
	}

	tokens.clear();

	/// copy output to tokens collection
	while(output.size() > 0)
	{
		tokens.push_back(output.front());
		output.pop();
	}

}


void Parser::Reset()
{
}


#define IS(item,type) (nullptr != dynamic_cast<type*>(item))

/// <summary>
/// construct an abstract syntax tree from the RPN token stream
/// </summary>
/// <param name="tokens">RPN-order token stream from which to build the syntax tree</param>
/// <returns></returns>
Program* Parser::BuildSyntaxTree(vector<Token>& tokens)
{
	stack<string> blockNames;
	stack<Node*> nodes;

	vector<Node*> statements;
	statements.reserve(1024);

	Program* program = new Program();
	nodes.push(program);

	for(auto current = begin(tokens); current != end(tokens); ++current)
	{
		statements.clear();

		switch(current->Type)
		{
		case TokenType::Identifier : 
			{
				nodes.push(new Identifier(*current));
			}
			break;

		case TokenType::Float		: nodes.push(new Float    (*current)); break;
		case TokenType::String		: nodes.push(new String	(*current)); break;
		case TokenType::Integer		: nodes.push(new Integer	(*current)); break;

		case TokenType::BlockStart	: nodes.push(new Block()); break;

		case TokenType::Function	: 
			{
				Function* f = new Function();
				f->Callee = new Identifier(*current);

				/// add any identifiers on the stack as parameters
				while(nodes.size() > 0 && IS(nodes.top(),Identifier))
				{
					f->Parameters.push_back(dynamic_cast<Identifier*>(nodes.top()));
					nodes.pop();
				}

				nodes.push(f);
			}
			break;

		case TokenType::Call:
			{
				Call *call = new Call();

				call->Callee = new Identifier(*current);

				/// add any identifiers on the stack as parameters
				while(nodes.size() > 0 && IS(nodes.top(),Expression))
				{
					call->Parameters.push_back(dynamic_cast<Expression*>(nodes.top()));
					nodes.pop();
				}

				nodes.push(call);
			}
			break;

		case TokenType::Operator:
			{
				BinaryOperator* op = new BinaryOperator();
				op->Op = StringToOperation(current->Text);

				op->Right = nodes.top();
				nodes.pop();

				op->Left = nodes.top();
				nodes.pop();

				nodes.push(op);				
			}
			break;

		case TokenType::BlockEnd:
			{
				Block* block = dynamic_cast<Block*>(nodes.top());
				if (nullptr == block)
				{
					Error("Unbalanced braces");
					continue;
				}

				nodes.pop();

				Block* container = dynamic_cast<Block*>(nodes.top());
				if (nullptr == container)
				{
					Error("No containing function or program");
					continue;
				}

				container->Statements.push_back(block);
			}
			break;

		case TokenType::EndOfStatement:
			{
				Block* block;

				while(nullptr == (block = dynamic_cast<Block*>(nodes.top())))
				{
					statements.push_back(nodes.top());
					nodes.pop();
				}

				if (nullptr == block)
				{
					Error("Statement with no containing block or program.");
					continue;
				}

				for(auto statement : statements)
				{
					block->Statements.push_back(statement);
				}

			}
			break;
		}

	}

	return program;
}

