#include "Visitor.h"

void Visitor::Visit(Node& node)
{
}
#define VISIT_EMPTY(type) void Visitor::Visit(type& node) { }

VISIT_EMPTY(Integer);
VISIT_EMPTY(Float);
VISIT_EMPTY(String);

#undef VISIT_EMPTY

void Visitor::Visit(Identifier& identifier)
{
	throw;
}

void Visitor::Visit(Function& function)
{
	function.Accept(*this);

	if(function.Body)
		function.Body->Accept(*this);
}

void Visitor::Visit(Call& call)
{
	call.Callee->Accept(*this);

	for(auto it : call.Parameters)
	{
		it->Accept(*this);
	}
}

void Visitor::Visit(Block& block)
{
	for(auto statement : block.Statements)
	{
		statement->Accept(*this);
	}
}

void Visitor::Visit(Program& program)
{
	program.Block::Accept(*this);
}


void Visitor::Visit(BinaryOperator& op) 
{
	if(op.Left)
		op.Left->Accept(*this);

	if (op.Right)
		op.Right->Accept(*this);

}




