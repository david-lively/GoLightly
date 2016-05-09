#include <iostream>
#include <minmax.h>
using namespace std;

#include "PrintVisitor.h"



void PrintVisitor::Visit(Node& node)
{
	Print("PrintVisitor: Node\n");
}

void PrintVisitor::Indent() { ++m_indentLevel; } 

void PrintVisitor::Unindent() { m_indentLevel = max(m_indentLevel - 1, 0); }

/* print method for everything */
template<typename T>
void PrintVisitor::Print(const T& message)
{
	cout << string(m_indentLevel,'\t') << message << endl;
}

#define VISIT_TYPEDCONSTANT(type) void PrintVisitor::Visit(type& node) { Print(node.Value); }


VISIT_TYPEDCONSTANT(Integer);
VISIT_TYPEDCONSTANT(Float);
VISIT_TYPEDCONSTANT(String);
VISIT_TYPEDCONSTANT(Identifier);

#undef VISIT_TYPEDCONSTANT

void PrintVisitor::Visit(Program& program)
{
	Print("Program");

	Indent();

	Visitor::Visit(program);

	Unindent();
}


void PrintVisitor::Visit(Function& function)
{
	Print("Function");

	Indent();

	Visitor::Visit(function);

	Unindent();
}

void PrintVisitor::Visit(Call& call)
{
	Print("Call");

	Indent();

	call.Callee->Accept(*this);

	Indent();

	for(auto param : call.Parameters)
	{
		param->Accept(*this);
	}

	Unindent();


	Unindent();
}

void PrintVisitor::Visit(Block& block)
{
	Print("Block");

	Indent();

	Visitor::Visit(block);

	Unindent();
}



void PrintVisitor::Visit(BinaryOperator& op) 
{
	Print(OperationToString(op.Op));

	Indent();

	Visitor::Visit(op);

	Unindent();
}





