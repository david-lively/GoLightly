#ifndef VISITOR_H
#define VISITOR_H

#include "AST.h"

/*
Interface class for Visitor types
*/
struct Visitor
{
	virtual ~Visitor() = 0 {}

	virtual void Visit(Program& program)		;
	virtual void Visit(Node& node)				;
	virtual void Visit(Identifier& identifier)	;
	virtual void Visit(Float& identifier)		;
	virtual void Visit(Integer& identifier)		;
	virtual void Visit(String& identifier)		;
	virtual void Visit(Function& function)		;
	virtual void Visit(Call& call)				;
	virtual void Visit(Block& block)			;
	virtual void Visit(BinaryOperator& op)		;
};

#endif