#ifndef PRINTVISITOR_H
#define PRINTVISITOR_H

#include <string>
using namespace std;

#include "Visitor.h"

/// prints an ASCII version of an abstact syntax tree
struct PrintVisitor : public Visitor
{
	PrintVisitor() : m_indentLevel(0)
	{
	}

	void Visit(Node& node)			override;
	void Visit(Identifier& node)	override;
	void Visit(Float& f)   override;
	void Visit(Integer& identifier) override;
	void Visit(String& identifier) 	override;


	void Visit(Program& program)	override;
	void Visit(Function& function)	override;
	void Visit(Call& call)	override;
	void Visit(Block& block)	override;

	void Visit(BinaryOperator& op)  override;

private:
	unsigned int m_indentLevel;

	void Indent();
	void Unindent();

	template<typename T>
	void Print(const T& message);
};


#endif