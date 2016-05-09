#include "AST.h"

#include "Visitor.h"

#ifdef ACCEPT
#undef ACCEPT
#endif

#define ACCEPT(type) void type::Accept(Visitor& visitor) { visitor.Visit(*this); }

ACCEPT(Node             )
ACCEPT(Integer			)
ACCEPT(Float			)
ACCEPT(String			)
ACCEPT(Identifier		)
ACCEPT(BinaryOperator	)
ACCEPT(Call				)
ACCEPT(Block			)
ACCEPT(Function			)
ACCEPT(Program			)
