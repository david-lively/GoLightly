#ifndef COMMON_H
#define COMMON_H

#include <map>
#include <string>
#include <iostream>
#include <ostream>

using namespace std;

#define STRINGIZE_HELPER(x) #x
#define STRINGIZE(x) STRINGIZE_HELPER(x)
#define WARNING(desc) message(__FILE__ "(" STRINGIZE(__LINE__) ") : Warning: " #desc)

/// #pragma WARNING(FIXME: because...)


enum class Operation
{
	Unknown = 0
	,ListEnd
	,ListDelimiter      /// , separate items in a comma-delimited list
	,ListStart		/// (Expression)
	,BlockStart
	,BlockEnd
	,Assign
	,Call
	,EqualTo
	,Increment
	,Decrement
	,GreaterThan
	,LessThan
	,Exponent			/// base ^ power - NOT an XOR!
	,Multiply			/// a * b
	,Divide				/// a / b
	,Add				/// a + b
	,Subtract			/// -
	,IncrementBy
	,DecrementBy
	,MultiplyBy
	,DivideBy
	,Function
	,Push
	,Pop
};

Operation StringToOperation(const string&);
string OperationToString(const Operation&);



inline void Abort(const string &message)
{
	cerr << "Error:" << message << endl;
	exit(EXIT_FAILURE);
}

inline void Expected(const string &s)
{
	Abort(s + " expected");
}

inline void Error(const string& message)
{
	Abort(message);
}

inline void Warning(const string& message)
{
	cerr << "warning: " << message << endl;
}


#endif