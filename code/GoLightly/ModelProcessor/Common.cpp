#include <map>
#include <string>

using namespace std;

#include "Common.h"

map<string,Operation> CreateOperationStringMap()
{
	map<string,Operation> result;

	result["="]  = Operation::Assign	    ;
	result["("]  = Operation::ListStart		;
	result[","]  = Operation::ListDelimiter ;
	result[")"] =  Operation::ListEnd		;
	result["^"]  = Operation::Exponent		;
	result["*"]  = Operation::Multiply		;
	result["/"]  = Operation::Divide		; 
	result["+"]  = Operation::Add			;
	result["-"]  = Operation::Subtract		;
	result["++"] = Operation::Increment     ;
	result["--"] = Operation::Decrement		;
	result["=="] = Operation::EqualTo		;
	result["+="] = Operation::IncrementBy	;
	result["-="] = Operation::DecrementBy	;
	result["*="] = Operation::MultiplyBy	;
	result["/="] = Operation::DivideBy		;

	result[">"]  = Operation::GreaterThan	;
	result["<"]  = Operation::LessThan		;
	result["{"]  = Operation::BlockStart	;
	result["}"]  = Operation::BlockEnd		;		
	result["()"] = Operation::Function		;



	return result;
}

static map<string,Operation> operationStrings = CreateOperationStringMap();

Operation StringToOperation(const string& text)
{
	auto it = operationStrings.find(text);

	if (it != end(operationStrings))
		return it->second;
	else
		return Operation::Unknown;
}

string OperationToString(const Operation& op)
{
	for(auto it = begin(operationStrings); it != end(operationStrings); ++it)
	{
		if (it->second == op)
			return it->first;
	}

	return "INVALID OPERATION VALUE";
}




