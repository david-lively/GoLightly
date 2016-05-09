#ifndef INPUT_H
#define INPUT_H

#include <array>

typedef std::array<bool,512> KeyboardState;

/// <summary>
/// TokenType of key state transitions
/// </summary>
#if _MSC_VER > 1600
/// C++11 scoped enums are not supported before VS2012
enum class InputTransition
#else
#pragma warning(disable : 4482)
enum InputTransition
#endif
{
	None
	,
	Press
	,
	Hold
	,
	Release
};


#endif