#version 430 core

uniform sampler2D Font;
uniform unsigned int CharacterCount;

in int Character;
out int vCharacter;
out unsigned int CharacterIndex;

void main()
{
	vCharacter = Character;
	CharacterIndex = gl_VertexID;
	gl_Position = vec4(0,0,0,1);
}

