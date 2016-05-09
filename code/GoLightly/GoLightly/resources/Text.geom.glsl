#version 410 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform vec2 ViewportSize;
uniform int FontSize;

/// top-left pixel coordinate of first character
uniform vec2 StartPosition;

in int vCharacter[1];
in unsigned int CharacterIndex[1];

out vec2 TexCoord;


void main(void)
{
	float padding = 1.f / ViewportSize.x;
	uint code = vCharacter[0] - ' ';

	/// size of a character cell in pixels
	vec2 charPixelSize = FontSize * vec2(20 / 32.f, 1);
	/// size of a character cell in texels
	vec2 texCharSize = vec2(20,31) / vec2(512.f,256.f);
	
	/// number of character rows and columns in the viewport
	vec2 columnsAndRows = vec2(ViewportSize.x * 1.f / charPixelSize.x, ViewportSize.y * 1.f / charPixelSize.y);

	vec2 drawSize = 1.f / columnsAndRows;
	vec2 topLeft = vec2(-1.f + CharacterIndex[0] * (drawSize.x + padding), 1.f) + StartPosition * drawSize;

	//vec2 topLeft = vec2(10,0);//vec2(-1.f + CharacterIndex[0] * (drawSize.x + padding), 1.f) + StartPosition;// * drawSize;
	vec2 texTopLeft = vec2(code % 16, code / 16) / vec2(16.f,8.f) + vec2(1/512.f,1/256.f);
	
	TexCoord = texTopLeft + vec2(texCharSize.x,0);
	gl_Position = vec4(topLeft.x + drawSize.x, topLeft.y, 0, 1);
	EmitVertex();

	TexCoord = texTopLeft + texCharSize;
	gl_Position = vec4(topLeft.x + drawSize.x, topLeft.y - drawSize.y,0,1);
	EmitVertex();

	TexCoord = texTopLeft + vec2(0,0);
	gl_Position = vec4(topLeft, 0, 1);
	EmitVertex();

	TexCoord = texTopLeft + vec2(0,texCharSize.y);
	gl_Position = vec4(topLeft.x, topLeft.y - drawSize.y,0,1);
	EmitVertex();

	EndPrimitive();

}

