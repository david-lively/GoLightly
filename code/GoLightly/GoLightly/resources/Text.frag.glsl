#version 430 core

uniform sampler2D Font;
uniform bool SDFText = true;

in vec2 TexCoord;

out vec4 fragmentColor;

void main() {

	vec4 texel = texture(Font,TexCoord);

	float dist = texel.a - 0.5f;

	if (dist <= 0)
	{
		fragmentColor = vec4(1);
	}
	else
		fragmentColor = vec4(0);
}

