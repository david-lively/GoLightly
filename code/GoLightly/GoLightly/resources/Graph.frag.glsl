#version 430 core

uniform vec4 Color;
in vec4 vWorldPosition;

out vec4 fragmentColor;

void main() {
	fragmentColor = Color;
}

