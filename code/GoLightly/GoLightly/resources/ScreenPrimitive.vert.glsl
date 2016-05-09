#version 430 core
/// http://www.lighthouse3d.com/cg-topics/code-samples/opengl-3-3-glsl-1-5-sample/
uniform vec2 ViewportSize;
uniform vec2 Location;
uniform vec2 Size;
uniform vec4 Color;

in vec2 Position;

out vec4 vColor;
out vec4 vWorldPosition;

void main()
{
	vec2 pixelSize = 2.f / ViewportSize;

	vWorldPosition = vec4(
		-1 + Location.x * pixelSize.x + Position.x * Size.x * pixelSize.x
		, 1 - Location.y * pixelSize.y - Position.y * Size.y * pixelSize.y
		, 0, 1);

	gl_Position = vWorldPosition;

	vColor = Color;

//    vec4 pos = vec4(Position,1);
//
//	vWorldPosition = (World * pos).xyz;
//	vTexCoord = vec2(Position.xz) + vec2(0.5);
//
//    gl_Position = Projection * View * World * pos;
//
//	vColor = vec4(1);
}

