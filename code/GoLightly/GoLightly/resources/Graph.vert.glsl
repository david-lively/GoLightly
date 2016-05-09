#version 430 core
/// http://www.lighthouse3d.com/cg-topics/code-samples/opengl-3-3-glsl-1-5-sample/
uniform vec2 ViewportSize;
uniform vec2 Location;
uniform vec2 Size;
uniform int SampleCount;
uniform vec2 MinMax;
uniform vec4 Color;

/// note - normalized x,y need to be converted to screen coordinates
in vec2 Position;

out vec4 vWorldPosition;

void main()
{
	/// pixel size in normalized screen coordinates (-1,+1)-(+1,-1)
	vec2 pixelSize = 2.f / ViewportSize;

	float normY = (Position.y - MinMax.y) / (MinMax.x - MinMax.y);

	vWorldPosition = vec4(
		-1 + Location.x * pixelSize.x + Position.x * Size.x / SampleCount * pixelSize.x
		, 1 - Location.y * pixelSize.y - (1 - normY) * Size.y * pixelSize.y
		, 0, 1);


	//float x = -1 + Position.x * pixelSize.x * Size.x / SampleCount;
	//float y = 1 - TopLeft.y * pixelSize.y - (1 - normY) * pixelSize.y * Size.y;

	//vec4 worldPos = vec4(x,y, 0, 1);

	//vWorldPosition = worldPos;

	gl_Position = vWorldPosition;

//    vec4 pos = vec4(Position,1);
//
//	vWorldPosition = (World * pos).xyz;
//	vTexCoord = vec2(Position.xz) + vec2(0.5);
//
//    gl_Position = Projection * View * World * pos;
//
//	vColor = vec4(1);
}

