#version 430 core
uniform sampler2D CudaTexture;
uniform vec2 TextureSize;
uniform vec2 WindowSize;
in vec3 Position;
out vec4 vColor;
out vec3 vWorldPosition;
out vec2 vTexCoord;
void main()
{
	vec4 worldPos = vec4(Position, 1);
	/// scale down the quad coordinates to match the texture aspect ratio
	if (TextureSize.x > TextureSize.y) 
		worldPos.y *= TextureSize.y / TextureSize.x;
	else
		worldPos.x *= TextureSize.x / TextureSize.y;
	float windowAspect = WindowSize.x / WindowSize.y;
	if (windowAspect > 1) 
		worldPos.y *= windowAspect;
	else if (windowAspect < 1) 
		worldPos.x /= windowAspect;
	if (abs(worldPos.x) > 1) worldPos.xy /= abs(worldPos.x);
	if (abs(worldPos.y) > 1) worldPos.xy /= abs(worldPos.y);
	vWorldPosition = worldPos.xyz;
	vTexCoord = vec2(Position.x,-Position.y);
	gl_Position = worldPos;

	vColor = vec4(1);
}

