#version 430 core
uniform sampler2D CudaTexture;
uniform float ColorScale = 100.f;
in vec2 vTexCoord;
in vec3 vWorldPosition;
in vec4 vColor;
out vec4 fragmentColor;
vec4 saturate(vec4 val) { return clamp(val,vec4(0),vec4(1)); }
void main() {
	vec2 texCoord = vTexCoord / 2 + vec2(0.5);
	vec4 texel = texture2D(CudaTexture,texCoord);
	float r = texel.r * ColorScale;
	float b = -r;
	float dielectric = texel.g;
	float g = (dielectric >= 0.5) ? 0 : dielectric * 2;
	vec4 t = saturate(vec4(r,g,b,1));
	fragmentColor = saturate(vec4(r,g,b,1));
}
