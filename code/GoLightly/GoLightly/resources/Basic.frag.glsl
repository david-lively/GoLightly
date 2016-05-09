#version 430 core

uniform sampler2D CudaTexture;
uniform float ColorScale = 100.f;

in vec2 vTexCoord;
in vec3 vWorldPosition;
in vec4 vColor;

out vec4 fragmentColor;

vec4 saturate(vec4 val)
{
	return clamp(val,vec4(0),vec4(1));
}

void main() {
	vec2 texCoord = vTexCoord / 2 + vec2(0.5);



	vec4 texel = texture2D(CudaTexture,texCoord);
	//texel.b = 1.f;

	//fragmentColor = texel;
	//return;

	/// texel.r = field value, scale so we can see it in the output	
	float r = texel.r * ColorScale;
	/// blue so we can see the valleys
	float b = -r;
	
	float dielectric = texel.g;
	float g = (dielectric >= 0.5) ? 0 : dielectric * 2;
	
	vec4 t = saturate(vec4(r,g,b,1));

	//vec4 waveMin = vec4(1,0,0,1);
	//vec4 waveMid = vec4(1,1,1,1);
	//vec4 waveMax = vec4(0,0,1,1);

	//fragmentColor.r = mix(1,0,t.r);
	//fragmentColor.b = mix(1,0,t.b);
	//fragmentColor.g = mix(1,0,t.g);
	//fragmentColor.a = 1;
	fragmentColor = saturate(vec4(r,g,b,1));
	
	//if (t.r > 0)
	//{
	//	fragmentColor = mix(waveMin,waveMid,t.r);
	//}
	//else 
	//{
	//	fragmentColor = mix(waveMid,waveMax,t.b);
	//}

	//fragmentColor.g = t.g;

	/////// show the grid
	////if (texel.a > 0.5)
	////	r = g = b = 0.25f;


	//fragmentColor = saturate(vec4(r,g,b,1));
}

