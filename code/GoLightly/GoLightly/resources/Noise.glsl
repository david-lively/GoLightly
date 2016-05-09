
vec3 permute(vec3 x) {
	return mod(((x * 34.0) +1.0) *x , 289.0) ; 
}

vec3 taylorInvSqrt(vec3 r) {
	return 1.79284291400159 - 0.85373472095314 * r ; 
}


float snoise(vec2 P) {
	const vec2 C = vec2(0.211324865405187134 , 
	0.366025403784438597) ; 

	vec2 i =floor(P + dot(P , C.yy)) ;
	vec2 x0 = P - i + dot(i , C.xx) ;

	vec2 i1 ;
	i1.x = step(x0.y , x0.x) ; 
	i1.y = 1.0 - i1.x ;
	
	vec4 x12 = x0.xyxy + vec4(C.xx , C.xx * 2.0 - 1.0) ;
	x12.xy -= i1 ;
	
	i = mod(i ,289.0) ; 
	
	vec3 p = permute(permute(i.y + vec3(0.0 , i1.y , 1.0))
	+ i.x + vec3(0.0 , i1.x , 1.0)) ;
	
	vec3 m = max(0.5 - vec3(dot(x0 , x0) , dot(x12.xy , x12.xy) ,
	dot(x12.zw , x12.zw)) , 0.0) ;
	m = m*m ;
	m = m*m ;

	vec3 x = fract(p *(1.0 / 41.0)) * 2.0 - 1.0 ;
	vec3 gy = abs(x) - 0.5 ;
	vec3 ox =floor(x + 0.5) ; 
	vec3 gx = x - ox ;

	m *= taylorInvSqrt(gx * gx + gy * gy) ;

	vec3 g ;
	g.x = gx.x * x0.x + gy.x * x0.y ;
	g.yz = gx.yz * x12.xz + gy.yz * x12.yw ;

	return 130.0 * dot(m , g) ;
}