#version 300 es

precision highp float;
uniform vec2 u_Dimensions;
uniform float u_Time;
uniform float u_Scheme;
uniform float u_Reflect;

vec3 lightPos = vec3(7.0, -1.0, 4.0);
float nearClip = 0.1;
float farClip = 1000.0;
float fov = radians(16.0);
vec3 cameraPos = vec3(0.0, -.7, 4.0);
vec3 cameraLook = vec3(0.0, 0.0, -1.0);
vec3 ref = vec3(0.0, 0.0, 0.0);
float epsilon = .002;
float max_steps = 200.0;
float max_dist = 1000.0;
vec4 colorScene(vec3 rd, float t, float type);

const vec3 peach = vec3(255.0 / 255.0, 215.0 / 255.0, 199.0 / 255.0);
const vec3 cream = vec3(246.0 / 255.0, 236.0 / 255.0, 228.0 / 255.0);
const vec3 cherry = vec3(202.0 / 255.0, 22.0 / 255.0, 31.0 / 255.0);
const vec3 light_grey = vec3(227.0 / 255.0, 227.0 / 255.0, 230.0 / 255.0);

const vec3 teal = vec3(28.0 / 255.0, 131.0 / 255.0, 139.0 / 255.0);
const vec3 mint = vec3(146.0 / 255.0, 223.0 / 255.0, 195.0 / 255.0);
const vec3 bright_blue = vec3(151.0 / 255.0, 233.0 / 255.0, 228.0 / 255.0);
const vec3 bright_red = vec3(230.0 / 255.0, 56.0 / 255.0, 71.0 / 255.0);
const vec3 silver = vec3(127.0 / 255.0, 134.0 / 255.0, 138.0 / 255.0);
const vec3 dark_grey = vec3(72.0 / 255.0, 73.0 / 255.0, 77.0 / 255.0);
const vec3 yellow = vec3(198.0 / 255.0, 173.0 / 255.0, 16.0 / 255.0);

float fbm(const in vec3 uv);
const vec3 counterCol[11] = vec3[](peach, cream, cherry, light_grey, teal, mint, bright_blue, bright_red, silver, dark_grey, yellow);
out vec4 out_Col;

// SDF functions
float sdSphere(vec3 p, float s)
{
  return length(p) - s;
}

float sdCappedCone(in vec3 p, in vec3 c )
{
    vec2 q = vec2( length(p.xz), p.y );
    vec2 v = vec2( c.z*c.y/c.x, -c.z );
    vec2 w = v - q;
    vec2 vv = vec2( dot(v,v), v.x*v.x );
    vec2 qv = vec2( dot(v,w), v.x*w.x );
    vec2 d = max(qv,0.0)*qv/vv;
    return sqrt( dot(w,w) - max(d.x,d.y) ) * sign(max(q.y*v.x-q.x*v.y,w.y));
}

float udRoundBox( vec3 p, vec3 b, float r )
{
  return length(max(abs(p)-b,0.0))-r;
}

float sdTorus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

float sdCylinder( vec3 p, vec3 c )
{
  return length(p.xz-c.xy)-c.z;
}

float udBox( vec3 p, vec3 b )
{
  return length(max(abs(p)-b,0.0));
}
// objects
vec2 counterSDF(vec3 p) {
	p = vec3(p.x, p.y - 0.63, p.z + .2);
	return vec2(udRoundBox(p / .3, vec3(10.0, .1, 1.7), 1.0) * .3, 2.0);
}

// exponential smooth min (k = 32);
float smin( float a, float b, float k )
{
    float res = exp( -k*a ) + exp( -k*b );
    return -log( res ) / k;
}

vec3 opRep( vec3 p, vec3 c )
{
    vec3 q = mod(p,c)-0.5 * c;
    return q;
}
// include noraml pos and changing pos to cut out bottom half of cream
vec2 creamSDF(vec3 p, vec3 tp) {
	p.x += .5;
	tp.x += .5;
	vec3 trans = tp + vec3(0.0, .28, 0.0);
	float t1 = sdTorus(trans, vec2(.16, .091));
	vec3 trans2 = tp + vec3(0.0, .32, 0.0);
	float t2 = sdTorus(trans2, vec2(.08, .1));
	vec3 trans3 = tp + vec3(0.0, .35, 0.0);
	float t3 = sdTorus(trans3, vec2(.02, .1));
	float sdf = smin(t1, t2, 200.0);
	sdf = min(sdf, t3);
	vec3 trans4 = p + vec3(-0.1, .81, .4);
	float cutBox = udBox(trans4, vec3(1.0, .53, 1.0));
	sdf = max(cutBox, sdf);
	return vec2(sdf, 3.0);
}

vec2 cherrySDF(vec3 p) {
	p.x += .5;
	vec3 trans = p + vec3(0.0, .45, -0.0);
	float cherry = sdSphere(trans, .04);
	return vec2(cherry, 4.0);
}
mat4 rotZ(float a) {
	return mat4(vec4(cos(a), sin(a), 0.0, 0.0),
	vec4(-sin(a), cos(a), 0.0, 0.0),
	vec4(0.0, 0.0, 1.0, 0.0),
	vec4(0.0, 0.0, 0.0, 1.0));
}

vec2 glassSDF(vec3 p) {
	p.x += .5;
	vec3 trans = p + vec3(0.0, -.3, 0.0);
	float glass = sdCappedCone(trans / .6, vec3(2.5, 1.1, 1.0)) * .6;
	float c = cos(radians(180.0));
	float s = cos(radians(180.0));
	vec3 rotbase = vec3(rotZ(radians(180.0)) * vec4(p, 1.0));
	vec3 trans2 = rotbase + vec3(0.0, -.02, 0.0);
	float base = sdCappedCone(trans2 / .3, vec3(3.0, 1.5, 1.0)) * .3;
	vec3 trans3 = p + vec3(0.0, .3, 0.0);
	float rim = sdTorus(trans3, vec2(.3, .01));
	float sdf = smin(glass, base, 70.0);
	vec3 trans4 = p + vec3(0.0, -.3, 0.0);
	float bottomRim = sdTorus(trans4, vec2(.2, .02));
	sdf =  smin(rim, sdf, 30.0);
	sdf = smin(sdf, bottomRim, 30.0);
	return vec2(sdf, 1.0);
}

vec2 straw(vec3 p) {
	p.x += .5;
	vec3 trans = p + vec3(-0.1, .3, .0);
	vec3 rot = vec3(rotZ(radians(-35.0)) * vec4(trans, 1.0));
	float straw = sdCylinder(rot, vec3(.03, .05, .03));
	vec3 trans2 = p + vec3(-0.6, .3, .4);
	float cutBox = udBox(trans2, vec3(.5, .5, 1.0));
	float sdf = max(cutBox, straw);
	return vec2(sdf, 5.0);
}

vec2 tiles(vec3 pos) {
	pos.z = pos.z + 5.5;
	vec3 repeat = opRep(pos, vec3(1.05, .83, 0.0));
	return vec2(udRoundBox(repeat, vec3(.3, .2, .01), .2), 7.0);
}

// Union (with material data)
vec2 opU( vec2 d1, vec2 d2 )
{
    return (d1.x < d2.x) ? d1 : d2;
}

vec2 sceneSDF(vec3 p) {
	// value for time between 0 and 1
	float intervalLength = 170.0;
	float time = mod(u_Time, intervalLength) / intervalLength; // t is 0 - 1
	// tp = time based position
	vec3 tp = vec3(p.xyz);	
	// sink the cream into glass
		tp.y += (sin(8.0 * time) -  8.0 * time) / 4.0;
		//tp.y -= abs(.1 * (sin(2.0 * (time * 4.0)) / (.4*(time*5.0)+.5))) + .01;
	vec2 sdf = opU(glassSDF(p), creamSDF(p, tp));
	sdf = opU(sdf, straw(p));
	sdf = opU(sdf, cherrySDF(tp));
	sdf = opU(sdf, counterSDF(p));
	sdf = opU(sdf, tiles(p));
	return sdf;
}

float checkersGradBox( in vec2 p )
{
	// filter kernel
    vec2 w = fwidth(p) + 0.001;
    // analytical integral (box filter)
    vec2 i = 2.0*(abs(fract((p-0.5*w)*0.5)-0.5)-abs(fract((p+0.5*w)*0.5)-0.5))/w;
    // xor pattern
    return 0.5 - 0.5*i.x*i.y;                  
}


vec3 rayCast(vec2 pixel) {
    vec3 camLook = normalize(ref - cameraPos);
    vec3 camRight = normalize(cross(camLook, vec3(0.0, 1.0, 0.0)));
    vec3 camUp = normalize(cross(camRight, camLook));

	float len = 0.1;// length(ref - cameraPos);
	vec3 V = camUp * len * tan(fov / 2.0);
	vec3 H = camRight * len * (u_Dimensions.x / u_Dimensions.y ) * tan(fov / 2.0);
	vec3 point = len * camLook + pixel.x * H + pixel.y * V;
	return normalize(point);
}

vec3 getNormal(vec3 pos) {
    vec2 e = vec2(0.0, .01);
    return normalize( vec3( sceneSDF(pos + e.yxx).x - sceneSDF(pos - e.yxx).x,
                            sceneSDF(pos + e.xyx).x - sceneSDF(pos - e.xyx).x,
                            sceneSDF(pos + e.xxy).x - sceneSDF(pos - e.xxy).x));
}

vec2 rayMarch(vec3 origin, vec3 dir) {
	// distance to march
	float t = 0.01;
	// distance along the ray
	vec2 dist = vec2(0.01, 0.0);
	for(float i = 0.0; i < max_steps; i++) {
		dist = sceneSDF(vec3(origin + t * dir));
		if(dist.x < epsilon) {
			// return previously marched distance
			return vec2(t, dist.y);
		}
		// add distance to closest object to t
		t += dist.x;

		// return if gone too far along ray
		if(t >= max_dist) {
			return vec2(-1.0, 0.0);
		}
	}
	return vec2(-1.0, 0.0);
}

void main() {
	vec2 pixel = (2.0 * gl_FragCoord.xy - u_Dimensions.xy) / -u_Dimensions.y;
	vec3 rayDir = normalize(rayCast(pixel));

	vec2 object = rayMarch(cameraPos, rayDir);
	float t = object.x;
	float type = object.y;
	
	vec3 point = cameraPos + t * rayDir;

	vec3 lightDir = normalize(lightPos - point);
	if(t != -1.0) { // color the hit objects
	// treat glass differently for reflectivity
		if(type == 1.0) { // glass
			vec3 normal = getNormal(point);
			vec3 H = normalize((cameraPos + lightPos - point) / 2.0);
			float diffuseTerm = dot(normalize(normal), normalize(lightPos - point));
   			diffuseTerm = clamp(diffuseTerm, 0.0, 1.0);
			float ambientTerm = 0.2;
			float lightIntensity = diffuseTerm + ambientTerm;

			float specularIntensity = max(pow(dot(H, normal), 60.0), 0.0);	
			float translucency = 1.0;
			vec4 reflectCol = vec4(1.0);
			vec4 baseCol = vec4(peach * (lightIntensity + specularIntensity), 1.0);
			// if reflectivity in gui is set, find reflected color
			if(u_Reflect == 1.0) {
				// find reflected color
				vec3 reflectedRayDir = normalize(reflect(rayDir, normal));
				// raymarch to find the hit object
				vec2 hitObject = rayMarch(point, reflectedRayDir);
				reflectCol = colorScene(reflectedRayDir, hitObject.x, hitObject.y);
				out_Col = vec4(baseCol.xyz * reflectCol.xyz * 4.0, 1.0);
			} else {
				out_Col = baseCol;
			}
			
	} else {	
		vec4 color = colorScene(rayDir, t, type);
		out_Col = color;
		}
	} else { // make back wall grey
		out_Col = vec4(dark_grey, 1.0);
	}
	
}

vec4 colorScene(vec3 rd, float t, float type) {
    // calculate normal
	vec3 normal = getNormal(cameraPos + t * rd);

	// computer blinn phong and lambertian shading
	vec3 point = cameraPos + t * rd;
	vec3 H = normalize((cameraPos + lightPos - point) / 2.0);
    float diffuseTerm = dot(normalize(normal), normalize(lightPos - point));
    diffuseTerm = clamp(diffuseTerm, 0.0, 1.0);
	float ambientTerm = 0.2;
	float lightIntensity = diffuseTerm + ambientTerm;
	 if (type == 2.0) { // counter
	 	// vary counter surface print
	 	float specularIntensity = max(pow(dot(H, normal), 30.0), 0.0);
	 	if(u_Scheme == 0.0) {
			float det = mod(t * 40.0, 11.0);
			vec4 color = vec4(counterCol[int(det)], 1.0); 
			return color * vec4(lightIntensity + (specularIntensity * mint), 1.0);
	 	} else if (u_Scheme == 1.0) { 
			float f = checkersGradBox(5.0 * point.xz);
        	vec3 col = .5 + f*vec3(0.6);
			return vec4(col, 1.0) * vec4(lightIntensity + (specularIntensity * mint), 1.0);
	 	} else if (u_Scheme == 2.0) {
		 	float det = mod(normal.y * 40.0, 11.0);
			vec4 color = vec4(counterCol[int(det)], 1.0); 
			return color * vec4(lightIntensity + (specularIntensity * mint), 1.0);
		}

	} else if (type == 3.0) { // cream
		return vec4(cream * lightIntensity, 1.0);
	} else if (type == 4.0) { // cherry
		float specularIntensity = max(pow(dot(H, normal), 10.0), 0.0);	
		return vec4(cherry * (lightIntensity + specularIntensity * .06), 1.0);
	} else if (type == 5.0) { // straw
		vec3 col = mint;
		if(u_Scheme == 1.0) {
			col = teal;
		} else if (u_Scheme == 2.0) {
			col = bright_blue;
		}
		return vec4(col * (lightIntensity), 1.0);
	} else if (type == 7.0) { // tiles
		float specularIntensity = max(pow(dot(H, normal), 60.0), 0.0);
		vec3 col = cherry;
		if(u_Scheme == 1.0) {
			col = mint;
		} else if (u_Scheme == 2.0) {
			col = yellow;
		}
		return vec4(col * (lightIntensity + (specularIntensity * bright_blue)), 1.0);
	} else {
		return vec4(0.0);
	}
}