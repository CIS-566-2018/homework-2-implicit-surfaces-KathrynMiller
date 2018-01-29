#version 300 es

precision highp float;
uniform vec2 u_Dimensions;

vec3 lightPos = vec3(0.0, 0.0, 1.0);
float nearClip = 0.1;
float farClip = 1000.0;
float fov = radians(80.0);
vec3 cameraPos = vec3(0.0, .0, 1.0);
vec3 cameraLook = vec3(0.0, 0.0, -1.0);
vec3 cameraUp = vec3(0.0, 1.0, 0.0);
vec3 cameraRight = vec3(1.0, 0.0, 0.0);
vec3 ref = vec3(0.0, 0.0, 0.0);
float epsilon = .002;
float max_steps = 200.0;
float max_dist = 1000.0;

out vec4 out_Col;

// functions to return rotation/translation matrices given x, y, z values
mat4 transMat(vec3 t) {
	return mat4(vec4(1.0, 0.0, 0.0, 0.0),
	vec4(0.0, 1.0, 0.0, 0.0),
	vec4(0.0, 0.0, 1.0, 0.0),
	vec4(t.x, t.y, t.z, 1.0));
}

mat4 rotX(float a) {
	return mat4(vec4(1, 0.0, 0.0, 0.0),
	vec4(0, cos(a), sin(a), 0.0),
	vec4(0.0, -sin(a), cos(a), 0.0),
	vec4(0.0, 0.0, 0.0, 1.0));
}

mat4 rotY(float a) {
	return mat4(vec4(cos(a), 0.0, -sin(a), 0.0),
	vec4(0.0, 1.0, 0.0, 0.0),
	vec4(sin(a), 0.0, cos(a), 0.0),
	vec4(0.0, 0.0, 0.0, 1.0));
}

mat4 rotZ(float a) {
	return mat4(vec4(cos(a), sin(a), 0.0, 0.0),
	vec4(-sin(a), cos(a), 0.0, 0.0),
	vec4(0.0, 0.0, 1.0, 0.0),
	vec4(0.0, 0.0, 0.0, 1.0));
}


// SDF functions


float sdSphere(vec3 p, float s)
{
  return length(p) - s;
}

float sdPlane( vec3 p, vec4 n )
{
  // n must be normalized
  return dot(p,n.xyz) + n.w;
}

float sdBox(vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
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

vec2 counterSDF(vec3 p) {
	vec3 trans = vec3(transMat(vec3(0.0, -0.25, 0.0)) * vec4(p, 1.0));
	return vec2(udRoundBox(trans / .1, vec3(10.0, .5, 1.0), .2), 2.0) * .1;
}


// exponential smooth min (k = 32);
float smin( float a, float b, float k )
{
    float res = exp( -k*a ) + exp( -k*b );
    return -log( res )/k;
}

vec2 glassSDF(vec3 p) {
	vec3 trans = vec3(transMat(vec3(0.0, -.1, 0.0)) * vec4(p, 1.0));
	float glass = sdCappedCone(trans / .6, vec3(4.0, 1.0, 1.0)) * .6;
	vec3 rotbase = vec3(rotZ(radians(180.0)) * vec4(p, 1.0));
	vec3 trans2 = vec3(transMat(vec3(0.0, -.00002, 0.0)) * vec4(rotbase, 1.0));
	float base = sdCappedCone(trans2 / .3, vec3(3.0, 1.5, 1.0)) * .3;
	vec3 trans3 =  vec3(transMat(vec3(0.0, .5, 0.0)) * vec4(p, 1.0));
	float rim = sdTorus(trans3, vec2(.2, .01));
	float sdf = smin(glass, base, 60.0);
	sdf =  smin(rim, sdf, 30.0);
	return vec2(sdf, 1.0);
}


// Union (with material data)
vec2 opU( vec2 d1, vec2 d2 )
{
    return (d1.x < d2.x) ? d1 : d2;
}

vec2 sceneSDF(vec3 p) {
	vec2 sdf = opU(glassSDF(p), counterSDF(p));
	return sdf;
}


// return point in world space
vec3 rayCast(vec2 pixel) {
	float len = length(ref - cameraPos);
	vec3 V = cameraUp * len * tan(fov / 2.0);
	vec3 H = cameraRight * len * tan(fov / 2.0);
	vec3 point = ref + pixel.x * H + pixel.y * V;
	return point;
}

vec3 getNormal(vec3 pos) {
    vec2 e = vec2(0.0, 0.001);
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
			return dist;
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

float intersect( float d1, float d2 )
{
    return max(d1,d2);
}

float sub( float d1, float d2 )
{
    return max(-d1,d2);
}

float add( float d1, float d2 )
{
    return min(d1,d2);
}

void main() {
	lightPos = cameraPos;
	vec2 pixel = (2.0 * gl_FragCoord.xy - u_Dimensions.xy) / -u_Dimensions.y;
	vec3 rayDir = normalize(rayCast(pixel) - cameraPos);

	vec2 object = rayMarch(rayDir, cameraPos);
	float t = object.x;
	float type = object.y;
	vec3 normal = getNormal(cameraPos + t * rayDir);
	vec3 point = cameraPos + t * rayDir;
	vec3 lightDir = normalize(lightPos - point);
	// calculate lambertian shading
	float diffuseTerm = dot(normal, normalize(vec3(lightPos - point)));
    diffuseTerm = clamp(diffuseTerm, 0.0, 1.0);
	if(t != -1.0) {
		//out_Col = vec4(normal, 1.0);
		out_Col = vec4(vec3(.5, .5, 0.0) * diffuseTerm, 1.0);
	} else {
		out_Col = vec4(0.5, 0.0, 0.2, 1.0);
	}
	
}
