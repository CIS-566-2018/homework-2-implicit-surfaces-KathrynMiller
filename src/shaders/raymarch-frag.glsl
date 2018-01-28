#version 300 es

precision highp float;
uniform vec2 u_Dimensions;

vec3 lightPos = vec3(0.0, 4.0, 5.0);
float nearClip = 0.1;
float farClip = 1000.0;
float fov = 45.0;
vec3 cameraPos = vec3(0.0, 0.0, 7.0);
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

mat4 rotXZ(float a) {
	return mat4(vec4(cos(a), -sin(a), 0.0, 0.0),
	vec4(sin(a), cos(a), 0.0, 0.0),
	vec4(1.0, 0.0, 1.0, 0.0),
	vec4(1.0, 0.0, 0.0, 1.0));
}

mat4 rotY(float a) {
	return mat4(vec4(cos(a), 0.0, sin(a), 0.0),
	vec4(-sin(a), 0.0, cos(a), 0.0),
	vec4(1.0, 0.0, 1.0, 0.0),
	vec4(1.0, 0.0, 0.0, 1.0));
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

float bunSDF(vec3 p) {
	// top bun
	float sdf = sdSphere(p, 1.0);
	vec3 transBox = vec3(transMat(vec3(.0, -0.5, .0)) * vec4(p, 1.0));
	float box = sdBox(transBox, vec3(1.2, 0.5, 1.2));
	sdf = max(-box, sdf);
	// bottom bun
	vec3 bottomBunTrans = vec3(transMat(vec3(0.0, -1.0, 0.0)) * vec4(p, 1.0));
	//translate bottom box
	vec3 transBox2 = vec3(transMat(vec3(.0, -1.9, .0)) * vec4(p, 1.0));
	float cut = sdBox(transBox2, vec3(1.2, 0.7, 1.2));
	float bottomBun = sdSphere(bottomBunTrans, 1.0);
	bottomBun = max(-cut, bottomBun);
	bottomBun = max(-box, bottomBun);
	sdf = min(sdf, bottomBun);
	return sdf;
}

float pattySDF(vec3 p) {
	// create patty
	vec3 transBox = vec3(transMat(vec3(.0, -1.5, .0)) * vec4(p, 1.0));
	vec3 transBox2 = vec3(transMat(vec3(.0, .8, .0)) * vec4(p, 1.0));
	float cut1 = sdBox(transBox, vec3(1.2, 0.9, 1.2));
	float cut2 = sdBox(transBox2, vec3(1.2, 0.9, 1.2));
	vec3 pattyTrans = vec3(transMat(vec3(0.0, -0.3, 0.0)) * vec4(p, 1.0));
	float sdf = sdSphere(pattyTrans, 1.1);
	sdf = max(-cut1, sdf);
	sdf = max(-cut2, sdf);
	return sdf;
}

float counterSDF(vec3 p) {
	vec3 trans = vec3(transMat(vec3(0.0, -1.0, 0.0)) * vec4(p, 1.0));
	return sdPlane(trans, vec4(0.0, 0.0, 1.0, 1.0));
}

float shakeSDF(vec3 p) {
	vec3 trans = vec3(transMat(vec3(-2.0, -1.0, 0.0)) * vec4(p, 1.0));
	float glass = sdCappedCone(trans / 4.0, vec3(4.0, 1.0, 1.0)) * 4.0;
	vec3 trans2 = vec3(transMat(vec3(-2.0, -1.0, 0.0)) * vec4(p, 1.0));
	trans2 = vec3(rotXZ(180.0) * vec4(trans, 1.0));
	float base = sdCappedCone(trans2 / 4.0, vec3(4.0, 1.0, 1.0)) * 4.0;
	
	return glass;
}

float sceneSDF(vec3 p) {
	float sdf = min(bunSDF(p), pattySDF(p));
	sdf = min(sdf, counterSDF(p));
	sdf = min(sdf, shakeSDF(p));
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
    vec2 e = vec2(0.0, 0.000001);
    return normalize( vec3( sceneSDF(pos + e.yxx) - sceneSDF(pos - e.yxx),
                            sceneSDF(pos + e.xyx) - sceneSDF(pos - e.xyx),
                            sceneSDF(pos + e.xxy) - sceneSDF(pos - e.xxy)));
}

float rayMarch(vec3 origin, vec3 dir) {
	// distance to march
	float t = 0.01;
	// distance along the ray
	float dist = 0.01;
	for(float i = 0.0; i < max_steps; i++) {
		dist = sceneSDF(vec3(origin + t * dir));
		if(dist < epsilon) {
			return dist;
			// return previously marched distance
			return t;
		}
		// add distance to closest object to t
		t += dist;

		// return if gone too far along ray
		if(t >= max_dist) {
			return -1.0;
		}
	}
	return -1.0;
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
	vec2 pixel = (2.0 * gl_FragCoord.xy - u_Dimensions.xy) / -u_Dimensions.y;
	vec3 rayDir = rayCast(pixel);

	float t = rayMarch(rayDir, cameraPos);
	vec3 normal = getNormal(cameraPos + t * rayDir);
	vec3 point = cameraPos + t * rayDir;
	vec3 lightDir = normalize(lightPos - point);
	// calculate lambertian shading
	float diffuseTerm = dot(normal, normalize(vec3(lightPos - point)));
    diffuseTerm = clamp(diffuseTerm, 0.0, 1.0);
	if(t != -1.0) {
		//out_Col = vec4(diffuseTerm);
		out_Col = vec4(vec3(normal), 1.0);
	} else {
		out_Col = vec4(0.5, 0.0, 0.2, 1.0);
	}
	
}
