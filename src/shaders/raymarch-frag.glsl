#version 300 es

precision highp float;
uniform vec2 u_Dimensions;
uniform float u_Time;

vec3 lightPos = vec3(2.0, -1.0, 2.0);
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
	//p.y -= .03;
	vec3 trans = vec3(transMat(vec3(0.0, -0.63, 1.0 / 3.0)) * vec4(p, 1.0));
	return vec2(udRoundBox(trans / .3, vec3(10.0, .1, 1.7), 1.0) * .3, 2.0);
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
	vec3 trans = vec3(transMat(vec3(0.0, .28, 0.0)) * vec4(tp, 1.0));
	float t1 = sdTorus(trans, vec2(.16, .091));
	vec3 trans2 = vec3(transMat(vec3(0.0, .32, 0.0)) * vec4(tp, 1.0));
	float t2 = sdTorus(trans2, vec2(.08, .1));
	vec3 trans3 = vec3(transMat(vec3(0.0, .35, 0.0)) * vec4(tp, 1.0));
	float t3 = sdTorus(trans3, vec2(.02, .1));
	float sdf = smin(t1, t2, 200.0);
	sdf = min(sdf, t3);
	vec3 trans4 = vec3(transMat(vec3(-0.1, .81, .4)) * vec4(p, 1.0));
	float cutBox = udBox(trans4, vec3(1.0, .53, 1.0));
	sdf = max(cutBox, sdf);
	return vec2(sdf, 3.0);
}

vec2 cherrySDF(vec3 p) {
	p.x += .5;
	vec3 trans = vec3(transMat(vec3(0.0, .45, -0.0)) * vec4(p, 1.0));
	float cherry = sdSphere(trans, .04);
	return vec2(cherry, 4.0);
}

vec2 glassSDF(vec3 p) {
	p.x += .5;
	vec3 trans = vec3(transMat(vec3(0.0, -.3, 0.0)) * vec4(p, 1.0));
	float glass = sdCappedCone(trans / .6, vec3(2.5, 1.1, 1.0)) * .6;
	vec3 rotbase = vec3(rotZ(radians(180.0)) * vec4(p, 1.0));
	vec3 trans2 = vec3(transMat(vec3(0.0, -.02, 0.0)) * vec4(rotbase, 1.0));
	float base = sdCappedCone(trans2 / .3, vec3(3.0, 1.5, 1.0)) * .3;
	vec3 trans3 =  vec3(transMat(vec3(0.0, .3, 0.0)) * vec4(p, 1.0));
	float rim = sdTorus(trans3, vec2(.3, .01));
	float sdf = smin(glass, base, 70.0);
	vec3 trans4 =  vec3(transMat(vec3(0.0, -.3, 0.0)) * vec4(p, 1.0));
	float bottomRim = sdTorus(trans4, vec2(.2, .02));
	sdf =  smin(rim, sdf, 30.0);
	sdf = smin(sdf, bottomRim, 30.0);
	return vec2(sdf, 1.0);
}

vec2 straw(vec3 p) {
	p.x += .5;
	vec3 trans =  vec3(transMat(vec3(-0.1, .3, .0)) * vec4(p, 1.0));
	vec3 rot = vec3(rotZ(radians(-35.0)) * vec4(trans, 1.0));
	float straw = sdCylinder(rot, vec3(.03, .05, .03));
	vec3 trans2 = vec3(transMat(vec3(-0.6, .3, .4)) * vec4(p, 1.0));
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
// subtract first from second material
vec2 opS(vec2 d1, vec2 d2) {
	if(-d1.x > d2.x) {
	//	return vec2(-d1.x, 0.0);
	}
	//return d2;
	return (-d1.x > d2.x) ? -d1 : d2;
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

// 
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
	if(t != -1.0) {
		vec4 color = colorScene(rayDir, t, type);
		out_Col = color;
	} else {
		out_Col = vec4(dark_grey, 1.0);
	}
	
}

float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k )
{
    float res = 1.0;
    for( float t = mint; t < maxt; )
    {
        float h = sceneSDF(ro + rd * t).x;
        if( h < 0.001 )
            return 0.0;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}

// 1 = glass, 2 = counter, 3 = tiles
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

	
	if(type == 1.0) { // glass
		float specularIntensity = max(pow(dot(H, normal), 90.0), 0.0);	
		float translucency = 1.0;
		return vec4(peach * (lightIntensity + (specularIntensity * bright_red * .1)), 1.0);
	} else if (type == 2.0) { // counter
		float specularIntensity = max(pow(dot(H, normal), 40.0), 0.0);
		float fbm = fbm(vec3(t));
		float det = mod(cos(length(vec3(rd.x - .003, rd.yz))) * 60.0 * fbm, 11.0); // does cool smooshy line thing
		//float det = mod(t * 40.0, 11.0);
		vec4 color = vec4(counterCol[int(det)], 1.0); 
		return color * vec4(lightIntensity + (specularIntensity * mint), 1.0);
		//return vec4(silver * (lightIntensity + (specularIntensity * mint)), 1.0);
	} else if (type == 3.0) { // cream
		return vec4(cream * lightIntensity, 1.0);
	} else if (type == 4.0) { // cherry
		float specularIntensity = max(pow(dot(H, normal), 10.0), 0.0);	
		return vec4(cherry * (lightIntensity + specularIntensity * .06), 1.0);
	} else if (type == 5.0) { // straw
		float specularIntensity = max(pow(dot(H, normal), 40.0), 0.0);
		return vec4(mint * (lightIntensity + specularIntensity * .06), 1.0);
	} else if (type == 7.0) { // tiles
		float specularIntensity = max(pow(dot(H, normal), 60.0), 0.0);
		return vec4(cherry * (lightIntensity + (specularIntensity * bright_blue)), 1.0);
	}
}


float mod289(float x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 mod289(vec4 x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 perm(vec4 x){return mod289(((x * 34.0) + 1.0) * x);}

float noise(in vec3 p){
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);

    vec4 b = vec4(a.x, a.x, a.y, a.y) + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + vec4(a.z, a.z, a.z, a.z);
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0);

    vec4 o1 = fract(k3 * (1.0 / 41.0));
    vec4 o2 = fract(k4 * (1.0 / 41.0));

    vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return o4.y * d.y + o4.x * (1.0 - d.y);
}

float fbm(const in vec3 uv)
{
    float a = 0.3;
    float f = 9.0;
    float n = 0.;
    int it = 9;
    for(int i = 0; i < 32; i++)
    {
        if(i<it)
        {
            n += noise(uv*f)*a;
            a *= .5;
            f *= 2.;
        }
    }
    return n;
}