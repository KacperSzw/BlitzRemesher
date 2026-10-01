#version 450
#ifndef COVERAGE
layout(location=0) in vec3 world;
layout(location=1) in vec3 shadingNormal;
layout(location=2) in vec3 rgb;
#endif
layout(set=0,binding=1,std430) readonly buffer Faces {uint values[];} faces;
layout(push_constant) uniform Camera {vec4 right;vec4 up;vec4 forward;vec4 center;vec4 depth;uvec4 options;} camera;
#ifdef COVERAGE
layout(location=0) out uint coverage;
#else
layout(location=0) out vec4 attributes;
layout(location=1) out vec4 colors;
layout(location=2) out uvec2 witness;
#endif
void main(){
    uint face=faces.values[gl_PrimitiveID];
    if(!gl_FrontFacing&&camera.options.z==0&&(face&65536)==0)discard;
#ifdef COVERAGE
    coverage=1;
#else
    vec3 n=shadingNormal;float l=dot(n,n);vec3 geometric=cross(dFdx(world),dFdy(world));
    // Screen derivatives orient the flat normal consistently with the front
    // face convention, including zero/missing vertex normals.
    if(l>0)n=normalize(n);else n=dot(geometric,geometric)>0?normalize(geometric):vec3(0);
    if(l>0&&!gl_FrontFacing)n=-n;
    attributes=vec4(n,float((face&65535)+1));colors=vec4(rgb,1);
    witness=uvec2(gl_PrimitiveID+1,floatBitsToUint(gl_FragCoord.z));
#endif
}
