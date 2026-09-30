#version 450
layout(location=0) in vec3 position;
#ifndef COVERAGE
layout(location=1) in vec3 normal;
layout(location=2) in vec4 color;
layout(location=0) out vec3 world;
layout(location=1) out vec3 shadingNormal;
layout(location=2) out vec3 rgb;
#endif
layout(set=0,binding=0,std430) readonly buffer Metadata {vec4 low;vec4 extent;uint invalid;uint clipped[4];uint bitsOffset;uint rankOffset;uint exactOffset;} meta;
layout(set=0,binding=2,std430) readonly buffer Precision {uint words[];} exactData;
layout(push_constant) uniform Camera {vec4 right;vec4 up;vec4 forward;vec4 center;vec4 depth;uvec4 options;} camera;
void main(){
    vec3 p=camera.options.x!=0?meta.low.xyz+meta.extent.xyz*position:position;
    if(camera.options.x!=0&&meta.bitsOffset!=0){uint vertex=uint(gl_VertexIndex),word=exactData.words[meta.bitsOffset+vertex/32],bit=1u<<(vertex%32);
        if((word&bit)!=0){uint rank=exactData.words[meta.rankOffset+vertex/32]+bitCount(word&(bit-1)),offset=meta.exactOffset+rank*3;p=vec3(uintBitsToFloat(exactData.words[offset]),uintBitsToFloat(exactData.words[offset+1]),uintBitsToFloat(exactData.words[offset+2]));}}
    vec3 v=p-camera.center.xyz;float z=camera.forward.w-dot(v,camera.forward.xyz);
    float w=camera.options.y!=0?z:1;
    float depth=camera.options.y!=0?camera.depth.x*z+camera.depth.y:(z-camera.center.w)*camera.depth.z;
    gl_Position=vec4(dot(v,camera.right.xyz)*camera.right.w,dot(v,camera.up.xyz)*camera.up.w,depth,w);
    #ifndef COVERAGE
    world=v*camera.depth.w;shadingNormal=dot(normal,normal)>0?normalize(normal):vec3(0);rgb=color.rgb;
    #endif
}
