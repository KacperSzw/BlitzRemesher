#version 450
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec4 color;
layout(location=0) out vec3 world;
layout(location=1) out vec3 shadingNormal;
layout(location=2) out vec3 rgb;
layout(set=0,binding=0,std430) readonly buffer Metadata {vec4 low;vec4 extent;uint invalid;} meta;
layout(push_constant) uniform Camera {vec4 right;vec4 up;vec4 forward;vec4 center;vec4 depth;uvec4 options;} camera;
void main(){
    vec3 p=camera.options.x!=0?meta.low.xyz+meta.extent.xyz*position:position;
    vec3 v=p-camera.center.xyz;float z=camera.forward.w-dot(v,camera.forward.xyz);
    float w=camera.options.y!=0?z:1;
    float depth=camera.options.y!=0?camera.depth.x*z+camera.depth.y:(z-camera.center.w)*camera.depth.z;
    gl_Position=vec4(dot(v,camera.right.xyz)*camera.right.w,dot(v,camera.up.xyz)*camera.up.w,depth,w);
    world=v*camera.depth.w;shadingNormal=dot(normal,normal)>0?normalize(normal):vec3(0);rgb=color.rgb;
}
