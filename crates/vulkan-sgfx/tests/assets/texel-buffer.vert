#version 450
layout(set=0,binding=4) uniform isamplerBuffer data;
layout(location=0) flat out ivec4 result;
void main(){ result=texelFetch(data,gl_VertexIndex);gl_Position=vec4(0,0,0,1); }
