#version 450
#extension GL_ARB_shader_draw_parameters : require
layout(location=0) in vec2 pos; layout(location=1) in float record; layout(push_constant) uniform Params { int base; } params; layout(location=0) flat out uint instance; layout(location=1) flat out uint valid; void main(){ instance=uint(gl_InstanceIndex)-2u; valid=(gl_BaseInstanceARB==2 && gl_BaseVertexARB==params.base && gl_DrawIDARB==1 && record==(instance<2u?7.0:8.0))?1u:0u; gl_Position=vec4(pos,0,1); }
