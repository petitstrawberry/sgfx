#version 450
layout(set=0,binding=0) uniform sampler2D sampled; layout(location=0) flat in uint instance; layout(location=1) flat in uint valid; layout(location=0) out vec4 color; void main(){ if(uint((gl_FragCoord.x-2.0)*0.75)!=instance)discard; color=valid==1u?texture(sampled,vec2(0.5)):vec4(1,0,1,1); }
