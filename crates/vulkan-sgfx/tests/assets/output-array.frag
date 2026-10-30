#version 450
layout(location=0) out vec4 colors[2];
void main(){ colors[0]=vec4(-2,3,0.5,1); colors[1]=vec4(gl_FragCoord.xy,2,1); }
