#version 450
layout(location=0) in vec4 position;
layout(location=0) out vec4 color;
void main() { gl_Position=position; gl_PointSize=1.0; color=position; }
