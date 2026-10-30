#version 450
layout(location=0) out float color;
void main() { color=gl_FragCoord.x; gl_FragDepth=0.375; }
