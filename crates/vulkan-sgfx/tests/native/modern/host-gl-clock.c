#define GL_SILENCE_DEPRECATION
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
int main(void) {
 CGLPixelFormatAttribute attrs[]={kCGLPFAOpenGLProfile,(CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,kCGLPFAAccelerated,(CGLPixelFormatAttribute)0};
 CGLPixelFormatObj pixel=NULL;GLint count=0;CGLContextObj ctx=NULL;
 CGLError error=CGLChoosePixelFormat(attrs,&pixel,&count);if(error||!pixel){printf("choose=%d\n",error);return 1;}
 error=CGLCreateContext(pixel,NULL,&ctx);CGLDestroyPixelFormat(pixel);if(error){printf("create=%d\n",error);return 1;}
 CGLSetCurrentContext(ctx);
 printf("renderer=%s version=%s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
 GLint bits=0;glGetQueryiv(GL_TIMESTAMP,GL_QUERY_COUNTER_BITS,&bits);printf("timestamp_bits=%d error=%x\n",bits,glGetError());
 GLuint fbo,texture;glGenFramebuffers(1,&fbo);glBindFramebuffer(GL_FRAMEBUFFER,fbo);glGenTextures(1,&texture);glBindTexture(GL_TEXTURE_2D,texture);glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,8,8,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texture,0);
 GLuint q[2];glGenQueries(2,q);glQueryCounter(q[0],GL_TIMESTAMP);glClearColor(1,0,0,1);glClear(GL_COLOR_BUFFER_BIT);glFinish();usleep(10000);glQueryCounter(q[1],GL_TIMESTAMP);glFinish();
 for(int i=0;i<2;i++){GLuint64 value=0;GLint available=0;glGetQueryObjectiv(q[i],GL_QUERY_RESULT_AVAILABLE,&available);glGetQueryObjectui64v(q[i],GL_QUERY_RESULT,&value);printf("query[%d]=%llu available=%d error=%x\n",i,value,available,glGetError());}
 CGLSetCurrentContext(NULL);CGLDestroyContext(ctx);return 0;
}
