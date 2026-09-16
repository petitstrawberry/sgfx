/* Linux Mesa/VirGL control for the offscreen workload in src/main.rs.
 * This deliberately uses the Linux DRM render node and system EGL/GLES. */
#define _POSIX_C_SOURCE 200809L
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <gbm.h>

#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

struct config {
    unsigned width, height, draws, frames, warmup, uniform_every;
};

static void fail(const char *message) {
    fprintf(stderr, "ERROR: %s\n", message);
    exit(1);
}

static unsigned parse_number(const char *text) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || *end || value > UINT32_MAX) fail("invalid number");
    return (unsigned)value;
}

static struct config parse_args(int argc, char **argv) {
    struct config config = {1280, 800, 1200, 120, 20, 20};
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) fail("missing option value");
        const unsigned value = parse_number(argv[index + 1]);
        if (!strcmp(argv[index], "--width")) config.width = value;
        else if (!strcmp(argv[index], "--height")) config.height = value;
        else if (!strcmp(argv[index], "--draws")) config.draws = value;
        else if (!strcmp(argv[index], "--frames")) config.frames = value;
        else if (!strcmp(argv[index], "--warmup")) config.warmup = value;
        else if (!strcmp(argv[index], "--uniform-every")) config.uniform_every = value;
        else fail("unknown option");
    }
    if (!config.width || !config.height || config.width > 8192 || config.height > 8192 ||
        config.draws > 20000 || !config.frames || config.frames > 10000 ||
        config.warmup > 10000) fail("invalid dimensions or counts");
    return config;
}

static double now_ms(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &time)) fail("clock_gettime failed");
    return (double)time.tv_sec * 1000.0 + (double)time.tv_nsec / 1000000.0;
}

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double percentile(const double *samples, unsigned count, unsigned numerator) {
    double *sorted = malloc((size_t)count * sizeof(*sorted));
    if (!sorted) fail("out of memory");
    memcpy(sorted, samples, (size_t)count * sizeof(*sorted));
    qsort(sorted, count, sizeof(*sorted), compare_double);
    const double result = sorted[(count - 1) * numerator / 100];
    free(sorted);
    return result;
}

static GLuint compile_shader(GLenum stage, const char *source) {
    const GLuint shader = glCreateShader(stage);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char message[4096];
        glGetShaderInfoLog(shader, sizeof(message), NULL, message);
        fprintf(stderr, "shader: %s\n", message);
        fail("shader compilation failed");
    }
    return shader;
}

static GLuint make_program(void) {
    const char *vertex_source =
        "#version 300 es\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    const char *fragment_source =
        "#version 300 es\n"
        "precision highp float;\n"
        "uniform vec4 color;\n"
        "out vec4 output_color;\n"
        "void main() { output_color = color; }\n";
    const GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    const GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char message[4096];
        glGetProgramInfoLog(program, sizeof(message), NULL, message);
        fprintf(stderr, "program: %s\n", message);
        fail("program link failed");
    }
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    return program;
}

int main(int argc, char **argv) {
    const struct config config = parse_args(argc, argv);
    const int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (fd < 0) fail("cannot open /dev/dri/renderD128");
    struct gbm_device *gbm = gbm_create_device(fd);
    if (!gbm) fail("gbm_create_device failed");
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!get_display) fail("EGL platform display extension missing");
    EGLDisplay display = get_display(EGL_PLATFORM_GBM_KHR, gbm, NULL);
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, NULL, NULL))
        fail("EGL GBM display initialization failed");
    if (!eglBindAPI(EGL_OPENGL_ES_API)) fail("EGL OpenGL ES binding failed");
    const EGLint attributes[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_NONE,
    };
    EGLConfig egl_config;
    EGLint count = 0;
    if (!eglChooseConfig(display, attributes, &egl_config, 1, &count) || !count)
        fail("no EGL RGBA8 ES3 window configuration");
    const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    EGLContext context = eglCreateContext(display, egl_config, EGL_NO_CONTEXT, context_attributes);
    if (context == EGL_NO_CONTEXT) fail("EGL context creation failed");
    struct gbm_surface *gbm_surface = gbm_surface_create(
        gbm, config.width, config.height, GBM_FORMAT_ARGB8888, GBM_BO_USE_RENDERING);
    if (!gbm_surface) fail("GBM render surface creation failed");
    EGLSurface surface = eglCreatePlatformWindowSurface(display, egl_config, gbm_surface, NULL);
    if (surface == EGL_NO_SURFACE || !eglMakeCurrent(display, surface, surface, context))
        fail("EGL GBM surface activation failed");
    const char *renderer = (const char *)glGetString(GL_RENDERER);
    if (!renderer || !strstr(renderer, "virgl")) {
        fprintf(stderr, "GL renderer: %s\n", renderer ? renderer : "(none)");
        fail("non-VirGL renderer refused");
    }
    const GLuint program = make_program();
    const GLint color = glGetUniformLocation(program, "color");
    if (color < 0) fail("color uniform not found");
    const GLfloat vertices[] = {-1.f, -1.f, 1.f, -1.f, 0.f, 1.f};
    GLuint buffer = 0;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glFinish();

    double *calls = calloc(config.frames, sizeof(double));
    double *waits = calloc(config.frames, sizeof(double));
    double *totals = calloc(config.frames, sizeof(double));
    if (!calls || !waits || !totals) fail("out of memory");
    printf("RUN backend=mesa-virgl-gles renderer=%s width=%u height=%u draws=%u "
           "uniform_every=%u warmup=%u frames=%u\n",
           renderer, config.width, config.height, config.draws, config.uniform_every,
           config.warmup, config.frames);
    fflush(stdout);
    for (unsigned frame = 0; frame < config.warmup + config.frames; ++frame) {
        const double started = now_ms();
        glViewport(0, 0, config.width, config.height);
        glClearColor(0.05f, 0.08f, 0.12f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        if (config.draws) {
            glUseProgram(program);
            glBindBuffer(GL_ARRAY_BUFFER, buffer);
            glUniform4f(color, 1.f, 1.f, 1.f, 1.f);
            for (unsigned draw = 0; draw < config.draws; ++draw) {
                if (config.uniform_every && draw && draw % config.uniform_every == 0) {
                    const float brightness = (draw / config.uniform_every) % 2 ? 0.6f : 1.f;
                    glUniform4f(color, brightness, brightness, brightness, 1.f);
                }
                glDrawArrays(GL_TRIANGLES, 0, 3);
            }
        }
        const double issued = now_ms();
        glFinish();
        const double completed = now_ms();
        if (glGetError() != GL_NO_ERROR) fail("OpenGL error during benchmark");
        if (frame >= config.warmup) {
            const unsigned index = frame - config.warmup;
            calls[index] = issued - started;
            waits[index] = completed - issued;
            totals[index] = completed - started;
        }
    }
    const size_t pixel_bytes = (size_t)config.width * config.height * 4;
    unsigned char *pixels = malloc(pixel_bytes);
    if (!pixels) fail("out of memory");
    glReadPixels(0, 0, config.width, config.height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    if (glGetError() != GL_NO_ERROR) fail("OpenGL readback failed");
    const unsigned char *edge = pixels +
        ((size_t)(config.height / 2) * config.width) * 4;
    const unsigned char *center = pixels +
        ((size_t)(config.height / 2) * config.width + config.width / 2) * 4;
    if (config.draws && !memcmp(edge, center, 4)) {
        fprintf(stderr, "center=[%u,%u,%u,%u] edge=[%u,%u,%u,%u]\n",
                center[0], center[1], center[2], center[3],
                edge[0], edge[1], edge[2], edge[3]);
        fail("draw did not change center pixel");
    }
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (size_t index = 0; index < pixel_bytes; ++index)
        hash = (hash ^ pixels[index]) * UINT64_C(0x100000001b3);
    double sum = 0;
    for (unsigned index = 0; index < config.frames; ++index) sum += totals[index];
    printf("RESULT backend=mesa-virgl-gles draws=%u frames=%u fps=%.2f "
           "median_ms=%.3f p95_ms=%.3f calls_ms=%.3f finish_ms=%.3f "
           "center=[%u,%u,%u,%u] edge=[%u,%u,%u,%u] fnv64=%016" PRIx64 "\n",
           config.draws, config.frames, 1000.0 * config.frames / sum,
           percentile(totals, config.frames, 50), percentile(totals, config.frames, 95),
           percentile(calls, config.frames, 50), percentile(waits, config.frames, 50),
           center[0], center[1], center[2], center[3],
           edge[0], edge[1], edge[2], edge[3], hash);
    free(pixels);
    free(calls);
    free(waits);
    free(totals);
    glDeleteBuffers(1, &buffer);
    glDeleteProgram(program);
    eglDestroySurface(display, surface);
    eglDestroyContext(display, context);
    eglTerminate(display);
    gbm_surface_destroy(gbm_surface);
    gbm_device_destroy(gbm);
    close(fd);
    return 0;
}
