/* Private queue and proxy wrappers preserve the application's listeners,
 * user data and event queue, including Wine's GPU child wl_surface. */
#define _GNU_SOURCE
#include <wayland-client.h>
#include "scarlet-sgfx-client.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct image_buffer {
    struct wl_buffer *proxy;
    struct image_buffer *next;
    uint32_t index;
    int busy;
};
struct sgfx_wayland {
    struct wl_display *display;
    struct wl_event_queue *queue;
    void *display_wrapper;
    struct wl_surface *surface_wrapper;
    struct wl_registry *registry;
    struct wp_scarlet_sgfx_v1 *factory;
    uint32_t factory_name;
    struct image_buffer *buffers;
    struct wl_callback *frame;
    uint32_t count;
    int lost;
};

static void global(void *data, struct wl_registry *registry, uint32_t name,
                   const char *interface, uint32_t version) {
    struct sgfx_wayland *ctx = data;
    if (!ctx->factory && version >= 1 && !strcmp(interface, "wp_scarlet_sgfx_v1")) {
        ctx->factory = wl_registry_bind(registry, name, &wp_scarlet_sgfx_v1_interface, 1);
        ctx->factory_name = name;
    }
}
static void global_remove(void *data, struct wl_registry *registry, uint32_t name) {
    (void)registry;
    struct sgfx_wayland *ctx = data;
    if (name == ctx->factory_name) ctx->lost = 1;
}
static const struct wl_registry_listener registry_listener = { global, global_remove };
static void release(void *data, struct wl_buffer *buffer) {
    (void)buffer;
    ((struct image_buffer *)data)->busy = 0;
}
static const struct wl_buffer_listener buffer_listener = { release };
static void frame_done(void *data, struct wl_callback *callback, uint32_t time) {
    (void)time;
    struct sgfx_wayland *ctx = data;
    if (ctx->frame == callback) ctx->frame = NULL;
    wl_callback_destroy(callback);
}
static const struct wl_callback_listener frame_listener = { frame_done };

void sgfx_wayland_destroy(struct sgfx_wayland *ctx) {
    if (!ctx) return;
    if (ctx->frame) wl_callback_destroy(ctx->frame);
    struct image_buffer *buffer = ctx->buffers;
    while (buffer) {
        struct image_buffer *next = buffer->next;
        wl_buffer_destroy(buffer->proxy);
        free(buffer);
        buffer = next;
    }
    if (ctx->factory) wp_scarlet_sgfx_v1_destroy(ctx->factory);
    if (ctx->registry) wl_registry_destroy(ctx->registry);
    if (ctx->surface_wrapper) wl_proxy_wrapper_destroy(ctx->surface_wrapper);
    if (ctx->display_wrapper) wl_proxy_wrapper_destroy(ctx->display_wrapper);
    if (ctx->queue) wl_event_queue_destroy(ctx->queue);
    if (ctx->display) wl_display_flush(ctx->display);
    free(ctx);
}

struct sgfx_wayland *sgfx_wayland_create(struct wl_display *display,
                                       struct wl_surface *surface) {
    if (!display) return NULL;
    struct sgfx_wayland *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->display = display;
    ctx->queue = wl_display_create_queue(display);
    ctx->display_wrapper = wl_proxy_create_wrapper(display);
    if (!ctx->queue || !ctx->display_wrapper) goto fail;
    wl_proxy_set_queue(ctx->display_wrapper, ctx->queue);
    ctx->registry = wl_display_get_registry(ctx->display_wrapper);
    if (!ctx->registry || wl_registry_add_listener(ctx->registry, &registry_listener, ctx)) goto fail;
    if (wl_display_roundtrip_queue(display, ctx->queue) < 0 || !ctx->factory || ctx->lost) goto fail;
    if (surface) {
        ctx->surface_wrapper = wl_proxy_create_wrapper(surface);
        if (!ctx->surface_wrapper) goto fail;
        wl_proxy_set_queue((struct wl_proxy *)ctx->surface_wrapper, ctx->queue);
    }
    return ctx;
fail:
    sgfx_wayland_destroy(ctx);
    return NULL;
}

int sgfx_wayland_supported(struct wl_display *display) {
    struct sgfx_wayland *ctx = sgfx_wayland_create(display, NULL);
    if (!ctx) return 0;
    sgfx_wayland_destroy(ctx);
    return 1;
}

int sgfx_wayland_register(struct sgfx_wayland *ctx, int native_handle,
                          uint32_t width, uint32_t height) {
    if (!ctx || ctx->lost || !ctx->surface_wrapper || ctx->count >= 8) return -1;
    struct image_buffer *buffer = calloc(1, sizeof(*buffer));
    if (!buffer) return -1;
    /* The syscall duplicates, rather than consumes, the caller's native handle. */
    int fd = syscall(0x53440000UL, native_handle, O_CLOEXEC);
    if (fd < 0) { free(buffer); return -1; }
    buffer->proxy = wp_scarlet_sgfx_v1_create_buffer(ctx->factory, fd, width, height);
    close(fd); /* libwayland owns the marshalled copy until it flushes. */
    if (!buffer->proxy) { free(buffer); return -1; }
    buffer->index = ctx->count;
    wl_buffer_add_listener(buffer->proxy, &buffer_listener, buffer);
    buffer->next = ctx->buffers;
    ctx->buffers = buffer;
    ctx->count++;
    /* Registration is synchronous: protocol errors must fail swapchain creation,
     * rather than leave an invalid image waiting forever for a release. */
    if (wl_display_roundtrip_queue(ctx->display, ctx->queue) < 0) {
        ctx->lost = 1;
        return -1;
    }
    return (int)buffer->index;
}

int sgfx_wayland_dispatch(struct sgfx_wayland *ctx) {
    if (!ctx || ctx->lost) return -1;
    while (wl_display_prepare_read_queue(ctx->display, ctx->queue) != 0) {
        if (wl_display_dispatch_queue_pending(ctx->display, ctx->queue) < 0) goto fail;
    }
    if (wl_display_flush(ctx->display) < 0 && errno != EAGAIN) {
        wl_display_cancel_read(ctx->display);
        goto fail;
    }
    struct pollfd pollfd = { wl_display_get_fd(ctx->display), POLLIN, 0 };
    int result = poll(&pollfd, 1, 0);
    if (result > 0 && (pollfd.revents & POLLIN)) {
        if (wl_display_read_events(ctx->display) < 0) goto fail;
    } else {
        wl_display_cancel_read(ctx->display);
        if (result < 0 && errno != EINTR) goto fail;
        if (result > 0 && (pollfd.revents & (POLLERR | POLLHUP | POLLNVAL))) goto fail;
    }
    if (wl_display_dispatch_queue_pending(ctx->display, ctx->queue) < 0 || ctx->lost) goto fail;
    return 0;
fail:
    ctx->lost = 1;
    return -1;
}

int sgfx_wayland_available(struct sgfx_wayland *ctx, uint32_t index) {
    if (!ctx || ctx->lost) return 0;
    for (struct image_buffer *b = ctx->buffers; b; b = b->next)
        if (b->index == index) return !b->busy;
    return 0;
}
int sgfx_wayland_present(struct sgfx_wayland *ctx, uint32_t index,
                         uint32_t width, uint32_t height) {
    if (sgfx_wayland_dispatch(ctx) < 0) return -1;
    /* FIFO must not coalesce a previous queued frame into this one. */
    if (ctx->frame) return 1;
    struct image_buffer *buffer = ctx->buffers;
    while (buffer && buffer->index != index) buffer = buffer->next;
    if (!buffer || buffer->busy) return -1;
    ctx->frame = wl_surface_frame(ctx->surface_wrapper);
    if (!ctx->frame) return -1;
    wl_callback_add_listener(ctx->frame, &frame_listener, ctx);
    wl_surface_attach(ctx->surface_wrapper, buffer->proxy, 0, 0);
    if (wl_surface_get_version(ctx->surface_wrapper) >= 4)
        wl_surface_damage_buffer(ctx->surface_wrapper, 0, 0, width, height);
    else wl_surface_damage(ctx->surface_wrapper, 0, 0, width, height);
    wl_surface_commit(ctx->surface_wrapper);
    buffer->busy = 1;
    if (wl_display_flush(ctx->display) < 0 && errno != EAGAIN) {
        ctx->lost = 1;
        return -1;
    }
    return 0;
}
