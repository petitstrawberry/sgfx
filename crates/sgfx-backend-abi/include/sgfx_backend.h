#ifndef SGFX_BACKEND_H
#define SGFX_BACKEND_H
#include <stddef.h>
#include <stdint.h>

/* ABI v2: 64-bit little-endian, in-process, no unwinding across calls.
 * Borrowed spans remain valid until the call returns. Drivers never retain
 * command words or upload pointers; accepted GPU work owns its transport.
 * Objects are destroyed in their creating library, which remains resident.
 * Device/context/session calls are serialized. Receipts may be observed from
 * multiple threads; destroy a receipt only after its last observer returns.
 */
#define SGFX_BACKEND_ABI_VERSION 2
#define SGFX_BACKEND_ENTRY "sgfx_backend_get_api_v2"
enum sgfx_status {
    SGFX_OK = 0, SGFX_INVALID = 1, SGFX_UNSUPPORTED = 2,
    SGFX_OUT_OF_MEMORY = 3, SGFX_DEVICE_LOST = 4,
    SGFX_INITIALIZATION_FAILED = 5, SGFX_BUSY = 6, SGFX_ABI_MISMATCH = 7
};
enum sgfx_disposition { SGFX_REJECTED = 0, SGFX_ACCEPTED = 1, SGFX_PARTIAL = 2 };
enum sgfx_completion { SGFX_PENDING = 0, SGFX_COMPLETE = 1 };
enum sgfx_capability {
    SGFX_RENDERING = 1, SGFX_PRESENTATION = 2, SGFX_IMAGE_UPLOAD = 4,
    SGFX_IMAGE_READBACK = 8, SGFX_DEPTH = 16, SGFX_ASYNC = 32,
    SGFX_PROGRAMMABLE_GRAPHICS = 64, SGFX_TEXTURE_ARRAYS = 128,
    SGFX_DEPTH_SAMPLING = 256, SGFX_IMAGE_MIPS = 512
};
typedef void *sgfx_object;
#define SGFX_CPU_AARCH64_LSE UINT64_C(1)
/* Only set CPU bits confirmed by the initialized process runtime. */
typedef struct { uint32_t size, reserved; uint64_t cpu_features; } sgfx_host_info;
typedef struct { const uint8_t *data; size_t len; } sgfx_bytes;
typedef struct { const uint32_t *data; size_t len; } sgfx_slots;
typedef struct { const uint64_t *data; size_t len; } sgfx_words;
typedef struct { uint32_t x, y, width, height; } sgfx_rect;
typedef struct { uint64_t table; sgfx_words words; size_t count; } sgfx_batch;
typedef struct { uint32_t width, height; int32_t handle; uint32_t reserved; } sgfx_image_info;
typedef struct { uint32_t disposition; int32_t error; sgfx_object receipt; } sgfx_submit_result;

typedef struct sgfx_backend_api {
    uint32_t version, size;
    uint8_t name[64], gpu_backend[32];
    int32_t (*open)(sgfx_bytes path, sgfx_object *out, uint64_t *caps);
    void (*drop_device)(sgfx_object);
    int32_t (*create_context)(sgfx_object, sgfx_object *out);
    void (*drop_context)(sgfx_object);
    int32_t (*create_session)(sgfx_object, sgfx_words metadata, sgfx_slots targets, sgfx_object *out, uint64_t *caps);
    void (*drop_session)(sgfx_object);
    int32_t (*sync_resources)(sgfx_object, sgfx_words metadata);
    /* image transfers one owned duplicate on success; caller closes it. */
    int32_t (*image)(sgfx_object, uint32_t slot, sgfx_image_info *out);
    int32_t (*readback)(sgfx_object, uint32_t slot, uint8_t *out, size_t len, uint32_t stride, sgfx_rect);
    /* import consumes the owned handle on success and on failure. */
    int32_t (*import_bgra)(sgfx_object, uint32_t slot, int32_t handle);
    int32_t (*release_import)(sgfx_object, uint32_t slot);
    int32_t (*execute)(sgfx_object, const sgfx_batch *);
    void (*submit)(sgfx_object, const sgfx_batch *, sgfx_submit_result *out);
    /* UINT64_MAX waits indefinitely; zero polls. */
    int32_t (*wait)(sgfx_object receipt, uint64_t timeout_ns, uint32_t *out);
    void (*drop_receipt)(sgfx_object);
} sgfx_backend_api;

/* Optional low-level resource/queue extension. Negotiate after the main table.
 * Image objects own a library-private reference; map_image retains another.
 * Output pixels/bytes are written directly into the caller's allocation.
 * Resource IDs here are slots validated by the host against its branded table.
 * Device/context/resource/queue/image calls are serialized. Receipt clone and
 * wait are thread-safe. Each clone requires exactly one drop_receipt. */
#define SGFX_DRIVER_ENTRY "sgfx_backend_get_driver_api_v2"
enum sgfx_validation { SGFX_VALIDATE_SHADER = 1, SGFX_VALIDATE_RENDER_PIPELINE = 2, SGFX_VALIDATE_COMPUTE_PIPELINE = 3 };
typedef struct sgfx_driver_api {
    uint32_t version, size;
    int32_t (*create_resources)(sgfx_object context, sgfx_words, sgfx_object *out);
    void (*drop_resources)(sgfx_object);
    int32_t (*sync_resources)(sgfx_object, sgfx_words);
    int32_t (*release_buffer)(sgfx_object, uint32_t slot);
    int32_t (*validate)(sgfx_object, uint32_t kind, uint32_t slot);
    int32_t (*read_buffer)(sgfx_object, uint32_t slot, uint64_t offset, uint8_t *out, size_t len);
    int32_t (*create_image)(sgfx_object context, uint32_t width, uint32_t height, sgfx_object *out, sgfx_image_info *info);
    void (*drop_image)(sgfx_object);
    int32_t (*map_image)(sgfx_object resources, uint32_t slot, sgfx_object image);
    int32_t (*unmap_image)(sgfx_object, uint32_t slot);
    int32_t (*create_queue)(sgfx_object context, sgfx_object *out);
    void (*drop_queue)(sgfx_object);
    void (*submit)(sgfx_object queue, sgfx_object resources, const sgfx_batch *, sgfx_submit_result *out);
    int32_t (*read_texture)(sgfx_object context, sgfx_object resources, uint32_t slot, uint8_t *out, size_t len);
    void (*clone_receipt)(sgfx_object);
    int32_t (*release_texture)(sgfx_object, uint32_t slot);
    int32_t (*release_bind_group)(sgfx_object, uint32_t slot);
} sgfx_driver_api;
typedef int32_t (*sgfx_get_driver_api)(uint32_t version, size_t size, sgfx_driver_api *out);

typedef int32_t (*sgfx_get_api)(uint32_t version, size_t size, const sgfx_host_info *host, sgfx_backend_api *out);
/* A successful call initializes the entire table. An incompatible version or
 * short output buffer returns SGFX_ABI_MISMATCH without writing the table. */
#ifdef __cplusplus
extern "C" {
#endif
int32_t sgfx_backend_get_driver_api_v2(uint32_t version, size_t size, sgfx_driver_api *out);
int32_t sgfx_backend_get_api_v2(uint32_t version, size_t size, const sgfx_host_info *host, sgfx_backend_api *out);
#ifdef __cplusplus
}
#endif

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(void *) == 8, "SGFX ABI v2 requires 64-bit pointers");
_Static_assert(sizeof(sgfx_batch) == 32, "SGFX batch layout");
_Static_assert(sizeof(sgfx_submit_result) == 16, "SGFX receipt layout");
_Static_assert(sizeof(sgfx_driver_api) == 144, "SGFX low-level table layout");
_Static_assert(offsetof(sgfx_driver_api, release_bind_group) == 136, "SGFX low-level table tail");
_Static_assert(sizeof(sgfx_backend_api) == 224, "SGFX function table layout");
_Static_assert(offsetof(sgfx_backend_api, open) == 104, "SGFX function table prefix");
#endif
#endif
