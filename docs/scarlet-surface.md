# Native SWS surfaces

The native ICD supports `VK_KHR_surface`, `VK_KHR_swapchain` and the experimental
`VK_SGFX_scarlet_surface` instance extension. Use the matching Scarlet
Vulkan-Loader fork; an upstream loader does not recognize this private extension.
Its structure value is private and has not been assigned by Khronos.

`vkCreateScarletSurfaceSGFX` accepts a borrowed SWS window ID. The application
keeps that window alive until every swapchain and its Vulkan surface have been
destroyed. Destroying a Vulkan surface does not destroy the application window.
The create-info C layout is:

```c
struct VkScarletSurfaceCreateInfoSGFX {
    VkStructureType sType; /* (VkStructureType)0x7f530001 */
    const void *pNext;     /* NULL */
    VkFlags flags;        /* 0 */
    uint32_t window;      /* sws_window_create_ex result */
};
```

The function takes `(VkInstance, const VkScarletSurfaceCreateInfoSGFX *,
const VkAllocationCallbacks *, VkSurfaceKHR *)` and returns `VkResult`.
Custom allocation callbacks are currently unsupported by the ICD.
`vkGetPhysicalDeviceScarletPresentationSupportSGFX(VkPhysicalDevice, uint32_t)`
returns `VkBool32` for the queue family.

The native runtime loads `/lib/libsws_client_c.so` through Scarlet's resident
loader. GLFW and the ICD must resolve this same library so window IDs, input
queues, and GPU buffer release notifications share one SWS connection.
SGFX backend discovery continues to use `/lib/sgfx/*.sgfx-driver`.

Surface size follows the SWS window. Resize returns `VK_ERROR_OUT_OF_DATE_KHR`
for the old swapchain; recreate it using the new surface capabilities. Old and
new swapchains keep separate buffer generations and share retirement tracking.
Presentation supports BGRA8 UNORM, FIFO, and opaque composition. The ICD remains
an experimental, non-conformant Vulkan subset; see [vulkan-sgfx.md](vulkan-sgfx.md).
