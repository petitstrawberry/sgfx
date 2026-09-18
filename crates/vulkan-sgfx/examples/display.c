/* Ordinary Vulkan application: link to the system libvulkan, select an ICD
 * through its standard JSON manifest, and present through VK_KHR_display.
 * No SGFX or SWS symbols are used by this application. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(call) do { VkResult result = (call); if (result != VK_SUCCESS) { \
    fprintf(stderr, "%s: VkResult %d\n", #call, result); return 1; } } while (0)

int main(void) {
    VkApplicationInfo application = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Vulkan display check", .apiVersion = VK_API_VERSION_1_0};
    const char *instance_extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_DISPLAY_EXTENSION_NAME};
    VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &application, .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = instance_extensions};
    VkInstance instance;
    CHECK(vkCreateInstance(&instance_info, NULL, &instance));
    uint32_t count = 0;
    CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
    if (!count) return 1;
    VkPhysicalDevice *physicals = calloc(count, sizeof(*physicals));
    if (!physicals) return 1;
    CHECK(vkEnumeratePhysicalDevices(instance, &count, physicals));
    VkPhysicalDevice physical = physicals[0];
    free(physicals);
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical, &properties);
    printf("Vulkan loader device: %s\n", properties.deviceName);
    CHECK(vkGetPhysicalDeviceDisplayPropertiesKHR(physical, &count, NULL));
    if (!count) return 1;
    VkDisplayPropertiesKHR *displays = calloc(count, sizeof(*displays));
    if (!displays) return 1;
    CHECK(vkGetPhysicalDeviceDisplayPropertiesKHR(physical, &count, displays));
    VkDisplayKHR display = displays[0].display;
    free(displays);
    CHECK(vkGetDisplayModePropertiesKHR(physical, display, &count, NULL));
    if (!count) return 1;
    VkDisplayModePropertiesKHR *modes = calloc(count, sizeof(*modes));
    if (!modes) return 1;
    CHECK(vkGetDisplayModePropertiesKHR(physical, display, &count, modes));
    VkExtent2D extent = modes[0].parameters.visibleRegion;
    VkDisplaySurfaceCreateInfoKHR surface_info = {
        .sType = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
        .displayMode = modes[0].displayMode, .planeIndex = 0,
        .transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .alphaMode = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR, .imageExtent = extent};
    free(modes);
    VkSurfaceKHR surface;
    CHECK(vkCreateDisplayPlaneSurfaceKHR(instance, &surface_info, NULL, &surface));
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
    VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
    if (!families) return 1;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < count; ++i) {
        VkBool32 present;
        CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physical, i, surface, &present));
        if (present && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { family = i; break; }
    }
    free(families);
    if (family == UINT32_MAX) return 1;
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    const char *device_extension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = &device_extension};
    VkDevice device;
    CHECK(vkCreateDevice(physical, &device_info, NULL, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);
    VkSurfaceCapabilitiesKHR capabilities;
    CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &capabilities));
    CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, NULL));
    if (!count) return 1;
    VkSurfaceFormatKHR *formats = calloc(count, sizeof(*formats));
    if (!formats) return 1;
    CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, formats));
    VkSurfaceFormatKHR format = formats[0];
    free(formats);
    uint32_t requested = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount && requested > capabilities.maxImageCount) requested = capabilities.maxImageCount;
    VkSwapchainCreateInfoKHR swapchain_info = {.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface, .minImageCount = requested, .imageFormat = format.format,
        .imageColorSpace = format.colorSpace, .imageExtent = extent, .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR, .clipped = VK_TRUE};
    VkSwapchainKHR swapchain;
    CHECK(vkCreateSwapchainKHR(device, &swapchain_info, NULL, &swapchain));
    CHECK(vkGetSwapchainImagesKHR(device, swapchain, &count, NULL));
    VkImage *images = calloc(count, sizeof(*images));
    VkImageView *views = calloc(count, sizeof(*views));
    VkFramebuffer *framebuffers = calloc(count, sizeof(*framebuffers));
    if (!images || !views || !framebuffers) return 1;
    CHECK(vkGetSwapchainImagesKHR(device, swapchain, &count, images));
    VkAttachmentDescription attachment = {.format = format.format, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};
    VkAttachmentReference reference = {.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1, .pColorAttachments = &reference};
    VkRenderPassCreateInfo pass_info = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &attachment, .subpassCount = 1, .pSubpasses = &subpass};
    VkRenderPass pass;
    CHECK(vkCreateRenderPass(device, &pass_info, NULL, &pass));
    for (uint32_t i = 0; i < count; ++i) {
        VkImageViewCreateInfo view_info = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = images[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format.format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        CHECK(vkCreateImageView(device, &view_info, NULL, &views[i]));
        VkFramebufferCreateInfo framebuffer_info = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = pass, .attachmentCount = 1, .pAttachments = &views[i],
            .width = extent.width, .height = extent.height, .layers = 1};
        CHECK(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffers[i]));
    }
    VkCommandPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = family};
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(device, &pool_info, NULL, &pool));
    VkCommandBufferAllocateInfo allocate = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer command;
    CHECK(vkAllocateCommandBuffers(device, &allocate, &command));
    VkSemaphoreCreateInfo semaphore_info = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore acquired, rendered;
    CHECK(vkCreateSemaphore(device, &semaphore_info, NULL, &acquired));
    CHECK(vkCreateSemaphore(device, &semaphore_info, NULL, &rendered));
    VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(device, &fence_info, NULL, &fence));
    for (uint32_t frame = 0; frame < 60; ++frame) {
        uint32_t index;
        CHECK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &index));
        CHECK(vkResetCommandBuffer(command, 0));
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK(vkBeginCommandBuffer(command, &begin));
        VkClearValue clear = {.color = {.float32 = {0.05f, 0.08f, 0.12f, 1}}};
        clear.color.float32[(frame / 20) % 3] = 0.8f;
        VkRenderPassBeginInfo pass_begin = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = pass, .framebuffer = framebuffers[index], .renderArea = {{0, 0}, extent},
            .clearValueCount = 1, .pClearValues = &clear};
        vkCmdBeginRenderPass(command, &pass_begin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdEndRenderPass(command);
        CHECK(vkEndCommandBuffer(command));
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1, .pWaitSemaphores = &acquired, .pWaitDstStageMask = &wait_stage,
            .commandBufferCount = 1, .pCommandBuffers = &command,
            .signalSemaphoreCount = 1, .pSignalSemaphores = &rendered};
        CHECK(vkQueueSubmit(queue, 1, &submit, fence));
        VkPresentInfoKHR present = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1, .pWaitSemaphores = &rendered, .swapchainCount = 1,
            .pSwapchains = &swapchain, .pImageIndices = &index};
        CHECK(vkQueuePresentKHR(queue, &present));
        CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        CHECK(vkResetFences(device, 1, &fence));
    }
    struct timespec pause = {2, 0};
    nanosleep(&pause, NULL);
    CHECK(vkDeviceWaitIdle(device));
    vkDestroyFence(device, fence, NULL);
    vkDestroySemaphore(device, rendered, NULL);
    vkDestroySemaphore(device, acquired, NULL);
    vkDestroyCommandPool(device, pool, NULL);
    for (uint32_t i = 0; i < count; ++i) {
        vkDestroyFramebuffer(device, framebuffers[i], NULL);
        vkDestroyImageView(device, views[i], NULL);
    }
    free(framebuffers); free(views); free(images);
    vkDestroyRenderPass(device, pass, NULL);
    vkDestroySwapchainKHR(device, swapchain, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    vkDestroyInstance(instance, NULL);
    puts("PASS: system Vulkan loader, KHR_display, 60 presentations and clean shutdown");
    return 0;
}
