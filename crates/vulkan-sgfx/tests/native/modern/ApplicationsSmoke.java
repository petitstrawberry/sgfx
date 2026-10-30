import java.nio.*;
import org.lwjgl.*;
import org.lwjgl.system.*;
import org.lwjgl.sdl.SDLVersion;
import org.lwjgl.vulkan.*;
import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.KHRSynchronization2.*;
import static org.lwjgl.vulkan.KHRTimelineSemaphore.*;
import static org.lwjgl.vulkan.EXTHostQueryReset.*;
import static org.lwjgl.vulkan.KHRDynamicRendering.*;
import static org.lwjgl.vulkan.KHRDrawIndirectCount.*;
public class ApplicationsSmoke {
 static boolean timing,statistics;
 static void check(int r,String op){if(r!=VK_SUCCESS)throw new AssertionError(op+" "+r);}
 static long shader(VkDevice d,int kind,String text){
  long compiler=org.lwjgl.util.shaderc.Shaderc.shaderc_compiler_initialize();
  long result=org.lwjgl.util.shaderc.Shaderc.shaderc_compile_into_spv(compiler,text,kind,"smoke.glsl","main",0);
  try(MemoryStack s=MemoryStack.stackPush()){
   if(org.lwjgl.util.shaderc.Shaderc.shaderc_result_get_compilation_status(result)!=0)throw new AssertionError(org.lwjgl.util.shaderc.Shaderc.shaderc_result_get_error_message(result));
   LongBuffer h=s.mallocLong(1);check(vkCreateShaderModule(d,VkShaderModuleCreateInfo.calloc(s).sType$Default().pCode(org.lwjgl.util.shaderc.Shaderc.shaderc_result_get_bytes(result)),null,h),"shader");return h.get(0);
  }finally{org.lwjgl.util.shaderc.Shaderc.shaderc_result_release(result);org.lwjgl.util.shaderc.Shaderc.shaderc_compiler_release(compiler);}
 }
 static long[] buffer(VkDevice d,long size,int usage){try(MemoryStack s=MemoryStack.stackPush()){
  LongBuffer h=s.mallocLong(1);check(vkCreateBuffer(d,VkBufferCreateInfo.calloc(s).sType$Default().size(size).usage(usage).sharingMode(VK_SHARING_MODE_EXCLUSIVE),null,h),"graphics buffer");long b=h.get(0);
  VkMemoryRequirements r=VkMemoryRequirements.calloc(s);vkGetBufferMemoryRequirements(d,b,r);check(vkAllocateMemory(d,VkMemoryAllocateInfo.calloc(s).sType$Default().allocationSize(r.size()).memoryTypeIndex(Integer.numberOfTrailingZeros(r.memoryTypeBits())),null,h),"graphics memory");long m=h.get(0);check(vkBindBufferMemory(d,b,m,0),"graphics bind");return new long[]{b,m};
 }}
 static void graphics(VkDevice d,VkQueue queue,long pool){try(MemoryStack s=MemoryStack.stackPush()){
  LongBuffer h=s.mallocLong(1);PointerBuffer p=s.mallocPointer(1);
  long vs=shader(d,org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_vertex_shader,"#version 450\nlayout(location=0) in vec2 pos; void main(){ gl_Position=vec4(pos,0,1); }");
  long fs=shader(d,org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_fragment_shader,"#version 450\nlayout(location=0) out vec4 color; void main(){ color=vec4(0,1,0,1); }");
  check(vkCreateImage(d,VkImageCreateInfo.calloc(s).sType$Default().imageType(VK_IMAGE_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM).extent(e->e.set(8,8,1)).mipLevels(1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT).tiling(VK_IMAGE_TILING_OPTIMAL).usage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT),null,h),"graphics image");long image=h.get(0);
  VkMemoryRequirements req=VkMemoryRequirements.calloc(s);vkGetImageMemoryRequirements(d,image,req);check(vkAllocateMemory(d,VkMemoryAllocateInfo.calloc(s).sType$Default().allocationSize(req.size()).memoryTypeIndex(Integer.numberOfTrailingZeros(req.memoryTypeBits())),null,h),"image memory");long memory=h.get(0);check(vkBindImageMemory(d,image,memory,0),"image bind");
  check(vkCreateImageView(d,VkImageViewCreateInfo.calloc(s).sType$Default().image(image).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1)),null,h),"image view");long view=h.get(0);
  check(vkCreatePipelineLayout(d,VkPipelineLayoutCreateInfo.calloc(s).sType$Default(),null,h),"pipeline layout");long layout=h.get(0);
  VkPipelineShaderStageCreateInfo.Buffer stages=VkPipelineShaderStageCreateInfo.calloc(2,s);stages.get(0).sType$Default().stage(VK_SHADER_STAGE_VERTEX_BIT).module(vs).pName(s.UTF8("main"));stages.get(1).sType$Default().stage(VK_SHADER_STAGE_FRAGMENT_BIT).module(fs).pName(s.UTF8("main"));
  VkPipelineRenderingCreateInfo formats=VkPipelineRenderingCreateInfo.calloc(s).sType$Default().pColorAttachmentFormats(s.ints(VK_FORMAT_R8G8B8A8_UNORM));
  VkGraphicsPipelineCreateInfo.Buffer info=VkGraphicsPipelineCreateInfo.calloc(1,s).sType$Default().pNext(formats.address()).pStages(stages).layout(layout).renderPass(0).subpass(0)
    .pVertexInputState(VkPipelineVertexInputStateCreateInfo.calloc(s).sType$Default().pVertexBindingDescriptions(VkVertexInputBindingDescription.calloc(1,s).binding(0).stride(8).inputRate(VK_VERTEX_INPUT_RATE_VERTEX)).pVertexAttributeDescriptions(VkVertexInputAttributeDescription.calloc(1,s).location(0).binding(0).format(VK_FORMAT_R32G32_SFLOAT).offset(0)))
    .pInputAssemblyState(VkPipelineInputAssemblyStateCreateInfo.calloc(s).sType$Default().topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST))
    .pViewportState(VkPipelineViewportStateCreateInfo.calloc(s).sType$Default().pViewports(VkViewport.calloc(1,s).x(0).y(0).width(8).height(8).minDepth(0).maxDepth(1)).pScissors(VkRect2D.calloc(1,s).offset(o->o.set(2,2)).extent(e->e.set(4,4))))
    .pRasterizationState(VkPipelineRasterizationStateCreateInfo.calloc(s).sType$Default().polygonMode(VK_POLYGON_MODE_FILL).cullMode(VK_CULL_MODE_NONE).frontFace(VK_FRONT_FACE_COUNTER_CLOCKWISE).lineWidth(1))
    .pMultisampleState(VkPipelineMultisampleStateCreateInfo.calloc(s).sType$Default().rasterizationSamples(VK_SAMPLE_COUNT_1_BIT))
    .pColorBlendState(VkPipelineColorBlendStateCreateInfo.calloc(s).sType$Default().pAttachments(VkPipelineColorBlendAttachmentState.calloc(1,s).colorWriteMask(15)));
  check(vkCreateGraphicsPipelines(d,0,info,null,h),"dynamic pipeline");long pipeline=h.get(0);
  long[] args=buffer(d,64,VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT),counter=buffer(d,4,VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT),vertices=buffer(d,24,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT),indices=buffer(d,8,VK_BUFFER_USAGE_INDEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT),output=buffer(d,512,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  if(timing)check(vkCreateQueryPool(d,VkQueryPoolCreateInfo.calloc(s).sType$Default().queryType(VK_QUERY_TYPE_TIMESTAMP).queryCount(2),null,h),"graphics queries");long queries=timing?h.get(0):0;
  check(vkCreateQueryPool(d,VkQueryPoolCreateInfo.calloc(s).sType$Default().queryType(VK_QUERY_TYPE_OCCLUSION).queryCount(2),null,h),"occlusion pool");long visibility=h.get(0);
  int statMask=VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT|VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT|VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT|VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
  int statResult=vkCreateQueryPool(d,VkQueryPoolCreateInfo.calloc(s).sType$Default().queryType(VK_QUERY_TYPE_PIPELINE_STATISTICS).queryCount(1).pipelineStatistics(statMask),null,h);
  if(statistics)check(statResult,"statistics pool");else if(statResult!=VK_ERROR_FEATURE_NOT_PRESENT)throw new AssertionError("unsupported statistics accepted");long counters=statistics?h.get(0):0;
  for(int indexed=0;indexed<2;indexed++){
   check(vkAllocateCommandBuffers(d,VkCommandBufferAllocateInfo.calloc(s).sType$Default().commandPool(pool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1),p),"graphics commands");VkCommandBuffer cmd=new VkCommandBuffer(p.get(0),d);
   check(vkBeginCommandBuffer(cmd,VkCommandBufferBeginInfo.calloc(s).sType$Default()),"graphics begin");if(timing)vkCmdResetQueryPool(cmd,queries,0,2);vkCmdResetQueryPool(cmd,visibility,0,2);if(statistics){vkCmdResetQueryPool(cmd,counters,0,1);vkCmdBeginQuery(cmd,counters,0,0);}
   vkCmdUpdateBuffer(cmd,vertices[0],0,s.floats(-1,-1,3,-1,-1,3));vkCmdUpdateBuffer(cmd,indices[0],0,s.shorts((short)1,(short)2,(short)3,(short)0));vkCmdUpdateBuffer(cmd,counter[0],0,s.ints(1));vkCmdUpdateBuffer(cmd,args[0],0,indexed==0?s.ints(3,1,0,0,0,0,0,0):s.ints(3,1,0,-1,0,0,0,0,0,0));
   VkImageMemoryBarrier2.Buffer barrier=VkImageMemoryBarrier2.calloc(1,s).sType$Default().oldLayout(indexed==0?VK_IMAGE_LAYOUT_UNDEFINED:VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL).srcStageMask(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR).dstStageMask(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR).srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).image(image).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1));
   vkCmdPipelineBarrier2KHR(cmd,VkDependencyInfo.calloc(s).sType$Default().pImageMemoryBarriers(barrier));
   VkRenderingAttachmentInfo.Buffer attachment=VkRenderingAttachmentInfo.calloc(1,s).sType$Default().imageView(view).imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL).loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR).storeOp(VK_ATTACHMENT_STORE_OP_STORE).clearValue(c->c.color(v->v.float32(0,1).float32(3,1)));
   vkCmdBeginRenderingKHR(cmd,VkRenderingInfo.calloc(s).sType$Default().renderArea(r->r.extent(e->e.set(8,8))).layerCount(1).pColorAttachments(attachment));
   vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);vkCmdBindVertexBuffers(cmd,0,s.longs(vertices[0]),s.longs(0));if(timing)vkCmdWriteTimestamp2KHR(cmd,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,queries,0);
   vkCmdBeginQuery(cmd,visibility,0,0);
   if(indexed==0)vkCmdDrawIndirectCountKHR(cmd,args[0],0,counter[0],0,2,16);else{vkCmdBindIndexBuffer(cmd,indices[0],0,VK_INDEX_TYPE_UINT16);vkCmdDrawIndexedIndirectCountKHR(cmd,args[0],0,counter[0],0,2,20);}
   vkCmdEndQuery(cmd,visibility,0);vkCmdBeginQuery(cmd,visibility,1,0);vkCmdEndQuery(cmd,visibility,1);
   if(timing)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,queries,1);vkCmdEndRenderingKHR(cmd);if(statistics)vkCmdEndQuery(cmd,counters,0);
   barrier.oldLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL).srcStageMask(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR).dstStageMask(VK_PIPELINE_STAGE_2_COPY_BIT_KHR).srcAccessMask(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT_KHR).dstAccessMask(VK_ACCESS_2_TRANSFER_READ_BIT_KHR);vkCmdPipelineBarrier2KHR(cmd,VkDependencyInfo.calloc(s).sType$Default().pImageMemoryBarriers(barrier));
   vkCmdCopyImageToBuffer(cmd,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,output[0],VkBufferImageCopy.calloc(1,s).imageSubresource(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1)).imageExtent(e->e.set(8,8,1)));
   if(timing)vkCmdCopyQueryPoolResults(cmd,queries,0,2,output[0],256,24,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
   vkCmdCopyQueryPoolResults(cmd,visibility,0,2,output[0],320,24,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
   check(vkEndCommandBuffer(cmd),"graphics end");check(vkQueueSubmit2KHR(queue,VkSubmitInfo2.calloc(1,s).sType$Default().pCommandBufferInfos(VkCommandBufferSubmitInfo.calloc(1,s).sType$Default().commandBuffer(cmd)),0),"graphics submit");check(vkQueueWaitIdle(queue),"graphics idle");
   check(vkMapMemory(d,output[1],0,512,0,p),"graphics readback");ByteBuffer pixels=MemoryUtil.memByteBuffer(p.get(0),512).order(ByteOrder.LITTLE_ENDIAN);
   for(int y=0;y<8;y++)for(int x=0;x<8;x++){int expected=x>=2&&x<6&&y>=2&&y<6?0xff00ff00:0xff0000ff;if(pixels.getInt((y*8+x)*4)!=expected)throw new AssertionError("pixel "+x+","+y+" indexed="+indexed+" value="+Integer.toHexString(pixels.getInt((y*8+x)*4)));}
   if(timing&&(pixels.getLong(280)<=pixels.getLong(256)||pixels.getLong(264)!=1||pixels.getLong(288)!=1))throw new AssertionError("graphics query copy "+pixels.getLong(256)+".."+pixels.getLong(280));
   if(pixels.getLong(320)==0||pixels.getLong(328)!=1||pixels.getLong(344)!=0||pixels.getLong(352)!=1)throw new AssertionError("occlusion visible/empty "+pixels.getLong(320)+","+pixels.getLong(344));
   LongBuffer samples=s.mallocLong(4);check(vkGetQueryPoolResults(d,visibility,0,2,samples,16,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),"occlusion host results");if(samples.get(0)!=pixels.getLong(320)||samples.get(1)!=1||samples.get(2)!=0||samples.get(3)!=1)throw new AssertionError("occlusion copy mismatch");
   vkResetQueryPoolEXT(d,visibility,0,1);if(vkGetQueryPoolResults(d,visibility,0,1,samples,16,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)!=VK_NOT_READY)throw new AssertionError("occlusion host reset");
   if(statistics){LongBuffer stats=s.mallocLong(5);check(vkGetQueryPoolResults(d,counters,0,1,stats,40,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT|VK_QUERY_RESULT_WAIT_BIT),"statistics results");if(stats.get(0)!=3||stats.get(1)!=1||stats.get(2)==0||stats.get(3)==0||stats.get(4)!=1)throw new AssertionError("statistics values");}
   vkUnmapMemory(d,output[1]);vkFreeCommandBuffers(d,pool,cmd);
  }
  vkDestroyQueryPool(d,visibility,null);if(statistics)vkDestroyQueryPool(d,counters,null);
  if(timing)vkDestroyQueryPool(d,queries,null);for(long[] b:new long[][]{args,counter,vertices,indices,output}){vkDestroyBuffer(d,b[0],null);vkFreeMemory(d,b[1],null);}
  vkDestroyPipeline(d,pipeline,null);vkDestroyPipelineLayout(d,layout,null);vkDestroyImageView(d,view,null);vkDestroyImage(d,image,null);vkFreeMemory(d,memory,null);vkDestroyShaderModule(d,vs,null);vkDestroyShaderModule(d,fs,null);
  System.out.println("SCOPED_QUERIES_NATIVE PASS occlusion visible empty copy host-reset statistics_supported="+statistics);
  System.out.println("MODERN_GRAPHICS_NATIVE PASS dynamic rendering GPU-written indirect count indexed baseVertex=-1 pixels=128 timestamp_supported="+timing+"");
 }}

 public static void main(String[] args)throws Exception{
  if(SDLVersion.SDL_GetVersion()<3004000)throw new AssertionError("SDL dependency loading");
  VK.create("/lib/libvulkan.so.1");
  try(MemoryStack s=MemoryStack.stackPush()){
   PointerBuffer p=s.mallocPointer(1);LongBuffer h=s.mallocLong(1);IntBuffer n=s.mallocInt(1);
   VkInstanceCreateInfo ii=VkInstanceCreateInfo.calloc(s).sType$Default().pApplicationInfo(VkApplicationInfo.calloc(s).sType$Default().apiVersion(VK_MAKE_VERSION(1,1,0)));
   check(vkCreateInstance(ii,null,p),"instance");VkInstance instance=new VkInstance(p.get(0),ii);
   check(vkEnumeratePhysicalDevices(instance,n,null),"adapter count");PointerBuffer adapters=s.mallocPointer(n.get(0));check(vkEnumeratePhysicalDevices(instance,n,adapters),"adapters");VkPhysicalDevice physical=new VkPhysicalDevice(adapters.get(0),instance);
   VkPhysicalDeviceSynchronization2Features support=VkPhysicalDeviceSynchronization2Features.calloc(s).sType$Default();
   VK11.vkGetPhysicalDeviceFeatures2(physical,VkPhysicalDeviceFeatures2.calloc(s).sType$Default().pNext(support));
   if(!support.synchronization2())throw new AssertionError("sync2 query");
   VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeature=VkPhysicalDeviceTimelineSemaphoreFeatures.calloc(s).sType$Default().timelineSemaphore(true);
   vkGetPhysicalDeviceQueueFamilyProperties(physical,n,null);VkQueueFamilyProperties.Buffer families=VkQueueFamilyProperties.calloc(n.get(0),s);vkGetPhysicalDeviceQueueFamilyProperties(physical,n,families);timing=families.get(0).timestampValidBits()!=0;
   VkPhysicalDeviceFeatures hardware=VkPhysicalDeviceFeatures.calloc(s);vkGetPhysicalDeviceFeatures(physical,hardware);statistics=hardware.pipelineStatisticsQuery();
   VkPhysicalDeviceHostQueryResetFeatures hostReset=VkPhysicalDeviceHostQueryResetFeatures.calloc(s).sType$Default().hostQueryReset(true);timelineFeature.pNext(hostReset.address());support.pNext(timelineFeature.address());
   VkDeviceQueueCreateInfo.Buffer queues=VkDeviceQueueCreateInfo.calloc(1,s).sType$Default().queueFamilyIndex(0).pQueuePriorities(s.floats(1));
   VkPhysicalDeviceDynamicRenderingFeatures rendering=VkPhysicalDeviceDynamicRenderingFeatures.calloc(s).sType$Default().dynamicRendering(true);hostReset.pNext(rendering.address());
   VkDeviceCreateInfo di=VkDeviceCreateInfo.calloc(s).sType$Default().pNext(support.address()).pQueueCreateInfos(queues).pEnabledFeatures(VkPhysicalDeviceFeatures.calloc(s).multiDrawIndirect(true).drawIndirectFirstInstance(true).pipelineStatisticsQuery(statistics)).ppEnabledExtensionNames(s.pointers(s.UTF8(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME),s.UTF8(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME),s.UTF8(VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME),s.UTF8(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME),s.UTF8(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME)));
   check(vkCreateDevice(physical,di,null,p),"device");VkDevice device=new VkDevice(p.get(0),physical,di);vkGetDeviceQueue(device,0,0,p);VkQueue queue=new VkQueue(p.get(0),device);
   long[] buffers=new long[2],memories=new long[2];
   for(int i=0;i<2;i++){
    check(vkCreateBuffer(device,VkBufferCreateInfo.calloc(s).sType$Default().size(256).usage(VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT),null,h),"buffer");buffers[i]=h.get(0);
    VkMemoryRequirements req=VkMemoryRequirements.calloc(s);vkGetBufferMemoryRequirements(device,buffers[i],req);
    check(vkAllocateMemory(device,VkMemoryAllocateInfo.calloc(s).sType$Default().allocationSize(req.size()).memoryTypeIndex(0),null,h),"memory");memories[i]=h.get(0);check(vkBindBufferMemory(device,buffers[i],memories[i],0),"bind");
   }
   check(vkCreateCommandPool(device,VkCommandPoolCreateInfo.calloc(s).sType$Default().queueFamilyIndex(0),null,h),"pool");long pool=h.get(0);
   check(vkAllocateCommandBuffers(device,VkCommandBufferAllocateInfo.calloc(s).sType$Default().commandPool(pool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1),p),"commands");VkCommandBuffer command=new VkCommandBuffer(p.get(0),device);
   check(vkCreateEvent(device,VkEventCreateInfo.calloc(s).sType$Default().flags(VK_EVENT_CREATE_DEVICE_ONLY_BIT_KHR),null,h),"event");long event=h.get(0);
   check(vkCreateSemaphore(device,VkSemaphoreCreateInfo.calloc(s).sType$Default().pNext(VkSemaphoreTypeCreateInfo.calloc(s).sType$Default().semaphoreType(VK_SEMAPHORE_TYPE_TIMELINE_KHR)),null,h),"timeline");long timeline=h.get(0);
   check(vkCreateSemaphore(device,VkSemaphoreCreateInfo.calloc(s).sType$Default(),null,h),"binary");long binary=h.get(0);
   check(vkCreateFence(device,VkFenceCreateInfo.calloc(s).sType$Default(),null,h),"fence");long fence=h.get(0);
   int queryResult=vkCreateQueryPool(device,VkQueryPoolCreateInfo.calloc(s).sType$Default().queryType(VK_QUERY_TYPE_TIMESTAMP).queryCount(2),null,h);if(timing)check(queryResult,"query pool");else if(queryResult!=VK_ERROR_FEATURE_NOT_PRESENT)throw new AssertionError("unavailable timer advertised");long query=timing?h.get(0):0;
   if(timing)vkResetQueryPoolEXT(device,query,0,2);
   check(vkBeginCommandBuffer(command,VkCommandBufferBeginInfo.calloc(s).sType$Default()),"begin");
   if(timing){vkCmdResetQueryPool(command,query,0,2);vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,query,0);}
   vkCmdFillBuffer(command,buffers[0],0,256,0x12345678);
   VkBufferMemoryBarrier2.Buffer barrier=VkBufferMemoryBarrier2.calloc(1,s).sType$Default().srcStageMask(VK_PIPELINE_STAGE_2_CLEAR_BIT_KHR).srcAccessMask(VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR).dstStageMask(VK_PIPELINE_STAGE_2_COPY_BIT_KHR).dstAccessMask(VK_ACCESS_2_TRANSFER_READ_BIT_KHR).srcQueueFamilyIndex(0).dstQueueFamilyIndex(0).buffer(buffers[0]).offset(4).size(16);
   VkDependencyInfo dependency=VkDependencyInfo.calloc(s).sType$Default().pBufferMemoryBarriers(barrier);
   vkCmdSetEvent2KHR(command,event,dependency);vkCmdWaitEvents2KHR(command,s.longs(event),VkDependencyInfo.calloc(1,s).sType$Default().pBufferMemoryBarriers(barrier));
   vkCmdCopyBuffer(command,buffers[0],buffers[1],VkBufferCopy.calloc(1,s).size(256));
   VkBufferMemoryBarrier2.Buffer host=VkBufferMemoryBarrier2.calloc(1,s).sType$Default().srcStageMask(VK_PIPELINE_STAGE_2_COPY_BIT_KHR).srcAccessMask(VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR).dstStageMask(VK_PIPELINE_STAGE_2_HOST_BIT_KHR).dstAccessMask(VK_ACCESS_2_HOST_READ_BIT_KHR).srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).buffer(buffers[1]).size(VK_WHOLE_SIZE);
   vkCmdPipelineBarrier2KHR(command,VkDependencyInfo.calloc(s).sType$Default().pBufferMemoryBarriers(host));vkCmdResetEvent2KHR(command,event,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR);if(timing)vkCmdWriteTimestamp2KHR(command,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,query,1);check(vkEndCommandBuffer(command),"end");
   VkSemaphoreSubmitInfo.Buffer wait=VkSemaphoreSubmitInfo.calloc(1,s).sType$Default().semaphore(timeline).value(1).stageMask(VK_PIPELINE_STAGE_2_COPY_BIT_KHR);
   VkSemaphoreSubmitInfo.Buffer firstSignal=VkSemaphoreSubmitInfo.calloc(1,s).sType$Default().semaphore(binary).value(73).stageMask(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR);
   VkSemaphoreSubmitInfo.Buffer secondWait=VkSemaphoreSubmitInfo.calloc(1,s).sType$Default().semaphore(binary).value(999).stageMask(VK_PIPELINE_STAGE_2_NONE_KHR);
   VkSemaphoreSubmitInfo.Buffer last=VkSemaphoreSubmitInfo.calloc(1,s).sType$Default().semaphore(timeline).value(2).stageMask(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR);
   VkSubmitInfo2.Buffer submits=VkSubmitInfo2.calloc(2,s);
   submits.get(0).sType$Default().pWaitSemaphoreInfos(wait).pCommandBufferInfos(VkCommandBufferSubmitInfo.calloc(1,s).sType$Default().commandBuffer(command).deviceMask(0)).pSignalSemaphoreInfos(firstSignal);
   submits.get(1).sType$Default().pWaitSemaphoreInfos(secondWait).pSignalSemaphoreInfos(last);
   check(vkQueueSubmit2KHR(queue,submits,fence),"submit2");
   if(vkGetFenceStatus(device,fence)!=VK_NOT_READY)throw new AssertionError("premature fence");
   check(vkSignalSemaphoreKHR(device,VkSemaphoreSignalInfo.calloc(s).sType$Default().semaphore(timeline).value(1)),"host signal");
   check(vkWaitForFences(device,fence,true,5_000_000_000L),"wait fence");
   check(vkGetSemaphoreCounterValueKHR(device,timeline,h),"counter");if(h.get(0)!=2)throw new AssertionError("timeline retirement");
   check(vkMapMemory(device,memories[1],0,256,0,p),"map result");IntBuffer data=MemoryUtil.memIntBuffer(p.get(0),64);for(int i=0;i<64;i++)if(data.get(i)!=0x12345678)throw new AssertionError("GPU result index="+i+" value="+data.get(i));vkUnmapMemory(device,memories[1]);
   if(timing){LongBuffer stamps=s.mallocLong(4);int queryFlags=VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT|VK_QUERY_RESULT_WAIT_BIT;
   check(vkGetQueryPoolResults(device,query,0,2,stamps,16,queryFlags),"query results");
   if(stamps.get(0)<=0 || stamps.get(2)<stamps.get(0) || stamps.get(1)!=1 || stamps.get(3)!=1)throw new AssertionError("GPU timestamps "+stamps.get(0)+","+stamps.get(1)+","+stamps.get(2)+","+stamps.get(3));
   System.out.println("QUERY_NATIVE PASS timestamp_ns="+stamps.get(0)+".."+stamps.get(2));
   vkResetQueryPoolEXT(device,query,0,1);if(vkGetQueryPoolResults(device,query,0,1,stamps,16,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)!=VK_NOT_READY)throw new AssertionError("host reset availability");
   vkDestroyQueryPool(device,query,null);
   }else System.out.println("QUERY_NATIVE_UNAVAILABLE PASS timestampValidBits=0 query pool rejected");
   graphics(device,queue,pool);
   vkDestroyFence(device,fence,null);vkDestroySemaphore(device,binary,null);vkDestroySemaphore(device,timeline,null);vkDestroyEvent(device,event,null);vkDestroyCommandPool(device,pool,null);
   for(int i=0;i<2;i++){vkDestroyBuffer(device,buffers[i],null);vkFreeMemory(device,memories[i],null);}
   vkDestroyDevice(device,null);vkDestroyInstance(instance,null);VK.destroy();
   System.out.println("SYNC2_NATIVE PASS GPU copy bytes=256 events mixed semaphores timeline host gate fence");
  }
 }
}
