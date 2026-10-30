import java.nio.*;
import org.lwjgl.*;
import org.lwjgl.system.*;
import org.lwjgl.sdl.SDLVersion;
import org.lwjgl.vulkan.*;
import org.lwjgl.util.vma.*;
import static org.lwjgl.util.vma.Vma.*;
import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.KHRSynchronization2.*;
import static org.lwjgl.vulkan.KHRTimelineSemaphore.*;
import static org.lwjgl.vulkan.EXTHostQueryReset.*;
import static org.lwjgl.vulkan.KHRDynamicRendering.*;
import static org.lwjgl.vulkan.KHRDrawIndirectCount.*;
public class ApplicationsSmoke {
 static boolean timing,statistics,compute;
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
 static void vmaAllocations(VkDevice d,VkQueue q,long pool){try(MemoryStack s=MemoryStack.stackPush()){
  PointerBuffer p=s.mallocPointer(1),allocation=s.mallocPointer(1);LongBuffer h=s.mallocLong(1);
  VkInstance instance=d.getPhysicalDevice().getInstance();
  check(vmaCreateAllocator(VmaAllocatorCreateInfo.calloc(s).instance(instance).physicalDevice(d.getPhysicalDevice()).device(d).vulkanApiVersion(VK_MAKE_VERSION(1,2,0)).pVulkanFunctions(VmaVulkanFunctions.calloc(s).set(instance,d)),p),"VMA allocator");long allocator=p.get(0);
  final long bytes=64L*1024*1024;
  check(vmaCreateBuffer(allocator,VkBufferCreateInfo.calloc(s).sType$Default().size(bytes).usage(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT),VmaAllocationCreateInfo.calloc(s).usage(VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE).flags(VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT),h,allocation,null),"VMA dedicated 64MiB buffer");long source=h.get(0),memory=allocation.get(0);
  VmaAllocationInfo details=VmaAllocationInfo.calloc(s);vmaGetAllocationInfo(allocator,memory,details);
  check(vkCreateBuffer(d,VkBufferCreateInfo.calloc(s).sType$Default().size(4).usage(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT),null,h),"dedicated alias candidate");long alias=h.get(0);
  if(vkBindBufferMemory(d,alias,details.deviceMemory(),0)!=VK_ERROR_INITIALIZATION_FAILED)throw new AssertionError("dedicated buffer allocation reused by a different buffer");vkDestroyBuffer(d,alias,null);
  long[] output=buffer(d,8,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  check(vkAllocateCommandBuffers(d,VkCommandBufferAllocateInfo.calloc(s).sType$Default().commandPool(pool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1),p),"VMA GPU commands");VkCommandBuffer cmd=new VkCommandBuffer(p.get(0),d);check(vkBeginCommandBuffer(cmd,VkCommandBufferBeginInfo.calloc(s).sType$Default()),"VMA begin");
  vkCmdUpdateBuffer(cmd,source,0,s.ints(0x13579bdf));vkCmdUpdateBuffer(cmd,source,bytes-4,s.ints(0x2468ace0));
  vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,VkMemoryBarrier.calloc(1,s).sType$Default().srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT),null,null);
  vkCmdCopyBuffer(cmd,source,output[0],VkBufferCopy.calloc(2,s).apply(0,r->r.srcOffset(0).dstOffset(0).size(4)).apply(1,r->r.srcOffset(bytes-4).dstOffset(4).size(4)));
  check(vkEndCommandBuffer(cmd),"VMA end");check(vkQueueSubmit(q,VkSubmitInfo.calloc(s).sType$Default().pCommandBuffers(s.pointers(cmd)),0),"VMA submit");check(vkQueueWaitIdle(q),"VMA idle");
  check(vkMapMemory(d,output[1],0,8,0,p),"VMA result map");IntBuffer got=MemoryUtil.memByteBuffer(p.get(0),8).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();if(got.get(0)!=0x13579bdf||got.get(1)!=0x2468ace0)throw new AssertionError("VMA GPU boundary copy");vkUnmapMemory(d,output[1]);
  vkFreeCommandBuffers(d,pool,cmd);vkDestroyBuffer(d,output[0],null);vkFreeMemory(d,output[1],null);vmaDestroyBuffer(allocator,source,memory);vmaDestroyAllocator(allocator);
  System.out.println("VMA_DEDICATED_NATIVE PASS actual VMA dedicated 64MiB allocation GPU boundary updates and copies");
 }}
 static void compute(VkDevice d,VkQueue queue,long pool){
  if(!compute){System.out.println("COMPUTE_NATIVE_UNAVAILABLE PASS graphics-only queue retained after isolated dispatch probe");return;}
  try(MemoryStack s=MemoryStack.stackPush()){
   LongBuffer h=s.mallocLong(1);PointerBuffer p=s.mallocPointer(1);
   long[] data=buffer(d,64,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   check(vkMapMemory(d,data[1],0,64,0,p),"compute initialize");ByteBuffer bytes=MemoryUtil.memByteBuffer(p.get(0),64).order(ByteOrder.LITTLE_ENDIAN);for(int i=0;i<16;i++)bytes.putInt(i*4,13);vkUnmapMemory(d,data[1]);
   VkDescriptorSetLayoutBinding.Buffer bindings=VkDescriptorSetLayoutBinding.calloc(1,s).binding(0).descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
   check(vkCreateDescriptorSetLayout(d,VkDescriptorSetLayoutCreateInfo.calloc(s).sType$Default().pBindings(bindings),null,h),"compute descriptor layout");long setLayout=h.get(0);
   check(vkCreatePipelineLayout(d,VkPipelineLayoutCreateInfo.calloc(s).sType$Default().pSetLayouts(s.longs(setLayout)),null,h),"compute pipeline layout");long layout=h.get(0);
   long module=shader(d,org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_compute_shader,"#version 450\nlayout(local_size_x=4) in;layout(set=0,binding=0,std430) buffer Data { uint values[]; };void main(){uint i=gl_GlobalInvocationID.x;values[i]=values[i]*3u+i*7u+6699u;}");
   VkPipelineShaderStageCreateInfo stage=VkPipelineShaderStageCreateInfo.calloc(s).sType$Default().stage(VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(s.UTF8("main"));
   check(vkCreateComputePipelines(d,0,VkComputePipelineCreateInfo.calloc(1,s).sType$Default().stage(stage).layout(layout),null,h),"compute pipeline");long pipeline=h.get(0);
   check(vkCreateDescriptorPool(d,VkDescriptorPoolCreateInfo.calloc(s).sType$Default().maxSets(1).pPoolSizes(VkDescriptorPoolSize.calloc(1,s).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)),null,h),"compute descriptor pool");long descriptorPool=h.get(0);
   check(vkAllocateDescriptorSets(d,VkDescriptorSetAllocateInfo.calloc(s).sType$Default().descriptorPool(descriptorPool).pSetLayouts(s.longs(setLayout)),h),"compute descriptor set");long set=h.get(0);
   vkUpdateDescriptorSets(d,VkWriteDescriptorSet.calloc(1,s).sType$Default().dstSet(set).dstBinding(0).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).pBufferInfo(VkDescriptorBufferInfo.calloc(1,s).buffer(data[0]).offset(16).range(32)),null);
   check(vkAllocateCommandBuffers(d,VkCommandBufferAllocateInfo.calloc(s).sType$Default().commandPool(pool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1),p),"compute commands");VkCommandBuffer cmd=new VkCommandBuffer(p.get(0),d);
   check(vkBeginCommandBuffer(cmd,VkCommandBufferBeginInfo.calloc(s).sType$Default()),"compute begin");vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,s.longs(set),null);vkCmdDispatch(cmd,2,1,1);
   vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,VkMemoryBarrier.calloc(1,s).sType$Default().srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT).dstAccessMask(VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT),null,null);vkCmdDispatch(cmd,2,1,1);
   check(vkEndCommandBuffer(cmd),"compute end");check(vkQueueSubmit(queue,VkSubmitInfo.calloc(s).sType$Default().pCommandBuffers(s.pointers(cmd)),0),"compute submit");check(vkQueueWaitIdle(queue),"compute idle");
   check(vkMapMemory(d,data[1],0,64,0,p),"compute readback");bytes=MemoryUtil.memByteBuffer(p.get(0),64).order(ByteOrder.LITTLE_ENDIAN);
   for(int i=0;i<16;i++){int expected=i>=4&&i<12?(13*3+(i-4)*7+6699)*3+(i-4)*7+6699:13;if(bytes.getInt(i*4)!=expected)throw new AssertionError("compute GPU word "+i+" "+bytes.getInt(i*4)+" expected="+expected);}
   vkUnmapMemory(d,data[1]);vkFreeCommandBuffers(d,pool,cmd);vkDestroyDescriptorPool(d,descriptorPool,null);vkDestroyPipeline(d,pipeline,null);vkDestroyShaderModule(d,module,null);vkDestroyPipelineLayout(d,layout,null);vkDestroyDescriptorSetLayout(d,setLayout,null);vkDestroyBuffer(d,data[0],null);vkFreeMemory(d,data[1],null);
   System.out.println("COMPUTE_NATIVE PASS native SSBO GPU read-modify-write dispatches=2 offset=16 bytes=32 sentinels=8");
  }
 }
 static void graphics(VkDevice d,VkQueue queue,long pool){try(MemoryStack s=MemoryStack.stackPush()){
  LongBuffer h=s.mallocLong(1);PointerBuffer p=s.mallocPointer(1);
  long vs=shader(d,org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_vertex_shader,"#version 450\n#extension GL_ARB_shader_draw_parameters : require\nlayout(set=0,binding=1) uniform isamplerBuffer words; layout(location=0) in vec2 pos; layout(location=1) in float record; layout(location=2) in vec4 normal; layout(location=3) in ivec2 signedPair; layout(location=4) in ivec3 wide; layout(push_constant) uniform Params { int base; } params; layout(location=0) flat out uint instance; layout(location=1) flat out uint valid; void main(){ instance=uint(gl_InstanceIndex)-2u; valid=(wide==ivec3(-2147483647-1,1234,2147483647) && (texelFetch(words,0)==ivec4(-9,0,0,1) || (texelFetch(words,0)==ivec4(-128,0,0,1) && texelFetch(words,1).x==127 && texelFetch(words,2).x==-9 && texelFetch(words,3).x==3 && texelFetch(words,4).x==-1 && texelFetch(words,5).x==64 && texelFetch(words,6).x==-64)) && gl_BaseInstanceARB==2 && gl_BaseVertexARB==params.base && gl_DrawIDARB==1 && record==(instance<2u?7.0:8.0) && abs(normal.x+1.0)<0.001 && abs(normal.y-1.0)<0.001 && abs(normal.z+64.0/127.0)<0.001 && abs(normal.w-64.0/127.0)<0.001 && signedPair==(instance<2u?ivec2(-32768,1234):ivec2(32767,-9)))?1u:0u; gl_Position=vec4(pos,0,1); }");
  long fs=shader(d,org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_fragment_shader,"#version 450\nlayout(set=0,binding=0) uniform sampler2D sampled; layout(location=0) flat in uint instance; layout(location=1) flat in uint valid; layout(location=0) out vec4 color; void main(){ if(uint((gl_FragCoord.x-2.0)*0.75)!=instance)discard; float a=degrees(radians(gl_FragCoord.x)); float values[4]=float[4](1,2,3,4); uint i=uint(gl_FragCoord.x)&3u; values[i]=values[(i+1u)&3u]+7.0; gl_FragDepth=0.375; bool ok=abs(a-gl_FragCoord.x)<0.001 && values[i]==float(((i+1u)&3u)+1u)+7.0; color=valid==1u && ok?textureGrad(sampled,vec2(0.5),vec2(0.125,0),vec2(0,0.125)):vec4(1,0,1,1); }");
  long pointVs=shader(d,org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_vertex_shader,"#version 450\nlayout(location=0) flat out uint instance; layout(location=1) flat out uint valid; void main(){uint x=uint(gl_VertexIndex)%4u,y=uint(gl_VertexIndex)/4u; instance=uint((float(x)+0.5)*0.75); valid=1u; gl_Position=vec4((float(x)+2.5)/4.0-1.0,1.0-(float(y)+2.5)/4.0,0,1); gl_PointSize=1.0; }");
  check(vkCreateImage(d,VkImageCreateInfo.calloc(s).sType$Default().imageType(VK_IMAGE_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM).extent(e->e.set(8,8,1)).mipLevels(4).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT).tiling(VK_IMAGE_TILING_OPTIMAL).usage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT),null,h),"graphics image");long image=h.get(0);
  VkMemoryRequirements req=VkMemoryRequirements.calloc(s);vkGetImageMemoryRequirements(d,image,req);check(vkAllocateMemory(d,VkMemoryAllocateInfo.calloc(s).sType$Default().allocationSize(req.size()).memoryTypeIndex(Integer.numberOfTrailingZeros(req.memoryTypeBits())),null,h),"image memory");long memory=h.get(0);check(vkBindImageMemory(d,image,memory,0),"image bind");
  check(vkCreateImageView(d,VkImageViewCreateInfo.calloc(s).sType$Default().image(image).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1)),null,h),"image view");long view=h.get(0);
  check(vkCreateImage(d,VkImageCreateInfo.calloc(s).sType$Default().imageType(VK_IMAGE_TYPE_2D).format(VK_FORMAT_D32_SFLOAT).extent(e->e.set(8,8,1)).mipLevels(1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT).tiling(VK_IMAGE_TILING_OPTIMAL).usage(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT),null,h),"fragment depth image");long depth=h.get(0);
  vkGetImageMemoryRequirements(d,depth,req);check(vkAllocateMemory(d,VkMemoryAllocateInfo.calloc(s).sType$Default().allocationSize(req.size()).memoryTypeIndex(Integer.numberOfTrailingZeros(req.memoryTypeBits())),null,h),"fragment depth memory");long depthMemory=h.get(0);check(vkBindImageMemory(d,depth,depthMemory,0),"fragment depth bind");
  check(vkCreateImageView(d,VkImageViewCreateInfo.calloc(s).sType$Default().image(depth).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_D32_SFLOAT).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).levelCount(1).layerCount(1)),null,h),"fragment depth view");long depthView=h.get(0);
  check(vkCreateImage(d,VkImageCreateInfo.calloc(s).sType$Default().imageType(VK_IMAGE_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM).extent(e->e.set(1,1,1)).mipLevels(1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT).tiling(VK_IMAGE_TILING_OPTIMAL).usage(VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT),null,h),"sample image");long texture=h.get(0);
  vkGetImageMemoryRequirements(d,texture,req);check(vkAllocateMemory(d,VkMemoryAllocateInfo.calloc(s).sType$Default().allocationSize(req.size()).memoryTypeIndex(Integer.numberOfTrailingZeros(req.memoryTypeBits())),null,h),"sample memory");long textureMemory=h.get(0);check(vkBindImageMemory(d,texture,textureMemory,0),"sample bind");
  check(vkCreateImageView(d,VkImageViewCreateInfo.calloc(s).sType$Default().image(texture).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1)),null,h),"sample view");long textureView=h.get(0);
  check(vkCreateSampler(d,VkSamplerCreateInfo.calloc(s).sType$Default().minFilter(VK_FILTER_LINEAR).magFilter(VK_FILTER_LINEAR).mipmapMode(VK_SAMPLER_MIPMAP_MODE_LINEAR).anisotropyEnable(true).maxAnisotropy(16),null,h),"anisotropic sampler");long sampler=h.get(0);
  check(vkCreateDescriptorSetLayout(d,VkDescriptorSetLayoutCreateInfo.calloc(s).sType$Default().flags(KHRPushDescriptor.VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR).pBindings(VkDescriptorSetLayoutBinding.calloc(2,s).apply(0,b->b.binding(0).descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).stageFlags(VK_SHADER_STAGE_FRAGMENT_BIT)).apply(1,b->b.binding(1).descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER).stageFlags(VK_SHADER_STAGE_VERTEX_BIT))),null,h),"push descriptor layout");long pushLayout=h.get(0);
  check(vkCreatePipelineLayout(d,VkPipelineLayoutCreateInfo.calloc(s).sType$Default().pSetLayouts(s.longs(pushLayout)).pPushConstantRanges(VkPushConstantRange.calloc(1,s).stageFlags(VK_SHADER_STAGE_VERTEX_BIT).size(4)),null,h),"pipeline layout");long layout=h.get(0);
  VkPipelineShaderStageCreateInfo.Buffer stages=VkPipelineShaderStageCreateInfo.calloc(2,s);stages.get(0).sType$Default().stage(VK_SHADER_STAGE_VERTEX_BIT).module(vs).pName(s.UTF8("main"));stages.get(1).sType$Default().stage(VK_SHADER_STAGE_FRAGMENT_BIT).module(fs).pName(s.UTF8("main"));
  VkPipelineRenderingCreateInfo formats=VkPipelineRenderingCreateInfo.calloc(s).sType$Default().pColorAttachmentFormats(s.ints(VK_FORMAT_R8G8B8A8_UNORM));
  VkGraphicsPipelineCreateInfo.Buffer info=VkGraphicsPipelineCreateInfo.calloc(1,s).sType$Default().pNext(formats.address()).pStages(stages).layout(layout).renderPass(0).subpass(0)
    .pVertexInputState(VkPipelineVertexInputStateCreateInfo.calloc(s).sType$Default().pNext(VkPipelineVertexInputDivisorStateCreateInfoEXT.calloc(s).sType$Default().pVertexBindingDivisors(VkVertexInputBindingDivisorDescriptionEXT.calloc(1,s).binding(1).divisor(2)).address()).pVertexBindingDescriptions(VkVertexInputBindingDescription.calloc(2,s).apply(0,b->b.binding(0).stride(8).inputRate(VK_VERTEX_INPUT_RATE_VERTEX)).apply(1,b->b.binding(1).stride(24).inputRate(VK_VERTEX_INPUT_RATE_INSTANCE))).pVertexAttributeDescriptions(VkVertexInputAttributeDescription.calloc(5,s).apply(0,a->a.location(0).binding(0).format(VK_FORMAT_R32G32_SFLOAT).offset(0)).apply(1,a->a.location(1).binding(1).format(VK_FORMAT_R32_SFLOAT).offset(0)).apply(2,a->a.location(2).binding(1).format(VK_FORMAT_R8G8B8A8_SNORM).offset(4)).apply(3,a->a.location(3).binding(1).format(VK_FORMAT_R16G16_SINT).offset(8)).apply(4,a->a.location(4).binding(1).format(VK_FORMAT_R32G32B32_SINT).offset(12))))
    .pInputAssemblyState(VkPipelineInputAssemblyStateCreateInfo.calloc(s).sType$Default().topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST))
    .pViewportState(VkPipelineViewportStateCreateInfo.calloc(s).sType$Default().pViewports(VkViewport.calloc(1,s).x(0).y(0).width(8).height(8).minDepth(0).maxDepth(1)).pScissors(VkRect2D.calloc(1,s).offset(o->o.set(2,2)).extent(e->e.set(4,4))))
    .pRasterizationState(VkPipelineRasterizationStateCreateInfo.calloc(s).sType$Default().polygonMode(VK_POLYGON_MODE_FILL).cullMode(VK_CULL_MODE_NONE).frontFace(VK_FRONT_FACE_COUNTER_CLOCKWISE).lineWidth(1))
    .pMultisampleState(VkPipelineMultisampleStateCreateInfo.calloc(s).sType$Default().rasterizationSamples(VK_SAMPLE_COUNT_1_BIT))
    .pColorBlendState(VkPipelineColorBlendStateCreateInfo.calloc(s).sType$Default().pAttachments(VkPipelineColorBlendAttachmentState.calloc(1,s).colorWriteMask(15)));
  formats.depthAttachmentFormat(VK_FORMAT_D32_SFLOAT);info.pDepthStencilState(VkPipelineDepthStencilStateCreateInfo.calloc(s).sType$Default().depthTestEnable(true).depthWriteEnable(true).depthCompareOp(VK_COMPARE_OP_ALWAYS));
  long[] pipelines=new long[5];
  for(int mode=0;mode<5;mode++){
   VkPipelineColorBlendAttachmentState color=info.pColorBlendState().pAttachments().get(0);color.blendEnable(mode==2||mode==3).colorBlendOp(mode==2?VK_BLEND_OP_MIN:mode==3?VK_BLEND_OP_MAX:VK_BLEND_OP_ADD).alphaBlendOp(mode==2?VK_BLEND_OP_MIN:mode==3?VK_BLEND_OP_MAX:VK_BLEND_OP_ADD);
   if(mode==4){stages.get(0).module(pointVs);info.pVertexInputState(VkPipelineVertexInputStateCreateInfo.calloc(s).sType$Default());info.pInputAssemblyState().topology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);}
   check(vkCreateGraphicsPipelines(d,0,info,null,h),"dynamic pipeline mode="+mode);pipelines[mode]=h.get(0);
  }
  long[] args=buffer(d,64,VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT),counter=buffer(d,4,VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT),vertices=buffer(d,64L*1024*1024,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT),indices=buffer(d,8,VK_BUFFER_USAGE_INDEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT),output=buffer(d,768,VK_BUFFER_USAGE_TRANSFER_DST_BIT),instances=buffer(d,96,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  long[] texels=buffer(d,64,VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  check(vkCreateBufferView(d,VkBufferViewCreateInfo.calloc(s).sType$Default().buffer(texels[0]).format(VK_FORMAT_R32_SINT).offset(16).range(32),null,h),"signed texel buffer view");long texelView=h.get(0);
  check(vkCreateBufferView(d,VkBufferViewCreateInfo.calloc(s).sType$Default().buffer(texels[0]).format(VK_FORMAT_R8_SINT).offset(20).range(7),null,h),"signed byte texel view");long byteView=h.get(0);
  if(timing)check(vkCreateQueryPool(d,VkQueryPoolCreateInfo.calloc(s).sType$Default().queryType(VK_QUERY_TYPE_TIMESTAMP).queryCount(2),null,h),"graphics queries");long queries=timing?h.get(0):0;
  check(vkCreateQueryPool(d,VkQueryPoolCreateInfo.calloc(s).sType$Default().queryType(VK_QUERY_TYPE_OCCLUSION).queryCount(2),null,h),"occlusion pool");long visibility=h.get(0);
  int statMask=VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT|VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT|VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT|VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
  int statResult=vkCreateQueryPool(d,VkQueryPoolCreateInfo.calloc(s).sType$Default().queryType(VK_QUERY_TYPE_PIPELINE_STATISTICS).queryCount(1).pipelineStatistics(statMask),null,h);
  if(statistics)check(statResult,"statistics pool");else if(statResult!=VK_ERROR_FEATURE_NOT_PRESENT)throw new AssertionError("unsupported statistics accepted");long counters=statistics?h.get(0):0;
  for(int indexed=0;indexed<7;indexed++){
   final long activeTexelView=indexed==5?byteView:texelView;
   check(vkAllocateCommandBuffers(d,VkCommandBufferAllocateInfo.calloc(s).sType$Default().commandPool(pool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1),p),"graphics commands");VkCommandBuffer cmd=new VkCommandBuffer(p.get(0),d);
   check(vkBeginCommandBuffer(cmd,VkCommandBufferBeginInfo.calloc(s).sType$Default()),"graphics begin");if(timing)vkCmdResetQueryPool(cmd,queries,0,2);vkCmdResetQueryPool(cmd,visibility,0,2);if(statistics){vkCmdResetQueryPool(cmd,counters,0,1);vkCmdBeginQuery(cmd,counters,0,0);}
   vkCmdUpdateBuffer(cmd,texels[0],0,s.ints(99,99,99,99,-9,0x03f77f80,0x00c040ff,31,47,51,63,77,88,88,88,88));vkCmdUpdateBuffer(cmd,vertices[0],64L*1024*1024-24,s.floats(-1,-1,3,-1,-1,3));vkCmdUpdateBuffer(cmd,indices[0],0,s.shorts((short)1,(short)2,(short)3,(short)0));vkCmdUpdateBuffer(cmd,instances[0],0,s.ints(Float.floatToRawIntBits(99),0,0,0,0,0,Float.floatToRawIntBits(99),0,0,0,0,0,Float.floatToRawIntBits(7),0x40c07f80,0x04d28000,Integer.MIN_VALUE,1234,Integer.MAX_VALUE,Float.floatToRawIntBits(8),0x40c07f80,0xfff77fff,Integer.MIN_VALUE,1234,Integer.MAX_VALUE));vkCmdUpdateBuffer(cmd,counter[0],0,s.ints(2));vkCmdUpdateBuffer(cmd,args[0],0,indexed!=1?s.ints(0,0,0,0,indexed==4?16:3,3,0,2):s.ints(0,0,0,0,0,3,3,0,-1,2));
   if(indexed==0){
    VkImageMemoryBarrier2.Buffer sampleBarrier=VkImageMemoryBarrier2.calloc(1,s).sType$Default().oldLayout(VK_IMAGE_LAYOUT_UNDEFINED).newLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL).srcStageMask(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR).dstStageMask(VK_PIPELINE_STAGE_2_COPY_BIT_KHR).dstAccessMask(VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR).srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).image(texture).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1));
    vkCmdPipelineBarrier2KHR(cmd,VkDependencyInfo.calloc(s).sType$Default().pImageMemoryBarriers(sampleBarrier));
    vkCmdClearColorImage(cmd,texture,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VkClearColorValue.calloc(s).float32(1,1).float32(3,1),VkImageSubresourceRange.calloc(1,s).aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1));
    sampleBarrier.oldLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL).srcAccessMask(VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR).dstAccessMask(VK_ACCESS_2_SHADER_SAMPLED_READ_BIT_KHR).srcStageMask(VK_PIPELINE_STAGE_2_COPY_BIT_KHR).dstStageMask(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT_KHR);vkCmdPipelineBarrier2KHR(cmd,VkDependencyInfo.calloc(s).sType$Default().pImageMemoryBarriers(sampleBarrier));
   }
   VkImageMemoryBarrier2.Buffer barrier=VkImageMemoryBarrier2.calloc(1,s).sType$Default().oldLayout(indexed==0?VK_IMAGE_LAYOUT_UNDEFINED:VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL).srcStageMask(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR).dstStageMask(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR).srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).image(image).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1));
   vkCmdPipelineBarrier2KHR(cmd,VkDependencyInfo.calloc(s).sType$Default().pImageMemoryBarriers(barrier));
   VkImageMemoryBarrier2.Buffer depthBarrier=VkImageMemoryBarrier2.calloc(1,s).sType$Default().oldLayout(indexed==0?VK_IMAGE_LAYOUT_UNDEFINED:VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL).srcStageMask(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR).dstStageMask(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR).srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).image(depth).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).levelCount(1).layerCount(1));vkCmdPipelineBarrier2KHR(cmd,VkDependencyInfo.calloc(s).sType$Default().pImageMemoryBarriers(depthBarrier));
   VkRenderingAttachmentInfo depthAttachment=VkRenderingAttachmentInfo.calloc(s).sType$Default().imageView(depthView).imageLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL).loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR).storeOp(VK_ATTACHMENT_STORE_OP_STORE).clearValue(c->c.depthStencil(z->z.depth(1)));
   VkRenderingAttachmentInfo.Buffer attachment=VkRenderingAttachmentInfo.calloc(1,s).sType$Default().imageView(view).imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL).loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR).storeOp(VK_ATTACHMENT_STORE_OP_STORE).clearValue(c->c.color(v->v.float32(0,1).float32(3,1)));
   vkCmdBeginRenderingKHR(cmd,VkRenderingInfo.calloc(s).sType$Default().renderArea(r->r.extent(e->e.set(8,8))).layerCount(1).pColorAttachments(attachment).pDepthAttachment(depthAttachment));
   vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipelines[indexed>=5?0:indexed]);vkCmdBindVertexBuffers(cmd,0,s.longs(vertices[0],instances[0]),s.longs(64L*1024*1024-24,0));vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_VERTEX_BIT,0,s.ints(indexed==1?-1:0));KHRPushDescriptor.vkCmdPushDescriptorSetKHR(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,layout,0,VkWriteDescriptorSet.calloc(2,s).apply(0,w->w.sType$Default().dstBinding(0).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1).pImageInfo(VkDescriptorImageInfo.calloc(1,s).sampler(sampler).imageView(textureView).imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))).apply(1,w->w.sType$Default().dstBinding(1).descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER).descriptorCount(1).pTexelBufferView(s.longs(activeTexelView))));if(timing)vkCmdWriteTimestamp2KHR(cmd,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,queries,0);
   vkCmdBeginQuery(cmd,visibility,0,0);
   if(indexed!=1)vkCmdDrawIndirectCountKHR(cmd,args[0],0,counter[0],0,2,16);else{vkCmdBindIndexBuffer(cmd,indices[0],0,VK_INDEX_TYPE_UINT16);vkCmdDrawIndexedIndirectCountKHR(cmd,args[0],0,counter[0],0,2,20);}
   vkCmdEndQuery(cmd,visibility,0);vkCmdBeginQuery(cmd,visibility,1,0);vkCmdEndQuery(cmd,visibility,1);
   if(timing)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,queries,1);vkCmdEndRenderingKHR(cmd);if(statistics)vkCmdEndQuery(cmd,counters,0);
   barrier.oldLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL).srcStageMask(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR).dstStageMask(VK_PIPELINE_STAGE_2_COPY_BIT_KHR).srcAccessMask(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT_KHR).dstAccessMask(VK_ACCESS_2_TRANSFER_READ_BIT_KHR);vkCmdPipelineBarrier2KHR(cmd,VkDependencyInfo.calloc(s).sType$Default().pImageMemoryBarriers(barrier));
   vkCmdCopyImageToBuffer(cmd,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,output[0],VkBufferImageCopy.calloc(1,s).imageSubresource(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1)).imageExtent(e->e.set(8,8,1)));
   depthBarrier.oldLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL).srcAccessMask(VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT_KHR).dstAccessMask(VK_ACCESS_2_TRANSFER_READ_BIT_KHR);vkCmdPipelineBarrier2KHR(cmd,VkDependencyInfo.calloc(s).sType$Default().pImageMemoryBarriers(depthBarrier));
   vkCmdCopyImageToBuffer(cmd,depth,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,output[0],VkBufferImageCopy.calloc(1,s).bufferOffset(512).imageSubresource(r->r.aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).layerCount(1)).imageExtent(e->e.set(8,8,1)));
   if(timing)vkCmdCopyQueryPoolResults(cmd,queries,0,2,output[0],256,24,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
   vkCmdCopyQueryPoolResults(cmd,visibility,0,2,output[0],320,24,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
   check(vkEndCommandBuffer(cmd),"graphics end");check(vkQueueSubmit2KHR(queue,VkSubmitInfo2.calloc(1,s).sType$Default().pCommandBufferInfos(VkCommandBufferSubmitInfo.calloc(1,s).sType$Default().commandBuffer(cmd)),0),"graphics submit");check(vkQueueWaitIdle(queue),"graphics idle");
   check(vkMapMemory(d,output[1],0,768,0,p),"graphics readback");ByteBuffer pixels=MemoryUtil.memByteBuffer(p.get(0),768).order(ByteOrder.LITTLE_ENDIAN);
   for(int y=0;y<8;y++)for(int x=0;x<8;x++){int expected=x>=2&&x<6&&y>=2&&y<6?(indexed==2?0xff000000:indexed==3?0xff00ffff:0xff00ff00):0xff0000ff;if(pixels.getInt((y*8+x)*4)!=expected)throw new AssertionError("pixel "+x+","+y+" indexed="+indexed+" value="+Integer.toHexString(pixels.getInt((y*8+x)*4)));}
   for(int y=0;y<8;y++)for(int x=0;x<8;x++){float expected=x>=2&&x<6&&y>=2&&y<6?0.375f:1.0f;float value=pixels.getFloat(512+(y*8+x)*4);if(value!=expected)throw new AssertionError("fragment depth "+x+","+y+" mode="+indexed+" got="+value);}
   if(timing&&(pixels.getLong(280)<=pixels.getLong(256)||pixels.getLong(264)!=1||pixels.getLong(288)!=1))throw new AssertionError("graphics query copy "+pixels.getLong(256)+".."+pixels.getLong(280));
   if(pixels.getLong(320)==0||pixels.getLong(328)!=1||pixels.getLong(344)!=0||pixels.getLong(352)!=1)throw new AssertionError("occlusion visible/empty "+pixels.getLong(320)+","+pixels.getLong(344));
   LongBuffer samples=s.mallocLong(4);check(vkGetQueryPoolResults(d,visibility,0,2,samples,16,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),"occlusion host results");if(samples.get(0)!=pixels.getLong(320)||samples.get(1)!=1||samples.get(2)!=0||samples.get(3)!=1)throw new AssertionError("occlusion copy mismatch");
   vkResetQueryPoolEXT(d,visibility,0,1);if(vkGetQueryPoolResults(d,visibility,0,1,samples,16,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)!=VK_NOT_READY)throw new AssertionError("occlusion host reset");
   if(statistics){LongBuffer stats=s.mallocLong(5);check(vkGetQueryPoolResults(d,counters,0,1,stats,40,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT|VK_QUERY_RESULT_WAIT_BIT),"statistics results");if(stats.get(0)!=(indexed==4?48:9)||stats.get(1)!=(indexed==4?48:3)||stats.get(2)==0||stats.get(3)==0||stats.get(4)!=1)throw new AssertionError("statistics values");}
   vkUnmapMemory(d,output[1]);vkFreeCommandBuffers(d,pool,cmd);
  }
  vkDestroyQueryPool(d,visibility,null);if(statistics)vkDestroyQueryPool(d,counters,null);
  if(timing)vkDestroyQueryPool(d,queries,null);for(long[] b:new long[][]{args,counter,vertices,indices,output,instances}){vkDestroyBuffer(d,b[0],null);vkFreeMemory(d,b[1],null);}
  vkDestroyBufferView(d,byteView,null);vkDestroyBufferView(d,texelView,null);vkDestroyBuffer(d,texels[0],null);vkFreeMemory(d,texels[1],null);
  vkDestroySampler(d,sampler,null);vkDestroyImageView(d,textureView,null);vkDestroyImage(d,texture,null);vkFreeMemory(d,textureMemory,null);vkDestroyDescriptorSetLayout(d,pushLayout,null);
  for(long pipeline:pipelines)vkDestroyPipeline(d,pipeline,null);vkDestroyImageView(d,depthView,null);vkDestroyImage(d,depth,null);vkFreeMemory(d,depthMemory,null);vkDestroyShaderModule(d,pointVs,null);vkDestroyPipelineLayout(d,layout,null);vkDestroyImageView(d,view,null);vkDestroyImage(d,image,null);vkFreeMemory(d,memory,null);vkDestroyShaderModule(d,vs,null);vkDestroyShaderModule(d,fs,null);
  System.out.println("SCOPED_QUERIES_NATIVE PASS occlusion visible empty copy host-reset statistics_supported="+statistics);
  System.out.println("VULKAN_REQUIREMENTS_NATIVE PASS push descriptors anisotropy16 divisor2 firstInstance2 BaseVertex signed DrawIndex1 after zero-count entry");
  System.out.println("MODERN_GRAPHICS_NATIVE PASS dynamic rendering GPU-written indirect count indexed baseVertex=-1 pixels=448 depth_words=448 vertex_backing=64MiB points MinMax angles dynamic-stores gradients packed-inputs signed-texel-buffer R32 offset16 R8 offset20 range7 R32-restored mip-zero-render timestamp_supported="+timing+"");
 }}

 static void depthTransfers(VkDevice d,VkQueue q,long pool){try(MemoryStack s=MemoryStack.stackPush()){
  LongBuffer h=s.mallocLong(1);PointerBuffer p=s.mallocPointer(1);long[] image=new long[2],memory=new long[2];
  for(int i=0;i<2;i++){
   check(vkCreateImage(d,VkImageCreateInfo.calloc(s).sType$Default().imageType(VK_IMAGE_TYPE_2D).format(VK_FORMAT_D32_SFLOAT).extent(e->e.set(4,4,1)).mipLevels(1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT).tiling(VK_IMAGE_TILING_OPTIMAL).usage(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT),null,h),"depth image");image[i]=h.get(0);
   VkMemoryRequirements r=VkMemoryRequirements.calloc(s);vkGetImageMemoryRequirements(d,image[i],r);check(vkAllocateMemory(d,VkMemoryAllocateInfo.calloc(s).sType$Default().allocationSize(r.size()).memoryTypeIndex(Integer.numberOfTrailingZeros(r.memoryTypeBits())),null,h),"depth memory");memory[i]=h.get(0);check(vkBindImageMemory(d,image[i],memory[i],0),"depth bind");
  }
  check(vkCreateImageView(d,VkImageViewCreateInfo.calloc(s).sType$Default().image(image[1]).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_D32_SFLOAT).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).levelCount(1).layerCount(1)),null,h),"depth-only view");long view=h.get(0);
  long vs=shader(d,org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_vertex_shader,"#version 450\nvoid main(){vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex],0.5,1);}");
  check(vkCreatePipelineLayout(d,VkPipelineLayoutCreateInfo.calloc(s).sType$Default(),null,h),"depth-only layout");long layout=h.get(0);
  VkGraphicsPipelineCreateInfo.Buffer info=VkGraphicsPipelineCreateInfo.calloc(1,s).sType$Default().pNext(VkPipelineRenderingCreateInfo.calloc(s).sType$Default().depthAttachmentFormat(VK_FORMAT_D32_SFLOAT).address()).layout(layout).pStages(VkPipelineShaderStageCreateInfo.calloc(1,s).sType$Default().stage(VK_SHADER_STAGE_VERTEX_BIT).module(vs).pName(s.UTF8("main")))
   .pVertexInputState(VkPipelineVertexInputStateCreateInfo.calloc(s).sType$Default()).pInputAssemblyState(VkPipelineInputAssemblyStateCreateInfo.calloc(s).sType$Default().topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST))
   .pViewportState(VkPipelineViewportStateCreateInfo.calloc(s).sType$Default().pViewports(VkViewport.calloc(1,s).width(4).height(4).minDepth(0).maxDepth(1)).pScissors(VkRect2D.calloc(1,s).extent(e->e.set(2,2))))
   .pRasterizationState(VkPipelineRasterizationStateCreateInfo.calloc(s).sType$Default().polygonMode(VK_POLYGON_MODE_FILL).lineWidth(1).depthBiasEnable(true).depthBiasConstantFactor(10)).pMultisampleState(VkPipelineMultisampleStateCreateInfo.calloc(s).sType$Default().rasterizationSamples(VK_SAMPLE_COUNT_1_BIT))
   .pColorBlendState(VkPipelineColorBlendStateCreateInfo.calloc(s).sType$Default()).pDepthStencilState(VkPipelineDepthStencilStateCreateInfo.calloc(s).sType$Default().depthTestEnable(true).depthWriteEnable(true).depthCompareOp(VK_COMPARE_OP_ALWAYS));
  check(vkCreateGraphicsPipelines(d,0,info,null,h),"vertex-only depth pipeline");long pipeline=h.get(0);
  long[] src=buffer(d,64,VK_BUFFER_USAGE_TRANSFER_SRC_BIT),dst=buffer(d,128,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  check(vkMapMemory(d,src[1],0,64,0,p),"depth upload map");FloatBuffer data=MemoryUtil.memByteBuffer(p.get(0),64).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();for(int i=0;i<16;i++)data.put(i,i/16.0f);vkUnmapMemory(d,src[1]);
  for(int clear=0;clear<3;clear++){
   check(vkAllocateCommandBuffers(d,VkCommandBufferAllocateInfo.calloc(s).sType$Default().commandPool(pool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1),p),"depth command");VkCommandBuffer c=new VkCommandBuffer(p.get(0),d);check(vkBeginCommandBuffer(c,VkCommandBufferBeginInfo.calloc(s).sType$Default()),"depth begin");
   VkImageMemoryBarrier.Buffer b=VkImageMemoryBarrier.calloc(clear==0?2:1,s);for(int i=0;i<b.capacity();i++)b.get(i).sType$Default().oldLayout(clear==0?VK_IMAGE_LAYOUT_UNDEFINED:VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL).newLayout(clear==2?VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL).image(image[clear==0?i:1]).srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).levelCount(1).layerCount(1));
   vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,null,null,b);
   VkBufferImageCopy.Buffer region=VkBufferImageCopy.calloc(1,s).imageSubresource(r->r.aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).layerCount(1)).imageExtent(e->e.set(4,4,1));
   if(clear==0){
    vkCmdCopyBufferToImage(c,src[0],image[0],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,region);b.limit(1);b.get(0).oldLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL).srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT);vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,null,null,b);
    vkCmdCopyImageToBuffer(c,image[0],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,dst[0],region);region.bufferOffset(64);
    vkCmdCopyImage(c,image[0],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,image[1],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VkImageCopy.calloc(1,s).srcSubresource(r->r.aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).layerCount(1)).dstSubresource(r->r.aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).layerCount(1)).extent(e->e.set(4,4,1)));
   }else if(clear==2){
    vkCmdBeginRenderingKHR(c,VkRenderingInfo.calloc(s).sType$Default().renderArea(r->r.extent(e->e.set(4,4))).layerCount(1).pDepthAttachment(VkRenderingAttachmentInfo.calloc(s).sType$Default().imageView(view).imageLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL).loadOp(VK_ATTACHMENT_LOAD_OP_LOAD).storeOp(VK_ATTACHMENT_STORE_OP_STORE)));
    vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);vkCmdDraw(c,3,1,0,0);vkCmdEndRenderingKHR(c);
   }else vkCmdClearDepthStencilImage(c,image[1],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VkClearDepthStencilValue.calloc(s).depth(0.625f),VkImageSubresourceRange.calloc(1,s).aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).levelCount(1).layerCount(1));
   b.limit(1);b.get(0).image(image[1]).oldLayout(clear==2?VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL).srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT);vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,null,null,b);
   vkCmdCopyImageToBuffer(c,image[1],VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,dst[0],region);check(vkEndCommandBuffer(c),"depth end");check(vkQueueSubmit(q,VkSubmitInfo.calloc(s).sType$Default().pCommandBuffers(s.pointers(c)),0),"depth submit");check(vkQueueWaitIdle(q),"depth idle");
   check(vkMapMemory(d,dst[1],0,128,0,p),"depth readback map");IntBuffer bits=MemoryUtil.memByteBuffer(p.get(0),128).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();for(int i=0;i<(clear==0?32:16);i++){int want=Float.floatToRawIntBits(clear==0?(i%16)/16.0f:0.625f);if(clear==2 && i%4<2 && i/4<2){float depth=Float.intBitsToFloat(bits.get(i));if(!(depth>0.5f&&depth<0.501f))throw new AssertionError("native depth bias "+depth);continue;}if(bits.get(i)!=want)throw new AssertionError("depth clear="+clear+" word="+i+" value="+Integer.toHexString(bits.get(i))+" expected="+Integer.toHexString(want));}vkUnmapMemory(d,dst[1]);vkFreeCommandBuffers(d,pool,c);
  }
  vkDestroyPipeline(d,pipeline,null);vkDestroyPipelineLayout(d,layout,null);vkDestroyShaderModule(d,vs,null);vkDestroyImageView(d,view,null);
  for(int i=0;i<2;i++){vkDestroyImage(d,image[i],null);vkFreeMemory(d,memory[i],null);}for(long[] b:new long[][]{src,dst}){vkDestroyBuffer(d,b[0],null);vkFreeMemory(d,b[1],null);}
  System.out.println("DEPTH_ONLY_NATIVE PASS vertex-only dynamic rendering D32 writes static depth bias untouched texels preserved");
  System.out.println("DEPTH_TRANSFER_NATIVE PASS GPU D32 upload copy readback clear 32 exact float words");
 }}
 static void capabilityCapacity(VkDevice d,VkQueue q,long pool){try(MemoryStack s=MemoryStack.stackPush()){
  LongBuffer h=s.mallocLong(1);PointerBuffer p=s.mallocPointer(1);long[] images=new long[1100],memories=new long[1100],views=new long[1100];
  long[] read=buffer(d,8,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  for(int round=0;round<2;round++){
   check(vkAllocateCommandBuffers(d,VkCommandBufferAllocateInfo.calloc(s).sType$Default().commandPool(pool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1),p),"capacity command");VkCommandBuffer c=new VkCommandBuffer(p.get(0),d);check(vkBeginCommandBuffer(c,VkCommandBufferBeginInfo.calloc(s).sType$Default()),"capacity begin");
   for(int i=0;i<images.length;i++){try(MemoryStack t=MemoryStack.stackPush()){
    check(vkCreateImage(d,VkImageCreateInfo.calloc(t).sType$Default().imageType(VK_IMAGE_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM).extent(e->e.set(1,1,1)).mipLevels(1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT).tiling(VK_IMAGE_TILING_OPTIMAL).usage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT),null,h),"capacity image");images[i]=h.get(0);
    VkMemoryRequirements r=VkMemoryRequirements.calloc(t);vkGetImageMemoryRequirements(d,images[i],r);check(vkAllocateMemory(d,VkMemoryAllocateInfo.calloc(t).sType$Default().allocationSize(r.size()).memoryTypeIndex(Integer.numberOfTrailingZeros(r.memoryTypeBits())),null,h),"capacity memory");memories[i]=h.get(0);check(vkBindImageMemory(d,images[i],memories[i],0),"capacity bind");
    check(vkCreateImageView(d,VkImageViewCreateInfo.calloc(t).sType$Default().image(images[i]).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM).subresourceRange(v->v.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1)),null,h),"capacity view");views[i]=h.get(0);
    final int phase=round;
    vkCmdBeginRenderingKHR(c,VkRenderingInfo.calloc(t).sType$Default().renderArea(a->a.extent(e->e.set(1,1))).layerCount(1).pColorAttachments(VkRenderingAttachmentInfo.calloc(1,t).sType$Default().imageView(views[i]).imageLayout(VK_IMAGE_LAYOUT_GENERAL).loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR).storeOp(VK_ATTACHMENT_STORE_OP_STORE).clearValue(v->v.color(col->col.float32(0,phase==0?1:0).float32(1,phase==0?0:1).float32(2,0).float32(3,1)))));vkCmdEndRenderingKHR(c);
   }}
   vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,VkMemoryBarrier.calloc(1,s).sType$Default().srcAccessMask(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT),null,null);
   for(int end=0;end<2;end++)vkCmdCopyImageToBuffer(c,images[end==0?0:1099],VK_IMAGE_LAYOUT_GENERAL,read[0],VkBufferImageCopy.calloc(1,s).bufferOffset(end*4).imageSubresource(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1)).imageExtent(e->e.set(1,1,1)));
   check(vkEndCommandBuffer(c),"capacity end");check(vkQueueSubmit(q,VkSubmitInfo.calloc(s).sType$Default().pCommandBuffers(s.pointers(c)),0),"capacity submit");check(vkQueueWaitIdle(q),"capacity idle");
   check(vkMapMemory(d,read[1],0,8,0,p),"capacity readback");IntBuffer got=MemoryUtil.memByteBuffer(p.get(0),8).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();for(int end=0;end<2;end++)if(got.get(end)!=(round==0?0xff0000ff:0xff00ff00))throw new AssertionError("capacity pixel round="+round+" end="+end);vkUnmapMemory(d,read[1]);
   vkFreeCommandBuffers(d,pool,c);for(int i=0;i<images.length;i++){vkDestroyImageView(d,views[i],null);vkDestroyImage(d,images[i],null);vkFreeMemory(d,memories[i],null);}
  }
  vkDestroyBuffer(d,read[0],null);vkFreeMemory(d,read[1],null);
  System.out.println("GPU_HANDLES_NATIVE PASS 1100 live GPU images plus completion handles two rounds exact pixels retirement and reuse");
 }}
 static void mipTargets(VkDevice d,VkQueue q,long pool){try(MemoryStack s=MemoryStack.stackPush()){
  LongBuffer h=s.mallocLong(1);PointerBuffer p=s.mallocPointer(1);long[] images=new long[2],memories=new long[2];
  for(int i=0;i<2;i++){
   final int size=i==0?8:4;
   check(vkCreateImage(d,VkImageCreateInfo.calloc(s).sType$Default().imageType(VK_IMAGE_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM).extent(e->e.set(size,size,1)).mipLevels(i==0?4:1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT).tiling(VK_IMAGE_TILING_OPTIMAL).usage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT),null,h),"mip image");images[i]=h.get(0);
   VkMemoryRequirements r=VkMemoryRequirements.calloc(s);vkGetImageMemoryRequirements(d,images[i],r);check(vkAllocateMemory(d,VkMemoryAllocateInfo.calloc(s).sType$Default().allocationSize(r.size()).memoryTypeIndex(Integer.numberOfTrailingZeros(r.memoryTypeBits())),null,h),"mip memory");memories[i]=h.get(0);check(vkBindImageMemory(d,images[i],memories[i],0),"mip bind");
  }
  check(vkCreateImageView(d,VkImageViewCreateInfo.calloc(s).sType$Default().image(images[0]).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_R8G8B8A8_UNORM).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(1).levelCount(1).layerCount(1)),null,h),"mip-one view");long view=h.get(0);
  long[] upload=buffer(d,320,VK_BUFFER_USAGE_TRANSFER_SRC_BIT),read=buffer(d,320,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  check(vkMapMemory(d,upload[1],0,320,0,p),"mip upload");IntBuffer data=MemoryUtil.memByteBuffer(p.get(0),320).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();for(int i=0;i<80;i++)data.put(i,i<64?0xff0000ff:0xffff0000);vkUnmapMemory(d,upload[1]);
  check(vkAllocateCommandBuffers(d,VkCommandBufferAllocateInfo.calloc(s).sType$Default().commandPool(pool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1),p),"mip command");VkCommandBuffer c=new VkCommandBuffer(p.get(0),d);check(vkBeginCommandBuffer(c,VkCommandBufferBeginInfo.calloc(s).sType$Default()),"mip begin");
  VkImageMemoryBarrier.Buffer barriers=VkImageMemoryBarrier.calloc(2,s);for(int i=0;i<2;i++)barriers.get(i).sType$Default().oldLayout(VK_IMAGE_LAYOUT_UNDEFINED).newLayout(VK_IMAGE_LAYOUT_GENERAL).image(images[i]).srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).subresourceRange(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(VK_REMAINING_MIP_LEVELS).layerCount(1));
  vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,null,null,barriers);
  VkBufferImageCopy.Buffer regions=VkBufferImageCopy.calloc(2,s);regions.get(0).imageSubresource(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1)).imageExtent(e->e.set(8,8,1));regions.get(1).bufferOffset(256).imageSubresource(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(1).layerCount(1)).imageExtent(e->e.set(4,4,1));vkCmdCopyBufferToImage(c,upload[0],images[0],VK_IMAGE_LAYOUT_GENERAL,regions);
  vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,VkMemoryBarrier.calloc(1,s).sType$Default().srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT).dstAccessMask(VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT),null,null);
  VkRenderingAttachmentInfo.Buffer attachment=VkRenderingAttachmentInfo.calloc(1,s).sType$Default().imageView(view).imageLayout(VK_IMAGE_LAYOUT_GENERAL).loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR).storeOp(VK_ATTACHMENT_STORE_OP_STORE).clearValue(v->v.color(col->col.float32(0,0).float32(1,1).float32(2,0).float32(3,1)));
  vkCmdBeginRenderingKHR(c,VkRenderingInfo.calloc(s).sType$Default().renderArea(r->r.offset(o->o.set(1,1)).extent(e->e.set(2,2))).layerCount(1).pColorAttachments(attachment));vkCmdEndRenderingKHR(c);
  attachment.loadOp(VK_ATTACHMENT_LOAD_OP_LOAD);
  vkCmdBeginRenderingKHR(c,VkRenderingInfo.calloc(s).sType$Default().renderArea(r->r.extent(e->e.set(4,4))).layerCount(1).pColorAttachments(attachment));
  vkCmdClearAttachments(c,VkClearAttachment.calloc(1,s).aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).colorAttachment(0).clearValue(v->v.color(col->col.float32(0,1).float32(1,1).float32(2,0).float32(3,1))),VkClearRect.calloc(1,s).rect(r->r.extent(e->e.set(1,1))).layerCount(1));vkCmdEndRenderingKHR(c);
  vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,VkMemoryBarrier.calloc(1,s).sType$Default().srcAccessMask(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT),null,null);
  VkImageBlit.Buffer blit=VkImageBlit.calloc(1,s).srcSubresource(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(1).layerCount(1)).dstSubresource(r->r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1));blit.srcOffsets(1).set(4,4,1);blit.dstOffsets(1).set(4,4,1);vkCmdBlitImage(c,images[0],VK_IMAGE_LAYOUT_GENERAL,images[1],VK_IMAGE_LAYOUT_GENERAL,blit,VK_FILTER_NEAREST);
  regions.limit(1);vkCmdCopyImageToBuffer(c,images[0],VK_IMAGE_LAYOUT_GENERAL,read[0],regions);regions.get(0).bufferOffset(256).imageExtent(e->e.set(4,4,1));vkCmdCopyImageToBuffer(c,images[1],VK_IMAGE_LAYOUT_GENERAL,read[0],regions);
  check(vkEndCommandBuffer(c),"mip end");check(vkQueueSubmit(q,VkSubmitInfo.calloc(s).sType$Default().pCommandBuffers(s.pointers(c)),0),"mip submit");check(vkQueueWaitIdle(q),"mip idle");
  check(vkMapMemory(d,read[1],0,320,0,p),"mip readback");IntBuffer got=MemoryUtil.memByteBuffer(p.get(0),320).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();for(int i=0;i<80;i++){int x=(i-64)%4,y=(i-64)/4;int want=i<64?0xff0000ff:i==64?0xff00ffff:x>=1&&x<=2&&y>=1&&y<=2?0xff00ff00:0xffff0000;if(got.get(i)!=want)throw new AssertionError("mip pixel "+i+" got="+Integer.toHexString(got.get(i))+" want="+Integer.toHexString(want));}vkUnmapMemory(d,read[1]);
  vkFreeCommandBuffers(d,pool,c);vkDestroyImageView(d,view,null);for(int i=0;i<2;i++){vkDestroyImage(d,images[i],null);vkFreeMemory(d,memories[i],null);}for(long[] b:new long[][]{upload,read}){vkDestroyBuffer(d,b[0],null);vkFreeMemory(d,b[1],null);}
  System.out.println("MIP_ATTACHMENT_NATIVE PASS mip1 partial clear LOAD clearAttachments resume exact80pixels mip0 and outside-area preserved");
 }}
 static void floatTargets(VkDevice d,VkQueue q,long pool){try(MemoryStack s=MemoryStack.stackPush()){
  LongBuffer h=s.mallocLong(1);PointerBuffer p=s.mallocPointer(1);
  long vs=shader(d,org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_vertex_shader,"#version 450\nvoid main(){vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex],0,1);}");
  long fs=shader(d,org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_fragment_shader,"#version 450\nlayout(location=0) out vec4 color[1];void main(){color[0]=vec4(-2,3,0.5,1);}");
  check(vkCreatePipelineLayout(d,VkPipelineLayoutCreateInfo.calloc(s).sType$Default(),null,h),"float pipeline layout");long layout=h.get(0);
  for(int format:new int[]{VK_FORMAT_R16G16B16A16_SFLOAT,VK_FORMAT_R32G32B32A32_SFLOAT}){
   int component=format==VK_FORMAT_R16G16B16A16_SFLOAT?2:4;
   check(vkCreateImage(d,VkImageCreateInfo.calloc(s).sType$Default().imageType(VK_IMAGE_TYPE_2D).format(format).extent(e->e.set(4,4,1)).mipLevels(1).arrayLayers(1).samples(VK_SAMPLE_COUNT_1_BIT).tiling(VK_IMAGE_TILING_OPTIMAL).usage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT),null,h),"floating-point image format="+format);long image=h.get(0);
   VkMemoryRequirements r=VkMemoryRequirements.calloc(s);vkGetImageMemoryRequirements(d,image,r);check(vkAllocateMemory(d,VkMemoryAllocateInfo.calloc(s).sType$Default().allocationSize(r.size()).memoryTypeIndex(Integer.numberOfTrailingZeros(r.memoryTypeBits())).pNext(VkMemoryDedicatedAllocateInfo.calloc(s).sType$Default().image(image).address()),null,h),"float memory");long memory=h.get(0);check(vkBindImageMemory(d,image,memory,0),"float bind");
   check(vkCreateImageView(d,VkImageViewCreateInfo.calloc(s).sType$Default().image(image).viewType(VK_IMAGE_VIEW_TYPE_2D).format(format).subresourceRange(a->a.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1)),null,h),"float view");long view=h.get(0);
   VkPipelineShaderStageCreateInfo.Buffer stages=VkPipelineShaderStageCreateInfo.calloc(2,s);stages.get(0).sType$Default().stage(VK_SHADER_STAGE_VERTEX_BIT).module(vs).pName(s.UTF8("main"));stages.get(1).sType$Default().stage(VK_SHADER_STAGE_FRAGMENT_BIT).module(fs).pName(s.UTF8("main"));
   VkGraphicsPipelineCreateInfo.Buffer info=VkGraphicsPipelineCreateInfo.calloc(1,s).sType$Default().pNext(VkPipelineRenderingCreateInfo.calloc(s).sType$Default().pColorAttachmentFormats(s.ints(format)).address()).layout(layout).pStages(stages)
    .pVertexInputState(VkPipelineVertexInputStateCreateInfo.calloc(s).sType$Default()).pInputAssemblyState(VkPipelineInputAssemblyStateCreateInfo.calloc(s).sType$Default().topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST))
    .pViewportState(VkPipelineViewportStateCreateInfo.calloc(s).sType$Default().pViewports(VkViewport.calloc(1,s).width(4).height(4).minDepth(0).maxDepth(1)).pScissors(VkRect2D.calloc(1,s).extent(e->e.set(2,2))))
    .pRasterizationState(VkPipelineRasterizationStateCreateInfo.calloc(s).sType$Default().polygonMode(VK_POLYGON_MODE_FILL).cullMode(VK_CULL_MODE_NONE).lineWidth(1)).pMultisampleState(VkPipelineMultisampleStateCreateInfo.calloc(s).sType$Default().rasterizationSamples(VK_SAMPLE_COUNT_1_BIT))
    .pColorBlendState(VkPipelineColorBlendStateCreateInfo.calloc(s).sType$Default().pAttachments(VkPipelineColorBlendAttachmentState.calloc(1,s).blendEnable(true).srcColorBlendFactor(VK_BLEND_FACTOR_ONE).dstColorBlendFactor(VK_BLEND_FACTOR_ONE).colorBlendOp(VK_BLEND_OP_ADD).srcAlphaBlendFactor(VK_BLEND_FACTOR_ONE).dstAlphaBlendFactor(VK_BLEND_FACTOR_ONE).alphaBlendOp(VK_BLEND_OP_ADD).colorWriteMask(15)));
   check(vkCreateGraphicsPipelines(d,0,info,null,h),"float blend pipeline");long pipeline=h.get(0);long[] out=buffer(d,256,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   check(vkAllocateCommandBuffers(d,VkCommandBufferAllocateInfo.calloc(s).sType$Default().commandPool(pool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1),p),"float command");VkCommandBuffer c=new VkCommandBuffer(p.get(0),d);check(vkBeginCommandBuffer(c,VkCommandBufferBeginInfo.calloc(s).sType$Default()),"float begin");
   VkImageMemoryBarrier.Buffer barrier=VkImageMemoryBarrier.calloc(1,s).sType$Default().oldLayout(VK_IMAGE_LAYOUT_UNDEFINED).newLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL).image(image).srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).subresourceRange(a->a.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1));vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,null,null,barrier);
   vkCmdBeginRenderingKHR(c,VkRenderingInfo.calloc(s).sType$Default().renderArea(a->a.extent(e->e.set(4,4))).layerCount(1).pColorAttachments(VkRenderingAttachmentInfo.calloc(1,s).sType$Default().imageView(view).imageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL).loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR).storeOp(VK_ATTACHMENT_STORE_OP_STORE).clearValue(a->a.color(v->v.float32(0,1.5f).float32(1,-0.5f).float32(2,0.25f).float32(3,1)))));
   vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);vkCmdDraw(c,3,1,0,0);vkCmdEndRenderingKHR(c);barrier.oldLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL).newLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL).srcAccessMask(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT);vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,null,null,barrier);
   vkCmdCopyImageToBuffer(c,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,out[0],VkBufferImageCopy.calloc(1,s).imageSubresource(a->a.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1)).imageExtent(e->e.set(4,4,1)));check(vkEndCommandBuffer(c),"float end");check(vkQueueSubmit(q,VkSubmitInfo.calloc(s).sType$Default().pCommandBuffers(s.pointers(c)),0),"float submit");check(vkQueueWaitIdle(q),"float idle");
   check(vkMapMemory(d,out[1],0,256,0,p),"float map");ByteBuffer bytes=MemoryUtil.memByteBuffer(p.get(0),256).order(ByteOrder.LITTLE_ENDIAN);
   float[] clear={1.5f,-0.5f,0.25f,1},blend={-0.5f,2.5f,0.75f,2};int[] halfClear={0x3e00,0xb800,0x3400,0x3c00},halfBlend={0xb800,0x4100,0x3a00,0x4000};
   for(int y=0;y<4;y++)for(int x=0;x<4;x++)for(int channel=0;channel<4;channel++){boolean inside=x<2&&y<2;int offset=((y*4+x)*4+channel)*component;int value=component==2?Short.toUnsignedInt(bytes.getShort(offset)):bytes.getInt(offset);int expected=component==2?(inside?halfBlend:halfClear)[channel]:Float.floatToRawIntBits((inside?blend:clear)[channel]);if(value!=expected)throw new AssertionError("float format="+format+" pixel="+x+","+y+" channel="+channel+" got="+Integer.toHexString(value)+" expected="+Integer.toHexString(expected));}
   vkUnmapMemory(d,out[1]);vkFreeCommandBuffers(d,pool,c);vkDestroyBuffer(d,out[0],null);vkFreeMemory(d,out[1],null);vkDestroyPipeline(d,pipeline,null);vkDestroyImageView(d,view,null);vkDestroyImage(d,image,null);vkFreeMemory(d,memory,null);
  }
  vkDestroyPipelineLayout(d,layout,null);vkDestroyShaderModule(d,vs,null);vkDestroyShaderModule(d,fs,null);
  System.out.println("FLOAT_COLOR_NATIVE PASS RGBA16F RGBA32F GPU clear draw additive blend raw readback 128 exact components negative and above-one values");
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
   compute=(families.get(0).queueFlags()&VK_QUEUE_COMPUTE_BIT)!=0;
   VkPhysicalDeviceProperties properties=VkPhysicalDeviceProperties.calloc(s);vkGetPhysicalDeviceProperties(physical,properties);if(compute!=(properties.limits().maxComputeWorkGroupInvocations()>0))throw new AssertionError("compute queue/limit mismatch");
   VkPhysicalDeviceVulkan11Properties p11=VkPhysicalDeviceVulkan11Properties.calloc(s).sType$Default();
   VkPhysicalDeviceMaintenance3Properties maintenance=VkPhysicalDeviceMaintenance3Properties.calloc(s).sType$Default();p11.pNext(maintenance.address());
   VK11.vkGetPhysicalDeviceProperties2(physical,VkPhysicalDeviceProperties2.calloc(s).sType$Default().pNext(p11.address()));
   if(p11.maxMemoryAllocationSize()!=256L*1024*1024||p11.maxMemoryAllocationSize()!=maintenance.maxMemoryAllocationSize()||properties.limits().maxImageDimension2D()<4096)throw new AssertionError("allocation/texture properties");
   System.out.println("CORE11_PROPERTIES_NATIVE PASS maxAllocation="+p11.maxMemoryAllocationSize()+" maxImage2D="+properties.limits().maxImageDimension2D());
   VkPhysicalDeviceFeatures hardware=VkPhysicalDeviceFeatures.calloc(s);vkGetPhysicalDeviceFeatures(physical,hardware);statistics=hardware.pipelineStatisticsQuery();
   VkPhysicalDeviceHostQueryResetFeatures hostReset=VkPhysicalDeviceHostQueryResetFeatures.calloc(s).sType$Default();timelineFeature.pNext(hostReset.address());
   VkPhysicalDeviceVulkan12Features promoted=VkPhysicalDeviceVulkan12Features.calloc(s).sType$Default().bufferDeviceAddress(true).runtimeDescriptorArray(true);promoted.pNext(timelineFeature.address());
   VK11.vkGetPhysicalDeviceFeatures2(physical,VkPhysicalDeviceFeatures2.calloc(s).sType$Default().pNext(promoted.address()));
   if(!promoted.timelineSemaphore()||!promoted.hostQueryReset()||!promoted.drawIndirectCount()||promoted.timelineSemaphore()!=timelineFeature.timelineSemaphore()||promoted.hostQueryReset()!=hostReset.hostQueryReset()||promoted.bufferDeviceAddress()||promoted.runtimeDescriptorArray())throw new AssertionError("promoted/extension feature agreement timeline="+promoted.timelineSemaphore()+"/"+timelineFeature.timelineSemaphore()+" reset="+promoted.hostQueryReset()+"/"+hostReset.hostQueryReset()+" indirect="+promoted.drawIndirectCount()+" bda="+promoted.bufferDeviceAddress()+" arrays="+promoted.runtimeDescriptorArray());
   promoted.pNext(0);support.pNext(promoted.address());
   VkDeviceQueueCreateInfo.Buffer queues=VkDeviceQueueCreateInfo.calloc(1,s).sType$Default().queueFamilyIndex(0).pQueuePriorities(s.floats(1));
   VkPhysicalDeviceDynamicRenderingFeatures rendering=VkPhysicalDeviceDynamicRenderingFeatures.calloc(s).sType$Default().dynamicRendering(true);promoted.pNext(rendering.address());
   if(!hardware.samplerAnisotropy()||properties.limits().maxSamplerAnisotropy()<16)throw new AssertionError("host anisotropy capability");
   VkPhysicalDeviceVulkan11Features drawParameters=VkPhysicalDeviceVulkan11Features.calloc(s).sType$Default().shaderDrawParameters(true);VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT divisors=VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT.calloc(s).sType$Default().vertexAttributeInstanceRateDivisor(true);rendering.pNext(drawParameters.address());drawParameters.pNext(divisors.address());
   VkDeviceCreateInfo di=VkDeviceCreateInfo.calloc(s).sType$Default().pNext(support.address()).pQueueCreateInfos(queues).pEnabledFeatures(VkPhysicalDeviceFeatures.calloc(s).multiDrawIndirect(true).drawIndirectFirstInstance(true).pipelineStatisticsQuery(statistics).samplerAnisotropy(true)).ppEnabledExtensionNames(s.pointers(s.UTF8(KHRPushDescriptor.VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME),s.UTF8(EXTVertexAttributeDivisor.VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME),s.UTF8(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME),s.UTF8(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME),s.UTF8(VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME),s.UTF8(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME),s.UTF8(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME)));
   check(vkCreateDevice(physical,di,null,p),"device");VkDevice device=new VkDevice(p.get(0),physical,di);vkGetDeviceQueue(device,0,0,p);VkQueue queue=new VkQueue(p.get(0),device);
   System.out.println("CORE12_FEATURES_NATIVE PASS aggregate enablement aliases agree unsupported bits cleared");
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
   vmaAllocations(device,queue,pool);compute(device,queue,pool);graphics(device,queue,pool);depthTransfers(device,queue,pool);mipTargets(device,queue,pool);floatTargets(device,queue,pool);capabilityCapacity(device,queue,pool);
   vkDestroyFence(device,fence,null);vkDestroySemaphore(device,binary,null);vkDestroySemaphore(device,timeline,null);vkDestroyEvent(device,event,null);vkDestroyCommandPool(device,pool,null);
   for(int i=0;i<2;i++){vkDestroyBuffer(device,buffers[i],null);vkFreeMemory(device,memories[i],null);}
   vkDestroyDevice(device,null);vkDestroyInstance(instance,null);VK.destroy();
   System.out.println("SYNC2_NATIVE PASS GPU copy bytes=256 events mixed semaphores timeline host gate fence");
  }
 }
}
