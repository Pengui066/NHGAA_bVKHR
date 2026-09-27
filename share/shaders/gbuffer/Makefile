all: strand_gbuffer.vert.spv strand_gbuffer.frag.spv fullscreen.vert.spv shading.frag.spv

strand_gbuffer.vert.spv: strand_gbuffer.vert ../scene_graph/camera.glsl
	glslc -O -g -c strand_gbuffer.vert

strand_gbuffer.frag.spv: strand_gbuffer.frag ../scene_graph/camera.glsl ../anti-aliasing/gpaa.glsl
	glslc -O -g -c strand_gbuffer.frag

fullscreen.vert.spv: fullscreen.vert
	glslc -O -g -c fullscreen.vert

shading.frag.spv: shading.frag ../scene_graph/camera.glsl ../scene_graph/shadow_maps.glsl ../scene_graph/lights.glsl ../self-shadowing/approximate_deep_shadows.glsl ../shading/kajiya-kay.glsl ../volumes/local_ambient_occlusion.glsl ../scene_graph/params.glsl ../strands/strand.glsl ../volumes/bounding_box.glsl ../volumes/sample_volume.glsl ../utils/math.glsl ../self-shadowing/linearize_depth.glsl ../self-shadowing/tex2Dproj.glsl
	glslc -O -g -c shading.frag
