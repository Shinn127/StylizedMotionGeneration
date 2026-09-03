from __future__ import annotations

import numpy as np
from pyray import Color, Rectangle, Vector2
from raylib import *
from raylib.defines import *

from stylized_motion.anim.environment import set_shader_value_cubemap


class Renderer:
    """Pass scheduler for the viewer's deferred render targets and shaders."""

    def __init__(self, view):
        self.view = view

    def _bind_material(self, shader_key, material, ground_pattern=False):
        view = self.view
        if view.shading != "pbr":
            return
        prefix = shader_key
        for index, channel in enumerate(material.base_color):
            view.material_base_color_ptr[index] = channel
        view.metallic_ptr[0] = material.metallic
        view.roughness_ptr[0] = material.roughness
        view.material_ao_ptr[0] = material.ao
        view.use_base_color_map_ptr[0] = int(material.uses_base_color_map)
        view.use_metallic_roughness_map_ptr[0] = int(material.uses_metallic_roughness_map)
        view.use_normal_map_ptr[0] = int(material.uses_normal_map)
        view.ground_pattern_ptr[0] = int(ground_pattern)
        SetShaderValue(view.shaders[shader_key], view.shader_locs[f"{prefix}_material_base_color"], view.material_base_color_ptr, SHADER_UNIFORM_VEC4)
        SetShaderValue(view.shaders[shader_key], view.shader_locs[f"{prefix}_metallic"], view.metallic_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders[shader_key], view.shader_locs[f"{prefix}_roughness"], view.roughness_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders[shader_key], view.shader_locs[f"{prefix}_ao"], view.material_ao_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders[shader_key], view.shader_locs[f"{prefix}_use_base_color_map"], view.use_base_color_map_ptr, SHADER_UNIFORM_INT)
        SetShaderValue(view.shaders[shader_key], view.shader_locs[f"{prefix}_use_metallic_roughness_map"], view.use_metallic_roughness_map_ptr, SHADER_UNIFORM_INT)
        SetShaderValue(view.shaders[shader_key], view.shader_locs[f"{prefix}_use_normal_map"], view.use_normal_map_ptr, SHADER_UNIFORM_INT)
        SetShaderValue(view.shaders[shader_key], view.shader_locs[f"{prefix}_ground_pattern"], view.ground_pattern_ptr, SHADER_UNIFORM_INT)
        if material.base_color_map is not None:
            SetShaderValueTexture(view.shaders[shader_key], view.shader_locs[f"{prefix}_base_color_map"], material.base_color_map)
        if material.metallic_roughness_map is not None:
            SetShaderValueTexture(view.shaders[shader_key], view.shader_locs[f"{prefix}_metallic_roughness_map"], material.metallic_roughness_map)
        if material.normal_map is not None:
            SetShaderValueTexture(view.shaders[shader_key], view.shader_locs[f"{prefix}_normal_map"], material.normal_map)

    def render_shadow(self):
        view = self.view
        begin_shadow_map(view.shadow_map, view.shadow_light)
        light_view_proj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
        light_clip_near = rlGetCullDistanceNear()
        light_clip_far = rlGetCullDistanceFar()
        view.light_clip_near_ptr[0] = light_clip_near
        view.light_clip_far_ptr[0] = light_clip_far
        SetShaderValue(view.shaders["shadow"], view.shader_locs["shadow_light_clip_near"], view.light_clip_near_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["shadow"], view.shader_locs["shadow_light_clip_far"], view.light_clip_far_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["skinned_shadow"], view.shader_locs["skinned_shadow_light_clip_near"], view.light_clip_near_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["skinned_shadow"], view.shader_locs["skinned_shadow_light_clip_far"], view.light_clip_far_ptr, SHADER_UNIFORM_FLOAT)

        for render_object in view.scene.objects:
            shader_key = "skinned_shadow" if render_object.skinned else "shadow"
            render_object.model.materials[0].shader = view.shaders[shader_key]
            DrawModel(render_object.model, render_object.position, render_object.scale, WHITE)
        end_shadow_map()
        return light_view_proj

    def render_gbuffer(self):
        view = self.view
        begin_gbuffer(view.gbuffer, view.camera.cam3d)
        cam_view = rlGetMatrixModelview()
        cam_proj = rlGetMatrixProjection()
        cam_inv_proj = MatrixInvert(cam_proj)
        cam_inv_view_proj = MatrixInvert(MatrixMultiply(cam_view, cam_proj))
        cam_clip_near = rlGetCullDistanceNear()
        cam_clip_far = rlGetCullDistanceFar()
        view.cam_clip_near_ptr[0] = cam_clip_near
        view.cam_clip_far_ptr[0] = cam_clip_far
        SetShaderValue(view.shaders["basic"], view.shader_locs["basic_cam_clip_near"], view.cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["basic"], view.shader_locs["basic_cam_clip_far"], view.cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)
        if view.shading == "legacy":
            SetShaderValue(view.shaders["basic"], view.shader_locs["basic_specularity"], view.specularity_ptr, SHADER_UNIFORM_FLOAT)
            SetShaderValue(view.shaders["basic"], view.shader_locs["basic_glossiness"], view.glossiness_ptr, SHADER_UNIFORM_FLOAT)
            SetShaderValue(view.shaders["skinned_basic"], view.shader_locs["skinned_basic_specularity"], view.specularity_ptr, SHADER_UNIFORM_FLOAT)
            SetShaderValue(view.shaders["skinned_basic"], view.shader_locs["skinned_basic_glossiness"], view.glossiness_ptr, SHADER_UNIFORM_FLOAT)
        else:
            SetShaderValue(view.shaders["basic"], view.shader_locs["basic_metallic"], view.metallic_ptr, SHADER_UNIFORM_FLOAT)
            SetShaderValue(view.shaders["basic"], view.shader_locs["basic_roughness"], view.roughness_ptr, SHADER_UNIFORM_FLOAT)
            SetShaderValue(view.shaders["skinned_basic"], view.shader_locs["skinned_basic_metallic"], view.metallic_ptr, SHADER_UNIFORM_FLOAT)
            SetShaderValue(view.shaders["skinned_basic"], view.shader_locs["skinned_basic_roughness"], view.roughness_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["skinned_basic"], view.shader_locs["skinned_basic_cam_clip_near"], view.cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["skinned_basic"], view.shader_locs["skinned_basic_cam_clip_far"], view.cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)

        for render_object in view.scene.objects:
            shader_key = "skinned_basic" if render_object.skinned else "basic"
            self._bind_material(shader_key, render_object.material, render_object.model is view.ground_model)
            render_object.model.materials[0].shader = view.shaders[shader_key]
            draw_color = render_object.draw_color if render_object.draw_color is not None else WHITE
            DrawModel(render_object.model, render_object.position, render_object.scale, draw_color)
        end_gbuffer(view.screen_width, view.screen_height)
        return cam_view, cam_proj, cam_inv_proj, cam_inv_view_proj

    def render_ssao(self, cam_view, cam_proj, cam_inv_proj):
        view = self.view
        BeginTextureMode(view.ssao_front)
        BeginShaderMode(view.shaders["ssao"])
        SetShaderValueTexture(view.shaders["ssao"], view.shader_locs["ssao_gbuffer_normal"], view.gbuffer.normal)
        SetShaderValueTexture(view.shaders["ssao"], view.shader_locs["ssao_gbuffer_depth"], view.gbuffer.depth)
        SetShaderValueMatrix(view.shaders["ssao"], view.shader_locs["ssao_cam_view"], cam_view)
        SetShaderValueMatrix(view.shaders["ssao"], view.shader_locs["ssao_cam_proj"], cam_proj)
        SetShaderValueMatrix(view.shaders["ssao"], view.shader_locs["ssao_cam_inv_proj"], cam_inv_proj)
        SetShaderValue(view.shaders["ssao"], view.shader_locs["ssao_cam_clip_near"], view.cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["ssao"], view.shader_locs["ssao_cam_clip_far"], view.cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)
        view.ssao_intensity_ptr[0] = view.ssao_intensity
        SetShaderValue(view.shaders["ssao"], view.shader_locs["ssao_intensity"], view.ssao_intensity_ptr, SHADER_UNIFORM_FLOAT)
        ClearBackground(WHITE)
        DrawTextureRec(view.ssao_front.texture, Rectangle(0, 0, view.ssao_front.texture.width, -view.ssao_front.texture.height), Vector2(0.0, 0.0), WHITE)
        EndShaderMode()
        EndTextureMode()

    def render_ssao_blur(self, cam_inv_proj):
        view = self.view
        view.blur_inv_texture_resolution.x = 1.0 / view.ssao_front.texture.width
        view.blur_inv_texture_resolution.y = 1.0 / view.ssao_front.texture.height
        BeginTextureMode(view.ssao_back)
        BeginShaderMode(view.shaders["blur"])
        view.blur_direction.x = 1.0
        view.blur_direction.y = 0.0
        SetShaderValueTexture(view.shaders["blur"], view.shader_locs["blur_gbuffer_normal"], view.gbuffer.normal)
        SetShaderValueTexture(view.shaders["blur"], view.shader_locs["blur_gbuffer_depth"], view.gbuffer.depth)
        SetShaderValueTexture(view.shaders["blur"], view.shader_locs["blur_input_texture"], view.ssao_front.texture)
        SetShaderValueMatrix(view.shaders["blur"], view.shader_locs["blur_cam_inv_proj"], cam_inv_proj)
        SetShaderValue(view.shaders["blur"], view.shader_locs["blur_cam_clip_near"], view.cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["blur"], view.shader_locs["blur_cam_clip_far"], view.cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["blur"], view.shader_locs["blur_inv_texture_resolution"], ffi.addressof(view.blur_inv_texture_resolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(view.shaders["blur"], view.shader_locs["blur_direction"], ffi.addressof(view.blur_direction), SHADER_UNIFORM_VEC2)
        DrawTextureRec(view.ssao_back.texture, Rectangle(0, 0, view.ssao_back.texture.width, -view.ssao_back.texture.height), Vector2(0, 0), WHITE)
        EndShaderMode()
        EndTextureMode()

        BeginTextureMode(view.ssao_front)
        BeginShaderMode(view.shaders["blur"])
        view.blur_direction.x = 0.0
        view.blur_direction.y = 1.0
        SetShaderValueTexture(view.shaders["blur"], view.shader_locs["blur_input_texture"], view.ssao_back.texture)
        SetShaderValue(view.shaders["blur"], view.shader_locs["blur_direction"], ffi.addressof(view.blur_direction), SHADER_UNIFORM_VEC2)
        DrawTextureRec(view.ssao_front.texture, Rectangle(0, 0, view.ssao_front.texture.width, -view.ssao_front.texture.height), Vector2(0, 0), WHITE)
        EndShaderMode()
        EndTextureMode()

    def render_lighting(self, cam_inv_view_proj, light_view_proj):
        view = self.view
        light = view.scene.directional_light
        BeginTextureMode(view.lighted)
        BeginShaderMode(view.shaders["lighting"])
        SetShaderValueTexture(view.shaders["lighting"], view.shader_locs["lighting_gbuffer_color"], view.gbuffer.color)
        SetShaderValueTexture(view.shaders["lighting"], view.shader_locs["lighting_gbuffer_normal"], view.gbuffer.normal)
        SetShaderValueTexture(view.shaders["lighting"], view.shader_locs["lighting_gbuffer_depth"], view.gbuffer.depth)
        set_shader_value_texture_slot(
            view.shaders["lighting"], view.shader_locs["lighting_ssao"],
            view.ssao_front.texture, view.ssao_texture_slot_ptr,
        )
        set_shader_value_shadow_map(view.shaders["lighting"], view.shader_locs["lighting_shadow_map"], view.shadow_map, view.shadow_texture_slot_ptr)
        if view.shading == "pbr":
            set_shader_value_texture_slot(
                view.shaders["lighting"], view.shader_locs["lighting_material_ao"],
                view.gbuffer.material_ao, view.material_ao_texture_slot_ptr,
            )
            set_shader_value_cubemap(view.shaders["lighting"], view.shader_locs["lighting_environment_map"], view.ibl.environment, view.environment_texture_slot_ptr)
            set_shader_value_cubemap(view.shaders["lighting"], view.shader_locs["lighting_irradiance_map"], view.ibl.irradiance, view.irradiance_texture_slot_ptr)
            set_shader_value_cubemap(view.shaders["lighting"], view.shader_locs["lighting_prefilter_map"], view.ibl.prefilter, view.prefilter_texture_slot_ptr)
            rlEnableShader(view.shaders["lighting"].id)
            rlActiveTextureSlot(view.brdf_lut_texture_slot_ptr[0])
            rlEnableTexture(view.ibl.brdf_lut.id)
            rlSetUniform(view.shader_locs["lighting_brdf_lut"], view.brdf_lut_texture_slot_ptr, SHADER_UNIFORM_INT, 1)
            SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_prefilter_max_lod"], view.prefilter_max_lod_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_cam_pos"], ffi.addressof(view.camera.cam3d.position), SHADER_UNIFORM_VEC3)
        SetShaderValueMatrix(view.shaders["lighting"], view.shader_locs["lighting_cam_inv_view_proj"], cam_inv_view_proj)
        SetShaderValueMatrix(view.shaders["lighting"], view.shader_locs["lighting_light_view_proj"], light_view_proj)
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_light_dir"], ffi.addressof(light.direction), SHADER_UNIFORM_VEC3)
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_sun_color"], ffi.addressof(light.color), SHADER_UNIFORM_VEC3)
        view.sun_strength_ptr[0] = light.intensity
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_sun_strength"], view.sun_strength_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_sky_color"], ffi.addressof(view.sky_color), SHADER_UNIFORM_VEC3)
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_sky_strength"], view.sky_strength_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_ground_strength"], view.ground_strength_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_ambient_strength"], view.ambient_strength_ptr, SHADER_UNIFORM_FLOAT)
        if view.shading == "legacy":
            SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_exposure"], view.exposure_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_cam_clip_near"], view.cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_cam_clip_far"], view.cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_light_clip_near"], view.light_clip_near_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_light_clip_far"], view.light_clip_far_ptr, SHADER_UNIFORM_FLOAT)
        if view.shading == "pbr":
            view.ibl_strength_ptr[0] = view.ibl.strength
            SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_ibl_strength"], view.ibl_strength_ptr, SHADER_UNIFORM_FLOAT)
            view.use_ibl_ptr[0] = int(view.ibl.enabled)
            SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_use_ibl"], view.use_ibl_ptr, SHADER_UNIFORM_INT)
            view.debug_mode_ptr[0] = LIGHTING_DEBUG_MODES[view.debug_view]
            SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_debug_mode"], view.debug_mode_ptr, SHADER_UNIFORM_INT)
            view.shadow_texel_size.x = 1.0 / float(view.shadow_map.depth.width)
            view.shadow_texel_size.y = 1.0 / float(view.shadow_map.depth.height)
            SetShaderValue(view.shaders["lighting"], view.shader_locs["lighting_shadow_texel_size"], ffi.addressof(view.shadow_texel_size), SHADER_UNIFORM_VEC2)
        ClearBackground(RAYWHITE)
        DrawTextureRec(view.gbuffer.color, Rectangle(0, 0, view.gbuffer.color.width, -view.gbuffer.color.height), Vector2(0, 0), WHITE)
        EndShaderMode()
        EndTextureMode()

    def render_tonemap(self):
        view = self.view
        if view.shading != "pbr":
            return
        if view.debug_view != "final":
            self.render_debug()
            return
        BeginTextureMode(view.tonemapped)
        BeginShaderMode(view.shaders["tonemap"])
        view.exposure_ptr[0] = view.exposure
        SetShaderValueTexture(view.shaders["tonemap"], view.shader_locs["tonemap_input_texture"], view.lighted.texture)
        SetShaderValue(view.shaders["tonemap"], view.shader_locs["tonemap_exposure"], view.exposure_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(view.shaders["tonemap"], view.shader_locs["tonemap_tone_curve"], view.tone_curve_ptr, SHADER_UNIFORM_INT)
        ClearBackground(BLACK)
        DrawTextureRec(view.lighted.texture, Rectangle(0, 0, view.lighted.texture.width, -view.lighted.texture.height), Vector2(0, 0), WHITE)
        EndShaderMode()
        EndTextureMode()

    def render_debug(self):
        """Render the active debug view into the display target.

        Debug output deliberately bypasses ACES tonemapping and FXAA so the
        raw debug quantities stay readable; the display shader applies only
        exposure and linear-to-sRGB where radiance-like quantities need it.
        """
        view = self.view
        BeginTextureMode(view.tonemapped)
        BeginShaderMode(view.shaders["debug"])
        set_shader_value_texture_slot(
            view.shaders["debug"], view.shader_locs["debug_gbuffer_color"],
            view.gbuffer.color, view.debug_gbuffer_color_slot_ptr,
        )
        set_shader_value_texture_slot(
            view.shaders["debug"], view.shader_locs["debug_gbuffer_normal"],
            view.gbuffer.normal, view.debug_gbuffer_normal_slot_ptr,
        )
        set_shader_value_texture_slot(
            view.shaders["debug"], view.shader_locs["debug_gbuffer_depth"],
            view.gbuffer.depth, view.debug_gbuffer_depth_slot_ptr,
        )
        set_shader_value_texture_slot(
            view.shaders["debug"], view.shader_locs["debug_ssao"],
            view.ssao_front.texture, view.debug_ssao_slot_ptr,
        )
        set_shader_value_texture_slot(
            view.shaders["debug"], view.shader_locs["debug_lighted"],
            view.lighted.texture, view.debug_lighted_slot_ptr,
        )
        view.debug_mode_ptr[0] = DEBUG_MODES.index(view.debug_view)
        SetShaderValue(view.shaders["debug"], view.shader_locs["debug_mode"], view.debug_mode_ptr, SHADER_UNIFORM_INT)
        view.exposure_ptr[0] = view.exposure
        SetShaderValue(view.shaders["debug"], view.shader_locs["debug_exposure"], view.exposure_ptr, SHADER_UNIFORM_FLOAT)
        ClearBackground(BLACK)
        DrawTextureRec(view.gbuffer.color, Rectangle(0, 0, view.gbuffer.color.width, -view.gbuffer.color.height), Vector2(0, 0), WHITE)
        EndShaderMode()
        EndTextureMode()

    def render_frame(self, global_rot, global_pos, sample_index, compare_global_rot=None, compare_global_pos=None):
        light_view_proj = self.render_shadow()
        cam_view, cam_proj, cam_inv_proj, cam_inv_view_proj = self.render_gbuffer()
        self.render_ssao(cam_view, cam_proj, cam_inv_proj)
        self.render_ssao_blur(cam_inv_proj)
        self.render_lighting(cam_inv_view_proj, light_view_proj)
        view = self.view
        if view.tpos is not None and view.tdir is not None:
            BeginTextureMode(view.lighted)
            BeginMode3D(view.camera.cam3d)
            draw_trajectory(global_pos[0], global_rot[0], view.tpos[sample_index], view.tdir[sample_index])
            EndMode3D()
            EndTextureMode()
        if view.skeleton_enabled:
            self.render_skeleton(global_rot, global_pos, view.left_model_offset)
            if view.compare_mode and compare_global_rot is not None and compare_global_pos is not None:
                self.render_skeleton(compare_global_rot, compare_global_pos, view.right_model_offset)
        self.render_tonemap()

    def render_skeleton(self, global_rot, global_pos, model_offset):
        """Overlay the character skeleton onto the lighting target, GenoView debug style.

        ``skeleton_overlay_pose`` cuts the full simulation-root pose down to the
        subtree rooted at the rig's character root joint, dropping the virtual
        simulation root and static rig nodes pinned at the world origin.
        """
        view = self.view
        positions, rotations, parents = skeleton_overlay_pose(
            global_pos, global_rot, view.full_parents, view.skeleton_root_index
        )
        positions = positions + np.asarray([model_offset.x, model_offset.y, model_offset.z], dtype=np.float32)
        BeginTextureMode(view.lighted)
        BeginMode3D(view.camera.cam3d)
        draw_skeleton(positions, rotations, parents, view.skeleton_color)
        EndMode3D()
        EndTextureMode()

    def draw_output(self):
        view = self.view
        output_texture = view.tonemapped.texture if view.shading == "pbr" else view.lighted.texture
        if view.shading == "pbr" and view.debug_view != "final":
            DrawTextureRec(output_texture, Rectangle(0, 0, output_texture.width, -output_texture.height), Vector2(0, 0), WHITE)
            return
        view.fxaa_inv_texture_resolution.x = 1.0 / output_texture.width
        view.fxaa_inv_texture_resolution.y = 1.0 / output_texture.height
        BeginShaderMode(view.shaders["fxaa"])
        SetShaderValueTexture(view.shaders["fxaa"], view.shader_locs["fxaa_input_texture"], output_texture)
        SetShaderValue(view.shaders["fxaa"], view.shader_locs["fxaa_inv_texture_resolution"], ffi.addressof(view.fxaa_inv_texture_resolution), SHADER_UNIFORM_VEC2)
        DrawTextureRec(output_texture, Rectangle(0, 0, output_texture.width, -output_texture.height), Vector2(0, 0), WHITE)
        EndShaderMode()


from stylized_motion.anim.genoview import DEBUG_MODES, LIGHTING_DEBUG_MODES, draw_skeleton, draw_trajectory, ffi, skeleton_overlay_pose
from stylized_motion.anim.render_targets import (
    begin_gbuffer,
    begin_shadow_map,
    end_gbuffer,
    end_shadow_map,
    set_shader_value_shadow_map,
    set_shader_value_texture_slot,
)
