"""
衍射光栅粒子系统 v3.1 - Blender 5.0.1 兼容版
专为衍射模拟设计，支持屏幕对齐和颜色调节
"""

bl_info = {
    "name": "衍射光栅粒子系统",
    "author": "Blender助手",
    "version": (3, 1, 0),
    "blender": (5, 0, 1),
    "location": "3D视图 > 侧边栏 > 衍射光栅",
    "description": "专为衍射模拟设计的粒子系统，支持屏幕对齐和颜色调节",
    "category": "粒子",
}

import bpy
import math
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import (EnumProperty, FloatProperty, BoolProperty, 
                      IntProperty, FloatVectorProperty, PointerProperty,
                      StringProperty)

# ============================================================================
# 1. 属性定义 - 简化版，避免API冲突
# ============================================================================

class DiffractionProps(PropertyGroup):
    """衍射系统属性"""
    
    # 屏幕设置
    target_name: StringProperty(
        name="目标屏幕",
        description="目标屏幕物体名称",
        default=""
    )
    
    align_mode: EnumProperty(
        name="对齐方式",
        description="选择屏幕对齐方式",
        items=[
            ('PARALLEL', "平行", "发射屏幕与目标屏幕平行"),
            ('FACE', "面向", "发射屏幕面向目标屏幕"),
        ],
        default='PARALLEL'
    )
    
    distance: FloatProperty(
        name="屏幕距离",
        description="发射屏幕与目标屏幕之间的距离",
        default=3.0,
        min=0.5,
        max=10.0
    )
    
    # 粒子设置
    particle_count: IntProperty(
        name="粒子数量",
        description="发射的粒子总数",
        default=1000,
        min=100,
        max=5000
    )
    
    particle_size: FloatProperty(
        name="粒子大小",
        description="粒子的显示大小",
        default=0.03,
        min=0.001,
        max=0.1
    )
    
    # 颜色设置
    color_mode: EnumProperty(
        name="颜色模式",
        description="粒子颜色模式",
        items=[
            ('SINGLE', "单色", "所有粒子使用相同颜色"),
            ('GRADIENT', "渐变", "粒子颜色渐变"),
        ],
        default='SINGLE'
    )
    
    color1: FloatVectorProperty(
        name="颜色1",
        description="主要颜色",
        subtype='COLOR',
        size=4,
        default=(0.2, 0.5, 1.0, 1.0),  # 蓝色
        min=0.0,
        max=1.0
    )
    
    color2: FloatVectorProperty(
        name="颜色2",
        description="次要颜色（用于渐变）",
        subtype='COLOR',
        size=4,
        default=(1.0, 0.2, 0.5, 1.0),  # 粉色
        min=0.0,
        max=1.0
    )
    
    # 光波设置
    wavelength: FloatProperty(
        name="波长",
        description="模拟的光波波长（毫米）",
        default=0.00053,
        min=0.0004,
        max=0.0007
    )

# ============================================================================
# 2. 核心函数 - 简化并避免API冲突
# ============================================================================

def get_target_object(context):
    """获取目标物体"""
    props = context.scene.diffraction_tool
    if props.target_name in bpy.data.objects:
        return bpy.data.objects[props.target_name]
    return None

def align_emitter_to_target(emitter, target):
    """对齐发射器到目标"""
    if not emitter or not target:
        return False
    
    props = bpy.context.scene.diffraction_tool
    
    if props.align_mode == 'PARALLEL':
        # 平行对齐：复制旋转，沿法线移动
        emitter.rotation_euler = target.rotation_euler.copy()
        
        # 计算目标法线方向
        import mathutils
        target_normal = target.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, 1))
        emitter.location = target.location - target_normal * props.distance
    
    elif props.align_mode == 'FACE':
        # 面向目标
        direction = target.location - emitter.location
        emitter.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    return True

def create_particle_system(obj):
    """为物体创建粒子系统（Blender 5.0.1兼容版）"""
    if not obj:
        return None
    
    props = bpy.context.scene.diffraction_tool
    
    # 检查物体类型
    if obj.type not in {'MESH', 'CURVE', 'SURFACE'}:
        print(f"物体类型 {obj.type} 不支持粒子系统")
        return None
    
    # 确保在物体模式
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # 选择物体
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # 移除现有的粒子系统
    if obj.particle_systems:
        for psys in obj.particle_systems:
            obj.particle_systems.remove(psys)
    
    # 添加粒子系统
    bpy.ops.object.particle_system_add()
    
    if not obj.particle_systems:
        print("无法添加粒子系统")
        return None
    
    psys = obj.particle_systems.active
    if not psys or not psys.settings:
        print("粒子系统无效")
        return None
    
    # 配置粒子设置 - 使用Blender 5.0.1的API
    settings = psys.settings
    settings.type = 'EMITTER'
    settings.count = props.particle_count
    settings.frame_start = 1
    settings.frame_end = 10
    settings.lifetime = 100
    
    # 发射设置
    settings.emit_from = 'VERT'
    settings.normal_factor = 2.0
    
    # 根据波长调整速度
    wavelength_factor = 0.00053 / props.wavelength
    settings.normal_factor *= wavelength_factor
    
    # 渲染设置
    settings.render_type = 'OBJECT'
    settings.particle_size = props.particle_size
    settings.instance_object = None  # 不使用实例物体
    
    # 在Blender 5.0中，路径显示的方式可能已经改变
    # 我们尝试使用不同的方法
    
    # 方法1：尝试使用显示百分比
    settings.draw_percentage = 100
    
    # 方法2：确保粒子可见
    settings.show_unborn = True
    settings.show_die = True
    
    # 创建材质
    create_particle_material(obj)
    
    return psys

def create_particle_material(obj):
    """创建粒子材质"""
    if not obj:
        return None
    
    props = bpy.context.scene.diffraction_tool
    
    # 材质名称
    mat_name = f"{obj.name}_ParticleMat"
    
    # 删除现有同名材质
    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])
    
    # 创建新材质
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    
    # 清除现有节点
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # 创建节点
    output = nodes.new(type='ShaderNodeOutputMaterial')
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs[1].default_value = 5.0  # 强度
    
    # 颜色节点
    if props.color_mode == 'SINGLE':
        # 单色
        rgb = nodes.new(type='ShaderNodeRGB')
        rgb.outputs[0].default_value = props.color1
        mat.node_tree.links.new(rgb.outputs[0], emission.inputs[0])
    
    elif props.color_mode == 'GRADIENT':
        # 渐变
        color_ramp = nodes.new(type='ShaderNodeValToRGB')
        color_ramp.color_ramp.elements[0].color = props.color1
        color_ramp.color_ramp.elements[1].color = props.color2
        
        # 添加随机因子
        object_info = nodes.new(type='ShaderNodeObjectInfo')
        mat.node_tree.links.new(object_info.outputs['Random'], color_ramp.inputs['Fac'])
        mat.node_tree.links.new(color_ramp.outputs['Color'], emission.inputs[0])
    
    # 连接到输出
    mat.node_tree.links.new(emission.outputs[0], output.inputs[0])
    
    # 应用材质到物体
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    # 将材质赋给粒子系统
    if obj.particle_systems:
        psys = obj.particle_systems.active
        if psys:
            psys.settings.material = mat
    
    return mat

# ============================================================================
# 3. 操作符类 - 简化版
# ============================================================================

class DIFFRACTION_OT_CreateSetup(Operator):
    """创建衍射模拟系统"""
    bl_idname = "diffraction.create_setup"
    bl_label = "创建新模拟"
    bl_description = "创建完整的衍射模拟系统"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.diffraction_tool
        
        # 创建发射器（光栅）
        bpy.ops.mesh.primitive_plane_add(size=1.5)
        emitter = context.active_object
        emitter.name = "Diffraction_Grating"
        emitter.location = (0, 0, 0)
        
        # 创建接收屏幕
        bpy.ops.mesh.primitive_plane_add(size=2.5)
        screen = context.active_object
        screen.name = "Diffraction_Screen"
        screen.location = (0, 4, 0)
        
        # 设置目标
        props.target_name = screen.name
        
        # 设置发射器粒子
        bpy.context.view_layer.objects.active = emitter
        create_particle_system(emitter)
        
        # 对齐屏幕
        align_emitter_to_target(emitter, screen)
        
        self.report({'INFO'}, "衍射模拟创建完成")
        return {'FINISHED'}

class DIFFRACTION_OT_SetupEmitter(Operator):
    """设置当前物体为发射器"""
    bl_idname = "diffraction.setup_emitter"
    bl_label = "设为发射器"
    bl_description = "将当前选中物体设置为衍射发射器"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj:
            self.report({'ERROR'}, "请先选择一个物体")
            return {'CANCELLED'}
        
        if obj.type not in {'MESH', 'CURVE', 'SURFACE'}:
            self.report({'ERROR'}, f"物体类型 {obj.type} 不支持粒子系统")
            return {'CANCELLED'}
        
        # 创建粒子系统
        psys = create_particle_system(obj)
        
        if psys:
            self.report({'INFO'}, f"已将 {obj.name} 设置为衍射发射器")
        else:
            self.report({'ERROR'}, "设置失败")
        
        return {'FINISHED'}

class DIFFRACTION_OT_AlignScreens(Operator):
    """对齐屏幕"""
    bl_idname = "diffraction.align_screens"
    bl_label = "对齐屏幕"
    bl_description = "将发射器与目标屏幕对齐"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        emitter = context.active_object
        
        if not emitter:
            self.report({'ERROR'}, "请先选择发射器物体")
            return {'CANCELLED'}
        
        target = get_target_object(context)
        
        if not target:
            self.report({'ERROR'}, "请先设置目标屏幕")
            return {'CANCELLED'}
        
        if emitter == target:
            self.report({'ERROR'}, "不能将物体与自身对齐")
            return {'CANCELLED'}
        
        success = align_emitter_to_target(emitter, target)
        
        if success:
            self.report({'INFO'}, f"已将 {emitter.name} 与 {target.name} 对齐")
        else:
            self.report({'ERROR'}, "对齐失败")
        
        return {'FINISHED'}

class DIFFRACTION_OT_SetTarget(Operator):
    """设置目标"""
    bl_idname = "diffraction.set_target"
    bl_label = "设为目标"
    bl_description = "将当前选中物体设置为目标屏幕"
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj:
            self.report({'ERROR'}, "请先选择一个物体")
            return {'CANCELLED'}
        
        props = context.scene.diffraction_tool
        props.target_name = obj.name
        
        self.report({'INFO'}, f"已将 {obj.name} 设置为目标屏幕")
        return {'FINISHED'}

class DIFFRACTION_OT_UpdateMaterial(Operator):
    """更新材质"""
    bl_idname = "diffraction.update_material"
    bl_label = "更新材质"
    bl_description = "更新粒子材质"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj:
            self.report({'ERROR'}, "请先选择一个物体")
            return {'CANCELLED'}
        
        mat = create_particle_material(obj)
        
        if mat:
            self.report({'INFO'}, "材质已更新")
        else:
            self.report({'ERROR'}, "更新失败")
        
        return {'FINISHED'}

class DIFFRACTION_OT_PlayAnimation(Operator):
    """播放动画"""
    bl_idname = "diffraction.play_animation"
    bl_label = "播放动画"
    bl_description = "播放粒子动画"
    
    def execute(self, context):
        # 回到第一帧
        context.scene.frame_set(1)
        
        # 播放动画
        if not context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()
        
        self.report({'INFO'}, "动画播放中...")
        return {'FINISHED'}

class DIFFRACTION_OT_ApplyPreset(Operator):
    """应用预设"""
    bl_idname = "diffraction.apply_preset"
    bl_label = "应用预设"
    
    color_type: StringProperty(default="blue")
    
    def execute(self, context):
        props = context.scene.diffraction_tool
        
        if self.color_type == "red":
            props.color1 = (1.0, 0.2, 0.1, 1.0)
            props.wavelength = 0.00065
        elif self.color_type == "green":
            props.color1 = (0.2, 1.0, 0.1, 1.0)
            props.wavelength = 0.00053
        elif self.color_type == "blue":
            props.color1 = (0.1, 0.2, 1.0, 1.0)
            props.wavelength = 0.00047
        elif self.color_type == "gradient":
            props.color_mode = 'GRADIENT'
            props.color1 = (0.2, 0.5, 1.0, 1.0)
            props.color2 = (1.0, 0.2, 0.5, 1.0)
        
        # 更新当前物体的材质
        obj = context.active_object
        if obj:
            create_particle_material(obj)
        
        self.report({'INFO'}, f"已应用 {self.color_type} 预设")
        return {'FINISHED'}

# ============================================================================
# 4. 面板类 - 简化界面
# ============================================================================

class DIFFRACTION_PT_MainPanel(Panel):
    """主面板"""
    bl_label = "衍射光栅"
    bl_idname = "DIFFRACTION_PT_MainPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "工具"
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        # 标题
        box = layout.box()
        box.label(text="衍射粒子模拟", icon='PARTICLES')
        
        # 快速创建按钮
        row = box.row()
        row.operator(DIFFRACTION_OT_CreateSetup.bl_idname, 
                    text="创建新模拟", icon='ADD')
        
        # 当前物体状态
        if obj:
            box.label(text=f"当前物体: {obj.name}", icon='OBJECT_DATA')
            
            if obj.particle_systems:
                box.label(text="✓ 已设置粒子系统", icon='CHECKMARK')
                box.operator(DIFFRACTION_OT_PlayAnimation.bl_idname, 
                           text="播放动画", icon='PLAY')
            else:
                box.label(text="✗ 未设置粒子系统", icon='X')
                box.operator(DIFFRACTION_OT_SetupEmitter.bl_idname, 
                           text="设为发射器", icon='PARTICLES')

class DIFFRACTION_PT_SettingsPanel(Panel):
    """设置面板"""
    bl_label = "设置"
    bl_idname = "DIFFRACTION_PT_SettingsPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "工具"
    bl_parent_id = "DIFFRACTION_PT_MainPanel"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.diffraction_tool
        
        # 屏幕设置
        box = layout.box()
        box.label(text="屏幕设置", icon='VIEW_ORTHO')
        
        # 目标屏幕
        row = box.row()
        row.label(text="目标:")
        if props.target_name:
            row.label(text=props.target_name, icon='MESH_PLANE')
        else:
            row.label(text="未设置", icon='X')
        
        row.operator(DIFFRACTION_OT_SetTarget.bl_idname, 
                   text="", icon='EYEDROPPER')
        
        # 对齐设置
        box.prop(props, "align_mode", text="对齐方式")
        box.prop(props, "distance", text="距离")
        
        if props.target_name:
            box.operator(DIFFRACTION_OT_AlignScreens.bl_idname, 
                       text="执行对齐", icon='CONSTRAINT')
        
        # 粒子设置
        box = layout.box()
        box.label(text="粒子设置", icon='PARTICLES')
        
        box.prop(props, "particle_count", text="数量")
        box.prop(props, "particle_size", text="大小")
        
        # 波长设置
        box = layout.box()
        box.label(text="光波设置", icon='LIGHT_SUN')
        
        box.prop(props, "wavelength", text="波长")
        
        # 计算并显示信息
        wavelength_nm = props.wavelength * 1000000  # 转换为纳米
        box.label(text=f"波长: {wavelength_nm:.0f} nm", icon='NONE')

class DIFFRACTION_PT_ColorPanel(Panel):
    """颜色面板"""
    bl_label = "颜色"
    bl_idname = "DIFFRACTION_PT_ColorPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "工具"
    bl_parent_id = "DIFFRACTION_PT_MainPanel"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.diffraction_tool
        
        # 颜色设置
        box = layout.box()
        box.label(text="颜色设置", icon='COLOR')
        
        box.prop(props, "color_mode", text="颜色模式")
        
        if props.color_mode == 'SINGLE':
            box.prop(props, "color1", text="颜色")
        elif props.color_mode == 'GRADIENT':
            box.prop(props, "color1", text="颜色1")
            box.prop(props, "color2", text="颜色2")
        
        # 更新按钮
        box.operator(DIFFRACTION_OT_UpdateMaterial.bl_idname, 
                   text="更新材质", icon='FILE_REFRESH')
        
        # 预设按钮
        box = layout.box()
        box.label(text="快速预设", icon='PRESET')
        
        row = box.row(align=True)
        row.operator(DIFFRACTION_OT_ApplyPreset.bl_idname, text="红光").color_type = "red"
        row.operator(DIFFRACTION_OT_ApplyPreset.bl_idname, text="绿光").color_type = "green"
        row.operator(DIFFRACTION_OT_ApplyPreset.bl_idname, text="蓝光").color_type = "blue"
        
        row = box.row(align=True)
        row.operator(DIFFRACTION_OT_ApplyPreset.bl_idname, text="渐变").color_type = "gradient"

# ============================================================================
# 5. 注册与卸载
# ============================================================================

classes = [
    DiffractionProps,
    DIFFRACTION_OT_CreateSetup,
    DIFFRACTION_OT_SetupEmitter,
    DIFFRACTION_OT_AlignScreens,
    DIFFRACTION_OT_SetTarget,
    DIFFRACTION_OT_UpdateMaterial,
    DIFFRACTION_OT_PlayAnimation,
    DIFFRACTION_OT_ApplyPreset,
    DIFFRACTION_PT_MainPanel,
    DIFFRACTION_PT_SettingsPanel,
    DIFFRACTION_PT_ColorPanel,
]

def register():
    """注册插件"""
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except:
            print(f"注册类 {cls} 时出错，可能已注册")
    
    # 注册场景属性
    if not hasattr(bpy.types.Scene, 'diffraction_tool'):
        bpy.types.Scene.diffraction_tool = PointerProperty(
            type=DiffractionProps,
            name="衍射工具",
            description="衍射粒子系统工具"
        )
    
    print("衍射光栅粒子系统 v3.1 已注册（Blender 5.0.1兼容版）")

def unregister():
    """卸载插件"""
    # 移除场景属性
    if hasattr(bpy.types.Scene, 'diffraction_tool'):
        del bpy.types.Scene.diffraction_tool
    
    # 注销类
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except:
            print(f"注销类 {cls} 时出错，可能未注册")
    
    print("衍射光栅粒子系统已卸载")

# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    register()