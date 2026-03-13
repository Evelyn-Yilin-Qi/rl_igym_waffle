"""
配置工具函数
用于从新格式的配置文件中提取启用的组件
"""
from omegaconf import OmegaConf


def get_enabled_component(config, component_type="components"):
    """
    从配置中获取启用的组件
    
    Args:
        config: OmegaConf配置对象，包含 components 列表
        component_type: 组件类型键名，默认为 "components"
    
    Returns:
        tuple: (name, params) - 启用的组件名称和参数字典
    
    Raises:
        ValueError: 如果没有找到启用的组件，或者有多个启用的组件
    """
    # 获取 components 列表（OmegaConf 支持属性访问）
    components = getattr(config, component_type, [])
    
    # 转换为普通 Python 列表以便处理
    components_list = OmegaConf.to_container(components, resolve=True) if hasattr(OmegaConf, 'to_container') else list(components)
    
    enabled_components = [comp for comp in components_list if comp.get('enabled', False)]
    
    if len(enabled_components) == 0:
        available_names = [c.get('name', 'unknown') for c in components_list]
        raise ValueError(f"No enabled component found in config. Available components: {available_names}")
    
    if len(enabled_components) > 1:
        enabled_names = [c.get('name', 'unknown') for c in enabled_components]
        raise ValueError(f"Multiple enabled components found: {enabled_names}. Only one component should be enabled at a time.")
    
    enabled_comp = enabled_components[0]
    name = enabled_comp.get('name')
    params = enabled_comp.get('params', {})
    
    return name, params
