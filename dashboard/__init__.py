"""
Web界面模块 - 提供基于Dash的交互式Web界面
"""

from .app import app
from .layouts import main_layout, create_header, create_footer
from .callbacks import register_callbacks

__all__ = ['app', 'main_layout', 'create_header', 'create_footer', 'register_callbacks']