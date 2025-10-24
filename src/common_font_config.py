"""
通用字体配置模块，用于解决matplotlib字体警告问题
"""
import matplotlib.pyplot as plt
import logging
import os
import warnings

# 全局过滤所有字体相关警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*findfont:.*")
warnings.filterwarnings("ignore", message=".*font.*")
warnings.filterwarnings("ignore", message=".*Font.*")
warnings.filterwarnings("ignore", message=".*text.usetex.*")
warnings.filterwarnings("ignore", message=".*Matplotlib is building the font cache.*")

# 设置非交互式后端，避免显示相关警告
plt.switch_backend('agg')

def setup_matplotlib_fonts():
    """
    配置matplotlib使用系统可用字体，避免字体警告
    """
    # 尽量使用非交互式后端，避免显示相关警告
    try:
        plt.switch_backend('agg')
    except Exception:
        pass

    # 避免访问私有或不稳定的字体管理器内部结构，直接使用通用配置

    # 直接设置通用字体族与常见备选，不依赖FontManager内部结构
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "DejaVu Sans", "Liberation Sans", "FreeSans", "Noto Sans", "Ubuntu", "sans"
    ]
    plt.rcParams['font.serif'] = ["DejaVu Serif", "serif"]
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 再次禁用常见的字体相关警告
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*findfont:.*")
    warnings.filterwarnings("ignore", message=".*font.*")
    warnings.filterwarnings("ignore", message=".*Font.*")

    logging.info("已配置matplotlib通用字体族，移除对私有属性的访问")