import pygame
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import math
import time
import platform
import tkinter as tk
from tkinter import filedialog
import random
import io

# 图像处理函数
def generate_normal_map(image_path, output_path, strength=5.0):
    """生成法向贴图"""
    try:
        img = Image.open(image_path).convert('L')  # 转换为灰度图
        width, height = img.size
        pixels = np.array(img)
        
        # 创建法向贴图数组
        normal_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(1, height-1):
            for x in range(1, width-1):
                # 计算梯度
                dx = (pixels[y, x+1] - pixels[y, x-1]) / 255.0
                dy = (pixels[y+1, x] - pixels[y-1, x]) / 255.0
                
                # 计算法向量
                dz = 1.0 / strength
                length = math.sqrt(dx*dx + dy*dy + dz*dz)
                dx /= length
                dy /= length
                dz /= length
                
                # 转换到RGB范围 (0-255)
                r = int((dx + 1) * 127.5)
                g = int((dy + 1) * 127.5)
                b = int((dz + 1) * 127.5)
                
                normal_map[y, x] = [r, g, b]
        
        # 边界处理
        normal_map[0] = normal_map[1]
        normal_map[-1] = normal_map[-2]
        normal_map[:,0] = normal_map[:,1]
        normal_map[:,-1] = normal_map[:,-2]
        
        # 保存法向贴图
        result = Image.fromarray(normal_map)
        result.save(output_path)
        return True
    except Exception as e:
        print(f"生成法向贴图错误: {e}")
        return False

def apply_uniform_blur(image_path, output_path, radius=3):
    """应用均匀模糊"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        pixels = np.array(img)
        
        # 创建模糊后的图像数组
        blurred = np.zeros_like(pixels)
        
        # 应用简单盒式模糊
        for y in range(radius, height-radius):
            for x in range(radius, width-radius):
                # 计算邻域平均值
                neighborhood = pixels[y-radius:y+radius+1, x-radius:x+radius+1]
                blurred[y, x] = neighborhood.mean(axis=(0,1))
        
        # 边界处理
        for y in range(height):
            for x in range(width):
                if x < radius or x >= width-radius or y < radius or y >= height-radius:
                    min_x = max(0, x-radius)
                    max_x = min(width, x+radius+1)
                    min_y = max(0, y-radius)
                    max_y = min(height, y+radius+1)
                    neighborhood = pixels[min_y:max_y, min_x:max_x]
                    blurred[y, x] = neighborhood.mean(axis=(0,1))
        
        # 保存模糊后的图像
        result = Image.fromarray(blurred)
        result.save(output_path)
        return True
    except Exception as e:
        print(f"均匀模糊错误: {e}")
        return False

def apply_radial_blur(image_path, output_path, center=None, strength=0.02):
    """应用径向模糊"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        pixels = np.array(img)
        
        # 设置模糊中心（默认图像中心）
        if center is None:
            center = (width // 2, height // 2)
        cx, cy = center
        
        # 创建模糊后的图像数组
        blurred = np.zeros_like(pixels)
        
        # 应用径向模糊
        for y in range(height):
            for x in range(width):
                # 计算当前点到中心的向量
                dx = x - cx
                dy = y - cy
                distance = math.sqrt(dx*dx + dy*dy)
                
                # 如果点在中心，直接使用原像素
                if distance == 0:
                    blurred[y, x] = pixels[y, x]
                    continue
                
                # 归一化方向向量
                dx /= distance
                dy /= distance
                
                # 采样多个点并平均
                total = np.zeros(3, dtype=np.float32)
                count = 0
                
                # 沿着方向采样
                for i in range(int(distance * strength) + 1):
                    sample_x = int(x - dx * i)
                    sample_y = int(y - dy * i)
                    
                    if 0 <= sample_x < width and 0 <= sample_y < height:
                        total += pixels[sample_y, sample_x]
                        count += 1
                
                if count > 0:
                    blurred[y, x] = total / count
                else:
                    blurred[y, x] = pixels[y, x]
        
        # 保存径向模糊后的图像
        result = Image.fromarray(blurred)
        result.save(output_path)
        return True
    except Exception as e:
        print(f"径向模糊错误: {e}")
        return False

def generate_noise_image(width=256, height=256):
    """生成随机噪声图像"""
    # 创建随机噪声数组
    noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    
    # 转换为PIL图像
    img = Image.fromarray(noise, 'L')  # 'L' 表示灰度模式
    return img

def fractal_brownian_motion(noise_images, output_path):
    """分形布朗运动 - 叠加多个噪声图生成新的噪声图"""
    if not noise_images:
        return False
    
    try:
        # 确保所有噪声图尺寸相同（使用第一张图的尺寸）
        width, height = noise_images[0].size
        
        # 创建结果数组
        result = np.zeros((height, width), dtype=np.float32)
        
        # 设置权重（指数衰减）
        weights = [1.0 / (2 ** i) for i in range(len(noise_images))]
        total_weight = sum(weights)
        
        # 归一化权重
        weights = [w / total_weight for w in weights]
        
        # 叠加所有噪声图
        for i, img in enumerate(noise_images):
            # 确保图像是灰度图
            if img.mode != 'L':
                img = img.convert('L')
            
            # 调整图像尺寸
            img = img.resize((width, height))
            
            # 转换为数组并应用权重
            arr = np.array(img, dtype=np.float32) * weights[i]
            result += arr
        
        # 归一化到0-255范围
        result = (result - result.min()) / (result.max() - result.min()) * 255
        result = result.astype(np.uint8)
        
        # 保存结果
        fbm_img = Image.fromarray(result, 'L')
        fbm_img.save(output_path)
        return True
    except Exception as e:
        print(f"分形布朗运动错误: {e}")
        return False

def find_chinese_font():
    """尝试查找系统中支持中文的字体"""
    # 常见中文字体路径
    font_paths = [
        # Windows 系统字体
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
        
        # macOS 系统字体
        "/System/Library/Fonts/PingFang.ttc",  # 苹方
        "/System/Library/Fonts/STHeiti Light.ttc",  # 华文黑体
        "/System/Library/Fonts/STHeiti Medium.ttc",
        
        # Linux 系统字体
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",    # 文泉驿正黑
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans CJK
        
        # 跨平台通用字体
        "simhei.ttf",  # 尝试当前目录下的字体
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            return path
    
    # 如果找不到任何中文字体，尝试使用默认字体
    try:
        return pygame.font.get_default_font()
    except:
        return None

class ImageProcessor:
    def __init__(self):
        pygame.init()
        self.width, self.height = 1000, 850  # 增加高度以容纳新按钮
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("简易处理与生成")
        
        # 查找并加载中文字体
        self.font_path = find_chinese_font()
        self.font = None
        self.small_font = None
        
        if self.font_path:
            try:
                # 加载中文字体
                self.font = pygame.font.Font(self.font_path, 24)
                self.small_font = pygame.font.Font(self.font_path, 18)
                print(f"使用字体: {self.font_path}")
            except Exception as e:
                print(f"加载字体失败: {e}")
                self.font = pygame.font.SysFont(None, 24)
                self.small_font = pygame.font.SysFont(None, 18)
        else:
            print("警告：找不到中文字体，使用系统默认字体")
            self.font = pygame.font.SysFont(None, 24)
            self.small_font = pygame.font.SysFont(None, 18)
        
        # 状态变量
        self.image_path = None
        self.image_surface = None
        self.output_dir = "output"
        self.status = "就绪 - 点击'上传图片'选择图像"
        self.processing = False
        self.process_type = ""
        self.last_process_time = 0
        
        # 噪声图相关变量
        self.noise_preview_images = []  # 存储预览的噪声图
        self.noise_counter = 0  # 噪声图计数器
        self.fbm_counter = 0    # 分形布朗运动计数器
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 按钮定义 - 第一行
        self.buttons_row1 = [
            {"id": "upload", "rect": pygame.Rect(50, 30, 150, 50), "text": "上传图片", "color": (70, 130, 180)},
            {"id": "normal", "rect": pygame.Rect(220, 30, 150, 50), "text": "法向贴图", "color": (60, 179, 113)},
            {"id": "blur", "rect": pygame.Rect(390, 30, 150, 50), "text": "均匀模糊", "color": (218, 165, 32)},
            {"id": "radial", "rect": pygame.Rect(560, 30, 150, 50), "text": "径向模糊", "color": (205, 92, 92)},
        ]
        
        # 按钮定义 - 第二行（新功能）
        self.buttons_row2 = [
            {"id": "noise", "rect": pygame.Rect(50, 100, 150, 50), "text": "生成噪声图", "color": (100, 150, 200)},
            {"id": "fbm", "rect": pygame.Rect(220, 100, 150, 50), "text": "分形布朗运动", "color": (150, 100, 200)},
        ]
        
        # 按钮定义 - 清除按钮（底部右侧）
        self.clear_buttons = [
            {"id": "clear_original", "rect": pygame.Rect(600, 750, 120, 40), "text": "清除预览", "color": (200, 100, 100)},
            {"id": "clear_noise", "rect": pygame.Rect(730, 750, 120, 40), "text": "清除噪音预览", "color": (200, 100, 100)},
            {"id": "clear_all", "rect": pygame.Rect(860, 750, 120, 40), "text": "全部清除", "color": (200, 100, 100)},
        ]
        
        # 合并所有按钮
        self.all_buttons = self.buttons_row1 + self.buttons_row2 + self.clear_buttons
    
    def open_file_dialog(self):
        """打开文件对话框让用户选择图像"""
        try:
            # 初始化Tkinter
            root = tk.Tk()
            root.withdraw()  # 隐藏主窗口
            
            # 设置文件类型
            file_types = [
                ("图片文件", "*.jpg;*.jpeg;*.png;*.bmp;*.tga;*.tif"),
                ("所有文件", "*.*")
            ]
            
            # 打开文件对话框
            file_path = filedialog.askopenfilename(
                title="选择图片",
                filetypes=file_types
            )
            
            # 销毁Tkinter窗口
            root.destroy()
            
            return file_path
        except Exception as e:
            print(f"打开文件对话框失败: {e}")
            return None
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and not self.processing:  # 左键点击且不在处理中
                    pos = pygame.mouse.get_pos()
                    
                    # 检查按钮点击
                    for button in self.all_buttons:
                        if button["rect"].collidepoint(pos):
                            if button["id"] == "upload":  # 上传图片
                                self.upload_image()
                            elif button["id"] == "normal":  # 法向贴图
                                self.start_processing("normal")
                            elif button["id"] == "blur":  # 均匀模糊
                                self.start_processing("uniform_blur")
                            elif button["id"] == "radial":  # 径向模糊
                                self.start_processing("radial_blur")
                            elif button["id"] == "noise":  # 生成噪声图
                                self.generate_noise()
                            elif button["id"] == "fbm":    # 分形布朗运动
                                self.start_processing("fbm")
                            elif button["id"] == "clear_original":  # 清除原始预览
                                self.clear_original_preview()
                            elif button["id"] == "clear_noise":    # 清除噪音预览
                                self.clear_noise_preview()
                            elif button["id"] == "clear_all":       # 全部清除
                                self.clear_all_previews()
    
    def upload_image(self):
        """让用户选择并上传图片"""
        try:
            # 打开文件对话框
            file_path = self.open_file_dialog()
            
            if not file_path:
                self.status = "未选择图片"
                return
            
            # 验证文件是否存在
            if not os.path.exists(file_path):
                self.status = f"文件不存在: {file_path}"
                return
            
            # 验证是否是图片文件
            try:
                with Image.open(file_path) as img:
                    img.verify()  # 验证文件完整性
            except Exception as e:
                self.status = f"无效的图片文件: {str(e)}"
                return
            
            self.image_path = file_path
            self.status = f"已上传图片: {os.path.basename(self.image_path)}"
            
            # 加载图片用于显示
            try:
                self.image_surface = pygame.image.load(self.image_path).convert()
            except pygame.error as e:
                self.status = f"加载图片错误: {str(e)}"
                self.image_surface = None
        except Exception as e:
            self.status = f"上传错误: {str(e)}"
    
    def generate_noise(self):
        """生成随机噪声图并添加到预览池"""
        try:
            # 生成噪声图
            noise_img = generate_noise_image(256, 256)
            
            # 保存噪声图
            self.noise_counter += 1
            noise_path = os.path.join(self.output_dir, f"noise_{self.noise_counter}.png")
            noise_img.save(noise_path)
            
            # 将PIL图像转换为Pygame表面
            img_bytes = io.BytesIO()
            noise_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            noise_surface = pygame.image.load(img_bytes)
            
            # 添加到预览池（最多4张）
            if len(self.noise_preview_images) >= 4:
                # 如果已经有4张，先清除所有预览图
                self.noise_preview_images = []
            
            # 添加新噪声图
            self.noise_preview_images.append({
                "surface": noise_surface,
                "image": noise_img,
                "path": noise_path
            })
            
            self.status = f"已生成噪声图: noise_{self.noise_counter}.png"
            return True
        except Exception as e:
            self.status = f"生成噪声图错误: {str(e)}"
            return False
    
    def clear_original_preview(self):
        """清除原始图像预览"""
        self.image_path = None
        self.image_surface = None
        self.status = "已清除原始图像预览"
    
    def clear_noise_preview(self):
        """清除噪声预览池"""
        self.noise_preview_images = []
        self.status = "已清除噪声预览池"
    
    def clear_all_previews(self):
        """清除所有预览"""
        self.image_path = None
        self.image_surface = None
        self.noise_preview_images = []
        self.status = "已清除所有预览"
    
    def start_processing(self, process_type):
        if process_type == "fbm":
            # 特殊处理分形布朗运动
            if not self.noise_preview_images:
                self.status = "错误: 请先生成噪声图"
                return
        else:
            # 其他处理需要上传的图片
            if not self.image_path:
                self.status = "错误: 请先上传图片"
                return
            
            if not self.image_surface:
                self.status = "错误: 图片加载失败，请重新上传"
                return
        
        self.processing = True
        self.process_type = process_type
        self.status = f"正在处理{self.get_process_name(process_type)}..."
        self.last_process_time = time.time()
    
    def get_process_name(self, process_type):
        names = {
            "normal": "法向贴图",
            "uniform_blur": "均匀模糊",
            "radial_blur": "径向模糊",
            "fbm": "分形布朗运动"
        }
        return names.get(process_type, "处理")
    
    def do_processing(self):
        if not self.processing or not self.process_type:
            return
        
        # 确保有足够的时间更新UI
        if time.time() - self.last_process_time < 0.5:
            return
        
        success = False
        
        if self.process_type == "normal":
            # 创建输出文件名
            file_name = os.path.basename(self.image_path)
            file_base, file_ext = os.path.splitext(file_name)
            output_path = os.path.join(self.output_dir, f"{file_base}_normal{file_ext}")
            success = generate_normal_map(self.image_path, output_path)
        elif self.process_type == "uniform_blur":
            file_name = os.path.basename(self.image_path)
            file_base, file_ext = os.path.splitext(file_name)
            output_path = os.path.join(self.output_dir, f"{file_base}_blur{file_ext}")
            success = apply_uniform_blur(self.image_path, output_path, radius=5)
        elif self.process_type == "radial_blur":
            file_name = os.path.basename(self.image_path)
            file_base, file_ext = os.path.splitext(file_name)
            output_path = os.path.join(self.output_dir, f"{file_base}_radial{file_ext}")
            success = apply_radial_blur(self.image_path, output_path, strength=0.03)
        elif self.process_type == "fbm":
            # 分形布朗运动
            self.fbm_counter += 1
            output_path = os.path.join(self.output_dir, f"fbm_{self.fbm_counter}.png")
            
            # 提取所有噪声图像对象
            noise_images = [item["image"] for item in self.noise_preview_images]
            
            # 执行分形布朗运动
            success = fractal_brownian_motion(noise_images, output_path)
            
            if success:
                # 加载新生成的FBM图像
                try:
                    # 使用字节流方法加载图像
                    fbm_img = Image.open(output_path)
                    
                    # 转换为Pygame表面
                    img_bytes = io.BytesIO()
                    fbm_img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    fbm_surface = pygame.image.load(img_bytes)
                    
                    # 添加到预览池（最多4张）
                    if len(self.noise_preview_images) >= 4:
                        # 如果已经有4张，先清除所有预览图
                        self.noise_preview_images = []
                    
                    # 添加新生成的FBM图像
                    self.noise_preview_images.append({
                        "surface": fbm_surface,
                        "image": fbm_img,
                        "path": output_path
                    })
                except Exception as e:
                    print(f"加载FBM图像错误: {e}")
                    success = False
        
        if success:
            self.status = f"{self.get_process_name(self.process_type)}已保存至: {os.path.basename(output_path)}"
        else:
            self.status = f"{self.get_process_name(self.process_type)}处理失败"
        
        self.processing = False
        self.process_type = ""
    
    def draw_text(self, text, font, color, x, y, centered=False):
        """安全绘制文本的方法"""
        try:
            if not font:
                # 如果字体未加载，使用默认字体
                font = pygame.font.SysFont(None, 24)
            
            text_surface = font.render(text, True, color)
            if centered:
                text_rect = text_surface.get_rect(center=(x, y))
                self.screen.blit(text_surface, text_rect)
            else:
                self.screen.blit(text_surface, (x, y))
        except Exception as e:
            print(f"绘制文本错误: {e}")
            # 在错误位置绘制一个红色矩形作为错误指示
            pygame.draw.rect(self.screen, (255, 0, 0), (x, y, 200, 30))
    
    def draw_ui(self):
        # 绘制背景
        self.screen.fill((30, 30, 40))
                
        # 绘制第一行按钮
        for button in self.buttons_row1:
            color = button["color"]
            if self.processing:
                # 处理中时按钮变暗
                color = tuple(max(0, c - 80) for c in color)
            
            pygame.draw.rect(self.screen, color, button["rect"], border_radius=10)
            pygame.draw.rect(self.screen, (200, 200, 200), button["rect"], 2, border_radius=10)
            
            # 绘制按钮文本
            self.draw_text(button["text"], self.font, (255, 255, 255), 
                          button["rect"].centerx, button["rect"].centery, centered=True)
        
        # 绘制第二行按钮
        for button in self.buttons_row2:
            color = button["color"]
            if self.processing:
                # 处理中时按钮变暗
                color = tuple(max(0, c - 80) for c in color)
            
            pygame.draw.rect(self.screen, color, button["rect"], border_radius=10)
            pygame.draw.rect(self.screen, (200, 200, 200), button["rect"], 2, border_radius=10)
            
            # 绘制按钮文本
            self.draw_text(button["text"], self.font, (255, 255, 255), 
                          button["rect"].centerx, button["rect"].centery, centered=True)
        
        # 绘制清除按钮
        for button in self.clear_buttons:
            pygame.draw.rect(self.screen, button["color"], button["rect"], border_radius=8)
            pygame.draw.rect(self.screen, (200, 200, 200), button["rect"], 2, border_radius=8)
            
            # 使用小字体绘制按钮文本
            self.draw_text(button["text"], self.small_font, (255, 255, 255), 
                          button["rect"].centerx, button["rect"].centery, centered=True)
        
        # 绘制状态信息
        self.draw_text(self.status, self.font, (220, 220, 100), 50, 170)
        
        # 绘制说明
        instructions = [
            "使用说明:",
            "1. 点击'上传图片'按钮选择图片",
            "2. 点击'法向贴图'生成法向贴图",
            "3. 点击'均匀模糊'应用均匀模糊",
            "4. 点击'径向模糊'应用径向模糊",
            "5. 点击'生成噪声图'创建随机噪声",
            "6. 点击'分形布朗运动'叠加噪声图",
            "7. 使用底部按钮清除预览内容",
            "处理结果保存在output目录中"
        ]
        
        for i, text in enumerate(instructions):
            self.draw_text(text, self.small_font, (180, 180, 255), 50, 220 + i * 30)
        
        # 绘制原始图像预览
        preview_x = 600
        preview_y = 170  # 下移预览区，避免覆盖按钮
        preview_size = (250, 250)  # 缩小预览框尺寸
        
        if self.image_surface:
            # 调整图像大小以适应预览区域
            scaled_img = pygame.transform.scale(self.image_surface, preview_size)
            self.screen.blit(scaled_img, (preview_x, preview_y))
        else:
            # 如果没有图片，显示提示信息
            self.draw_text("无原始图像", self.small_font, (150, 150, 200), 
                          preview_x + preview_size[0] // 2, preview_y + preview_size[1] // 2, centered=True)
        
        # 绘制预览框
        pygame.draw.rect(self.screen, (100, 100, 150), 
                        (preview_x - 10, preview_y - 10, 
                         preview_size[0] + 20, preview_size[1] + 20), 2, border_radius=8)
        
        # 绘制预览标题
        self.draw_text("原始图像预览", self.small_font, (200, 200, 200), 
                      preview_x + preview_size[0] // 2, preview_y - 25, centered=True)
        
        # 绘制噪声图预览区域
        noise_preview_x = 600
        noise_preview_y = 450  # 下移噪声预览区
        noise_preview_size = 120  # 每个预览小图的大小
        
        # 绘制预览框
        pygame.draw.rect(self.screen, (100, 100, 150), 
                        (noise_preview_x - 10, noise_preview_y - 10, 
                         noise_preview_size * 2 + 20, noise_preview_size * 2 + 20), 2, border_radius=8)
        
        # 绘制预览标题
        self.draw_text("噪声图预览池 (最多4张)", self.small_font, (200, 200, 200), 
                      noise_preview_x + noise_preview_size, noise_preview_y - 25, centered=True)
        
        # 绘制噪声图预览
        for i, noise in enumerate(self.noise_preview_images):
            row = i // 2
            col = i % 2
            x = noise_preview_x + col * noise_preview_size
            y = noise_preview_y + row * noise_preview_size
            
            # 调整图像大小以适应预览区域
            scaled_noise = pygame.transform.scale(noise["surface"], (noise_preview_size, noise_preview_size))
            self.screen.blit(scaled_noise, (x, y))
            
            # 绘制文件名
            file_name = os.path.basename(noise["path"])
            self.draw_text(file_name, self.small_font, (220, 220, 100), 
                          x + noise_preview_size // 2, y + noise_preview_size + 15, centered=True)
        
        # 如果没有噪声图，显示提示
        if not self.noise_preview_images:
            self.draw_text("无噪声图", self.small_font, (150, 150, 200), 
                          noise_preview_x + noise_preview_size, noise_preview_y + noise_preview_size, centered=True)
        
        # 绘制分隔线
        pygame.draw.line(self.screen, (100, 100, 150), (0, 90), (self.width, 90), 2)
        pygame.draw.line(self.screen, (100, 100, 150), (0, 160), (self.width, 160), 2)
        
        # 绘制底部信息
        self.draw_text("简易处理与生成 v1.0 by chuyueyu | 按ESC退出", self.small_font, (150, 150, 200), 
                      self.width // 2, self.height - 50, centered=True)
    
    def run(self):
        clock = pygame.time.Clock()
        
        while True:
            self.handle_events()
            
            # 处理图像（如果正在处理中）
            if self.processing:
                self.do_processing()
            
            self.draw_ui()
            pygame.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    app = ImageProcessor()
    app.run()