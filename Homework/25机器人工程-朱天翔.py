#感谢我亲密无间的AI战友！！！

import cv2
import numpy as np
import json
import os
from pathlib import Path



class ColorRange:
    def __init__(self, config_path=None):
        self.color_ranges = {}
        if config_path:
            self.load_config(config_path)
        else:
            self.set_default_ranges()
    
    def load_config(self, config_path):
        try:
            if not os.path.exists(config_path):
                homework_config = os.path.join(os.path.dirname(__file__), "config.json")
                if os.path.exists(homework_config):
                    config_path = homework_config
                else:
                    print(f"配置文件 {config_path} 未找到，使用默认颜色范围")
                    self.set_default_ranges()
                    return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            color_configs = config.get("color_ranges", {})
            for color_name, ranges in color_configs.items():
                self.add_color_range(color_name, ranges)
                
            print(f"成功加载配置文件: {config_path}")
                
        except Exception as e:
            print(f"读取配置文件错误: {e}，使用默认颜色范围")
            self.set_default_ranges()
    
    def set_default_ranges(self):
        self.add_color_range("红", [
            {"lower": [0, 100, 100], "upper": [10, 255, 255]},
            {"lower": [170, 100, 100], "upper": [180, 255, 255]}
        ])
        self.add_color_range("蓝", [
            {"lower": [100, 150, 50], "upper": [130, 255, 255]}
        ])
        self.add_color_range("紫", [
            {"lower": [130, 50, 50], "upper": [160, 255, 255]}
        ])
        print("使用默认颜色范围")
    
    def add_color_range(self, color_name, ranges):
        self.color_ranges[color_name] = []
        for range_dict in ranges:
            lower = np.array(range_dict["lower"], dtype=np.uint8)
            upper = np.array(range_dict["upper"], dtype=np.uint8)
            self.color_ranges[color_name].append((lower, upper))
    
    def get_color_mask(self, hsv_frame, color_name):
        if color_name not in self.color_ranges:
            return None
        
        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.color_ranges[color_name]:
            color_mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)
        
        return mask
    
    def get_all_color_masks(self, hsv_frame):
        masks = {}
        for color_name in self.color_ranges.keys():
            masks[color_name] = self.get_color_mask(hsv_frame, color_name)
        return masks


class BallDetector:
    def __init__(self, config_path=None):
        self.color_range = ColorRange(config_path)
        self.roi = None
        self.min_pixel_ratio = 0.01
        self.load_config(config_path)
    
    def load_config(self, config_path):
        try:
            if not config_path or not os.path.exists(config_path):
                homework_config = os.path.join(os.path.dirname(__file__), "config.json")
                if os.path.exists(homework_config):
                    config_path = homework_config
                else:
                    print(f"配置文件 {config_path} 未找到，使用默认参数")
                    self.roi = (300, 200, 200, 200)
                    self.min_pixel_ratio = 0.01
                    return
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            roi_config = config.get("roi", {})
            self.roi = (
                roi_config.get("x", 300),
                roi_config.get("y", 200),
                roi_config.get("width", 200),
                roi_config.get("height", 200)
            )
            
            self.min_pixel_ratio = config.get("min_pixel_ratio", 0.01)
            
            print(f"成功加载ROI配置: {self.roi}")
            print(f"最小像素比例: {self.min_pixel_ratio}")
            
        except Exception as e:
            print(f"读取配置文件错误: {e}，使用默认参数")
            self.roi = (300, 200, 200, 200)
            self.min_pixel_ratio = 0.01
    
    def set_roi(self, x, y, width, height):
        self.roi = (x, y, width, height)
    
    def process_frame(self, frame):
        # 旋转图像180度
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        result_frame = frame.copy()
        
        if self.roi:
            x, y, w, h = self.roi
            h_frame, w_frame = frame.shape[:2]
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)
            
            roi_frame = frame[y:y+h, x:x+w]
        else:
            roi_frame = frame
            x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
        
        hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        
        color_masks = self.color_range.get_all_color_masks(hsv_roi)
        
        color_counts = {}
        for color_name, mask in color_masks.items():
            if mask is not None:
                color_counts[color_name] = np.sum(mask > 0)
            else:
                color_counts[color_name] = 0
        
        total_pixels = w * h
        if total_pixels == 0:
            total_pixels = 1
        
        ball_state = "无球"
        max_ratio = 0
        
        for color_name, count in color_counts.items():
            ratio = count / total_pixels
            if ratio > self.min_pixel_ratio and ratio > max_ratio:
                max_ratio = ratio
                ball_state = color_name + "球"
        
        cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result_frame, f"状态: {ball_state}", (10, 30), font, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f"ROI像素比: {max_ratio:.3f}", (10, 70), font, 0.7, (255, 0, 0), 2)
        
        y_offset = 110
        for color_name, count in color_counts.items():
            ratio = count / total_pixels
            cv2.putText(result_frame, f"{color_name}: {count} ({ratio:.3f})", 
                       (10, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return result_frame, ball_state
    
    def try_open_video(self, video_path):
        """尝试使用不同的后端打开视频"""
        backends = [
            cv2.CAP_ANY,      # 自动选择
            cv2.CAP_FFMPEG,   # FFmpeg
            cv2.CAP_MSMF,     # Microsoft Media Foundation
            cv2.CAP_DSHOW,    # DirectShow
        ]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(video_path, backend)
                if cap.isOpened():
                    # 尝试读取一帧
                    ret, frame = cap.read()
                    if ret:
                        print(f"成功使用后端 {backend} 打开视频")
                        return cap
                    else:
                        cap.release()
                else:
                    if cap:
                        cap.release()
            except Exception as e:
                print(f"后端 {backend} 失败: {e}")
                if 'cap' in locals():
                    cap.release()
        
        return None
    
    def check_video_file(self, video_path):
        print(f"检查视频文件: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"文件不存在: {video_path}")
            return False
        
        file_size = os.path.getsize(video_path)
        print(f"文件大小: {file_size} 字节")
        if file_size == 0:
            print("文件大小为0，可能是空文件")
            return False
        
        # 尝试使用不同的后端打开视频       ！！！！！！！！！！



        cap = self.try_open_video(video_path)
        if cap is None:
            print("所有后端都无法打开视频文件")
            return False
        
        # 检查视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频属性: {width}x{height}, {fps:.2f} fps, 总帧数: {frame_count}")
        
        cap.release()
        return True
    
    def process_video(self, video_path, output_path=None, display=True):
        if not self.check_video_file(video_path):
            print(f"视频文件检查失败: {video_path}")
            return []
        
        # 使用最佳后端打开视频  ！！！！！！！！！！！！！！！！！！！



        cap = self.try_open_video(video_path)
        if cap is None:
            print(f"无法打开视频文件: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        
        print(f"开始处理视频: {video_path}")
        print(f"视频信息: {width}x{height}, {fps:.2f} fps, 总帧数: {total_frames}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, ball_state = self.process_frame(frame)
            results.append(ball_state)
            
            if writer:
                writer.write(processed_frame)
            
            if display:
                cv2.imshow('Ball Detection', processed_frame)
                key = cv2.waitKey(25) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
                elif key == ord('s'):
                    cv2.imwrite(f"frame_{frame_count}.jpg", processed_frame)
                    print(f"已保存第 {frame_count} 帧")
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"进度: {progress:.1f}% ({frame_count}/{total_frames}), 当前状态: {ball_state}")
        
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"视频处理完成，共处理 {frame_count} 帧")
        return results


def main():
    config_path = r"D:\朱天翔\华南理工大学\Github-clone\assignment1-basics-of-opencv-Aishiterus-z\Homework\config.json"
    
    detector = BallDetector(config_path)
    
    video_files = [
        r"D:\朱天翔\华南理工大学\Github-clone\assignment1-basics-of-opencv-Aishiterus-z\Homework\output.avi",
        r"D:\朱天翔\华南理工大学\Github-clone\assignment1-basics-of-opencv-Aishiterus-z\Homework\output1.avi"
    ]
    
    print("检查视频文件...")
    valid_videos = []
    for video_path in video_files:
        if detector.check_video_file(video_path):
            valid_videos.append(video_path)
        print("-" * 50)
    
    if not valid_videos:
        print("没有可用的视频文件，尝试使用相对路径...")
        # 尝试使用相对路径
        video_files_relative = [
            "output.avi",
            "output1.avi"
        ]
        for video_path in video_files_relative:
            if os.path.exists(video_path):
                print(f"找到相对路径视频: {video_path}")
                valid_videos.append(video_path)
    
    if not valid_videos:
        print("没有可用的视频文件，程序退出")
        return
    
    for i, video_path in enumerate(valid_videos):
        print(f"\n{'='*50}")
        print(f"处理第 {i+1} 个视频: {video_path}")
        print(f"{'='*50}")
        
        results = detector.process_video(video_path, 
                                       output_path=f"processed_output_{i+1}.avi",
                                       display=True)
        
        if results:
            state_counts = {}
            for state in results:
                state_counts[state] = state_counts.get(state, 0) + 1
            
            print(f"\n视频 {i+1} 检测结果统计:")
            for state, count in state_counts.items():
                percentage = count / len(results) * 100
                print(f"  {state}: {count} 帧 ({percentage:.1f}%)")
            
            results_file = f"video_{i+1}_results.txt"
            with open(results_file, "w", encoding="utf-8") as f:
                f.write(f"视频 {video_path} 检测结果统计:\n")
                for state, count in state_counts.items():
                    percentage = count / len(results) * 100
                    f.write(f"  {state}: {count} 帧 ({percentage:.1f}%)\n")
                
                f.write("\n逐帧检测结果:\n")
                for frame_idx, state in enumerate(results):
                    f.write(f"帧 {frame_idx}: {state}\n")
            
            print(f"详细结果已保存到: {results_file}")


if __name__ == "__main__":
    main()