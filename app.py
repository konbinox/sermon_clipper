#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import shutil
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 限制

# 临时目录
TEMP_DIR = Path('/tmp/sermon_clipper')
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 保持最新任务的状态
tasks = {}


# ============================================================
# 路由
# ============================================================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    """提交任务"""
    data = request.json
    
    task_id = datetime.now().strftime('%Y%m%d%H%M%S')
    tasks[task_id] = {
        'status': 'processing',
        'progress': 0,
        'message': '任务已提交',
        'result': None
    }
    
    # 启动后台线程处理
    thread = threading.Thread(target=process_video, args=(task_id, data))
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'processing'})


@app.route('/status/<task_id>')
def status(task_id):
    """查询任务状态"""
    task = tasks.get(task_id, {})
    return jsonify(task)


@app.route('/download/<task_id>')
def download(task_id):
    """下载视频"""
    task = tasks.get(task_id, {})
    if task.get('status') == 'completed' and task.get('result'):
        return send_file(
            task['result'],
            as_attachment=True,
            download_name=f"sermon_{task_id}.mp4"
        )
    return jsonify({'error': '任务未完成或不存在'}), 404


# ============================================================
# 视频处理核心（轻量版，只保留必要功能）
# ============================================================

def process_video(task_id, data):
    """后台处理视频"""
    task = tasks[task_id]
    
    try:
        task['message'] = '获取视频...'
        task['progress'] = 5
        
        # 1. 下载视频
        video_url = data.get('url')
        if not video_url:
            raise ValueError('请提供视频链接')
        
        task_dir = TEMP_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用 yt-dlp 下载
        import yt_dlp
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': str(task_dir / 'video.%(ext)s'),
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_file = task_dir / f"video.{info['ext']}"
        
        task['message'] = '提取音频...'
        task['progress'] = 10
        
        # 2. 提取音频
        audio_file = task_dir / 'audio.wav'
        cmd = ['ffmpeg', '-y', '-i', str(video_file), '-ar', '16000', '-ac', '1', '-vn', str(audio_file)]
        subprocess.run(cmd, check=True, capture_output=True)
        
        task['message'] = '转录中（约3-5分钟）...'
        task['progress'] = 15
        
        # 3. 转录（使用 faster-whisper）
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_file), language="zh", beam_size=5)
        
        transcript = []
        for seg in segments:
            transcript.append({
                'start': seg.start,
                'end': seg.end,
                'text': seg.text.strip()
            })
        
        # 保存转录
        with open(task_dir / 'transcript.json', 'w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False)
        
        task['message'] = '提取精华...'
        task['progress'] = 60
        
        # 4. 简化版精华提取（基于提纲匹配）
        topics = data.get('topics', [])
        outlines = [t['title'] for t in topics]
        
        # 为每个提纲找到匹配的段落
        selected_segments = []
        total_duration = 0
        target_seconds = int(data.get('duration', 8)) * 60
        
        for outline in outlines:
            best_match = None
            best_score = 0
            for seg in transcript:
                if outline in seg['text']:
                    score = len(seg['text'])
                    if score > best_score:
                        best_score = score
                        best_match = seg
            
            if best_match:
                selected_segments.append(best_match)
                total_duration += best_match['end'] - best_match['start']
            
            if total_duration >= target_seconds:
                break
        
        # 5. 合成视频
        task['message'] = '合成视频...'
        task['progress'] = 85
        
        # 按时序排序
        selected_segments.sort(key=lambda x: x['start'])
        
        # 创建片段文件
        temp_files = []
        concat_lines = []
        for i, seg in enumerate(selected_segments):
            start = seg['start']
            dur = seg['end'] - start
            temp_file = task_dir / f'temp_{i:03d}.mp4'
            cmd = ['ffmpeg', '-y', '-ss', str(start), '-i', str(video_file),
                   '-t', str(dur), '-c:v', 'libx264', '-c:a', 'aac', '-preset', 'fast', str(temp_file)]
            subprocess.run(cmd, check=True, capture_output=True)
            temp_files.append(temp_file)
            concat_lines.append(f"file '{temp_file.name}'")
        
        # 合并
        concat_file = task_dir / 'concat.txt'
        with open(concat_file, 'w') as f:
            f.write('\n'.join(concat_lines))
        
        output_file = task_dir / 'output.mp4'
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
               '-i', str(concat_file), '-c', 'copy', str(output_file)]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 清理临时文件
        for f in temp_files:
            f.unlink()
        concat_file.unlink()
        
        task['result'] = str(output_file)
        task['status'] = 'completed'
        task['progress'] = 100
        task['message'] = '处理完成！'
        
    except Exception as e:
        task['status'] = 'failed'
        task['message'] = str(e)
        task['progress'] = 0


# ============================================================
# 启动
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
