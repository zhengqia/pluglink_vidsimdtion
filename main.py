from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os,webbrowser,cv2,socketio,json,threading,sys
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor,as_completed
from multiprocessing import cpu_count
from flask_socketio import SocketIO, emit
from flask import Blueprint,Flask, request, jsonify,render_template,send_from_directory,redirect, url_for
import pandas as pd
import tensorflow as tf
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment,Color
from openpyxl.styles.numbers import FORMAT_PERCENTAGE_00
import openpyxl.utils as u
from openpyxl.utils import get_column_letter
from sklearn.metrics.pairwise import cosine_similarity

"""
默认插件样本
此文件名必须为main.py
在插件目录下必须要有__init__.py文件，该文件可以为空
升级文件请自行安排自己的升级服务器
"""
#注册插件蓝图-自定义名字，尽量个性化，避免与其它插件冲突
plugin_blueprint = Blueprint('vidsimdtion', __name__)

# 目录检测函数
def get_base_path(subdir=None):
    """
    获取基础路径，可选地附加一个子目录。
    :param subdir: 可选的子目录名。
    :return: 根据是否打包以及是否指定子目录返回相应的路径。
    """
    if getattr(sys, 'frozen', False):
        # 如果应用程序被打包成了单一文件
        base_path = sys._MEIPASS
    else:
        # 正常执行时使用文件的当前路径
        base_path = os.path.dirname(os.path.abspath(__file__))

    # 如果指定了子目录，将其追加到基路径
    if subdir:
        base_path = os.path.join(base_path, subdir)

    return base_path

"""<视频相似度检测>"""
# 初始化模型,需要下载resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# 指定权重文件的路径
weights_path = os.path.join(get_base_path(), 'models', 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

# 实例化模型，不加载预训练的imagenet权重
model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))

# 加载本地权重文件
model.load_weights(weights_path)

# 保留你想要的输出层，这里以模型的倒数第二层为例
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
#model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

"""
以下代码仅为方便您前端网页调试的示例，在正式安装插件的环境下必须删除

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/<path:filename>')
def serve_views(filename):
    return send_from_directory('static/', filename)

@app.route('/', methods=['GET'])
def index_page():
    return send_from_directory(app.static_folder, 'index.html')
"""

@plugin_blueprint.route('/', methods=['POST','GET'])  #蓝图环境不需要加路由器名称
#@app.route('/vidsimdtion/', methods=['POST','GET'])
def handle_vidsimdtion():
    try:
        action = request.args.get('action') if request.method == 'GET' else request.form.get('action')
        base_path = get_base_path()  # 获取基础路径
        api_path = os.path.join(base_path, 'api.json')  # 构建完整的文件路径

        print(f'当前于【vidsimdtion插件】/路由action状态码：{action}')
        if action == 'load_api':
            if os.path.exists(api_path):
                with open(api_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return jsonify(data)
            else:
                 return jsonify({'error': 'No saved directory path found.'}), 404

        if action == 'start':
            directory_path = request.form.get('dir')

            if not os.path.exists(directory_path) or not os.listdir(directory_path):
                return jsonify({'error': '该文件夹为空或不存在'}), 400

            with open(api_path, 'w', encoding='utf-8') as f:
                json.dump({'dir': directory_path}, f)

            return start_vidsimdtion()

        if action == 'save':
            try:
                directory_path = request.form.get('dir')

                # 检查 directory_path 是否是一个非空字符串
                if not isinstance(directory_path, str) or not directory_path:
                    print('error:文件夹路径不能为空')
                    return jsonify({'error': '文件夹路径不能为空'}), 400

                with open(api_path, 'w', encoding='utf-8') as f:
                    json.dump({'dir': directory_path}, f)

                print('成功保存到 api.json')
                return jsonify({'成功保存到':'api.json'})

            except Exception as e:
                print(f"错误 - {str(e)}")

        if action == 'download':
            directory = os.getcwd()  # 确保这里是文件的正确位置
            filename = 'outdata.xlsx'  # 确保文件名和位置正确
            file_path = os.path.join(directory, filename)
            if not os.path.exists(file_path):
                return jsonify({'error': '还未生成表格，请先执行检测'}), 404

            try:
                return send_from_directory(directory, filename, as_attachment=True)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    except Exception as e:
        print(f"错误 - {str(e)}")

# 主执行函数
def start_vidsimdtion(isAPI=False):
    base_path = get_base_path()  # 获取基础路径
    api_path = os.path.join(base_path, 'api.json')  # 构建完整的文件路径
    print(f'读取API文件路径：{api_path}')
    with open(api_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 提取dir键的值
    dir = data.get('dir', None)

    video_paths = find_videos_in_directory(dir)
    if not video_paths:
        print('该目录中没有找到视频文件')
        if not isAPI:
            return jsonify({'error': '该目录中没有找到视频文件'}), 400

    if not isAPI:
        socketio.emit('video_count', {'total': len(video_paths)})

    similarities = process_and_calculate_similarities(video_paths,isAPI)
    save_similarities_to_excel(similarities)  # 保存到Excel
    if not isAPI:
        return jsonify({'similarities': similarities})


# 保存为excel表格
def save_similarities_to_excel(similarities, file_path='outdata.xlsx'):
    # 解析相似度数据
    video_pairs = json.loads(similarities)

    # 获取所有视频文件名
    videos = set()
    for pair in video_pairs.keys():
        v1, v2 = pair.split(" - ")
        videos.update([v1, v2])
    videos = sorted(videos)

    # 创建相似度矩阵，指定dtype为object
    similarity_matrix = pd.DataFrame(index=videos, columns=videos, dtype=object)
    for pair, similarity in video_pairs.items():
        v1, v2 = pair.split(" - ")
        similarity_percentage = f"{similarity * 100:.2f}%"
        similarity_matrix.at[v1, v2] = similarity_percentage
        similarity_matrix.at[v2, v1] = similarity_percentage
    for video in videos:
        similarity_matrix.at[video, video] = "100.00%"

    # 将数据写入Excel文件
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        similarity_matrix.to_excel(writer, index_label='Video', sheet_name='Similarities')


def process_and_calculate_similarities(video_paths,isAPI=False):
    video_features = process_videos(video_paths, model,isAPI)
    #video_features = [extract_features(path) for path in video_paths]
    similarities = cosine_similarity(np.array(video_features))
    video_names = [os.path.basename(path) for path in video_paths]
    similarity_df = pd.DataFrame(similarities, index=video_names, columns=video_names)

    # 创建一个字典来存储非重复的相似度结果
    similarity_dict = {}
    for i in range(len(video_names)):
        for j in range(i + 1, len(video_names)):
            #similarity_dict[f"{video_names[i]} - {video_names[j]}"] = similarity_df.iloc[i, j]
            similarity_dict[f"{video_names[i]} - {video_names[j]}"] = float(similarity_df.iloc[i, j])

    # 将字典转换为JSON格式
    similarity_json = json.dumps(similarity_dict, indent=4)

    print("Similarities in JSON format:")
    print(similarity_json)

    return similarity_json

    # print("Similarities:", similarity_df)
    # return similarities


def extract_and_process_frames(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    frames = preprocess_input(frames)
    features = model.predict(frames)
    features = features.reshape(features.shape[0], -1)  # 改变形状以使每个样本只有一行特征
    return np.mean(features, axis=0)  # 使用平均特征表示整个视频

def process_videos(video_paths, model,isAPI=False):
    video_features = []
    total_videos = len(video_paths)
    for index, video_path in enumerate(video_paths):
        features = extract_and_process_frames(video_path, model)
        video_features.append(features)
        # 实时更新处理进度到前端
        progress = (index + 1) / total_videos * 100
        if not isAPI:
           socketio.emit('progress_update', {'current': index + 1, 'total': total_videos, 'progress': progress})
    return video_features

def find_videos_in_directory(directory_path):
    video_paths = []
    for file_name in os.listdir(directory_path):
        if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):  # 添加更多视频格式扩展名
            video_paths.append(os.path.join(directory_path, file_name))
    return video_paths




"""
插件独立测试专用，正式环境要注释

def open_browser():
    webbrowser.open('http://localhost:8966/') 


if __name__ == '__main__':
    #在此处调用函数，其它部分可以删除
    open_browser()
    #app.run(host='0.0.0.0', debug=True, port=8966, use_reloader=False)
    socketio.run(app, host='0.0.0.0', port=8966, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
"""