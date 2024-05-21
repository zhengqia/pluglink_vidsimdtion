import sys,os,time,socketio
from plugins.vidsimdtion.main import start_vidsimdtion

"""
插件环境下要采用绝对路径：
from plugins.videosyn.Web_Actions import running_merge_videos

plugins：为PlugLink框架默认插件集中路径；
videosyn（示例）：这是当前插件的文件夹名；
Web_Actions（示例）：这是你调用的文件。
"""

"""
以下部分不要修改，否则可能会导致无法在大框架中实现api
仅一处可以修改，即Runconn函数下的if execute_merge_videos:代码块
在此代码块下写入您的函数即可，尽量仅用函数，最好不要写入太长的代码，养成良好的代码习惯。
print_messages函数是测试函数，不需要用到注释掉就好。
"""
# 获取插件名称
def test_plugin_name():
    current_file_path = os.path.abspath(__file__)
    current_folder_path = os.path.dirname(current_file_path)
    current_folder_name = os.path.basename(current_folder_path)
    return current_folder_name

# 事件测试
def test_connection(pluginname):
    result = f"{pluginname} （来自API:{pluginname}消息）Testing connection..."
    return result

# 这是测试函数
def print_messages():
    for i in range(5):  # 假设我们要打印5条信息
        print(f"信息{i}: 这是第 {i} 条信息")
        time.sleep(1)  # 暂停1秒

def Runconn(pluginname):
    execute_merge_videos = True
    if len(sys.argv) > 1:
        function_name = sys.argv[1]
        if function_name == 'test_connection':
            print(f'（来自API:{pluginname}消息）Executing test_connection().')
            result = test_connection(pluginname)
            print(result)
            execute_merge_videos = False

    if execute_merge_videos:
        #这里运行插件的代码（输入你的主函数即可）
        print(f'（来自API:{pluginname}消息）Executing start_vidsimdtion(True).')
        start_vidsimdtion(True)
        #print_messages()

if __name__ == "__main__":
    pluginname = test_plugin_name()
    try:
        Runconn(pluginname)
        result = (f"Success:Connection {pluginname}")
    except Exception as e:
        result = (f"Error:Connection {pluginname} - {str(e)}")

    print(result, file=sys.stderr if "Error" in result else sys.stdout)
