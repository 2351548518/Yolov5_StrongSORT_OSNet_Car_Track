import time
from flask import Flask,render_template,request,jsonify
app = Flask(__name__)
# 设置静态文件缓存过期时间
# app.send_file_max_age_default = timedelta(seconds=1)


# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():

    return render_template('index.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=5000, debug=True)