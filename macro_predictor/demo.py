import cv2
import numpy as np
import tensorflow.compat.v1 as tf

from model import *

tf.disable_v2_behavior()


# 加载opencv自带的人脸识别器
CASC_PATH = '../env/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
# 人脸七种微表情
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


def format_image(image):
    # image如果为彩色图：image.shape[0][1][2](水平、垂直像素、通道数)
    if len(image.shape) > 2 and image.shape[2] == 3:
        # 将图片变为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
        # 调整scaleFactor参数的大小，可以增加识别的灵敏度，推荐1.1
        faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    # 如果图片中没有检测到人脸，则返回None
    if not len(faces) > 0:
        return None, None
    # max_are_face包含了人脸的坐标，大小
    max_are_face = faces[0]
    # 在所有人脸中选一张最大的脸
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face

    # 这两步可有可无
    face_coor = max_are_face
    image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    # 调整图片大小，变为48*48
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("problem during resize")
        return None, None

    return image, face_coor


def demo(modelPath, showBox=True):
    # 调用模型分析人脸微表情
    #    tf.reset_default_graph()
    face_x = tf.placeholder(tf.float32, [None, 2304])
    y_conv = deepnn(face_x)
    probs = tf.nn.softmax(y_conv)

    # 加载模型
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(modelPath)
    sess = tf.Session()
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restore models sucsses!!\nNOTE: Press 'a' on keyboard to capture face.")

    # feelings_facesy用来存储emojis表情
    feelings_faces = []
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', 1))

    # 获取笔记本的摄像头，
    video_captor = cv2.VideoCapture(0)

    print("Restore models sucsses!!\nNOTE: Press 'a' on keyboard to capture face.")
    emoji_face = []
    result = None
    while True:
        # 获取摄像头的每帧图片，若获得，则ret的值为True,frame就是每一帧的图像，是个三维矩阵
        ret, frame = video_captor.read()

        detected_face, face_coor = format_image(frame)
        if showBox:
            if face_coor is not None:
                # 获取人脸的坐标,并用矩形框出
                [x, y, w, h] = face_coor
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # 每隔10ms刷新一次，并且等当键盘输入a的时候，截取图像，因为是64位系统所以必须要0xFF == ord('a')
        if cv2.waitKey(1) & 0xFF == ord('a'):
            if detected_face is not None:
                cv2.imwrite('a.jpg', detected_face)
                print(detected_face)
                print("获取成功")
                # 将图片变为tensorflow可以接受的格式
                tensor = image_to_tensor(detected_face)
                print(tensor)
                result = sess.run(probs, feed_dict={face_x: tensor})
                print(result)

        if result is not None:
            for index, emotion in enumerate(EMOTIONS):
                # 将七种微表情的文字添加到图片中
                cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                # 将七种微表情的概率用矩形表现出来
                cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                              (255, 0, 0), -1)
                # 获取人脸微表情相应的emojis表情
                emoji_face = feelings_faces[np.argmax(result[0])]

            # 将emojis表情添加到图片中的指定位置 方法1：
            frame[200:320, 10:130, :] = emoji_face[0:120, 0:120, :]
            cv2.imwrite('b.jpg', frame)
            # 将emojis表情添加到图片中的指定位置 方法2：
            # for c in range(0, 1):
            #     frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)

        cv2.imshow('face', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    video_captor.release()
    cv2.destroyAllWindows()


demo("./models", True)
