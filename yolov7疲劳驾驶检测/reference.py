import shutil
from tqdm import *
from PIL import Image
import copy
import sys
import traceback
import os

import numpy as np
import time
import cv2
from input_reader import InputReader
from tracker import Tracker
from scipy.spatial import distance as euclidean_distance
# from EAR import eye_aspect_ratio
# from MAR import mouth_aspect_ratio
#拿掉了，因为后面重新写了一边
from models.experimental import attempt_load
from utils.general import check_img_size
from tempfile import NamedTemporaryFile
from utils.torch_utils import TracedModel
from mydetect import detect
# from model_service.pytorch_model_service import PTServingBaseService
def eye_aspect_ratio(eye):
    # 计算垂直距离
    A = euclidean_distance.euclidean(eye[1], eye[5])
    B = euclidean_distance.euclidean(eye[2], eye[4])

    # 计算水平距离
    C = euclidean_distance.euclidean(eye[0], eye[3])

    # 计算眼睛纵横比
    ear = (A + B) / (2.0 * C)

    return ear
def mouth_aspect_ratio(landmarks):
    # 嘴巴外轮廓的特征点
    A = landmarks[60]  # 51: 上嘴唇低部
    B = landmarks[64]  # 57: 下嘴唇顶部
    C = landmarks[62]  # 48: 左嘴角
    D = landmarks[58]  # 54: 右嘴角

    # 嘴巴宽度
    width = euclidean_distance.euclidean(C, D)
    hight = euclidean_distance.euclidean(A,B)
    mar = hight/width
    return mar
#直接不继承了
class fatigue_driving_detection():
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path

        self.is_save = True #是否保存抽帧的图片
        self.result_list=['Normal']#检测结果的列表

        self.capture = 'test.mp4'

        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.first = True

        self.standard_pose = [180, 40, 80]
        self.look_around_frame = 0
        self.eyes_closed_frame = 0
        self.mouth_open_frame = 0
        self.use_phone_frame = 0
        # lStart, lEnd) = (42, 48)
        self.lStart = 42
        self.lEnd = 48
        # (rStart, rEnd) = (36, 42)
        self.rStart = 36
        self.rEnd = 42
        # (mStart, mEnd) = (49, 66)
        self.mStart = 49
        self.mEnd = 66
        self.EYE_AR_THRESH = 0.2
        self.MOUTH_AR_THRESH = 0.6
        self.frame_3s = self.fps * 2
        self.face_detect = 0

        self.weights = "best.pt"
        self.imgsz = 640

        self.device = 'cpu'  # 大赛后台使用CPU判分

        model = attempt_load(model_path, map_location=self.device)
        self.stride = int(model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

        self.model = TracedModel(model, self.device, self.imgsz)


        self.need_reinit = 0
        self.failures = 0

        self.tracker = Tracker(self.width, self.height, threshold=None, max_threads=4, max_faces=4,
                          discard_after=10, scan_every=3, silent=True, model_type=3,
                          model_dir=None, no_gaze=False, detection_threshold=0.6,
                          use_retinaface=0, max_feature_updates=900,
                          static_model=True, try_hard=False)

        # self.temp = NamedTemporaryFile(delete=False)  # 用来存储视频的临时文件

    def _preprocess(self, data):
        # preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        with open(self.capture, 'wb') as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                    # self.capture = self.temp.name  # Pass temp.name to VideoCapture()
                except Exception:
                    return {"message": "There was an error processing the file"}
        return 'ok'

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        print(data)
        result = {"result": {"category": 0, "duration": 6000}}
        #判断视频文件夹是否存在
        if(os.path.exists('video_images/' + self.capture)):
            pass
        else:
            os.mkdir('video_images/' + self.capture)  # 创建为视频名称的文件夹

        self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps)
        source_name = self.input_reader.name
        now = time.time()
        frame_n = 0
        while self.input_reader.is_open():
            if not self.input_reader.is_open() or self.need_reinit == 1:
                self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps, use_dshowcapture=False, dcap=None)
                if self.input_reader.name != source_name:
                    print(f"Failed to reinitialize camera and got {self.input_reader.name} instead of {source_name}.")
                    # sys.exit(1)
                self.need_reinit = 2
                time.sleep(0.02)
                continue
            if not self.input_reader.is_ready():
                time.sleep(0.02)
                continue

            ret, frame = self.input_reader.read()

            self.need_reinit = 0

            try:
                if frame is not None:
                    # 剪裁主驾驶位
                    frame1 = frame
                    frame = frame[:, 200:, :]

                    # 检测驾驶员是否接打电话 以及低头的人脸
                    bbox = detect(self.model, frame, self.stride, self.imgsz)
                    # print(bbox)     #返回的是检测的矩形框

                    for box in bbox:
                        if box[0] == 0:
                            self.face_detect = 1
                        if box[0] == 1:
                            self.use_phone_frame += 1

                    # 检测驾驶员是否张嘴、闭眼、转头
                    faces = self.tracker.predict(frame)
                    if len(faces) > 0:

                        face_num = 0
                        max_x = 0
                        for face_num_index, f in enumerate(faces):
                            if max_x <= f.bbox[3]:
                                face_num = face_num_index
                                max_x = f.bbox[3]

                        f = faces[face_num]
                        f = copy.copy(f)
                        # 检测是否转头
                        if np.abs(self.standard_pose[0] - f.euler[0]) >= 45 or np.abs(self.standard_pose[1] - f.euler[1]) >= 45 or \
                                np.abs(self.standard_pose[2] - f.euler[2]) >= 45:
                            self.look_around_frame += 1
                        else:
                            self.look_around_frame = 0

                        # 检测是否闭眼
                        leftEye = f.lms[self.lStart:self.lEnd]
                        rightEye = f.lms[self.rStart:self.rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        # average the eye aspect ratio together for both eyes
                        ear = (leftEAR + rightEAR) / 2.0

                        if ear < self.EYE_AR_THRESH:
                            self.eyes_closed_frame += 1
                        else:
                            self.eyes_closed_frame = 0
                        # print(ear, eyes_closed_frame)

                        # 检测是否张嘴
                        mar = mouth_aspect_ratio(f.lms)

                        if mar > self.MOUTH_AR_THRESH:
                            self.mouth_open_frame += 1
                    else:
                        if self.face_detect:
                            self.look_around_frame += 1
                            self.face_detect = 0

                    # 可视化
                    cv2.imwrite('1.jpg', frame)
                    img = cv2.imread('1.jpg')

                    #如果监测出来目标则画框，如果检测不出来直接传原图像
                    if(bbox):
                        img2 = cv2.rectangle(img, (int(bbox[0][1]), int(bbox[0][2])),
                                             (int(bbox[0][3]), int(bbox[0][4])), color=(255, 0, 255), thickness=2, )
                    else:
                        img2 = img
                    # img2 = cv2.rectangle(img,(int(bbox[0][1]),int(bbox[0][2])),(int(bbox[0][3]),int(bbox[0][4])),color=(255, 0, 255),thickness=2,)
                    img2_name = 'img' + str(frame_n) +'.jpg'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    try:                                        #对于没有检测到框的选择跳过
                        cv2.putText(img,str(self.result_list[frame_n]), (int(bbox[0][1]), int(bbox[0][2])+40), font, 1.5, (255, 0, 255), 3)
                    except IndexError:
                        pass
                    # print(img2_name)

                    #
                    cv2.imwrite('video_images/'+self.capture+'/'+img2_name,img2) #图片保存
                    frame_n+=1
                    #

                    # print(self.look_around_frame)
                    if self.use_phone_frame >= self.frame_3s:
                        result['result']['category'] = 'Use_phone_over3s'
                        self.result_list.append(result['result']['category'])
                        # break

                    elif self.look_around_frame >= self.frame_3s:
                        result['result']['category'] = 'Look_around_over3s'
                        self.result_list.append(result['result']['category'])
                        # print(self.result_list)
                        # break

                    elif self.mouth_open_frame >= self.frame_3s:
                        result['result']['category'] = 'Mouth_open_over3s'
                        self.result_list.append(result['result']['category'])
                        # print(self.result_list)
                        # break

                    elif self.eyes_closed_frame >= self.frame_3s:
                        result['result']['category'] = 'Eyes_closed_over3s'
                        self.result_list.append(result['result']['category'])
                        # print(self.result_list)
                        # break
                    else:
                        result['result']['category'] = 'Normal'
                        self.result_list.append(result['result']['category'])

                    self.failures = 0
                else:
                    break
            except Exception as e:
                if e.__class__ == KeyboardInterrupt:
                    print("Quitting")
                    break
                traceback.print_exc()
                self.failures += 1
                if self.failures > 30:   # 失败超过30次就默认返回
                    break
            del frame
        final_time = time.time()
        duration = int(np.round((final_time - now) * 1000))
        result['result']['duration'] = duration
        # print(result)
        # print(self.result_list)
        print('The test video was framed and contained a total of '+str(len(self.result_list)-1)+' images')
        time.sleep(2)


        # 合成视频
        # 要被合成的多张图片所在文件夹
        # 路径分隔符最好使用“/”,而不是“\”,“\”本身有转义的意思；或者“\\”也可以。
        # 因为是文件夹，所以最后还要有一个“/”
        file_dir = 'video_images/'+self.capture
        list = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                list.append(file)  # 获取目录下文件名列表
        list = sorted(list,key= lambda x: int(x[3:-4])) #按图片顺序排序
        # print(list)

        # VideoWriter是cv2库提供的视频保存方法，将合成的视频保存到该路径中
        # 'MJPG'意思是支持jpg格式图片
        # fps = 5代表视频的帧频为5，如果图片不多，帧频最好设置的小一点
        # (1280,720)是生成的视频像素1280*720，一般要与所使用的图片像素大小一致，否则生成的视频无法播放
        # 定义保存视频目录名称和压缩格式，像素为1280*720
        video = cv2.VideoWriter('video_images/'+self.capture[:-4]+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1320, 1080))

        for i in tqdm(range(1, len(list))):
            # 读取图片
            img_name = 'video_images/'+self.capture+'/' + list[i - 1]
            img = cv2.imread(img_name)
            # print(img_name)
            # resize方法是cv2库提供的更改像素大小的方法
            # 将图片转换为1280*720像素大小
            img = cv2.resize(img, (1320, 1080))
            # 写入视频
            video.write(img)
            # print(img_name+" has been written")

        # 释放资源
        video.release()
        print('The new video is composited in '+'video_images/'+self.capture[:-4]+'.avi')

        ###是否删除图片文件
        if(self.is_save):
            pass
        else:
            shutil.rmtree('video_images/' + self.capture)

        # os.startfile(path='video_images/'+self.capture[:-4]+'.avi',operation='open')

        return result

    def _postprocess(self, data):
        # os.remove(self.temp.name)
        return data
if __name__ == '__main__':
    det =fatigue_driving_detection(model_name='berberber_model',model_path='best.pt')  #读入yolo训练好的权重文件
    det.capture = 'use_phone3.mp4'    #待测试的视频文件
    det.is_save =True           #不保存视频的抽帧图片
    det._inference('The test begins.....')  #开始预测
    print(det.result_list)


