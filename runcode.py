import cv2 as cv
import numpy as np
import mediapipe as mp
 
# 视频设备号
DEVICE_NUM = 0

# 检测摄像头是否打开
def camera_status(cap):
    return cap.isOpened()

# 检测左手是否存在
def left_exist_fun(result,i):
    left_exit = result.multi_handedness[i].classification[0].label == "Left"
    return left_exit

# 手指检测
# point1-手掌0点位置，point2-手指尖点位置，point3手指根部点位置
def finger_stretch_detect(point1, point2, point3,landmark7):
    result = 0
    add_result = 0
    #计算向量的L2范数
    dist1 = np.linalg.norm((point2 - point1), ord=2)
    dist2 = np.linalg.norm((point3 - point1), ord=2)
    dist3 = np.linalg.norm((landmark7 - point1), ord=2)
    if dist2 > dist1:
        result = 1
    if dist3 > dist2:
        add_result = 1
    
    return result,add_result

# 检测左手的含义
def detect_left_meaning(handLms, frame_width, frame_height):
    
    landmark = np.empty((21, 2)) # 声明变量
    figure = np.zeros(5)
    add_figure = np.zeros(5)

    for j, lm in enumerate(handLms.landmark):
        xPos = int(lm.x * frame_width)
        yPos = int(lm.y * frame_height)
        landmark_ = [xPos, yPos]
        landmark[j,:] = landmark_

    # 通过判断手指尖与手指根部到0位置点的距离判断手指是否伸开(拇指检测到17点的距离)
    for k in range (5):
        if k == 0:
            figure_,add_figure_ = finger_stretch_detect(landmark[17],landmark[4*k+2],landmark[4*k+4],landmark[7])
        else:    
            figure_,add_figure_ = finger_stretch_detect(landmark[0],landmark[4*k+1],landmark[4*k+4],landmark[7])
    
        figure[k] = figure_
        add_figure[k] = add_figure_

    # 判断手势
    if (figure[0] == 0) and (figure[1] == 0) and (figure[2] == 0) and (figure[3] == 0) and (figure[4] == 0):
        gesture = "symbol"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 0) and (figure[3] == 0) and (figure[4] == 0) and (add_figure[1] == 0):
        gesture = "one"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 0) and (figure[4] == 0):
        gesture = "two"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 1) and (figure[4] == 0):
        gesture = "three"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 1) and (figure[4] == 1):
        gesture = "four"
    elif (figure[0] == 1) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 1) and (figure[4] == 1):
        gesture = "five"
    elif (figure[0] == 1) and (figure[1] == 0)and (figure[2] == 0) and (figure[3] == 0) and (figure[4] == 1):
        gesture = "six"
    elif (figure[0] == 1) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 0) and (figure[4] == 0) and (add_figure[0] == 0):
        gesture = "seven"
    elif (figure[0] == 1) and (figure[1] == 1)and (figure[2] == 0) and (figure[3] == 0) and (figure[4] == 0):
        gesture = "eight"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 0) and (figure[3] == 0) and (figure[4] == 0) and (add_figure[1] == 1):
        gesture = "nine"
    else:
        gesture = "not in detect range..."
    
    return gesture

# 检测右手是否存在
def right_exist_fun(result,i):
    right_exit = result.multi_handedness[i].classification[0].label == "Right"
    return right_exit

# 检测右手表示的值
def detect_right_value(handLms, frame_width, frame_height):
    
    landmark = np.empty((21, 2)) # 声明变量
    figure = np.zeros(5)
    add_figure = np.zeros(5)

    for j, lm in enumerate(handLms.landmark):
        xPos = int(lm.x * frame_width)
        yPos = int(lm.y * frame_height)
        landmark_ = [xPos, yPos]
        landmark[j,:] = landmark_

    # 通过判断手指尖与手指根部到0位置点的距离判断手指是否伸开(拇指检测到17点的距离)
    for k in range (5):
        if k == 0:
            figure_,add_figure_ = finger_stretch_detect(landmark[17],landmark[4*k+2],landmark[4*k+4],landmark[7])
        else:    
            figure_,add_figure_ = finger_stretch_detect(landmark[0],landmark[4*k+1],landmark[4*k+4],landmark[7])
    
        figure[k] = figure_
        add_figure[k] = add_figure_

    # 判断右手的值
    if (figure[0] == 0) and (figure[1] == 0) and (figure[2] == 0) and (figure[3] == 0) and (figure[4] == 0):
        gesture = "zero"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 0) and (figure[3] == 0) and (figure[4] == 0) and (add_figure[1] == 0):
        gesture = "one"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 0) and (figure[4] == 0):
        gesture = "two"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 1) and (figure[4] == 0):
        gesture = "three"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 1) and (figure[4] == 1):
        gesture = "four"
    elif (figure[0] == 1) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 1) and (figure[4] == 1):
        gesture = "five"
    elif (figure[0] == 1) and (figure[1] == 0)and (figure[2] == 0) and (figure[3] == 0) and (figure[4] == 1):
        gesture = "six"
    elif (figure[0] == 1) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 0) and (figure[4] == 0) and (add_figure[0] == 0):
        gesture = "seven"
    elif (figure[0] == 1) and (figure[1] == 1)and (figure[2] == 0) and (figure[3] == 0) and (figure[4] == 0):
        gesture = "eight"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 0) and (figure[3] == 0) and (figure[4] == 0) and (add_figure[1] == 1):
        gesture = "nine"
    else:
        gesture = "not in detect range..."
    
    return gesture

# 检测右手表示的符号
def detect_right_symbol(handLms, frame_width, frame_height):
        
    landmark = np.empty((21, 2)) # 声明变量
    figure = np.zeros(5)

    for j, lm in enumerate(handLms.landmark):
        xPos = int(lm.x * frame_width)
        yPos = int(lm.y * frame_height)
        landmark_ = [xPos, yPos]
        landmark[j,:] = landmark_

    # 通过判断手指尖与手指根部到0位置点的距离判断手指是否伸开(拇指检测到17点的距离)
    for k in range (5):
        if k == 0:
            figure_,_ = finger_stretch_detect(landmark[17],landmark[4*k+2],landmark[4*k+4],landmark[7])
        else:    
            figure_,_ = finger_stretch_detect(landmark[0],landmark[4*k+1],landmark[4*k+4],landmark[7])
    
        figure[k] = figure_

    if (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 0) and (figure[3] == 0) and (figure[4] == 0):
        gesture = "plus"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 0) and (figure[4] == 0):
        gesture = "minus"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 1) and (figure[4] == 0):
        gesture = "multiply"
    elif (figure[0] == 0) and (figure[1] == 1)and (figure[2] == 1) and (figure[3] == 1) and (figure[4] == 1):
        gesture = "divide"
    else:
        gesture = None
    
    return gesture

# 将检测到的数的string转为number
def string2num(string):
    if string == "zero":
        string = 0
    elif string == "one":
        string = 1
    elif string == "two":
        string = 2
    elif string == "three":
        string = 3
    elif string == "four":
        string = 4
    elif string == "five":
        string = 5
    elif string == "six":
        string = 6
    elif string == "seven":
        string = 7
    elif string == "eight":
        string = 8
    elif string == "nine":
        string = 9
    else:
        string = 0
    
    return string

# 检测到的运算数被保存在数组中，此函数将数组转为对应的运算数值
def ndarray2value(num_ndarray):
    num_value = 0
    for i in range(len(num_ndarray)):
        num_value = num_value + num_ndarray[i]*10**i
    return num_value

# 运算函数
def compute_out(num1_value,num2_value,sig):
    if sig == "plus":
        num3_value = num1_value + num2_value
    elif sig == "minus":
        num3_value = num1_value - num2_value
    elif sig == "multiply":
        num3_value = num1_value * num2_value
    elif sig == "divide":
        if num2_value == 0:
            sig = None
            print("除数不能为0")
            num3_value = None
        else:
            num3_value = num1_value / num2_value
    else:
        print("请重新表示运算表达式！")
    
    return num3_value

# 检测函数
def detect():
    # 接入USB摄像头时，注意修改cap设备的编号
    cap = cv.VideoCapture(DEVICE_NUM) 

    # 检测摄像头是否有问题
    if not camera_status(cap): # 检测摄像头是否有问题
        print("Can not open camera.")
        exit()

    # 加载手部检测函数
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()

    # 加载绘制函数，并设置手部关键点和连接线的形状、颜色
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=int(5))
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(10))

    # 定义运算变量
    num1 = np.array(None) # 运算数1
    num1_string = None # 运算数1的字符
    num2 = np.array(None) # 运算数2
    num2_string = None # 运算数2的字符
    sig = None # 检测符号

    # 定义检测变量
    left_exit = 0 # 左手是否存在，0表示不存在
    right_exit = 0 # 右手是否存在，0表示不存在
    left_value = None # 左手的值
    num_frame = 0 # 运算帧数

    # 开始读取摄像头检测到的视频帧
    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1) # 翻转后为正确的左右手检测
        if not ret: # 检测是否能成功读取帧
            print("Can not receive frame (stream end?). Exiting...")
            break

        #mediaPipe的图像要求是RGB，所以此处需要转换图像的格式
        frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands.process(frame_RGB)

        #读取视频图像的高和宽，用于后续获取手部关键点的坐标
        frame_height = frame.shape[0]
        frame_width  = frame.shape[1]

        # 判断是否当前帧出现了手
        if result.multi_hand_landmarks:

            # 对检测到的多个手进行分别处理 
            for i, handLms in enumerate(result.multi_hand_landmarks):
                
                #为每个手绘制关键点和连接线
                mpDraw.draw_landmarks(frame, handLms,mpHands.HAND_CONNECTIONS,landmark_drawing_spec=handLmsStyle,connection_drawing_spec=handConStyle)
            
                # 检测当前帧是否为右手。此处先检测右手，是因为左手检测到之后，下一帧开始检测右手，
                # 因此把检测左手放到最后，从而在进入下一次循环时开始对右手信息进行处理
                right_exit = right_exist_fun(result,i)

                if right_exit: # 当右手存在时
                    if left_exit: # 是否已检测到左手
                        
                        #print(left_value + " left value")
                        num_frame += 1 # 检测帧数+1
                        if left_value == "symbol": # 当左手表示符号出现时
                            if (num1.any() != None) and (num2.any() == None): # 当运算数1已经获取，且运算数2尚未获取时，才可以获得运算符号
                                    sig = detect_right_symbol(handLms, frame_width, frame_height)
                                    print(sig)
                            elif (num1.any() != None) and (num2.any() != None):
                                num1_value = ndarray2value(num1)
                                num2_value = ndarray2value(num2)
                                print("num1 = %d" %num1_value)
                                print("num2 = %d" %num2_value)
                                num3_value = compute_out(num1_value,num2_value,sig)
                                print(num3_value)
                                formula = str(num1_value) + str(sig) + str(num2_value) + "=" + str(num3_value)
                                cv.putText(frame, formula, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        elif left_value == "not in detect range...": # 当左手手势不规范时，不采取操作
                            break

                        else: # 当左手手势表示接下来要出现数字时
                            if sig == None: # 当运算符号还没有获取时，表示现在需要获取运算数1
                                
                                if num1.all() == None:
                                    num1 = np.array([0,0,0,0,0,0,0,0,0])

                                if num_frame % 50 == 0: # 每50帧检测一次
                                    num1_string = detect_right_value(handLms, frame_width, frame_height)
                                    num1_pos = string2num(left_value)
                                    num1[num1_pos-1] = string2num(num1_string)
                                    print(num1_string + " num1")
                            else:
                                if num2.all() == None:
                                    num2 = np.array([0,0,0,0,0,0,0,0,0])

                                if num_frame % 50 == 0: # 每50帧检测一次
                                    num2_string = detect_right_value(handLms, frame_width, frame_height)
                                    num2_pos = string2num(left_value)
                                    num2[num2_pos-1] = string2num(num2_string)
                                    print(num2_string + " num2")
                    #cv.putText(frame, "right_value", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 检测左手是否存在，若不存在，则跳出if
                left_exit = left_exist_fun(result, i)

                if left_exit: # 当左手存在时

                    left_value = detect_left_meaning(handLms, frame_width, frame_height) # 获得左手的含义
                    #cv.putText(frame, "left_value", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        else:
            # 定义运算变量
            num1 = np.array(None) # 运算数1
            num1_string = None # 运算数1的字符
            num2 = np.array(None) # 运算数2
            num2_string = None # 运算数2的字符
            sig = None # 检测符号
            num1_value = 0
            num2_value = 0
            num3_value = 0

            # 定义检测变量
            left_exit = 0 # 左手是否存在，0表示不存在
            right_exit = 0 # 右手是否存在，0表示不存在
            left_value = None # 左手的值
            num_frame = 0 # 运算帧数
            
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


# 主函数
if __name__ == '__main__':
    detect()