import cv2
import numpy as np

class Stitcher:
    # 拼接函数
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # 获取输入图片
        (imageB,imageA) = images
        # 检测A、B图片的SIFT关键特征点，并计算特征描述
        (kpsA,featuresA) = self.detectAndDescribe(imageA)
        (kpsB,featuresB) = self.detectAndDescribe(imageB)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则，提取匹配结果
        # H是 3*3 视角变换矩阵
        (matches, H, status) = M

        # 将图片A进行视角变换，result是变换后的图片
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        self.cv_show('result1',result)

        # 将图片B传入result图片最左端
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        self.cv_show('result1',result)

        # 检测是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result, vis)

        # 返回匹配结果
        return result

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 调用SIFT特征点检测，得出图像特征点，以及特征描述
    def detectAndDescribe(self, image):
        # 将图像转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor = cv2.xfeatures2d.SIFT_create()

        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(gray, None)

        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的特征描述
        return (kps , features)

    # 根据两个图像的特征，使用BF暴力匹配得出符合的四个特征点，应用RANSAC算法即使算变换矩阵
    def matchKeypoints(self, kpsA, kpsB, featureA, featureB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.BFMatcher()

        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        # K=2,返回每个特征点在 B 图像中的两个最近邻特征点
        rawMatches = matcher.knnMatch(featureA, featureB, 2)

        matches = []
        for m in rawMatches:
            # 有两个特征点，且当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 存储两个点在featuresA，featuresB中的索引值
                # trainIdx：表示匹配对中的特征点在图像 B（训练图像）中的索引值。
                # queryIdx：表示匹配对中的特征点在图像 A（查询图像）中的索引值。
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 对于 matches 中的每个匹配对，从图像 A 的特征点列表 kpsA 中根据匹配对中的索引提取出对应的特征点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 使用 RANSAC 算法计算视角变换矩阵 H
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # 返回结果
            return (matches, H, status)

        # 如果匹配对小于4时，返回None
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, mathces, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]

        # 初始化可视化图片，将A、B图左右连接到一起
        vis = np.zeros((max(hA,hB), wA + wB, 3), dtype="uint8") # 设定图像h、w和通道数
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA: ] = imageB

        # 联合遍历，画出匹配对
        # trainIdx：表示匹配对中的特征点在图像 B（训练图像）中的索引值。
        # queryIdx：表示匹配对中的特征点在图像 A（查询图像）中的索引值。
        # zip(mathces, status)：这个函数将 mathces和status中的元素一一对应地组合成一个新的可迭代对象
        # 每次迭代返回一个匹配对以及它的状态。
        for ((trainIdx, queryIdx), s) in zip(mathces, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                # [0]在图像 A 中，第 queryIdx 个特征点的 x 坐标值
                # [1]在图像 A 中，第 queryIdx 个特征点的 y 坐标值
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                # 画线,ptA 和 ptB 分别是连线的起点和终点坐标
                cv2.line(vis, ptA, ptB, (0,255,0), 1)

        # 返回可视化结果
        return vis

#################

# 读取拼接图片
imageA = cv2.imread("left_01.png")
imageB = cv2.imread("right_01.png")

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA,imageB],showMatches=True)

# 显示所有图片
cv2.imshow("Image A",imageA)
cv2.imshow("Image B",imageB)
cv2.imshow("Keypoint Matches",vis)
cv2.imshow("Keypoint Matches",vis)
cv2.imshow("Result", result)

cv2.waitKey(0) # 阻塞程序的执行，直到用户按下任意键，然后返回所按下的按键的 ASCII 码值
cv2.destroyAllWindows()