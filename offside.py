
import cv2
import numpy as np
import processing
from collections import deque
import math

frame = None
originalFrame = None
roi_hist_A, roi_hist_B = None, None
roi = None
nameitr=0
team = None
innnn=1
roi1 = None

teamA = np.array([])
teamB = np.array([])
op = None
newTeamB = np.array([])
newTeamA = np.array([])
points = []
M = None

minimumDistance = 0
previousTeam = None
previousPasser = -1

limits = None
centerOfBall = None

kernel = np.array([[0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0]], dtype=np.uint8)

previousGrad = 0
passes = 0

vel, previousVel = 0, 0

ballPoints = deque()

def select_points(event, x, y, flag, param):
    global points, frame, originalFrame

    if event == cv2.EVENT_LBUTTONUP:
        if len(points) < 8:
            points.append([x, y])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        else:
            print('You have already selected 8 points')


def get_boundaryPoints():
    global frame, points
    end_points = []
    cv2.namedWindow('input field')
    cv2.setMouseCallback('input field', select_points)
    while True:
        cv2.imshow('input field', frame)
        key = cv2.waitKey(1) & 0xFF
        if len(points) >= 8:
            points = np.array(points, dtype=np.float32)
            #print(points)
            points[:, 1] *= (-1)
            for i in range(0, 5, 2):
                m1 = (points[i + 1][1] - points[i][1]) / (points[i + 1][0] - points[i][0])
                m2 = (points[i + 3][1] - points[i + 2][1]) / (points[i + 3][0] - points[i + 2][0])
                A = np.array([[m1, -1], [m2, -1]])
                A_inv = np.linalg.inv(A)
                B = np.array([points[i][1] - m1 * points[i][0], points[i + 2][1] - m2 * points[i + 2][0]])
                B *= (-1)
                p = np.dot(A_inv, B)
                end_points.append(np.int16(p))
            m1 = (points[7][1] - points[6][1]) / (points[7][0] - points[6][0])
            m2 = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
            A = np.array([[m1, -1], [m2, -1]])
            A_inv = np.linalg.inv(A)
            B = np.array([points[6][1] - m1 * points[6][0], points[0][1] - m2 * points[0][0]])
            B *= (-1)
            p = np.dot(A_inv, B)
            end_points.append(np.int16(p))
            end_points = np.array(end_points)
            end_points[:, 1] *= (-1)
            break
        elif key == ord("q"):
            break
    cv2.destroyWindow('input field')
    return end_points


def get_coordinates():

    global M, teamA, teamB, op, newTeamB, newTeamA, ball_new, centerOfBall
    newTeamB = np.array([])
    newTeamA = np.array([])
    op = orig_op.copy()
    if centerOfBall is not None:
        new = np.dot(M, [centerOfBall[0], centerOfBall[1], 1])
        ball_new = [new[0] / new[2], new[1] / new[2]]
        op = cv2.circle(op, (int(ball_new[0]), int(ball_new[1])), 3, (255, 0, 0), -1)

    if len(teamB) > 0:
        for i in range(len(teamB)):
            new_pt = np.dot(M, [teamB[i][0], teamB[i][1], 1])
            newTeamB = np.append(newTeamB, [new_pt[0] / new_pt[2], new_pt[1] / new_pt[2]])

        newTeamB = np.int16(newTeamB).reshape(-1, 2)
        for i in range(len(teamB)):
            op = cv2.circle(op, (newTeamB[i][0], newTeamB[i][1]), 5, (0, 255, 0), -1)

    if len(teamA) > 0:
        for i in range(len(teamA)):
            new_pt = np.dot(M, [teamA[i][0], teamA[i][1], 1])
            newTeamA = np.append(newTeamA, [new_pt[0] / new_pt[2], new_pt[1] / new_pt[2]])

        newTeamA = np.int16(newTeamA).reshape(-1, 2)
        for i in range(len(teamA)):
            op = cv2.circle(op, (newTeamA[i][0], newTeamA[i][1]), 5, (0, 0, 255), -1)


def detect_players():
    global frame, roi_hist_A, roi_hist_B, teamA, teamB, innnn
    teamA = []
    teamB = []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cnt_thresh = 180
    if roi_hist_A is not None:
        backProjectionA = cv2.calcBackProject([hsv], [0, 1], roi_hist_A, [0, 180, 0, 256], 1)
        maskA = processing.applyMorphTransforms2(backProjectionA)
        #cv2.imshow('mask a', maskA)
        
        contours = cv2.findContours(maskA.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours) > 0:
            c = sorted(contours, key=cv2.contourArea, reverse=True)
            for i in range(len(c)):
                if cv2.contourArea(c[i]) < cnt_thresh:
                    break

                x, y, w, h = cv2.boundingRect(c[i])
                h += 5
                y -= 5
                if h < 0.8 * w:
                    continue
                elif h / float(w) > 3:
                    continue

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                M = cv2.moments(c[i])
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                foot = (center[0], int(center[1] + h * 1.5))
                teamA.append(foot)
                cv2.circle(frame, foot, 5, (0, 0, 255), -1)
    if roi_hist_B is not None:
        backProjectionB = cv2.calcBackProject([hsv], [0, 1], roi_hist_B, [0, 180, 0, 256], 1)
        maskB = processing.applyMorphTransforms2(backProjectionB)
        #tmp='mask'+str(innnn)+".jpg"
        #cv2.imwrite(tmp, maskB)
        #innnn+=1
        
        #cv2.imshow('mask b', maskB)

        contours = cv2.findContours(maskB.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours) > 0:
            c = sorted(contours, key=cv2.contourArea, reverse=True)
            for i in range(len(c)):
                if cv2.contourArea(c[i]) < cnt_thresh:
                    break
                x, y, w, h = cv2.boundingRect(c[i])
                h += 5
                y -= 5
                if h < 0.9 * w:
                    continue
                elif h / float(w) > 3:
                    continue

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                M = cv2.moments(c[i])
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                foot = (center[0], int(center[1] + h * 1.2))

                teamB.append(foot)
                cv2.circle(frame, foot, 5, (0, 0, 255), -1)


def track_ball():

    global grad, previousGrad, passes, centerOfBall, ballPoints, frame, vel, previousVel, previousPasser, previousTeam, minimumDistance
    ballPoints.appendleft(centerOfBall)

    if len(ballPoints) > 2:
        if len(ballPoints) > 20:

            for i in xrange(1, 20):
                cv2.line(frame, ballPoints[i - 1], ballPoints[i], (0, 0, 255), 2, cv2.LINE_AA)
        else:
            for i in xrange(1, len(ballPoints)):
                cv2.line(frame, ballPoints[i - 1], ballPoints[i], (0, 0, 255), 2, cv2.LINE_AA)


    l = len(ballPoints)
    if l >= 10:
        grad = np.arctan2((ballPoints[9][1] - ballPoints[0][1]), (ballPoints[9][0] - ballPoints[0][0]))
        grad = grad * (180.0 / np.pi)
        grad %= 360

        vel = math.sqrt((ballPoints[9][1] - ballPoints[0][1]) ** 2 + (ballPoints[9][0] - ballPoints[0][0]) ** 2) / 10
        if (math.fabs(grad - previousGrad) >= 20):
            if len(teamA) != 0 and len(teamB) != 0:
                get_coordinates()
                detectPasser()

                # print(passerIndex)

                if ((previousTeam != team) or (passerIndex != previousPasser)) and minimumDistance < 10000:
                    # print(minimumDistance)
                    # print(str(team) + str(passerIndex))
                    if (team == 'A'):

                        detectOffside()

                    passes += 1
                    #print('Ball Passed ' + str(passes))
                prevPasser = passerIndex
                previousTeam = team


        previousGrad = grad
        previousVel = vel

def detectPasser():
    global ball_new, teamA, teamB, passerIndex, team, minimumDistance
    teamA_min_ind = closest_node(ball_new, newTeamA)
    teamB_min_ind = closest_node(ball_new, newTeamB)
    # print(np.asarray(teamA[teamA_min_ind]))
    # print(np.asarray(centerOfBall))
    teamA_min = np.sum(([np.asarray(newTeamA[teamA_min_ind])] - np.asarray(ball_new)) ** 2, axis=1)
    teamB_min = np.sum(([np.asarray(newTeamB[teamB_min_ind])] - np.asarray(ball_new)) ** 2, axis=1)
    minimumDistance = min(teamB_min, teamA_min)
    if (teamA_min < teamB_min):
        # print("Ball passed by TeamA player")

        passerIndex = teamA_min_ind
        # print(passerIndex)
        team = 'A'
    else:
        # print("Ball passed by TeamB player")
        passerIndex = teamB_min_ind
        team = 'B'


def drawOffsideLine():
    global M, newTeamB, op, frame
    if len(newTeamB) > 0:
        M_inv = np.linalg.inv(M)
        last_def = np.argmin(newTeamB[:,0])
        p1 = np.dot(M_inv, [newTeamB[last_def][0], 0, 1])
        p2 = np.dot(M_inv, [newTeamB[last_def][0], op.shape[0] - 1, 1])

        points = [(int(p1[0] / p1[2]), int(p1[1] / p1[2])), (int(p2[0] / p2[2]), int(p2[1] / p2[2]))]
        frame = cv2.line(frame, points[0], points[1], (255, 0, 0), 2)


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    node = np.array([node[0], node[1]])
    # print(nodes)
    # print(node)
    dist_2 = np.sum((nodes - node) ** 2, axis=1)

    return np.argmin(dist_2)



def detectOffside():
    if roi1 is None:
        return None
    global newTeamA, newTeamB, passerIndex
    if len(newTeamB) > 0:
        if len(newTeamA) > 0:
            # newTeamA.sort()
            newTeamB.sort()
            # print(newTeamA)
            if (newTeamB[0][0] < newTeamA[passerIndex][0]):
                # if (teamB[0][0] > teamA[passerIndex][0]):
                # print(passerIndex)
                # Assuming no goalie
                print('Offside')
            else:
                print('Not Offside')
        else:
            print('Not Offside')
    else:
        print('Not Offside')

if __name__ == '__main__':
        
    camera = cv2.VideoCapture("a3.mp4")

    orig_op = cv2.imread('soccer_half_field.jpeg')
    op = orig_op.copy()
    foregroundbackground = cv2.createBackgroundSubtractorMOG2(history=20, detectShadows=False)
    flag = False

    while True:
        (grabbed, frame) = camera.read()

        if not grabbed:
            break

        frame = processing.resize(frame, width=400)

        originalFrame = frame.copy()

        frame2 = processing.removeBG(originalFrame.copy(), foregroundbackground)

        detect_players()

        if roi is not None:
            centerOfBall, cnt = processing.detectBallThresh(frame2, limits)
            if cnt is not None:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)
                cv2.circle(frame, centerOfBall, 2, (0, 0, 255), -1)
                track_ball()

        if M is not None:
            src = np.int32(src)

            for i in range(4):
                frame = cv2.circle(frame.copy(), (src[i][0], src[i][1]), 3, (255, 0, 255), -1)

            #cv2.polylines(frame, np.int32([src]), True, (255, 0, 0), 2, cv2.LINE_AA)

            get_coordinates()

            drawOffsideLine()

        cv2.imshow('camera view', frame)
        name='abc'+str(nameitr)+'.jpg'
        cv2.imwrite(name,frame)
        nameitr=nameitr+1
        #cv2.imshow('top view', op)

        if flag:
            t = 1
        else:
            t = 100

        key = cv2.waitKey(t) & 0xFF

        if key == ord("q"):
            break
        elif key == ord('i') and (roi_hist_A is None or roi_hist_B is None):
            flag = True
            roi_hist_A, roi_hist_B = processing.getHistogram(frame)

            roi = processing.getROIvid(originalFrame, 'input ball')
            if roi is not None:
                limits = processing.getLimits(roi)
                processing.check()

            src = get_boundaryPoints()
            src = np.float32(src)
            dst = np.float32([[0, 0], [0, op.shape[0]], [op.shape[1], op.shape[0]], [op.shape[1], 0]])
            M = cv2.getPerspectiveTransform(src, dst)

    camera.release()
    cv2.destroyAllWindows()
