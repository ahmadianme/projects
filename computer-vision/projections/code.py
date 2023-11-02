import numpy as np
import pygame
from math import *
from pprint import pprint



print()
print('Welcome to projection test by Mehran Ahmadian.')
print()

print('1: Parrallel - Orthographic - Front')
print('2: Parrallel - Orthographic - Top')
print('3: Parrallel - Orthographic - Side')
print('4: Parrallel - Axonometric - Isometric')
print('5: Parrallel - Oblique - Cabinet')
print('6: Parrallel - Oblique - Cavalier')
print('7: prespective - One Point')
print('8: prespective - Two Points')
print('9: prespective - Three Points')
print()

projectionType = input('Please Select projection type by entering 1 to 9: ')







projectionMatrices = {
    'parrallel': {
        'orthographic': {
            'top': {
                'm': [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                'xAngle': 1.57075,
                'yAngle': 0,
                'zAngle': 0,
            },
            'front': {
                'm': [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                'xAngle': 0,
                'yAngle': 0,
                'zAngle': 0,
            },
            'side': {
                'm': [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                'xAngle': 0,
                'yAngle': -1.57075,
                'zAngle': 0,
            },
            'axonometric': {
                'isometric': {
                    'm': [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                    ],
                    'xAngle': 0.7875,
                    'yAngle': -0.6101,
                    'zAngle': -0.5246,
                },
            },
        },
        'oblique': {
            'cabinet': {
                'm': [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                'xAngle': 0.19,
                'yAngle': -0.30,
                'zAngle': -0.02,
            },
            'cavalier': {
                'm': [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                'xAngle': 0.2,
                'yAngle': -0.69,
                'zAngle': -0.12,
            },
        }
    },
    'prespective': {
        'onePoint': {
            'm': [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1/4, 1],
            ],
            'xAngle': 0,
            'yAngle': -0.35,
            'zAngle': 0,
        },
        'twoPoint': {
            'm': [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [1/4, 0, 1/4, 1],
            ],
            'xAngle': 0,
            'yAngle': -0.29,
            'zAngle': 0,
        },
        'threePoint': {
            'm': [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [1/4, 1/4, 1/4, 1],
            ],
            'xAngle': 0.14,
            'yAngle': -0.32,
            'zAngle': -0.04,
        },
    }
}





WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RED = (255, 0, 0)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)
LIME = (0, 255, 0)
OLIVE = (128, 128, 0)
CYAN = (0, 255, 255)
GRREN = (0, 128, 0)
GRAY = (100, 100, 100)
ORANGE = (255, 165, 0)


pointColors = [
    RED,
    BLUE,
    MAGENTA,
    LIME,
    # OLIVE,
    CYAN,
    GRREN,
    BLACK,
    ORANGE,
]







geometericPoints = [n for n in range(8)]
geometericPoints[0] = [[-1], [-1],   [1],    [-1]]
geometericPoints[1] = [[1],  [-1],   [1],    [-1]]
geometericPoints[2] = [[1],  [1],    [1],    [-1]]
geometericPoints[3] = [[-1], [1],    [1],    [-1]]
geometericPoints[4] = [[-1], [-1],   [-1],   [-1]]
geometericPoints[5] = [[1],  [-1],   [-1],   [-1]]
geometericPoints[6] = [[1],  [1],    [-1],   [-1]]
geometericPoints[7] = [[-1], [1],    [-1],   [-1]]











if projectionType == '1':
    projection = projectionMatrices['parrallel']['orthographic']['front']
elif projectionType == '2':
    projection = projectionMatrices['parrallel']['orthographic']['top']
elif projectionType == '3':
    projection = projectionMatrices['parrallel']['orthographic']['side']
elif projectionType == '4':
    projection = projectionMatrices['parrallel']['orthographic']['axonometric']['isometric']
elif projectionType == '5':
    projection = projectionMatrices['parrallel']['oblique']['cabinet']
elif projectionType == '6':
    projection = projectionMatrices['parrallel']['oblique']['cavalier']
elif projectionType == '7':
    projection = projectionMatrices['prespective']['onePoint']
elif projectionType == '8':
    projection = projectionMatrices['prespective']['twoPoint']
elif projectionType == '9':
    projection = projectionMatrices['prespective']['threePoint']
else:
    print('Invalid Projection Type. Please Enter 1 to 9.')
    exit()


projection_matrix = np.matrix(projection['m'])
angle_x = projection['xAngle']
angle_y = projection['yAngle']
angle_z = projection['zAngle']




scale = 100
WINDOW_SIZE = 800
ROTATE_SPEED = 0.01
window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()



def dotProduct(a, b):
    return np.dot(np.matrix(a), np.matrix(b)).tolist()




def drawLine(i, j, points):
    pygame.draw.line(window, GRAY, (points[i][0], points[i][1]) , (points[j][0], points[j][1]), width=2)




while True:
    clock.tick(60)
    window.fill((255,255,255))
    rotation_x = [[1, 0, 0, 0],
                  [0, cos(angle_x), -sin(angle_x), 0],
                  [0, sin(angle_x), cos(angle_x), 0],
                  [0, 0, 0, 1],
                  ]

    rotation_y = [[cos(angle_y), 0, sin(angle_y), 0],
                  [0, 1, 0, 0],
                  [-sin(angle_y), 0, cos(angle_y), 0],
                  [0, 0, 0, 1],
                  ]

    rotation_z = [[cos(angle_z), -sin(angle_z), 0, 0],
                  [sin(angle_z), cos(angle_z), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

    points = [0 for _ in range(len(geometericPoints))]
    i = 0
    for point in geometericPoints:
        rotate_x = dotProduct(rotation_x, point)
        rotate_y = dotProduct(rotation_y, rotate_x)
        rotate_z = dotProduct(rotation_z, rotate_y)
        point_2d = dotProduct(projection_matrix, rotate_z)


        if point_2d[3][0] != 1:
            point_2d[1][0] /= point_2d[3][0]
            point_2d[2][0] /= point_2d[3][0]
            point_2d[3][0] /= point_2d[3][0]


        x = (point_2d[0][0] * scale) + WINDOW_SIZE / 2
        y = (point_2d[1][0] * scale) + WINDOW_SIZE / 2

        points[i] = (x,y)
        i += 1
        pygame.draw.circle(window, pointColors[i-1], (x, y), 8)



    pygame.draw.line(window, ORANGE, (points[4][0], points[4][1]), (points[5][0], points[5][1]), width=5)
    pygame.draw.line(window, BLUE, (points[4][0], points[4][1]), (points[7][0], points[7][1]), width=5)
    pygame.draw.line(window, RED, (points[0][0], points[0][1]), (points[4][0], points[4][1]), width=5)

    drawLine(0, 1, points)
    drawLine(0, 3, points)
    # drawLine(0, 4, points)
    drawLine(1, 2, points)
    drawLine(1, 5, points)
    drawLine(2, 6, points)
    drawLine(2, 3, points)
    drawLine(3, 7, points)
    # drawLine(4, 5, points)
    # drawLine(4, 7, points)
    drawLine(6, 5, points)
    drawLine(6, 7, points)






    pygame.display.update()
