#include <GLUT/GLUT.h>
#include <iostream>
#include <cmath>


GLfloat xRotated, yRotated, zRotated;
int armRot = 0, forearmRot = 0, r = 5;

int selfRotationAngle = 0;
int earthS;

void init(void)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    float mat[4];
    mat[0] = 0.5;
    mat[1] = 0.2;
    mat[2] = 0.4;
    mat[3] = 1.0;
    glMaterialfv(GL_FRONT, GL_AMBIENT, mat);
    mat[0] = 0.7;
    mat[1] = 0.1;
    mat[2] = 0.5;
    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat);
    mat[0] = 0.4;
    mat[1] = 0.3;
    mat[2] = 0.8;
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat);
    glMaterialf(GL_FRONT, GL_SHININESS, 100 * 128.0);
}


void display(void)
{
    // clear the drawing buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // sun
    glPushMatrix();
    glRotatef(selfRotationAngle, 0, 1.0, 0);
    glutSolidSphere(4, 20, 20);
    glPopMatrix();
    
    // earch
    glPushMatrix();
    glRotatef(armRot, 1.0, 1.0, 0);  //rotation around sun
    glTranslatef(10, -10, 0);
    
    glRotatef(selfRotationAngle, 0, 1.0, 0); //rotation around self
    glutSolidSphere(2, 20, 20);
    glPopMatrix();
    
    // moon
    glPushMatrix();
    glRotatef(armRot, 1, 1.0, 0); //rotation around sun
    glTranslatef(10, -10, 0);
    
    glRotatef(armRot, 0, 1, 0); //rotation around earth
    glTranslatef(5, 0, 0);
    
    glRotatef(selfRotationAngle, 0, 1, 0); //rotation around self
    glutSolidSphere(1, 20, 20);
    glPopMatrix();
    
    // light
    glPushMatrix();
    GLfloat lightpos[4] = { 40, 40, 0, 1 };
    glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    
    glutSwapBuffers();
    glFlush();
}


void reshape(int w, int h)
{
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(80, (GLfloat)w / (GLfloat)h, 10.0, 70.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 4.0, 32.0, 0.0, 0.0, 0, 0.0, 1.0, 0.0);
}


void timer(int v) {
    armRot += 5;
    selfRotationAngle += 5;
    
    glutPostRedisplay();
    glutTimerFunc(70, timer, 0);
}


int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(700, 700);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(argv[0]);
    glutTimerFunc(100, timer, 0);
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMainLoop();
    return 0;
}
