###############################################################################
"""
pendulum.py

desc: A mildly interactive pygame animated simulation of circular physical
      pendulum.

auth: Craig Wm. Versek (cversek@gmail.com) circa 2008?
"""
###############################################################################
import sys,os
from math import sin,cos,pi,atan2,sqrt
import pygame
from pygame.locals import *
import pygame.surfarray

COLOR = {'black' : (0,0,0),
         'red'   : (255,0,0),
         'green' : (0,255,0),
         'blue'  : (0,0,255)}

SCREEN_WIDTH  = 800
SCREEN_HEIGHT = 800
SCREEN_DIM    = (SCREEN_WIDTH,SCREEN_HEIGHT)
SCREEN_CENTER = (SCREEN_WIDTH//2,SCREEN_HEIGHT//2) 
DEFAULT_INTEGRATION_METHOD = 'Steormer-Verlet'

def gen_pendulum_physics_Heun(dt,theta0,theta_dot0,m,g,l):
    a = float(-m*g*l)
    b = float(m*l*l)
    #initialize generalized coordinates
    q = theta0        
    p = b*theta_dot0
    while True:
        #Update using Heun's Method
        q_dot1 = p/b
        p_dot1 = a*sin(q)
        q_dot2 = (p+dt*p_dot1)/b
        p_dot2 = a*sin(q+dt*q_dot1)
        q += dt/2.0*(q_dot1 + q_dot2)
        p += dt/2.0*(p_dot1 + p_dot2)
        yield q

def gen_pendulum_physics_Steormer_Verlet(dt,theta0,theta_dot0,m,g,l):
    a = float(-m*g*l)
    b = float(m*l*l)
    #initialize generalized coordinates
    q = theta0        
    p = b*theta_dot0
    #this is the force pulling back, note 'a' should be negative 
    f = lambda x: a*sin(x)
    #integrate via the leap-frog method
    while True:
        q_dot = p + 0.5*dt*f(q)
        q += dt*q_dot
        p  = q_dot + 0.5*dt*f(q)  #note that q has changed in between
        yield (q, q_dot)


def gen_pendulum_physics_RK4(dt,theta0,theta_dot0,m,g,l):
    #agregate the physical constants    
    a = 1.0/(m*l*l)    
    b = float(-m*g*l)
    #initialize generalized coordinates
    q = theta0        
    p = theta_dot0/a
    #this is the physics! ... Hamiltonian style
    q_dot = lambda p_arg: a*p_arg        
    p_dot = lambda q_arg: b*sin(q_arg)  
    #Integrate using Runge-Kutta 4th Order Method
    while True:
        k1 = dt*q_dot(p)
        h1 = dt*p_dot(q)        
        k2 = dt*q_dot(p + h1/2.0)
        h2 = dt*p_dot(q + k1/2.0)
        k3 = dt*q_dot(p + h2/2.0)
        h3 = dt*p_dot(q + k2/2.0)
        k4 = dt*q_dot(p + h3)
        h4 = dt*p_dot(q + k3)
        q += ((k1+k4)/2.0 + k2+k3)/3.0
        p += ((h1+h4)/2.0 + h2+h3)/3.0
        yield (q,q_dot(p))


PHYSICS_GENERATORS = {
    'Heun':            gen_pendulum_physics_Heun,
    'Steormer-Verlet': gen_pendulum_physics_Steormer_Verlet,
    'Runge-Kutta 4':   gen_pendulum_physics_RK4,
}

class Pendulum(pygame.sprite.Sprite):
    """renders a fixed pivot pendulum and updates motion according to differential equation"""
    def __init__(self,pivot_vect,length,bob_radius,bob_mass,init_angle, integration_method = DEFAULT_INTEGRATION_METHOD):
        pygame.sprite.Sprite.__init__(self) #call Sprite initializer
        phys_gen_init = PHYSICS_GENERATORS[integration_method]
        self.phys_gen = phys_gen_init(dt = 0.01,theta0 = init_angle,theta_dot0 = 0,m = bob_mass,g = 9.8,l = length/1000.0)        
        self.length = length
        self.bob_radius = bob_radius
        self.bob_mass   = bob_mass
        #deflection from plumb
        self.angle      = init_angle
        #positioning attributes
        self.pivot_vect = pivot_vect         #vector from topleft to pivot of pendulum
        swinglen = (length + bob_radius)     #whole
        #these next two attributes are used by the RenderPlain container class
        self.image  = pygame.Surface((swinglen*2,swinglen*2)).convert()          #create surface just big enough to fit swing
        self.rect   = self.image.get_rect()
        self.rect.topleft = (pivot_vect[0] - swinglen, pivot_vect[1] - swinglen) #place so that pivot is at center 
        #calculate the initial relative bob position in the image
        self.bob_X    = int(length*sin(init_angle)+self.rect.width//2)
        self.bob_Y    = int(length*cos(init_angle)+self.rect.height//2)
        self.bob_rect = None
        #render the pendulum from the parameters
        self._render()
    def _render(self):
        #clear the pendulum surface
        self.image.fill(COLOR['black'])
        bob_pos = (self.bob_X,self.bob_Y)
        #draw the tether
        pygame.draw.aaline(self.image,COLOR['red'],(self.rect.width//2,self.rect.height//2),bob_pos,True)
        #draw the bob
        self.bob_rect = pygame.draw.circle(self.image,COLOR['blue'],bob_pos,self.bob_radius,0)
        x1,y1 = self.bob_rect.topleft
        x2,y2 = self.rect.topleft
        #make the reference absolute
        self.bob_rect.topleft = (x1+x2,y1+y2)  #make the reference absolute
    def update(self):
        #coords relative to pivot
        self.angle = self.phys_gen.__next__()[0]
        angle  = self.angle
        length = self.length
        X = int(length*sin(angle))
        Y = int(length*cos(angle))
        self.bob_X = X + self.rect.width//2
        self.bob_Y = Y + self.rect.height//2
        self._render()

    def update_held(self,mouse_pos):
        m_x,m_y = mouse_pos
        self.bob_X = m_x - self.lever_x - self.bob_rect.width//2
        self.bob_Y = m_y - self.lever_y - self.bob_rect.height//2
        self.length = sqrt((self.bob_X - self.rect.width//2)**2 + (self.bob_Y - self.rect.height//2)**2)
        #calculate the new parameters        
        self.angle = atan2(self.bob_X - self.rect.width//2, self.bob_Y - self.rect.height//2)
        self._render()

    def grab(self,mouse_pos):
        m_x, m_y = mouse_pos
        b_x, b_y = self.bob_rect.center
        self.held = True
        self.lever_x = m_x - b_x
        self.lever_y = m_y - b_y
        print("Grabbed the bob a vector from center (%d,%d)" % (self.lever_x,self.lever_y))

    def point_on_bob(self,point):
        x,y = point
        return self.bob_rect.collidepoint(x,y)

    def release(self):
        #create new physics by dropping pendulum at current angle
        self.phys_gen = gen_pendulum_physics_RK4(0.01,self.angle,0,self.bob_mass,9.8,self.length/1000.0)
   

def main():
    """this function is called when the program starts.
    it initializes everything it needs, then runs in
    a loop until the function returns.
    """
    #Initialize Everything
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_DIM)
    pygame.display.set_caption('Pendulum Simulation')
    #pygame.mouse.set_visible(0)

    #Create The Backgound
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill(COLOR['black'])
    #Prepare Objects
    clock = pygame.time.Clock()
    p1 = Pendulum(pivot_vect=SCREEN_CENTER,length=300,bob_radius=50,bob_mass=1,init_angle=pi/5)
    free_group = pygame.sprite.RenderPlain((p1,))
    held_group = pygame.sprite.RenderPlain()
    #Display The Background
    screen.blit(background, (0, 0))
    pygame.display.flip()

    #Main Loop
    while True:
        clock.tick(60)
        #Handle Input Events
        for event in pygame.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
            elif event.type == MOUSEBUTTONDOWN:
                print("Mouse Button Down")
                mouse_pos = pygame.mouse.get_pos()
                for p in free_group:
                    if p.point_on_bob(mouse_pos):     #if user clicked on the bob grab it
                        p.grab(mouse_pos)
                        held_group.add(p)
                        free_group.remove(p)
            elif event.type is MOUSEBUTTONUP:
                print("Mouse Button Up")
                for p in held_group:
                    p.release()
                    free_group.add(p)
                    held_group.remove(p)
        free_group.update()
        #send the mouse position to the held bobs so we can move them
        mouse_pos = pygame.mouse.get_pos()
        for p in held_group:
            p.update_held(mouse_pos)
        screen.blit(background,(0,0))
        free_group.draw(screen)
        held_group.draw(screen)
        #pygame.draw.circle(screen, COLOR['blue'], SCREEN_CENTER, 50, 0)
        pygame.display.flip()

if __name__ == "__main__":
    main()