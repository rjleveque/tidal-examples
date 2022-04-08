
from pylab import *

G = 6.674e-11    # gravitational constant 

Re = 6371e3      # radius of earth (m)
Me = 5.972e24    # mass of earth (kg)

Mm = 7.348e22    # mass of moon (kg)
Rm = 3.844e8     # distance moon to earth (m)

Ms = 1.989e30    # mass of sun (kg)
Rs = 149.6e9     # distance sun to earth (m)

g = G*Me/Re**2

Ag = 1.5*G*Mm*Re/Rm**3
epsilon_moon = 1.5*G*Mm*Re / Rm**3
epsilon_sun = 1.5*G*Ms*Re / Rs**3

cosd = lambda x: cos(x*pi/180)
sind = lambda x: sin(x*pi/180)

def plot_tidal_forces(gshift=False):

    figure(figsize=(12,5))

    # plot unit circle for earth:
    theta = linspace(-pi,pi,100)
    x = cos(theta)
    y = sin(theta)
    plot(x,y,'b')
    axis('equal')
    axis('off')

    theta = linspace(-pi,pi,60)
    Ahor = Ag*sin(2*theta)

    if gshift:
        # shift g so to eliminate constant term
        Aver = Ag*cos(2*theta)
    else:
        Aver = Ag*(cos(2*theta) + 1/3.)

    # scale for good size arrows relative to unit circle:
    Av = Aver * 0.2e6
    Ah = Ahor * 0.2e6

    x = cos(theta)
    y = sin(theta)
    dx = Av*cos(theta) + Ah*sin(theta)
    dy = Av*sin(theta) - Ah*cos(theta)
    #plot([x,x+dx], [y,y+dy], 'b')
    for k in range(len(x)):
        arrow(x[k],y[k],dx[k],dy[k],width=0.009,length_includes_head=True)
        
    arrow(1.7,0,0.3,0,width=0.02)
    text(2.3,-0.02,'moon', fontsize=15)
    title('Tidal forces on Earth');

def plot_tidal_bulges(theta_moon=0, theta_sun=None):

    r_earth = 1 #  lambda theta: ones(theta.shape)
    r_sea =  1.15  # lambda theta: 1.1*r_earth(theta)
    
    theta = linspace(-180, 180, 500)
    x_earth = r_earth * cosd(theta)
    y_earth = r_earth * sind(theta)
    x_sea = r_sea * cosd(theta)
    y_sea = r_sea * sind(theta)
    r_bulge_moon = 0.2*epsilon_moon*Re/(2*g) * cosd(2*(theta - theta_moon))
    x_bulge_moon = x_sea + r_bulge_moon*cosd(theta)
    y_bulge_moon = y_sea + r_bulge_moon*sind(theta)
    
    x_bulge_total = x_bulge_moon
    y_bulge_total = y_bulge_moon

    figure(figsize=(13,9))
    
    plot(x_sea, y_sea, 'b--', label='Undisturbed sea level')
    
    # bulge due to moon:
    #fill(x2,y2,color=[.5,.5,1])


    if theta_sun is not None:
        plot(x_bulge_moon, y_bulge_moon, 'r', label='bulge due to moon')

        r_bulge_sun = 0.2*epsilon_sun*Re/(2*g) * cosd(2*(theta - theta_sun))
        x_bulge_sun = x_sea + r_bulge_sun*cosd(theta)
        y_bulge_sun = y_sea + r_bulge_sun*sind(theta)
        plot(x_bulge_sun, y_bulge_sun, 'c', label='bulge due to sun')

        xs = 1.3*cosd(theta_sun)
        ys = 1.3*sind(theta_sun)
        arrow(xs, ys, 0.2*xs, 0.2*ys,width=0.01,length_includes_head=True)
        text(1.6*xs, 1.6*ys, 'sun', fontsize=15, ha='center')
        
        # adjust total bulge
        x_bulge_total = x_bulge_total + r_bulge_sun*cosd(theta)
        y_bulge_total = y_bulge_total + r_bulge_sun*sind(theta)
        
    else:
        plot(x_bulge_moon, y_bulge_moon, 'b', label='bulge due to moon')
        xs = ys = 0.


    fill(x_bulge_total,y_bulge_total,color=[.6,.6,1])
    
    if theta_sun is not None:
        plot(x_bulge_total,y_bulge_total,'b',label='total bulge')
    
    # plot earth last to cover others:
    fill(x_earth,y_earth,color=[.7,1,.7])
    plot(x_earth,y_earth,'g')
    
    legend(loc='lower left', fontsize=12)

    #arrow(1.3,0,0.3,0,width=0.01)
    #text(1.7,-0.01,'moon', fontsize=15)
    xm = 1.3*cosd(theta_moon)
    ym = 1.3*sind(theta_moon)
    arrow(xm, ym, 0.2*xm, 0.2*ym,width=0.01,length_includes_head=True)
    text(1.4*xm, 1.4*ym, 'moon', fontsize=15, ha='center')

    axis('equal')
    xlim(min(-1.5, 1.4*xm, 1.4*xs), max(1.5, 1.4*xm, 1.4*xs))
    ylim(min(-1.5, 1.4*ym, 1.4*ys), max(1.5, 1.4*ym, 1.4*ys))
    
    axis('off')
    title('\nOcean depth with tidal bulges (greatly amplified)', fontsize=15);
    
    if 0:
        figure()
        #plot(lam, r3(lam-lam_sun*pi/180), 'm')
        #plot(lam, r2(lam-lam_moon*pi/180), 'b')
        plot(theta, x_bulge_moon - x_sea, 'm')
        plot(theta, x_bulge_total - x_sea, 'b')
    
