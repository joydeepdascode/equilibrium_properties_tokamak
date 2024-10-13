### Program for T1 Tokamak

from numpy import loadtxt,linspace,arange,zeros,array,append,meshgrid
from matplotlib.pyplot import xlabel,ylabel,plot,show,contour,colorbar,xticks,yticks,contourf,title,savefig,imshow
from math import pi,cos,sin,tan,sqrt,atan,atan2,modf,floor
from scipy import interpolate

"""Data_of_T1_Tokamak"""

grid_horizon=      #No. of horizontal grid points
grid_vertical=     #No. of vertical   grid points
rdim=     #Physical R dimension of the grid (m) =   0.850000024    
zdim=      #Physical Z dimension of the grid (m) =    1.29999995    
rmid=     #R coordinate of the geometric center (m) =   0.750000000    
rmin=     #R at the leftmost end of the grid (m) =   0.250000000    
zmid=      #Z coordinate at the centre of the grid (m) =   0.00000000    
bcenter=      #Value of Magnetic Field at the magnetic axis


"""Evaluate_min_max_of_R,Z"""


rmax=rmin+rdim
zmin=zmid-(zdim/2)
zmax=zmid+(zdim/2)


r=linspace(rmin,rmax,grid_horizon)     #Creating r, z axis for later use 
z=linspace(zmin,zmax,grid_vertical)
rdif=r[2]-r[1]
zdif=z[2]-z[1]

dr = (rmax-rmin)/grid_horizon          #Grid Size to be used later
dz = (zmax-zmin)/grid_vertical

#################################################################

"""Generate_fpol_function"""

fpol=loadtxt("T1_fpol.txt",float)

xlabel("N(psi), in terms of grids")
ylabel("F(psi)")
plot(abs(fpol))
savefig('T1_fpol.png', dpi=300, bbox_inches='tight')
show()

################################################################

"""Generate_psizr_function"""

#Loading Data for psizr contour, along with limiters

psizr=loadtxt("T1_psizr.txt",float)
rlimiter=loadtxt("T1_rlimiter.txt",float)
zlimiter=loadtxt("T1_zlimiter.txt",float)
rbbbs=loadtxt("T1_rbbbs.txt",float)
zbbbs=loadtxt("T1_zbbbs.txt",float)


#Plot of psizr
x,y=meshgrid(r,z)
contour(x,y,psizr,20)
colorbar()
xlabel("R(in m)")
ylabel("Z(in m)")
title("psizr")
plot(rlimiter,zlimiter,color="red")
plot(rbbbs,zbbbs,color="black")
savefig('T1_psizr.png', dpi=300, bbox_inches='tight')
show()


################################################################


"""B_radial"""

Br=zeros(shape=(65,65))                       
for iz in range(0,grid_horizon):
      for i in range(0,64):
            Br[i,iz]=-(1/(rmid*bcenter))*(psizr[i+1,iz]-psizr[i,iz])/dz

      
for i in range(0,65):
      for j in range(0,64):
            Br[i,j]=(1/(rmin+i*(rmax-rmin)/64))*Br[i,j]


contour(x,y,Br,20)
colorbar()
title("B_radial")
xlabel("R(in m)")
ylabel("Z(in m)")
savefig('T1_B_radial_contour.png', dpi=300, bbox_inches='tight')
show()
title("B_radial")
imshow(Br, extent=[rmin, rmax, zmin,zmax-zdif], origin='lower')
colorbar()
xlabel("R(in m)")
ylabel("Z(in m)")
savefig('T1_B_radial_rgb.png', dpi=300, bbox_inches='tight')
show()


##################################


"""B_z"""


Bz=zeros(shape=(grid_horizon,grid_vertical))
for ir in range(0,grid_horizon):
      for j in range(0,grid_vertical-1):
            Bz[ir,j]=(1/(bcenter*rmid))*(psizr[ir,j+1]-psizr[ir,j])/dr
     
      
for i in range(0,grid_vertical):
      for j in range(0,grid_horizon-1):
            Bz[i,j]=(1/(rmin+i*(rmax-rmin)/(grid_horizon-1)))*Bz[i,j]



contour(x,y,Bz,20)
title("B_z")
colorbar()
xlabel("R(in m)")
ylabel("Z(in m)")
savefig('T1_B_z_contour.png', dpi=300, bbox_inches='tight')
show()
imshow(Bz, extent=[rmin, rmax-rdif, zmin,zmax], origin='lower')
colorbar()
title("B_z")
xlabel("R(in m)")
ylabel("Z(in m)")
savefig('T1_B_z_rgb.png', dpi=300, bbox_inches='tight')
show()



#############################################


##### B_Zeta Re-evaluation #####


"""B_zeta"""


dim=65

min1=min(psizr[0])
for i in range(1,dim):
      min2=min(psizr[i])
      if(min2<min1):
            min1=min2

max1=max(psizr[0])
for i in range(1,dim):
      max2=max(psizr[i])
      if(max2>max1):
            max1=max2

xf=linspace(min1,max1,dim)
dpsi=xf[4]-xf[3]


Btor=zeros(shape=(dim,dim))
for i in range(0,dim):
      for j in range(0,dim):
            rem1=(psizr[i,j]%dpsi)
            q1=psizr[i,j]//dpsi
            if(rem1<dpsi/2):
                  q=q1
            else:
                  q=q1+1
            ipsi=q+abs(min1//dpsi)      #index of psizr
            index=int(ipsi)
            Btor[i,j]=(1/(bcenter))*(1/(rmin+j*(rmax-rmin)/dim))*fpol[index]

            
x,y=meshgrid(r,z)
contour(x,y,Btor,20)

colorbar()
xlabel("R")
ylabel("Z")
title("B_tor")
savefig('T1_B_tor_contour.png', dpi=300, bbox_inches='tight')
show()
title("B_tor")
imshow(Btor, extent=[rmin, rmax, zmin,zmax], origin='lower')
colorbar()
xlabel("R(in m)")
ylabel("Z(in m)")
title("B_tor")
savefig('T1_B_tor_rgb.png', dpi=300, bbox_inches='tight')
show()



###########   Constructing_Grids  when dsi0=0.1 ##############


def Bp(i,j):                                 #Defining function of Poloidal Magnetic Field
      return sqrt(Br[i,j]**2+Bz[i,j]**2)



#### EVALUATION OF CORRECTED delta_Si   #####
        



dsi0=0.1


point=3
slength=zeros(shape=(point))
n_i=zeros(shape=(point))
      

fr = interpolate.interp2d(x,y,Br,kind='linear')     #Interpolation used to evaluate approximate value of Poloidal magnetic field at updated points   
fz = interpolate.interp2d(x,y,Bz,kind='linear')
drl = dr
R0 = 5*dr
a=1                            #Initial Line
b=4                            #Final Line


for i in range(a,b):
      rdi0=R0+i*drl     #Initial r value from rmid for i-th line
      zdi0=0.0       #Initial z value from zmid for i-th line
      si0=0.0        #Initial length of i-th line(which is to be updated below)
     
      
      ThetaOld = -1    #Reference point so that the below loop stops if during updation of point theta moves in opposite direction
      Theta = 0        #Angle at which the updated point lies
      
      
      while( Theta > ThetaOld ):
            Bbr=fr(rmid+rdi0,zdi0+zmid)
            Bbz=fz(rmid+rdi0,zdi0+zmid)
            Bbp=sqrt(Bbr**2+Bbz**2)
             
            drd1=dsi0*(-Bbr/Bbp)               #Euler's Method
            drz1=dsi0*(-Bbz/Bbp)
            
            rdij=rdi0+drd1                     
            zdij=zdi0+drz1
            sij=si0+dsi0
            slength[i-1]=abs(sij)
            n_i[i-1]=int(abs(sij)/dsi0)

            rdi0=rdij
            zdi0=zdij
            si0=sij
            
            ThetaOld = Theta
            Theta=(2*pi+atan2(zdi0,rdi0))%(2*pi)
            
            
      
           

dsii=zeros(shape=(point))      ##### DEFINING ARRAY FOR CORRECTED delta_Si

for v in range(0,point):
      dsii[v]=slength[v]/(n_i[v]-1) 



######  EVALUATION OF GRID POINTS IN POLOIDAL PLANE  ######


r_i=[[],[],[]]                 # Defining List to store new values of the grid
z_i=[[],[],[]]

for i in range(a,b):
      rdi0=R0+i*drl       #Initial r value from rmid for i-th line
      zdi0=0.0            #Initial z value from zmid for i-th line
           
      
      ThetaOld = -1
      Theta = 0           
      
      r_ij=r_i[i-1]
      z_ij=z_i[i-1]
      while(Theta > ThetaOld):
            Bbr=fr(rmid+rdi0,zdi0+zmid)
            Bbz=fz(rmid+rdi0,zdi0+zmid)
            Bbp=sqrt(Bbr**2+Bbz**2)
            
            drd1=dsii[i-1]*(-Bbr/Bbp)             
            drz1=dsii[i-1]*(-Bbz/Bbp)
            
            rdij=rdi0+drd1
            zdij=zdi0+drz1
                 
            
            rdi0=rdij
            zdi0=zdij

            
            ThetaOld = Theta
            Theta=(2*pi+atan2(zdi0,rdi0))%(2*pi)
            
            
            r_ij.append(rdi0+rmid)
            z_ij.append(zdi0+zmid)


### Plotting Grids over psizr function ####

x,y=meshgrid(r,z)
contour(x,y,psizr,20)
colorbar()
xlabel("R(in m)")
ylabel("Z(in m)")
title("psizr")
for n in range(0,point):
      plot(r_i[n],z_i[n],'o')
plot(rlimiter,zlimiter,color="red")
plot(rbbbs,zbbbs,color="black")
savefig('T1_grid_ds_0_dot_1.png', dpi=300, bbox_inches='tight')
show()

print("When dsi0 = 0.1")

###########   Constructing_Grids  when dsi0=0.001 ##############


def Bp(i,j):               #Defining function of Poloidal Magnetic Field
      return sqrt(Br[i,j]**2+Bz[i,j]**2)



#### EVALUATION OF CORRECTED delta_Si   #####


dsi0=0.001


slength=zeros(shape=(point))
n_i=zeros(shape=(point))
      




for i in range(a,b):
      rdi0=R0+i*drl     #Initial r value from rmid for i-th line
      zdi0=0.0       #Initial z value from zmid for i-th line
      si0=0.0        #Initial length of i-th line(which is to be updated below)
     
      
      ThetaOld = -1    #Reference point so that the below loop stops if during updation of point theta moves in opposite direction
      Theta = 0        #Angle at which the updated point lies
      
      
      while( Theta > ThetaOld ):
            Bbr=fr(rmid+rdi0,zdi0+zmid)
            Bbz=fz(rmid+rdi0,zdi0+zmid)
            Bbp=sqrt(Bbr**2+Bbz**2)
             
            drd1=dsi0*(-Bbr/Bbp)               #Euler's Method
            drz1=dsi0*(-Bbz/Bbp)
            
            rdij=rdi0+drd1                     
            zdij=zdi0+drz1
            sij=si0+dsi0
            slength[i-1]=abs(sij)
            n_i[i-1]=int(abs(sij)/dsi0)

            rdi0=rdij
            zdi0=zdij
            si0=sij
            
            ThetaOld = Theta
            Theta=(2*pi+atan2(zdi0,rdi0))%(2*pi)
            
           

dsii=zeros(shape=(point))      ##### DEFINING ARRAY FOR CORRECTED delta_Si

for v in range(0,point):
      dsii[v]=slength[v]/(n_i[v]-1) 



######  EVALUATION OF GRID POINTS IN POLOIDAL PLANE  ######


r_i=[[],[],[]]                 # Defining List to store new values of the grid
z_i=[[],[],[]]

for i in range(a,b):
      rdi0=R0+i*drl       #Initial r value from rmid for i-th line
      zdi0=0.0            #Initial z value from zmid for i-th line
           
      
      ThetaOld = -1
      Theta = 0           
      
      r_ij=r_i[i-1]
      z_ij=z_i[i-1]
      while(Theta > ThetaOld):
            Bbr=fr(rmid+rdi0,zdi0+zmid)
            Bbz=fz(rmid+rdi0,zdi0+zmid)
            Bbp=sqrt(Bbr**2+Bbz**2)
            
            drd1=dsii[i-1]*(-Bbr/Bbp)             
            drz1=dsii[i-1]*(-Bbz/Bbp)
            
            rdij=rdi0+drd1
            zdij=zdi0+drz1
                 
            
            rdi0=rdij
            zdi0=zdij

            
            ThetaOld = Theta
            Theta=(2*pi+atan2(zdi0,rdi0))%(2*pi)
            
            
            r_ij.append(rdi0+rmid)
            z_ij.append(zdi0+zmid)


### Plotting Grids over psizr function ####

x,y=meshgrid(r,z)
contour(x,y,psizr,20)
colorbar()
xlabel("R(in m)")
ylabel("Z(in m)")
title("psizr")
for n in range(0,point):
      plot(r_i[n],z_i[n],'o')
plot(rlimiter,zlimiter,color="red")
plot(rbbbs,zbbbs,color="black")
savefig('T1_grid_ds_0_dot_001.png', dpi=300, bbox_inches='tight')
show()

print("When dsi0 = 0.001")