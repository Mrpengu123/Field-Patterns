%Code by Ornella Mattie and Kyle Smith
%6/6/2024

clear all
close all
format long

%Mediums
med1 = 1;
med2 = 1;
med3 = 1;

%Medium Function
g = @(x) x;

%Forwards Coefficients 
Jf = @(a,b) (g(a)+g(b))/(2*g(a));
%Backwards Coefficients
Jb = @(a,b) (g(b)-g(a))/(2*g(a));
%Transfer Coefficients 
Jt = @(a,b) (2*g(a))/(g(b)+g(a));
%Reflection Coefficients
Jr = @(a,b) (g(b)-g(a))/(g(b)+g(a));

%Green function (e.g., 4 points in the unit cell -> 4 entries for the Green function, here you can consider one cell only)
g=zeros(4,4);

%Define the angle from 0 to 2pi that represents k:
phi=linspace(0,2*pi,2000);

%By definition omega is related to the eigenvalues of the matrix
%representing the Green function, so for a 4x4 matrix we expect a max of 4
%eigenvalues -> 4 omegas
omega1=zeros(length(phi),1);
omega2=zeros(length(phi),1);
omega3=zeros(length(phi),1);
omega4=zeros(length(phi),1);

%The 4 omegas for the imaginary eigenvalues
omega1i=zeros(length(phi),1);
omega2i=zeros(length(phi),1);
omega3i=zeros(length(phi),1);
omega4i=zeros(length(phi),1);

%Take the calculations you already have for the Green function matrix and
%modify them so that the characteristic lines that end up in the previous
%cell are multiplied by exp(-1*phi) and those that end up in the next cell
%by exp(i*phi) 

%If your lines end up in multiple cells then add them with the respective
%exp(.) multiples

for l=1:1:length(phi)    
g(1,1)= Jb(med1, med2) * Jt(med2, med1) * Jr(med1, med3)  +  exp(1i*phi(l)) * Jb(med1, med2) * Jr(med2, med1) * Jf(med2, med3) * Jt(med3, med1) + Jf(med1, med2) * Jr(med2, med1) * Jb(med2, med3) * Jt(med3, med1);
g(1,2)= exp(-1i*phi(l)) * Jb(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jf(med3, med1)  +  Jb(med1, med2) * Jr(med2, med1) * (Jb(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jf(med3,med1)) + Jf(med1, med2) * Jr(med2, med1) * (Jf(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jb(med3, med1))  +  exp(1i*phi(l)) * Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jb(med3, med1);
g(1,3)= exp(-1i*phi(l)) * Jb(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jb(med3, med1)  +  Jb(med1, med2) * Jr(med2, med1) * (Jb(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jb(med3,med1)) + Jf(med1, med2) * Jr(med2, med1) * (Jf(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jf(med3, med1))  +  exp(1i*phi(l)) * Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jf(med3, med1);
g(1,4)= exp(-1i*phi(l)) * Jb(med1, med2) * Jr(med2, med1) * Jb(med2, med3) * Jt(med3, med1) + Jf(med1, med2) * Jr(med2, med1) * Jf(med2, med3) * Jt(med3, med1)  +  Jf(med1, med2) * Jt(med2, med1) * Jr(med1, med3);

g(2,1)= Jt(med1, med2) * Jb(med2, med3) * Jt(med3, med1);
g(2,2)= exp(-1i*phi(l)) * Jt(med1, med2) * (Jf(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jf(med3,med1))  +  Jr(med1, med2) * Jt(med3, med3) * Jb(med3, med1);  
g(2,3)= exp(-1i*phi(l)) * Jt(med1, med2) * (Jf(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jb(med3,med1))  +  Jr(med1, med2) * Jt(med2, med3) * Jf(med3, med1);
g(2,4)= exp(-1i*phi(l)) * exp(-1i*phi(l)) * Jt(med1, med2) * Jf(med2, med3) * Jt(med3, med1)  +  exp(-1i*phi(l)) * Jr(med1, med2) * Jr(med1, med3);

g(3,1)= exp(1i*phi(l)) * Jb(med1, med2) * Jr(med2, med1) * Jf(med2, med3) * Jt(med3,med1) + Jf(med1, med2) * Jr(med2, med1) * Jb(med2, med3) * Jt(med3, med1)  +  exp(1i*phi(l)) * exp(1i*phi(l)) * Jt(med1, med2) * Jf(med2, med3) * Jt(med3, med1);
g(3,2)= Jr(med1, med2) * Jr(med1, med3)  +  exp(1i*phi(l)) * Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jb(med3, med1);
g(3,3)= Jt(med1, med2) * (Jb(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jf(med3,med1))  +  exp(1i*phi(l)) * Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jf(med3, med1);
g(3,4)= Jt(med1, med2) * (Jb(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jb(med3,med1));

g(4,1)= Jf(med1, med2) * Jt(med2, med1) * Jr(med1, med3)  +  exp(1i*phi(l)) * Jf(med1, med2) * Jr(med2, med1) * Jf(med2, med3) * Jt(med3,med1) + Jb(med1, med2) * Jr(med2, med1) * Jb(med2, med3) * Jt(med3, med1);
g(4,2)= exp(-1i*phi(l)) * Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jf(med3, med1)  +  Jf(med1, med2) * Jr(med2, med1) * (Jb(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jf(med3,med1)) + Jb(med1, med2) * Jr(med2, med1) * (Jf(med2, med3) * Jr(med3, med1) * Jb(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jf(med3, med1))  +  exp(1i*phi(l)) * Jb(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jb(med3, med1);
g(4,3)= exp(-1i*phi(l)) * Jf(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jb(med3, med1)  +  Jf(med1, med2) * Jr(med2, med1) * (Jb(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jf(med2, med3) * Jr(med3, med1) * Jb(med3,med1)) + Jb(med1, med2) * Jr(med2, med1) * (Jf(med2, med3) * Jr(med3, med1) * Jf(med3, med1) + Jb(med2, med3) * Jr(med3, med1) * Jb(med3, med1))  +  exp(1i*phi(l)) * Jb(med1, med2) * Jt(med2, med1) * Jt(med1, med3) * Jf(med3, med1);
g(4,4)= exp(-1i*phi(l)) * Jf(med1, med2) * Jr(med2, med1) * Jb(med2, med3) * Jt(med3,med1) + Jf(med1, med2) * Jr(med2, med1) * Jf(med2, med3) * Jt(med3, med1)  +  Jb(med1, med2) * Jt(med2, med1) * Jr(med1, med3);  
%Compute determinant and eigenvalues of the matrix g
det(g);
[V,D,W]=eig(g);
z=diag(D);
%Use the fact that the eigenvalues z are such that z=exp(i*omega) and
%compute real omegas
omega1(l)=real(log(z(1))*1i);
omega2(l)=real(log(z(2))*1i);
omega3(l)=real(log(z(3))*1i);
omega4(l)=real(log(z(4))*1i);

%compute imaginary omegas
omega1i(l)=imag(log(z(1))*1i);
omega2i(l)=imag(log(z(2))*1i);
omega3i(l)=imag(log(z(3))*1i);
omega4i(l)=imag(log(z(4))*1i);
end

%Plot real omegas against k
figurehLine1=plot(phi,omega1,'.','Color',[0 0.2 0.6]);
hold all
hLine2=plot(phi,omega2,'.','Color',[0 0.2 0.6]);
hLine3=plot(phi,omega3,'.','Color',[0 0.2 0.6]);
hLine4=plot(phi,omega4,'.','Color',[0 0.2 0.6]);

%Plot imaginary omegas against k
figurehLine5=plot(phi,omega1i,'.','Color',[1 0 0]);
hold all
hLine6=plot(phi,omega2i,'.','Color',[1 0 0]);
hLine7=plot(phi,omega3i,'.','Color',[1 0 0]);
hLine8=plot(phi,omega4i,'.','Color',[1 0 0]);

%Formating for the plot
axis([0 2*pi -pi pi])
xticks([0 pi/2 pi 3*pi/2 2*pi])
yticks([-pi -pi/2 0 pi/2 pi])
set(gca,'fontsize',16);
xticklabels({'0','\pi/2','\pi','3\pi/2','2\pi'})
yticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'})
xlabel('$$k$$','Interpreter','Latex','fontsize',20)
ylabel('$$\omega$$','Interpreter','Latex','fontsize',20)
% t=title('$$a$$)');
% set(t,'Interpreter','Latex','fontsize',20);