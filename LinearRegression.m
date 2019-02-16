clc
clear all
close all

%Moataz Ahmed Samy, 34-764
%Machine Learning ASSIGNMENT 1
%Linear Regression with Multiple Variables

file= xlsread('completeData.csv');

%Mean Normalization Approach
%Hypothesis (Three Features)

oldTheta0=10;
oldTheta1=5;
oldTheta2=2;
oldTheta3=1;

m= 17999;
x0= ones(1,m)';

x1s= (file(1:m,4)).^1;
x2s= (file(1:m,8)).^1;
x3s= (file(1:m,5)).^1;
 
x1= (x1s-mean(x1s))/std(x1s);
x2= (x2s-mean(x2s))/std(x2s);
x3= (x3s-mean(x3s))/std(x3s);
y= ((file(1:m,3))-mean(file(1:m,3)))/std(file(1:m,3));


alpha= 0.01;

J= 0;
oldJ= 1;

counter= 1;

x= [x0 x1 x2 x3];

%Gradient Decent

flag=1;
while flag==1

oldJ= (1/(2*m))*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) + (oldTheta3*x(:,4)) - y).^2);    
    
theta0= oldTheta0 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) + (oldTheta3*x(:,4)) - y).*x(:,1)));
theta1= oldTheta1 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) + (oldTheta3*x(:,4)) - y).*x(:,2)));
theta2= oldTheta2 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) + (oldTheta3*x(:,4)) - y).*x(:,3)));
theta3= oldTheta3 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) + (oldTheta3*x(:,4)) - y).*x(:,4)));
J= (1/(2*m))*sum(((theta0*x(:,1)) + (theta1*x(:,2)) + (theta2*x(:,3)) + (theta3*x(:,4)) - y).^2);

oldTheta0= theta0;
oldTheta1= theta1;
oldTheta2= theta2;
oldTheta3= theta3;

mse(counter)= J;
counter= counter+1;

q=(oldJ - J)./oldJ;
if q <.000001;
    flag=0;
end
end
axis= [1:length(mse)];
figure(1)
plot(axis, mse);
title('Hypothesis 1 (Three Features in First Degree Order)');
xlabel('Number of Iterations');
ylabel('MSE');

%Normal Equation

RxxInverse= inv(x'*x);
Rxy= x'*y;
theta= RxxInverse*Rxy;

theta0nEQ= theta(1);
theta1nEQ= theta(2);
theta2nEQ= theta(3);
theta3nEQ= theta(4);

%MSE 

errorH1= (1/(2*m))*sum(((theta0*x(:,1)) + (theta1*x(:,2)) + (theta2*x(:,3)) - y).^2);

%Test

n= length(file(:,1))-m;

x0= ones(1,n)';
x1= ((file(m+1:m+n,4))-mean(file(m+1:m+n,4)))/std(file(m+1:m+n,4));
x2= ((file(m+1:m+n,8))-mean(file(m+1:m+n,8)))/std(file(m+1:m+n,8));
yTest= ((file(m+1:m+n,3))-mean(file(m+1:m+n,3)))/std(file(m+1:m+n,3));

xTest= [x0 x1 x2];
errorTest1= (1/(2*n))*sum(((theta0*xTest(:,1)) + (theta1*xTest(:,2)) + (theta2*xTest(:,3)) - yTest).^2);

%Predictions

h1= (((theta0*x0) + (theta1*x1) + (theta2*x2))*std(file(m+1:m+n,3))) + mean(file(m+1:m+n,3));
%% 

%Hyphothesis 2

flag=1;
while flag==1

oldJ= (1/(2*m))*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) - y).^2);    
    
theta0= oldTheta0 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) - y).*x(:,1)));
theta1= oldTheta1 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) - y).*x(:,2)));
theta2= oldTheta2 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) - y).*x(:,3)));

J= (1/(2*m))*sum(((theta0*x(:,1)) + (theta1*x(:,2)) + (theta2*x(:,3)) - y).^2);

oldTheta0= theta0;
oldTheta1= theta1;
oldTheta2= theta2;

mse(counter)= J;
counter= counter+1;

q=(oldJ - J)./oldJ;
if q <.000001;
    flag=0;
end
end
axis= [1:length(mse)];
figure(2)
plot(axis, mse);
title('Hypothesis 2 (Two Features in First Degree Order)');
xlabel('Number of Iterations');
ylabel('MSE');

%Normal Equation

RxxInverse= inv(x'*x);
Rxy= x'*y;
theta= RxxInverse*Rxy;

theta0nEQ= theta(1);
theta1nEQ= theta(2);
theta2nEQ= theta(3);

%MSE 

errorH2= (1/(2*m))*sum(((theta0*x(:,1)) + (theta1*x(:,2)) + (theta2*x(:,3)) - y).^2);

%Test

n= length(file(:,1))-m;

x0= ones(1,n)';
x1= ((file(m+1:m+n,4))-mean(file(m+1:m+n,4)))/std(file(m+1:m+n,4));
x2= ((file(m+1:m+n,8))-mean(file(m+1:m+n,8)))/std(file(m+1:m+n,8));
yTest= ((file(m+1:m+n,3))-mean(file(m+1:m+n,3)))/std(file(m+1:m+n,3));

xTest= [x0 x1 x2];
errorTest2= (1/(2*n))*sum(((theta0*xTest(:,1)) + (theta1*xTest(:,2)) + (theta2*xTest(:,3)) - yTest).^2);

%Predictions

h2= (((theta0*x0) + (theta1*x1) + (theta2*x2))*std(file(m+1:m+n,3))) + mean(file(m+1:m+n,3));
%% 

%Hypothesis 3

x1s= (file(1:m,4)).^1;
x2s= (file(1:m,8)).^2;

x0= ones(1,m)';
x1= (x1s-mean(x1s))/std(x1s);
x2= (x2s-mean(x2s))/std(x2s);
y= ((file(1:m,3))-mean(file(1:m,3)))/std(file(1:m,3));

x= [x0 x1 x2];

flag=1;
while flag==1

oldJ= (1/(2*m))*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) - y).^2);    
    
theta0= oldTheta0 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) - y).*x(:,1)));
theta1= oldTheta1 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) - y).*x(:,2)));
theta2= oldTheta2 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) - y).*x(:,3)));

J= (1/(2*m))*sum(((theta0*x(:,1)) + (theta1*x(:,2)) + (theta2*x(:,3)) - y).^2);

oldTheta0= theta0;
oldTheta1= theta1;
oldTheta2= theta2;

mse(counter)= J;
counter= counter+1;

q=(oldJ - J)./oldJ;
if q <.000001;
    flag=0;
end
end
axis= [1:length(mse)];
figure(3)
plot(axis, mse);
title('Hypothesis 3 (Two Features till Second Degree Order)');
xlabel('Number of Iterations');
ylabel('MSE');

%Normal Equation

RxxInverse= inv(x'*x);
Rxy= x'*y;
theta= RxxInverse*Rxy;

theta0nEQ= theta(1);
theta1nEQ= theta(2);
theta2nEQ= theta(3);

%MSE 

errorH3= (1/(2*m))*sum(((theta0*x(:,1)) + (theta1*x(:,2)) + (theta2*x(:,3)) - y).^2);

%Test

n= length(file(:,1))-m;

x0= ones(1,n)';
x1s= ((file(m+1:m+n,4))).^1;
x2s= ((file(m+1:m+n,8))).^2;

x1= (x1s-mean(x1s))/std(x1s);
x2= (x2s-mean(x2s))/std(x2s);

yTest= ((file(m+1:m+n,3))-mean(file(m+1:m+n,3)))/std(file(m+1:m+n,3));

xTest= [x0 x1 x2];
errorTest3= (1/(2*n))*sum(((theta0*xTest(:,1)) + (theta1*xTest(:,2)) + (theta2*xTest(:,3)) - yTest).^2);

%Predictions

h3= (((theta0*(x0)) + (theta1*(x1)) + (theta2*(x2)))*std(file(m+1:m+n,3))) + mean(file(m+1:m+n,3));
%% 

%Hypothesis 4

x1s= (file(1:m,4)).^1;
x2s= (file(1:m,8)).^2;
x3s= (file(1:m,5)).^3;

x0= ones(1,m)';
x1= (x1s-mean(x1s))/std(x1s);
x2= (x2s-mean(x2s))/std(x2s);
x3= (x3s-mean(x3s))/std(x3s);
y= ((file(1:m,3))-mean(file(1:m,3)))/std(file(1:m,3));

x= [x0 x1 x2 x3];

flag=1;
while flag==1

oldJ= (1/(2*m))*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) + (oldTheta3*x(:,4)) - y).^2);    
    
theta0= oldTheta0 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) + (oldTheta3*x(:,4)) - y).*x(:,1)));
theta1= oldTheta1 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) + (oldTheta3*x(:,4)) - y).*x(:,2)));
theta2= oldTheta2 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) + (oldTheta3*x(:,4)) - y).*x(:,3)));
theta3= oldTheta3 - ((alpha/m)*sum(((oldTheta0*x(:,1)) + (oldTheta1*x(:,2)) + (oldTheta2*x(:,3)) + (oldTheta3*x(:,4)) - y).*x(:,4)));

J= (1/(2*m))*sum(((theta0*x(:,1)) + (theta1*x(:,2)) + (theta2*x(:,3)) + (theta3*x(:,4)) - y).^2);

oldTheta0= theta0;
oldTheta1= theta1;
oldTheta2= theta2;
oldTheta3= theta3;

mse(counter)= J;
counter= counter+1;

q=(oldJ - J)./oldJ;
if q <.000001;
    flag=0;
end
end
axis= [1:length(mse)];
figure(4)
plot(axis, mse);
title('Hypothesis 4 (Three Features till Third Degree Order)');
xlabel('Number of Iterations');
ylabel('MSE');

%Normal Equation

RxxInverse= inv(x'*x);
Rxy= x'*y;
theta= RxxInverse*Rxy;

theta0nEQ= theta(1);
theta1nEQ= theta(2);
theta2nEQ= theta(3);
theta3nEQ= theta(4);

%MSE 

errorH4= (1/(2*m))*sum(((theta0*x(:,1)) + (theta1*x(:,2)) + (theta2*x(:,3)) + (theta3*x(:,4)) - y).^2);

%Test

n= length(file(:,1))-m;

x0= ones(1,n)';

x1s= ((file(m+1:m+n,4))).^1;
x2s= ((file(m+1:m+n,8))).^2;
x3s= ((file(m+1:m+n,5))).^3;


x1= (x1s-mean(x1s))/std(x1s);
x2= (x2s-mean(x2s))/std(x2s);
x3= (x3s-mean(x3s))/std(x3s);
yTest= ((file(m+1:m+n,3))-mean(file(m+1:m+n,3)))/std(file(m+1:m+n,3));

xTest= [x0 x1 x2 x3];
errorTest4= (1/(2*n))*sum(((theta0*xTest(:,1)) + (theta1*xTest(:,2)) + (theta2*xTest(:,3)) + (theta3*xTest(:,4)) - yTest).^2);

%Predictions

h4= (((theta0*x0) + (theta1*x1) + (theta2*(x2)) + (theta3*(x3)))*std(file(m+1:m+n,3))) + mean(file(m+1:m+n,3));
