clc
clear all
close all

%Logistic Regression

file= xlsread('heart_DD.csv');
trainingSet= length(file(:,1))*0.7;
% trainingSet= 250;
testSet= length(file(:,1))*0.3;

%Normalization Approach
%Hypothesis 1 (Three Features + Theta0)

oldTheta0=10;
oldTheta1=5;
oldTheta2=2;
oldTheta3=1;

x1s= (file(1:trainingSet,1));
x2s= (file(1:trainingSet,4));
x3s= (file(1:trainingSet,5));

x1= file(1:trainingSet,1)/max(x1s);
x2= file(1:trainingSet,4)/max(x2s); 
x3= file(1:trainingSet,5)/max(x3s); 
y= file(1:trainingSet,14);

x0= ones(1,trainingSet)';
m= trainingSet;

alpha= 0.01;

J= 0;
oldJ= 1;

counterN= 1;


%Gradient Decent

flag=1;
while flag==1
   
oldJ= (1/m)*sum((-y.*log(1./(1+exp(-((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3))))))-((1-y).*log(1-(1./(1+exp(-((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3))))))));

ntheta0= oldTheta0 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3) - y).*x0));
ntheta1= oldTheta1 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3) - y).*x1));
ntheta2= oldTheta2 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3) - y).*x2));
ntheta3= oldTheta3 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3) - y).*x3));

J= (1/m)*sum((-y.*log(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2) + (ntheta3*x3))))))-((1-y).*log(1-(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2) + (ntheta3*x3))))))));

oldTheta0= ntheta0;
oldTheta1= ntheta1;
oldTheta2= ntheta2;
oldTheta3= ntheta3;

costFunction(counterN)= J;
counterN= counterN+1;

q=(oldJ - J)./oldJ;
if q <.000001;
    flag=0;
end
end
axis= [1:length(costFunction)];
figure(1)
plot(axis, costFunction);
title('Hypothesis 1 (Three Features in First Degree Order)');
xlabel('Number of Iterations');
ylabel('Error');

%MSE 

errorH1= (1/m)*sum((-y.*log(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2) + (ntheta3*x3))))))-((1-y).*log(1-(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2) + (ntheta3*x3))))))));

%Test
x1= file(trainingSet+1:trainingSet+testSet,1)/max(file(trainingSet+1:trainingSet+testSet,1));
x2= file(trainingSet+1:trainingSet+testSet,4)/max(file(trainingSet+1:trainingSet+testSet,4)); 
x3= file(trainingSet+1:trainingSet+testSet,5)/max(file(trainingSet+1:trainingSet+testSet,5)); 
yTest= file(trainingSet+1:trainingSet+testSet,14);

x0= ones(1,testSet)';
xTest= [x0 x1 x2 x3];
errorTest1= (1/testSet)*sum((-yTest.*log(1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)) + (ntheta3*xTest(:,4)))))))-((1-yTest).*log(1-(1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)) + (ntheta3*xTest(:,4)))))))));

%Predictions

h1= 1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)) + (ntheta3*xTest(:,4)))));
%% 

%Hypothesis 2 (Two Features + Theta0)

oldTheta0=10;
oldTheta1=5;
oldTheta2=2;

x1s= (file(1:trainingSet,1));
x2s= (file(1:trainingSet,4));

x1= file(1:trainingSet,1)/max(file(1:trainingSet,1));
x2= file(1:trainingSet,4)/max(file(1:trainingSet,4)); 
y= file(1:trainingSet,14);

x0= ones(1,trainingSet)';
m= trainingSet;

alpha= 0.01;

J= 0;
oldJ= 1;

counterN= 1;


%Gradient Decent

flag=1;
while flag==1
   
% oldJ= (1/m)*sum((-y.*log((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2)))-((1-y).*log(1-((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2)))));
oldJ= (1/m)*sum((-y.*log(1./(1+exp(-((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2))))))-((1-y).*log(1-(1./(1+exp(-((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2))))))));

ntheta0= oldTheta0 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) - y).*x0));
ntheta1= oldTheta1 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) - y).*x1));
ntheta2= oldTheta2 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) - y).*x2));

% J= (1/m)*sum((-y.*log((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2)))-((1-y).*log(1-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2)))));
J= (1/m)*sum((-y.*log(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2))))))-((1-y).*log(1-(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2))))))));

oldTheta0= ntheta0;
oldTheta1= ntheta1;
oldTheta2= ntheta2;

costFunction(counterN)= J;
counterN= counterN+1;

q=(oldJ - J)./oldJ;
if q <.000001;
    flag=0;
end
end
axis= [1:length(costFunction)];
figure(2)
plot(axis, costFunction);
title('Hypothesis 2 (Two Features in First Degree Order)');
xlabel('Number of Iterations');
ylabel('Error');

%MSE 

errorH2= (1/m)*sum((-y.*log(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2))))))-((1-y).*log(1-(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2))))))));

%Test
x1= file(trainingSet+1:trainingSet+testSet,1)/max(file(trainingSet+1:trainingSet+testSet,1));
x2= file(trainingSet+1:trainingSet+testSet,4)/max(file(trainingSet+1:trainingSet+testSet,4)); 
yTest= file(trainingSet+1:trainingSet+testSet,14);

x0= ones(1,testSet)';
xTest= [x0 x1 x2];
errorTest2= (1/testSet)*sum((-yTest.*log(1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)))))))-((1-yTest).*log(1-(1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)))))))));

%Predictions

h2= 1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)))));
%% 

%Hypothesis 3 (Two Features till Second Degree)

oldTheta0=10;
oldTheta1=5;
oldTheta2=2;

x1s= (file(1:trainingSet,1)).^1;
x2s= (file(1:trainingSet,4)).^2;

x1= x1s/max(x1s);
x2= x2s/max(x2s); 
y= file(1:trainingSet,14);

x0= ones(1,trainingSet)';
m= trainingSet;

alpha= 0.01;

J= 0;
oldJ= 1;

counterN= 1;


%Gradient Decent

flag=1;
while flag==1
   
% oldJ= (1/m)*sum((-y.*log((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2)))-((1-y).*log(1-((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2)))));
oldJ= (1/m)*sum((-y.*log(1./(1+exp(-((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2))))))-((1-y).*log(1-(1./(1+exp(-((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2))))))));

ntheta0= oldTheta0 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) - y).*x0));
ntheta1= oldTheta1 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) - y).*x1));
ntheta2= oldTheta2 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) - y).*x2));

% J= (1/m)*sum((-y.*log((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2)))-((1-y).*log(1-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2)))));
J= (1/m)*sum((-y.*log(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2))))))-((1-y).*log(1-(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2))))))));

oldTheta0= ntheta0;
oldTheta1= ntheta1;
oldTheta2= ntheta2;

costFunction(counterN)= J;
counterN= counterN+1;

q=(oldJ - J)./oldJ;
if q <.000001;
    flag=0;
end
end
axis= [1:length(costFunction)];
figure(3)
plot(axis, costFunction);
title('Hypothesis 3 (Two Features till Second Degree Order)');
xlabel('Number of Iterations');
ylabel('Error');

%MSE 

errorH3= (1/m)*sum((-y.*log(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2))))))-((1-y).*log(1-(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2))))))));

%Test

x1s= (file(trainingSet+1:trainingSet+testSet,1)).^1;
x2s= (file(trainingSet+1:trainingSet+testSet,4)).^2;

x1= x1s/max(x1s);
x2= x2s/max(x2s); 
yTest= file(trainingSet+1:trainingSet+testSet,14);

x0= ones(1,testSet)';
xTest= [x0 x1 x2];
errorTest3= (1/testSet)*sum((-yTest.*log(1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)))))))-((1-yTest).*log(1-(1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)))))))));

%Predictions

h3= 1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)))));
%% 

%Hypothesis 4 (Three Features till Third Degree)

oldTheta0=10;
oldTheta1=5;
oldTheta2=2;
oldTheta3=1;

x1s= (file(1:trainingSet,1)).^1;
x2s= (file(1:trainingSet,4)).^2;
x3s= (file(1:trainingSet,5)).^3;


x1= x1s/max(x1s);
x2= x2s/max(x2s); 
x3= x3s/max(x3s);
y= file(1:trainingSet,14);

x0= ones(1,trainingSet)';
m= trainingSet;

alpha= 0.01;

J= 0;
oldJ= 1;

counterN= 1;


%Gradient Decent

flag=1;
while flag==1
   
oldJ= (1/m)*sum((-y.*log(1./(1+exp(-((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3))))))-((1-y).*log(1-(1./(1+exp(-((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3))))))));

ntheta0= oldTheta0 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3) - y).*x0));
ntheta1= oldTheta1 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3) - y).*x1));
ntheta2= oldTheta2 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3) - y).*x2));
ntheta3= oldTheta3 - ((alpha/m)*sum(((oldTheta0*x0) + (oldTheta1*x1) + (oldTheta2*x2) + (oldTheta3*x3) - y).*x3));

J= (1/m)*sum((-y.*log(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2) + (ntheta3*x3))))))-((1-y).*log(1-(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2) + (ntheta3*x3))))))));

oldTheta0= ntheta0;
oldTheta1= ntheta1;
oldTheta2= ntheta2;
oldTheta3= ntheta3;

costFunction(counterN)= J;
counterN= counterN+1;

q=(oldJ - J)./oldJ;
if q <.000001;
    flag=0;
end
end
axis= [1:length(costFunction)];
figure(4)
plot(axis, costFunction);
title('Hypothesis 4 (Three Features till Third Degree Order)');
xlabel('Number of Iterations');
ylabel('Error');

%MSE 

errorH4= (1/m)*sum((-y.*log(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2) + (ntheta3*x3))))))-((1-y).*log(1-(1./(1+exp(-((ntheta0*x0) + (ntheta1*x1) + (ntheta2*x2) + (ntheta3*x3))))))));

%Test

x1s= (file(trainingSet+1:trainingSet+testSet,1)).^1;
x2s= (file(trainingSet+1:trainingSet+testSet,4)).^2;
x3s= (file(trainingSet+1:trainingSet+testSet,5)).^3;

x1= x1s/max(x1s);
x2= x2s/max(x2s); 
x3= x3s/max(x3s);
yTest= file(trainingSet+1:trainingSet+testSet,14);

x0= ones(1,testSet)';
xTest= [x0 x1 x2 x3];
errorTest4= (1/testSet)*sum((-yTest.*log(1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)) + (ntheta3*xTest(:,4)))))))-((1-yTest).*log(1-(1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)) + (ntheta3*xTest(:,4)))))))));

%Predictions

h4= 1./(1+exp(-((ntheta0*xTest(:,1)) + (ntheta1*xTest(:,2)) + (ntheta2*xTest(:,3)) + (ntheta3*xTest(:,4)))));
