%PhD Thesis - Daniel Furtado Leite - UNICAMP
%Section 7.2 Classification of rotating Gaussians
%This algorithm generates the data stream
%--------------------------------------------------------------------------
clear all; close all; clc;
%Enter the center and standard deviation of the 1st Gaussian
%Class 1
    Center1 = [4 4];                    %Center
    StdDev1 = [.8 .8];                  %Standard deviation
    NoEx1 = 100;                        %Number of examples
    for (j=1:NoEx1)                     %For all examples
        for (i=1:size(Center1,2))       %For all dimensions
            Data(j,i) = Center1(i) + StdDev1(i)*randn;
        end
    end
    %Display
    figure(1), plot(Data(:,1),Data(:,2),'b*'), hold on
    DataC1M1 = Data;
%--------------------------------------------------------------------------
%Enter the center and standard deviation of the 2nd Gaussian
%Class 2
    Center2 = [6 6];                    %Center
    StdDev2 = [.8 .8];                  %Standard deviation
    NoEx2 = 100;                        %Number of examples
    for (j=1:NoEx2)                     %For all examples
        for (i=1:size(Center2,2))       %For all dimensions
            Data(j,i) = Center2(i) + StdDev2(i)*randn;
        end
    end
    %Display
    figure(1), plot (Data(:,1),Data(:,2),'ko')
    title('200 initial samples - no concept drift')
    axis([0 10 0 10]), xlabel('x1'), ylabel('x2')
    DataC2M1 = Data;
%--------------------------------------------------------------------------
%Plot: Initial location of the twin Gaussians
x = 0:.1:10; y = 0:.1:10;
for(i=1:size(x,2))
    for(j=1:size(y,2))
        C1(i,j) = exp(-( ((x(i)-Center1(1))^2)/(2*StdDev1(1)^2) + ((y(j)-Center1(2))^2)/(2*StdDev1(2)^2) ));
        C2(i,j) = exp(-( ((x(i)-Center2(1))^2)/(2*StdDev2(1)^2) + ((y(j)-Center2(2))^2)/(2*StdDev2(2)^2) ));
    end
end
figure(2), mesh(x,y,C1), hold on, mesh(x,y,C2)
title('Initial location of the twin Gaussians')
axis([0 10 0 10]), xlabel('x1'), ylabel('x2')
% --------------------------------------------------------------------------
% --------------------------------------------------------------------------
%90 degree anticlockwise rotation around (5,5)
%Initial position of Class 1
theta1 = 225*pi/180;
Dtheta1 = .45*pi/180;
StdDev1 = [.8 .8];
NoEx1 = 200;                            %Number of examples
for (j=1:NoEx1)                     %For all examples
    Center1 = [5+sqrt(2)*cos(theta1+Dtheta1) 5+sqrt(2)*sin(theta1+Dtheta1)];
    theta1 = theta1 + Dtheta1;
    for (i=1:size(Center1,2))       %For all dimensions
        Data1(j,i) = Center1(i) + StdDev1(i)*randn;
        cc1(j,i) = Center1(i);
    end
end
%Display
figure(3)
plot (Data1(:,1),Data1(:,2),'b*'), hold on, plot(cc1(:,1),cc1(:,2),'r*')
DataC1M2 = Data1;
%--------------------------------------------------------------------------
%90 degree anticlockwise rotation around (5,5)
%Initial position of Class 2
theta2 = 45*pi/180;
Dtheta2 = .45*pi/180;
StdDev2 = [.8 .8];
NoEx2 = 200;                            %Number of examples
for (j=1:NoEx2)                     %For all examples
    Center2 = [5+sqrt(2)*cos(theta2+Dtheta2) 5+sqrt(2)*sin(theta2+Dtheta2)];
    theta2 = theta2 + Dtheta2;
    for (i=1:size(Center2,2))       %For all dimensions
        Data2(j,i) = Center2(i) + StdDev2(i)*randn;
        cc2(j,i) = Center2(i);
    end
end
%Display
figure(3), axis([0 10 0 10])
plot (Data2(:,1),Data2(:,2),'ko'), hold on, plot(cc2(:,1),cc2(:,2),'ro')
title('Samples during concept drift')
xlabel('x1'), ylabel('x2')
DataC2M2 = Data2;
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%Plot: Final location of the twin Gaussians
x = 0:.1:10; y = 0:.1:10;
for(i=1:size(x,2))
    for(j=1:size(y,2))
        C1(i,j) = exp(-( ((x(i)-Center1(1))^2)/(2*StdDev1(1)^2) + ((y(j)-Center1(2))^2)/(2*StdDev1(2)^2) ));
        C2(i,j) = exp(-( ((x(i)-Center2(1))^2)/(2*StdDev2(1)^2) + ((y(j)-Center2(2))^2)/(2*StdDev2(2)^2) ));
    end
end
figure(4), mesh(x,y,C1), hold on, mesh(x,y,C2)
title('Final location of the twin Gaussians')
axis([0 10 0 10]), xlabel('x1'), ylabel('x2')
%--------------------------------------------------------------------------
clear Data C1 C2 Center1 Center2 NoEx1 NoEx2 StdDev1 StdDev2 i j x y
%Adding new column representing class label: [x C]
%1st moment
DataC1M1 = [DataC1M1 zeros(size(DataC1M1,1),1)];
DataC2M1 = [DataC2M1 ones(size(DataC2M1,1),1)];
%2nd moment
DataC1M2 = [DataC1M2 zeros(size(DataC1M2,1),1)];
DataC2M2 = [DataC2M2 ones(size(DataC2M2,1),1)];
%--------------------------------------------------------------------------
%Shuffle the data
%1st moment
DataM1 = [DataC1M1; DataC2M1];
for(i=1:1000)
    j = round(rand*(size(DataM1,1)-1))+1;
    k = round(rand*(size(DataM1,1)-1))+1;
    aux(1,:) = DataM1(j,:);
    DataM1(j,:) = DataM1(k,:);
    DataM1(k,:) = aux(1,:);
end
%2nd moment
DataM2 = [];
for(i=1:size(DataC1M2,1))
    if(round(rand)==0)
        DataM2 = [DataM2; DataC1M2(i,:)]; 
    else
        DataM2 = [DataM2; DataC2M2(i,:)];
    end
end
Data = [DataM1; DataM2];
clear aux i j k
%--------------------------------------------------------------------------
save DataStream Data
%--------------------------------------------------------------------------
