% Evolving Granular Neural Network - eGNN
% Daniel Furtado Leite - 2010
%--------------------------------------------------------------------------
%Loading data
clear all; close all; clc
load eGNNParameters
n = size(l,1);                          %No. of inputs
%--------------------------------------------------------------------------
%Granules
for(i=1:c)
    Granule(i,:) = [l(i,:) lambda(i,:) Lambda(i,:) L(i,:)];
end
%--------------------------------------------------------------------------
x = [0:.001:10]';
%Computing the membership value for all x and all granules
for(i = 1:c)
    yx1(:,i) = trapmf(x, [Granule(i,1) Granule(i,3) Granule(i,5) Granule(i,7)]);    %Feature 1
    yx2(:,i) = trapmf(x, [Granule(i,2) Granule(i,4) Granule(i,6) Granule(i,8)]);    %Feature 2
end
%--------------------------------------------------------------------------
%Plot
figure(1)
subplot(2,1,1)          %Feature 1
for(i = 1:c)
    if(Target(i)==0)
        plot(x, [yx1(:,i)],'.r'), hold on
    elseif(Target(i)==1)
        plot(x, [yx1(:,i)],'.g'), hold on
    end
end
subplot(2,1,2)
for(i = 1:c)            %Feature 2
    if(Target(i)==0)
        plot(x, [yx2(:,i)],'.r'), hold on
    elseif(Target(i)==1)
        plot(x, [yx2(:,i)],'.g'), hold on
    end
end
%--------------------------------------------------------------------------
