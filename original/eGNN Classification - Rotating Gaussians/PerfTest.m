% Evolving Granular Neural Network - eGNN
% Daniel Furtado Leite - 2010
%--------------------------------------------------------------------------
%Loading data
clear all; close all; clc
load DataStream Data
X = [Data(:,1:2)];                              %Inputs
Y = [Data(:,3)];                                %Outputs
n = size(X,2);                                  %No. of inputs
m = size(Y,2);                                  %No. of outputs
load eGNNParameters
%--------------------------------------------------------------------------
Correct = 0;
Wrong = 0;
for (counter = 1:size(X,1))
    x = X(counter,:);
    y = Y(counter,:);
    %Feeding x into the neural network
    %Computing the computability of x and the existing granules
    for(i = 1:c)                    %For all granules
        for(j = 1:n)                %For all features
            if (x(j) >= l(i,j) && x(j) < lambda(i,j))
                Xnorm(i,j) = (x(j)-l(i,j))/(lambda(i,j)-l(i,j));
            elseif (x(j) >= lambda(i,j) && x(j) <= Lambda(i,j))
                Xnorm(i,j) = 1;
            elseif (x(j) > Lambda(i,j) && x(j) <= L(i,j))
                Xnorm(i,j) = (L(i,j)-x(j))/(L(i,j)-Lambda(i,j));
            else
                Xnorm(i,j) = 0;
            end
        end
    end                             %Xnorm is the normalized input for each granule
%-------------------------------------------------------------------------- 
    %Multiplying Xnorm by the weights w of each feature
    Xnorm = Xnorm.*w;               %Algebraic product
%--------------------------------------------------------------------------               
    %T-S neuron using nullnorm aggregation (T-norm above) - Computando compatibilidade o entre x e \gamma
    for (i = 1:c)
        if (Xnorm(i,:)>e(i))
            o(i) = min(Xnorm(i,:));         %T-norm min
        elseif (Xnorm(i,:)<e(i))
            o(i) = max(Xnorm(i,:));         %S-norm max
        else
            o(i) = e(i);                    %Absorbing in intermediary squares
        end
    end                                     %o is the compatibility between x and the existing granules
%--------------------------------------------------------------------------           
    %Multiplying o by the weigths delta of each granule
    o = o.*delta;                   %Algebraic product
%--------------------------------------------------------------------------
    %Computing the winner granule for x
    [valor,pos] = max(o);                   %pos is the g-th winner \nu
    
    if(o==0)                                %No granules were activated
        %Approximating the output to the nearest granule
        %Calculating the center of the granules
        for(i=1:c)
            Granule(i,:) = [l(i,:) lambda(i,:) Lambda(i,:) L(i,:)];
            Center(i,:) = [Granule(i,1)+(Granule(i,5)-Granule(i,3))/2 Granule(i,2)+(Granule(i,6)-Granule(i,4))/2];
            dist(i) = norm(x-Center(i,:),2);  %Euclidean distance
        end
        [trash,pos] = min(dist);
    end
    
%-------------------------------------------------------------------------
%This part of the code must be reviewed for each case

%     %Class 1 conviction
%     Conv1(counter) = o(1)+o(3)+o(5)-o(2)-o(4);
%     Pred1(counter) = Target(pos);
%     T = [Pred1;Conv1]';

    %Classes 1 2 conviction
%     Conv1(counter) = o(1)+o(3)+o(5)-o(2)-o(4);
%     ConvC1 = (Conv1-min(Conv1))/(max(Conv1)-min(Conv1)); 
%     Conv2(counter) = -o(1)-o(3)-o(5)+o(2)+o(4);
%     ConvC2 = (Conv2-min(Conv2))/(max(Conv2)-min(Conv2)); 
    
    Conv1(counter) = o(1)+o(3)-o(2)-o(4);
    ConvC1 = (Conv1-min(Conv1))/(max(Conv1)-min(Conv1)); 
    Conv2(counter) = -o(1)-o(3)+o(2)+o(4);
    ConvC2 = (Conv2-min(Conv2))/(max(Conv2)-min(Conv2)); 

%-------------------------------------------------------------------------
    
    if(y==0)
        Target1(counter,:) = [1 0];
    else
        Target1(counter,:) = [0 1];
    end

%--------------------------------------------------------------------------
    %Computing the output error
    yEstimated = Target(pos,:);              %Estimated class
    y;                                      %Class accompanying x
    epsilon = yEstimated - y;               %Error
%--------------------------------------------------------------------------     
    %Counting rights and wrongs
    if(counter>size(Data,1)/2)
        if (epsilon == 0)
            Correct = Correct + 1;
        else
            Wrong = Wrong + 1;
        end
    end
%--------------------------------------------------------------------------
    %Display
    yEst(counter,:) = yEstimated;
end
Correct = Correct
Wrong = Wrong
%--------------------------------------------------------------------------

plotroc([Target1]',[[ConvC1; ConvC2]']')