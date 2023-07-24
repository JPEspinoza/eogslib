% Evolving Granular Neural Network - eGNN
% Daniel Furtado Leite - 2010
%--------------------------------------------------------------------------
%Loading data
clear all; close all; clc
load eGNNParameters
x1 = 0:0.1:10;
x2 = 0:0.1:10;
for(i=1:size(x1,2))
    for(j=1:size(x2,2))
        X((i-1)*100+j,:) = [x1(i) x2(j)];           %Inputs
    end
end
n = size(X,2);                                      %No. of inputs
m = 1;                                              %No. of outputs
%--------------------------------------------------------------------------
%Tracing the decision boundary
for (counter = 1:size(X,1))
    x = X(counter,:);
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
%--------------------------------------------------------------------------
    %Estimated class
    T(counter) = Target(pos,:);
%--------------------------------------------------------------------------     
end
figure(1)
    for(i=1:size(X,1))
        if(T(i) == 0)
            plot(X(i,1),X(i,2),'*r'),hold on    %Class 1 
        elseif(T(i) == 1)
            plot(X(i,1),X(i,2),'*g'), hold on   %Class 2
        end
    end
%     %Ploting the centers of the granules
%     for(i=1:c)
%         if(Target(i) == 0)
%             plot(Center(i,1),Center(i,2),'bo'),hold on 
%         elseif(Target(i) == 1)
%             plot(Center(i,1),Center(i,2),'ko'),hold on 
%         end
%     end
%--------------------------------------------------------------------------       
load Datastream
%Visualizing the 2nd moment
for(i=size(Data,1)/2:size(Data,1))
    if(Data(i,3)==0)
        hold on, plot(Data(i,1),Data(i,2),'<b')
    elseif(Data(i,3)==1)
        hold on, plot(Data(i,1),Data(i,2),'^m')
    end
end
%--------------------------------------------------------------------------

    