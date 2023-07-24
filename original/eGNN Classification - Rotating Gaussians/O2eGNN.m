% eGNN for classification of numeric data - Daniel Leite - 2012
% This neural model uses min-max neurons, but can be easily changed for other
% types of T-S neurons, which implement any aggregation function
% (triangular norms, uninorms, nullnorms, compensatory and averaging op.)
%--------------------------------------------------------------------------
%Loading data
clear all; close all; clc
load DataStream Data
X = [Data(:,1:2)];                              %Inputs
Y = [Data(:,3)];                                %Outputs
n = size(X,2);                                  %No. of inputs
m = size(Y,2);                                  %No. of outputs
%--------------------------------------------------------------------------
%Parameters
Rho = .85;                                       %Granularity
hr = 40;                                        %No. of iterations to adapt the granules size / Inactivity: threshold for deletion
eta = 2;
Beta = 0;                                       %Weights (w) drop off constant
chi = 0.1;                                      %Neutral elements (e) increasing constant
zeta = 0.9;                                     %Weights (delta) adjusting constant
c = 0;                                          %No. of granules
w = [];                                         %Weights matrix - evolving layer
e = [];                                         %Vector relat. to the nullneurons neutral element
delta = [];                                     %Weights vector - aggregation layer
counter = 1;
alpha = 0;
%--------------------------------------------------------------------------
%Normalization
Me=minmax(X');
for i=1:size(X,1)
    X(i,:)=(X(i,:)-Me(:,1)')./(Me(:,2)'-Me(:,1)');
end
%--------------------------------------------------------------------------
%Simulating data stream
tic     %Starting chronometer
for(h=1:size(X,1))

    %Chose data pairs sequentially
    x = X(h,:);
    y = Y(h);
    if(h==1)    %First step

        %Creating granule
        c = c+1;
        w(c,1:n) = 1;
        e(c) = 0;
        delta(c) = 1;
        coef(c) = y;                    %Class
        %Input granule
        for (j=1:n)                     %Using trapezoidal mf
            l(c,j) = x(j) - Rho/2;        %Lower bound
            lambda(c,j) = x(j);         %Intermediary point 1
            Lambda(c,j) = x(j);         %Intermediary point 2
            L(c,j) = x(j) + Rho/2;        %Upper bound
            if(l(c,j)<0) l(c,j) = 0; end
            if(L(c,j)>1) L(c,j) = 1; end
        end
        
        %Storing the example
        PointInG = zeros(1,2011);
        PointInG(c,1) = h;
        Act(c)=0;
        
        %Test
        
        %Prediction error - vectors for plot
        C(h) = round(rand);         %Class - flip coin
        StoreNoRule(h) = 0;         %No. of rules
        VecRho(h) = Rho;            %Granule size along iterations
        
        %Desired output becomes available
        if(y==C(h))
            Right(h) = 1; Wrong(h) = 0;
        else
            Right(h) = 0; Wrong(h) = 1;
        end
        
%--------------------------------------------------------------------------

    else        %Other steps
        
        %Test
        
        %Compute compatibility between x and granules
        for(i=1:c)                    %For all granules
            for(j=1:n)                %For all features
                if(x(j)>=l(i,j) && x(j)<=L(i,j)) %Non-empty intersection
                    Xnorm(i,j) = 1 - (abs(x(j)-l(i,j)) + abs(x(j)-lambda(i,j)) + abs(x(j)-Lambda(i,j)) + abs(x(j)-L(i,j)))/4;
                else
                    Xnorm(i,j) = 0;
                end
            end
        end
        
        %Aggregation layer weights
        Xnorm = Xnorm.*w;               

        %T-S neuron: min
        for (i=1:c)
            o(i) = min(Xnorm(i,:));        
        end
        
        %Output layer weights
        o = o.*delta;

        %T-S neuron: max
        [garbage,I] = max(o);   %Granule I is the closest

        
        %Vector for plot ROC (out)
        oClass1 = 0;
        oClass2 = 0;
        for (i=1:size(coef,2))
            if(coef(i)==0)
                oClass1 = max(oClass1,o(i));
            elseif(coef(i)==1)
                oClass2 = max(oClass2,o(i));
            end
        end
        out(h,1) = oClass1/(oClass1+oClass2);
        out(h,2) = oClass2/(oClass1+oClass2);

        
        
        %Prediction error - vectors for plot
        C(h) = coef(I);             %Class predicted
        StoreNoRule(h) = c;         %No. of rules

        %Desired output becomes available
        if(y==C(h))
            Right(h) = 1; Wrong(h) = 0;
        else
            Right(h) = 0; Wrong(h) = 1;
        end

        %------------------------------------------------------------------
        
        %Train
        
        %Rules that can fit x
        I = [];
        for (i=1:c)
            if (o(i)>0)
                if(y==coef(i))
                    I = [I,i];
                end
            end
        end       
        
        Flag = 0;
        if(size(I,2)==0)
            Flag = 1;           %No rule encloses x
        end

        %------------------------------------------------------------------
        
        if(Flag==1)         %Case 0: No granule accommodates x

            %Creating granule
            c = c+1;
            w(c,1:n) = 1;
            e(c) = 0;
            delta(c) = 1;
            coef(c) = y;                    %Class
            %Input granule
            for (j=1:n)                     %Using trapezoidal mf
                l(c,j) = x(j) - Rho/2;              %Lower bound
                lambda(c,j) = x(j);         %Intermediary point 1
                Lambda(c,j) = x(j);         %Intermediary point 2
                L(c,j) = x(j) + Rho/2;              %Upper bound
            end
        
            %Storing the example
            PointInG(c,1) = h;
            Act(c)=0;
            
        else        %Adaptation of the most qualified granule
            
            %Case >2: more than one rule fits x
            if(size(I,2)>=2)  
                [garbage,pos] = max(o(I));   
                aux = []; aux = I(pos);
                I = []; I = aux;        %Granule I should be used for training
            end
            
            %Adapting granule I - antecedent part
            for (j=1:n)
                if (x(j) > ((lambda(I,j)+Lambda(I,j))/2)-Rho/2 && x(j) < lambda(I,j))
                    lambda(I,j) = x(j);       %Core expansion
                end
                if (x(j) > lambda(I,j) && x(j) < ((lambda(I,j)+Lambda(I,j))/2))
                    lambda(I,j) = x(j);       %Core contraction
                end
                if (x(j) > ((lambda(I,j)+Lambda(I,j))/2) && x(j) < Lambda(I,j))
                    Lambda(I,j) = x(j);       %Core contraction
                end
                if (x(j) > Lambda(I,j) && x(j) < ((lambda(I,j)+Lambda(I,j))/2)+Rho/2)
                    Lambda(I,j) = x(j);       %Core expansion
                end
            end
            
            %Checking if support conctraction is necessary
            for(j=1:n)
                if(((lambda(I,j)+Lambda(I,j))/2)-Rho/2 > l(I,j))
                    l(I,j) = ((lambda(I,j)+Lambda(I,j))/2)-Rho/2;
                    if(((lambda(I,j)+Lambda(I,j))/2)-Rho/2 > lambda(I,j))
                        lambda(I,j) = ((lambda(I,j)+Lambda(I,j))/2)-Rho/2;
                    end
                end
                if(((lambda(I,j)+Lambda(I,j))/2)+Rho/2 < L(I,j))
                    L(I,j) = ((lambda(I,j)+Lambda(I,j))/2)+Rho/2;
                    if(((lambda(I,j)+Lambda(I,j))/2)+Rho/2 < Lambda(I,j))
                        Lambda(I,j) = ((lambda(I,j)+Lambda(I,j))/2)+Rho/2;
                    end
                end
            end
                      
            %Adapting aggregation layer weights
            for(j=1:n)
                w(I,j) = w(I,j) - Beta*x(j)*Xnorm(I,j);
            end
            
            %Storing the example
            [garbage,pos] = find(PointInG(I,:) == 0,1,'first');
            PointInG(I,pos) = h;
            Act(I)=0;

        end         %End loop ELSE adaptation
        
    end             %End loop ELSE h = 2, ...
        
    clear Xnorm o
    %----------------------------------------------------------------------

    %Deleting granules
    Act = Act + 1;
    line = -1;
    for(K=1:c)
        if(Act(K)>=hr)
            %Deleting granule K
            l(K,:) = [];
            lambda(K,:) = [];
            Lambda(K,:) = [];
            L(K,:) = [];
            PointInG(K,:) = [];
            coef(K) = [];
            w(K,:) = [];
            e(K) = [];
            delta(K) = [];
            line = K;
            break
        end
    end
    if(line>=0)
        Act(line) = [];
        c = c-1;
    end

    %----------------------------------------------------------------------
    
    %Adapt granules size
    if(counter==1)
        beta = c;       %No. of granules before
    end
    counter = counter + 1;
    if(counter == hr)
        chi = c;        %No. of granules after
        difference = chi - beta;
        %Change granules size
        if(difference>=eta)     
            Rho = (1+difference/counter)*Rho;       %Increasing size
        else                                          
            Rho = (1-2*difference/counter)*Rho;     %Decreasing size
        end
        counter = 1;
    end
    VecRho(h) = Rho;        %Granules size along iterations
                
    %Check if support and core conctraction is necessary
    for(i=1:c)
        for(j=1:n)
            if(((lambda(i,j)+Lambda(i,j))/2)-Rho/2>l(i,j))
                l(i,j) = ((lambda(i,j)+Lambda(i,j))/2)-Rho/2;
                if(((lambda(i,j)+Lambda(i,j))/2)-Rho/2>lambda(i,j))
                    lambda(i,j) = ((lambda(i,j)+Lambda(i,j))/2)-Rho/2;
                end
            end
            if(((lambda(i,j)+Lambda(i,j))/2)+Rho/2<L(i,j))
                L(i,j) = ((lambda(i,j)+Lambda(i,j))/2)+Rho/2;
                if(((lambda(i,j)+Lambda(i,j))/2)+Rho/2<Lambda(i,j))
                    Lambda(i,j) = ((lambda(i,j)+Lambda(i,j))/2)+Rho/2;
                end
            end
        end
    end
        
    %----------------------------------------------------------------------
    
    Acc(h) = sum(Right)/(sum(Right)+sum(Wrong))*100;
    
end     %End data stream

ElapsedTime = toc;

%--------------------------------------------------------------------------

%Display

TotalTime_TimePerSample_Frequency = [ElapsedTime ElapsedTime/h h/ElapsedTime]
ElapsedTime/h
AverageNoRules_DeviationNoRules = [sum(StoreNoRule)/h std(StoreNoRule)]
Accuracy = Acc(h)

figure(1)
    plot(1:h,Acc,'k','LineWidth',2), hold on
    xlabel('Time index'), ylabel('Accuracy (%)');
    axis([0 h+5 0 100]);
    
figure(2)
    plot(1:h,StoreNoRule,'k','LineWidth',2), hold on, grid off
    xlabel('Time index'),ylabel('Number of rules');
    axis([0 h+5 0 max(StoreNoRule)+2]);
    
figure(3)
    plot(1:h,VecRho,'k','LineWidth',2), hold on, grid off
    xlabel('Time index'),ylabel('Granularity');
    axis([0 h+5 0 max(VecRho)+2]);

%--------------------------------------------------------------------------


