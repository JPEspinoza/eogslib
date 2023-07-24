%Evolving Optimal Granular System (eOGS) - adapted for classification (2 classes)
%By: Daniel Leite
%V3. December, 18th, 2018

%Optimal Rule-based Granular Systems from Data Streams
%IEEE Transactions on Fuzzy Systems, 2019
%DOI: 10.1109/TFUZZ.2019.2911493 

clear all
close all
clc

%--------------------------------------------------------------------------
%Manual Setting of the Parameters

aut = 1;                %Manual mode. Set aut = 1 for automatic

alpha = 0.1;            %Static Alpha-cut

S = 1/(2*pi);           %Stigler or maximal dispersion
% S = 0.01;               %Minimal approach for Sigma
OV = 2;                 %Opening the variance in Beta: default = 2;

V = 1000;                %Legth of the time-window: useful for parameter adaptation and deleting

VV = 1;                 %Useful for merging
MDTM = 0.01;             %Minimal distance to merge

TW1 = 1;                %Smoothness level of the numerical prediction
TW2 = 1;                %Smoothness level of the bounds

SmoothJump = 0.1;       %Univariate time series or smooth output expected
% SmoothJump = 0.99;       %Multivariate time series or unordered samples

%--------------------------------------------------------------------------
%Block Sigma and alpha to avoid numerical problem
if alpha <= 0       alpha = 0.001;   end
if alpha >= 1       alpha = 0.999;   end
if S <= 0.0001      S = 0.0001;      end
if S > 1/(2*pi)     S = 1/(2*pi);    end

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%Automatic Setting of the Parameters

Priority = 1;           %Priority objective (1), (2)

MaxSingError = 0.1;    %(1) Maximum estimation error (RMSE) - Range: [0,1]
MaxGranError = 1;     %(2) Maximum estimation error (GEI) - Range: [0,1]
MinSp = 0.6;            %(3) Minimum specificity - Range: [0,1]
MaxNoRules = 8;        %(4) Maximum number of rules - Range: [0,1]

SIni = S; MDTMIni = MDTM; OVIni = OV;
    
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%Dataset

Data = xlsread('Dados_EyeState');

%X = Data;                   %Non-normalized inputs and output
X = [Data(:,1:15)]; %Non-normalized inputs and output

H = size(X,1);              %Number of samples
n = size(X,2)-1;            %Number of attributes
m = 1;

%Normalizing all attributes in [0,1]
for j=1:n+1
    X(:,j) = (X(:,j)-min(X(:,j)))./(max(X(:,j))-min(X(:,j)));
end

Dataset = X;
clear h j x X
MSE(1) = 0;
GEI(1) = 0;
count = 0;
SamplesInGranule = zeros(1,V+2);
warning off

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%Loop (data stream)
tic
for h = 1:size(Dataset,1)
    
    %Given the h-th input sample of the dataset
    x = Dataset(h,1:size(Dataset,2)-1);
    y = Dataset(h,size(Dataset,2));
    
    if h == 1 %First sample of the data stream
        
        % Create first granule
        c = 1;                                %Number of rules
        Mu(1,:) = x;                          %Central point
        Sigma(1,1:n) = S;                     %Dispersion
        l(1,:) = x - sqrt(-2*Sigma(1,:).*log(alpha*(sqrt(2*pi*Sigma(1,:)))));   %Lower bounds
        L(1,:) = x + sqrt(-2*Sigma(1,:).*log(alpha*(sqrt(2*pi*Sigma(1,:)))));   %Upper bounds
        Sp(1) = 1 - ((2*sqrt(2))/n) * (sum(sqrt(-Sigma(1,:).*log(alpha*sqrt(2*pi*Sigma(1,:))))));  %Specificity
        if Sp(1) < 0 Sp(1) = 0; end 
        if Sp(1) > 1 Sp(1) = 1; end
        
        %Output y becomes available
        Mu = [Mu(1,:) y];                   %Central point
        Sigma = [Sigma(1,:) S];             %Dispersion
        u(1,:) = y - sqrt(-2*Sigma(1,n+1:end).*log(alpha*(sqrt(2*pi*Sigma(1,n+1:end)))));   %Lower bounds
        U(1,:) = y + sqrt(-2*Sigma(1,n+1:end).*log(alpha*(sqrt(2*pi*Sigma(1,n+1:end)))));   %Upper bounds
        A(1,:) = zeros(1,n+1);              %Coefficients of affine function
        A(1,1) = y;                         %Coefficients of affine function
        v(1) = 1;                           %Number of times that granule 1 was achieved in V iterations
        SamplesInGranule(1,1) = h;          %Samples in Granule considering V previous iterations
        
        % A crystal ball prediction 
        yest(h) = 0.5;  
        uest(h) = 0.4;
        Uest(h) = 0.6;
        
    else %Samples h = 2, ...
        
        %Check if sample x is within any existing granule
        I = [];     
        for i = 1:c
            for j = 1:n
                if x(j) <= l(i,j) || x(j) >= L(i,j)
                    break
                else
                    if j ~= n
                    else
                        I = [I i];      %Granules that fit x
                    end
                end
            end
        end
            
        %3 possible cases:
        % (1) No granule fits x
        % (2) One granule fits x
        % (3) More than one granule fits x
        
        if isempty(I) == 1      %(1) No granule fits x
            Flag = 0;
            %Which is closest granule?
            %Distance between x and Mu
            Dist = []; PP = [];
            for i = 1:c
                PP(i) = sum(A(i,2:end) ~= 0);  %Punish premature granules: coefficients in A are not developed
                Dist(i) = exp(2*PP(i)) * norm(x-Mu(i,1:n));
            end
            [trash,Winner] = min(Dist);

        elseif length(I) == 1   %(2) I see a winner
            Flag = 1;
            Winner = I;
                
        else                    %(3) Conflict, we should decide about a winner granule
            Flag = 1;
            
            %Which is closest granule?
            %Distance between x and Mu
            Dist = []; PP = [];
            for i = 1:c
                PP(i) = sum(A(i,2:end) ~= 0);  %Punish premature granules: coefficients in A are not developed
                Dist(i) = exp(2*PP(i)) * norm(x-Mu(i,1:n));
            end
            [trash,Winner] = min(Dist);
            
%             %Decision based on the maximum specificity
%             PP = [];
%             for i=1:c
%                 if i ~= I
%                     SpAux(i) = 0;
%                 else
%                     PP(i) = sum(A(i,2:end) ~= 0);  %Punish premature granules: coefficients in A are not developed
%                     SpAux(i) = exp(PP(i)) * Sp(i);
%                 end
%             end
%             [trash,Winner] = max(SpAux);    %A winner emerging from the shadows

        end
        
        clear trash SpAux
        
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        %Prediction
        
        %------------------------------------------------------------------
        %Numerical prediction
        %------------------------------------------------------------------
        %1 - The winner takes all approach
%         yest(h) = A(Winner,1) + sum( A(Winner,2:end).*x );
        %------------------------------------------------------------------

        %2 - Smooth approach considering active granules I
        Distance = [];
        yest(h) = 0;
        if isempty(I) == 1 || length(I) == 1
            yest(h) = A(Winner,1) + sum( A(Winner,2:end).*x );
        else            
            for i=1:size(I,2)
                Distance(i) = norm(x-Mu(I(i),1:n));
            end
            Distance = Distance/sum(Distance);
            for i=1:size(I,2)
                if Distance(i) < 0.0001
                    Distance(i) = 0.0001;
                end
            end
            Similarity = 1./Distance;
            Similarity = Similarity/sum(Similarity);
        
            %Punish premature granules: coefficients in A are not developed
            Total = 0; PP = [];
            for i=1:size(I,2)
                PP(i) = sum(A(I(i),2:end) ~= 0);
            end
            Similarity = exp(2*PP).*Similarity;
            Similarity = Similarity/sum(Similarity);
           
            for i=1:size(I,2)
                yest(h) = yest(h) + Similarity(i) * (A(I(i),1) + sum( A(I(i),2:end).*x ));
            end
        end
        
        %Quasi-linear projection in case of jumps
        Dif = [];
        if h > TW1
            if yest(h) > yest(h-1)+SmoothJump || yest(h) < yest(h-1)-SmoothJump || yest(h) > 1 || yest(h) < 0
                for i = 1:TW1-1
                    Dif(i) = (Dataset(h-i,size(Dataset,2)) - Dataset(h-i-1,size(Dataset,2)));
                end
                if isempty(Dif)
                    Dif = 0;
                end
                yest(h) = Dataset(h-1,size(Dataset,2)) + mean(Dif);
                if yest(h) > 1 
                    yest(h) = 1;
                elseif yest(h) < 0
                    yest(h) = 0;
                end
            end
        end
 
        %------------------------------------------------------------------
        %Granular prediction
        %------------------------------------------------------------------
        %1 - The winner takes all approach
        %Inclusion function   
%         for j = 1:n
%             if A(Winner,j) >= 0
%                 AuxMin(j) = l(Winner,j);
%                 AuxMax(j) = L(Winner,j);
%             else
%                 AuxMin(j) = L(Winner,j);
%                 AuxMax(j) = l(Winner,j);
%             end
%         end
%         uIncFunc = A(Winner,1) + sum( A(Winner,2:end).*AuxMin );
%         UIncFunc = A(Winner,1) + sum( A(Winner,2:end).*AuxMax );
%         %Consequent aggregation AND
%         if A(Winner,2:end) ~= 0
%             uest(h) = max(u(Winner),uIncFunc);
%             Uest(h) = min(U(Winner),UIncFunc);
%         else
%             uest(h) = u(Winner);
%             Uest(h) = U(Winner);
%         end
        %------------------------------------------------------------------
        %2 - Smooth approach considering active granules I

        if isempty(I) == 1 || length(I) == 1
            
            for j = 1:n
                if A(Winner,j+1) >= 0
                    AuxMin(j) = l(Winner,j);
                    AuxMax(j) = L(Winner,j);
                else
                    AuxMin(j) = L(Winner,j);
                    AuxMax(j) = l(Winner,j);
                end
            end
            uIncFunc = []; UIncFunc = [];
            uIncFunc = A(Winner,1) + sum( A(Winner,2:end).*AuxMin );
            UIncFunc = A(Winner,1) + sum( A(Winner,2:end).*AuxMax );
            %Consequent aggregation AND
            if A(Winner,2:end) ~= 0
                uest(h) = max(u(Winner),uIncFunc);
                Uest(h) = min(U(Winner),UIncFunc);
            else
                uest(h) = u(Winner);
                Uest(h) = U(Winner);
            end
        
        else

            %Inclusion function
            aux3 = []; aux4 = [];
            for k = 1:size(I,2)
                for j = 1:n
                    if A(I(k),j+1) >= 0
                        AuxMin(j) = l(I(k),j);
                        AuxMax(j) = L(I(k),j);
                    else
                        AuxMin(j) = L(I(k),j);
                        AuxMax(j) = l(I(k),j);
                    end
                end
                aux3(k) = A(I(k),1) + sum( A(I(k),2:end).*AuxMin );
                aux4(k) = A(I(k),1) + sum( A(I(k),2:end).*AuxMax );
            end
            uIncFunc = 0; UIncFunc = 0;
            for k = 1:size(I,2)
                uIncFunc = uIncFunc + Similarity(k)*aux3(k);
                UIncFunc = UIncFunc + Similarity(k)*aux4(k);
            end
            
            aux5 = 0; aux6 = 0;
            for k = 1:size(I,2)
                aux5 = aux5 + Similarity(k) * u(I(k));
                aux6 = aux6 + Similarity(k) * U(I(k));
            end
            
            %Consequent aggregation AND
            for k = 1:size(I,2)
                if A(I(k),2:end) ~= 0
                    uest(h) = max(aux5,uIncFunc);
                    Uest(h) = min(aux6,UIncFunc);
                else
                    uest(h) = aux5;
                    Uest(h) = aux6;
                end
            end
            
        end

        %Quasi-linear projection of the inferior bound in case of jumps
        Dif = [];
        if h > TW2
            if uest(h) > uest(h-1)+SmoothJump || uest(h) < uest(h-1)-SmoothJump
                for i = 1:TW2-1
%                     Dif(i) = (Dataset(h-i,size(Dataset,2)) - uest(h-i));
                    Dif(i) = (Uest(h-i) - uest(h-i));
                end
                if isempty(Dif)
                    Dif(1) = (Uest(h-1) - uest(h-1));
                end
                uest(h) = Dataset(h-1,size(Dataset,2)) - mean(Dif)/2;
                if uest(h) > 1 
                    uest(h) = 1;
                elseif uest(h) < 0
                    uest(h) = 0;
                end
                if uest(h) >= Uest(h)
                    uest(h) = Uest(h) - SmoothJump;
                end
            end
        end
        
        %Quasi-linear projection of the superior bound in case of jumps
        Dif = [];
        if h > TW2
            if Uest(h) > Uest(h-1)+SmoothJump || Uest(h) < Uest(h-1)-SmoothJump
                for i = 1:TW2-1
%                     Dif(i) = (Dataset(h-i,size(Dataset,2)) + Uest(h-i));
                    Dif(i) = (Uest(h-i) - uest(h-i));
                end
                if isempty(Dif)
                    Dif(1) = (Uest(h-1) - uest(h-1));
                end
                Uest(h) = Dataset(h-1,size(Dataset,2)) + mean(Dif)/2;
                if Uest(h) > 1 
                    Uest(h) = 1;
                elseif Uest(h) < 0
                    Uest(h) = 0;
                end
                if Uest(h) <= uest(h)
                    Uest(h) = uest(h) + SmoothJump;
                end
            end
        end

        %------------------------------------------------------------------
        %Smoothness level of the numerical prediction
        Dif = [];
        if h>TW1
            for i = 1:TW1-1
                Dif(i) = (Dataset(h-i,size(Dataset,2)) - Dataset(h-i-1,size(Dataset,2)));
            end
            if isempty(Dif)
                Dif = 0;
            end
            yest(h) = (1/TW1)*(yest(h) + (TW1-1)*(Dataset(h-1,size(Dataset,2)) + mean(Dif)));
        end

        %Smoothness level of the prediction bounds
        Difu = []; DifU = [];
        if h>TW2
            for i = 1:TW2-1
                Difu(i) = uest(h-i) - uest(h-i-1);
                DifU(i) = Uest(h-i) - Uest(h-i-1);
            end
            if isempty(Difu)
                Difu = 0;
            end
            if isempty(DifU)
                DifU = 0;
            end
            uest(h) = (1/TW2)*(uest(h) + (TW2-1)*(uest(h-1) + mean(Difu)));
            Uest(h) = (1/TW2)*(Uest(h) + (TW2-1)*(Uest(h-1) + mean(DifU)));
        end

        %Guarantee that premature polynomials do not give numerical
            %prediction of the bounds
        if yest(h) < uest(h)
            uest(h) = yest(h)-0.05;
        elseif yest(h) > Uest(h)
            Uest(h) = yest(h)+0.05;
        end
        
        if uest(h) >= Uest(h)
            uest(h) = Uest(h)-0.05;
        end
        if Uest(h) <= uest(h)
            Uest(h) = uest(h)+0.05;
        end
        
        %These constraints are useful if the time series does not represent
        %a smooth dynamic system, i.e., the samples are temporaly uncorrelated
        if yest(h) > 1 
            yest(h) = 1;
        elseif yest(h) < 0
            yest(h) = 0;
        end
        if uest(h) > 1 
            uest(h) = 1;
        elseif uest(h) < 0
            uest(h) = 0;
        end
        if Uest(h) > 1 
            Uest(h) = 1;
        elseif Uest(h) < 0
            Uest(h) = 0;
        end
           
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        
        %Finally, y becomes available
        %Supervised adaptation of the winner rule (or) structural expansion
        if Flag == 0
            %Create granule
            c = c + 1;                          %Number of rules
            Mu(c,1:n) = x;                      %Central point
            Mu(c,n+1:n+m) = y;                      
            Sigma(c,1:n+m) = S;                 %Dispersion
            l(c,:) = x - sqrt(-2*Sigma(c,1:n).*log(alpha*(sqrt(2*pi*Sigma(c,1:n)))));   %Lower bounds
            L(c,:) = x + sqrt(-2*Sigma(c,1:n).*log(alpha*(sqrt(2*pi*Sigma(c,1:n)))));   %Upper bounds
            Sp(c) = 1 - ((2*sqrt(2))/n) * (sum(sqrt(-Sigma(c,1:n).*log(alpha*sqrt(2*pi*Sigma(c,1:n))))));  %Specificity
            if Sp(c) < 0 Sp(c) = 0; end 
            if Sp(c) > 1 Sp(c) = 1; end
            u(c,:) = y - sqrt(-2*Sigma(c,n+1:end).*log(alpha*(sqrt(2*pi*Sigma(c,n+1:end)))));  %Lower bounds
            U(c,:) = y + sqrt(-2*Sigma(c,n+1:end).*log(alpha*(sqrt(2*pi*Sigma(c,n+1:end)))));   %Upper bounds
            A(c,:) = zeros(1,n+1);              %Coefficients of affine function
            A(c,1) = y;                         %Coefficients of affine function
            v(c) = 1;                           %Number of times that granule c was achieved in V iterations
            SamplesInGranule(c,1) = h;          %Samples in Granule considering V previous iterations
            if h>V
                [row,trash1,trash2] = find(SamplesInGranule==h-V);
                SamplesInGranule(row,:) = [SamplesInGranule(row,2:end) zeros(size(row,1),1)];
                v(row) = v(row)-1;
            end
            
        else
           
            %Adapt the winner
               %Adapt the variance
            beta = (v(Winner)*(Mu(Winner,:)-[l(Winner,:) u(Winner,:)]) + OV*abs(Mu(Winner,:)-[x y]))./((v(Winner)+1)*(Mu(Winner,:)-[l(Winner,:) u(Winner,:)]));
            Sigma(Winner,:) = beta.*Sigma(Winner,:);
            %--------------------------------------------------------------------------
            %Constrain Sigma to avoid numerical problem
            for j = 1:size(Sigma,2)
                if Sigma(Winner,j) <= 0.0001
                    Sigma(Winner,j) = 0.0001;
                end
                if Sigma(Winner,j) > 1/(2*pi)
                    Sigma(Winner,j) = 1/(2*pi);
                end
            end
            %--------------------------------------------------------------------------
            %Adapt the central point
            Mu(Winner,:) = (v(Winner)*Mu(Winner,:)+[x y])/(v(Winner)+1);
            
            %Adapt the counter of activations of the winner granule
            v(Winner) = v(Winner)+1;
            
            %Adapt the time window V
            aux = []; aux = find(SamplesInGranule(Winner,:)==0);
            SamplesInGranule(Winner,aux(1)) = h; aux = [];
                       
            if h>V          
                [row,trash1,trash2] = find(SamplesInGranule==h-V);
                SamplesInGranule(row,:) = [SamplesInGranule(row,2:end) zeros(size(row,1),1)];
                v(row) = v(row)-1;
            end

            %Adapt the bounds
            aux1 = Mu(Winner,:) - sqrt(-2*Sigma(Winner,:).*log(alpha*(sqrt(2*pi*Sigma(Winner,:)))));
            l(Winner,:) = aux1(1:n);
            u(Winner,:) = aux1(n+1:end);
            aux2 = Mu(Winner,:) + sqrt(-2*Sigma(Winner,:).*log(alpha*(sqrt(2*pi*Sigma(Winner,:)))));
            L(Winner,:) = aux2(1:n);
            U(Winner,:) = aux2(n+1:end);
            clear aux1 aux2
            
            %Recalculate the specificity
            Sp(Winner) = 1 - ((2*sqrt(2))/n) * (sum(sqrt(-Sigma(Winner,1:n).*log(alpha*sqrt(2*pi*Sigma(Winner,1:n))))));
            if Sp(Winner) < 0 Sp(Winner) = 0; end 
            if Sp(Winner) > 1 Sp(Winner) = 1; end
            
            %Recursive Least Squares - for linear consequent functions
            [trash,Pos] = find(SamplesInGranule(Winner,:) == 0,1,'first');
            RelevantData = Dataset(SamplesInGranule(Winner,1:Pos-1),:);
            YY = RelevantData(:,end);
            XX = [ones(Pos-1,1) RelevantData(:,1:end-1)];
            Aaux = []; Aaux = A(Winner,:); %Protection in case of numerical problem (quadratic XX with incomplete rank)
            A(Winner,:) = (XX\YY)';
            NrNaN = sum(isnan(A(Winner,:)));
            if NrNaN > 0
                A(Winner,:) = Aaux;
            end
            
            %Sliding - for zero-order consequent functions
%             A(Winner,1) = Mu(Winner,end);

            %--------------------------------------------------------------
            %Adaptation of the remaining active granules
            if length(I)>1
                for i = 1:size(I,2)
                    if I(i) ~= Winner
                        %Adapt granule I(i)
                        %Adapt the variance
                        beta = (v(I(i))*(Mu(I(i),:)-[l(I(i),:) u(I(i),:)]) + OV*abs(Mu(I(i),:)-[x y]))./((v(I(i))+1)*(Mu(I(i),:)-[l(I(i),:) u(I(i),:)]));
                        Sigma(I(i),:) = beta.*Sigma(I(i),:);
                        %--------------------------------------------------------------------------
                        %Constrain Sigma to avoid numerical problem
                        for j = 1:size(Sigma,2)
                            if Sigma(I(i),j) <= 0.0001
                                Sigma(I(i),j) = 0.0001;
                            end
                            if Sigma(I(i),j) > 1/(2*pi)
                                Sigma(I(i),j) = 1/(2*pi);
                            end
                        end
                        %--------------------------------------------------------------------------
                        %Adapt the central point
                        Mu(I(i),:) = (v(I(i))*Mu(I(i),:)+[x y])/(v(I(i))+1);

                        %Adapt the counter of activations of the winner granule
                        v(I(i)) = v(I(i))+1;
                        %Adapt the time window V
                        aux = []; aux = find(SamplesInGranule(I(i),:)==0);
                        SamplesInGranule(I(i),aux(1)) = h; aux = [];

                        %Adapt the bounds
                        aux1 = Mu(I(i),:) - sqrt(-2*Sigma(I(i),:).*log(alpha*(sqrt(2*pi*Sigma(I(i),:)))));
                        l(I(i),:) = aux1(1:n);
                        u(I(i),:) = aux1(n+1:end);
                        aux2 = Mu(I(i),:) + sqrt(-2*Sigma(I(i),:).*log(alpha*(sqrt(2*pi*Sigma(I(i),:)))));
                        L(I(i),:) = aux2(1:n);
                        U(I(i),:) = aux2(n+1:end);
                        clear aux1 aux2

                        %Recalculate the specificity
                        Sp(I(i)) = 1 - ((2*sqrt(2))/n) * (sum(sqrt(-Sigma(I(i),1:n).*log(alpha*sqrt(2*pi*Sigma(I(i),1:n))))));
                        if Sp(I(i)) < 0 Sp(I(i)) = 0; end 
                        if Sp(I(i)) > 1 Sp(I(i)) = 1; end
                        
                        %Recursive Least Squares - for linear consequent functions
                        [trash,Pos] = find(SamplesInGranule(I(i),:) == 0,1,'first');
                        RelevantData = Dataset(SamplesInGranule(I(i),1:Pos-1),:);
%                         RelevantData = [RelevantData; [x y]];
                        YY = RelevantData(:,end);
                        XX = [ones(Pos-1,1) RelevantData(:,1:end-1)];
                        Aaux = []; Aaux = A(I(i),:);
                        A(I(i),:) = (XX\YY)';
                        NrNaN = sum(isnan(A(I(i),:)));
                        if NrNaN > 0
                            A(I(i),:) = Aaux;
                        end
                    end
                end 
            end
            %--------------------------------------------------------------
        end
        %------------------------------------------------------------------
    end
    
    %Delete granules
    aux = []; aux = 1; cont = [];
    for j=1:c
        if aux == 1
            cont = 0;
            for i=1:c
                if SamplesInGranule(i,1)==0
                    c = c - 1;                       
                    Mu(i,:) = [];                      
                    Sigma(i,:) = [];
                    l(i,:) = [];
                    L(i,:) = [];
                    Sp(i) = [];
                    u(i,:) = [];
                    U(i,:) = [];
                    A(i,:) = [];
                    v(i) = [];
                    SamplesInGranule(i,:) = [];
                    break
                else
                    cont = cont + 1;
                end
            end
            if cont == c
                aux = 0;
            end
        end
    end
    
    %----------------------------------------------------------------------
    %Merge granules
    
    count = count + 1;
    if count == round(VV)
    
        %Calculate the minimum distance between midpoints Mu
        minimal = 9999999; I1 = 0; I2 = 0;
        for i1=1:c
            for i2=i1+1:c
                aux = norm(Mu(i1,:)-Mu(i2,:))/n;
                if aux>0 & aux<minimal
                    minimal = aux;
                    I1 = i1;                %Granules I1 and I2 are the closest
                    I2 = i2;
                end
            end
        end
        
        if I1 ~= 0 && I2 ~= 0       %There are candidates to be merged
            
            if minimal < MDTM       %Unique criterion for merging
            
%             if A(I1,2:end) ~= 0                         %Granules should have some maturity
%                 if A(I2,2:end) ~= 0                     %Granules should have some maturity

                    %Check if inclinations are similar: < |10 degrees|
%                     theta1 = atan(A(I1,2:end))*180/pi;
%                     theta2 = atan(A(I2,2:end))*180/pi;
                    %if abs(theta1 - theta2) < 10        %Good to merge for a positive

                        %Create a pseudo-granule
%                         MU = (Mu(I1,:)+Mu(I2,:))/2;               %Average of means
                        MU = (v(I1)*Mu(I1,:) + v(I2)*Mu(I2,:))/(v(I1)+v(I2)); %Weighted average of means
                        
                        SIGMA = max(Sigma(I1,:),Sigma(I2,:));     %Maximum variations
%                         SIGMA = (Sigma(I1,:)+Sigma(I2,:))/2;      %Average variations
%                         SIGMA = min(Sigma(I1,:),Sigma(I2,:));     %Minimum variations
%                         SIGMA = (v(I1)*Sigma(I1,:)+v(I2)*Sigma(I2,:))/(v(I1)+v(I2));  %Weighted average of variations
                        
                        ll = MU(1,1:n) - sqrt(-2*SIGMA(1,1:n).*log(alpha*(sqrt(2*pi*SIGMA(1,1:n)))));          %Lower bounds
                        LL = MU(1,1:n) + sqrt(-2*SIGMA(1,1:n).*log(alpha*(sqrt(2*pi*SIGMA(1,1:n)))));          %Upper bounds
                        SP = 1 - ((2*sqrt(2))/n) * (sum(sqrt(-SIGMA(1,1:n).*log(alpha*sqrt(2*pi*SIGMA(1,1:n))))));   %Specificity
                        if SP < 0 SP = 0; end 
                        if SP > 1 SP = 1; end

                        uu = MU(1,n+1:end) - sqrt(-2*SIGMA(1,n+1:end).*log(alpha*(sqrt(2*pi*SIGMA(1,n+1:end)))));  %Lower bounds
                        UU = MU(1,n+1:end) + sqrt(-2*SIGMA(1,n+1:end).*log(alpha*(sqrt(2*pi*SIGMA(1,n+1:end)))));  %Upper bounds
                        
                        AA = (A(I1,:)+A(I2,:))/2;                              %Average of conseq coef.
%                         AA = (v(I1)*A(I1,:)+v(I2)*A(I2,:))/(v(I1)+v(I2));      % Weighted average of conseq coef.

                        %Replace granule I1
                        Mu(I1,:) = MU;
                        Sigma(I1,:) = SIGMA;
                        l(I1,:) = ll;
                        L(I1,:) = LL;
                        Sp(I1) = SP;
                        u(I1,:) = uu;
                        U(I1,:) = UU;
                        A(I1,:) = AA;
                        [trash,Pos1] = find(SamplesInGranule(I1,:) == 0,1,'first');
                        [trash,Pos2] = find(SamplesInGranule(I2,:) == 0,1,'first');
                        SamplesInGranule(I1,Pos1:Pos1+Pos2-2) = SamplesInGranule(I2,1:Pos2-1);
                        SamplesInGranule(I1,1:Pos1+Pos2-2) = sort(SamplesInGranule(I1,1:Pos1+Pos2-2));

                        aux = []; aux = SamplesInGranule(I1,1:Pos1+Pos2-2);
                        for i = 2:size(aux,2)
                            if ~isempty(aux(i))
                                if aux(i) == aux(i-1);
                                    aux(size(aux,2)+1) = 0;
                                    aux(i) = [];
                                end
                            end
                        end
                        
                        SamplesInGranule(I1,1:Pos1+Pos2-2) = aux;          
                        v(I1) = find(SamplesInGranule(I1,:) == 0,1,'first')-1;

                        %Delete granule I2
                        c = c - 1;                       
                        Mu(I2,:) = [];                      
                        Sigma(I2,:) = [];
                        l(I2,:) = [];
                        L(I2,:) = [];
                        Sp(I2) = [];
                        u(I2,:) = [];
                        U(I2,:) = [];
                        A(I2,:) = [];
                        v(I2) = [];
                        SamplesInGranule(I2,:) = [];
                    %end
%                 end
%             end

            end
   
        end
        
        count = 0;
    
    end
    
    %----------------------------------------------------------------------
    %----------------------------------------------------------------------
    %Automatic mode
    
    if aut == 1
        
        %Constraint on the maximum number of rules
        if c > MaxNoRules
            alpha = alpha - 0.01;
            MDTM = MDTM + 0.001;
            if alpha <= 0.001
                alpha = 0.001;   
            end
            if MDTM >= 0.03
                MDTM = 0.03;
            end
        else
            MDTM = MDTMIni;
        end
        
        %Constraint on the minimum specificity level
        if mean(Sp) < MinSp
            alpha = alpha + 0.01;
            OV = OV - 0.002;
            if alpha >= 0.99
                alpha = 0.99;   
            end
            if OV <= 1.5
                OV = 1.5;
            end
        else
            OV = OVIni;
        end
        
        %Constraint on the granular or numerical prediction error
        if Priority == 1
            %Constrain on the granular error
            if GEI(h) > MaxGranError
                alpha = alpha + 0.01;
                S = S - 0.004;
                if alpha >= 0.99
                    alpha = 0.99;   
                end
                if S <= 0.01
                    S = 0.01;
                end
            else
                S = SIni;
                alpha = alpha - 0.001;
                if alpha <= 0.001
                    alpha = 0.001;   
                end
            end
        else
            %Constrain on the numerical error
            if sqrt(MSE(h)) > MaxSingError
                alpha = alpha - 0.01;
                S = S + 0.004;
                if alpha <= 0.001
                    alpha = 0.001;   
                end
                if S >= 1/(2*pi)
                    S = 1/(2*pi);
                end
            else
                S = SIni;
            end
        end
     
    end
 
    %----------------------------------------------------------------------
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------    
    %Vectors for plot
    Rules(h) = c;                       %Number of rules
    MSE(h+1) = (h/(h+1))*MSE(h) + (1/(h+1))*(yest(h)-y)^2;
    RMSE(h+1) = sqrt(MSE(h+1));         %Root mean square error
    ISD(h) = (yest(h)-y)^2;             %Instantaneous square deviation
    Spec(h) = mean(Sp);
    Alpha(h) = alpha;
    
    %Granular Error Index: GEI
    if y > uest(h) && y < Uest(h)
        WITHIN = 1;
    else
        WITHIN = 0;
    end
    GEI(h+1) = (h/(h+1))*GEI(h) + (1/(h+1))*(1-WITHIN*(1-(Uest(h)-uest(h))));
    
end

%--------------------------------------------------------------------------
%Display
%Error indices
Root_Mean_Square_Error = RMSE(end)
Granular_Error_Index = GEI(end)
%Average specificity
Avg_Spec = mean(Spec)
%Total time
Elapsed_time_seconds = toc
%Total number of rules
Number_of_rules = c

%--------------------------------------------------------------------------
%Plots

figure('units','normalized','outerposition',[0 0.1 1 0.9])

subplot(3,2,1)
plot(Dataset(:,n+1),'k'), hold on
plot(yest,'r')
plot(uest,'b')
plot(Uest,'b')
axis([0 size(Dataset,1) -0.2 1.2])
title('eOGS: numerical and granular prediction')
legend('Actual data','Numerical estimation','Granular estimation')
xlabel('Time index')
ylabel('Amplitude')

subplot(3,2,2)
plot(Rules,'k')
axis([0 size(Dataset,1) -0.1 max(Rules)+1])
title('eOGS: evolution of the rule base')
legend('# of rules')
xlabel('Time index')
ylabel('# of rules')

subplot(3,2,3)
plot(Alpha,'k')
axis([0 size(Dataset,1) min(Alpha)-0.005 max(Alpha)+0.005])
title('eOGS: alpha cut')
legend('Alpha-cut')
xlabel('Time index')
ylabel('Alpha level')

subplot(3,2,4)
plot(Spec,'k')
axis([0 size(Dataset,1) min(Spec)-0.005 max(Spec)+0.005])
title('eOGS: average instantaneous specificity')
legend('Avg specificity')
xlabel('Time index')
ylabel('Specificity')

subplot(3,2,5)
plot(ISD,'k')
axis([0 size(Dataset,1) -0.005 1.005])
title('eOGS: instantaneous square deviation (ISD)')
legend('ISD')
xlabel('Time index')
ylabel('ISD')

subplot(3,2,6)
plot(RMSE,'k'), hold on
plot(GEI,'r')
axis([0 size(Dataset,1) -0.005 1.005])
title('eOGS: RMS error and Granular error')
legend('RMSE','MGE')
xlabel('Time index')
ylabel('RMSE')

%--------------------------------------------------------------------------
figure(2)
plot(Dataset(:,n+1),'k'), hold on
plot(yest,'r')
% plot(uest,'b')
% plot(Uest,'b')
axis([0 size(Dataset,1) -0.2 1.2])
title('eOGS: numerical prediction')
legend('Actual data','Numerical estimation')
xlabel('Time index')
ylabel('Amplitude')

% % figure(3)
% % plot(Dataset(:,n+1),'k'), hold on
% % % plot(yest,'r')
% % plot(uest,'b')
% % plot(Uest,'b')
% % axis([0 size(Dataset,1) -0.2 1.2])
% % title('eOGS: paving the path')
% % legend('Actual data','Granular estimation')
% % xlabel('Time index')
% % ylabel('Amplitude')

figure(4)
x = 0:0.0001:1; 
for i=1:c
    y = gaussmf(x,[Sigma(i,n+1) Mu(i,n+1)]); 
    plot(x,y), hold on
end
title('eOGS: output Gaussians')
xlabel('Y')
ylabel('Membership degree')

%----------------------------

% Classification accuracy
Est = (round(yest))'; Right = 0; Error = 0;
for i = 1:size(Dataset,1)
    if Dataset(i,n+1) - Est(i) == 0
        Right = Right + 1;
    else
        Error = Error + 1;
    end
end
Classification_accuracy = (Right/(Right+Error))*100
