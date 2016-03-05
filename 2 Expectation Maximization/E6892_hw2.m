clear;
close all
 
load mnist_mat
[row_Xtrain,column_Xtrain] = size(Xtrain);
[row_Xtest,column_Xtest] = size(Xtest);
number1 = sum(ytrain);
number0 = column_Xtrain - number1;
Sample1 = Xtrain(:,1+number0:column_Xtrain);
Sample0 = Xtrain(:,1:number0);
y1 = ytrain(:,1+number0:column_Xtrain);
y0 = ytrain(:,1:number0);

sigma = 1.5;
lambda = 1;
T = 100;
d = row_Xtrain;%rows of data

w = zeros(d,1);
W = zeros(d,T);
One = ones(number1,1);
I = eye(d);
x = zeros(d,d);
for i = 1:column_Xtrain
    a = Xtrain(:,i) * Xtrain(:,i)';%15*15
    x = x + a;
end
for t = 1:T
    s0 = -Sample0.' * w ./ sigma;%5842*1
    Eqt_yequals_0 = Sample0.' * w + sigma .* (-normpdf(s0)./normcdf(s0));%5842*1
    s1 = -Sample1.' * w ./ sigma;%5949*1
    Eqt_yequals_1 = Sample1.' * w + sigma .* (normpdf(s1))./(One-normcdf(s1));%5949*1
    
    Eqt_fi = [Eqt_yequals_0' Eqt_yequals_1'];
    xiEqt = Xtrain * (Eqt_fi.');
    w = pinv(lambda .* I + x ./ (sigma^2)) * (xiEqt./(sigma^2));%1*1
    W(:,t) = w;   
end

s = Xtest.' * w .* (1/sigma);%1991*1
yeq1 = log(normcdf(s));
yeq0 = log(1-normcdf(s));
lnpy_0 = d/2 * log(lambda/2/pi) - lambda * (w.' * w)/2 + yeq0;
p0 = exp(lnpy_0);

lnpy_1 = d/2 * log(lambda/2/pi) - lambda * (w.' * w)/2 + yeq1;
p1 = exp(lnpy_1);

predicProb0 = p0 ./(p0 + p1);
predicProb1 = p1 ./(p0 + p1);

%b
lnpy = zeros(1,T);
for t = 1:T;
    s_0 = Sample0.' * W(:,t) .* (1/sigma);
    s_1 = Sample1.' * W(:,t) .* (1/sigma);
    yeq1_2 = log(normcdf(s_1));
    yeq0_2 = log(1-normcdf(s_0));
    lnpy(t) = d/2 * log(lambda/2/pi) - lambda * (W(:,t).' * W(:,t))/2 + sum(yeq0_2) + sum(yeq1_2);
end
t = 1:1:T;
figure,plot(t,lnpy);

%c
x_wrong = zeros(1,column_Xtest);
p = zeros(1,column_Xtest);
confsMatrix = zeros(2,2);
number1iny = sum(ytest);
number0iny = column_Xtest - number1iny;
for i = 1:column_Xtest
    if (lnpy_0(i)>lnpy_1(i))
        p(i) = 0;
    else p(i) = 1;
    end
    if (i<number0iny+1)
        if (p(i) == ytest(i))
            confsMatrix(1,1) = confsMatrix(1,1) + 1;
        else
            confsMatrix(1,2) = confsMatrix(1,2) + 1;%4's classified as 9's
            x_wrong(i)= 1;
        end
    else
        if (p(i) == ytest(i))
            confsMatrix(2,2) = confsMatrix(2,2) + 1;
        else
            confsMatrix(2,1) = confsMatrix(2,1) + 1;%9's classified as 4's
            x_wrong(i)= 1;
        end
    end
end
rightNum = confsMatrix(1,1) + confsMatrix(2,2);
wrongNum = confsMatrix(1,2) + confsMatrix(2,1);
correctness = rightNum/column_Xtest;

%d, show three misclassifed images
misImg = zeros(15,3);
j = 1;
index_mis = zeros(1,3);%records the positions of the misclassified numbers in Xtest
for i = 1:column_Xtest
    if(x_wrong(i) == 1)
        misImg(:,j) = Xtest(:,i);
        index_mis(j) = i;
        j = j+1;
        if (j == 4)
            break;
        end
    end
end
figure;
for i = 1:3
    x2 = Q * misImg(:,i);
    x_show = reshape(x2,28,28);
    subplot(1,3,i),imshow(x_show);
    fprintf('The predictive probility equals 0 of the %d misclassified is = %d\n',i,predicProb0(index_mis(i)));
    fprintf('The predictive probility equals 1 of the %d misclassified is = %d\n',i,predicProb1(index_mis(i)));
end

%e, three most ambiguous predictions
prdicProb = abs(predicProb0 - predicProb1);
[ambPred, index] = sort(prdicProb,'ascend');

figure;
for i = 1:3
    x3 = Q * Xtest(:,index(i));
    x_show3 = reshape(x3,28,28);    
    subplot(1,3,i),imshow(x_show3);
    fprintf('The predictive probility equals 0 of the %d ambiguous number is = %d\n',i,predicProb0(index(i)));
    fprintf('The predictive probility equals 1 of the %d ambiguous number is = %d\n',i,predicProb1(index(i)));
end

%f
number_t = [1,5,10,25,50,100];

figure;
for i = 1:length(number_t)
    x4 = Q * W(:,number_t(i));
    x_show4 = reshape(x4,28,28);   
    subplot(2,3,i)
    imshow(x_show4);
    title(number_t(i));    
end