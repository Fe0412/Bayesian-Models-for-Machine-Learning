clear;
close all
 
load mnist_mat
[row_Xtrain,column_Xtrain] = size(Xtrain);
[row_Xtest,column_Xtest] = size(Xtest);
number1 = sum(ytrain);
number0 = column_Xtrain - number1;
Sample1 = Xtrain(:,1+number0:column_Xtrain);
Sample0 = Xtrain(:,1:number0);
%calculate the means and variances of x
u1=mean(Sample1')';
u0=mean(Sample0')';
sigm1=var(Sample1')';
sigm0=var(Sample0')'; 

%calculate the means and variances of t distribution
a = 1;
b = 1;
c = 1;
e = 1;
f = 1;
un1 = (number1*u1)/(a+number1);
un0 = (number0*u0)/(a+number0);
an1 = 1 + number1;
an0 = 0 + number0;
bn1 = b + number1/2;
bn0 = b + number0/2;
cn1 = c + number1*sigm1/2 + a*number1*(u1 .* u1)/(2*(a + number1));
cn0 = c + number0*sigm0/2 + a*number0*(u0 .* u0)/(2*(a + number0));
freedom1 = 2*bn1;
freedom0 = 2*bn0;
sigma1 = sqrt(cn1*(an1 + 1)/(bn1*an1));
sigma0 = sqrt(cn0*(an0 + 1)/(bn0*an0));

%normalization
X1minus = bsxfun(@minus,Xtest',un1')';
X0minus = bsxfun(@minus,Xtest',un0')';
X1 = bsxfun(@times,X1minus,(1./sigma1));
X0 = bsxfun(@times,X0minus,(1./sigma0));
psum1 = 1;
psum0 = 1;

%Posterior predictive
for i = 1:15
    p_x_yequal1 = tpdf(X1(i,:),freedom1);
    p_x_yequal0 = tpdf(X0(i,:),freedom0);
    psum1 = psum1 .* p_x_yequal1;
    psum0 = psum0 .* p_x_yequal0;
end
p_yequal1_y = (e + number1)/(column_Xtrain + e + f);
p_yequal0_y = (f + number0)/(column_Xtrain + e + f);
px1 = psum1 * p_yequal1_y;
px0 = psum0 * p_yequal0_y;
p = zeros(1,column_Xtest);
x_wrong = zeros(1,column_Xtest);

%QUESTION b, confusion matrix
confsMatrix = zeros(2,2);
number1iny = sum(ytest);
number0iny = column_Xtest - number1iny;
for i = 1:column_Xtest
    if (px0(i)>px1(i))
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
predicProb0 =  px0.*(1./(px0 + px1));
predicProb1 =  px1.*(1./(px0 + px1));
%a
correctness = rightNum/column_Xtest;

%c, show three misclassifed images
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

for i = 1:3
    x2 = Q * misImg(:,i);
    x_show = reshape(x2,28,28);
    subplot(2,3,i)
    imshow(x_show);
    fprintf('The predictive probility equals 0 of the %d misclassified is = %d\n',i,predicProb0(index_mis(i)));
    fprintf('The predictive probility equals 1 of the %d misclassified is = %d\n',i,predicProb1(index_mis(i)));
end

%d, three most ambiguous predictions
prdicProb = abs(predicProb0 - predicProb1);
[ambPred, index] = sort(prdicProb,'ascend');
for i = 1:3
    x3 = Q * Xtest(:,index(i));
    x_show3 = reshape(x3,28,28);
    subplot(2,3,i+3)
    imshow(x_show3);
    fprintf('The predictive probility equals 0 of the %d ambiguous number is = %d\n',i,predicProb0(index(i)));
    fprintf('The predictive probility equals 1 of the %d ambiguous number is = %d\n',i,predicProb1(index(i)));
end