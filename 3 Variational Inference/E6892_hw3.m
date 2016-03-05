clear;
close all
 
load data3.mat
%a
[N, d] = size(X);
t = 500;
mu = zeros(d, 1);
sigma = zeros(d, d);
a = zeros(d, 1);
b = zeros(d, 1);
akt = zeros(d, 1);
b0t = zeros(d, 1);
L = zeros(t, 1);

a0 = 10^(-16);
b0 = 10^(-16);
for i = 1:d
    b0t(i) = b0;
    a(i) = a0;
    b(i) = b0;
end
e0 = 1;
e = e0;
f0 = 1;
f = f0;

xx = zeros(d, d);
xy = zeros(d, 1);
for i = 1:N
    xx = xx + X(i,:)'*X(i,:);%X(i,:)':d*1
    xy = xy + X(i,:)'*y(i);
end

Eqalpha = zeros(d,1);
for i = 1:t
che2 = a./b;
    sigma = pinv(diag(a./b) + e/f*xx);
    mu = sigma*(e/f)*xy;
    e = e0 + N/2;
    
    yxmu = 0;
    for j = 1:N
        yxmu = yxmu + (y(j) - X(j,:)*mu)^2 + X(j,:)*sigma*X(j,:)';
    end
    
    f = f0 + 0.5*yxmu;
    mumuandsigma = mu*mu' + sigma;
    Elnpw = 0;
    Elnpa = 0;
    Elnqa = 0; 
    for m = 1:d
        a(m) = a0+0.5;
    end
    for k = 1:d
        b(k) = b0t(k) + 0.5*mumuandsigma(k,k);
        Elnpw = Elnpw + 0.5*(psi(a(k))-log(b(k))) - 0.5*a(k)./b(k)*mumuandsigma(k,k);
        Elnpa = Elnpa + (a0 - 1)*(psi(a(k))-log(b(k))) - b0*a(k)/b(k);
        Elnqa = Elnqa + a(k) - log(b(k)) + log(gamma(a(k))) + (1 - a(k))*psi(a(k));
        Eqalpha(k) = a(k)./b(k);
    end
    Elnpy = N/2*(psi(e)-log(f)) - 0.5*e/f*yxmu;
    Elnpl = (e0 - 1)*(psi(e)-log(f)) - f0*e/f;
    Elnqw = 0.5*log(det(10*sigma));
    Elnql = e - log(f) + gammaln(e) + (1 - e)*psi(e);
    L(i) = Elnpy + Elnpw + Elnpa + Elnpl + Elnqw + Elnqa + Elnql;
    
end
figure;
plot(L);
figure;
stem(1./Eqalpha);
Eqlambda = e/f;

y_hat = X*mu;
figure;
plot(z,y_hat, z,y,'.', z,10*sinc(z));

