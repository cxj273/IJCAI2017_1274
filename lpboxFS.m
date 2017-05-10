function [A,V,obj] = lpboxFS(xTr,yTr,lambdaA,P)

% P is the number of selected features
% lambdaA is the regularization parameter
rho = 1; 
mu = 1.05; 
MaxIter = 200; 

[Dim,N] = size(xTr); 
classlabels = unique(yTr); 
numclass = numel(classlabels); 
y = zeros(N,1);
for iter1 = 1:numclass
   ind = yTr == classlabels(iter1);
   y(ind) = iter1; 
end
yTr = full(sparse(y,1:N,1)); 

V = ones(Dim,1); V1 = zeros(size(V)); V2 = zeros(size(V));  
A = rand(numclass,Dim);   ones1 = ones(size(V)); 
y1 = 0;     y2 = zeros(size(V));  y3 = zeros(size(V)); 

PHI = xTr*xTr';   Id = eye(Dim); obj =0; 
for iter1 = 1:MaxIter
    dV = diag(V);
    % update projection mapping A
    A = (yTr*xTr'*dV)/((dV*xTr)*(xTr'*dV) + lambdaA*Id); 
    % compute V
    PSI = A'*A; 
    THETA = xTr*yTr'*A; 
    V =(2*(PHI.*PSI') + rho*(ones1*ones1')+2*rho*Id)\(2*diag(THETA) + rho*((P-(y1/rho))*ones1+ V1 - (y2/rho) + V2 - (y3/rho)));  
    % Projection on Sb and Sp
    V1 = PSb(V+(y2/rho)); 
    V2 = PSp(V+(y3/rho)); 
    % update the parameters
    y1 = y1 + rho*(V'*ones1 - P); 
    y2 = y2 + rho*(V-V1); 
    y3 = y3 + rho*(V-V2);
    rho = rho*mu;
    % compute the objective function 
    obj(iter1) = norm(A*diag(V)*xTr - yTr,'fro')^2 + lambdaA*norm(A,'fro')^2;
end

end

function X = PSb(X,a,b)
if ~exist('a','var')
    a = 0; b =1;
end
A = X(:); 
ind1 = A<a; ind2 = A>b;
A(ind1)=a; A(ind2) = b; 
X = reshape(A, size(X));
end

function X = PSp(X)
Dim = length(X); 
ones1 = ones(Dim,1);
t0 = sqrt(Dim)/(2*norm(X-(ones1/2),2)); 
X1 = (ones1/2) + t0*(X-(ones1/2)); 
X2 = (ones1/2) - t0*(X-(ones1/2)); 
d1 = norm(X-X1,2); d2 = norm(X-X2,2);
if d1>d2 
    X = X2;
else
    X = X1;
end
end

