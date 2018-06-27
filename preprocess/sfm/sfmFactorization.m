% [Motion, Shape, T] = factorization(Wo, iterMax1, iterMax2, stopError1, stopError2)
% 
% This function computes the 3D object shape from missing and degenerate data. 
% 
% 
% Input arguments:
% 
%     Wo - The data matrix is defined as:
%                 Wo = [ u_1^1  ...  u_P^1
%                        v_1^1  ...  v_P^1
%                          .    .      .
%                          .     .     .
%                          .      .    .
%                        u_1^F  ...  u_P^F
%                        v_1^F  ...  v_P^F ]
%                 The missing data entries must be identified with NaN.
%                 
%     iterMax1 [Typical values: 3000-15000] - Maximum number of iterations for 
%                                             the missing data algorithm
%     iterMax2 [100-500] - Maximum number of iterations for Rigid Factorization
%     stopError1 [10^(-4) - 10^(-7)] - The missing data algorithm stops if the
%                                      error between two consecutive iterations
%                                      is lower than this threshold
%     stopError2 [10^(-1) - 10^(-3)] - Rigid Factorization algorithm stops if 
%                                      the error between two consecutive iterations 
%                                      is lower than this threshold
% 
% Note that Rigid Factorization is one step of the global missing data algorithm.
% 
% 
% Output arguments:
% 
%     Motion - The motion matrix stacks the camera matrices in all frames. The
%              camera matrix in frame f is a Stiefel matrix and is composed by lines 
%              2*f-1 and 2*f.
%     Shape - 3D object shape
%     T - Translation vector
%     
% 
% For details, see:
% 
%    Marques, M., and Costeira, J.. "Estimating 3D shape from degenerate sequences with missing data", 
%    Computer Vision and Image Understanding, 113(2):261-272, February 2009.
             
function [Motion, Shape, T] = sfmFactorization(Wo, iterMax1, iterMax2)

M = not(isnan(Wo));
Wo(M == 0) = 0;
Wo = (sum(Wo,2)./sum(M,2)*ones(1,size(Wo,2))).*not(M) + Wo.*M;
W = Wo;

nImgs = floor(size(W,1)/2);
iter1 = 0;

ind = find(sum(M,1) == 2*nImgs);


T = mean(W,2);


[o1,e,o2]=svd(W);
E=e(1:3,1:3);
O1=o1(:,1:3);
O2=o2(:,1:3);
R=O1*sqrtm(E);
S=sqrtm(E)*O2';
    
try
while iter1 < iterMax1
    %if(~mod(iter1,10))
    %    disp(['Iteration ' num2str(iter1) '/' num2str(iterMax1)])
    %end
    W = W - T*ones(1,size(W,2));
    Woit = Wo - T*ones(1,size(W,2));

    iterAux = 0;

    while iterAux < iterMax2
        Motion = [];        
        for i = 1:nImgs
            A_f = projStiefel(R(2*i-1:2*i,:)');
            Motion = [Motion; A_f'];
        end
        Shape = pinv(Motion)*W;
        R = W*pinv(Shape);
        iterAux = iterAux + 1;
    end

    W = Motion*Shape + T*ones(1,size(W,2));
    W = Motion*Shape.*not(M) + Woit.*M + T*ones(1,size(W,2));

    iter1 = iter1 + 1;
    
    T = mean(W,2);
    Motionret=Motion;
    Tret=T;
    Shaperet=Shape;
end

catch
    Motion=Motionret;
    T=Tret;
    Shape=Shaperet;
    
end

end

function W = projStiefel(Wo)

[U,D,V] = svd(Wo,'econ');
c = mean(diag(D));

W = c*U*V';

end
