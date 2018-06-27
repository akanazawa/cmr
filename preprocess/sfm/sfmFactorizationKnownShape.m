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
             
function [Motion, T, W] = sfmFactorizationKnownShape(Wo, Shape, iterMax1)

M = not(isnan(Wo));
Wo(M == 0) = 0;
Wo = (sum(Wo,2)./sum(M,2)*ones(1,size(Wo,2))).*not(M) + Wo.*M;
W = Wo;

nImgs = floor(size(W,1)/2);
iter1 = 0;


T = mean(W,2);

R = repmat([1 0 0;0 1 0],nImgs,1);

try
while iter1 < iterMax1

    W = W - T*ones(1,size(W,2));
    Woit = Wo - T*ones(1,size(W,2));


    Motion = [];
    for i = 1:nImgs
        A_f = projStiefel(R(2*i-1:2*i,:)');
        Motion = [Motion; A_f'];
    end
    %Shape = pinv(Motion)*W;
    R = W*pinv(Shape);

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

W = Motion*Shape + T*ones(1,size(W,2));

end

function W = projStiefel(Wo)
[U,D,V] = svd(Wo,'econ');
c = mean(diag(D));
W = c*U*V';
end