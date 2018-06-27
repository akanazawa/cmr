function [R] = alignSfmModel(S,lrEdges,horzEdges,vertEdges)
%ALIGNSFMMODEL Summary of this function goes here
%   Detailed explanation goes here
% S is 3 X N
% z is vertical axis in the shape
% y axis along length of the car (y increases along a horzEdge)
% YZ plane is the symmetry plane with X increasing from object's anatomical
% right to left

std_s = mean(sum(S.*S, 1));
S = S/std_s;

if(~isempty(lrEdges) && size(lrEdges,2)~=2)
    lrEdges = lrEdges';
end

if(~isempty(horzEdges) && size(horzEdges,2)~=2)
    horzEdges = horzEdges';
end

if(~isempty(vertEdges) && size(vertEdges,2)~=2)
    vertEdges = vertEdges';
end

edges = {lrEdges,horzEdges,vertEdges};
directions = zeros(3,3);
directionsTarget = eye(3);

%% computing desired x,y,z directions
for d = 1:3
    for i=1:size(edges{d},1)
        vec = S(:,edges{d}(i,1)) - S(:,edges{d}(i,2));
        if(norm(vec)>0.1)
            directions(:,d) = directions(:,d) + vec/norm(vec);
        end
    end
    if(norm(directions(:,d)) > 0)
        directions(:,d) = directions(:,d)/norm(directions(:,d));
    end
end

%% orthogonalization
isAvailable = ones(1,3);
for d = 2:3
    directions(:,d) = directions(:,d) - directions(:,1)*((directions(:,1))'*(directions(:,d)));
    if(norm(directions(:,d)) > 0)
        directions(:,d) = directions(:,d)/norm(directions(:,d));
    else
        isAvailable(d) = 0 ;
    end
end

%% computing rotation x axis as mirror axis
Rx = vrrotvec2mat(vrrotvec(directions(:,1),directionsTarget(:,1)));

%% other direction    
R = Rx;
directions = Rx*directions;
for d=2:3
    if(isAvailable(d))
        Ryz = vrrotvec2mat(vrrotvec(directions(:,d),directionsTarget(:,d)));
        R = Ryz*R;
        directions = Ryz*directions;
        return
    end
end

end