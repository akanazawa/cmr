function [c, R, t, err_val] = reprojMaskMinimize(P, S, mask, c_init, R_init, t_init)
    
    vis_inds = ~isnan(P(1,:));
    P_vis = P(:, vis_inds);
    S_vis = S(:, vis_inds);

    S_non_vis = S(:, ~vis_inds);

    q_init = dcm2quat(R_init);
    x_init = [c_init' t_init' q_init];

    mask_dist = bwdist(mask);
    function err = reprojError(x)
        c_iter = x(1);
        t_iter = x(2:3);
        q_iter = x(:, 4:7); q_iter = q_iter/norm(q_iter);
        R_iter = quat2dcm(q_iter); R_iter = R_iter(1:2, :);

        proj_vis = c_iter*R_iter*S_vis;
        proj_vis = proj_vis + t_iter';
        err_kp = proj_vis - P_vis;
        err_kp = sum(sum(err_kp.*err_kp));
        
        proj_non_vis = c_iter*R_iter*S_non_vis;
        proj_non_vis = proj_non_vis + t_iter';
        if ~isempty(proj_non_vis)
            err_mask = chamferLossInterp(mask_dist, proj_non_vis);
        else
            err_mask = 0;
        end
        err = err_kp + double(err_mask);
    end
    
    %disp(reprojError(x_init));
    [x_final, err_val] = fminunc(@reprojError, x_init);
    %disp(reprojError(x_final));
    c = x_final(1);
    t = x_final(2:3);
    R = quat2dcm(x_final(4:7));
end


function err_chamfer = chamferLoss(mask_dist, points)
    err_chamfer = 0;
    points = round(points);
    imH = size(mask_dist, 1);
    imW = size(mask_dist, 2);
    for p = 1:size(points,2)
        pt_orig = points(:, p);
        pt = points(:, p); %(x,y)
        
        pt(1) = max(pt(1), 1);
        pt(2) = max(pt(2), 1);

        pt(1) = min(pt(1), imW);
        pt(2) = min(pt(2), imH);
        
        err_pt = (pt_orig - pt); err_pt = sum(sum(err_pt.*err_pt));
        err_mask = mask_dist(pt(2), pt(1));
        err_chamfer = err_chamfer + err_pt + err_mask*err_mask;
    end
end

function err_chamfer = chamferLossInterp(mask_dist, points)
    err_chamfer = 0;
    imH = size(mask_dist, 1);
    imW = size(mask_dist, 2);
    for p = 1:size(points,2)
        pt_orig = points(:, p);
        pt = points(:, p); %(x,y)
        
        pt(1) = max(pt(1), 1);
        pt(2) = max(pt(2), 1);

        pt(1) = min(pt(1), imW);
        pt(2) = min(pt(2), imH);
        
        points(:, p) = pt;
        err_pt = (pt_orig - pt); err_pt = sum(sum(err_pt.*err_pt));
        err_chamfer = err_chamfer + err_pt;
    end

    err_mask = interp2(mask_dist, points(1,:), points(2,:), 'linear');
    err_mask = sum(sum(err_mask.*err_mask));
    err_chamfer = err_chamfer + err_mask;
end