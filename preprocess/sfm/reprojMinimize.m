function [c, R, t] = reprojMinimize(P, S, c_init, R_init, t_init)
    
    vis_inds = ~isnan(P(1,:));
    P = P(:, vis_inds);
    S = S(:, vis_inds);

    q_init = dcm2quat(R_init);
    x_init = [c_init' t_init' q_init];

    function err = reprojError(x)
        c_iter = x(1);
        t_iter = x(2:3);
        q_iter = x(:, 4:7); q_iter = q_iter/norm(q_iter);
        R_iter = quat2dcm(q_iter); R_iter = R_iter(1:2, :);

        proj = c_iter*R_iter*S;
        proj = proj + t_iter';
        err = proj - P;
        err = sum(sum(err.*err));
    end
    
    %disp(reprojError(x_init));
    [x_final, err_val] = fminunc(@reprojError, x_init);
    %disp(reprojError(x_final));
    c = x_final(1);
    t = x_final(2:3);
    R = quat2dcm(x_final(4:7));
end