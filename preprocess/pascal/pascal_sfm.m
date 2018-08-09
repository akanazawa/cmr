function [sfm_anno, sfm_verts, sfm_faces, kp_perm] = pascal_sfm(class_data, kp_names, horz_edges, vert_edges)
    kp_perm = findKpsPerm(kp_names);

    kps_all = [];
    vis_all = [];
    box_scale = [];
    n_objects = length(class_data);
    box_trans = zeros(n_objects, 2);

    left_inds = cellfun(@(x) ~isempty(x),strfind(kp_names,'Left'));
    left_inds = left_inds | cellfun(@(x) ~isempty(x),strfind(kp_names,'L_'));
    left_inds = left_inds | cellfun(@(x) ~isempty(x),strfind(kp_names,'left'));
    left_inds = find(left_inds);

    right_inds = kp_perm(left_inds);
    lr_edges = [left_inds right_inds];

    %% Construct keypoint matrix    
    for b = 1:n_objects
        % bbox to normalize
        bbox_h = class_data(b).bbox.y2 - class_data(b).bbox.y1 + 1;
        bbox_w = class_data(b).bbox.x2 - class_data(b).bbox.x1 + 1;
        box_scale(b) = max(bbox_w, bbox_h);
        kps_b = class_data(b).parts(1:2, :)/box_scale(b);
        % Add flipped data
        kps_b_flipped = kps_b(:, kp_perm);
        kps_b_flipped(1, :) = -kps_b_flipped(1, :);

        vis_b = vertcat(class_data(b).parts(3, :), class_data(b).parts(3, :));
        vis_b_flipped = vis_b(:, kp_perm);

        % Mean center here,,
        % keyboard;
        box_trans(b, :) = mean(kps_b(:, vis_b(1,:)>0), 2);
        box_trans_flipped = mean(kps_b_flipped(:, vis_b_flipped(1,:)>0), 2);
        %kps_b = kps_b - box_trans(b, :)';
        kps_b = kps_b - box_trans(b);
        kps_b_flipped = kps_b_flipped - box_trans_flipped;

        kps_all = vertcat(kps_all, kps_b, kps_b_flipped);

        vis_all = vertcat(vis_all, vis_b, vis_b_flipped);
        % keyboard
        % sfigure(2); clf;
        % scatter(kps_b(1,:), kps_b(2,:));
        % hold on;
        % scatter(kps_b_flipped(1,:), kps_b_flipped(2,:));
    end

    %% Compute mean shape and poses
    kps_all(~vis_all) = nan;
    [~, S, ~] = sfmFactorization(kps_all, 30, 10);
    % show3dModel(S, kp_names, 'convex_hull');
    %cameratoolbar

    %% Align mean shape to canonical directions
    good_model = 0;
    flip = 0;
    while(~good_model)
        R = alignSfmModel(S, lr_edges, horz_edges, vert_edges);
        Srot = R*S;
        show3dModel(Srot, kp_names, 'convex_hull');
        user_in = input('Is this model aligned ? "y" will save and "n" will realign after flipping \n','s');
        if(strcmp(user_in,'y'))
            good_model = 1;
            disp('Ok !')
        else
            flip = mod(flip+1,2);
            S = diag([1 1 -1])*S;        
        end
        close all;
    end
    S = Srot;
    max_dist = max(pdist(S'));
    S_scale = 2. / max_dist;
    fprintf('Scale Shape by %.2g\n', S_scale);
    S = S*S_scale;
    [M,T,~] = sfmFactorizationKnownShape(kps_all, S, 50);

    %%
    sfm_anno = struct;
    for bx = 1:n_objects
        b = 2*bx-1;
        motion = M([2*b-1, 2*b], :);
        scale = norm(motion(1,:));
        rot = motion/scale;
        rot = [rot;cross(rot(1,:),rot(2,:))];
        if(det(rot)<0)
            rot(3,:) = -rot(3,:);
        end
        % reproj = motion * S + T([2*b-1, 2*b], :);
        % reproj2 = rot * S;
        % reproj2 = scale * (reproj2(1:2, :)) + T([2*b-1, 2*b], :);
        % norm(reproj - reproj2);
        
        kps_b = (kps_all([2*b-1, 2*b], :) + box_trans(bx))*box_scale(bx);
        scale = scale*box_scale(bx);
        trans = T([2*b-1, 2*b], :)*box_scale(bx) + box_trans(bx)*box_scale(bx);

        [scale, rot, trans, err_sfm_reproj] = reprojMaskMinimize(kps_b, S, class_data(bx).mask,  scale, rot, trans);
        %[scale, rot, trans] = reprojMinimize(kps_b, S,  scale, rot, trans);
        sfm_anno(bx).rot = rot;
        sfm_anno(bx).scale = scale;
        sfm_anno(bx).trans = trans;
        sfm_anno(bx).err_sfm_reproj = err_sfm_reproj/(box_scale(bx)*box_scale(bx)*length(kp_perm));
    end

    %% Compute and save convex hull
    sfm_verts = S;
    sfm_faces = delaunay(S(1,:), S(2, :), S(3, :));
    sfm_faces = [sfm_faces(:, [1,2,3]); sfm_faces(:, [1,2,4]); sfm_faces(:, [1,3,4]); sfm_faces(:, [4,2,3])];

end

function kpsPerm = findKpsPerm(part_names)

    leftInds = cellfun(@(x) ~isempty(x),strfind(part_names,'Left'));
    leftInds = leftInds | cellfun(@(x) ~isempty(x),strfind(part_names,'L_'));
    leftInds = leftInds | cellfun(@(x) ~isempty(x),strfind(part_names,'left'));

    rightInds = cellfun(@(x) ~isempty(x),strfind(part_names,'Right'));
    rightInds = rightInds | cellfun(@(x) ~isempty(x),strfind(part_names,'R_'));
    rightInds = rightInds | cellfun(@(x) ~isempty(x),strfind(part_names,'right'));

    flipNames = part_names;
    flipNames(leftInds) = strrep(flipNames(leftInds),'L_','R_');
    flipNames(leftInds) = strrep(flipNames(leftInds),'Left','Right');
    flipNames(leftInds) = strrep(flipNames(leftInds),'left','right');

    flipNames(rightInds) = strrep(flipNames(rightInds),'R_','L_');
    flipNames(rightInds) = strrep(flipNames(rightInds),'Right','Left');
    flipNames(rightInds) = strrep(flipNames(rightInds),'right','left');

    kpsPerm = zeros(length(flipNames),1);
    for i=1:length(flipNames)
        kpsPerm(i) = find(ismember(part_names,flipNames{i}));
    end

end