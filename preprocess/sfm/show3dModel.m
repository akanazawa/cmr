function show3dModel(Shape, kp_names, type)    

    n_kp = size(Shape,2);
    colormap('hsv');
    %%%%%%%%%%%%% VISUALIZATION of 3d model %%%%%%%%%%%%%%
    if strcmp(type, 'convex_hull')
        %%%% visualize convex hull %%%

        T = delaunay(Shape(1,:), Shape(2,:), Shape(3,:));
        
        tetramesh(T, Shape'); %camorbit(20,0)
    else
        % visualize keypoint 3d locations in isolation
        plot3(Shape(1,:), Shape(2,:), Shape(3,:),'o', 'MarkerSize', 12, 'LineWidth', 8); hold on;
    end
    if~isempty(kp_names)
        hold on;
        for i=1:n_kp
            h = text(Shape(1,i), Shape(2,i), Shape(3,i), kp_names{i}, 'FontSize', 10, 'Color', [0 0 0], 'BackgroundColor', [1 1 1]);
            set(h,'interpreter','none');
        end
    end
    xlabel('x');ylabel('y');zlabel('z');
end