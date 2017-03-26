function myVisualize(test_image, heatMaps, predict, param, rectangle)

model = param.model(param.modelID);
np = model.np;
im = imread(test_image);
facealpha = 0.6; % for limb transparency
[x_start, x_end, y_start, y_end] = get_display_range(rectangle, im);
predict = bsxfun(@minus, predict, [x_start, y_start]); % offset due to display range


% plot all parts and background
truncate = zeros(1,np);
for part = 1:np+1
    response = heatMaps{end}(:,:,part);
    max_value = max(max(response));
    
    % plot predictions on heat maps
    if(part~=np+1)
        if(max_value <= 0.15)
            truncate(part) = 1;
        end
    end
end

% plot full pose
figure(2)
imshow(im(y_start:y_end, x_start:x_end, :));
hold on;
bodyHeight = max(predict(:,2)) - min(predict(:,2));
plot_visible_limbs(model, facealpha, predict, truncate, bodyHeight/30);
plot(predict(:,1), predict(:,2), 'k.', 'MarkerSize', bodyHeight/32);
title('Full Pose');



%% function area
function [x_start, x_end, y_start, y_end] = get_display_range(rectangle, im)
    x_start = max(rectangle(1), 1);
    x_end = min(rectangle(1)+rectangle(3), size(im,2));
    y_start = max(rectangle(2), 1);
    y_end = min(rectangle(2)+rectangle(4), size(im,1));
    center = [(x_start + x_end)/2, (y_start + y_end)/2];
    % enlarge range
    x_start = max(1, round(x_start - (center(1) - x_start) * 0.2));
    x_end = min(size(im,2), round(x_end + (center(1) - x_start) * 0.2));
    y_start = max(1, round(y_start - (center(2) - y_start) * 0.2));
    y_end = min(size(im,1), round(y_end + (center(2) - y_start) * 0.2));

function plot_visible_limbs(model, facealpha, predict, truncate, stickwidth)
    % plot limbs as ellipses
    limbs = model.limbs;
    colors = hsv(length(limbs));

    for p = 1:size(limbs,1) % visibility?
        if(truncate(limbs(p,1))==1 || truncate(limbs(p,2))==1)
            continue;
        end
        X = predict(limbs(p,:),1);
        Y = predict(limbs(p,:),2);

        if(~sum(isnan(X)))
            a = 1/2 * sqrt((X(2)-X(1))^2+(Y(2)-Y(1))^2);
            b = stickwidth;
            t = linspace(0,2*pi);
            XX = a*cos(t);
            YY = b*sin(t);
            w = atan2(Y(2)-Y(1), X(2)-X(1));
            x = (X(1)+X(2))/2 + XX*cos(w) - YY*sin(w);
            y = (Y(1)+Y(2))/2 + XX*sin(w) + YY*cos(w);
            h = patch(x,y,colors(p,:));
            set(h,'FaceAlpha',facealpha);
            set(h,'EdgeAlpha',0);
        end
    end