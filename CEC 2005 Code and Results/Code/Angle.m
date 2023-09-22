x=25;
y=20;

cosine=pdist2(x,y,'cosine');
fprintf("cosine similarity");
cosine_similarity=1-cosine
fprintf("Angle between them");
theta=acosd(cosine_similarity) 


v=[x;y]
center=repmat([x,y],1,length(x))
%x_center = x)
%y_center = y(3)
%center = repmat([x_center; y_center], 1, length(x))
R = [cos(theta) -sin(theta); sin(theta) cos(theta)]

vo = R*(v - center) + center

% pick out the vectors of rotated x- and y-data
x_rotated = vo(1,:);
y_rotated = vo(2,:);
% make a plot
plot(x, y, 'k-', x_rotated, y_rotated, 'r-', x_center, y_center, 'bo')
axis equal