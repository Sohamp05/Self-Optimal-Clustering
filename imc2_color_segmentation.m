function [clusters, cluster_centers, GS, SI] = imc2_color_segmentation()
    % IMC-2: Improved Mountain Clustering Technique Version-2
    % For color image segmentation
    
    disp('input the image name in the format --> imread(image name, image format); ');
    a = input('Enter image data : ');
    [f, h, k] = size(a);
    nk = input('no. of clusters required : '); % nk is no. of clusters required
    
    % Extract RGB values
    for j = 1:f
        for i = 1:h
            R(j,i) = a(j,i,1); % extracting R value from original matrix a
            G(j,i) = a(j,i,2); % extracting G value from original matrix a
            B(j,i) = a(j,i,3); % extracting B value from original matrix a
        end
    end
    
    n = f*h; % n is total no. of data points
    
    % Create data matrix x where each row is [R, G, B]
    for j = 1:n
        x(j,:) = [R(j), G(j), B(j)];
    end
    
    % Convert to double for calculations
    x = double(x);
    
    % Step 1: Normalize data points to unit hypercube
    x_normalized = normalize_data(x);
    
    % IMC-2 Algorithm
    [clusters, cluster_centers] = imc2_algorithm(x_normalized, nk);
    
    % Calculate validation indices
    GS = calculate_global_silhouette(x_normalized, clusters);
    SI = calculate_separation_index(x_normalized, clusters, cluster_centers);
    
    disp(['Global Silhouette Index (GS): ', num2str(GS)]);
    disp(['Separation Index (SI): ', num2str(SI)]);
    
    % Visualize results
    visualize_clusters(a, clusters, f, h);
end

function x_norm = normalize_data(x)
    % Step 1: Normalize each dimension to [0,1]
    [n, D] = size(x);
    x_norm = zeros(n, D);
    
    for d = 1:D
        x_min = min(x(:,d));
        x_max = max(x(:,d));
        if x_max > x_min
            x_norm(:,d) = (x(:,d) - x_min) / (x_max - x_min);
        else
            x_norm(:,d) = x(:,d);
        end
    end
end

function [clusters, cluster_centers] = imc2_algorithm(x, M)
    % IMC-2 Main Algorithm
    [n, D] = size(x);
    
    % Initialize
    clusters = zeros(n, 1);
    cluster_centers = zeros(M, D);
    remaining_data = x;
    remaining_indices = 1:n;
    
    for m = 1:M
        if isempty(remaining_data)
            break;
        end
        
        % Step 2: Calculate threshold value d1 with heuristic factor
        d1 = calculate_threshold(remaining_data, M, D);
        
        % Step 3: Calculate potential values using mountain function
        P = calculate_potential(remaining_data, d1);
        
        % Step 4: Select cluster center with highest potential
        [~, max_idx] = max(P);
        center = remaining_data(max_idx, :);
        cluster_centers(m, :) = center;
        
        % Step 5: Assign data points to cluster
        distances = sqrt(sum((remaining_data - center).^2, 2));
        cluster_members = distances <= d1;
        
        % Assign cluster labels
        original_indices = remaining_indices(cluster_members);
        clusters(original_indices) = m;
        
        % Step 6: Remove clustered data points
        remaining_data = remaining_data(~cluster_members, :);
        remaining_indices = remaining_indices(~cluster_members);
    end
    
    % Step 8: Assign remaining points to nearest clusters
    if any(clusters == 0)
        unassigned = find(clusters == 0);
        for i = 1:length(unassigned)
            idx = unassigned(i);
            distances = sqrt(sum((x(idx, :) - cluster_centers).^2, 2));
            [~, nearest_cluster] = min(distances);
            clusters(idx) = nearest_cluster;
        end
    end
end

function d1 = calculate_threshold(x, M, D)
    % Step 2: Calculate threshold with IMC-2 improvement
    [n, ~] = size(x);
    
    % Calculate pairwise distances
    sum_distances = 0;
    count = 0;
    
    for i = 1:n
        for j = i+1:n
            sum_distances = sum_distances + norm(x(i,:) - x(j,:));
            count = count + 1;
        end
    end
    
    % Original threshold calculation
    if count > 0
        avg_distance = sum_distances / count;
        d1_base = avg_distance / (2 * D);
    else
        d1_base = 0.1; % Default value
    end
    
    % IMC-2 heuristic factor D (based on number of clusters)
    % From paper: D = 1 - 1/M
    heuristic_factor = 1 - 1/M;
    
    % Apply heuristic factor to improve threshold
    d1 = d1_base * heuristic_factor;
    
    % Ensure minimum threshold
    d1 = max(d1, 0.01);
end

function P = calculate_potential(x, d1)
    % Step 3: Calculate potential using mountain function
    [n, ~] = size(x);
    P = zeros(n, 1);
    
    for r = 1:n
        sum_exp = 0;
        for j = 1:n
            distance_sq = sum((x(r,:) - x(j,:)).^2);
            sum_exp = sum_exp + exp(-distance_sq / (2 * d1^2));
        end
        P(r) = sum_exp;
    end
end

function GS = calculate_global_silhouette(x, clusters)
    % Step 9: Calculate Global Silhouette Index
    M = max(clusters);
    cluster_silhouettes = zeros(M, 1);
    
    for m = 1:M
        cluster_points = find(clusters == m);
        if length(cluster_points) <= 1
            cluster_silhouettes(m) = 0;
            continue;
        end
        
        silhouette_values = zeros(length(cluster_points), 1);
        
        for i = 1:length(cluster_points)
            point_idx = cluster_points(i);
            
            % Calculate a(i) - average distance within cluster
            same_cluster = cluster_points(cluster_points ~= point_idx);
            if ~isempty(same_cluster)
                a_i = mean(sqrt(sum((x(point_idx,:) - x(same_cluster,:)).^2, 2)));
            else
                a_i = 0;
            end
            
            % Calculate b(i) - minimum average distance to other clusters
            b_i = inf;
            for k = 1:M
                if k ~= m
                    other_cluster = find(clusters == k);
                    if ~isempty(other_cluster)
                        avg_dist = mean(sqrt(sum((x(point_idx,:) - x(other_cluster,:)).^2, 2)));
                        b_i = min(b_i, avg_dist);
                    end
                end
            end
            
            % Calculate silhouette value
            if max(a_i, b_i) > 0
                silhouette_values(i) = (b_i - a_i) / max(a_i, b_i);
            else
                silhouette_values(i) = 0;
            end
        end
        
        cluster_silhouettes(m) = mean(silhouette_values);
    end
    
    GS = mean(cluster_silhouettes);
end

function SI = calculate_separation_index(x, clusters, centers)
    % Calculate Separation Index
    M = max(clusters);
    [n, ~] = size(x);
    
    % Calculate numerator: sum of squared distances from points to centers
    numerator = 0;
    for j = 1:n
        cluster_id = clusters(j);
        distance_sq = sum((x(j,:) - centers(cluster_id,:)).^2);
        numerator = numerator + distance_sq;
    end
    
    % Calculate denominator: minimum distance between cluster centers
    min_center_distance = inf;
    for k = 1:M
        for l = k+1:M
            center_distance = norm(centers(k,:) - centers(l,:));
            min_center_distance = min(min_center_distance, center_distance);
        end
    end
    
    if min_center_distance > 0
        SI = numerator / (n * min_center_distance^2);
    else
        SI = inf;
    end
end

function visualize_clusters(original_image, clusters, f, h)
    % Visualize clustering results
    figure;
    
    % Original image
    subplot(2,2,1);
    imshow(original_image);
    title('Original Image');
    
    % Create segmented image
    segmented = zeros(f, h, 3);
    M = max(clusters);
    colors = hsv(M); % Generate distinct colors for each cluster
    
    for i = 1:f*h
        cluster_id = clusters(i);
        [row, col] = ind2sub([f, h], i);
        segmented(row, col, :) = colors(cluster_id, :);
    end
    
    subplot(2,2,2);
    imshow(segmented);
    title('Segmented Image (IMC-2)');
    
    % Show cluster distribution
    subplot(2,2,3);
    histogram(clusters, 1:M+1);
    title('Cluster Distribution');
    xlabel('Cluster ID');
    ylabel('Number of Points');
    
    % Show individual clusters
    subplot(2,2,4);
    cluster_image = zeros(f, h);
    for i = 1:f*h
        [row, col] = ind2sub([f, h], i);
        cluster_image(row, col) = clusters(i);
    end
    imagesc(cluster_image);
    colormap(jet);
    colorbar;
    title('Cluster Map');
end