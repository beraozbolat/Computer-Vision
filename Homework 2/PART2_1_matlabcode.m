image_folder = 'part2';  
segments_folder = 'part2/matlab_segments';  
output_folder = 'feature_matrices';  


nscale = 4;               % Number of scales
norient = 4;              % Number of orientations
minWaveLength = 4;        % Smallest wavelength
mult = 1.7;                 % Scaling factor for wavelength
sigmaOnf = 0.5;          % Ratio of standard deviation to frequency
dThetaOnSigma = 1.2;      % Ratio of angular spread
Lnorm = 0;                % No normalization
feedback = 1;             % Display progress

compute_and_save_all_features(image_folder, segments_folder, output_folder, ...
                              nscale, norient, minWaveLength, mult, ...
                              sigmaOnf, dThetaOnSigma, Lnorm, feedback);



 for img_idx = 1:10
    img_path = fullfile(image_folder, sprintf('%d.jpg', img_idx));
    im = imread(img_path);
    
    if size(im, 3) == 3
        im = rgb2gray(im); % Convert to grayscale
    end
    im = double(im);

    [EO, ~] = gaborconvolve(im, nscale, norient, minWaveLength, mult, ...
                            sigmaOnf, dThetaOnSigma, Lnorm, feedback);

    figure('Name', sprintf('Gabor Filters Visualization - Image %d', img_idx), ...
           'NumberTitle', 'off');
    for s = 1:nscale
        for o = 1:norient
            subplot(nscale, norient, (s-1)*norient + o);
            gabor_magnitude = abs(EO{s, o});
            imagesc(gabor_magnitude);
            colormap gray;
            axis off;
            title(sprintf('Scale %d, Orientation %d', s, o));
        end
    end
    subtitle(sprintf('Gabor Filter Results for Image %d', img_idx));
end





function compute_and_save_all_features(image_folder, segments_folder, output_folder, ...
                                       nscale, norient, minWaveLength, mult, ...
                                       sigmaOnf, dThetaOnSigma, Lnorm, feedback)
  
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    

    for img_idx = 1:10

        img_path = fullfile(image_folder, sprintf('%d.jpg', img_idx));
        im = imread(img_path);
       
        seg_path = fullfile(segments_folder, sprintf('%d_segments.mat', img_idx));
        segments = load(seg_path).segments;  
        
        [superpixel_features, ~] = compute_gabor_features(im, segments, ...
                                                          nscale, norient, ...
                                                          minWaveLength, mult, ...
                                                          sigmaOnf, dThetaOnSigma, ...
                                                          Lnorm, feedback);

        unique_segments = unique(segments);
        Ni = length(unique_segments);
        feature_matrix = zeros(Ni, nscale * norient);  
        
        for s = 1:nscale
            for o = 1:norient
                col_idx = (s - 1) * norient + o;
                feature_matrix(:, col_idx) = superpixel_features{s, o};
            end
        end
        
        save(fullfile(output_folder, sprintf('feature_matrices_%d.mat', img_idx)), 'feature_matrix');
        fprintf('Saved feature matrix for image %d\n', img_idx);
    end
end



function [superpixel_features, scale_orientation_map] = compute_gabor_features(im, segments, ...
                                                                              nscale, norient, ...
                                                                              minWaveLength, mult, ...
                                                                              sigmaOnf, dThetaOnSigma, ...
                                                                              Lnorm, feedback)
    if size(im, 3) == 3
        im = rgb2gray(im);
    end
    im = double(im);
    
    [EO, ~] = gaborconvolve(im, nscale, norient, minWaveLength, mult, ...
                            sigmaOnf, dThetaOnSigma, Lnorm, feedback);
    
    superpixel_features = cell(nscale, norient);
    
    scale_orientation_map = cell(nscale, norient);
    
    unique_segments = unique(segments);
    Ni = length(unique_segments);  
    
    for s = 1:nscale
        for o = 1:norient
            scale_orientation_map{s, o} = sprintf('Scale %d, Orientation %d', s, o);
                  
            gabor_magnitude = abs(EO{s, o});
          
            feature_vector = zeros(Ni, 1);
            
            for i = 1:Ni
               
                mask = (segments == unique_segments(i));
                
             
                feature_vector(i) = mean(gabor_magnitude(mask));
            end
            
           
            superpixel_features{s, o} = feature_vector;
        end
    end
end

function [EO, BP] = gaborconvolve(im, nscale, norient, minWaveLength, mult, ...
			    sigmaOnf, dThetaOnSigma, Lnorm, feedback)
    
    if ndims(im) == 3
        warning('Colour image supplied: Converting to greyscale');
        im = rgb2gray(im);
    end
    
    if ~exist('Lnorm','var'), Lnorm = 0;  end
    if ~exist('feedback','var'), feedback = 0;  end    
    if ~isa(im,'double'),  im = double(im);  end
    
    [rows cols] = size(im);					
    imagefft = fft2(im);                 % Fourier transform of image
    EO = cell(nscale, norient);          % Pre-allocate cell array
    BP = cell(nscale,1);
    
    % Pre-compute some stuff to speed up filter construction
    % Set up X and Y matrices with ranges normalised to +/- 0.5
    % The following code adjusts things appropriately for odd and even values
    % of rows and columns.
    if mod(cols,2)
        xrange = [-(cols-1)/2:(cols-1)/2]/(cols-1);
    else
        xrange = [-cols/2:(cols/2-1)]/cols; 
    end
    
    if mod(rows,2)
        yrange = [-(rows-1)/2:(rows-1)/2]/(rows-1);
    else
        yrange = [-rows/2:(rows/2-1)]/rows; 
    end
    
    [x,y] = meshgrid(xrange, yrange);
    
    radius = sqrt(x.^2 + y.^2);       % Matrix values contain *normalised* radius from centre.
    theta = atan2(y,x);               % Matrix values contain polar angle.
                                  
    radius = ifftshift(radius);       % Quadrant shift radius and theta so that filters
    theta  = ifftshift(theta);        % are constructed with 0 frequency at the corners.
    radius(1,1) = 1;                  % Get rid of the 0 radius value at the 0
                                      % frequency point (now at top-left corner)
                                      % so that taking the log of the radius will 
                                      % not cause trouble.
    sintheta = sin(theta);
    costheta = cos(theta);
    clear x; clear y; clear theta;    % save a little memory
    
    % Filters are constructed in terms of two components.
    % 1) The radial component, which controls the frequency band that the filter
    %    responds to
    % 2) The angular component, which controls the orientation that the filter
    %    responds to.
    % The two components are multiplied together to construct the overall filter.
    
    % Construct the radial filter components...
    % First construct a low-pass filter that is as large as possible, yet falls
    % away to zero at the boundaries.  All log Gabor filters are multiplied by
    % this to ensure no extra frequencies at the 'corners' of the FFT are
    % incorporated. This keeps the overall norm of each filter not too dissimilar.
    lp = lowpassfilter([rows,cols],.45,15);   % Radius .45, 'sharpness' 15

    logGabor = cell(1,nscale);

    for s = 1:nscale
        wavelength = minWaveLength*mult^(s-1);
        fo = 1.0/wavelength;                  % Centre frequency of filter.
        logGabor{s} = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2));  
        logGabor{s} = logGabor{s}.*lp;        % Apply low-pass filter
        logGabor{s}(1,1) = 0;                 % Set the value at the 0
                                              % frequency point of the filter 
                                              % back to zero (undo the radius fudge).
        % Compute bandpass image for each scale 
        if Lnorm == 2       % Normalize filters to have same L2 norm
            L = sqrt(sum(logGabor{s}(:).^2));
        elseif Lnorm == 1   % Normalize to have same L1
            L = sum(sum(abs(real(ifft2(logGabor{s})))));
        elseif Lnorm == 0   % No normalization
            L = 1;
        else
            error('Lnorm must be 0, 1 or 2');
        end
        
        logGabor{s} = logGabor{s}./L;        
        BP{s} = ifft2(imagefft .* logGabor{s});   
    end
    
    % The main loop...
    for o = 1:norient,                   % For each orientation.
        if feedback
            fprintf('Processing orientation %d \r', o);
        end
    
        angl = (o-1)*pi/norient;           % Calculate filter angle.
        wavelength = minWaveLength;        % Initialize filter wavelength.

        % Pre-compute filter data specific to this orientation
        % For each point in the filter matrix calculate the angular distance from the
        % specified filter orientation.  To overcome the angular wrap-around problem
        % sine difference and cosine difference values are first computed and then
        % the atan2 function is used to determine angular distance.
        ds = sintheta * cos(angl) - costheta * sin(angl);     % Difference in sine.
        dc = costheta * cos(angl) + sintheta * sin(angl);     % Difference in cosine.
        dtheta = abs(atan2(ds,dc));                           % Absolute angular distance.

        % Calculate the standard deviation of the angular Gaussian
        % function used to construct filters in the freq. plane.
        thetaSigma = pi/norient/dThetaOnSigma;  
        spread = exp((-dtheta.^2) / (2 * thetaSigma^2));  
        
        for s = 1:nscale,                    % For each scale.
            filter = logGabor{s} .* spread;  % Multiply by the angular spread to get the filter

            if Lnorm == 2      % Normalize filters to have the same L2 norm ** why sqrt 2 **????
                L = sqrt(sum(real(filter(:)).^2 + imag(filter(:)).^2 ))/sqrt(2);
                filter = filter./L;  
            elseif Lnorm == 1  % Normalize to have same L1
                L = sum(sum(abs(real(ifft2(filter)))));
                filter = filter./L;              
            elseif Lnorm == 0   % No normalization
                ;
            end

            % Do the convolution, back transform, and save the result in EO
            EO{s,o} = ifft2(imagefft .* filter);    
            
            wavelength = wavelength * mult;       % Finally calculate Wavelength of next filter
        end                                       % ... and process the next scale

    end  % For each orientation
    
    if feedback, fprintf('                                        \r'); end




end

function f = lowpassfilter(sze, cutoff, n)
    
    if cutoff < 0 | cutoff > 0.5
	error('cutoff frequency must be between 0 and 0.5');
    end
    
    if rem(n,1) ~= 0 | n < 1
	error('n must be an integer >= 1');
    end

    if length(sze) == 1
	rows = sze; cols = sze;
    else
	rows = sze(1); cols = sze(2);
    end

    [radius, u1, u2] = filtergrid(rows,cols);

    f = 1.0 ./ (1.0 + (radius ./ cutoff).^(2*n));   % The filter
    
end


function [radius, u1, u2] = filtergrid(rows, cols)

    % Handle case where rows, cols has been supplied as a 2-vector
    if nargin == 1 && length(rows) == 2  
        tmp = rows;
        rows = tmp(1);
        cols = tmp(2);
    end
    
    % Set up X and Y spatial frequency matrices, u1 and u2 The following code
    % adjusts things appropriately for odd and even values of rows and columns
    % so that the 0 frequency point is placed appropriately.  See
    % https://blogs.uoregon.edu/seis/wiki/unpacking-the-matlab-fft/
    if mod(cols,2)
        u1range = [-(cols-1)/2:(cols-1)/2]/cols;
    else
        u1range = [-cols/2:(cols/2-1)]/cols; 
    end
    
    if mod(rows,2)
        u2range = [-(rows-1)/2:(rows-1)/2]/rows;
    else
        u2range = [-rows/2:(rows/2-1)]/rows; 
    end
    
    [u1,u2] = meshgrid(u1range, u2range);
    
    % Quadrant shift so that filters are constructed with 0 frequency at
    % the corners
    u1 = ifftshift(u1);
    u2 = ifftshift(u2);
    
    % Construct spatial frequency values in terms of normalised radius from
    % centre. 
    radius = sqrt(u1.^2 + u2.^2);     
                          
end