% Image Analysis and Enhancement Program

% Step 1: Read the input image
[file, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files'}, 'Select an Image');
if isequal(file, 0)
    disp('No image selected.');
    return;
end
img = imread(fullfile(path, file));
imshow(img);
title('Original Image');

% Step 2: Determine image type
if islogical(img)
    disp('Image type: Binary');
elseif size(img, 3) == 1
    disp('Image type: Grayscale');
else
    disp('Image type: RGB');
    img = rgb2gray(img); % Convert RGB to grayscale for further processing
    disp('Converted to Grayscale for analysis.');
end

% Step 3: Analyze the image histogram
figure;
imhist(img);
title('Image Histogram');

mean_intensity = mean(img(:));
if mean_intensity < 75
    disp('Issue: Image is too dark.');
    enhancement_type = 'brighten';
elseif mean_intensity > 180
    disp('Issue: Image is too bright.');
    enhancement_type = 'darken';
else
    contrast = max(img(:)) - min(img(:));
    if contrast < 50
        disp('Issue: Low contrast.');
        enhancement_type = 'contrast';
    else
        disp('Image is normal.');
        enhancement_type = 'none';
    end
end

% Step 4: Enhance the image based on the detected issue
switch enhancement_type
    case 'brighten'
        enhanced_img = img + 50; % Brighten by increasing pixel intensity
    case 'darken'
        enhanced_img = img - 50; % Darken by decreasing pixel intensity
    case 'contrast'
        enhanced_img = imadjust(img); % Increase contrast
    otherwise
        enhanced_img = img; % No enhancement needed
end

% Step 5: Display the results
figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(enhanced_img);
title('Enhanced Image');
disp('Enhancement completed.');
