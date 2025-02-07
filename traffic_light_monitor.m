% Traffic Light Monitoring System in MATLAB
% Uses Computer Vision to Detect and Track Traffic Light Changes

% 1. Initialize Camera
vid = videoinput('winvideo', 1, 'YUY2_640x480'); % Adjust for your camera
preview(vid); % Show live camera feed
pause(2); % Allow time to initialize

% 2. Capture Initial Frame
frame = getsnapshot(vid); % Take a snapshot
frame = im2double(frame); % Normalize pixel values

% 3. Define Region of Interest (ROI) - Select Manually or Auto-detect
roi = imcrop(frame); % Manually select the traffic light region
imshow(roi); % Display the selected region

% 4. Color Detection and Monitoring Loop
previousColor = ""; % Store the last detected color
colors = ["Red", "Yellow", "Green"];

while true
    % Capture Frame and Extract ROI
    frame = getsnapshot(vid); % Capture a new frame
    roi = imcrop(frame); % Crop to traffic light region
    hsvImage = rgb2hsv(roi); % Convert to HSV color space
    H = hsvImage(:,:,1); % Extract the Hue channel
    
    % 5. Define Color Thresholds for Detection
    redMask = (H > 0.9 | H < 0.1);  % Red (Hue wraps around)
    yellowMask = (H > 0.10 & H < 0.18); % Yellow
    greenMask = (H > 0.25 & H < 0.40);  % Green
    
    % 6. Count Pixels in Each Mask
    redPixels = sum(redMask(:));
    yellowPixels = sum(yellowMask(:));
    greenPixels = sum(greenMask(:));
    
    % 7. Determine Active Traffic Light
    [~, idx] = max([redPixels, yellowPixels, greenPixels]);
    currentColor = colors(idx);
    
    % 8. Trigger Event if Color Changes
    if ~strcmp(currentColor, previousColor)
        disp(['Traffic Light Changed to: ', currentColor]);
        previousColor = currentColor;
    end
    
    pause(1); % Delay to process frames in real-time
end

% 9. Cleanup (Optional)
stop(vid);
clear vid;
