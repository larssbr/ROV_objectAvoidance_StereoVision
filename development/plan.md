
Todays plan.

- 1 make that picture black and white, before doing anything else
DONE --> GIVES VORSE RESULTS
TESTED: YES

- 2 floating window clahe.
It means, taking the clahe of smaller ROI in image, instead of taking clahe at the whole image
DONE: NOT
TESTED:


- 3 try Theese methods Respecte, fft method,  LP
DONE:

TESTED:


- 4 imporve disparity image, by using a scale sin(20) to sin(50)
    make an array of values from sin(20) to sin(50)
    then multiply that array by the matrix
DONE:  YES
TESTED = yes, looks good so far

created: camereaAngleAdjuster(img):
disparity_visual_adjusted = camereaAngleAdjuster(disparity_visual)

- 5 make a "forventningbilde" of the bacground. so that 
disp - "forventningbilde" = objectPlane





"forventningbilde" is from (hightOverBottum/cos(20)) to (hightOverBottum/cos(50)) 
hightOverBottum = alltitude meassurment from ROV


--->Possible tips and ticks


Dilate to make dense pixel are more dense
errode to remove "Noise" or "smaller" objects in image

DONE : 

TESTED :

