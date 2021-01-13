# Library-population-statistics-based-on-fasterrcnn
As an important public place in Colleges and universities, how to use its resources efficiently is of vital importance. It is of great practical significance to make full use of library resources by using library surveillance video and counting the number of people in various areas of the library. This topic focuses on the method of human detection in Library video, and uses the corresponding classifier to detect and count the target.

The test result(The image is from the library of UESTC):

![image](https://github.com/Alkaid-AI/Library-population-statistics-based-on-fasterrcnn/blob/master/images/1.jpg)

From the test result we can see that we have a high accuracy to detect the people in library and I also add a statistics function to count the total population at the bottom right corner of the image: "Total: 4"
And also I add a function to allow us to count population from a camera which can extract samples of the surveillance video in the real situlation of library. 

In addition, considering that the location of the monitors for library is beneath the ceiling and all we want is the statistics of the number of people in each region which is similar to the estimation of crowd density, I finally in-tend to use the existing mall data sets to replace the library video which is not available due to limited time and money. Another point to consider is that most of the people in the library are in sitting posture. Using the above data sets may lead to poor generalization ability of the model when applied to actual scenes and that maybe why one person is missed to be detected in the above test image. If there is a chance to get the real library dataset I think it will be better to improve the accuracy.
