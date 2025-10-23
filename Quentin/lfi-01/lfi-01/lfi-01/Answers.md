# What are the problems of this clustering algorithm ?

This algorithm works, but it can have several defaults :

1. The results and the efficiency depends on the initialisation
2. You have to fix a k value in advance
3. Can be long to compute on large images
4. It supposes that the distance we percieve between colors are well represented in the color space we use (RGB is not the best for example)

# How can I improve the results ?

Here are some ideas to improve this algorithm :

- Change the color space for one that represents better the distance between colors according to Euclide (LAB for example) (will fix point 4.)
- Use a smart strategy for the initialisation (will improve point 1.)
- Maybe add some pre-processing on the image such as a noise reduction ? (can improve the results we obtain)