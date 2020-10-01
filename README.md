# Analysing Rock Microstructures

I have received data from people who have previously worked on this task.
They have done the following:
  1. Transfer learning using ResNet50 (code availble) [74.65 training accuracy - 5 epochs]
              - Available in Transfer Learning in Keras.ipynb
  2. YOLOv3 (code not available)

I have been tasked with finding out alternative machine learning approaches to further improve the predictions.

The issue, as usual, is the data. The images available to our team are micrographs, these are basically images that someone painstakingly took. Naturally this involves a lot of man-hours and resources. Bottom line, we gotta work with what we have for now!

Steps I'm going to take:
- S1. Data augmentation
- S2. Try other models using transfer learning
- S3. Try to come up with models that could capture the essence
- S4. Maybe try running more number of epochs on the existing Transfer Learning sheet?
  
  
  S4. Increasing epochs on the Transfer Learning notebook.
          I increased the number of epochs from 5 to 10. The training accuracy improved from 75 to 92.7, this isn't really an accurate measure since we have only about 103 images of our positive class, which is not enough by any standard. I also should probably ask, why they stopped at 5 epochs lol.
  
  
 Note:
  I am not a geologist, or a person who has a lot of experience with the field. I did some reading up on these topics, but apart from that my process is purely based on visual identificiation of differences between the two classes and to use my experience in machine learning to come up with a solution
