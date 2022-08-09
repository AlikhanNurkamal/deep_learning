# Face emotion recognition using CNN #

## Problem ##
In this project I tried to build a model that can recognize an emotion based on an image of a person's face. There are 7 different labels of emotions (anger, fear, neutral, etc.) and the model should recognize the facial expression just by looking at an image of a person.
One of the major questions here is if it is possible to build an accurate model with high training and validation accuracies.

## Dataset ##
I have used the "fer2013" dataset, which is also in this directory but is archived. The link to the dataset is provided: https://drive.google.com/file/d/1-ltonXDdaCc-iRyV79bPBxaWPpUn28wB/view.

### Labels description ###
In the implementation stage the numeric representations of labels were used, so it might be not clear.
**Thus, here is the description of labels:**
* 0: Anger
* 1: Disgust
* 2: Fear
* 3: Happiness
* 4: Sadness
* 5: Surprise
* 6: Neutral

## Results ##
As seen in the notebook, the **last** `training loss` was 0.1666 and the **last** `cross-validation loss` comprised 2.4466. However, only the best model based on a validation accuracy has been saved to the checkpoint directory: "best_model.h5". The best model's `training loss` was 0.9351, `validation loss` was 1.1902, `training accuracy` comprised 61.02%, and the `validation accuracy` was 55.52%.
However, as the model continued training, it was overfitting more and more, since the training accuracy has been increasing, while the validation accuracy stayed relatively the same. This might be due to the lack of regularization or Dropout layers during training.

## Final thoughts ##
Well, the model does have low accuracy and high losses thus performs poorly on the unseen data in the testing dataset. It is thought that in order to improve the model, I need a larger dataset with balanced distribution of different classes of images.

## Acknowledgements ##
I would like to express acknowledgements to YouTube channel <a href="https://www.youtube.com/watch?v=T3yR9DZT2mQ&ab_channel=WhenMathsMeetCoding">***When Maths Meet Coding***</a>.