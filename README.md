# Detection-of-COVID-19-from-medical-images
The source code is now under refactoring. We will release the code as soon as possible after the work is done.

There are four parts of our code, which are the classification code, the regression code, the knowledge distilling code and the utils code, respectively.

##classification
prerequirements
pytorch==1.3.0
torchvision
Pillow
opencv-python
pandas
numpy
scikit-learn

1. train the network
in the classification directory,run the following command.
'''
python train_msnet.py
'''
after trained, run the test.py to reproduce the classification results in our paper.
'''
python test.py 
'''

##regression
prerequirements
pytorch==1.3.0
torchvision
Pillow
opencv-python
pandas
numpy
scikit-learn
scipy

1. train the network
in the regression directory,run the following command.
'''
python train_regression.py
'''
after trained, run the test.py to reproduce the regression results in our paper.
'''
python test.py 
'''

##knowledge distilling
prerequirements
pytorch==1.3.0
torchvision
Pillow
opencv-python
pandas
numpy
scikit-learn
scipy

1. train teachers networks
in the distiling directory,run the following command.
'''
python train_teachers.py
'''
2. after the teachers networks were well trained, run the following command to train the target student network.
'''
python train_student.py 
'''
3. after all the models were trained, run the test_student.py to reproduce the results in our paper.
'''
python test_student.py

## we provide the evaluation metric code and the regression metrics code in the utils part
1. MSE
2. RMSE
3. R2
4. Pierson's correlation cofficient
5. MAE
ETC...
