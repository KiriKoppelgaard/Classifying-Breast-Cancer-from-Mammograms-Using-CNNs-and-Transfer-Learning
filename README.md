# Classifying Breast Cancer from Mammograms Using CNNs and Transfer Learning

This is the code for our data science exam, Spring 2022.

We use the DDSM and CBIS-DDSM dataset (https://www.kaggle.com/datasets/skooch/ddsm-mammography). 

/ Kiri Koppelgaard and Signe Kirk Brødbæk, MSc in Information Technology (Cognitive Science), Aarhus University
May 31, 2022

## Abstract 
Worldwide, breast cancer is the one of the most common cancers with over two million annual diagnoses. Detection of malignant tumours at an advanced stage of the disease often lead to more difficult treatment with higher mortality rates. Therefore, early detection is of utmost importance. The most common breast cancer screening method is mammography, which are usually manually inspected by radiologists. Recently, it has been shown that CNNbased algorithms can exceed radiologists’ interpretive accuracy (Trister et al., 2017). Potentially, CNN-based diagnosis tools could assist radiologists and help make the process of detecting breast cancer less expert dependent. In the current study, we develop three baseline CNNs and implement two transfer learning models, Inception-v3 and EfficientNetv2 to classify mammograms. All CNNs are trained and tested on the DDSM and CBIS-DDSM (Karssemeijer, 1998; Sawyer-Lee et al., 2016). The best performing model is Inception-v3 with an accuracy of 92.50 %. However, a relatively low specificity and a high sensitivity is obtained for the negative class, while the opposite is obtained for the non-negative classes. As a result, patients would be underdiagnosed by the model. Limitations include an imbalanced data set and under-regularised models. If any of these models were to be implemented as a reliable part of a CAD system, improvements must be made on the model performance. 

<i> Key words: Mammography, Breast Cancer, CNNs, Transfer Learning, DDSM, CBIS-DDSM </i>

## To run: 
- pip install -r requirements.txt
- python CNN.py 
- python transfer_learning.py
