# Hackaton-2021-Adi-Nimrod-Keshet

## input description 
File Description: the file “AllEEG.mat” contains all the data required to run algorithms and models, both behavioral and neural, from the dynamic WTP experiment (1.22GB). It can be opened only via MATLAB, as it is a “struct” type variable.

**Primary Fields:**
•	*SubID*: the identifier of the subject whose data is in the same row.
•	*BDMEEG*: data related to the BDM phase of the experiment, for the subject in the row.
•	*AdEEG*: data related to the Advertisements phase of the experiment, for the subject in the row.
•	*Triggers*: a time series containing all the triggers presents throughout the experiment for the subject. You will likely not require this.

**Secondary Structs:**
AllEEG(<index>).BDMEEG: Once you access this field, you will find another struct variable nested within it. Here, each row represents a different trial of a given subject from the BDM phase. The fields of this struct are:
1.	*EEG*: a matrix (double) containing the EEG recordings from 8 electrodes (rows) throughout 4.5 seconds sampled at 500 Hz (columns). The first second is an ITI (“baseline”), and the following 3.5 seconds are viewing of the product’s image.
2.	*Category*: Category of the product viewed in the trial.
3.	*Item*: Name of the item viewed in the trial.
4.	*Repetition*: The number of appearances of the same product.
5.	*Label*: The value given to the product on that trial, from 0 to 100.
6.	*nArtifacts*: the number of faulty electrodes found in the trial (0 to 8), by an automatic artifact detection pipeline.

AllEEG(<index>).AdEEG: In this field you will find another struct variable nested within it. Each row in the nested struct represents a different trial of a given subject from the Ads phase. The fields are:
1.	*EEG*: A matrix (double) containing the EEG recordings from 8 electrodes (rows) throughout 7 seconds sampled at 500 Hz (columns). The first second is an ITI, and the following 6 seconds are viewing of the product’s commercial. Note that each product has several different commercials, that each is viewed once.
2.	*Category*: Category of the product viewed in the trial.
3.	*Item*: Name of the item viewed in the trial.
4.	*Repetition*: The number of appearances of different commercials for the same product.
5.	*AdID*: The Identifier of the ad. This is also mentioned in the “Products.xlsx” file, which can be related to an actual commercial in the video ads folder.
6.	*ItemID*: The identifier of the item.
7.	*Label*: The value given to the product on that trial, from 0 to 100.
8.	*Liking*: The value given to the commercial that appeared on that trial, from 0 to 7.
9.	*nArtifacts*: the number of faulty electrodes found in the trial, by an automatic artifact detection pipeline. 
 
## models description
the neegpdr packedge ofers two models of analysis:
  1. **Liking Model**: this model predicts the liking score a subject will give a certain commercial based on EEG activity
  2. **Diff Model**: this model predicts how will watching a commercial effect a subject's affinity towards an item based on EEG recordings
## experimental procedure
  ![Picture1](https://user-images.githubusercontent.com/80317440/124106824-480c7b00-da6d-11eb-832d-0b56e25033b1.png)
