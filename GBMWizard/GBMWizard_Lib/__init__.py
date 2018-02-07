""" Meant to be imported by ModelSegmentation.py. Conveniently loads all other files in the folder and tells the user in console.
"""

from GBMWizardStep import *
from VolumeSelect import *
from Preprocess import *
from SkullStrip import *
from Segmentation import *
from Radiomics import *
from Review import *

print 'GBMWizard Correctly Loaded'