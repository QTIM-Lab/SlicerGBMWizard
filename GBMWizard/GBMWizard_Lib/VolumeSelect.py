""" This is Step 1. The user selects the pre- and, if preferred, post-contrast
    volumes.
"""

from __main__ import qt, ctk, slicer

from GBMWizardStep import *
from Helper import *

""" VolumeSelectStep inherits from GBMWizardStep, with itself inherits
    from a ctk workflow class. 
"""

class VolumeSelectStep(GBMWizardStep) :

    def __init__(self, stepid):

        """ This method creates a drop-down menu including the whole step.
            The description also acts as a tooltip for the button. There may be 
            some way to override this. The initialize method is inherited
            from ctk.
        """

        self.initialize( stepid )
        self.setName('1. Volume Selection')

        self.__parent = super(VolumeSelectStep, self)

        self.volumeLabels = ['t1Pre', 't1Post', 't2', 'flair']

    def createUserInterface(self):

        """ This method uses qt to create a user interface. qMRMLNodeComboBox
            is a drop down menu for picking MRML files. MRML files have to be
            added to a "scene," i.e. the main Slicer container, hence setMRMLScene.
        """

        self.__layout = self.__parent.createUserInterface()

        image_label = qt.QLabel('Test label')
        image = qt.QPixmap('C:/Users/azb22/Documents/GitHub/Public_qtim_tools/SlicerGBMWizard/GBMWizard/GBMWizard_Lib/VolumeSelect.png')
        image_label.setPixmap(image)
        image_label.alignment = 4
        self.__layout.addRow(image_label)

        step_label = qt.QLabel( 'This module requires four MRI volumes for GBM cases. Please select pre- and post-contrast T1 images, a T2 image, and a FLAIR image from a single patient visit. The ability to use multiple cases will be added in future versions.' )
        step_label.setWordWrap(True)
        self.__informationGroupBox = qt.QGroupBox()
        self.__informationGroupBox.setTitle('Information')
        self.__informationGroupBoxLayout = qt.QFormLayout(self.__informationGroupBox)

        self.__volumeSelectionGroupBox = qt.QGroupBox()
        self.__volumeSelectionGroupBox.setTitle('Volume Selection')
        self.__volumeSelectionGroupBoxLayout = qt.QFormLayout(self.__volumeSelectionGroupBox)

        t1PreScanLabel = qt.QLabel('Pre-Contrast T1 Image:')
        self.__t1PreVolumeSelector = slicer.qMRMLNodeComboBox()
        self.__t1PreVolumeSelector.toolTip = "Select a pre-contrast T1 image."
        self.__t1PreVolumeSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.__t1PreVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.__t1PreVolumeSelector.addEnabled = 0

        t1PostScanLabel = qt.QLabel('Post-Contrast T1 Image:')
        self.__t1PostVolumeSelector = slicer.qMRMLNodeComboBox()
        self.__t1PostVolumeSelector.toolTip = "Select a post-contrast T1 image."
        self.__t1PostVolumeSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.__t1PostVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.__t1PostVolumeSelector.addEnabled = 0

        t2ScanLabel = qt.QLabel('T2 Image:')
        self.__t2VolumeSelector = slicer.qMRMLNodeComboBox()
        self.__t2VolumeSelector.toolTip = "Select a T2 image."
        self.__t2VolumeSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.__t2VolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.__t2VolumeSelector.addEnabled = 0

        flairScanLabel = qt.QLabel('FLAIR Image:')
        self.__flairVolumeSelector = slicer.qMRMLNodeComboBox()
        self.__flairVolumeSelector.toolTip = "Select a FLAIR image."
        self.__flairVolumeSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.__flairVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.__flairVolumeSelector.addEnabled = 0

        self.__layout.addRow(self.__informationGroupBox)
        self.__informationGroupBoxLayout.addRow(step_label)

        self.__layout.addRow(self.__volumeSelectionGroupBox)
        self.__volumeSelectionGroupBoxLayout.addRow(t1PreScanLabel, self.__t1PreVolumeSelector)
        self.__volumeSelectionGroupBoxLayout.addRow(t1PostScanLabel, self.__t1PostVolumeSelector)
        self.__volumeSelectionGroupBoxLayout.addRow(t2ScanLabel, self.__t2VolumeSelector)
        self.__volumeSelectionGroupBoxLayout.addRow(flairScanLabel, self.__flairVolumeSelector)

        self.volumeNodes = [self.__t1PostVolumeSelector.currentNode(), self.__t1PostVolumeSelector.currentNode(), self.__t2VolumeSelector.currentNode(), self.__flairVolumeSelector.currentNode()]

        self.updateWidgetFromParameters(self.parameterNode())

        # This timer is a trick to wait for buttons to load BEFORE deleting them.
        qt.QTimer.singleShot(0, self.killButton)

    def validate(self, desiredBranchId):

        self.__parent.validate( desiredBranchId )

        pNode = self.parameterNode()

        # Temporary
        self.__parent.validationSucceeded(desiredBranchId)   
        return

        for volumeNode, volumeLabel in zip(self.volumeNodes, self.volumeLabels):
            if volumeNode is None:
                self.__parent.validationFailed(desiredBranchId, 'Error','Please select all four volumes to continue with the wizard.')
                break
            elif volumeNode.GetID() == '':
                self.__parent.validationFailed(desiredBranchId, 'Error','Please select a valid volume to threshold.')
                break
            else:
                volumeID = volumeNode.GetID()
                pNode.SetParameter(volumeLabel + 'ID', volumeID)
                pNode.SetParameter('original' + volumeLabel + 'ID', volumeID)
                self.__parent.validationSucceeded(desiredBranchId)    

    def killButton(self):

        # Find 'next' and 'back' buttons to control step flow in individual steps.
        stepButtons = slicer.util.findChildren(className='ctkPushButton')
        
        backButton = ''
        nextButton = ''
        for stepButton in stepButtons:
            if stepButton.text == 'Next':
                nextButton = stepButton
            if stepButton.text == 'Back':
                backButton = stepButton

        backButton.hide()

        # ctk creates an unwanted final page button. This method gets rid of it.
        bl = slicer.util.findChildren(text='ReviewStep')
        if len(bl):
          bl[0].hide()

    def onEntry(self, comingFrom, transitionType):

        super(VolumeSelectStep, self).onEntry(comingFrom, transitionType)

        self.updateWidgetFromParameters(self.parameterNode())

        pNode = self.parameterNode()
        pNode.SetParameter('currentStep', self.stepid)

        qt.QTimer.singleShot(0, self.killButton)

    def onExit(self, goingTo, transitionType):   

        super(GBMWizardStep, self).onExit(goingTo, transitionType) 

    def updateWidgetFromParameters(self, parameterNode):

        # Return if needed.
        pass