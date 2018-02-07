""" This is Step 2. The user has the option to register their pre- and post-contrast images
    using the module BRAINSFit. TO-DO: add a progress bar.
"""

from __main__ import qt, ctk, slicer

from GBMWizardStep import *
from Helper import *

""" PreprocessStep inherits from GBMWizardStep, with itself inherits
    from a ctk workflow class. 
"""

class PreprocessStep( GBMWizardStep ) :
    
    def __init__( self, stepid ):

        self.initialize( stepid )
        self.setName( '2. Preprocessing' )

        self.__parent = super( PreprocessStep, self )

        self.volumeLabels = ['t1Pre', 't1Post', 't2', 'flair']

    def createUserInterface( self ):
        
        """ This method uses qt to create a user interface of radio buttons to select
            a registration method. Note that BSpline registration is so slow and memory-consuming
            as to at one point break Slicer. There is an option to run it with limited memory,
            but this may take prohibitively long. <- NOTE this last comment was based on
            expert automated registration - not sure about other modules.
        """

        self.__layout = self.__parent.createUserInterface()

        step_label = qt.QLabel( """This step allows you to pre-process your data as necessary for deep learning segmentation. Your data may already be preprocessed, in which case you can skip this step. Note that for proper deep learning segmentation, your data will need to be A) registered, and B) resampled into isotropic space.
            """)
        step_label.setWordWrap(True)
        self.__informationGroupBox = qt.QGroupBox()
        self.__informationGroupBox.setTitle('Information')
        self.__informationGroupBoxLayout = qt.QFormLayout(self.__informationGroupBox)
        self.__informationGroupBoxLayout.addRow(step_label)
        self.__layout.addRow(self.__informationGroupBox)

        self.__registrationCollapsibleButton = ctk.ctkCollapsibleButton()
        self.__registrationCollapsibleButton.text = "Registration"
        self.__layout.addWidget(self.__registrationCollapsibleButton)
        self.__registrationLayout = qt.QFormLayout(self.__registrationCollapsibleButton)

        # Moving/Fixed Image Registration Order Options

        OrderGroupBox = qt.QGroupBox()
        OrderGroupBox.setTitle('Registration Base Volume')
        self.__registrationLayout.addRow(OrderGroupBox)

        OrderGroupBoxLayout = qt.QFormLayout(OrderGroupBox)

        self.__OrderRadio1 = qt.QRadioButton("Register to T2.")
        self.__OrderRadio1.toolTip = "Your images will be registered to T2 space."
        OrderGroupBoxLayout.addRow(self.__OrderRadio1)
        self.__OrderRadio1.setChecked(True)

        self.__OrderRadio2 = qt.QRadioButton("Register to FLAIR")
        self.__OrderRadio2.toolTip = "Your images will be registered to FLAIR space."
        OrderGroupBoxLayout.addRow(self.__OrderRadio2)

        self.__OrderRadio3 = qt.QRadioButton("Register to post-contrast T1")
        self.__OrderRadio3.toolTip = "Your images will be registered to post-contrast T1 space."
        OrderGroupBoxLayout.addRow(self.__OrderRadio3)

        self.__OrderRadio4 = qt.QRadioButton("Register to pre-contrast T1")
        self.__OrderRadio4.toolTip = "Your images will be registered to pre-contrast T1 space."
        OrderGroupBoxLayout.addRow(self.__OrderRadio4)

        self.__orderMapping = dict(zip(self.volumeLabels, [self.__OrderRadio1, self.__OrderRadio2, self.__OrderRadio3, self.__OrderRadio4]))

        # Registration Method Options

        RegistrationGroupBox = qt.QGroupBox()
        RegistrationGroupBox.setTitle('Registration Method')
        self.__registrationLayout.addRow(RegistrationGroupBox)

        RegistrationGroupBoxLayout = qt.QFormLayout(RegistrationGroupBox)

        self.__RegistrationRadio1 = qt.QRadioButton("Rigid Registration")
        self.__RegistrationRadio1.toolTip = """Computes a rigid registration on the pre-contrast image with respect to the post-contrast image. This will likely be the fastest registration method"""
        RegistrationGroupBoxLayout.addRow(self.__RegistrationRadio1)

        self.__RegistrationRadio2 = qt.QRadioButton("Affine Registration")
        self.__RegistrationRadio2.toolTip = "Computes a rigid and affine registration on the pre-contrast image with respect to the post-contrast image. This method may take longer than rigid registration, but has the ability to stretch or compress images in addition to rotation and translation."
        RegistrationGroupBoxLayout.addRow(self.__RegistrationRadio2)
        self.__RegistrationRadio2.setChecked(True)

        self.__RegistrationRadio3 = qt.QRadioButton("Deformable Registration")
        self.__RegistrationRadio3.toolTip = """Computes a BSpline Registration on the pre-contrast image with respect to the post-contrast image. This method is slowest and may be necessary for only severly distorted images."""
        RegistrationGroupBoxLayout.addRow(self.__RegistrationRadio3)

        # Output Volume Preference

        OutputGroupBox = qt.QGroupBox()
        OutputGroupBox.setTitle('Registration Output')
        self.__registrationLayout.addRow(OutputGroupBox)

        OutputGroupBoxLayout = qt.QFormLayout(OutputGroupBox)

        self.__OutputRadio1 = qt.QRadioButton("Create new volume.")
        self.__OutputRadio1.toolTip = "A new volume will be created with the naming convention \"[pre]_reg_[post]\"."
        OutputGroupBoxLayout.addRow(self.__OutputRadio1)
        self.__OutputRadio1.setChecked(True)

        self.__OutputRadio2 = qt.QRadioButton("Replace existing volume.")
        self.__OutputRadio2.toolTip = "Your registered volume will be overwritten at the end of this step."
        OutputGroupBoxLayout.addRow(self.__OutputRadio2)

        # Registration Button and Progress Indicator

        RunGroupBox = qt.QGroupBox()
        RunGroupBox.setTitle('Run Registration')
        self.__registrationLayout.addRow(RunGroupBox)

        RunGroupBoxLayout = qt.QFormLayout(RunGroupBox)

        self.__registrationButton = qt.QPushButton('Run registration')
        self.__registrationStatus = qt.QLabel('Register scans')
        self.__registrationStatus.alignment = 4 # This codes for centered alignment, although I'm not sure why.
        RunGroupBoxLayout.addRow(self.__registrationStatus)
        RunGroupBoxLayout.addRow(self.__registrationButton)
        self.__registrationButton.connect('clicked()', self.onRegistrationRequest)

    def killButton(self):

        # ctk creates an unwanted final page button. This method gets rid of it.
        bl = slicer.util.findChildren(text='ReviewStep')
        if len(bl):
            bl[0].hide()

    def validate(self, desiredBranchId):

        """ This checks to make sure you are not currently registering an image, and
            throws an exception if so.
        """

        self.__parent.validate( desiredBranchId )

        pNode = self.parameterNode()

        # Temporary
        self.__parent.validationSucceeded(desiredBranchId)   
        return

        if pNode.GetParameter('followupVolumeID') == '' or pNode.GetParameter('followupVolumeID') == None:
            self.__parent.validationSucceeded(desiredBranchId)
        else:   
            if self.__status == 'Uncalled':
                if self.__RegistrationRadio1.isChecked():
                    self.__parent.validationSucceeded(desiredBranchId)
                else:
                    self.__parent.validationFailed(desiredBranchId, 'Error','Please click \"Run Registration\" or select the \"No Registration\" option to continue.')
            elif self.__status == 'Completed':
                self.__parent.validationSucceeded(desiredBranchId)
            else:
                self.__parent.validationFailed(desiredBranchId, 'Error','Please wait until registration is completed.')

    def onEntry(self, comingFrom, transitionType):

        super(PreprocessStep, self).onEntry(comingFrom, transitionType)

        pNode = self.parameterNode()
        pNode.SetParameter('currentStep', self.stepid)

        # Helper.SetBgFgVolumes(pNode.GetParameter('baselineVolumeID'), pNode.GetParameter('followupVolumeID'))

        # A different attempt to get rid of the extra workflow button.
        qt.QTimer.singleShot(0, self.killButton)

    def onExit(self, goingTo, transitionType):

        super(GBMWizardStep, self).onExit(goingTo, transitionType) 

    def onRegistrationRequest(self, wait_for_completion=False):

        """ This method makes a call to a different slicer module, BRAINSFIT. 
            Note that this registration method computes a transform, which is 
            then applied to the followup volume in processRegistrationCompletion. 
            TO-DO: Add a cancel button and a progress bar.
        """

        pNode = self.parameterNode()

        # Determine the Fixed Volume
        for volumeLabel in self.volumeLabels:
            if self.__orderMapping[volumeLabel].isChecked():
                fixedLabel = volumeLabel
                fixedVolumeID = pNode.GetParameter(volumeLabel + 'ID')
                break

        fixedVolume = Helper.getNodeByID(fixedVolumeID)

        # TODO: Add Advanced Options Dropdown for these params.
        parameters = {}
        parameters["interpolationMode"] = 'Linear'
        parameters["initializeTransformMode"] = 'useMomentsAlign'
        parameters["samplingPercentage"] = .02

        for volumeLabel in self.volumeLabels:

            if volumeLabel == fixedLabel:
                continue

            movingVolume = Helper.getNodeByID(pNode.GetParameter(volumeLabel + 'ID'))

            # Registration Type Options.
            if self.__RegistrationRadio3.isChecked():
                BSplineTransform = slicer.vtkMRMLBSplineTransformNode()
                slicer.mrmlScene.AddNode(BSplineTransform)
                pNode.SetParameter(volumeLabel + 'RegistrationTransformID', BSplineTransform.GetID())
            else:
                LinearTransform = slicer.vtkMRMLLinearTransformNode()
                slicer.mrmlScene.AddNode(LinearTransform)
                pNode.SetParameter(volumeLabel + 'RegistrationTransformID', LinearTransform.GetID())

            if self.__RegistrationRadio1.isChecked():
                parameters['transformType'] = 'Rigid'
            elif self.__RegistrationRadio2.isChecked():
                parameters['transformType'] = 'Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine'
            elif self.__RegistrationRadio3.isChecked():
                parameters['transformType'] = 'BSpline'

            parameters["fixedVolume"] = fixedVolume
            parameters["movingVolume"] = movingVolume

            # Output options. TODO: Make this section a bit more logical.
            if self.__OutputRadio2.isChecked():
                parameters['outputVolume'] = movingVolume
                pNode.SetParameter(volumeLabel + 'RegistrationVolumeID', movingVolume.GetID())

            elif self.__OutputRadio1.isChecked():

                registrationID = pNode.GetParameter(volumeLabel + 'RegistrationVolumeID')
                
                if registrationID == None or registrationID == '':
                    registrationVolume = slicer.vtkMRMLScalarVolumeNode()
                    registrationVolume.SetScene(slicer.mrmlScene)
                    registrationVolume.SetName(movingVolume.GetName() + '_reg_' + fixedVolume.GetName())
                    slicer.mrmlScene.AddNode(registrationVolume)
                    pNode.SetParameter(volumeLabel + 'RegistrationVolumeID', registrationVolume.GetID())
                else:
                    registrationVolume = Helper.getNodeByID(registrationID)

                parameters['outputVolume'] = registrationVolume

            self.__cliNode = None
            self.__cliNode = slicer.cli.run(slicer.modules.brainsfit, self.__cliNode, parameters, wait_for_completion=wait_for_completion)

            # An event listener for the CLI. TODO: Add a progress bar.
            self.__cliObserverTag = self.__cliNode.AddObserver('ModifiedEvent', self.processRegistrationCompletion)
            self.__registrationStatus.setText('Wait ...')
            self.__registrationButton.setEnabled(0)

    def processRegistrationCompletion(self, node, event):

        """ This updates the registration button with the CLI module's convenient status
            indicator. Upon completion, it applies the transform to the followup node.
            Furthermore, it sets the followup node to be the baseline node in the viewer.
        """

        self.__status = node.GetStatusString()
        self.__registrationStatus.setText('Registration ' + self.__status)

        if self.__status == 'Completed':
            self.__registrationButton.setEnabled(1)

            pNode = self.parameterNode()

            if self.__OrderRadio1.isChecked():
                Helper.SetBgFgVolumes(pNode.GetParameter('followupVolumeID'), pNode.GetParameter('registrationVolumeID'))
            else:
                Helper.SetBgFgVolumes(pNode.GetParameter('registrationVolumeID'), pNode.GetParameter('baselineVolumeID'))

