""" This file is picked up by 3D Slicer and used to create a widget. GBMWizard
    (the class) specifies the Help and Acknowledgements qt box seen in Slicer.
    GBMWizardWidget start the main action of the module, creating a workflow
    from ctk and creating initial links to Slicer's MRML data. Most of this
    module is modeled after ChangeTracker by Andrey Fedorov, which can be found in
    the following GitHub repository: https://github.com/fedorov/ChangeTrackerPy

    vtk is a libary associated with image processing, ctk is a refined version of
    vtk meant specifically for medical imaging and is used here to create a
    step-by-step workflow, qt is a popular user interface library, and Slicer is a
    python library that helps developers hook into the 3D Slicer codebase.
    The program 3D Slicer has access to these libraries (and more), and is
    referenced here as __main__. GBMWizard_Lib is a folder that 
    contains the individual steps of the workflow and does most of the computational
    work. 

    This module is meant to create easy and effecient segmentations on high slice
    resolution medical images. It can calculate subtraction maps, register images,
    normalize images, create automatic 3D ROIs using Delaunay Triangulation, and 
    threshold intensities within an ROI.

    This module was made by Andrew Beers as part of QTIM and QICCR
    https://www.martinos.org/lab/qtim
    http://qiicr.org/

"""

from __main__ import vtk, qt, ctk, slicer
import os

import GBMWizard_Lib

class GBMWizard():

    def __init__(self, parent):

        """ This class specifies the Help + Acknowledgements section. One assumes
            that Slicer looks for a class with the same name as the file name. 
            Modifications to the parent result in modifications to the qt box that 
            then prints the relevant information.
        """
        parent.title = """GBM Wizard"""
        parent.categories = ["""GBM"""]
        parent.contributors = ["""Andrew Beers, QTIM @ MGH [https://qtim-lab.github.io/]"""]
        parent.helpText = """
        This module is meant to pre-process, skull-strip, segment and extract radiomic features from
        glioblastoma cases using DeepInfer, Docker, and deep learning.
        """;
        parent.acknowledgementText = """ This work was funded by the following grants: U24CA180918, U01CA154601, U24CA180927
        Module templated from ChangeTracker by Andrey Fedorov (BWH).
        """
        self.parent = parent
        self.collapsed = False

class GBMWizardWidget():

    def __init__(self, parent=None):
        """ It seems to be that Slicer creates an instance of this class with a
            qMRMLWidget parent. My understanding of parenthood when it comes to modules
            is currently limited -- I'll update this when I know more.
        """

        if not parent:
            self.parent = slicer.qMRMLWidget()
            self.parent.setLayout(qt.QVBoxLayout())
            self.parent.setMRMLScene(slicer.mrmlScene)
        else:
            self.parent = parent
            self.layout = self.parent.layout()

    def setup( self ):

        """ Slicer seems to call all methods of these classes upon entry. setup creates
            a workflow from ctk, which simply means that it creates a series of UI
            steps one can traverse with "next" / "previous" buttons. The steps themselves
            are contained within GBMWizard_Lib.
        """

        # Currently unclear on the difference between ctkWorkflow and
        # ctkWorkflowStackedWidget, but presumably the latter creates a UI
        # for the former.
        self.workflow = ctk.ctkWorkflow()
        workflowWidget = ctk.ctkWorkflowStackedWidget()
        workflowWidget.setWorkflow( self.workflow )

        # Create workflow steps.
        self.Step1 = GBMWizard_Lib.VolumeSelectStep('VolumeSelectStep')
        self.Step2 = GBMWizard_Lib.PreprocessStep('PreprocessStep')
        self.Step3 = GBMWizard_Lib.SkullStripStep('SkullStripStep')
        self.Step4 = GBMWizard_Lib.SegmentationStep('SegmentationStep')
        self.Step5 = GBMWizard_Lib.RadiomicsStep('RadiomicsStep')
        self.Step6 = GBMWizard_Lib.ReviewStep('ReviewStep')

        # Add the wizard steps to an array for convenience. Much of the following code
        # is copied wholesale from ChangeTracker.
        allSteps = []
        allSteps.append( self.Step1 )
        allSteps.append( self.Step2 )
        # allSteps.append( self.Step3 )
        allSteps.append( self.Step4 )
        allSteps.append( self.Step5 )
        allSteps.append( self.Step6 )

        # Adds transition functionality between steps.
        self.workflow.addTransition(self.Step1, self.Step2)
        self.workflow.addTransition(self.Step2, self.Step4)
        # self.workflow.addTransition(self.Step3, self.Step4)
        self.workflow.addTransition(self.Step4, self.Step5)
        self.workflow.addTransition(self.Step5, self.Step6)
        self.workflow.addTransition(self.Step6, self.Step1)

        """ The following code creates a 'parameter node' from the vtkMRMLScriptedModuleNode class. 
            A parameter node keeps track of module variables from step to step, in the case of
            ctkWorkflow, and when users leave the module to visit other modules. The code below
            searches to see if a parameter node already exists for GBMWizard among all
            available parameter nodes, and then creates one if it does not.
        """
        nNodes = slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLScriptedModuleNode')
        self.parameterNode = None
        for n in xrange(nNodes):
            compNode = slicer.mrmlScene.GetNthNodeByClass(n, 'vtkMRMLScriptedModuleNode')
            nodeid = None
            if compNode.GetModuleName() == 'GBMWizard':
                self.parameterNode = compNode
                # print 'Found existing GBMWizard parameter node'
                break
        if self.parameterNode == None:
            self.parameterNode = slicer.vtkMRMLScriptedModuleNode()
            self.parameterNode.SetModuleName('GBMWizard')
            slicer.mrmlScene.AddNode(self.parameterNode)

        # Individual workflow steps need to remember the parameter node too.
        for s in allSteps:
            s.setParameterNode(self.parameterNode)

        # Restores you to the correct step if you leave and then return to the module.
        currentStep = self.parameterNode.GetParameter('currentStep')
        if currentStep != '':
            # print 'Restoring GBMWizard workflow step to ', currentStep
            if currentStep == 'VolumeSelectStep':
                self.workflow.setInitialStep(self.Step1)
            if currentStep == 'PreprocessStep':
                self.workflow.setInitialStep(self.Step2)
            if currentStep == 'SkullStripStep':
                self.workflow.setInitialStep(self.Step3)
            if currentStep == 'SegmentationStep':
                self.workflow.setInitialStep(self.Step4)
            if currentStep == 'RadiomicsStep':
                self.workflow.setInitialStep(self.Step5)
            if currentStep == 'ReviewStep':
                self.workflow.setInitialStep(self.Step6)
        else:
            # print 'currentStep in parameter node is empty!'
            pass

        # Starts and show the workflow.
        self.workflow.start()
        workflowWidget.visible = True
        self.layout.addWidget( workflowWidget )

    def enter(self):
        """ A quick check to see if the file was loaded. Can be seen in the Python Interactor.
        """

        import GBMWizard_Lib
        print "Model GBM Module Correctly Entered"

        # test = GBMWizardTest()
        # test.runTest()
        pass

class GBMWizardTest():

    def delayDisplay(self,message,msec=1000):
        """This utility method displays a small dialog and waits.
        This does two things: 1) it lets the event loop catch up
        to the state of the test so that rendering and widget updates
        have all taken place before the test continues and 2) it
        shows the user/developer/tester the state of the test
        so that we'll know when it breaks.
        """
        print(message)
        self.info = qt.QDialog()
        self.infoLayout = qt.QVBoxLayout()
        self.info.setLayout(self.infoLayout)
        self.label = qt.QLabel(message,self.info)
        self.infoLayout.addWidget(self.label)
        qt.QTimer.singleShot(msec, self.info.close)
        self.info.exec_()

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """

        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.testGBMWizard()

    def testGBMWizard(self):
        """ Test the ChangeTracker module
        """
        self.delayDisplay("Starting the test")

        try:

            self.delayDisplay("Loading sample data")

            import SampleData
            sampleDataLogic = SampleData.SampleDataLogic()
            head = sampleDataLogic.downloadMRHead()
            braintumor1 = sampleDataLogic.downloadMRBrainTumor1()
            braintumor2 = sampleDataLogic.downloadMRBrainTumor2()

            self.delayDisplay("Getting scene variables")

            mainWindow = slicer.util.mainWindow()
            layoutManager = slicer.app.layoutManager()
            threeDView = layoutManager.threeDWidget(0).threeDView()
            redWidget = layoutManager.sliceWidget('Red')
            redController = redWidget.sliceController()
            viewNode = threeDView.mrmlViewNode()
            cameras = slicer.util.getNodes('vtkMRMLCameraNode*')

            mainWindow.moduleSelector().selectModule('GBMWizard')
            modelsegmentation_module = slicer.modules.modelsegmentation.widgetRepresentation().self()

            # self.delayDisplay('Select Volumes')
            # baselineNode = braintumor1
            # followupNode = braintumor2
            # modelsegmentation_module.Step1._VolumeSelectStep__enableSubtractionMapping.setChecked(True)
            # modelsegmentation_module.Step1._VolumeSelectStep__baselineVolumeSelector.setCurrentNode(baselineNode)
            # modelsegmentation_module.Step1._VolumeSelectStep__followupVolumeSelector.setCurrentNode(followupNode)

            # self.delayDisplay('Go Forward')
            # modelsegmentation_module.workflow.goForward()

            # self.delayDisplay('Register Images')
            # modelsegmentation_module.Step2.onRegistrationRequest(wait_for_completion=True)

            # self.delayDisplay('Go Forward')
            # modelsegmentation_module.workflow.goForward()

            # self.delayDisplay('Normalize Images')
            # modelsegmentation_module.Step3.onGaussianNormalizationRequest()

            # self.delayDisplay('Subtract Images')
            # modelsegmentation_module.Step3.onSubtractionRequest(wait_for_completion=True)

            # self.delayDisplay('Go Forward')
            # modelsegmentation_module.workflow.goForward()

            # self.delayDisplay('Load model')

            # displayNode = slicer.vtkMRMLMarkupsDisplayNode()
            # slicer.mrmlScene.AddNode(displayNode)
            # inputMarkup = slicer.vtkMRMLMarkupsFiducialNode()
            # inputMarkup.SetName('Test')
            # slicer.mrmlScene.AddNode(inputMarkup)
            # inputMarkup.SetAndObserveDisplayNodeID(displayNode.GetID())

            # modelsegmentation_module.Step4._ROIStep__clippingMarkupSelector.setCurrentNode(inputMarkup)

            # inputMarkup.AddFiducial(35,-10,-10)
            # inputMarkup.AddFiducial(-15,20,-10)
            # inputMarkup.AddFiducial(-25,-25,-10)
            # inputMarkup.AddFiducial(-5,-60,-15)
            # inputMarkup.AddFiducial(-5,5,60)
            # inputMarkup.AddFiducial(-5,-35,-30)

            # self.delayDisplay('Go Forward')
            # modelsegmentation_module.workflow.goForward()

            # self.delayDisplay('Set Thresholds')
            # modelsegmentation_module.Step5._ThresholdStep__threshRange.minimumValue = 50
            # modelsegmentation_module.Step5._ThresholdStep__threshRange.maximumValue = 150

            # self.delayDisplay('Go Forward')
            # modelsegmentation_module.workflow.goForward()

            # self.delayDisplay('Restart Module')
            # modelsegmentation_module.Step6.Restart()

            # self.delayDisplay('Test passed!')
            
        except Exception, e:
            import traceback
            traceback.print_exc()
            self.delayDisplay('Test caused exception!\n' + str(e))