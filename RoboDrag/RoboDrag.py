import logging
import os, time
from typing import Annotated, Optional
import xml.etree.ElementTree as ET
import math
import qt

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)





from slicer import vtkMRMLScalarVolumeNode


#
# RoboDrag
#


class RoboDrag(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("RoboDrag")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#RoboDrag">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # RoboDrag1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="RoboDrag",
        sampleName="RoboDrag1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "RoboDrag1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="RoboDrag1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="RoboDrag1",
    )

    # RoboDrag2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="RoboDrag",
        sampleName="RoboDrag2",
        thumbnailFileName=os.path.join(iconsPath, "RoboDrag2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="RoboDrag2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="RoboDrag2",
    )


#
# RoboDragParameterNode
#


@parameterNodeWrapper
class RoboDragParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# RoboDragWidget
#


class RoboDragWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.fromtransform = None
        self.totransform = None
        self.robot = None
        self.jointPositionsRad = []
        self.rootlink = None
        self.tiplink = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/RoboDrag.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = RoboDragLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.loadpushbutton.connect("clicked(bool)", self.onloadbutton)
        self.ui.streamstartbutton.connect("clicked(bool)", self.onstreamstartbutton)
        self.ui.streamstopbutton.connect("clicked(bool)", self.onstreamstopbutton)
        self.ui.opacitypushButton.connect("clicked(bool)", self.onopacitybutton)
        self.ui.robotColorButton.connect("colorChanged(QColor)", self.onRobotColorChanged)
        self.ui.zeropushButton.connect("clicked(bool)", self.onzerobutton)
        self.ui.checkBox.connect("toggled(bool)", self.onMoveGroupToggled)
                
        # Set appearence collapsible button to be collapsed and disabled initially
        self.ui.appCollapsibleButton.collapsed = True
        self.ui.appCollapsibleButton.enabled = False
        
        # Set checkbox for move group option. If toggled, enable moveCollapsibleButton and uncollapse
        # if toggled off, disable moveCollapsibleButton and collapse
        self.ui.moveCollapsibleButton.collapsed = True
        self.ui.moveCollapsibleButton.enabled = False
        self.ui.checkBox.enabled = False
        
        
        
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        # Stop streaming and remove observers before cleanup
        if self.logic:
            self.logic.removeObserver()
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Stop streaming when exiting module
        if self.logic:
            self.logic.removeObserver()
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[RoboDragParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)
   
    def onloadbutton(self) -> None:
            # Stop any prior streaming callbacks before we touch transforms
            self.logic.removeObserver()
                
            # 1. Get robot node
            robotNode = self.ui.ikrobotcombobox.currentNode()
            if not robotNode:
                print("Error: No robot selected.")
                return

            # 2. Extract URDF XML
            pnode = robotNode.GetNthNodeReference("parameter", 0)
            if not pnode:
                print("Error: No parameter node found for robot.")
                return
            urdf_xml = pnode.GetParameterAsString("robot_description")

            # 3. Auto-detect Root and Tip Links
            root_link, tip_link = self.logic.find_root_and_deepest_tip(urdf_xml)
            joint_names = self.logic.parse_joint_names_from_urdf(urdf_xml)
            limits = self.logic.parse_joint_limits_from_urdf(urdf_xml)
            
            # Store for later use
            self.rootlink = root_link
            self.tiplink = tip_link
            self.logic.joint_names = joint_names
            self.logic.last_ik_solution = [0.0] * len(joint_names)
            self.jointPositionsRad = [0.0] * len(joint_names)
            
            # Check if links were found
            if not root_link or not tip_link:
                print("Error: Could not auto-detect kinematic chain from URDF.")
                return
            print(f"Auto-detected Chain for IK: {root_link} -> {tip_link}")
            
            # 4 Find Slicer Transforms for Root and Tip
            try:
                root_tf_node, tip_tf_node = self.logic.findRobotTransforms(root_link, tip_link)
                self.totransform = root_tf_node  # This is our reference frame for IK (base)
            except RuntimeError as e:
                print(f"Error locating robot transforms: {e}")
                return
            
            # 5 Handle Probe Sphere
            # Check if sphere model already exists, if not create model and store from transform
            try:
                model = slicer.util.getNode("ProbeSphere")
            except slicer.util.MRMLNodeNotFoundException:
                model = None

            if model is not None:
                print("Sphere model already exists.")
                self.fromtransform = slicer.util.getNode("ProbeSphere_Transform")
            else:
                print("Creating sphere model...")
                model = self.logic.createSphereModel()
                self.fromtransform = self.logic.createLinearTransform()
                self.logic.applyTransformToModel(model, self.fromtransform)

            # 7. Call IK Setup with New Arguments
            try:
                self.robot = self.logic.setupikforRobot(
                    root_link=root_link, 
                    tip_link=tip_link, 
                    robotNode=robotNode, 
                    fromtransformname=self.fromtransform.GetName(), 
                    totransformname=self.totransform.GetName()
                )
            except RuntimeError as e:
                print(f"IK Setup Failed: {e}")
                return

            # 8. Enable buttons
            self.ui.zeropushButton.enabled = True
            self.ui.streamstartbutton.enabled = True
            self.ui.streamstopbutton.enabled = True
            self.ui.appCollapsibleButton.collapsed = False
            self.ui.appCollapsibleButton.enabled = True
            self.ui.checkBox.enabled = True
            
            # 9. Create Joint Sliders Dynamically
            container = self.ui.Jointtab.layout()
            if container is not None:
                # FIX: Iterate backwards to delete dynamic items but KEEP the zero button
                for i in reversed(range(container.count())):
                    item = container.itemAt(i)
                    widget = item.widget()
                    
                    # If this is your specific button, skip it!
                    if widget == self.ui.zeropushButton:
                        continue
                        
                    # Otherwise, remove from layout and destroy
                    if widget is not None:
                        container.takeAt(i)
                        widget.deleteLater()

                # Create sliders dynamically
                for i, joint_name in enumerate(joint_names):
                    joint_label = qt.QLabel(joint_name)
                    joint_slider = qt.QSlider(qt.Qt.Horizontal)

                    # Apply limits
                    lo_hi = limits.get(joint_name)
                    if lo_hi:
                        lo_deg = int(round(math.degrees(lo_hi[0])))
                        hi_deg = int(round(math.degrees(lo_hi[1])))
                        if lo_deg > hi_deg:
                            lo_deg, hi_deg = hi_deg, lo_deg
                    else:
                        lo_deg, hi_deg = -180, 180

                    joint_slider.setMinimum(lo_deg)
                    joint_slider.setMaximum(hi_deg)
                    joint_slider.setValue(0)
                    joint_slider.setTickInterval(10)
                    joint_slider.setTickPosition(qt.QSlider.TicksBelow)

                    # Connect slider change
                    joint_slider.valueChanged.connect(lambda value, idx=i: self.onJointSliderChanged(idx, value))

                    # Note: addWidget appends to the end. 
                    # If you want sliders BEFORE the button, use container.insertWidget(position, widget)
                    container.addWidget(joint_label)
                    container.addWidget(joint_slider)
            
    # def onloadbutton(self) -> None:
        
    #     # Get robot node
    #     robotNode = self.ui.ikrobotcombobox.currentNode()
        
    #     # check if sphere model already exists (slicer.util.getNode raises if not found)
    #     try:
    #         model = slicer.util.getNode("ProbeSphere")
    #     except slicer.util.MRMLNodeNotFoundException:
    #         model = None

    #     if model is not None:
    #         print("Sphere model already exists.")
    #         # Get linear transform into self.fromtransform
    #         self.fromtransform = slicer.util.getNode("ProbeSphere_Transform")
            
    #         # Reset to zero position
    #         identity_matrix = vtk.vtkMatrix4x4()
    #         identity_matrix.Identity()
    #         self.fromtransform.SetMatrixTransformToParent(identity_matrix)
    #         print("Reset sphere transform to identity.")
            
    #         # Get ros2 root into self.totransform
    #         self.totransform = self.logic.findRos2Root()
    #     else:
    #         print("Creating sphere model...")
    #         model = self.logic.createSphereModel()
    #         self.fromtransform = self.logic.createLinearTransform()

    #         self.logic.applyTransformToModel(model, self.fromtransform)
    #         self.totransform = self.logic.findRos2Root()
    #         print(f"Found ros2 root transform: {self.totransform.GetName()}")
        
    #     ikgroup = self.ui.grouplineEdit.text
    #     self.robot = self.logic.setupikforRobot(group=ikgroup, robotNode=robotNode, fromtransformname=self.fromtransform.GetName(), totransformname=self.totransform.GetName())

    #     # Parse URDF to get joint names
    #     # Get URDF XML from robot parameter node
    #     pnode = robotNode.GetNthNodeReference("parameter", 0)
    #     urdf_xml = pnode.GetParameterAsString("robot_description")
        
    #     # Parse joint names and store in logic
    #     joint_names = self.logic.parse_joint_names_from_urdf(urdf_xml)
    #     self.logic.joint_names = joint_names  # Store for publishing
    #     limits = self.logic.parse_joint_limits_from_urdf(urdf_xml)

    #     self.ui.jointCollapsibleButton.collapsed = False        
    #     self.ui.jointCollapsibleButton.enabled = True
        
    #     container = self.ui.jointScrollContents.layout()
        
    #     if container is not None:
    #         # Clear existing widgets
    #         while container.count():
    #             item = container.takeAt(0)
    #             widget = item.widget()
    #             if widget is not None:
    #                 widget.deleteLater()

    #         # Create sliders dynamically based on joint names
    #         for i, joint_name in enumerate(joint_names):
    #             joint_label = qt.QLabel(joint_name)
    #             joint_slider = qt.QSlider(qt.Qt.Horizontal)

    #             # Apply limits from URDF (convert radians to degrees)
    #             lo_hi = limits.get(joint_name)
    #             if lo_hi:
    #                 lo_deg = int(round(math.degrees(lo_hi[0])))
    #                 hi_deg = int(round(math.degrees(lo_hi[1])))
    #                 if lo_deg > hi_deg:
    #                     lo_deg, hi_deg = hi_deg, lo_deg
    #             else:
    #                 lo_deg, hi_deg = -180, 180

    #             joint_slider.setMinimum(lo_deg)
    #             joint_slider.setMaximum(hi_deg)
    #             joint_slider.setValue(0)
    #             joint_slider.setTickInterval(10)
    #             joint_slider.setTickPosition(qt.QSlider.TicksBelow)

    #             # Connect slider change to handler
    #             joint_slider.valueChanged.connect(lambda value, idx=i: self.onJointSliderChanged(idx, value))

    #             # Add to layout
    #             container.addWidget(joint_label)
    #             container.addWidget(joint_slider)
            
    #         # Resize jointPositionsRad to match actual joint count
    #         self.jointPositionsRad = [0.0] * len(joint_names)
        
        
    # def onstreamstartbutton(self) -> None:
    #         robotmodel = self.robot
    #         ikgroup = self.ui.grouplineEdit.text
            
    #         if robotmodel is None:
    #             print("Robot model is not set up.")
    #             return

    #         # 1. Find the deepest model (e.g. "tool0_model")
    #         deepest_model = self.logic.findDeepestModel()
            
    #         ee_name = ""
            
    #         if deepest_model:
    #             raw_model_name = deepest_model.GetName()
                
    #             # 2. Derive Link Name: Simply strip "_model"
    #             # "tool0_model" -> "tool0"
    #             if raw_model_name.endswith("_model"):
    #                 ee_name = raw_model_name[:-6]
    #             else:
    #                 ee_name = raw_model_name
                    
    #             print(f"Deepest Model: {raw_model_name}")
    #             print(f"End-Effector Target: {ee_name}")

    #         else:
    #             print("Could not find any robot models.")
    #             return

    #         # 3. Start Streaming
    #         self.logic.addObserverComputeIK(ee_name, robotmodel, ikgroup)

    def onstreamstartbutton(self) -> None:
            robotmodel = self.robot
            
            if robotmodel is None:
                print("Robot model is not set up.")
                return

            # 3. Start Streaming
            self.logic.addObserverComputeIK(robotmodel)
        
    def onstreamstopbutton(self) -> None:
        self.logic.removeObserver()
        print("Stopped streaming.")

        
    def onopacitybutton(self) -> None:
        opacity = self.ui.spinBox.value / 100.0
        robot = self.ui.robotcomboBox.currentNode()
        self.logic.setopacity(robot, opacity)
        
    def onRobotColorChanged(self) -> None:
        color = self.ui.robotColorButton.color    
        robotNode = self.ui.robotcomboBox.currentNode()
        self.logic.setRobotColor(robotNode, color)
        
    def onJointSliderChanged(self, idx: int, valueDeg: int) -> None:
        # Ensure array is large enough
        while len(self.jointPositionsRad) <= idx:
            self.jointPositionsRad.append(0.0)
        
        # store as radians
        self.jointPositionsRad[idx] = math.radians(valueDeg)

        # publish all joint values
        if self.logic is not None and self.logic.joint_state_publisher is not None:
            self.logic._publish_joint_state(self.jointPositionsRad)
        
        print(f"All joint values (rad): {[f'{j:.4f}' for j in self.jointPositionsRad]}")
        
    def onzerobutton(self) -> None:
        
        print("Resetting joint sliders to zero.")

        container = self.ui.Jointtab.layout()
        if container is None:
            return

        sliders = []
        for i in range(container.count()):
            widget = container.itemAt(i).widget()
            if isinstance(widget, qt.QSlider):
                sliders.append(widget)

        if not sliders:
            return

        for s in sliders:
            s.blockSignals(True)
            s.setValue(0)
            s.blockSignals(False)

        # Reset stored joint positions to match slider count
        self.jointPositionsRad = [0.0] * len(sliders)

        # Publish zero positions
        if self.logic is not None and self.logic.joint_state_publisher is not None:
            self.logic._publish_joint_state(self.jointPositionsRad)

    def onMoveGroupToggled(self, toggled: bool) -> None:
        if toggled:
            self.ui.moveCollapsibleButton.collapsed = False
            self.ui.moveCollapsibleButton.enabled = True
        else:
            self.ui.moveCollapsibleButton.collapsed = True
            self.ui.moveCollapsibleButton.enabled = False

#
# RoboDragLogic
#


class RoboDragLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.obsTag = None
        self.obsNode = None
        self.callback = None  
        self.toNode = None
        self.plangroup = None
        self.last_ik_solution = []  # Will be sized based on actual joint count
        self.joint_names = []  # Will be populated from URDF
        self.joint_state_publisher = None
        
    def getParameterNode(self):
        return RoboDragParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")
        
    
    def createSphereModel(self, name="ProbeSphere", radius_mm=20.0):
        r = radius_mm
        src = vtk.vtkSphereSource()
        src.SetRadius(r); src.SetThetaResolution(40); src.SetPhiResolution(40); src.Update()

        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        model.SetAndObservePolyData(src.GetOutput())

        disp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode", name+"_Display")
        disp.SetOpacity(0.6); disp.SetBackfaceCulling(0); disp.SetVisibility3D(True)
        disp.SetColor(0.9,0.3,0.3)
        model.SetAndObserveDisplayNodeID(disp.GetID())
        return model
    
    def createLinearTransform(self, name="ProbeSphere_Transform", showAxes=True):
        t = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)
        if not t.GetDisplayNode():
            tdisp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformDisplayNode", name+"_Display")
            tdisp.SetVisibility(True)  # “eye” in Data
            t.SetAndObserveDisplayNodeID(tdisp.GetID())
        if showAxes:
            t.GetDisplayNode().SetEditorVisibility(True)  # show 3D gizmo on selection
        return t

    def applyTransformToModel(self, modelNode, transformNode):
        """
        Parent a model under a transform (no hardening).
        The model will move with the transform and stay linked.
        """
        if modelNode is None or transformNode is None:
            raise ValueError("modelNode and transformNode are required")

        # Link the model to the transform in the MRML hierarchy
        modelNode.SetAndObserveTransformNodeID(transformNode.GetID())

        # Nudge MRML/3D view to update
        modelNode.Modified()
        

    def _allTransforms(self):
        s = slicer.mrmlScene
        return [s.GetNthNodeByClass(i, "vtkMRMLTransformNode")
                for i in range(s.GetNumberOfNodesByClass("vtkMRMLTransformNode"))]

    # def findLeafRobotTransform(self,prefix="ros2:tf2lookup:"):
    #     """
    #     Return the leaf (last) transform whose name starts with `prefix`.
    #     If multiple leaves exist, prefer names containing flange/tool/ee/end/specholder.
    #     """
    #     ts = [t for t in self._allTransforms() if t.GetName().startswith(prefix)]
    #     if not ts:
    #         raise RuntimeError(f"No transforms found with prefix '{prefix}'.")

    #     subset = set(ts)
    #     children = {t: [] for t in ts}
    #     for t in ts:
    #         p = t.GetParentTransformNode()
    #         if p in subset:
    #             children[p].append(t)

    #     # leaves = no children in this subset
    #     leaves = [t for t in ts if len(children[t]) == 0]
    #     if not leaves:
    #         # degenerate case: single node with self? fallback to deepest by ancestry
    #         leaves = ts

    #     if len(leaves) == 1:
    #         return leaves[0]

    #     # prefer typical end-effector names
    #     prefs = ("specholder", "flange", "tool", "ee", "end")
    #     lname = [t.GetName().lower() for t in leaves]
    #     for kw in prefs:
    #         for i, n in enumerate(lname):
    #             if kw in n:
    #                 return leaves[i]

    #     # fallback: deepest node by chain length
    #     def depth(t):
    #         d = 0
    #         cur = t
    #         while cur.GetParentTransformNode() in subset:
    #             cur = cur.GetParentTransformNode()
    #             d += 1
    #         return d
    #     leaves.sort(key=depth, reverse=True)
    #     return leaves[0]
    
    def find_root_and_deepest_tip(self, urdf_string):
            
            root_xml = ET.fromstring(urdf_string)

            # 1. Build TWO trees:
            #    - actuated_tree: for finding the Root (ignores fixed joints)
            #    - full_tree: for finding the Tip (includes fixed joints)
            actuated_tree = {}
            full_tree = {}
            
            all_child_links_actuated = set()

            for link in root_xml.findall("link"):
                name = link.get("name")
                if name:
                    actuated_tree.setdefault(name, [])
                    full_tree.setdefault(name, [])

            def is_actuated_joint(joint_elem):
                jtype = joint_elem.get("type", "")
                return jtype not in ("fixed", "")

            for joint in root_xml.findall("joint"):
                parent = joint.find("parent").get("link")
                child  = joint.find("child").get("link")
                if not parent or not child:
                    continue

                # Always add to full tree (for finding tip)
                full_tree.setdefault(parent, []).append(child)

                # Only add to actuated tree if movable (for finding root)
                if is_actuated_joint(joint):
                    actuated_tree.setdefault(parent, []).append(child)
                    all_child_links_actuated.add(child)

            # 2. Find Root using Actuated Tree
            # Candidates are nodes in actuated_tree that are never children (in the actuated graph)
            # This effectively skips 'world' and finds 'base_link'
            candidate_roots = list(set(actuated_tree.keys()) - all_child_links_actuated)
            
            # Filter: Must have at least one outgoing actuated joint
            candidate_roots = [r for r in candidate_roots if len(actuated_tree.get(r, [])) > 0]

            if not candidate_roots:
                return None, None

            # Pick root with most reachable actuated joints
            def reachable_count(start):
                visited = set()
                stack = [start]
                while stack:
                    n = stack.pop()
                    if n in visited: continue
                    visited.add(n)
                    stack.extend(actuated_tree.get(n, []))
                return len(visited)

            root_link = max(candidate_roots, key=reachable_count)

            # 3. Find Deepest Tip using FULL Tree
            # Now that we know where the arm starts (root_link), we walk down EVERYTHING
            # to find the furthest tool tip (even if attached via fixed joint).
            max_depth = -1
            deepest_tip = root_link
            
            # BFS using full_tree
            queue = [(root_link, 0)]
            while queue:
                node, depth = queue.pop(0)
                kids = full_tree.get(node, [])
                
                if not kids:
                    # Leaf node in full tree
                    if depth > max_depth:
                        max_depth = depth
                        deepest_tip = node
                    elif depth == max_depth:
                        # Tie-breaker: prefer standard ROS names
                        if "tool" in node or "flange" in node or "ee" in node:
                            deepest_tip = node
                else:
                    for k in kids:
                        queue.append((k, depth + 1))

            return root_link, deepest_tip

    def attachProbeTransformUnderLeaf(self, probeTransformName="ProbeSphere_Transform",
                                    prefix="ros2:tf2lookup:"):
        """
        Find robot leaf transform and parent `probeTransformName` under it.
        Creates a display node for the probe transform if needed.
        """
        # 1) find (or get) the probe transform node
        probeT = slicer.util.getNode(probeTransformName)
        if probeT is None:
            # create if missing
            probeT = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", probeTransformName)
        if not probeT.GetDisplayNode():
            d = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformDisplayNode", probeTransformName + "_Display")
            d.SetVisibility(True)
            probeT.SetAndObserveDisplayNodeID(d.GetID())

        # 2) find robot leaf
        leaf = self.findLeafRobotTransform(prefix=prefix)

        # 3) parent probe under the leaf
        probeT.SetAndObserveTransformNodeID(leaf.GetID())

        print(f"Attached '{probeTransformName}' under leaf transform '{leaf.GetName()}'")
        return dict(leafTransform=leaf, probeTransform=probeT)

    # def findRos2Root(self, prefix="ros2:tf2lookup:"):
    #     s = slicer.mrmlScene
    #     ts = [s.GetNthNodeByClass(i,"vtkMRMLTransformNode")
    #         for i in range(s.GetNumberOfNodesByClass("vtkMRMLTransformNode"))]
    #     ts = [t for t in ts if t.GetName().startswith(prefix)]
    #     if not ts:
    #         raise RuntimeError(f"No transforms with prefix '{prefix}' found.")
    #     subset = set(ts)
    #     roots = [t for t in ts if t.GetParentTransformNode() not in subset]
    #     return roots[0]  # the top-most (your “first ros2 tf lookup”)
    
    
    def findRobotTransforms(self, root_link_name, tip_link_name):
            """
            Locates the Slicer Transform nodes driving the robot by looking for 
            the visual models (e.g., 'base_link_model') and getting their parents.
            """
            
            def _get_parent_transform_of_model(link_name):
                # 1. Construct expected model name
                # SlicerROS2 usually appends '_model' to the link name
                model_name = f"{link_name}_model"
                
                # 2. Try to find the model node
                try:
                    model_node = slicer.util.getNode(model_name)
                except slicer.util.MRMLNodeNotFoundException:
                    model_node = None
                    
                # 3. If found, return its parent transform
                if model_node:
                    parent = model_node.GetParentTransformNode()
                    if parent:
                        print(f"Found transform for '{link_name}' via model '{model_name}': {parent.GetName()}")
                        return parent
                    else:
                        print(f"Warning: Model '{model_name}' exists but is not parented under a transform.")
                
                # 4. Fallback: Try finding the transform directly (e.g. for 'world' which has no mesh)
                # Common SlicerROS2 prefix
                fallback_name = f"ros2:tf2lookup:{link_name}"
                try:
                    tf_node = slicer.util.getNode(fallback_name)
                    print(f"Found transform directly: {fallback_name}")
                    return tf_node
                except slicer.util.MRMLNodeNotFoundException:
                    pass

                return None

            # --- Execution ---
            root_transform = _get_parent_transform_of_model(root_link_name)
            tip_transform = _get_parent_transform_of_model(tip_link_name)

            if not root_transform:
                raise RuntimeError(f"Could not find Transform for root link '{root_link_name}'")
            if not tip_transform:
                print(f"Warning: Could not find Transform for tip link '{tip_link_name}'")
                
            return root_transform, tip_transform

    def addObserver(self, fromTransformName, toTransformName):
        """
        Observe 'fromTransformName' and print its XYZ (origin) w.r.t. 'toTransformName'
        whenever the FROM transform is modified.
        """
        fromNode = slicer.util.getNode(fromTransformName)
        toNode   = slicer.util.getNode(toTransformName)
        
        if fromNode is None or toNode is None:
            raise RuntimeError("Transform nodes not found in scene.")

        # Print once initially
        self.printLocation(fromNode, toNode)

        # Remove a previous observer if any
        self.removeObserver()

        # Define the callback and keep a reference to it
        def onModified(caller, eventId):
            self.printLocation(fromNode, toNode)

        self.callback = onModified
        # Prefer TransformModifiedEvent for transforms; ModifiedEvent also works
        eventId = slicer.vtkMRMLTransformNode.TransformModifiedEvent
        self.obsTag  = fromNode.AddObserver(eventId, self.callback)
        self.obsNode = fromNode
        self.toNode = toNode
        return self.obsTag
        
    def printLocation(self, fromNode, toNode):
        m = vtk.vtkMatrix4x4()
        ok = slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(fromNode, toNode, m)
        if not ok:
            print("Could not compute transform between nodes.")
            return
        x, y, z = m.GetElement(0, 3), m.GetElement(1, 3), m.GetElement(2, 3)
        print(f"{fromNode.GetName()} wrt {toNode.GetName()}: x={x:.2f}, y={y:.2f}, z={z:.2f} mm")

    def removeObserver(self):
        if self.obsNode and self.obsTag is not None:
            try:
                self.obsNode.RemoveObserver(self.obsTag)
                print(f"[RoboDragLogic] Removed observer (tag={self.obsTag}) from {self.obsNode.GetName()}")
            except Exception as e:
                print(f"[RoboDragLogic] Error removing observer: {e}")
        else:
            if self.obsTag is not None:
                print(f"[RoboDragLogic] No obsNode to remove observer from (tag={self.obsTag})")
        self.obsNode = None
        self.obsTag = None
        self.callback = None

    def getOrReusePublisher(self, topic="/ghost/joint_states"):
        pubs = slicer.util.getNodesByClass("vtkMRMLROS2PublisherNode")
        # getNodesByClass might be list or dict depending on Slicer version
        if isinstance(pubs, dict):
            pubs = list(pubs.values())

        for p in pubs:
            # method name varies by build; try both
            if hasattr(p, "GetTopicName") and p.GetTopicName() == topic:
                print("Reusing existing publisher for topic:", topic)
                return p
            if hasattr(p, "GetTopic") and p.GetTopic() == topic:
                print("Reusing existing publisher for topic:", topic)
                return p

        print("Creating new publisher for topic:", topic)
        rosLogic = slicer.util.getModuleLogic("ROS2")
        rosNode = rosLogic.GetDefaultROS2Node()
        
        return rosNode.CreateAndAddPublisherNode("JointState", topic)
        
    # def setupikforRobot(self, group, robotNode, fromtransformname, totransformname):
    #     if group:
    #         self.plangroup = group
    #         print(f"Set IK group to '{group}'")
    #     else:
    #         raise RuntimeError("IK group name is required.")
        
    #     fromNode = slicer.util.getNode(fromtransformname)
    #     toNode   = slicer.util.getNode(totransformname)
        
    #     if fromNode is None or toNode is None:
    #         raise RuntimeError("Transform nodes not found.")
    #     else:
    #         self.obsNode = fromNode
    #         self.toNode = toNode
    #         print(f"Nodes set for IK: from '{fromtransformname}' to '{totransformname}'")
        
    #     bool = robotNode.setupIK(group)
        
    #     self.joint_state_publisher = self.getOrReusePublisher("/ghost/joint_states")
        
    #     # Initialize IK solution seed with correct size
    #     if self.joint_names:
    #         self.last_ik_solution = [0.0] * len(self.joint_names)
        
    #     if bool:
    #         print(f"IK setup for robot '{robotNode.GetName()}' and group '{group}' successful.")
    #         return robotNode
    #     else:
    #         raise RuntimeError(f"IK setup for robot '{robotNode.GetName()}' and group '{group}' failed.")
    
    
    def setupikforRobot(self, root_link, tip_link, robotNode, fromtransformname, totransformname):
        """
        Sets up KDL IK for the robot.
        Args:
            root_link (str): The name of the base link (e.g., 'base_link')
            tip_link (str): The name of the end-effector link (e.g., 'tool0')
            robotNode: The vtkMRMLROS2RobotNode instance
            fromtransformname: Name of the transform node moving the robot
            totransformname: Name of the reference transform node
        """
        if not root_link or not tip_link:
             raise RuntimeError("Both Root Link and Tip Link names are required for KDL.")

        # Setup internal node references for the dragging logic
        fromNode = slicer.util.getNode(fromtransformname)
        toNode   = slicer.util.getNode(totransformname)
        
        if fromNode is None or toNode is None:
            raise RuntimeError("Transform nodes not found.")
        else:
            self.obsNode = fromNode
            self.toNode = toNode
            print(f"Nodes set for IK: from '{fromtransformname}' to '{totransformname}'")
        
        # Call the new C++ KDL method instead of the old MoveIt setupIK
        # This returns True if the chain was successfully built from URDF
        success = robotNode.setupKDLIKWithLimits(root_link, tip_link)
        # ----------------------
        
        self.joint_state_publisher = self.getOrReusePublisher("/ghost/joint_states")
        
        # Initialize IK solution seed
        # Note: Ensure self.joint_names is populated before this call 
        # (usually done by parsing URDF)
        if hasattr(self, 'joint_names') and self.joint_names:
            self.last_ik_solution = [0.0] * len(self.joint_names)
        
        if success:
            print(f"KDL IK setup for robot '{robotNode.GetName()}' (Chain: {root_link} -> {tip_link}) successful.")
            return robotNode
        else:
            raise RuntimeError(f"KDL IK setup failed. Could not build chain from '{root_link}' to '{tip_link}'.")
        
    # def compute_ik_once(self, endeffectorname: str, robotmodel=None, ikgroup=None):

    #     # --- Get Slicer transform nodes ---
    #     fromNode = self.obsNode
    #     toNode   = self.toNode

    #     # --- Compute 4×4 transform between nodes ---
    #     targetPose = vtk.vtkMatrix4x4()
    #     if not slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(fromNode, toNode, targetPose):
    #         raise RuntimeError("Could not compute transform between nodes.")
        
    #     seed = self.last_ik_solution
    #     result_str = robotmodel.FindIK(ikgroup, targetPose, endeffectorname, seed, 0.05)

    #     if result_str and result_str.strip():
    #         # Parse comma-separated string into list of floats
    #         try:
    #             data = [float(x) for x in result_str.split(",")]

    #             print(f"[IK] Joint Solution: {data}")
    #             self.last_ik_solution = data
    #             # Publish the joint state solution
    #             self._publish_joint_state(data)
    #             return data
            
    #         except ValueError as e:
    #             print(f"[IK] Failed to parse solution: {e}")
    #             return None
    #     else:
    #         print(f"[IK] Empty result from FindIK")
    #         return None
    
    def compute_ik_once(self, robotmodel):
            # robotmodel is required because we need to call robotmodel.FindKDLIK()
            
            # --- Get Slicer transform nodes ---
            fromNode = self.obsNode
            toNode   = self.toNode

            if fromNode is None or toNode is None:
                return None

            # --- Compute 4×4 transform between nodes ---
            targetPose = vtk.vtkMatrix4x4()
            success = slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(fromNode, toNode, targetPose)
            
            if not success:
                raise RuntimeError("Could not compute transform between nodes.")

            # If we have no seed or a bad seed, try all zeros first
            seed = self.last_ik_solution if self.last_ik_solution and len(self.last_ik_solution) > 0 else []

            # call KDL IK
            result_str = robotmodel.FindKDLIK(targetPose, seed)

            if result_str and result_str.strip():
                try:
                    data = [float(x) for x in result_str.split(",")]
                    print(f"[IK] Solution found: {data}")
                    self.last_ik_solution = data
                    self._publish_joint_state(data)
                    return data
                except ValueError as e:
                    print(f"[IK] Failed to parse solution: {e}")
                    return None
            else:
                print(f"[IK] Empty result from FindKDLIK")
                return None

    def _publish_joint_state(self, joint_positions):
            """Publish joint state to topic, handling ghost prefix automatically."""
            if self.joint_state_publisher is None:
                return
            
            try:
                jsmsg = self.joint_state_publisher.GetBlankMessage()
                
                # Use stored joint names (these are CLEAN, e.g. "shoulder_pan_joint")
                original_names = self.joint_names
                num_joints = len(joint_positions)
                
                # --- DETECT GHOST TOPIC ---
                topic_name = ""
                if hasattr(self.joint_state_publisher, "GetTopic"):
                    topic_name = self.joint_state_publisher.GetTopic()
                elif hasattr(self.joint_state_publisher, "GetTopicName"):
                    topic_name = self.joint_state_publisher.GetTopicName()

                # --- GENERATE PUBLISH NAMES ---
                # Create a temporary list for this message only.
                publish_names = []
                
                if "ghost" in topic_name:
                    # Add 'ghost_' prefix on the fly
                    publish_names = [
                        f"ghost_{name}" if not name.startswith("ghost_") else name 
                        for name in original_names
                    ]
                else:
                    # Use clean names
                    publish_names = original_names
                # -------------------------------
                
                # Set header stuff...
                current_time = time.time()
                sec = int(current_time)
                nanosec = int((current_time - sec) * 1e9)
                header = jsmsg.GetHeader()
                timestamp = header.GetStamp()
                timestamp.SetSec(sec)
                timestamp.SetNanosec(nanosec)
        
                # Use the TEMPORARY list 'publish_names'
                jsmsg.SetName(publish_names)
                jsmsg.SetPosition(joint_positions)
                jsmsg.SetVelocity([0.0] * num_joints)
                jsmsg.SetEffort([float('nan')] * num_joints)
                
                self.joint_state_publisher.Publish(jsmsg)
            except Exception as e:
                print(f"[IK] Failed to publish joint state: {e}")

    
    # def addObserverComputeIK(self, endeffectorname: str, robotmodel=None, ikgroup=None):
    #     """
    #     Observe transform changes. Uses self.obsNode and self.toNode that should be
    #     set by setupikforRobot(). Each transform update triggers IK computation.
    #     """
    #     fromNode = self.obsNode
    #     toNode   = self.toNode
        
    #     if fromNode is None or toNode is None:
    #         raise RuntimeError("Transform nodes not found. Call setupikforRobot() first.")
        
    #     # Remove previous observer if any (but don't clear node references)
    #     if self.obsTag and self.obsNode:
    #         try:
    #             self.obsNode.RemoveObserver(self.obsTag)
    #         except Exception:
    #             pass
    #         self.obsTag = None

    #     def onModified(caller, eventId):
    #         # Trigger IK compute
    #         self.compute_ik_once(endeffectorname=endeffectorname, robotmodel=robotmodel, ikgroup=ikgroup)


    #     self.callback = onModified
    #     eventId = slicer.vtkMRMLTransformNode.TransformModifiedEvent
    #     self.obsTag  = fromNode.AddObserver(eventId, self.callback)

    #     return self.obsTag
    
    
    def addObserverComputeIK(self, robotmodel=None):
            """
            Observe transform changes. Uses self.obsNode and self.toNode that should be
            set by setupikforRobot(). Each transform update triggers IK computation.
            """
            fromNode = self.obsNode
            toNode   = self.toNode
            
            if fromNode is None or toNode is None:
                raise RuntimeError("Transform nodes not found. Call setupikforRobot() first.")
            
            # Remove previous observer if any to prevent duplicates
            # (but this will clear self.obsNode, so we restore it below)
            self.removeObserver()
            
            # Restore the node references that removeObserver() cleared
            self.obsNode = fromNode
            self.toNode = toNode

            def onModified(caller, eventId):
                # Trigger IK compute
                self.compute_ik_once(robotmodel=robotmodel)

            self.callback = onModified
            eventId = slicer.vtkMRMLTransformNode.TransformModifiedEvent
            self.obsTag  = fromNode.AddObserver(eventId, self.callback)

            return self.obsTag
    
    # Set robot opacity
    def setopacity(self, robotmodel, opacity):
        
        # Get number of model nodes under robotmodel
        numModels = robotmodel.GetNumberOfNodeReferences("model")  

        # Loop through each model node and set opacity
        for i in range(numModels):  
            modelNode = robotmodel.GetNthNodeReference("model", i)  
            displayNode = modelNode.GetDisplayNode()  
            if displayNode:  
                displayNode.SetOpacity(opacity)

    # Set robot color
    def setRobotColor(self, robotNode, color):
        
        # Get number of model nodes under robotmodel
        numModels = robotNode.GetNumberOfNodeReferences("model")  

        # Loop through each model node and set color
        for i in range(numModels):  
            modelNode = robotNode.GetNthNodeReference("model", i)  
            displayNode = modelNode.GetDisplayNode()  
            if displayNode:  
                r = color.red() / 255.0
                g = color.green() / 255.0
                b = color.blue() / 255.0
                displayNode.SetColor(r, g, b)
    
    # Parse urdf to get joint limits
    def parse_joint_limits_from_urdf(self, urdf_xml: str):        
        root = ET.fromstring(urdf_xml)
        limits = {}  # joint_name -> (lower, upper) floats

        for joint in root.findall("joint"):
            jtype = joint.get("type", "")
            name = joint.get("name", "")
            if jtype == "fixed" or not name:
                continue

            limit = joint.find("limit")
            if limit is None:
                continue

            lo = limit.get("lower")
            hi = limit.get("upper")
            if lo is None or hi is None:
                continue

            limits[name] = (float(lo), float(hi))

        return limits
    
    def setJointSlidersFromUrdfLimits(self, limits_rad, sliders):

        if len(sliders) != len(limits_rad):
            print(
                f"[RoboDrag] Slider count ({len(sliders)}) "
                f"!= joint count ({len(limits_rad)})"
            )

        for slider, (jointName, (lo_rad, hi_rad)) in zip(sliders, limits_rad.items()):
            lo_deg = int(round(math.degrees(lo_rad)))
            hi_deg = int(round(math.degrees(hi_rad)))

            if lo_deg > hi_deg:
                lo_deg, hi_deg = hi_deg, lo_deg

            slider.setRange(lo_deg, hi_deg)
            slider.setValue(0)

            print(f"[RoboDrag] {jointName}: {lo_deg}..{hi_deg} deg")
            
    def parse_joint_names_from_urdf(self, urdf_xml: str):
        root = ET.fromstring(urdf_xml)
        names = []

        for joint in root.findall("joint"):
            jtype = joint.get("type", "")
            name = joint.get("name", "")
            if not name:
                continue
            if jtype == "fixed":
                continue
            names.append(name)

        return names


    
        


#
# RoboDragTest
#


class RoboDragTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_RoboDrag1()

    def test_RoboDrag1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("RoboDrag1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = RoboDragLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
