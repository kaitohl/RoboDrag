import logging
import os, time
from typing import Annotated, Optional

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
        self.ui.ikpushButton.connect("clicked(bool)", self.onikbutton)
        self.ui.opacitypushButton.connect("clicked(bool)", self.onopacitybutton)
        self.ui.robotColorButton.connect("colorChanged(QColor)", self.onRobotColorChanged)
        
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
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
        
        # check if sphere model already exists (slicer.util.getNode raises if not found)
        try:
            model = slicer.util.getNode("ProbeSphere")
        except slicer.util.MRMLNodeNotFoundException:
            model = None

        if model is not None:
            print("Sphere model already exists.")
            # Get linear transform into self.fromtransform
            self.fromtransform = slicer.util.getNode("ProbeSphere_Transform")
            # Get ros2 root into self.totransform
            self.totransform = self.logic.findRos2Root()
        else:
            print("Creating sphere model...")
            model = self.logic.createSphereModel()
            self.fromtransform = self.logic.createLinearTransform()

            self.logic.applyTransformToModel(model, self.fromtransform)
            self.totransform = self.logic.findRos2Root()
            print(f"Found ros2 root transform: {self.totransform.GetName()}")
        
        ikgroup = self.ui.grouplineEdit.text
        robot = self.ui.robotlineEdit.text
        self.robot = self.logic.setupikforRobot(group=ikgroup, robotname=robot, fromtransformname=self.fromtransform.GetName(), totransformname=self.totransform.GetName())
        
    def onstreamstartbutton(self) -> None:
        ee = self.ui.eelineEdit.text
        robotmodel = self.robot
        ikgroup = self.ui.grouplineEdit.text
        
        if robotmodel is None:
            print("Robot model is not set up. Please load the robot first.")
            return
        
        if ee is None:
            print("End-effector transform name is not set.")
            return
    
        self.logic.addObserverComputeIK(ee, robotmodel, ikgroup)
        print("Started streaming.")

        
    def onstreamstopbutton(self) -> None:
        self.logic.removeObserver()
        print("Stopped streaming.")
    
    def onikbutton(self) -> None:
        self.logic.compute_ik_once(self.fromtransform.GetName(), self.totransform.GetName())
        
    def onopacitybutton(self) -> None:
        opacity = self.ui.spinBox.value / 100.0
        robot = self.ui.robotcomboBox.currentNode()
        self.logic.setopacity(robot, opacity)
        
    def onRobotColorChanged(self) -> None:

        # Get currently selected robot from your qMRMLNodeComboBox
        robotNode = self.ui.robotcomboBox.currentNode()  # or self.ui.comboBox.currentNode()
        color = self.ui.robotColorButton.color
        self.logic.setRobotColor(robotNode, color)
        

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
        self.last_ik_solution = [0.0] * 6
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

    def findLeafRobotTransform(self,prefix="ros2:tf2lookup:"):
        """
        Return the leaf (last) transform whose name starts with `prefix`.
        If multiple leaves exist, prefer names containing flange/tool/ee/end/specholder.
        """
        ts = [t for t in self._allTransforms() if t.GetName().startswith(prefix)]
        if not ts:
            raise RuntimeError(f"No transforms found with prefix '{prefix}'.")

        subset = set(ts)
        children = {t: [] for t in ts}
        for t in ts:
            p = t.GetParentTransformNode()
            if p in subset:
                children[p].append(t)

        # leaves = no children in this subset
        leaves = [t for t in ts if len(children[t]) == 0]
        if not leaves:
            # degenerate case: single node with self? fallback to deepest by ancestry
            leaves = ts

        if len(leaves) == 1:
            return leaves[0]

        # prefer typical end-effector names
        prefs = ("specholder", "flange", "tool", "ee", "end")
        lname = [t.GetName().lower() for t in leaves]
        for kw in prefs:
            for i, n in enumerate(lname):
                if kw in n:
                    return leaves[i]

        # fallback: deepest node by chain length
        def depth(t):
            d = 0
            cur = t
            while cur.GetParentTransformNode() in subset:
                cur = cur.GetParentTransformNode()
                d += 1
            return d
        leaves.sort(key=depth, reverse=True)
        return leaves[0]

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

    def findRos2Root(self, prefix="ros2:tf2lookup:"):
        s = slicer.mrmlScene
        ts = [s.GetNthNodeByClass(i,"vtkMRMLTransformNode")
            for i in range(s.GetNumberOfNodesByClass("vtkMRMLTransformNode"))]
        ts = [t for t in ts if t.GetName().startswith(prefix)]
        if not ts:
            raise RuntimeError(f"No transforms with prefix '{prefix}' found.")
        subset = set(ts)
        roots = [t for t in ts if t.GetParentTransformNode() not in subset]
        return roots[0]  # the top-most (your “first ros2 tf lookup”)

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
        if self.obsNode and self.obsTag:
            try:
                self.obsNode.RemoveObserver(self.obsTag)
            except Exception:
                pass
        self.obsNode = None
        self.obsTag = None
        
    def setupikforRobot(self, group, robotname, fromtransformname, totransformname):
        if group:
            self.plangroup = group
            print(f"Set IK group to '{group}'")
        else:
            raise RuntimeError("IK group name is required.")
        
        fromNode = slicer.util.getNode(fromtransformname)
        toNode   = slicer.util.getNode(totransformname)
        
        if fromNode is None or toNode is None:
            raise RuntimeError("Transform nodes not found.")
        else:
            self.obsNode = fromNode
            self.toNode = toNode
            print(f"Nodes set for IK: from '{fromtransformname}' to '{totransformname}'")
        
        rosLogic = slicer.util.getModuleLogic('ROS2')
        rosNode = rosLogic.GetDefaultROS2Node()
        t = rosNode.GetRobotNodeByName(robotname)
        bool = t.setupIK(group)
        self.joint_state_publisher = rosNode.CreateAndAddPublisherNode('JointState', "/ghost/ghost_joint_states")
        
        if bool:
            print(f"IK setup for robot '{robotname}' and group '{group}' successful.")
            return t
        else:
            raise RuntimeError(f"IK setup for robot '{robotname}' and group '{group}' failed.")
        

    def compute_ik_once(self, endeffectorname: str, robotmodel=None, ikgroup=None):

        # --- Get Slicer transform nodes ---
        fromNode = self.obsNode
        toNode   = self.toNode

        # --- Compute 4×4 transform between nodes ---
        targetPose = vtk.vtkMatrix4x4()
        if not slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(fromNode, toNode, targetPose):
            raise RuntimeError("Could not compute transform between nodes.")
        
        seed = self.last_ik_solution
        result_str = robotmodel.FindIK(ikgroup, targetPose, endeffectorname, seed, 0.05)

        if result_str and result_str.strip():
            # Parse comma-separated string into list of floats
            try:
                data = [float(x) for x in result_str.split(",")]

                print(f"[IK] Joint Solution: {data}")
                self.last_ik_solution = data
                # Publish the joint state solution
                self._publish_joint_state(data)
                return data
            
            except ValueError as e:
                print(f"[IK] Failed to parse solution: {e}")
                return None
        else:
            print(f"[IK] Empty result from FindIK")
            return None

    def _publish_joint_state(self, joint_positions):
        """Publish joint state to /ghost/ghost_joint_states topic."""
        if self.joint_state_publisher is None:
            return
        
        try:
            jsmsg = self.joint_state_publisher.GetBlankMessage()
            # Joint names from mycobot URDF
            joint_names = [
            'joint2_to_joint1_ghost',
            'joint3_to_joint2_ghost',
            'joint4_to_joint3_ghost',
            'joint5_to_joint4_ghost',
            'joint6_to_joint5_ghost',
            'joint6output_to_joint6_ghost'
            ]
            # Set header timestamp
            current_time = time.time()
            sec = int(current_time)
            nanosec = int((current_time - sec) * 1e9)
            header = jsmsg.GetHeader()
            timestamp = header.GetStamp()
            timestamp.SetSec(sec)
            timestamp.SetNanosec(nanosec)
    
            jsmsg.SetName(joint_names)
            jsmsg.SetPosition(joint_positions)
            jsmsg.SetVelocity([0.0] * 6)
            jsmsg.SetEffort([float('nan')] * 6)
            
            self.joint_state_publisher.Publish(jsmsg)
        except Exception as e:
            print(f"[IK] Failed to publish joint state: {e}")

    
    def addObserverComputeIK(self, endeffectorname: str, robotmodel=None, ikgroup=None):
        """
        Observe transform changes. Uses self.obsNode and self.toNode that should be
        set by setupikforRobot(). Each transform update triggers IK computation.
        """
        fromNode = self.obsNode
        toNode   = self.toNode
        
        if fromNode is None or toNode is None:
            raise RuntimeError("Transform nodes not found. Call setupikforRobot() first.")
        
        # Remove previous observer if any (but don't clear node references)
        if self.obsTag and self.obsNode:
            try:
                self.obsNode.RemoveObserver(self.obsTag)
            except Exception:
                pass
            self.obsTag = None

        def onModified(caller, eventId):
            # Trigger IK compute
            self.compute_ik_once(endeffectorname=endeffectorname, robotmodel=robotmodel, ikgroup=ikgroup)


        self.callback = onModified
        eventId = slicer.vtkMRMLTransformNode.TransformModifiedEvent
        self.obsTag  = fromNode.AddObserver(eventId, self.callback)

        return self.obsTag
    
    def setopacity(self, robotmodel, opacity):
        
        numModels = robotmodel.GetNumberOfNodeReferences("model")  

        for i in range(numModels):  
            modelNode = robotmodel.GetNthNodeReference("model", i)  
            displayNode = modelNode.GetDisplayNode()  
            if displayNode:  
                displayNode.SetOpacity(opacity)
                
    def setRobotColor(self, robotNode, color):

        numModels = robotNode.GetNumberOfNodeReferences("model")  

        for i in range(numModels):  
            modelNode = robotNode.GetNthNodeReference("model", i)  
            displayNode = modelNode.GetDisplayNode()  
            if displayNode:  
                r = color.red() / 255.0
                g = color.green() / 255.0
                b = color.blue() / 255.0
                displayNode.SetColor(r, g, b)
    
    
    
        


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
