import 'dart:typed_data';
import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import './logic/image_proc.dart';
import './logic/native_bridge.dart';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';

class CameraPickerImage extends StatefulWidget {
  @override
  _CameraPickerImageState createState() => new _CameraPickerImageState();
}

class _CameraPickerImageState extends State<CameraPickerImage> {
  CameraController cameraController;
  ImageProc _imageProc;
  String _resultsString = "No results yet.";

  @override
  void initState() {
    super.initState();
  }

  Future<Null> setup() async {
    List<CameraDescription> camerasAvailable = await availableCameras();
    cameraController = new CameraController(camerasAvailable[0], ResolutionPreset.low);
    await cameraController.initialize();
    print("Setup Done");
  }

  @override
  void dispose() {
    cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return new FutureBuilder(
      future: setup(),
      builder: mainBody,
    );
  }

  String timestamp() => DateTime.now().toIso8601String();

  void updateImageProc(ImageProc imageProc) {
    setState(() {
      _imageProc = imageProc;
    });
  }

  void updateResults(String resultString) {
    setState(() {
      _resultsString = resultString;
    });
  }

/*
  void takePicture() async {
    final Directory extDir = await getApplicationDocumentsDirectory();
    final String dirPath = '${extDir.path}/Pictures';
    await Directory(dirPath).create(recursive: true);
    final String filePath = '$dirPath/${timestamp()}.jpg';
    try {
      await cameraController.takePicture(filePath);
      ImageProc imageProc = await (new ImageProc(localFilePath: filePath)).setUp();
      updateImageProc(imageProc);
    } on CameraException catch (e) {
      print("Failed to take picture");
    }
  }
  */

  void takePicture() async {
    File imageFile = await ImagePicker.pickImage(source: ImageSource.camera);
    ImageProc imageProc = await (new ImageProc(localFilePath: imageFile.path, imageRaw: imageFile.readAsBytesSync(), rotate: true)).setUp();
    updateImageProc(imageProc);
  }

  void choosePicture() async {
    File imageFile = await ImagePicker.pickImage(source: ImageSource.gallery);
    ImageProc imageProc = await (new ImageProc(localFilePath: imageFile.path, imageRaw: imageFile.readAsBytesSync())).setUp();
    updateImageProc(imageProc);
  }

  Widget imageWidget() {
    if (_imageProc != null) {
      return _imageProc.displayImage;
    } else {
      return new Container();
    }
  }

  Widget getClassificationButton() {
    if (_imageProc != null) {
      return new RaisedButton(
        child: Text('Get Classification'),
        onPressed: () async {
          ClassificationJob classificationJob = new ClassificationJob(imageProc: _imageProc);
          await classificationJob.loadLabels();
          String result = await classificationJob.classify();
          updateResults(result);
        },
      );
    } else {
      return new Container();
    }
  }

  Widget mainBody(BuildContext context, AsyncSnapshot<ImageProc> snapshot) {
    if (snapshot.connectionState == ConnectionState.done) {
      ListView widgets = new ListView(
        children: [
          imageWidget(),
          new RaisedButton(
            child: Text('Take Picture'),
            onPressed: takePicture,
          ),
          new RaisedButton(
            child: Text('Choose From Gallery'),
            onPressed: choosePicture,
          ),
          getClassificationButton(),
          new Text(_resultsString),
        ],
      );
      return widgets;
    } else {
      return new Center(
        child: new Text("Loading"),
      );
    }
  }
}
