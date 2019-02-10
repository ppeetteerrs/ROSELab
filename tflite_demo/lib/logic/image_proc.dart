import 'dart:typed_data';
import 'dart:io';
import 'package:image/image.dart' as DartImage;
import 'package:flutter/material.dart';

class ImageProc {
  String localFilePath;
  DartImage.Image dartImage;
  Image displayImage;
  List<int> imageRaw;
  Uint8List imageUint8;
  Uint8List imageUint8RGB;
  Float64List imageUint8RGBNormalized;
  bool rotate;
  int height;
  int width;
  int dimension;

  ImageProc({@required this.localFilePath, this.imageRaw, this.height = 124, this.width = 124, this.rotate = false}) : dimension = height * width;

  Future<ImageProc> setUp() async {
    if (this.imageRaw == null) {
      print("Loading Image from $localFilePath");

      imageRaw = (new File(localFilePath)).readAsBytesSync();
    }

    DartImage.Image imageOriginal = decodeImage(imageRaw);

    if (height == null) {
      height = dartImage.height;
    }

    if (width == null) {
      width = dartImage.width;
    }

    List<int> cropSizes = getCropSize(imageOriginal);

    print(cropSizes);

    imageOriginal = DartImage.copyResize(imageOriginal, cropSizes[0], cropSizes[1]);

    dartImage = DartImage.copyCrop(imageOriginal, cropSizes[2], cropSizes[3], width, height);

    if (rotate) {
      dartImage = DartImage.copyRotate(dartImage, 90);
    }

    displayImage = Image.memory(
      DartImage.encodePng(dartImage),
      width: 250.0,
      height: 250.0,
      fit: BoxFit.contain,
    );

    imageUint8 = dartImage.getBytes();

    imageUint8RGB = reduceToRGB();

    List<double> imageUint8RGBNormalizedList = imageUint8RGB.map<double>((int item) {
      return (item - 128) / 128.0;
    }).toList();

    imageUint8RGBNormalized = Float64List.fromList(imageUint8RGBNormalizedList);

    print("Loaded image of dimensions $height x $width");

    return this;
  }

  DartImage.Image decodeImage(List<int> rawData) {
    if (this.localFilePath.endsWith(".png")) {
      return DartImage.decodePng(rawData);
    } else if (this.localFilePath.endsWith(".jpg") || this.localFilePath.endsWith("jpeg")) {
      return DartImage.decodeJpg(rawData);
    } else {
      return DartImage.decodeImage(rawData);
    }
  }

  Uint8List reduceToRGB() {
    Uint8List rgbArray = Uint8List(dimension * 3);
    for (int i = 0; i < dimension; i++) {
      rgbArray[i * 3] = imageUint8[i * 4];
      rgbArray[i * 3 + 1] = imageUint8[i * 4 + 1];
      rgbArray[i * 3 + 2] = imageUint8[i * 4 + 2];
    }
    return rgbArray;
  }

  List<int> getPixels({@required int x, @required int y}) {
    int startIndex = (y * 224 + x) * 4;
    List<int> pixels = imageUint8RGB.sublist(startIndex, startIndex + 3);
    return pixels;
  }

  List<double> getNormalizedPixels({@required int x, @required int y}) {
    int startIndex = (y * 224 + x) * 4;
    List<double> pixels = imageUint8RGBNormalized.sublist(startIndex, startIndex + 3);
    return pixels;
  }

  // Prints Image Pixels
  void printPixels() {
    for (int i = 0; i < dartImage.height; i++) {
      String lineOutput = "";
      for (int j = 0; j < dartImage.width; j++) {
        lineOutput += dartImage.getPixel(i, j).toString() + " ";
      }
      print(lineOutput);
    }
  }

  List<int> getCropSize(DartImage.Image image) {
    print("Original Dimension: ${image.width}x${image.height}");

    double currentRatio = image.width / image.height;
    double desiredRatio = width / height;

    List<int> results;
    if (currentRatio > desiredRatio) {
      int newWidth = (height * currentRatio).toInt();
      results = [newWidth, height, ((width - newWidth).abs() ~/ 2), 0];
    } else {
      int newHeight = (width ~/ currentRatio);
      results = [width, newHeight, 0, ((height - newHeight).abs() ~/ 2)];
    }

    print("Resized Dimensions: ${results.toString()}");
    return results;
  }
}
