import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:math';
import './image_proc.dart';

class ClassificationJob {
  static const platform = const MethodChannel('tfCall');
  static const nativeMethodName = "getTF";
  static const labelsPath = "labels/labels.txt";

  ImageProc imageProc;
  List<String> labels;
  List<double> results;
  List<ResultItem> sortedScores;

  ClassificationJob({@required this.imageProc});

  Future<void> setUp() async {
    await loadLabels();
  }

  Future<List<String>> loadLabels() async {
    String labelsText = await rootBundle.loadString(labelsPath);
    List<String> labelsList = labelsText.split("\n")..removeLast();
    labels = labelsList;
    return labelsList;
  }

  Future<String> classify() async {
    Stopwatch stopwatch = new Stopwatch();
    await getResults(imageProc.imageUint8RGBNormalized);
    await analyzeResults();
    String returnString = printResults(n: 3, timeTaken: stopwatch.elapsedMicroseconds);
    stopwatch.stop();
    return returnString;
  }

  Future<List<double>> getResults(Float64List imageData) async {
    try {
      // Obtain Results
      final List<double> result = await platform.invokeMethod(nativeMethodName, <String, Float64List>{
        "image": imageData,
      });

      results = result;

      return result;
    } on PlatformException catch (e) {
      throw e;
    }
  }

  Future<List<ResultItem>> analyzeResults() async {
    List<ResultItem> unsortedResults = new List();
    for (int i = 0; i < results.length; i++) {
      unsortedResults.add(new ResultItem(name: labels[i], probability: results[i]));
    }
    List<ResultItem> sortedResults = unsortedResults
      ..sort((ResultItem item1, ResultItem item2) {
        if (item1 > item2) {
          return -1;
        } else if (item1 < item2) {
          return 1;
        } else {
          return 0;
        }
      });
    sortedScores = sortedResults;
    return sortedResults;
  }

  String printResults({int n = 3, @required int timeTaken}) {
    String resultsString = "";
    for (int i = 0; i < n; i++) {
      resultsString += "    ${i + 1}. ${sortedScores[i].name} (${sortedScores[i].probability.toStringAsPrecision(3)}) \n";
    }
    resultsString += "    Time Take: ${timeTaken} microseconds.";
    return resultsString;
  }
}

class ResultItem {
  String name;
  double probability;
  ResultItem({
    @required this.name,
    @required this.probability,
  });

  operator >(ResultItem otherItem) {
    return this.probability > otherItem.probability;
  }

  operator >=(ResultItem otherItem) {
    return this.probability >= otherItem.probability;
  }

  operator <(ResultItem otherItem) {
    return this.probability < otherItem.probability;
  }

  operator <=(ResultItem otherItem) {
    return this.probability <= otherItem.probability;
  }

  operator -(ResultItem otherItem) {
    return this.probability - otherItem.probability;
  }

  operator +(ResultItem otherItem) {
    return this.probability + otherItem.probability;
  }
}
