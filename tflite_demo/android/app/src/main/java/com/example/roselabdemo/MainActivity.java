package com.example.roselabdemo;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.os.SystemClock;

import io.flutter.app.FlutterActivity;
import io.flutter.plugins.GeneratedPluginRegistrant;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends FlutterActivity {

    private static final String CHANNEL = "tfCall";

    private static final int FILTER_STAGES = 3;

    private static final float FILTER_FACTOR = 0.4f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        GeneratedPluginRegistrant.registerWith(this);

        new MethodChannel(getFlutterView(), CHANNEL).setMethodCallHandler(new MethodCallHandler() {
            @Override
            public void onMethodCall(MethodCall call, Result result) {
                if (call.method.equals("getTF")) {
                    try {

                        double[] imagePixels = call.argument("image");

                        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(imagePixels.length * 4);

                        byteBuffer.order(ByteOrder.nativeOrder());

                        byteBuffer.rewind();

                        for (int i = 0; i < imagePixels.length; i++) {
                            byteBuffer.putFloat((float) imagePixels[i]);
                        }

                        byteBuffer.rewind();

                        long startTime = SystemClock.uptimeMillis();


                        // TimeUnit.MILLISECONDS.sleep(1500);
                        final float[] results = tfInterpret(byteBuffer);

                        long endTime = SystemClock.uptimeMillis();
                        String timeTaken = Long.toString(endTime - startTime);

                        double[] resultsDouble = new double[results.length];

                        for (int i=0; i< results.length; i++) {
                            resultsDouble[i] = (double) results[i];
                        }

                        result.success(resultsDouble);
                    } catch (Exception e) {
                        result.success(e.getMessage());
                    }
                } else {
                    result.notImplemented();
                }
            }
        });
    }

    private float[] tfInterpret(ByteBuffer imageData) {
        String returnText = "Loading";
        String timeTaken = "0";
        float[][] labelProbArray;
        try {

            Interpreter tflite;
            tflite = new Interpreter(loadModel());
            List<String> labelsText = loadLabelList();
            labelProbArray = new float[1][labelsText.size()];

            long startTime = SystemClock.uptimeMillis();
            tflite.run(imageData, labelProbArray);
            long endTime = SystemClock.uptimeMillis();
            timeTaken = Long.toString(endTime - startTime);

            tflite.close();

            returnText = getTopLabel(labelProbArray, labelsText);

        } catch (IOException e) {
            // return "Error";
            return null;
        }
        // return returnText + " Took " + timeTaken + "ms.";
        return labelProbArray[0];
    }

    private String getTopLabel(float[][] labelProbs, List<String> labelTexts) {
        float maxProb = labelProbs[0][0];
        int maxIndex = 0;
        for (int i = 1; i < labelTexts.size(); i++) {
            if (labelProbs[0][i] > maxProb) {
                maxIndex = i;
                maxProb = labelProbs[0][i];
            }
        }
        return labelTexts.get(maxIndex) + " with probability " + Float.toString(maxProb);
    }

    private MappedByteBuffer loadModel() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("indiannet/model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(getAssets().open("indiannet/labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    /* Memory-map the model file in Assets. */
    /*
     * private MappedByteBuffer loadModelFile(Activity activity) throws IOException
     * { AssetFileDescriptor fileDescriptor =
     * activity.getAssets().openFd("mobilenet/mobilenet.tflite"); FileInputStream
     * inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
     * FileChannel fileChannel = inputStream.getChannel(); long startOffset =
     * fileDescriptor.getStartOffset(); long declaredLength =
     * fileDescriptor.getDeclaredLength(); return
     * fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
     * }
     */
}
