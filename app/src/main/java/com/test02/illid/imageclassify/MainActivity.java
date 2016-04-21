package com.test02.illid.imageclassify;

import android.annotation.TargetApi;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Build;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;

import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgproc.Imgproc.INTER_AREA;
import static org.opencv.imgproc.Imgproc.INTER_CUBIC;
import static org.opencv.imgproc.Imgproc.resize;
import static org.opencv.ml.SVM.*;

public class MainActivity extends AppCompatActivity {
    private Button train, predict;
    private int ImgWidth = 120;
    private int ImgHeight = 120;
    private File train_data = new File("/storage/emulated/0/Pictures/xKImageFinder/TrainInfo");
    private File test_data = new File("/storage/emulated/0/Pictures/xKImageFinder/MatInfo");
    //private File svm_save = new File("/storage/emulated/0/Pictures/xKImageFinder/SVM_data.xml");
    private ArrayList<Integer> Category = new ArrayList<>();
    private ArrayList<String> ImgPath = new ArrayList<>();
    private MatOfFloat descriptors = new MatOfFloat();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        train = (Button)findViewById(R.id.Button1);
        predict = (Button)findViewById(R.id.Button2);

        train.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new TrainTask().execute();
            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new PredictTask().execute();
            }
        });
    }

    private class TrainTask extends AsyncTask {
        int nLines = 0;
        int n = 0;
        int i = 0;
        Mat data_mat, res_mat, src, trainImg;
        SVM svm;
        TermCriteria cri;
        @TargetApi(Build.VERSION_CODES.KITKAT)
        @Override
        protected Object doInBackground(Object[] params) {
            try (BufferedReader br = new BufferedReader(new FileReader(train_data))) {
                String line;
                while ((line = br.readLine()) != null) {
                    // process the line.
                    nLines++;
                    if (nLines % 2 == 0){
                        Category.add(Integer.parseInt(line));
                    }
                    else{
                        ImgPath.add(line);
                    }
                }
            }catch(IOException e){
                Log.e("TrainTask", "Can't read train data file");
            }
            //res_mat.create(nLines/2, 1, CvType.CV_32FC1);
            //res_mat.empty();
            //data_mat.create();
            res_mat = new Mat();
            src = new Mat();
            data_mat = new Mat();
            trainImg = new Mat();
            for (String path : ImgPath) {
                src = imread(path, 1);
                resize(src, trainImg, new Size(ImgWidth, ImgHeight), 0, 0, INTER_CUBIC);
                HOGDescriptor hog = new HOGDescriptor(new Size(ImgWidth, ImgHeight), new Size(16,16), new Size(8,8), new Size(8,8), 9);
                hog.compute(trainImg, descriptors);
                n = 0;
                for (Float data : descriptors.toArray()) {
                    data_mat.put(i, n, data);
                    n++;
                }
                res_mat.put(i, 0, Category.get(i));
                i++;
            }
            svm.create();
            svm.setC(10.0);
            svm.setDegree(10.0);
            svm.setGamma(8.0);
            svm.setCoef0(1.0);
            svm.setType(SVM.C_SVC);
            svm.setKernel(SVM.RBF);
            svm.setNu(0.5);
            svm.setP(0.1);
            svm.train(data_mat, 1, res_mat);
            svm.save("/storage/emulated/0/Pictures/xKImageFinder/SVM_data.xml");
            return null;
        }
    }

    private class PredictTask extends AsyncTask {
        int nLines = 0;
        int n = 0;
        int i = 0;
        Mat data_mat, res_mat, src, trainImg;
        SVM svm;
        TermCriteria cri;
        @TargetApi(Build.VERSION_CODES.KITKAT)
        @Override
        protected Object doInBackground(Object[] params) {
            try (BufferedReader br = new BufferedReader(new FileReader(test_data))) {
                String line;
                while ((line = br.readLine()) != null) {
                    // process the line.
                    nLines++;
                    ImgPath.add(line);
                }
            }catch(IOException e){
                Log.e("TrainTask", "Can't read test data file");
            }
            //res_mat.create(nLines/2, 1, CvType.CV_32FC1);
            //res_mat.empty();
            //data_mat.create();
            res_mat = new Mat();
            src = new Mat();
            data_mat = new Mat();
            trainImg = new Mat();
            for (String path : ImgPath) {
                src = imread(path, 1);
                resize(src, trainImg, new Size(ImgWidth, ImgHeight), 0, 0, INTER_CUBIC);
                HOGDescriptor hog = new HOGDescriptor(new Size(ImgWidth, ImgHeight), new Size(16,16), new Size(8,8), new Size(8,8), 9);
                hog.compute(trainImg, descriptors);
                n = 0;
                for (Float data : descriptors.toArray()) {
                    data_mat.put(i, n, data);
                    n++;
                }
                res_mat.put(i, 0, Category.get(i));
                i++;
            }
            // load the svm from the external storage
            // ????
            // prediction result
            int result = (int) svm.predict(data_mat);

            return null;
        }
    }
}
