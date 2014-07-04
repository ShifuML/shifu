package ml.shifu.plugin.mahout.trainer;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class MahoutDataPair {
    double[] inputData;
    double[] outputData;
    boolean isEvalData = false;

    public MahoutDataPair(double[] data, double[] ideal) {
        inputData = data;
        outputData = ideal;
    }
    public MahoutDataPair(double[] data, double[] ideal, boolean isEvalData) {
        inputData = data;
        outputData = ideal;
        this.isEvalData = isEvalData;
    }

    public Vector getMahoutInputVector() {
        int inputLen = inputData.length;
        double[] inputList = new double[inputLen + outputData.length];
        for (int i = 0; i < inputLen; i++)
            inputList[i] = inputData[i];
        for (int i = 0; i < outputData.length; i++)
            inputList[i + inputLen] = outputData[i];
        return new DenseVector(inputList);
    }

    public Vector getMahoutEvalVector() {

        return new DenseVector(inputData);
    }

    public double[] getIdealData() {
        return outputData;
    }

    public boolean isEvalData() {
        return isEvalData;
    }

    public void setEvalData(boolean isEvalData) {
        this.isEvalData = isEvalData;
    }

}
