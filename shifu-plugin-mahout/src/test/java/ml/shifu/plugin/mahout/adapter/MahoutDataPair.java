package ml.shifu.plugin.mahout.adapter;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class MahoutDataPair {
    private int actual;
    private Vector vector;
    private double[] originalData;

    public MahoutDataPair(int targetValue, double[] data) {
        this.actual = targetValue;
        originalData = data;
        vector = new DenseVector(data);
    }

    public int getActual() {
        return actual;
    }

    public void setActual(int actual) {
        this.actual = actual;
    }

    public Vector getVector() {
        return vector;
    }

    public void setVector(Vector vector) {
        this.vector = vector;
    }

    public Vector getVectorAsInputVector() {
        int len = originalData.length;
        double[] inputVector = new double[len + 1];
        for (int i = 0; i < len; i++)
            inputVector[i] = originalData[i];
        inputVector[len] = actual;
        return new DenseVector(inputVector);
    }
}
