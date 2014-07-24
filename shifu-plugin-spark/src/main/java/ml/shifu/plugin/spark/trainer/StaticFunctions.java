package ml.shifu.plugin.spark.trainer;

import java.util.regex.Pattern;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

public class StaticFunctions {

    @SuppressWarnings("serial")
    public static class ParsePoint implements Function<String, LabeledPoint> {
        private Pattern COMMA;
        private int target;
        private int[] activeField;

        public ParsePoint(int targetID, int[] activeField, String splitter) {
            target = targetID;
            this.activeField = activeField;
            COMMA = Pattern.compile("\\" + splitter);
        }

        @Override
        public LabeledPoint call(String line) {
            String[] parts = COMMA.split(line);
            double y = Double.parseDouble(parts[target]);
            int len = activeField.length;
            double[] x = new double[len];
            for (int i = 0; i < len; i++) {
                x[i] = Double.parseDouble(parts[activeField[i]]);
            }
            return new LabeledPoint(y, Vectors.dense(x));
        }
    }

    @SuppressWarnings("serial")
    public static class ParseVector implements Function<String, Vector> {
        private Pattern COMMA;
        private int[] activeField;

        public ParseVector(int[] activeField, String splitter) {
            this.activeField = activeField;
            COMMA = Pattern.compile("\\" + splitter);
        }

        @Override
        public Vector call(String line) {
            String[] parts = COMMA.split(line);
            int len = activeField.length;
            double[] x = new double[len];
            for (int i = 0; i < len; i++) {
                x[i] = Double.parseDouble(parts[activeField[i]]);
            }
            return Vectors.dense(x);
        }
    }

    @SuppressWarnings("serial")
    public static class TargetVector implements Function<String, Double> {
        private Pattern COMMA;
        private int target;

        public TargetVector(int targetID, String splitter) {
            target = targetID;
            COMMA = Pattern.compile("\\" + splitter);
        }

        @Override
        public Double call(String line) {
            String[] parts = COMMA.split(line);
            return Double.parseDouble(parts[target]);
        }
    }
    // @SuppressWarnings("serial")
    // public static class SumMSECalculator implements
    // Function2<String, Double, Double> {
    // private Pattern COMMA;
    // private int target;
    //
    // public SumMSECalculator(int targetID, String splitter) {
    // target = targetID;
    // COMMA = Pattern.compile("\\" + splitter);
    // }
    //
    // @Override
    // public Double call(String line, Double predict) throws Exception {
    // double ideal = Double.parseDouble(COMMA.split(line)[target]);
    // return Math.pow(ideal - predict, 2.0);
    // }
    // }
}
