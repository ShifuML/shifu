package ml.shifu.plugin.spark.trainer;

import java.util.regex.Pattern;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

public class StaticFunctions {

    @SuppressWarnings("serial")
   public static class ParsePoint implements Function<String, LabeledPoint> {
        private static final Pattern COMMA = Pattern.compile("\\,");

        @Override
        public LabeledPoint call(String line) {
            String[] parts = COMMA.split(line);
            double y = Double.parseDouble(parts[0]);
            int len = parts.length;
            double[] x = new double[len];
            for (int i = 1; i < len; i++) {
                x[i] = Double.parseDouble(parts[i]);
            }
            return new LabeledPoint(y, Vectors.dense(x));
        }
    }

  @SuppressWarnings("serial")
public static class ParseVector implements Function<String, Vector> {
        private static final Pattern COMMA = Pattern.compile(",");

        @Override
        public Vector call(String line) {
            String[] parts = COMMA.split(line);

            int len = parts.length;
            double[] x = new double[len];
            for (int i = 1; i < len; i++) {
                    x[i] = Double.parseDouble(parts[i]);
            }
            return Vectors.dense(x);
        }
    }
}
