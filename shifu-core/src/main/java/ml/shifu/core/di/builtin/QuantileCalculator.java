package ml.shifu.core.di.builtin;


import org.dmg.pmml.Quantile;

import java.util.ArrayList;
import java.util.List;

public class QuantileCalculator {


    public List<Quantile> getEvenlySpacedQuantiles(List<Double> values, Integer num) {
        if (num <= 1) throw new RuntimeException("Quantile number should be >= 2, but got: " + num);

        List<Quantile> quantiles = new ArrayList<Quantile>();

        int size = values.size();


        for (int i = 0; i < num; i++) {
            Quantile quantile = new Quantile();
            quantile.setQuantileLimit(((double) i / (num - 1)) * 100);
            quantile.setQuantileValue(values.get((size - 1) * i / (num - 1)));
            quantiles.add(quantile);
        }

        return quantiles;
    }

    public List<Quantile> getSpecifiedQuantiles(List<Double> values, Double... nums) {
        List<Quantile> quantiles = new ArrayList<Quantile>();

        int size = values.size();

        for (Double n : nums) {
            if (n < 0.0 || n > 100.0) {
                throw new RuntimeException("Specified percentiles should be between 0 and 100, but got: " + n);
            }
            Quantile quantile = new Quantile();
            quantile.setQuantileLimit(n);
            quantile.setQuantileValue(values.get((int) Math.round((size - 1) * n / 100)));
            quantiles.add(quantile);
        }

        return quantiles;
    }

}
