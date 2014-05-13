package ml.shifu.shifu.di.builtin;


import ml.shifu.shifu.di.spi.Normalizer;
import ml.shifu.shifu.container.obj.ColumnConfig;

public class ZScoreNormalizer implements Normalizer {

    private Double stdDevCutOff = 4.0;

    @Override
    public Double normalize(ColumnConfig config, Object raw) {

        if (config.getColumnControl().getStdDevCutOff() != null) {
            stdDevCutOff = config.getColumnControl().getStdDevCutOff();
        }

        if (config.isCategorical()) {
            int index = config.getBinCategory().indexOf(raw);
            // TODO: use default. Not 0 !!!
            // Using the most frequent categorical value?
            if (index == -1) {
                return 0.0;
            } else {
                return computeZScore(config.getBinPosRate().get(index), config.getMean(), config.getStdDev(), stdDevCutOff);
            }
        } else {
            double value;
            try {
                value = Double.parseDouble(raw.toString());
            } catch (Exception e) {
                //log.debug("Not decimal format " + raw + ", using default!");
                value = ((config.getMean() == null) ? 0.0 : config.getMean());
            }

            return computeZScore(value, config.getMean(), config.getStdDev(), stdDevCutOff);
        }
    }

    /**
     * Compute the zscore, by original value, mean, standard deviation and standard deviation cutoff
     *
     * @param var          - original value
     * @param mean         - mean value
     * @param stdDev       - standard deviation
     * @param stdDevCutOff - standard deviation cutoff
     * @return zscore
     */
    public static double computeZScore(double var, double mean, double stdDev, double stdDevCutOff) {
        double maxCutOff = mean + stdDevCutOff * stdDev;
        if (var > maxCutOff) {
            var = maxCutOff;
        }

        double minCutOff = mean - stdDevCutOff * stdDev;
        if (var < minCutOff) {
            var = minCutOff;
        }

        if (stdDev > 0.00001) {
            return (var - mean) / stdDev;
        } else {
            return 0.0;
        }
    }
}
