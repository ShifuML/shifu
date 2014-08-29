package ml.shifu.core.di.builtin.stats;

import ml.shifu.core.container.ColumnBinningResult;
import ml.shifu.core.container.NumericalValueObject;
import ml.shifu.core.di.builtin.EqualPositiveColumnNumBinningCalculator;
import ml.shifu.core.di.builtin.KSIVCalculator;
import ml.shifu.core.di.builtin.QuantileCalculator;
import ml.shifu.core.di.builtin.WOECalculator;
import ml.shifu.core.util.CommonUtils;
import ml.shifu.core.util.PMMLUtils;
import org.dmg.pmml.ContStats;
import org.dmg.pmml.Interval;
import org.dmg.pmml.NumericInfo;
import org.dmg.pmml.UnivariateStats;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class BinomialUnivariateStatsContCalculator {
    private Logger log = LoggerFactory.getLogger(BinomialUnivariateStatsContCalculator.class);

    private NumericInfo numericInfo = new NumericInfo();
    private ContStats contStats = new ContStats();


    public void calculate(UnivariateStats univariateStats, List<NumericalValueObject> nvoList, int expectedBinNum) {

        calculateBasicStats(nvoList);
        calculateBinning(nvoList, expectedBinNum);

        univariateStats.setNumericInfo(numericInfo);
        univariateStats.setContStats(contStats);

    }

    private void calculateBasicStats(List<NumericalValueObject> nvoList) {

        Double sum;
        Double squaredSum;
        Double min = Double.MAX_VALUE;
        Double max = -Double.MAX_VALUE;


        Double EPS = 1e-6;

        sum = 0.0;
        squaredSum = 0.0;


        List<Double> validValues = new ArrayList<Double>();

        for (NumericalValueObject nvo : nvoList) {

            if (CommonUtils.isValidNumber(nvo.getValue())) {

                Double value = nvo.getValue();

                max = Math.max(max, value);
                min = Math.min(min, value);

                sum += value;
                squaredSum += value * value;

                validValues.add(value);
            }
        }


        numericInfo.setMaximum(max);
        numericInfo.setMinimum(min);


        int validSize = validValues.size();
        // mean and stdDev defaults to NaN
        if (validSize == 0 || sum.isInfinite() || squaredSum.isInfinite()) {
            return;
        }


        //it's ok while the voList is sorted;
        Double median = validValues.get(validSize / 2);

        Double mean = sum / validSize;
        Double stdDev = Math.sqrt((squaredSum - (sum * sum) / validSize + EPS)
                / (validSize - 1));

        Double interQuartileRange = validValues.get((int) Math.floor(validSize * 0.75)) - validValues.get((int) Math.floor(validSize * 0.25));

        numericInfo.setStandardDeviation(stdDev);
        numericInfo.setMean(mean);
        numericInfo.setMedian(median);
        numericInfo.setInterQuartileRange(interQuartileRange);

        QuantileCalculator quantileCalculator = new QuantileCalculator();

        Collections.sort(validValues);


        numericInfo.withQuantiles(quantileCalculator.getEvenlySpacedQuantiles(validValues, 11));

        contStats.setTotalValuesSum(sum);
        contStats.setTotalSquaresSum(squaredSum);


    }

    private void calculateBinning(List<NumericalValueObject> nvoList, int expectedBinNum) {
        EqualPositiveColumnNumBinningCalculator calculator = new EqualPositiveColumnNumBinningCalculator();
        ColumnBinningResult result = calculator.calculate(nvoList, expectedBinNum);

        int size = result.getLength();

        List<Interval> intervals = new ArrayList<Interval>();
        for (int i = 0; i < size; i++) {
            Interval interval = new Interval();
            interval.setClosure(Interval.Closure.OPEN_CLOSED);
            interval.setLeftMargin(result.getBinBoundary().get(i));
            if (i == size - 1) {
                interval.setRightMargin(Double.POSITIVE_INFINITY);
            } else {
                interval.setRightMargin(result.getBinBoundary().get(i + 1));
            }

            intervals.add(interval);

        }
        contStats.withIntervals(intervals);

        Map<String, String> extensionMap = new HashMap<String, String>();

        extensionMap.put("BinCountPos", result.getBinCountPos().toString());
        extensionMap.put("BinCountNeg", result.getBinCountNeg().toString());
        extensionMap.put("BinWeightedCountPos", result.getBinWeightedPos().toString());
        extensionMap.put("BinWeightedCountNeg", result.getBinWeightedNeg().toString());
        extensionMap.put("BinPosRate", result.getBinPosRate().toString());


        List<Double> woe = WOECalculator.calculate(result.getBinCountPos().toArray(), result.getBinCountNeg().toArray());
        extensionMap.put("BinWOE", woe.toString());

        KSIVCalculator ksivCalculator = new KSIVCalculator();
        ksivCalculator.calculateKSIV(result.getBinCountNeg(), result.getBinCountPos());
        extensionMap.put("KS", Double.valueOf(ksivCalculator.getKS()).toString());
        extensionMap.put("IV", Double.valueOf(ksivCalculator.getIV()).toString());

        contStats.withExtensions(PMMLUtils.createExtensions(extensionMap));

    }

}
