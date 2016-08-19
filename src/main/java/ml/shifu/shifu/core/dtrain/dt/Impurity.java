/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.dtrain.dt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;

/**
 * Different {@link #Impurity()} strategies to compute impurity and gain for each tree node.
 * 
 * <p>
 * {@link Entropy} and {@link Gini} are mostly for classification while {@link Variance} is for regression.
 * 
 * <p>
 * TODO FOR categorical feature, do a shuffle in {@link #computeImpurity(double[], ColumnConfig)} firstly, sort by
 * centroid
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public abstract class Impurity {

    /**
     * # of values collected, for example in {@link Variance}, count, sum and squaredSum are collected, statsSize is 3.
     * For Gini and Entropy, each class, count are selected, for binary classification, statsSize is 2.
     */
    protected int statsSize;

    /**
     * Per node, min instances, if less than this value, such gain info will be ignored.
     */
    protected int minInstancesPerNode = 1;

    /**
     * Min info gain, if less than this value, such gain info will be ignored.
     */
    protected double minInfoGain = 0d;

    /**
     * Compute impurity by feature statistics. Stats array are for all bins.
     */
    public abstract GainInfo computeImpurity(double[] stats, ColumnConfig confg);

    /**
     * Update bin stats value per feature.
     */
    public abstract void featureUpdate(double[] featuerStatistic, int binIndex, float label, float significance,
            float weight);

    /**
     * @return the statsSize
     */
    public int getStatsSize() {
        return statsSize;
    }

    /**
     * @param statsSize
     *            the statsSize to set
     */
    public void setStatsSize(int statsSize) {
        this.statsSize = statsSize;
    }

}

/**
 * Variance impurity value is computed by ((sumSquare - (sum * sum) / count) / count).
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
class Variance extends Impurity {

    public Variance() {
        // 3 are count, sum and sumSquare
        super.statsSize = 3;
    }

    public Variance(int minInstancesPerNode, double minInfoGain) {
        super.statsSize = 3;
        super.minInstancesPerNode = minInstancesPerNode;
        super.minInfoGain = minInfoGain;
    }

    @Override
    public GainInfo computeImpurity(double[] stats, ColumnConfig config) {
        double count = 0d, sum = 0d, sumSquare = 0d;
        int binSize = stats.length / super.statsSize;
        for(int i = 0; i < binSize; i++) {
            count += stats[i * super.statsSize];
            sum += stats[i * super.statsSize + 1];
            sumSquare += stats[i * super.statsSize + 2];
        }

        double impurity = getImpurity(count, sum, sumSquare);
        Predict predict = new Predict(count == 0d ? 0d : sum / count);

        double leftCount = 0d, leftSum = 0d, leftSumSquare = 0d;
        double rightCount = 0d, rightSum = 0d, rightSumSquare = 0d;
        List<GainInfo> internalGainList = new ArrayList<GainInfo>();
        Set<String> leftCategories = config.isCategorical() ? new HashSet<String>() : null;

        List<Pair> categoricalOrderList = null;
        if(config.isCategorical()) {
            // sort by predict and then pick the best split
            categoricalOrderList = getCategoricalOrderList(stats, binSize);
        }

        for(int i = 0; i < (binSize - 1); i++) {
            int index = i;
            if(config.isCategorical()) {
                index = categoricalOrderList.get(i).index;
            }
            leftCount += stats[index * super.statsSize];
            leftSum += stats[index * super.statsSize + 1];
            leftSumSquare += stats[index * super.statsSize + 2];
            rightCount = count - leftCount;
            rightSum = sum - leftSum;
            rightSumSquare = sumSquare - leftSumSquare;

            if(leftCount <= minInstancesPerNode || rightCount <= minInstancesPerNode) {
                continue;
            }

            double leftWeight = leftCount / count;
            double rightWeight = rightCount / count;
            double leftImpurity = getImpurity(leftCount, leftSum, leftSumSquare);
            double rightImpurity = getImpurity(rightCount, rightSum, rightSumSquare);
            double gain = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity;
            if(gain <= minInfoGain) {
                continue;
            }

            Split split = null;
            if(config.isCategorical()) {
                leftCategories.add(config.getBinCategory().get(i));
                split = new Split(config.getColumnNum(), FeatureType.CATEGORICAL, 0d, leftCategories);
            } else {
                split = new Split(config.getColumnNum(), FeatureType.CONTINUOUS, config.getBinBoundary().get(i + 1),
                        null);
            }

            Predict leftPredict = new Predict(leftCount == 0d ? 0d : leftSum / leftCount);
            Predict rightPredict = new Predict(rightCount == 0d ? 0d : rightSum / rightCount);

            internalGainList.add(new GainInfo(gain, impurity, predict, leftImpurity, rightImpurity, leftPredict,
                    rightPredict, split));
        }
        return GainInfo.getGainInfoByMaxGain(internalGainList);
    }

    protected List<Pair> getCategoricalOrderList(double[] stats, int binSize) {
        List<Pair> categoricalOrderList = new ArrayList<Pair>(binSize);
        for(int i = 0; i < binSize; i++) {
            // set default = double min to make it sorted at first
            double binPredict = Double.MIN_VALUE;
            if(stats[i * super.statsSize] != 0d) {
                binPredict = stats[i * super.statsSize + 1] / stats[i * super.statsSize];
            }
            // for variance, use predict value to sort
            categoricalOrderList.add(new Pair(i, binPredict));
        }
        Collections.sort(categoricalOrderList, new Comparator<Pair>() {
            @Override
            public int compare(Pair o1, Pair o2) {
                return Double.valueOf(o1.value).compareTo(Double.valueOf(o2.value));
            }
        });
        return categoricalOrderList;
    }

    protected double getImpurity(double count, double sum, double sumSquare) {
        return (count != 0d) ? ((sumSquare - (sum * sum) / count) / count) : 0d;
    }

    @Override
    public void featureUpdate(double[] featuerStatistic, int binIndex, float label, float significance, float weight) {
        featuerStatistic[binIndex * super.statsSize] += (significance * weight);
        featuerStatistic[binIndex * super.statsSize + 1] += (label * significance * weight);
        featuerStatistic[binIndex * super.statsSize + 2] += (label * label * significance * weight);
    }

}

/**
 * 
 * Reference:
 * 
 * https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_criterion.pyx#L1264
 * J. Friedman, Greedy Function Approximation: A Gradient Boosting Machine, The Annals of Statistics, Vol. 29, No. 5,
 * 2001.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
class FriedmanMSE extends Variance {

    public FriedmanMSE() {
        // 3 are count, sum and sumSquare
        super.statsSize = 3;
    }

    public FriedmanMSE(int minInstancesPerNode, double minInfoGain) {
        super.statsSize = 3;
        super.minInstancesPerNode = minInstancesPerNode;
        super.minInfoGain = minInfoGain;
    }

    @Override
    public GainInfo computeImpurity(double[] stats, ColumnConfig config) {
        double count = 0d, sum = 0d, sumSquare = 0d;
        int binSize = stats.length / super.statsSize;
        for(int i = 0; i < binSize; i++) {
            count += stats[i * super.statsSize];
            sum += stats[i * super.statsSize + 1];
            sumSquare += stats[i * super.statsSize + 2];
        }

        double impurity = getImpurity(count, sum, sumSquare);
        Predict predict = new Predict(count == 0d ? 0d : sum / count);

        double leftCount = 0d, leftSum = 0d, leftSumSquare = 0d;
        double rightCount = 0d, rightSum = 0d, rightSumSquare = 0d;
        List<GainInfo> internalGainList = new ArrayList<GainInfo>();
        Set<String> leftCategories = config.isCategorical() ? new HashSet<String>() : null;

        List<Pair> categoricalOrderList = null;
        if(config.isCategorical()) {
            // sort by predict and then pick the best split
            categoricalOrderList = getCategoricalOrderList(stats, binSize);
        }

        for(int i = 0; i < (binSize - 1); i++) {
            int index = i;
            if(config.isCategorical()) {
                index = categoricalOrderList.get(i).index;
            }
            leftCount += stats[index * super.statsSize];
            leftSum += stats[index * super.statsSize + 1];
            leftSumSquare += stats[index * super.statsSize + 2];
            rightCount = count - leftCount;
            rightSum = sum - leftSum;
            rightSumSquare = sumSquare - leftSumSquare;

            if(leftCount <= minInstancesPerNode || rightCount <= minInstancesPerNode) {
                continue;
            }

            double leftImpurity = getImpurity(leftCount, leftSum, leftSumSquare);
            double rightImpurity = getImpurity(rightCount, rightSum, rightSumSquare);

            double diff = rightCount * leftSum - leftCount * rightSum;
            double gain = (diff * diff) / (leftCount * rightCount * (leftCount + rightCount));
            if(gain <= minInfoGain) {
                continue;
            }

            Split split = null;
            if(config.isCategorical()) {
                leftCategories.add(config.getBinCategory().get(i));
                split = new Split(config.getColumnNum(), FeatureType.CATEGORICAL, 0d, leftCategories);
            } else {
                split = new Split(config.getColumnNum(), FeatureType.CONTINUOUS, config.getBinBoundary().get(i + 1),
                        null);
            }

            Predict leftPredict = new Predict(leftCount == 0d ? 0d : leftSum / leftCount);
            Predict rightPredict = new Predict(rightCount == 0d ? 0d : rightSum / rightCount);

            internalGainList.add(new GainInfo(gain, impurity, predict, leftImpurity, rightImpurity, leftPredict,
                    rightPredict, split));
        }
        return GainInfo.getGainInfoByMaxGain(internalGainList);
    }
}

/**
 * Entropy impurity value for classification tree. Entropy formula is SUM(- rate * log(rate) )
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
class Entropy extends Impurity {

    public Entropy(int numClasses, int minInstancesPerNode, double minInfoGain) {
        assert numClasses > 0;
        super.statsSize = numClasses;
        super.minInstancesPerNode = minInstancesPerNode;
        super.minInfoGain = minInfoGain;
    }

    @Override
    public GainInfo computeImpurity(double[] stats, ColumnConfig config) {
        int numClasses = super.statsSize;
        double[] statsByClasses = new double[numClasses];

        for(int i = 0; i < stats.length / numClasses; i++) {
            for(int j = 0; j < numClasses; j++) {
                double oneStatValue = stats[i * super.statsSize + j];
                statsByClasses[j] += oneStatValue;
            }
        }

        List<Pair> categoricalOrderList = null;
        if(config.isCategorical()) {
            // sort by predict and then pick the best split
            categoricalOrderList = getCategoricalOrderList(stats, stats.length / super.statsSize);
        }

        InternalEntropyInfo info = getEntropyInterInfo(statsByClasses);
        // prob only effective in binary classes
        Predict predict = new Predict(info.sumAll == 0d ? 0d : (statsByClasses[1] / info.sumAll),
                info.indexOfLargestElement);

        double[] leftStatByClasses = new double[numClasses];
        double[] rightStatByClasses = new double[numClasses];
        List<GainInfo> internalGainList = new ArrayList<GainInfo>();
        Set<String> leftCategories = config.isCategorical() ? new HashSet<String>() : null;
        for(int i = 0; i < (stats.length / numClasses - 1); i++) {
            int index = i;
            if(config.isCategorical()) {
                index = categoricalOrderList.get(i).index;
            }
            for(int j = 0; j < leftStatByClasses.length; j++) {
                leftStatByClasses[j] += stats[index * numClasses + j];
            }
            InternalEntropyInfo leftInfo = getEntropyInterInfo(leftStatByClasses);
            Predict leftPredict = new Predict(leftInfo.sumAll == 0d ? 0d : (leftStatByClasses[1] / leftInfo.sumAll),
                    leftInfo.indexOfLargestElement);

            for(int j = 0; j < leftStatByClasses.length; j++) {
                rightStatByClasses[j] = statsByClasses[j] - leftStatByClasses[j];
            }
            InternalEntropyInfo rightInfo = getEntropyInterInfo(rightStatByClasses);

            if(leftInfo.sumAll <= minInstancesPerNode || rightInfo.sumAll <= minInstancesPerNode) {
                continue;
            }

            Predict rightPredict = new Predict(
                    rightInfo.sumAll == 0d ? 0d : (rightStatByClasses[1] / rightInfo.sumAll),
                    rightInfo.indexOfLargestElement);

            double leftWeight = info.sumAll == 0d ? 0d : (leftInfo.sumAll / info.sumAll);
            double rightWeight = info.sumAll == 0d ? 0d : (rightInfo.sumAll / info.sumAll);

            double gain = info.impurity - leftWeight * leftInfo.impurity - rightWeight * rightInfo.impurity;
            if(gain <= minInfoGain) {
                continue;
            }

            Split split = null;
            if(config.isCategorical()) {
                leftCategories.add(config.getBinCategory().get(i));
                split = new Split(config.getColumnNum(), FeatureType.CATEGORICAL, 0d, leftCategories);
            } else {
                split = new Split(config.getColumnNum(), FeatureType.CONTINUOUS, config.getBinBoundary().get(i + 1),
                        null);
            }

            internalGainList.add(new GainInfo(gain, info.impurity, predict, leftInfo.impurity, rightInfo.impurity,
                    leftPredict, rightPredict, split));
        }
        return GainInfo.getGainInfoByMaxGain(internalGainList);
    }

    private List<Pair> getCategoricalOrderList(double[] stats, int binSize) {
        List<Pair> categoricalOrderList = new ArrayList<Pair>(binSize);
        for(int i = 0; i < binSize; i++) {
            // for entropy, use bin positive rate to sort
            double sum = stats[i * super.statsSize] + stats[i * super.statsSize + 1];
            double binPredict = 0d;
            if(sum != 0d) {
                binPredict = stats[i * super.statsSize + 1] / sum;
            }
            categoricalOrderList.add(new Pair(i, binPredict));
        }
        Collections.sort(categoricalOrderList, new Comparator<Pair>() {
            @Override
            public int compare(Pair o1, Pair o2) {
                return Double.valueOf(o1.value).compareTo(Double.valueOf(o2.value));
            }
        });
        return categoricalOrderList;
    }

    private InternalEntropyInfo getEntropyInterInfo(double[] statsByClasses) {
        double sumAll = 0;
        for(int i = 0; i < statsByClasses.length; i++) {
            sumAll += statsByClasses[i];
        }

        double impurity = 0d;
        int indexOfLargestElement = -1;
        if(sumAll != 0d) {
            double maxElement = Double.MIN_VALUE;
            for(int i = 0; i < statsByClasses.length; i++) {
                double rate = statsByClasses[i] / sumAll;
                if(rate != 0d) {
                    impurity -= rate * log2(rate);
                    if(statsByClasses[i] > maxElement) {
                        maxElement = statsByClasses[i];
                        indexOfLargestElement = i;
                    }
                }
            }
        }

        return new InternalEntropyInfo(sumAll, indexOfLargestElement, impurity);
    }

    private static class InternalEntropyInfo {
        double sumAll;
        double indexOfLargestElement;
        double impurity;

        public InternalEntropyInfo(double sumAll, double indexOfLargestElement, double impurity) {
            this.sumAll = sumAll;
            this.indexOfLargestElement = indexOfLargestElement;
            this.impurity = impurity;
        }
    }

    @Override
    public void featureUpdate(double[] featuerStatistic, int binIndex, float label, float significance, float weight) {
        // label + 0.1f to avoid 0.99999f is converted to 0
        featuerStatistic[binIndex * super.statsSize + (int) (label + 0.000001f)] += (significance * weight);
    }

    private double log2(double x) {
        return Math.log(x) / Math.log(2);
    }

}

/**
 * Gini impurity value for classification tree. Entropy formula is SUM(- rate * rate )
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
class Gini extends Impurity {

    public Gini(int numClasses, int minInstancesPerNode, double minInfoGain) {
        assert numClasses > 0;
        super.statsSize = numClasses;
        super.minInstancesPerNode = minInstancesPerNode;
        super.minInfoGain = minInfoGain;
    }

    @Override
    public GainInfo computeImpurity(double[] stats, ColumnConfig config) {
        int numClasses = super.statsSize;
        double[] statsByClasses = new double[numClasses];

        for(int i = 0; i < stats.length / numClasses; i++) {
            for(int j = 0; j < numClasses; j++) {
                double oneStatValue = stats[i * super.statsSize + j];
                statsByClasses[j] += oneStatValue;
            }
        }

        List<Pair> categoricalOrderList = null;
        if(config.isCategorical()) {
            // sort by predict and then pick the best split
            categoricalOrderList = getCategoricalOrderList(stats, stats.length / super.statsSize);
        }

        InternalEntropyInfo info = getGiniInfo(statsByClasses);
        // prob only effective in binary classes
        Predict predict = new Predict(info.sumAll == 0d ? 0d : statsByClasses[1] / info.sumAll,
                info.indexOfLargestElement);

        double[] leftStatByClasses = new double[numClasses];
        double[] rightStatByClasses = new double[numClasses];
        List<GainInfo> internalGainList = new ArrayList<GainInfo>();
        Set<String> leftCategories = config.isCategorical() ? new HashSet<String>() : null;
        for(int i = 0; i < (stats.length / numClasses - 1); i++) {
            int index = i;
            if(config.isCategorical()) {
                index = categoricalOrderList.get(i).index;
            }
            for(int j = 0; j < leftStatByClasses.length; j++) {
                leftStatByClasses[j] += stats[index * numClasses + j];
            }
            InternalEntropyInfo leftInfo = getGiniInfo(leftStatByClasses);
            Predict leftPredict = new Predict(leftInfo.sumAll == 0d ? 0d : leftStatByClasses[1] / leftInfo.sumAll,
                    leftInfo.indexOfLargestElement);

            for(int j = 0; j < leftStatByClasses.length; j++) {
                rightStatByClasses[j] = statsByClasses[j] - leftStatByClasses[j];
            }
            InternalEntropyInfo rightInfo = getGiniInfo(rightStatByClasses);

            if(leftInfo.sumAll <= minInstancesPerNode || rightInfo.sumAll <= minInstancesPerNode) {
                continue;
            }

            Predict rightPredict = new Predict(rightInfo.sumAll == 0d ? 0d : rightStatByClasses[1] / rightInfo.sumAll,
                    rightInfo.indexOfLargestElement);

            double leftWeight = info.sumAll == 0d ? 0d : (leftInfo.sumAll / info.sumAll);
            double rightWeight = info.sumAll == 0d ? 0d : (rightInfo.sumAll / info.sumAll);
            double gain = info.impurity - leftWeight * leftInfo.impurity - rightWeight * rightInfo.impurity;
            if(gain <= minInfoGain) {
                continue;
            }

            Split split = null;
            if(config.isCategorical()) {
                leftCategories.add(config.getBinCategory().get(i));
                split = new Split(config.getColumnNum(), FeatureType.CATEGORICAL, 0d, leftCategories);
            } else {
                split = new Split(config.getColumnNum(), FeatureType.CONTINUOUS, config.getBinBoundary().get(i + 1),
                        null);
            }

            internalGainList.add(new GainInfo(gain, info.impurity, predict, leftInfo.impurity, rightInfo.impurity,
                    leftPredict, rightPredict, split));
        }
        return GainInfo.getGainInfoByMaxGain(internalGainList);
    }

    private List<Pair> getCategoricalOrderList(double[] stats, int binSize) {
        List<Pair> categoricalOrderList = new ArrayList<Pair>(binSize);
        for(int i = 0; i < binSize; i++) {
            // for gini, use bin positive rate to sort
            double sum = stats[i * super.statsSize] + stats[i * super.statsSize + 1];
            double binPredict = 0d;
            if(sum != 0d) {
                binPredict = stats[i * super.statsSize + 1] / sum;
            }
            categoricalOrderList.add(new Pair(i, binPredict));
        }
        Collections.sort(categoricalOrderList, new Comparator<Pair>() {
            @Override
            public int compare(Pair o1, Pair o2) {
                return Double.valueOf(o1.value).compareTo(Double.valueOf(o2.value));
            }
        });
        return categoricalOrderList;
    }

    private InternalEntropyInfo getGiniInfo(double[] statsByClasses) {
        double sumAll = 0;
        for(int i = 0; i < statsByClasses.length; i++) {
            sumAll += statsByClasses[i];
        }

        double impurity = 0d;
        int indexOfLargestElement = -1;
        if(sumAll != 0d) {
            double maxElement = Double.MIN_VALUE;
            for(int i = 0; i < statsByClasses.length; i++) {
                double rate = statsByClasses[i] / sumAll;
                impurity -= rate * rate;
                if(statsByClasses[i] > maxElement) {
                    maxElement = statsByClasses[i];
                    indexOfLargestElement = i;
                }
            }
        }

        return new InternalEntropyInfo(sumAll, indexOfLargestElement, impurity);
    }

    private static class InternalEntropyInfo {
        double sumAll;
        double indexOfLargestElement;
        double impurity;

        public InternalEntropyInfo(double sumAll, double indexOfLargestElement, double impurity) {
            this.sumAll = sumAll;
            this.indexOfLargestElement = indexOfLargestElement;
            this.impurity = impurity;
        }
    }

    @Override
    public void featureUpdate(double[] featuerStatistic, int binIndex, float label, float significance, float weight) {
        // label + 0.1f to avoid 0.99999f is converted to 0
        featuerStatistic[binIndex * super.statsSize + (int) (label + 0.000001f)] += (significance * weight);
    }

}

class Pair {
    public Pair(int index, double value) {
        super();
        this.index = index;
        this.value = value;
    }

    int index;
    double value;
}
