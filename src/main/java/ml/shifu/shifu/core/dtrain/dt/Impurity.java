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

/**
 * TODO
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public abstract class Impurity {

    protected int statsSize;

    public abstract double impurity(double[] stats);

    public abstract double predict(double[] stats);

    public abstract double prob(double[] stats);

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

class Variance extends Impurity {

    public Variance() {
        super.statsSize = 3;
    }

    @Override
    public double impurity(double[] stats) {
        double count = stats[0];
        if(count != 0d) {
            return (stats[2] - (stats[1] * stats[1]) / count) / count;
        }
        return 0d;
    }

    @Override
    public double predict(double[] stats) {
        return stats[0] == 0d ? 0d : stats[1] / stats[0];
    }

    @Override
    public double prob(double[] stats) {
        return predict(stats);
    }

}

class Entropy extends Impurity {

    public Entropy() {
        super.statsSize = 2;
    }

    @Override
    public double impurity(double[] stats) {
        double totalCount = 0d;
        for(double d: stats) {
            totalCount += d;
        }

        if(totalCount == 0d) {
            return 0d;
        }

        double impurity = 0d;
        for(double stat: stats) {
            double rate = stat / totalCount;
            impurity += -rate * log2(rate);
        }

        return impurity;
    }

    private double log2(double x) {
        return Math.log(x) / Math.log(2);
    }

    @Override
    public double predict(double[] stats) {
        int maxClassIndex = -1;
        double maxStat = Double.MIN_VALUE;
        for(int i = 0; i < stats.length; i++) {
            if(stats[i] > maxStat) {
                maxStat = stats[i];
                maxClassIndex = i;
            }
        }
        return maxClassIndex * 1d;
    }

    @Override
    public double prob(double[] stats) {
        double totalCount = 0d;
        for(double stat: stats) {
            totalCount += stat;
        }

        if(totalCount == 0d) {
            return 0d;
        }
        return stats[1] / totalCount;
    }

}

class Gini extends Impurity {

    public Gini() {
        super.statsSize = 2;
    }

    @Override
    public double impurity(double[] stats) {
        double totalCount = 0d;
        for(double stat: stats) {
            totalCount += stat;
        }

        if(totalCount == 0d) {
            return 0d;
        }

        double impurity = 0d;
        for(double stat: stats) {
            double rate = stat / totalCount;
            impurity += -rate * rate;
        }

        return impurity;
    }

    @Override
    public double predict(double[] stats) {
        int maxClassIndex = -1;
        double maxStat = Double.MIN_VALUE;
        for(int i = 0; i < stats.length; i++) {
            if(stats[i] > maxStat) {
                maxStat = stats[i];
                maxClassIndex = i;
            }
        }
        return maxClassIndex * 1d;
    }

    @Override
    public double prob(double[] stats) {
        double totalCount = 0d;
        for(double stat: stats) {
            totalCount += stat;
        }

        if(totalCount == 0d) {
            return 0d;
        }
        return stats[1] / totalCount;
    }

}
