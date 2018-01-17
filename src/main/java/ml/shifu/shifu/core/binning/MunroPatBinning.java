/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.binning;

import org.apache.commons.lang.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ml.shifu.shifu.core.MunroPatEstimator;
import ml.shifu.shifu.util.Constants;

public class MunroPatBinning extends AbstractBinning<Double> {

    private MunroPatEstimator<Double> estimator = null;

    public MunroPatBinning(int binningNum, List<String> missingValList) {
        super(binningNum, missingValList, Constants.MAX_CATEGORICAL_BINC_COUNT);
        estimator = new MunroPatEstimator<Double>(binningNum);
    }

    public MunroPatBinning(int binningNum) {
        this(binningNum, null);
    }

    @Override
    public void addData(String val) {
        String fval = StringUtils.trimToEmpty(val);
        try {
            Double dval = Double.parseDouble(fval);
            estimator.add(dval);
        } catch (NumberFormatException e) {
            super.incInvalidValCnt();
        }
    }

    /**
     * set min/max, merge same bins
     * In a very skewed data array, this one may not be well performed
     * 
     * @param bins
     *            input bins
     * @return merged bins
     */
    private List<Double> binMerge(List<Double> bins) {
        List<Double> newBins = new ArrayList<Double>();
        if(bins.size() == 0) {
            bins.add(Double.NaN);
            return bins;
        }

        Double cur = bins.get(0);
        newBins.add(cur);

        int i = 1;
        while(i < bins.size()) {
            if(Math.abs(cur - bins.get(i)) > 1e-10) {
                newBins.add(bins.get(i));
            }
            cur = bins.get(i);
            i++;
        }

        if(newBins.size() == 1) {
            // special case since there is only 1 candidate in the bins
            double val = newBins.get(0);
            newBins = Arrays.asList(new Double[] { Double.NEGATIVE_INFINITY, val });
        } else if(newBins.size() == 2) {
            newBins.set(0, Double.NEGATIVE_INFINITY);
        } else {
            newBins.set(0, Double.NEGATIVE_INFINITY);
            // remove the max, and became open interval
            newBins.remove(newBins.size() - 1);
        }
        return newBins;
    }

    @Override
    public List<Double> getDataBin() {
        return binMerge(estimator.getQuantiles());
    }

    public List<Double> getUnMergedDataBin() {
        return estimator.getQuantiles();
    }

}
