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

import java.util.ArrayList;
import java.util.List;

import ml.shifu.shifu.util.QuickSort;

import org.apache.commons.lang.StringUtils;

public class NativeBinning extends AbstractBinning<Double> {

    private List<Double> array;
    private boolean mergeEnabled;
    private final static double EPS = 1e-5;

    public NativeBinning(int binningNum, boolean mergeEnabled) {
        super(binningNum);
        this.mergeEnabled = mergeEnabled;
        this.array = new ArrayList<Double>();
    }

    @Override
    public void addData(String val) {
        String fval = StringUtils.trimToEmpty(val);
        if(!isMissingVal(fval)) {
            double dval = 0;

            try {
                dval = Double.parseDouble(fval);
            } catch (NumberFormatException e) {
                super.incInvalidValCnt();
                return;
            }
            array.add(dval);
        } else {
            super.incMissingValCnt();
        }
    }

    @Override
    public List<Double> getDataBin() {
        QuickSort.sort(array);

        int actualBinSize = (int) Math.ceil((double) array.size() / (double) expectedBinningNum);
        int actualBiningNum = this.expectedBinningNum;

        List<Double> binBoundary = new ArrayList<Double>();
        binBoundary.add(Double.NEGATIVE_INFINITY);

        double prevData = array.get(0);
        int currBinSize = 0;
        int currBinIndex = 0;
        for(int i = 0; i < array.size(); i++) {

            double currData = array.get(i);
            currBinSize++;
            if(currBinSize >= actualBinSize) {
                if(currBinIndex == actualBiningNum - 1 && i != array.size() - 1) {
                    continue;
                }

                if(i == 0 || (mergeEnabled == true && Math.abs(currData - prevData) > EPS) || mergeEnabled == false) {
                    if(i == array.size() - 1) {
                        break;
                    }
                    currBinIndex++;
                    currBinSize = 0;
                    binBoundary.add(currData);
                }

            }

            prevData = currData;
        }

        // binBoundary.set(binBoundary.size() - 1, Double.POSITIVE_INFINITY);
        return binBoundary;
    }

}
