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
package ml.shifu.shifu.core.binning.obj;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang.StringUtils;

/**
 * Created by zhanhu on 7/6/16.
 */
public class NumBinInfo {

    private double leftThreshold;
    private double rightThreshold;

    private long totalInstCnt = 0;
    private long positiveInstCnt = 0;

    public NumBinInfo(double leftThreshold, double rightThreshold) {
        this.leftThreshold = leftThreshold;
        this.rightThreshold = rightThreshold;
    }

    public double getLeftThreshold() {
        return leftThreshold;
    }

    public void setLeftThreshold(double leftThreshold) {
        this.leftThreshold = leftThreshold;
    }

    public double getRightThreshold() {
        return rightThreshold;
    }

    public void setRightThreshold(double rightThreshold) {
        this.rightThreshold = rightThreshold;
    }

    public long getTotalInstCnt() {
        return totalInstCnt;
    }

    public void setTotalInstCnt(long totalInstCnt) {
        this.totalInstCnt = totalInstCnt;
    }

    public long getPositiveInstCnt() {
        return positiveInstCnt;
    }

    public void setPositiveInstCnt(long positiveInstCnt) {
        this.positiveInstCnt = positiveInstCnt;
    }

    public void incInstCnt(boolean isPositiveInst) {
        this.totalInstCnt ++;
        if ( isPositiveInst ) {
            this.positiveInstCnt ++;
        }
    }

    public static List<NumBinInfo> constructNumBinfo(String binsData, char fieldSeparator) {
        List<NumBinInfo> binInfos = new ArrayList<NumBinInfo>();

        List<Double> thresholds = new ArrayList<Double>();
        thresholds.add(Double.NEGATIVE_INFINITY);

        String[] fields = StringUtils.split(binsData, fieldSeparator);
        for ( String field : fields ) {
            Double val = null;
            try {
                val = Double.valueOf(field);
                thresholds.add(val);
            } catch (Exception e) {
                // skip illegal double
            }
        }

        thresholds.add(Double.POSITIVE_INFINITY);
        Collections.sort(thresholds);

        for ( int i = 0; i < thresholds.size() - 1; i ++ ) {
            binInfos.add(new NumBinInfo(thresholds.get(i), thresholds.get(i + 1)));
        }

        return binInfos;
    }

    public void mergeRight(NumBinInfo numBinInfo) {
        this.setRightThreshold(numBinInfo.getRightThreshold());
        this.setTotalInstCnt(this.totalInstCnt + numBinInfo.getTotalInstCnt());
        this.setPositiveInstCnt(this.positiveInstCnt + numBinInfo.getPositiveInstCnt());
    }

    @Override
    public NumBinInfo clone() {
        NumBinInfo another = new NumBinInfo(this.leftThreshold, this.rightThreshold);
        another.setTotalInstCnt(this.totalInstCnt);
        another.setPositiveInstCnt(this.positiveInstCnt);
        return another;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("([" + this.leftThreshold + ", " + this.rightThreshold + "), ");
        builder.append(this.totalInstCnt + ", " + this.positiveInstCnt + ")");
        return builder.toString();
    }
}
