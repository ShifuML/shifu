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

import org.apache.commons.lang.StringUtils;

/**
 * EqualIntervalBinning class
 * 
 * @Oct 20, 2014
 *
 */
public class EqualIntervalBinning extends AbstractBinning<Double> {
    
    private double maxVal = -Double.MAX_VALUE;
    private double minVal = Double.MAX_VALUE;

    /**
     * Empty constructor : it is just for bin merging
     */
    protected EqualIntervalBinning() {}
    
    /**
     * Constructor with expected bin number
     * @param binningNum
     */
    public EqualIntervalBinning(int binningNum) {
        this(binningNum, null);
    }
    
    /**
     * Constructor with expected bin number and missing value list
     * @param binningNum
     */
    public EqualIntervalBinning(int binningNum, List<String> missingValList) {
        super(binningNum, missingValList);
    }

    /* 
     * Add the value (in format of text) into histogram with frequency 1. 
     * First of all the input string will be trimmed and check whether it is missing value or not
     * If it is missing value, the missing value count will +1
     * After that, the input string will be parsed into double. If it is not a double, invalid value count will +1 
     * @see ml.shifu.shifu.core.binning.AbstractBinning#addData(java.lang.Object)
     */
    @Override
    public void addData(String val) {
        String fval = StringUtils.trimToEmpty(val);
        if ( !isMissingVal(fval) ) {
            double dval = 0;
            
            try {
                dval = Double.parseDouble(fval);
            } catch (NumberFormatException e) {
                // not a number? just ignore
                super.incInvalidValCnt();
                return;
            }

            process(dval);
        } else {
            super.incMissingValCnt();
        }
    }

    /* (non-Javadoc)
     * @see ml.shifu.shifu.core.binning.AbstractBinning#getDataBin()
     */
    @Override
    public List<Double> getDataBin() {
        List<Double> binBorders = new ArrayList<Double>();
        if ( maxVal < minVal ) {
            // no data, just return empty
            return binBorders;
        }
        
        double delta = (maxVal - minVal) * 0.0005;
        double startVal = minVal - delta;
        double endVal = maxVal + delta;
        
        double binInterval = (endVal - startVal) / super.expectedBinningNum;
        double val = startVal;
        for ( int i = 0; i < super.expectedBinningNum; i ++ ) {
            binBorders.add(val);
            val = val + binInterval;
        }
        
        binBorders.add(endVal);
        
        return binBorders;
    }

    /**
     * Process the new incoming data
     * @param dval
     */
    private void process(double dval) {
        if ( dval < this.minVal ) {
            this.minVal = dval;
        }
        
        if ( dval > this.maxVal ) {
            this.maxVal = dval;
        }
    }
    

    /* (non-Javadoc)
     * @see ml.shifu.shifu.core.binning.AbstractBinning#mergeBin(ml.shifu.shifu.core.binning.AbstractBinning)
     */
    @Override
    public void mergeBin(AbstractBinning<?> another) {
        EqualIntervalBinning binning = (EqualIntervalBinning) another;
        
        super.mergeBin(another);
        
        process(binning.minVal);
        process(binning.maxVal);
    }
    
    /**
     * convert @EqualIntervalBinning to String
     * @return
     */
    protected void stringToObj(String objValStr) {
        super.stringToObj(objValStr);

        String[] objStrArr = objValStr.split(Character.toString(FIELD_SEPARATOR), -1);
        maxVal = Double.parseDouble(objStrArr[4]);
        minVal = Double.parseDouble(objStrArr[5]);
    }
    
    /**
     * convert @EqualIntervalBinning to String
     * @return
     */
    public String objToString() {
        return super.objToString() 
                + Character.toString(FIELD_SEPARATOR) 
                + Double.toString(maxVal)
                + Character.toString(FIELD_SEPARATOR) 
                + Double.toString(minVal);
    }
}
