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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;

/**
 * AbstractBinning class
 * 
 * Oct 20, 2014
 *
 */
public abstract class AbstractBinning<T> {
    
    /**
     * Special characters for object serialization
     */
    public static final char FIELD_SEPARATOR = '\u0001';
    public static final char SETLIST_SEPARATOR = '\u0002';
    public static final char PAIR_SEPARATOR = '\u0003';
    
    /**
     * Missing data count &amp;&amp; invalid data count
     */
    protected int missingValCnt = 0;
    protected int invalidValCnt = 0;
    
    /**
     * Expected missing value set. The default missing value set only contain empty string ""
     */
    protected Set<String> missingValSet;
    
    /**
     * The expect bin number
     */
    protected int expectedBinningNum;
    
    /**
     * Empty constructor : it is just for bin merging bin
     */
    protected AbstractBinning(){}
    
    /**
     * Constructor with expected bin number
     * @param binningNum - bin number to expect
     */
    public AbstractBinning(int binningNum) {
        this(binningNum, null);
    }
    
    /**
     * Constructor with expected bin number and expected missing values
     * @param binningNum - bin number to expect
     * @param missingValList - missing value list
     */
    public AbstractBinning(int binningNum, List<String> missingValList) {
        this.expectedBinningNum = binningNum;
        this.missingValSet = new HashSet<String>();
        this.missingValSet.add("");

        if ( CollectionUtils.isNotEmpty(missingValList) ) {
            for ( String missingVal : missingValList ) {
                missingValSet.add( StringUtils.trimToEmpty(missingVal) );
            }
        }
    }
    
    /**
     * Get value missing count
     * @return missing value count
     */
    public int getMissingValCnt() {
        return missingValCnt;
    }

    /**
     * Get invalid value count
     * @return invalid value count
     */
    public int getInvalidValCnt() {
        return invalidValCnt;
    }
    
    /**
     * Add data into bin generator
     * @param val - value to add
     */
    public abstract void addData(String val);
    /**
     * Generate the bin boundary or bin category
     * @return bin boundaries
     */
    public abstract List<T> getDataBin();
    

    /**
     * Check some value is missing value or not
     * @param val - value to test
     * @return - is missing value or not
     */
    protected boolean isMissingVal(String val) {
        return missingValSet.contains(val);
    }
    
    /**
     * Increase the missing value count
     */
    protected void incMissingValCnt() {
        missingValCnt ++;
    }
    
    /**
     * Increase the invalid value count
     */
    protected void incInvalidValCnt() {
        invalidValCnt ++;
    }
    
    /**
     * Merge another binning info to this. Currently for the expected bin number, the max value will be used.
     * @param another - another binning to merge
     */
    public void mergeBin(AbstractBinning<?> another) {
        this.expectedBinningNum = Math.max(this.expectedBinningNum, another.expectedBinningNum);
        
        this.missingValCnt += another.missingValCnt;
        this.invalidValCnt += another.invalidValCnt;
        
        if ( missingValSet == null ) {
            missingValSet = new HashSet<String>();
            missingValSet.add("");
        }
        
        missingValSet.addAll(another.missingValSet);
    }
    
    /**
     * convert @AbstractBinning to String
     * @param objValStr String format of Binning Object
     */
    protected void stringToObj(String objValStr) {
        String[] objStrArr = objValStr.split(Character.toString(FIELD_SEPARATOR), -1);
        if ( objStrArr.length < 4 ) {
            throw new IllegalArgumentException("The size of argument is incorrect");
        }
        
        missingValCnt = Integer.parseInt(StringUtils.trim(objStrArr[0]));
        invalidValCnt = Integer.parseInt(StringUtils.trim(objStrArr[1]));
        expectedBinningNum = Integer.parseInt(StringUtils.trim(objStrArr[2]));
        
        if ( missingValSet == null ) {
            missingValSet = new HashSet<String>();
        } else {
            missingValSet.clear();
        }
        
        String[] elements = objStrArr[3].split(Character.toString(SETLIST_SEPARATOR), -1);
        for ( String element : elements ) {
            missingValSet.add(element);
        }
    }
    
    /**
     * convert @AbstractBinning to String
     * @return String format of Object
     */
    public String objToString() {
        List<String> strList = new ArrayList<String>();
        
        strList.add(Integer.toString(missingValCnt));
        strList.add(Integer.toString(invalidValCnt));
        strList.add(Integer.toString(expectedBinningNum));
        
        String missingValStr = StringUtils.join(missingValSet, SETLIST_SEPARATOR);
        strList.add(missingValStr);
        
        return StringUtils.join(strList, FIELD_SEPARATOR);
    }

    /**
     * Construct Binning class object from String
     * @param modelConfig - the @ModelConfig to use
     * @param columnConfig - the @ColumnConfig to create bin
     * @param objValStr - the string present of object
     * @return the Binning object for the ColumnConfig
     */
    public static AbstractBinning<?> constructBinningFromStr(ModelConfig modelConfig, ColumnConfig columnConfig, String objValStr) {
        AbstractBinning<?> binning;
        
        if ( columnConfig.isCategorical() ) {
            binning = new CategoricalBinning();
        } else {
            if ( modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval) ) {
                binning = new EqualIntervalBinning();
            } else {
                binning = new EqualPopulationBinning();
            }
        }
        
        binning.stringToObj(objValStr);
        
        return binning;
    }
}
