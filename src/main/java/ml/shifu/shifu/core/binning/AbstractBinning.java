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
import ml.shifu.shifu.util.Constants;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;

/**
 * AbstractBinning class
 */
public abstract class AbstractBinning<T> {

    /**
     * Special characters for object serialization
     */
    public static final char FIELD_SEPARATOR = '\u0001';
    public static final char SETLIST_SEPARATOR = '\u0002';
    public static final char PAIR_SEPARATOR = '\u0003';

    /**
     * Missing data count and invalid data count
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

    protected int maxCategorySize = Constants.MAX_CATEGORICAL_BINC_COUNT;

    /**
     * Empty constructor : it is just for bin merging bin
     */
    protected AbstractBinning() {
    }

    /**
     * Constructor with expected bin number
     * 
     * @param binningNum
     *            the binningNum
     */
    public AbstractBinning(int binningNum) {
        this(binningNum, null, Constants.MAX_CATEGORICAL_BINC_COUNT);
    }

    /**
     * Constructor with expected bin number
     * 
     * @param binningNum
     *            the binningNum
     * @param maxCategorySize
     *            max size of category list
     */
    public AbstractBinning(int binningNum, int maxCategorySize) {
        this(binningNum, null, maxCategorySize);
    }

    /**
     * Constructor with expected bin number and expected missing values
     * 
     * @param binningNum
     *            the binningNum
     * @param missingValList
     *            the missing value list
     * @param maxCategorySize
     *            max size of category list
     */
    public AbstractBinning(int binningNum, List<String> missingValList, int maxCategorySize) {
        this.expectedBinningNum = binningNum;
        this.missingValSet = new HashSet<String>();
        this.missingValSet.add("");

        if(CollectionUtils.isNotEmpty(missingValList)) {
            for(String missingVal: missingValList) {
                missingValSet.add(StringUtils.trimToEmpty(missingVal));
            }
        }
        this.maxCategorySize = maxCategorySize;
    }

    /**
     * Get value missing count
     * 
     * @return the missing count
     */
    public int getMissingValCnt() {
        return missingValCnt;
    }

    /**
     * Get invalid value count
     * 
     * @return invalid count
     */
    public int getInvalidValCnt() {
        return invalidValCnt;
    }

    /**
     * Add data into bin generator
     * 
     * @param val
     *            the value to be added
     */
    public abstract void addData(String val);

    /**
     * Generate the bin boundary or bin category
     * 
     * @return data bin list
     */
    public abstract List<T> getDataBin();

    /**
     * Check some value is missing value or not
     * 
     * @param val
     *            the value to be checked
     * @return if it is missing value
     */
    public boolean isMissingVal(String val) {
        return missingValSet.contains(val);
    }

    /**
     * Increase the missing value count
     */
    protected void incMissingValCnt() {
        missingValCnt++;
    }

    /**
     * Increase the invalid value count
     */
    protected void incInvalidValCnt() {
        invalidValCnt++;
    }

    /**
     * Merge another binning info to this. Currently for the expected bin number, the max value will be used.
     * 
     * @param another
     *            the second binning to be mergerd
     */
    public void mergeBin(AbstractBinning<?> another) {
        this.expectedBinningNum = Math.max(this.expectedBinningNum, another.expectedBinningNum);

        this.missingValCnt += another.missingValCnt;
        this.invalidValCnt += another.invalidValCnt;

        if(missingValSet == null) {
            missingValSet = new HashSet<String>();
            missingValSet.add("");
        }

        missingValSet.addAll(another.missingValSet);
    }

    /**
     * convert @AbstractBinning to String
     * 
     * @param objValStr
     *            value string
     */
    public void stringToObj(String objValStr) {
        String[] objStrArr = objValStr.split(Character.toString(FIELD_SEPARATOR), -1);
        if(objStrArr.length < 4) {
            throw new IllegalArgumentException("The size of argument is incorrect");
        }

        missingValCnt = Integer.parseInt(StringUtils.trim(objStrArr[0]));
        invalidValCnt = Integer.parseInt(StringUtils.trim(objStrArr[1]));
        expectedBinningNum = Integer.parseInt(StringUtils.trim(objStrArr[2]));

        if(missingValSet == null) {
            missingValSet = new HashSet<String>();
        } else {
            missingValSet.clear();
        }

        String[] elements = objStrArr[3].split(Character.toString(SETLIST_SEPARATOR), -1);
        for(String element: elements) {
            missingValSet.add(element);
        }
    }

    /**
     * convert @AbstractBinning to String
     * 
     * @return string type of binning
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
     * 
     * @param modelConfig
     *            - the @ModelConfig to use
     * @param columnConfig
     *            - the @ColumnConfig to create bin
     * @param objValStr
     *            - the string present of object
     * @return the Binning object for the ColumnConfig
     */
    public static AbstractBinning<?> constructBinningFromStr(ModelConfig modelConfig, ColumnConfig columnConfig,
            String objValStr) {
        AbstractBinning<?> binning;

        if(columnConfig.isCategorical()) {
            binning = new CategoricalBinning();
        } else {
            if(modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval)) {
                binning = new EqualIntervalBinning();
            } else {
                binning = new EqualPopulationBinning();
            }
        }

        binning.stringToObj(objValStr);

        return binning;
    }
}
