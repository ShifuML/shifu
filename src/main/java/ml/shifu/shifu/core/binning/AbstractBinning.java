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
 * @Oct 20, 2014
 *
 */
public abstract class AbstractBinning<T> {
    
    public static final char FIELD_SEPARATOR = '\u0001';
    public static final char SETLIST_SEPARATOR = '\u0002';
    public static final char PAIR_SEPARATOR = '\u0003';
    
    protected int missingValCnt = 0;
    protected int invalidValCnt = 0;
    
    protected int expectedBinningNum;
    protected Set<String> missingValSet;
    
    public AbstractBinning(){}
    
    public AbstractBinning(int binningNum) {
        this(binningNum, null);
    }
    
    public AbstractBinning(int binningNum, List<String> missingValList) {
        this.expectedBinningNum = binningNum;
        if ( CollectionUtils.isEmpty(missingValList) ) {
            this.missingValSet = new HashSet<String>();
            this.missingValSet.add("");
        } else {
            for ( String missingVal : missingValList ) {
                missingValSet.add( StringUtils.trimToEmpty(missingVal) );
            }
        }
    }
    
    public int getMissingValCnt() {
        return missingValCnt;
    }

    public int getInvalidValCnt() {
        return invalidValCnt;
    }
    
    public abstract void addData(String val);
    public abstract List<T> getDataBin();
    public abstract void mergeBin(AbstractBinning<?> another);
    
    protected Set<String> getMissingValSet() {
        return missingValSet;
    }
    
    protected boolean isMissingVal(String val) {
        return missingValSet.contains(val);
    }
    
    protected void incMissingValCnt() {
        missingValCnt ++;
    }
    
    protected void incInvalidValCnt() {
        invalidValCnt ++;
    }
    
    /**
     * convert @AbstractBinning to String
     * @return
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
     * @return
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
     * @param objValStr
     * @return
     */
    public static AbstractBinning<?> constructBinningFromStr(ModelConfig modelConfig, ColumnConfig columnConfig, String objValStr) {
        AbstractBinning<?> binning = null;
        
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
