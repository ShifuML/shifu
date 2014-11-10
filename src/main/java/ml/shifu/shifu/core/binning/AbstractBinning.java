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

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;

/**
 * AbstractBinning class
 * 
 * @Oct 20, 2014
 *
 */
public abstract class AbstractBinning<T> {
    
    protected int missingValCnt = 0;
    protected int invalidValCnt = 0;
    
    protected int expectedBinningNum;
    protected Set<String> missingValSet;
    
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
}
