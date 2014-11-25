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

import org.apache.commons.lang.StringUtils;

/**
 * CategoricalBinning class
 * 
 * @author zhanhu
 * @Oct 20, 2014
 *
 */
public class CategoricalBinning extends AbstractBinning<String> {

    private Set<String> categoricalVals;
    
    /**
     * @param binningNum
     */
    public CategoricalBinning(int binningNum) {
        this(binningNum, null);
    }

    /**
     * @param binningNum
     * @param missingValList
     */
    public CategoricalBinning(int binningNum, List<String> missingValList) {
        super(binningNum, missingValList);
        this.categoricalVals = new HashSet<String>();
    }

    /* (non-Javadoc)
     * @see ml.shifu.shifu.core.binning.AbstractBinning#addData(java.lang.Object)
     */
    @Override
    public void addData(String val) {
        String fval = StringUtils.trimToEmpty(val);
        if ( !isMissingVal(fval) ) {
            categoricalVals.add(fval);
        } else {
            super.incMissingValCnt();
        }
    }

    /* (non-Javadoc)
     * @see ml.shifu.shifu.core.binning.AbstractBinning#getDataBin()
     */
    @Override
    public List<String> getDataBin() {
        List<String> binningVals = new ArrayList<String>();
        binningVals.addAll(categoricalVals);
        return binningVals;
    }

}
