/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.container;

import ml.shifu.shifu.container.obj.ColumnConfig;

import java.util.List;

/**
 * reason code object
 */
public class ReasonResultObject {

    private ColumnConfig columnConfig;
    private Object varValue;
    private Integer hitBinNum;
    private List<String> reasonCodes;

    public ColumnConfig getColumnConfig() {
        return columnConfig;
    }

    public void setColumnConfig(ColumnConfig columnConfig) {
        this.columnConfig = columnConfig;
    }

    public Object getVarValue() {
        return varValue;
    }

    public void setVarValue(Object varValue) {
        this.varValue = varValue;
    }

    public Integer getHitBinNum() {
        return hitBinNum;
    }

    public void setHitBinNum(Integer hitBinNum) {
        this.hitBinNum = hitBinNum;
    }

    public List<String> getReasonCodes() {
        return reasonCodes;
    }

    public void setReasonCodes(List<String> reasonCodes) {
        this.reasonCodes = reasonCodes;
    }

}
