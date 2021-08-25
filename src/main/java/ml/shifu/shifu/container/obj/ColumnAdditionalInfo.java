package ml.shifu.shifu.container.obj;

import java.util.List;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class ColumnAdditionalInfo {

    private Integer columnNum;
    private Double psiStd;
    private Double cosine;
    private Double cosStd;
    private List<String> unitStats;

    public Integer getColumnNum() {
        return columnNum;
    }

    public void setColumnNum(Integer columnNum) {
        this.columnNum = columnNum;
    }

    public Double getPsiStd() {
        return psiStd;
    }

    public void setPsiStd(Double psiStd) {
        this.psiStd = psiStd;
    }

    public Double getCosine() {
        return cosine;
    }

    public void setCosine(Double cosine) {
        this.cosine = cosine;
    }

    public Double getCosStd() {
        return cosStd;
    }

    public void setCosStd(Double cosStd) {
        this.cosStd = cosStd;
    }

    public List<String> getUnitStats() {
        return unitStats;
    }

    public void setUnitStats(List<String> unitStats) {
        this.unitStats = unitStats;
    }
}
