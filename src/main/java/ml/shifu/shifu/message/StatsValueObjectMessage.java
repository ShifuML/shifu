/**
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
package ml.shifu.shifu.message;

import ml.shifu.shifu.container.ValueObject;

import java.util.List;


/**
 * StatsValueObjectMessage class is message class that contains @ValueObject for calculating stats
 */
public class StatsValueObjectMessage {

    private int totalMsgCnt;
    private int columnNum;
    private List<ValueObject> voList;
    private long missing;
    private long total;

    public StatsValueObjectMessage(int totalMsgCnt, int columnNum, List<ValueObject> voList, long missing, long total) {
        this.totalMsgCnt = totalMsgCnt;
        this.columnNum = columnNum;
        this.voList = voList;
        this.setMissing(missing);
        this.setTotal(total);
    }

    public StatsValueObjectMessage(int totalMsgCnt, int columnNum, List<ValueObject> voList) {
        this.totalMsgCnt = totalMsgCnt;
        this.columnNum = columnNum;
        this.voList = voList;
    }

    public int getTotalMsgCnt() {
        return totalMsgCnt;
    }


    public int getColumnNum() {
        return columnNum;
    }

    public List<ValueObject> getVoList() {
        return voList;
    }

    public long getMissing() {
        return missing;
    }

    public void setMissing(long missing) {
        this.missing = missing;
    }

    public long getTotal() {
        return total;
    }

    public void setTotal(long total) {
        this.total = total;
    }

}
