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

import ml.shifu.shifu.container.ColumnScoreObject;

import java.util.List;


/**
 * ColumnScoreMessage class is the message that contains scores for some column.
 * columnNum is column id.
 */
public class ColumnScoreMessage {

    private int totalMsgCnt;
    private int columnNum;
    private List<ColumnScoreObject> colScoreList;

    public ColumnScoreMessage(int totalMsgCnt, int columnNum, List<ColumnScoreObject> colScoreList) {
        this.totalMsgCnt = totalMsgCnt;
        this.columnNum = columnNum;
        this.colScoreList = colScoreList;
    }

    public int getTotalMsgCnt() {
        return totalMsgCnt;
    }

    public int getColumnNum() {
        return columnNum;
    }

    public List<ColumnScoreObject> getColScoreList() {
        return colScoreList;
    }

}
