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

import java.util.List;

/**
 * StatsPartRawDataMessage class is message class that contains part of data for calculating stats
 */
public class StatsPartRawDataMessage {

    private int totalMsgCnt;
    private List<String> rawDataList;

    public StatsPartRawDataMessage(int totalMsgCnt, List<String> rawDataList) {
        this.totalMsgCnt = totalMsgCnt;
        this.rawDataList = rawDataList;
    }

    public int getTotalMsgCnt() {
        return totalMsgCnt;
    }

    public List<String> getRawDataList() {
        return rawDataList;
    }

}