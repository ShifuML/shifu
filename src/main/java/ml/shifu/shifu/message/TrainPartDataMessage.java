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

import org.encog.ml.data.MLDataPair;

import java.util.List;

/**
 * TrainPartDataMessage class is message class that contains the data for model training
 */
public class TrainPartDataMessage {

    private int totalMsgCnt;
    private boolean isDryRun;
    private List<MLDataPair> mlDataPairList;

    public TrainPartDataMessage(int totalMsgCnt, boolean isDryRun, List<MLDataPair> mlDataPairList) {
        this.totalMsgCnt = totalMsgCnt;
        this.isDryRun = isDryRun;
        this.mlDataPairList = mlDataPairList;
    }

    public int getTotalMsgCnt() {
        return totalMsgCnt;
    }

    public boolean isDryRun() {
        return isDryRun;
    }

    public List<MLDataPair> getMlDataPairList() {
        return mlDataPairList;
    }

}
