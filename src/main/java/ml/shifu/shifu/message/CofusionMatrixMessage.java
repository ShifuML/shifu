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

import ml.shifu.shifu.container.ConfusionMatrixObject;

import java.util.List;


public class CofusionMatrixMessage {

    private int totalMsgCnt;

    private List<ConfusionMatrixObject> matrixs;


    public CofusionMatrixMessage(int totalMsgCnt,
                                 List<ConfusionMatrixObject> matrixs) {
        super();
        this.totalMsgCnt = totalMsgCnt;
        this.matrixs = matrixs;
    }

    /**
     * @return the matrixs
     */
    public List<ConfusionMatrixObject> getMatrixs() {
        return matrixs;
    }

    /**
     * @param matrixs the matrixs to set
     */
    public void setMatrixs(List<ConfusionMatrixObject> matrixs) {
        this.matrixs = matrixs;
    }

    /**
     * @return the totalMsgCnt
     */
    public int getTotalMsgCnt() {
        return totalMsgCnt;
    }

    /**
     * @param totalMsgCnt the totalMsgCnt to set
     */
    public void setTotalMsgCnt(int totalMsgCnt) {
        this.totalMsgCnt = totalMsgCnt;
    }

}
