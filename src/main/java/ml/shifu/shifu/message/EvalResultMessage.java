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

/**
 * EvalResultMessage class is the message for evaluation result
 */
public class EvalResultMessage {

    private int totalMsgCnt;

    public EvalResultMessage(int totalMsgCnt) {
        super();
        this.totalMsgCnt = totalMsgCnt;
    }

    public int getTotalMsgCnt() {
        return totalMsgCnt;
    }

}
