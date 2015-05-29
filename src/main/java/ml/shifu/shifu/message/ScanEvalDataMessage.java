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

import java.util.Scanner;

/**
 * ScanEvalDataMessage class is the message class that contains input for evaluation
 */
public class ScanEvalDataMessage {

    private int streamId;
    private int totalStreamCnt;
    private Scanner scanner;

    public ScanEvalDataMessage(int streamId, int totalStreamCnt, Scanner scanner) {
        this.streamId = streamId;
        this.totalStreamCnt = totalStreamCnt;
        this.scanner = scanner;
    }

    public int getStreamId() {
        return streamId;
    }

    public int getTotalStreamCnt() {
        return totalStreamCnt;
    }

    public Scanner getScanner() {
        return scanner;
    }

}
