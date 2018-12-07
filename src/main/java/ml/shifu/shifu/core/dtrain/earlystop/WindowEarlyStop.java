package ml.shifu.shifu.core.dtrain.earlystop;

import java.util.LinkedList;
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

public class WindowEarlyStop extends AbstractEarlyStopStrategy {

    private double globalMinimumError = Double.MAX_VALUE;
    private int bufferSize = 0;
    private int windowSize;

    public WindowEarlyStop(int windowSize) {
        this.windowSize = windowSize;
    }

    @Override public boolean shouldEarlyStop(int epochs, double[] weights, double trainingError, double validationError) {
        if ( validationError < this.globalMinimumError ) {
            this.globalMinimumError = validationError;
            this.bufferSize = 0;
            return false;
        } else {
            if ( this.bufferSize >= this.windowSize ) {
                return true;
            } else {
                this.bufferSize ++;
                return false;
            }
        }
    }
}
