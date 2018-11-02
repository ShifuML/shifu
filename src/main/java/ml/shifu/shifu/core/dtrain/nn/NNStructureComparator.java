package ml.shifu.shifu.core.dtrain.nn;

import org.encog.neural.flat.FlatNetwork;

import java.util.Arrays;
import java.util.Comparator;

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
public class NNStructureComparator implements Comparator<FlatNetwork> {

    @Override
    public int compare(FlatNetwork from, FlatNetwork to) {
        if ( from.getInputCount() == to.getInputCount()
                && from.getOutputCount() == to.getOutputCount()
                && Arrays.equals(from.getLayerCounts(), to.getLayerCounts())
                && Arrays.equals(from.getLayerFeedCounts(), to.getLayerFeedCounts()) ) {
            return 0;
        } else if (from.getInputCount() >= to.getInputCount()
                && from.getOutputCount() >= to.getOutputCount()
                && isLargeOrEqualArr(from.getLayerCounts(), to.getLayerCounts())
                && isLargeOrEqualArr(from.getLayerFeedCounts(), to.getLayerFeedCounts()) ) {
            return 1;
        } else {
            return -1;
        }
    }

    private boolean isLargeOrEqualArr(int[] fromArray, int[] toArray) {
        boolean result = true;
        if ( fromArray != null && toArray != null && fromArray.length >= toArray.length ) {
            for ( int i = 0; i < toArray.length; i ++ ) {
                if ( fromArray[i] < toArray[i]) {
                    result = false;
                    break;
                }
            }
        } else {
            result = false;
        }
        return result;
    }
}