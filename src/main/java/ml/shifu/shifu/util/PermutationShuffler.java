/*
 * Copyright [2013-2020] PayPal Software Foundation
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
package ml.shifu.shifu.util;

import java.util.concurrent.ThreadLocalRandom;

/**
 * A {@link Shuffler} implementation that provides permutation index mapping. A permutation of [0, {@link #recordSize})
 * is created internally to provide the mapping functionality.
 *
 * @author Junshi Guo
 */
public class PermutationShuffler implements Shuffler {

    /**
     * Max index range. Should be set on construction and never change for one instance.
     */
    private final int recordSize;

    /**
     * Internally held permutation mapping from index -> permutation[index].
     */
    private final int[] permutation;

    public PermutationShuffler(int recordSize) {
        assert recordSize > 0;
        this.recordSize = recordSize;
        this.permutation = new int[recordSize];
        this.refresh();
    }

    /**
     * Generate permutation array.
     */
    @Override
    public void refresh() {
        for(int i = 0; i < recordSize; i++) {
            this.permutation[i] = i;
        }
        int temp, pos;
        for(int i = recordSize; i > 1; i--) {
            pos = ThreadLocalRandom.current().nextInt(i);
            temp = permutation[pos];
            permutation[pos] = permutation[i - 1];
            permutation[i - 1] = temp;
        }
    }

    @Override
    public int getIndex(int i) {
        assert i >= 0 && i < recordSize;
        return permutation[i];
    }

    @Override
    public int getRecordSize() {
        return this.recordSize;
    }
}
