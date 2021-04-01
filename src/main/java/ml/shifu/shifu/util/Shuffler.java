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

/**
 * Shuffler provides the functionality of mapping one integer to another integer.
 *
 * @author Junshi Guo
 */
public interface Shuffler {

    /**
     * Mapping from one integer to another.
     *
     * @param i original index
     * @return mapped index
     */
    int getIndex(int i);

    /**
     * Total index size. The index should not exceed this max size.
     *
     * @return total index size
     */
    int getRecordSize();
}
