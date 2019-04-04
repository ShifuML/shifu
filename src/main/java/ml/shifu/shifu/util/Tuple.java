/*
 * Copyright [2013-2019] PayPal Software Foundation
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

public class Tuple <FIRST, SECOND> {

    private FIRST first;
    
    private SECOND second;

    public Tuple(FIRST first, SECOND second) {
        this.first = first;
        this.second = second;
    }

    /**
     * @return the first
     */
    public FIRST getFirst() {
        return first;
    }

    /**
     * @param first the first to set
     */
    public void setFirst(FIRST first) {
        this.first = first;
    }

    /**
     * @return the second
     */
    public SECOND getSecond() {
        return second;
    }

    /**
     * @param second the second to set
     */
    public void setSecond(SECOND second) {
        this.second = second;
    }
    
    
}
