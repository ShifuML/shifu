/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dvarsel.wrapper;

import ml.shifu.shifu.core.dvarsel.CandidateSeed;

/**
 * Created by zhanhu on 2015/4/14.
 */
public class SeedCredit {
    private int credit;
    private CandidateSeed seed;

    public SeedCredit(int credit, CandidateSeed seed) {
        this.credit = credit;
        this.seed = seed;
    }

    public int getCredit() {
        return credit;
    }

    public CandidateSeed getSeed() {
        return seed;
    }
}
